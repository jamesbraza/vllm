# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

import pytest
import torch

from vllm.multimodal.inputs import PlaceholderRange
from vllm.v1.kv_cache_interface import CrossAttentionSpec, FullAttentionSpec
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker.gpu_model_runner import GPUModelRunner


@dataclass
class MockMultiModalFeatureSpec:
    """Minimal mock for MultiModalFeatureSpec."""

    mm_position: PlaceholderRange


@dataclass
class MockCachedRequestState:
    """Minimal mock for CachedRequestState with mm_features."""

    mm_features: list[MockMultiModalFeatureSpec] | None = None


@dataclass
class MockInputBatch:
    """Minimal mock for InputBatch with request tracking."""

    req_ids: list[str]
    num_reqs: int
    req_id_to_index: dict[str, int]


class GetEncoderSeqLensTestHarness:
    def __init__(
        self,
        max_num_reqs: int,
        requests: dict[str, MockCachedRequestState],
        input_batch: MockInputBatch,
    ):
        self.requests = requests
        self.input_batch = input_batch
        # Create a real CpuGpuBuffer like GPUModelRunner does
        # - int32: sequence lengths are integers (matches gpu_model_runner.py:552)
        # - pin_memory=False: pinned memory is a CUDA optimization for faster
        #   CPU→GPU transfers; not needed for CPU-only tests
        self.encoder_seq_lens = CpuGpuBuffer(
            max_num_reqs,
            dtype=torch.int32,
            device=torch.device("cpu"),
            pin_memory=False,
        )

    _get_encoder_seq_lens = GPUModelRunner._get_encoder_seq_lens


# Common test constants with realistic values
# 336 = typical vision encoder output
# (e.g., CLIP ViT-L/14 produces 336 tokens per image)
ENCODER_SEQ_LEN = 336
# KV cache configuration matching typical model settings
BLOCK_SIZE = 16  # KV cache block size (tokens per block)
NUM_KV_HEADS = 8  # Number of key-value attention heads
HEAD_SIZE = 64  # Dimension of each attention head
MAX_NUM_REQS = 4  # Test batch size


@pytest.fixture
def cross_attention_spec() -> CrossAttentionSpec:
    return CrossAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=NUM_KV_HEADS,
        head_size=HEAD_SIZE,
        dtype=torch.float16,
    )


@pytest.fixture
def full_attention_spec() -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=NUM_KV_HEADS,
        head_size=HEAD_SIZE,
        dtype=torch.float16,
    )


class TestGetEncoderSeqLensRegression:
    """
    Regression tests for _get_encoder_seq_lens.

    These tests verify the fix where decode phase (empty num_scheduled_tokens)
    still correctly populates encoder_seq_lens from existing requests.
    """

    def test_returns_none_for_non_cross_attention_spec(
        self, full_attention_spec: FullAttentionSpec
    ):
        """Non-CrossAttentionSpec should return None."""
        harness = GetEncoderSeqLensTestHarness(
            max_num_reqs=MAX_NUM_REQS,
            requests={},
            input_batch=MockInputBatch(req_ids=[], num_reqs=0, req_id_to_index={}),
        )

        result = harness._get_encoder_seq_lens(
            num_scheduled_tokens={},
            kv_cache_spec=full_attention_spec,
            num_reqs=0,
        )

        assert result == (None, None)

    def test_prefill_phase_populates_encoder_seq_lens(
        self, cross_attention_spec: CrossAttentionSpec
    ):
        """Prefill phase: num_scheduled_tokens has entries, encoder_seq_lens is set."""
        req_id = "req-0"

        harness = GetEncoderSeqLensTestHarness(
            max_num_reqs=MAX_NUM_REQS,
            requests={
                req_id: MockCachedRequestState(
                    mm_features=[
                        MockMultiModalFeatureSpec(
                            mm_position=PlaceholderRange(
                                offset=0, length=ENCODER_SEQ_LEN
                            )
                        )
                    ]
                )
            },
            input_batch=MockInputBatch(
                req_ids=[req_id], num_reqs=1, req_id_to_index={req_id: 0}
            ),
        )

        # Prefill: num_scheduled_tokens maps req_id -> num tokens being scheduled
        # (e.g., 100 prompt tokens being processed in this forward pass)
        encoder_seq_lens, encoder_seq_lens_cpu = harness._get_encoder_seq_lens(
            num_scheduled_tokens={req_id: 100},
            kv_cache_spec=cross_attention_spec,
            num_reqs=1,
        )

        assert encoder_seq_lens_cpu[0] == ENCODER_SEQ_LEN
        assert encoder_seq_lens[0].item() == ENCODER_SEQ_LEN

    def test_decode_phase_with_empty_num_scheduled_tokens(
        self, cross_attention_spec: CrossAttentionSpec
    ):
        """
        REGRESSION TEST: During decode, num_scheduled_tokens is empty but
        encoder_seq_lens must still be populated from existing requests.

        This is the core bug that was fixed. Before the fix, the for loop
        would iterate over empty num_scheduled_tokens and skip populating
        encoder_seq_lens entirely, leaving it as zeros.
        """
        req_id = "req-0"

        harness = GetEncoderSeqLensTestHarness(
            max_num_reqs=MAX_NUM_REQS,
            requests={
                req_id: MockCachedRequestState(
                    mm_features=[
                        MockMultiModalFeatureSpec(
                            mm_position=PlaceholderRange(
                                offset=0, length=ENCODER_SEQ_LEN
                            )
                        )
                    ]
                )
            },
            input_batch=MockInputBatch(
                req_ids=[req_id], num_reqs=1, req_id_to_index={req_id: 0}
            ),
        )

        # Decode phase: num_scheduled_tokens is EMPTY because decoder tokens
        # are generated one at a time and don't appear in this dict
        encoder_seq_lens, encoder_seq_lens_cpu = harness._get_encoder_seq_lens(
            num_scheduled_tokens={},  # Empty! This is the decode phase.
            kv_cache_spec=cross_attention_spec,
            num_reqs=1,
        )

        # The fix ensures encoder_seq_lens is still populated
        assert encoder_seq_lens_cpu[0] == ENCODER_SEQ_LEN, (
            "encoder_seq_lens was not populated during decode phase. "
            "This indicates the fix for iterating over input_batch.req_ids "
            "when num_scheduled_tokens is empty has regressed."
        )
        assert encoder_seq_lens[0].item() == ENCODER_SEQ_LEN

    def test_decode_phase_multiple_requests(
        self, cross_attention_spec: CrossAttentionSpec
    ):
        """Multiple requests during decode all have encoder_seq_lens populated."""
        # Different encoder lengths simulate different image sizes/counts
        encoder_len_req0 = ENCODER_SEQ_LEN  # 336 tokens (1 standard image)
        encoder_len_req1 = 512  # 512 tokens (larger image or different encoder)
        requests = {
            "req-0": MockCachedRequestState(
                mm_features=[
                    MockMultiModalFeatureSpec(
                        mm_position=PlaceholderRange(offset=0, length=encoder_len_req0)
                    )
                ]
            ),
            "req-1": MockCachedRequestState(
                mm_features=[
                    MockMultiModalFeatureSpec(
                        mm_position=PlaceholderRange(offset=0, length=encoder_len_req1)
                    )
                ]
            ),
            "req-2": MockCachedRequestState(mm_features=None),  # Text-only request
        }

        harness = GetEncoderSeqLensTestHarness(
            max_num_reqs=MAX_NUM_REQS,
            requests=requests,
            input_batch=MockInputBatch(
                req_ids=["req-0", "req-1", "req-2"],
                num_reqs=3,
                req_id_to_index={"req-0": 0, "req-1": 1, "req-2": 2},
            ),
        )

        # Decode phase with empty num_scheduled_tokens
        encoder_seq_lens, encoder_seq_lens_cpu = harness._get_encoder_seq_lens(
            num_scheduled_tokens={},
            kv_cache_spec=cross_attention_spec,
            num_reqs=3,
        )

        assert encoder_seq_lens_cpu[0] == encoder_len_req0
        assert encoder_seq_lens_cpu[1] == encoder_len_req1
        assert encoder_seq_lens_cpu[2] == 0  # Text-only, no encoder tokens

    def test_empty_mm_features_list_handled(
        self, cross_attention_spec: CrossAttentionSpec
    ):
        """Empty mm_features list (not None) should result in encoder_seq_lens = 0."""
        req_id = "req-0"

        harness = GetEncoderSeqLensTestHarness(
            max_num_reqs=MAX_NUM_REQS,
            requests={req_id: MockCachedRequestState(mm_features=[])},  # Empty list
            input_batch=MockInputBatch(
                req_ids=[req_id], num_reqs=1, req_id_to_index={req_id: 0}
            ),
        )

        _, encoder_seq_lens_cpu = harness._get_encoder_seq_lens(
            num_scheduled_tokens={},
            kv_cache_spec=cross_attention_spec,
            num_reqs=1,
        )

        assert encoder_seq_lens_cpu[0] == 0

    def test_multiple_mm_features_summed(
        self, cross_attention_spec: CrossAttentionSpec
    ):
        """Multiple mm_features should have their lengths summed."""
        req_id = "req-0"
        # Simulate multiple images/features in one request
        # e.g., 3 images with different encoder output sizes
        feature_lens = [100, 200, 36]  # Individual feature lengths
        expected_total = sum(feature_lens)  # 336 total encoder tokens

        harness = GetEncoderSeqLensTestHarness(
            max_num_reqs=MAX_NUM_REQS,
            requests={
                req_id: MockCachedRequestState(
                    mm_features=[
                        MockMultiModalFeatureSpec(
                            mm_position=PlaceholderRange(
                                offset=0, length=feature_lens[0]
                            )
                        ),
                        MockMultiModalFeatureSpec(
                            mm_position=PlaceholderRange(
                                offset=feature_lens[0], length=feature_lens[1]
                            )
                        ),
                        MockMultiModalFeatureSpec(
                            mm_position=PlaceholderRange(
                                offset=feature_lens[0] + feature_lens[1],
                                length=feature_lens[2],
                            )
                        ),
                    ]
                )
            },
            input_batch=MockInputBatch(
                req_ids=[req_id], num_reqs=1, req_id_to_index={req_id: 0}
            ),
        )

        # num_scheduled_tokens maps req_id -> decoder tokens being scheduled
        # (e.g., 50 tokens from prompt being processed)
        _, encoder_seq_lens_cpu = harness._get_encoder_seq_lens(
            num_scheduled_tokens={req_id: 50},
            kv_cache_spec=cross_attention_spec,
            num_reqs=1,
        )

        assert encoder_seq_lens_cpu[0] == expected_total
