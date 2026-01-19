# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.sampling_params import SamplingParams
from vllm.sequence import (CompletionSequenceGroupOutput, IntermediateTensors,
                           SequenceData, SequenceDataDelta,
                           SequenceGroupMetadata, SequenceGroupMetadataDelta,
                           SequenceOutput, SequenceStage)

from .core.utils import create_dummy_prompt


@pytest.fixture
def sample_outputs():
    return [
        CompletionSequenceGroupOutput(samples=[
            SequenceOutput(parent_seq_id=0, output_token=i, logprobs={})
        ],
                                      prompt_logprobs=None) for i in range(5)
    ]


@pytest.fixture
def sampler_output(sample_outputs):
    return SamplerOutput(outputs=sample_outputs)


def test_sampler_output_initialization(sampler_output, sample_outputs):
    assert len(sampler_output) == len(sample_outputs)
    assert sampler_output.sampled_token_probs is None
    assert sampler_output.sampled_token_ids is None


def test_sampler_output_getitem(sampler_output, sample_outputs):
    assert sampler_output[2] == sample_outputs[2]


def test_sampler_output_setitem(sampler_output):
    new_output = CompletionSequenceGroupOutput(samples=[
        SequenceOutput(parent_seq_id=0, output_token=99, logprobs={})
    ],
                                               prompt_logprobs=None)
    sampler_output[2] = new_output
    assert sampler_output[2] == new_output


def test_sampler_output_len(sampler_output, sample_outputs):
    assert len(sampler_output) == len(sample_outputs)


def test_sampler_output_eq(sample_outputs):
    sampler_output1 = SamplerOutput(outputs=sample_outputs)
    sampler_output2 = SamplerOutput(outputs=sample_outputs.copy())
    sampler_output3 = SamplerOutput(outputs=sample_outputs[:-1])
    assert sampler_output1 == sampler_output2
    assert sampler_output1 != sampler_output3


def test_sequence_data_prefill():
    seq_data = SequenceData.from_seqs([1, 2, 3, 4])
    assert seq_data.get_num_uncomputed_tokens() == 4
    assert seq_data.get_num_computed_tokens() == 0
    # advance by 2
    seq_data.update_num_computed_tokens(2)
    assert seq_data.get_num_uncomputed_tokens() == 2
    assert seq_data.get_num_computed_tokens() == 2

    # advance by 1
    seq_data.update_num_computed_tokens(1)
    assert seq_data.get_num_uncomputed_tokens() == 1
    assert seq_data.get_num_computed_tokens() == 3

    # append tokens and reset, simulating recompute
    seq_data.append_token_id(1, logprob=0.0)
    seq_data.reset_state_for_recompute()
    assert seq_data.get_num_uncomputed_tokens() == 5
    assert seq_data.get_num_computed_tokens() == 0


def test_sequence_group_stage():
    _, seq_group = create_dummy_prompt("1", 12)
    assert seq_group.is_prefill() is True
    seq_group.update_num_computed_tokens(6)
    assert seq_group.is_prefill() is True
    seq_group.update_num_computed_tokens(5)
    assert seq_group.is_prefill() is True
    seq_group.update_num_computed_tokens(1)
    assert seq_group.is_prefill() is False
    seqs = seq_group.get_seqs()
    assert len(seqs) == 1
    seqs[0].data.append_token_id(1, logprob=0.0)
    for seq in seq_group.get_seqs():
        seq.reset_state_for_recompute()
    assert seq_group.is_prefill() is True
    seq_group.update_num_computed_tokens(5)
    assert seq_group.is_prefill() is True
    seq_group.update_num_computed_tokens(7)
    assert seq_group.is_prefill() is True
    seq_group.update_num_computed_tokens(1)
    assert seq_group.is_prefill() is False


def test_sequence_intermediate_tensors_equal():

    class AnotherIntermediateTensors(IntermediateTensors):
        pass

    intermediate_tensors = IntermediateTensors({})
    another_intermediate_tensors = AnotherIntermediateTensors({})
    assert intermediate_tensors != another_intermediate_tensors

    empty_intermediate_tensors_1 = IntermediateTensors({})
    empty_intermediate_tensors_2 = IntermediateTensors({})
    assert empty_intermediate_tensors_1 == empty_intermediate_tensors_2

    different_key_intermediate_tensors_1 = IntermediateTensors(
        {"1": torch.zeros([2, 4], dtype=torch.int32)})
    difference_key_intermediate_tensors_2 = IntermediateTensors(
        {"2": torch.zeros([2, 4], dtype=torch.int32)})
    assert (different_key_intermediate_tensors_1
            != difference_key_intermediate_tensors_2)

    same_key_different_value_intermediate_tensors_1 = IntermediateTensors(
        {"1": torch.zeros([2, 4], dtype=torch.int32)})
    same_key_different_value_intermediate_tensors_2 = IntermediateTensors(
        {"1": torch.zeros([2, 5], dtype=torch.int32)})
    assert (same_key_different_value_intermediate_tensors_1
            != same_key_different_value_intermediate_tensors_2)

    same_key_same_value_intermediate_tensors_1 = IntermediateTensors(
        {"1": torch.zeros([2, 4], dtype=torch.int32)})
    same_key_same_value_intermediate_tensors_2 = IntermediateTensors(
        {"1": torch.zeros([2, 4], dtype=torch.int32)})
    assert (same_key_same_value_intermediate_tensors_1 ==
            same_key_same_value_intermediate_tensors_2)


def test_sequence_group_metadata_delta_cross_block_table():
    """Test that cross_block_table is properly propagated via delta for
    encoder-decoder models.

    This is a regression test for a bug where cross_block_table was not
    included in SequenceGroupMetadataDelta, causing encoder-decoder models
    to fail during decode phase when using delta mode (SPMD).
    """
    # Create initial full metadata with cross_block_table
    seq_data = SequenceData.from_seqs([1, 2, 3, 4])
    initial_cross_block_table = [10, 11, 12]
    metadata = SequenceGroupMetadata(
        request_id="test_0",
        is_prompt=True,
        seq_data={0: seq_data},
        sampling_params=SamplingParams(temperature=0),
        block_tables={0: [1, 2, 3]},
        cross_block_table=initial_cross_block_table,
    )
    assert metadata.cross_block_table == initial_cross_block_table

    # Create a delta with updated cross_block_table.
    # is_prompt=False because deltas are only sent after the first prefill
    # (in SPMD/delta mode), so they always represent decode steps.
    updated_cross_block_table = [20, 21, 22, 23]
    delta = SequenceGroupMetadataDelta(
        seq_data_delta={
            0:
            SequenceDataDelta(
                new_output_token_ids=[5],
                new_cumulative_logprob=0.0,
                # 4 prompt tokens + 1 generated token = 5 total computed
                new_num_computed_tokens=5,
                new_stage=SequenceStage.DECODE,
            )
        },
        request_id="test_0",
        block_tables={0: [1, 2, 3, 4]},
        is_prompt=False,
        cross_block_table=updated_cross_block_table,
    )
    assert delta.cross_block_table == updated_cross_block_table

    # Apply delta and verify cross_block_table is updated
    metadata.apply_delta(delta)
    assert metadata.cross_block_table == updated_cross_block_table
    assert not metadata.is_prompt


def test_sequence_group_metadata_delta_cross_block_table_none():
    """Test that apply_delta preserves existing cross_block_table when
    delta has None (for non-encoder-decoder models)."""
    seq_data = SequenceData.from_seqs([1, 2, 3, 4])
    initial_cross_block_table = [10, 11, 12]
    metadata = SequenceGroupMetadata(
        request_id="test_0",
        is_prompt=True,
        seq_data={0: seq_data},
        sampling_params=SamplingParams(temperature=0),
        block_tables={0: [1, 2, 3]},
        cross_block_table=initial_cross_block_table,
    )

    # Create a delta WITHOUT cross_block_table (None).
    # is_prompt=False because deltas are only sent after the first prefill
    # (in SPMD/delta mode), so they always represent decode steps.
    delta = SequenceGroupMetadataDelta(
        seq_data_delta={
            0:
            SequenceDataDelta(
                new_output_token_ids=[5],
                new_cumulative_logprob=0.0,
                # 4 prompt tokens + 1 generated token = 5 total computed
                new_num_computed_tokens=5,
                new_stage=SequenceStage.DECODE,
            )
        },
        request_id="test_0",
        block_tables={0: [1, 2, 3, 4]},
        is_prompt=False,
    )
    assert delta.cross_block_table is None, "Test expects None default"

    # Apply delta - cross_block_table should be preserved
    metadata.apply_delta(delta)
    assert metadata.cross_block_table == initial_cross_block_table
    assert not metadata.is_prompt
