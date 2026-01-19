# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest  # noqa

from vllm.config import CacheConfig, SchedulerConfig
from vllm.core.scheduler import Scheduler
from vllm.sequence import (SequenceGroup, SequenceGroupMetadata,
                           SequenceGroupMetadataDelta)

from .utils import (append_new_token, create_dummy_prompt_encoder_decoder,
                    get_sequence_groups, schedule_and_update_computed_tokens)


def test_scheduler_schedule_simple_encoder_decoder():
    '''
    Test basic scheduler functionality in the context
    of an encoder/decoder model. Focus on testing
    enc/dec-specific functionality sense tests already
    exist for decoder-only functionality

    Test behavior:
    * Construct Scheduler
    * Construct dummy encoder/decoder sequence groups
    * Add dummy seq groups to scheduler backlog
    * Schedule the next seq group & validate:
        * Cross-attn block tables
        * Updated states of seq groups
        * Number of batched tokens
        * Number of blocks to copy/swap-in/swap-out
        * Number of scheduled seq groups
    * Repeat for both prefill- and decode-phase
    * Abort scheduled seq groups
    * Assert that aborted seq groups no longer appear in
      cross-attention block table
    '''

    block_size = 4
    num_seq_group = 4
    max_model_len = 16
    scheduler_config = SchedulerConfig(
        "generate",
        max_num_batched_tokens=64,
        max_num_seqs=num_seq_group,
        max_model_len=max_model_len,
    )
    cache_config = CacheConfig(block_size, 1.0, 1, "auto")
    cache_config.num_cpu_blocks = 16  # enc and dec prompts per seq_group
    cache_config.num_gpu_blocks = 16  # enc and dec prompts per seq_group
    scheduler = Scheduler(scheduler_config, cache_config, None)
    running: list[SequenceGroup] = []

    # Add seq groups to scheduler.
    req_id_list = []
    for i in range(num_seq_group):
        req_id = str(i)
        req_id_list.append(req_id)
        _, _, seq_group = create_dummy_prompt_encoder_decoder(
            req_id, block_size, block_size, block_size)
        scheduler.add_seq_group(seq_group)
        running.append(seq_group)

    # Schedule seq groups prefill.
    num_tokens = block_size * num_seq_group
    seq_group_meta_list, out = schedule_and_update_computed_tokens(scheduler)
    # - Verify that sequence group cross-attention block tables are
    #   registered with the block manager
    assert all([(req_id in scheduler.block_manager.cross_block_tables)
                for req_id in req_id_list])
    # - Validate sequence-group status
    assert set(get_sequence_groups(out)) == set(running)
    # - Validate number of batched tokens
    assert out.num_batched_tokens == num_tokens
    # - Validate there are no remaining blocks to swap
    assert (not out.blocks_to_copy and not out.blocks_to_swap_in
            and not out.blocks_to_swap_out)
    # - Validate all seq groups were scheduled
    assert len(seq_group_meta_list) == num_seq_group
    append_new_token(out, 1)

    # Schedule seq groups decode.
    seq_group_meta_list, out = schedule_and_update_computed_tokens(scheduler)
    # - Verify that sequence group metadata includes encoder attention
    #   and cross-attention metadata
    assert all([
        not ((seq_group_meta.encoder_seq_data is None) or
             (seq_group_meta.cross_block_table is None))
        for seq_group_meta in seq_group_meta_list
    ])
    # - Validate sequence-group status
    assert set(get_sequence_groups(out)) == set(running)
    # - Validate there is one batched token per seq group
    assert out.num_batched_tokens == num_seq_group
    # - Validate there are no remaining blocks to swap
    assert (not out.blocks_to_copy and not out.blocks_to_swap_in
            and not out.blocks_to_swap_out)
    # - Validate that all seq groups were scheduled
    assert len(seq_group_meta_list) == num_seq_group
    append_new_token(out, 1)

    # Abort sequences
    for req_id in req_id_list:
        scheduler.abort_seq_group(req_id)
        # - Verify that sequence group cross-attention block tables are
        #   NO LONGER registered with the block manager
        assert req_id not in scheduler.block_manager.cross_block_tables


def test_scheduler_encoder_decoder_delta_mode_cross_block_table():
    """
    Test that cross_block_table is properly propagated in delta mode
    for encoder-decoder models.

    This is a regression test for a bug where SequenceGroupMetadataDelta
    was missing the cross_block_table field, causing encoder-decoder models
    to fail during decode when using SPMD mode (send_delta_data=True).

    The bug: After prefill, subsequent decode steps in delta mode would
    not include cross_block_table, so the attention layer couldn't locate
    the encoder KV cache blocks.
    """
    # Small KV cache block size keeps test fast
    # while ensuring at least 1 block is allocated
    block_size = 4
    # Use at least 2 seq groups to verify fix works for concurrency sequences,
    # but keep small for test speed
    num_seq_group = 2
    scheduler_config = SchedulerConfig(
        "generate",
        max_num_batched_tokens=64,
        max_num_seqs=num_seq_group,
        max_model_len=
        16,  # High enough to accommodate 4-token prompt + decode tokens
        send_delta_data=True,  # Enable delta mode (SPMD)
    )
    cache_config = CacheConfig(block_size, 1.0, 1, "auto")
    cache_config.num_cpu_blocks = 16  # enc and dec prompts per seq_group
    cache_config.num_gpu_blocks = 16  # enc and dec prompts per seq_group
    scheduler = Scheduler(scheduler_config, cache_config, None)

    # Add encoder-decoder seq groups to scheduler
    for i in range(num_seq_group):
        _, _, seq_group = create_dummy_prompt_encoder_decoder(
            str(i), block_size, block_size, block_size)
        scheduler.add_seq_group(seq_group)

    # Schedule seq groups prefill.
    seq_group_meta_list, out = schedule_and_update_computed_tokens(scheduler)
    assert len(
        seq_group_meta_list
    ) == num_seq_group, "Incorrect number of seq groups during prefill"
    assert all(
        isinstance(meta, SequenceGroupMetadata)
        for meta in seq_group_meta_list), "Prefill should return full metadata"
    assert all(meta.cross_block_table is not None
               for meta in seq_group_meta_list
               ), "Expecting cross block table should be set during prefill"
    append_new_token(out, 1)

    # Schedule seq groups decode.
    seq_group_meta_list, out = schedule_and_update_computed_tokens(scheduler)
    assert len(
        seq_group_meta_list
    ) == num_seq_group, "Incorrect number of seq groups during decode"
    assert all(
        isinstance(meta, SequenceGroupMetadataDelta)
        for meta in seq_group_meta_list
    ), "Test should be covering the delta code path during decode"
    for meta in seq_group_meta_list:
        assert meta.cross_block_table is not None, (
            "Expecting cross block table present"
            " for encoder-decoder cross-attention")
        assert meta.cross_block_table, (
            "Expecting cross block table present to have block IDs")
