# coding=utf-8
# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Forward step utilities."""

from collections.abc import Iterable

import torch

from megatron.training import get_args
from megatron.core import parallel_state

from .utils import forward_step


class ForwardStep:
    """
    Forward step function with all the communications.
    We use a class here to hide the inference parameters
    from the outside caller.
    """

    def __init__(self, model):
        """Set values so we don't need to do it multiple times."""
        # Make sure model is in eval mode.
        if isinstance(model, Iterable):
            raise TypeError("Interleaving schedule is not supported for inference")

        model.eval()
        self.model = model

        # Pipelining arguments.
        args = get_args()
        self.pipeline_size_larger_than_one = (
            args.pipeline_model_parallel_size > 1)
        # Threshold of pipelining.
        self.pipelining_batch_x_seqlen = args.inference_batch_times_seqlen_threshold
        self.micro_batch_size = args.micro_batch_size

    def __call__(self, tokens, position_ids, attention_mask, inference_params):
        """Invocation of the forward methods. Note that inference_params
        is being modified by the forward step."""
        # Pipelining case.
        if self.pipeline_size_larger_than_one:
            return _with_pipelining_forward_step(self.model,
                                                 (tokens,
                                                  position_ids,
                                                  attention_mask),
                                                 inference_params,
                                                 self.micro_batch_size)
        else:
            return _no_pipelining_forward_step(self.model,
                                               (tokens,
                                                position_ids,
                                                attention_mask),
                                               inference_params)


def _get_recv_buffer_dtype(args):
    """Receive happens between the layers."""
    if args.fp32_residual_connection:
        return torch.float
    return args.params_dtype


def _allocate_recv_buffer(batch_size, sequence_length):
    """Receive happens between the layers with size [s, b, h]."""
    res = None
    if not parallel_state.is_pipeline_first_stage():
        args = get_args()
        recv_size = (sequence_length, batch_size, args.hidden_size)
        res = torch.empty(recv_size,
                          dtype=_get_recv_buffer_dtype(args),
                          device=torch.cuda.current_device())

    return res


def _no_pipelining_forward_step(model, inputs, inference_params):
    """If recv_buffer is none, we will allocate one on the fly."""
    # Run a simple forward pass.
    tokens, position_ids, attention_mask = inputs
    sequence_length = tokens.size(1)
    output_tensor = forward_step(model,
                            tokens,
                            position_ids=position_ids,
                            attention_mask=attention_mask,
                            tokentype_ids=None,
                            inference_params=inference_params)
    
    if inference_params:
        # Update the sequence length offset.
        inference_params.sequence_len_offset += sequence_length

    logits = None
    if parallel_state.is_pipeline_last_stage():
        logits = output_tensor

    return logits


def _with_pipelining_forward_step(model, inputs, inference_params, micro_batch_size):
    """No interleaving is supported."""
    tokens, position_ids, attention_mask = inputs
    sequence_length = tokens.size(1)
    batch_size = tokens.size(0)

    # Divide the batch dimension into micro batches.
    num_micro_batches, last_chunk = divmod(batch_size, micro_batch_size)

    if last_chunk > 0:
        num_micro_batches += 1

    # Preallocate memory for output logits.
    logits = None
    if parallel_state.is_pipeline_last_stage():
        args = get_args()
        logits = torch.empty(
            (batch_size, sequence_length, args.padded_vocab_size),
            dtype=torch.float32, device=torch.cuda.current_device())

    for micro_batch_index in range(num_micro_batches):
        # Slice among the batch dimension.
        start = micro_batch_index * micro_batch_size
        end = min(start + micro_batch_size, batch_size)
        this_micro_batch_size = end - start
        tokens2use = tokens[start: end, ...]
        position_ids2use = position_ids[start: end, ...]

        output = forward_step(model,
                         tokens2use,
                         position_ids=position_ids2use,
                         attention_mask=attention_mask,
                         tokentype_ids=None,
                         inference_params=inference_params)

        if inference_params:
            # Adjust the batch size offset to account for the micro-batch.
            inference_params.batch_size_offset += this_micro_batch_size

        # Copy logits.
        if parallel_state.is_pipeline_last_stage():
            logits[start: end, ...] = output

    if inference_params:
        # Once we are done with all the micro-batches, we can
        # adjust the sequence length offset.
        inference_params.sequence_len_offset += sequence_length
        # and reset the batch size offset
        inference_params.batch_size_offset = 0

    return logits
