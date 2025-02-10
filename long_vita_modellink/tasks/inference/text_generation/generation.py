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

"""Generation utilities."""
from typing import List

import torch
import torch.nn.functional as F

from megatron.training import get_args
from megatron.core import parallel_state

from .utils import pad_batch, top_k_logits, InferenceParams
from .forward_step import ForwardStep
from .beam_utils import BeamHypotheses
from .communication import broadcast_from_last_pipeline_stage


def beam_search(model, tokens, **kwargs):
    beam_size = kwargs.pop("beam_size", 1)
    stop_token = kwargs.pop("stop_token", 0)
    if not isinstance(stop_token, List):
        stop_token = [stop_token]
    num_return_gen = kwargs.pop("num_return_gen", 1)
    length_penalty = kwargs.pop("length_penalty", 1.0)
    args = get_args()

    if args.micro_batch_size > 1:
        raise NotImplementedError("The input prompt nums should not greater than 1 "
                                  "(i.e. micro_batch_size must be 1) in beam search mode.")

    # ==========================
    # Pad tokens
    # ==========================
    prompt_length, context_lengths, tokens = _pad_tokens(args, tokens, beam_size, num_return_gen)
    final_sequence_length = args.max_length_ori

    # ==========================
    # Forward step
    # ==========================
    forward_step = ForwardStep(model)

    # ==========================
    # InferenceParams
    # ==========================
    if args.use_kv_cache:
        inference_params = InferenceParams(beam_size, final_sequence_length)
    else:
        inference_params = None

    # ==========================
    # Build BeamHypotheses
    # ==========================
    beam_hyp = BeamHypotheses(beam_size, length_penalty)
    done = torch.zeros(1, dtype=torch.uint8, device=torch.cuda.current_device())
    scores = torch.zeros(beam_size, dtype=torch.float32, device=torch.cuda.current_device()).unsqueeze(1)
    scores_size_tensor, tokens_size_tensor = None, None
    output_scores, output_tokens = None, None

    # ==========================
    # Run inference
    # ==========================
    with torch.no_grad():
        tokens = tokens.repeat(beam_size, 1)
        _, seq_length = tokens.size()

        attention_mask = torch.tril(torch.ones(
            (args.micro_batch_size, seq_length, seq_length), device=tokens.device)).view(
            args.micro_batch_size, 1, seq_length, seq_length)
        attention_mask = (attention_mask < 0.5)
        position_ids = torch.arange(seq_length, dtype=torch.long,
                                    device=tokens.device)
        position_ids = position_ids.unsqueeze(0).expand_as(tokens)

        context_length, done, scores, tokens = yield from forward_loop(args,
                                                                       attention_mask=attention_mask,
                                                                       beam_hyp=beam_hyp,
                                                                       beam_size=beam_size,
                                                                       done=done,
                                                                       final_sequence_length=final_sequence_length,
                                                                       forward_step=forward_step,
                                                                       num_return_gen=num_return_gen,
                                                                       position_ids=position_ids,
                                                                       prompt_length=prompt_length,
                                                                       context_lengths=context_lengths,
                                                                       scores=scores,
                                                                       stop_token=stop_token,
                                                                       tokens=tokens,
                                                                       inference_params=inference_params)

        output_scores, output_tokens = _beam_search_post_process(beam_hyp=beam_hyp,
                                                                 beam_size=beam_size,
                                                                 done=done,
                                                                 num_return_gen=num_return_gen,
                                                                 output_scores=output_scores,
                                                                 output_tokens=output_tokens,
                                                                 context_length=context_length,
                                                                 prompt_length=prompt_length,
                                                                 scores=scores,
                                                                 scores_size_tensor=scores_size_tensor,
                                                                 tokens=tokens,
                                                                 tokens_size_tensor=tokens_size_tensor)

        yield output_tokens, context_lengths, torch.exp(output_scores)


def forward_loop(args, **kwargs):
    attention_mask = kwargs.pop("attention_mask")
    beam_hyp = kwargs.pop("beam_hyp")
    beam_size = kwargs.pop("beam_size")
    done = kwargs.pop("done")
    final_sequence_length = kwargs.pop("final_sequence_length")
    forward_step = kwargs.pop("forward_step")
    num_return_gen = kwargs.pop("num_return_gen")
    position_ids = kwargs.pop("position_ids")
    prompt_length = kwargs.pop("prompt_length")
    context_lengths = kwargs.pop("context_lengths")
    scores = kwargs.pop("scores")
    stop_token = kwargs.pop("stop_token")
    tokens = kwargs.pop("tokens")
    inference_params = kwargs.pop("inference_params")
    context_length = None

    for context_length in range(prompt_length, final_sequence_length):
        if inference_params:
            if context_length == prompt_length:
                input_tokens = tokens[:, :context_length]
                input_position_ids = position_ids[:, :context_length]
                input_attention_mask = attention_mask[:, :, :context_length, :context_length]
            else:
                input_tokens = tokens[:, context_length - 1:context_length]
                input_position_ids = position_ids[:, context_length - 1:context_length]
                input_attention_mask = attention_mask[:, :, context_length - 1:context_length, :context_length]
        else:
            input_tokens = tokens
            input_position_ids = position_ids
            input_attention_mask = attention_mask

        logits = forward_step(input_tokens, input_position_ids, input_attention_mask, inference_params)

        if parallel_state.is_pipeline_last_stage():
            vocab_size = logits.size(2)

            best_beam_ids, best_scores, best_words = _beam_candidates_with_sampling(args,
                                                                                    beam_size=beam_size,
                                                                                    context_length=context_length,
                                                                                    logits=logits,
                                                                                    prompt_length=prompt_length,
                                                                                    scores=scores,
                                                                                    stop_token=stop_token,
                                                                                    vocab_size=vocab_size,
                                                                                    inference_params=inference_params)

            done, scores, tokens = _beam_search_process(beam_hyp=beam_hyp,
                                                        beam_size=beam_size,
                                                        best_beam_ids=best_beam_ids,
                                                        best_scores=best_scores,
                                                        best_words=best_words,
                                                        context_length=context_length,
                                                        done=done,
                                                        prompt_length=prompt_length,
                                                        scores=scores,
                                                        stop_token=stop_token,
                                                        tokens=tokens,
                                                        inference_params=inference_params)

        done = broadcast_from_last_pipeline_stage(1, torch.uint8, done)
        if done:
            break

        tokens = broadcast_from_last_pipeline_stage(tokens.size(), torch.int64, tokens)

        yield tokens[:num_return_gen], context_lengths, torch.exp(scores[:num_return_gen])

    output_info = (context_length, done, scores, tokens)
    return output_info


def _beam_search_post_process(**kwargs):
    beam_hyp = kwargs.pop("beam_hyp")
    beam_size = kwargs.pop("beam_size")
    context_length = kwargs.pop("context_length")
    done = kwargs.pop("done")
    num_return_gen = kwargs.pop("num_return_gen")
    output_scores = kwargs.pop("output_scores")
    output_tokens = kwargs.pop("output_tokens")
    prompt_length = kwargs.pop("prompt_length")
    scores = kwargs.pop("scores")
    scores_size_tensor = kwargs.pop("scores_size_tensor")
    tokens = kwargs.pop("tokens")
    tokens_size_tensor = kwargs.pop("tokens_size_tensor")

    if parallel_state.is_pipeline_last_stage():
        if not done:
            for beam_id in range(beam_size):
                beam_hyp.add(tokens[beam_id].clone(),
                             scores[beam_id].squeeze(),
                             context_length + 1 - prompt_length)

        # rank based on scores
        sorted_hyps = sorted(beam_hyp.beams, key=lambda x: x[0], reverse=True)
        num_return_gen = min(num_return_gen, len(sorted_hyps))
        output_scores = [sorted_hyps[i][0] for i in range(num_return_gen)]
        output_tokens = [sorted_hyps[i][1] for i in range(num_return_gen)]
        output_scores = torch.stack(output_scores, dim=0)
        output_tokens = torch.stack(output_tokens, dim=0)
        scores_size_tensor = torch.tensor(output_scores.shape,
                                          dtype=torch.int64,
                                          device=torch.cuda.current_device())
        tokens_size_tensor = torch.tensor(output_tokens.shape,
                                          dtype=torch.int64,
                                          device=torch.cuda.current_device())
    scores_size_tensor = broadcast_from_last_pipeline_stage(1, torch.int64, scores_size_tensor)
    tokens_size_tensor = broadcast_from_last_pipeline_stage(2, torch.int64, tokens_size_tensor)
    output_scores = broadcast_from_last_pipeline_stage(tuple(scores_size_tensor),
                                                       torch.float32,
                                                       output_scores)
    output_tokens = broadcast_from_last_pipeline_stage(tuple(tokens_size_tensor),
                                                       torch.int64,
                                                       output_tokens)
    return output_scores, output_tokens


def _beam_search_process(**kwargs):
    beam_hyp = kwargs.pop("beam_hyp")
    beam_size = kwargs.pop("beam_size")
    best_beam_ids = kwargs.pop("best_beam_ids")
    best_scores = kwargs.pop("best_scores")
    best_words = kwargs.pop("best_words")
    context_length = kwargs.pop("context_length")
    done = kwargs.pop("done")
    prompt_length = kwargs.pop("prompt_length")
    scores = kwargs.pop("scores")
    stop_token = kwargs.pop("stop_token")
    tokens = kwargs.pop("tokens")
    inference_params = kwargs.pop("inference_params")

    next_beams = []
    for beam_token_rank, (token_id, beam_score, beam_id) in enumerate(
            zip(best_words, best_scores, best_beam_ids)
    ):
        if token_id.item() in stop_token:
            # if beam_token does not belong to top num_beams tokens, it should not be added
            is_beam_token_worse_than_top_num_beams = beam_token_rank >= beam_size
            if is_beam_token_worse_than_top_num_beams:
                continue
            beam_hyp.add(
                tokens[beam_id].clone(),
                beam_score,
                context_length + 1 - prompt_length
            )
        else:
            # add next predicted token since it is not eos_token
            next_beams.append((token_id, beam_score, beam_id))

        if len(next_beams) == beam_size:
            break

    if beam_hyp.is_done(best_scores.max().item(), context_length + 1 - prompt_length):
        done = torch.ones(1, dtype=torch.uint8, device=torch.cuda.current_device())
    best_batches = tokens.new([item[2] for item in next_beams])
    tokens = tokens[best_batches, :]
    tokens[:, context_length] = tokens.new([item[0] for item in next_beams])
    scores = scores.new([item[1] for item in next_beams]).unsqueeze(1)

    if inference_params:
        inference_params.swap_key_value_dict(best_batches)

    return done, scores, tokens


def _beam_candidates_with_sampling(args, **kwargs):
    beam_size = kwargs.pop("beam_size")
    context_length = kwargs.pop("context_length")
    logits = kwargs.pop("logits")
    prompt_length = kwargs.pop("prompt_length")
    scores = kwargs.pop("scores")
    vocab_size = kwargs.pop("vocab_size")
    stop_token = kwargs.pop("stop_token")
    inference_params = kwargs.pop("inference_params")

    if inference_params:
        index = -1
    else:
        index = context_length - 1

    try:
        logits = logits[:, index, :] / args.text_generation_config["temperature"]
    except ZeroDivisionError:
        logits = logits[:, index, :] * 10000

    if args.text_generation_config["top_k"] > 1 and (0.0 < args.text_generation_config["top_p"] <= 1.0):
        logits = top_k_logits(logits,
                              top_k=args.text_generation_config["top_k"],
                              top_p=args.text_generation_config["top_p"])

    log_probs = F.log_softmax(logits, dim=1)

    new_scores = log_probs + scores
    if context_length == prompt_length:
        indices, sorted_scores = _beam_candidates_at_beginning(args, beam_size, new_scores)
    else:
        indices, sorted_scores = _beam_candidates_at_later(args, beam_size, new_scores)

    best_beam_ids = torch.div(indices[: 2 * beam_size], vocab_size).trunc().long()
    best_words = indices[:2 * beam_size] % vocab_size
    best_scores = sorted_scores[: 2 * beam_size]

    return best_beam_ids, best_scores, best_words


def _beam_candidates_at_later(args, beam_size, new_scores):
    if args.text_generation_config['greedy']:
        sorted_scores, indices = torch.sort(new_scores.view(-1), descending=True)
    else:
        accumulate_logits = torch.exp(new_scores)
        accumulate_logits_sum = accumulate_logits.sum()
        if accumulate_logits_sum > 1e-5 and accumulate_logits_sum < 1.0:
            indices = torch.multinomial(accumulate_logits.view(-1), num_samples=2 * beam_size)
            sorted_scores = torch.gather(new_scores.view(-1), dim=0, index=indices)
        else:
            sorted_scores, indices = torch.sort(new_scores.view(-1), descending=True)

    return indices, sorted_scores


def _beam_candidates_at_beginning(args, beam_size, new_scores):
    if args.text_generation_config['greedy']:
        sorted_scores, indices = torch.sort(new_scores[0, :], descending=True)
    else:
        accumulate_logits = torch.exp(new_scores[0, :])
        accumulate_logits_sum = accumulate_logits.sum()
        if accumulate_logits_sum > 1e-5 and accumulate_logits_sum < 1.0:
            indices = torch.multinomial(accumulate_logits, num_samples=2 * beam_size)
            sorted_scores = torch.gather(new_scores[0, :], dim=0, index=indices)
        else:
            sorted_scores, indices = torch.sort(new_scores[0, :], descending=True)

    return indices, sorted_scores


def _pad_tokens(args, tokens, beam_size, num_return_gen):
    tokens, lengths = pad_batch(tokens, args)
    prompt_length = lengths.min().item()
    lengths = lengths.repeat(min(beam_size, num_return_gen)).cpu().numpy().tolist()

    return prompt_length, lengths, tokens
