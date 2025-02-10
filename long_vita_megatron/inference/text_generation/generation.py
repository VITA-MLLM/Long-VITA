# coding=utf-8
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

import torch
import torch.nn.functional as F

from megatron.training import get_args, get_tokenizer
from megatron.core import mpu
from megatron.inference.text_generation.communication import (
    copy_from_last_to_first_pipeline_stage,
    broadcast_from_last_pipeline_stage,
    broadcast_from_last_to_first_pipeline_stage)
from megatron.inference.text_generation.forward_step import ForwardStep
from megatron.inference.text_generation.beam_utils import BeamHypotheses
from megatron.inference.text_generation.generation import _build_attention_mask_and_position_ids


def generate_tokens_probs_and_return_on_first_stage(
        model, tokens, lengths,
        return_output_log_probs=False,
        do_sample=False,
        top_k=0, top_p=0.0,
        temperature=1.0,
        use_eod_token_for_early_termination=True,
        external_inputs=None,
        ):
    """Main token generation function.

    Args:
        model: no interleaving is supported.
        tokens: prompt tokens extended to be of size [b, max-sequence-length]
        lengths: original prompt length, size: [b]
        return_output_log_probs: flag to calculate the log probability of
            the generated tokens. Note that the log probability is the one
            from the original logit.
        top_k, top_p: top-k and top-p sampling parameters.
        temperature: sampling temperature.
        use_eod_token_for_early_termination: if True, do early termination if
            all the sequences have reached this token.
    Note: Outside of model, other parameters only need to be available on
          rank 0.

    Returns: Note that is size is adjusted to a lower value than
             max-sequence-length if generation is terminated early.
        tokens: prompt and generated tokens. size: [b, :]
        lengths: original prompt length, size: [b]
        output_log_probs: log probability of the tokens. size: [b, s, vocab_size]
    """

    args = get_args()
    tokenizer = get_tokenizer()

    batch_size = tokens.size(0)
    min_prompt_length = lengths.min().item()
    max_sequence_length = tokens.size(1)

    if max_sequence_length > args.max_position_embeddings:
        raise ValueError("Length of prompt + tokens_to_generate longer than allowed")
    
    if max_sequence_length * batch_size > args.max_tokens_to_oom:
        raise ValueError("Too many tokens.  " + str(max_sequence_length*batch_size) + " is greater than " + str(args.max_tokens_to_oom))

    # forward step.
    forward_step = ForwardStep(model, batch_size, max_sequence_length, external_inputs=external_inputs)

    # Added termination_id to support the case that we want to terminate the
    # generation once that id is generated.
    if hasattr(args, 'eos_id'):
        termination_id = args.eos_id
    else:
        termination_id = tokenizer.eod

    # ===================
    # Pre-allocate memory
    # ===================

    # Log probability of the sequence (prompt + generated tokens).
    output_log_probs = None
    output_log_probs_size = (batch_size, max_sequence_length - 1, args.padded_vocab_size)
    # Lengths of generated seuquence including including prompts.
    generated_sequence_lengths = None
    if mpu.is_pipeline_last_stage():
        if return_output_log_probs:
            output_log_probs = torch.empty(output_log_probs_size,
                                           dtype=torch.float32,
                                           device=torch.cuda.current_device())
        generated_sequence_lengths = torch.ones(
                batch_size, dtype=torch.int64,
                device=torch.cuda.current_device()) * max_sequence_length
    
    # Whether we have reached a termination id.
    is_generation_done = torch.zeros(batch_size, dtype=torch.uint8,
                                     device=torch.cuda.current_device())

    # =============
    # Run infernece
    # =============

    with torch.no_grad():
        # attention_mask, position_ids = _build_attention_mask_and_position_ids(
        #     tokens)

        _, seq_length = tokens.size()
        position_ids = torch.arange(seq_length, dtype=torch.long, device=tokens.device)
        position_ids = position_ids.unsqueeze(0).expand_as(tokens)

        prev_context_length = 0
        for context_length in range(min_prompt_length, max_sequence_length):

            # start of megatron_adaptation, here we change sample stratrgy
            # Pick the slice that we need to pass through the network.
            if args.use_kv_cache:
                tokens2use = tokens[:, prev_context_length:context_length]
                positions2use = position_ids[:, prev_context_length:context_length]
                # attention_mask2use = attention_mask[
                #     ..., prev_context_length:context_length, :context_length]
            else:
                tokens2use = tokens
                positions2use = position_ids
                # attention_mask2use = attention_mask

            tokens2use, positions2use, external_inputs2use = get_batch_on_this_cp_rank(tokens2use, positions2use, external_inputs)
            attention_mask2use = None
            forward_step.inference_params.external_inputs = external_inputs2use

            if args.logit_mask:
                logit_mask = torch.zeros_like(tokens2use).bool()
                cp_size = mpu.get_context_parallel_world_size()
                if cp_size == 1:
                    if args.use_kv_cache:
                        logit_mask[:, -1] = 1
                    else:
                        logit_mask[:, context_length - 1] = 1
                else:
                    # cp_token_length = tokens2use.size(1)
                    # cp_context_length = context_length % cp_token_length
                    # logit_mask[:, cp_context_length - 1] = 1

                    cp_token_length = int(tokens2use.size(1) / 2)
                    cp_context_length = context_length % cp_token_length
                    cp_output_block = context_length // cp_token_length
                    logit_mask[:, cp_context_length - 1] = 1
                    logit_mask[:, cp_token_length + cp_context_length - 1] = 1
                # print(f"context_length {context_length}")
                # print(f"cp_token_length {cp_token_length}")
                # print(f"cp_output_block {cp_output_block}")
                # print(f"cp_context_length {cp_context_length}")
            else:
                logit_mask = None
            forward_step.inference_params.logit_mask = logit_mask

            if not args.use_kv_cache:
                forward_step.inference_params.use_kv_cache = False

            # logits will be meanigful only in the last pipeline stage.
            logits = forward_step(tokens2use, positions2use, attention_mask2use)

            # if args.logit_mask:
            #     cp_size = mpu.get_context_parallel_world_size()
            #     if cp_size == 1:
            #         pass
            #     else:
            #         logits = logits.repeat(1, 2, 1).contiguous()
            logits = sync_output(logits)

            if mpu.is_pipeline_last_stage():
                # Always the last stage should have an output.
                assert logits is not None

                # Sample.
                if args.logit_mask:
                    cp_size = mpu.get_context_parallel_world_size()
                    if cp_size == 1:
                        if args.use_kv_cache:
                            last_token_logits = logits[:, -1, :]
                        else:
                            last_token_logits = logits[:, -1, :]
                    else:
                        # last_token_logits = logits[:, -1, :]
                        last_token_logits = logits[:, cp_output_block, :]
                else:
                    cp_size = mpu.get_context_parallel_world_size()
                    if cp_size == 1:
                        if args.use_kv_cache:
                            last_token_logits = logits[:, -1, :]
                        else:
                            last_token_logits = logits[:, context_length - 1, :]
                    else:
                        # last_token_logits = logits[:, context_length % tokens2use.size(1) - 1, :]
                        last_token_logits = logits[:, context_length - 1, :]

                _, new_sample = _sample_strategy(last_token_logits,
                                    do_sample=do_sample,
                                    top_k=top_k,
                                    top_p=top_p,
                                    temperature=temperature)
                
                # end of megatron_adaptation
                
                # If a prompt length is smaller or equal th current context
                # length, it means we have started generating tokens
                started = lengths <= context_length
                # Update the tokens.
                tokens[started, context_length] = new_sample[started]

                # Calculate the log probabilities.
                if return_output_log_probs:
                    last_token_logits = F.log_softmax(last_token_logits, dim=1)
                    output_log_probs[:, context_length - 1, :] = last_token_logits

            # Update the tokens on the first stage so the next input to
            # the network is correct.
            copy_from_last_to_first_pipeline_stage(batch_size, torch.int64,
                                                   tokens[:, context_length])

            # Update the context length for the next token generation.
            prev_context_length = context_length

            # Check if all the sequences have hit the termination_id.
            done = None
            if mpu.is_pipeline_last_stage():
                # TODO(rprenger) These stopping methods are tokenizer dependent
                # instead tokenization should be in the inference loop so stop sequences can be used
                done_token = (new_sample == termination_id).byte() & \
                        started.byte()
                
                just_finished = (done_token & ~is_generation_done).bool()
                generated_sequence_lengths[just_finished.view(-1)] = \
                    context_length + 1
                is_generation_done = is_generation_done | done_token
                done = torch.all(is_generation_done)

            if output_log_probs is None and False:
                output_log_probs = torch.empty(output_log_probs_size,
                                        dtype=torch.float32,
                                        device=torch.cuda.current_device())

            yield tokens[:, :(context_length + 1)], lengths, output_log_probs

            done = broadcast_from_last_pipeline_stage(1, torch.uint8,
                                                      tensor=done)
            if use_eod_token_for_early_termination and done:
                break
            
    # ===================================================
    # Update the length of based on max generated length.
    # ===================================================

    tokens = tokens[:, :(context_length + 1)]
    if mpu.is_pipeline_last_stage():
        if return_output_log_probs:
            output_log_probs = output_log_probs[:, :context_length]

    # ======================================
    # Broadcast to the first pipeline stage.
    # ======================================

    generated_sequence_lengths = broadcast_from_last_to_first_pipeline_stage(
        batch_size, torch.int64, generated_sequence_lengths)
    if return_output_log_probs:
        output_log_probs_size = (batch_size, context_length, args.padded_vocab_size)
        output_log_probs = broadcast_from_last_to_first_pipeline_stage(
            output_log_probs_size, torch.float32, output_log_probs)

    return tokens, lengths, output_log_probs


def beam_search_and_return_on_first_stage(
        model, tokens=None, lengths=0, 
        beam_size=0, 
        do_sample=False,
        stop_token=None, 
        num_return_gen=1, 
        length_penalty=1, 
        top_k=0, top_p=0.0,
        temperature=1.0):
    args = get_args()
    tokenizer = get_tokenizer()

    batch_size = tokens.size(0)
    assert(batch_size == 1)
    prompt_length = lengths.item()
    final_sequence_length = tokens.size(1)
    final_sequence_length = min(final_sequence_length, args.max_position_embeddings)

    # Added termination_id to support the case that we want to terminate the
    # generation once that id is generated.
    if hasattr(args, 'eos_id'):
        termination_id = args.eos_id
    else:
        termination_id = tokenizer.eod
    
    # If the context is too big, this happens
    if prompt_length >= final_sequence_length:
        raise ValueError("context length + tokens_to_generate too large")

    # forward step.
    forward_step = ForwardStep(model, beam_size, final_sequence_length)

    beam_hyp = BeamHypotheses(beam_size, length_penalty)
    best_batches = None
    done = torch.zeros(1, dtype=torch.uint8, device=torch.cuda.current_device())
    scores = torch.zeros(beam_size,
                         dtype=torch.float32,
                         device=torch.cuda.current_device()).unsqueeze(1)
    scores_size_tensor, tokens_size_tensor = None, None
    # =============
    # Run infernece
    # =============
    with torch.no_grad():
        tokens = tokens.repeat(beam_size, 1)
        lengths = lengths.repeat(beam_size, 1)
        attention_mask, position_ids = _build_attention_mask_and_position_ids(tokens)
        prev_context_length = 0
        for context_length in range(prompt_length, final_sequence_length):

            # start of megatron_adaptation, here we change sample stratrgy
            # Pick the slice that we need to pass through the network.
            if args.use_kv_cache:
                tokens2use = tokens[:, prev_context_length:context_length]
                positions2use = position_ids[:, prev_context_length:context_length]
                attention_mask2use = attention_mask[
                    ..., prev_context_length:context_length, :context_length]
            else:
                tokens2use = tokens
                positions2use = position_ids
                attention_mask2use = attention_mask

            # logits will be meanigful only in the last pipeline stage.
            logits = forward_step(tokens2use, positions2use, attention_mask2use)

            if mpu.is_pipeline_last_stage():
                vocab_size = logits.size(2)

                if args.use_kv_cache:
                    logits = logits[:, -1, :]
                else:
                    logits = logits[:, context_length - 1, :]

                try:
                    logits = logits / temperature
                except ZeroDivisionError:
                    logits = logits * 10000

                if top_k > 1 and (0.0 < top_p <= 1.0):
                    logits = top_k_logits(logits,
                                        top_k=top_k,
                                        top_p=top_p)

                log_probs = F.log_softmax(logits, dim=1)

                new_scores = log_probs + scores

                if context_length == prompt_length:  # if this is the first one
                    indices, sorted_scores = beam_candidates(do_sample, beam_size, new_scores, at_beginning=True)
                else:
                    indices, sorted_scores = beam_candidates(do_sample, beam_size, new_scores, at_beginning=False)

                best_beam_ids = torch.div(indices[: 2 * beam_size], vocab_size).trunc().long()
                best_words = indices[:2 * beam_size] % vocab_size
                best_scores = sorted_scores[: 2 * beam_size]

                # end of megatron_adaptation

                next_beams = []
                for beam_token_rank, (token_id, beam_score, beam_id) in enumerate(
                    zip(best_words, best_scores, best_beam_ids)
                ):
                    if token_id.item() == termination_id:
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
          
            # torch.distributed.barrier()
            done = broadcast_from_last_pipeline_stage(1, torch.uint8, done)
            if done:
                break

            # Update the tokens on the first stage so the next input to
            # the network is correct.
            copy_from_last_to_first_pipeline_stage(tokens.size(), torch.int64,
                                                   tokens)

            # set inference key values to make it consistent with best beam index
            best_batches = broadcast_from_last_pipeline_stage(beam_size, torch.int64, best_batches)
            if forward_step.inference_params:
                forward_step.inference_params.swap_key_value_dict(best_batches)

            # Update the context length for the next token generation.
            prev_context_length = context_length

            yield tokens[:num_return_gen, :(context_length + 1)], lengths, scores[:num_return_gen]

        if mpu.is_pipeline_last_stage():
            # if cannot find stop token, add open beams to hyps
            if not done:
                for beam_id in range(beam_size):
                    beam_hyp.add(tokens[beam_id].clone(), scores[beam_id].squeeze(), context_length + 1 - prompt_length)

            # rank based on scores
            sorted_hyps = sorted(beam_hyp.beams, key=lambda x: x[0], reverse=True)
            num_return_gen = min(num_return_gen, len(sorted_hyps))
            scores = [sorted_hyps[i][0] for i in range(num_return_gen)]
            tokens = [sorted_hyps[i][1] for i in range(num_return_gen)]
            scores = torch.stack(scores, dim=0)
            tokens = torch.stack(tokens, dim=0)
            scores_size_tensor = torch.tensor(scores.shape, dtype=torch.int64, device=torch.cuda.current_device())
            tokens_size_tensor = torch.tensor(tokens.shape, dtype=torch.int64, device=torch.cuda.current_device())

        scores_size_tensor = broadcast_from_last_pipeline_stage(1, torch.int64, scores_size_tensor)
        tokens_size_tensor = broadcast_from_last_pipeline_stage(2, torch.int64, tokens_size_tensor)

        scores = broadcast_from_last_to_first_pipeline_stage(tuple(scores_size_tensor), torch.float32, scores)
        tokens = broadcast_from_last_to_first_pipeline_stage(tuple(tokens_size_tensor), torch.int64, tokens)

    return tokens, lengths, scores


def beam_candidates(do_sample, beam_size, new_scores, at_beginning=False):
    if at_beginning:
        new_scores = new_scores[0, :]
    else:
        new_scores = new_scores.view(-1)

    if not do_sample:
        sorted_scores, indices = torch.sort(new_scores, descending=True)
    else:
        accumulate_logits = torch.exp(new_scores)
        accumulate_logits_sum = accumulate_logits.sum()
        if accumulate_logits_sum > 1e-5 and accumulate_logits_sum < 1.0:
            indices = torch.multinomial(accumulate_logits, num_samples=2 * beam_size)
            sorted_scores = torch.gather(new_scores, dim=0, index=indices)
        else:
            sorted_scores, indices = torch.sort(new_scores, descending=True)

    return indices, sorted_scores


def _sample_strategy(logits, do_sample, top_k=0, top_p=0.0, temperature=1.0):
    if not do_sample:
        prev = torch.argmax(logits, dim=-1).view(-1)
    else:
        logits = logits.float()
        logits /= temperature
        logits = top_k_logits(logits,
                              top_k=top_k,
                              top_p=top_p)
        logits = F.softmax(logits, dim=-1)
        prev = torch.multinomial(logits, num_samples=1).view(-1)
    return logits, prev


def top_k_logits(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):

    if top_k > 0:
        # Remove all tokens with a probability less than the
        # last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Cconvert to 1D
        sorted_logits, sorted_indices = torch.sort(
            logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1),
                                        dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token
        # above the threshold
        sorted_indices_to_remove[..., 1:] \
            = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        for i in range(sorted_indices.size(0)):
            indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
            logits[i][indices_to_remove] = filter_value

    return logits


def get_batch_on_this_cp_rank(tokens, position_ids, external_inputs):
    cp_size = mpu.get_context_parallel_world_size()
    cp_rank = mpu.get_context_parallel_rank()
    if cp_size == 1:
        return tokens, position_ids, external_inputs

    batch = {}
    batch["tokens"] = tokens
    batch["position_ids"] = position_ids
    batch["external_images"] = external_inputs["images"]
    batch["external_indices"] = external_inputs["indices"]

    from long_vita_megatron.training.utils import get_batch_on_this_cp_rank
    batch = get_batch_on_this_cp_rank(batch)

    tokens = batch["tokens"]
    position_ids = batch["position_ids"]
    external_inputs = {
        "images": batch["external_images"],
        "src_indices": batch["external_src_indices"],
        "tgt_indices": batch["external_tgt_indices"],
    }
    return tokens, position_ids, external_inputs


def sync_output(output):
    cp_size = mpu.get_context_parallel_world_size()
    cp_rank = mpu.get_context_parallel_rank()
    if cp_size == 1:
        return output

    output_list = [torch.empty_like(output) for _ in range(cp_size)]
    output_list[cp_rank] = output
    torch.distributed.all_gather(output_list, output, group=mpu.get_context_parallel_group())
    # print(f"output_list {len(output_list)}")
    output_list = [xx for x in output_list for xx in x.chunk(2, dim=1)]
    # print(f"output_list {len(output_list)}")

    index = torch.tensor([cp_rank, (2 * cp_size - cp_rank - 1)], device=output.device)
    index_list = [torch.empty_like(index) for _ in range(cp_size)]
    index_list[cp_rank] = index
    torch.distributed.all_gather(index_list, index, group=mpu.get_context_parallel_group())
    index = torch.cat(index_list)
    _, index = torch.sort(index, dim=0, descending=False)

    # print(f"index_list {index_list} index {index}")
    output_list = [output_list[x] for x in index.tolist()]
    output = torch.cat(output_list, dim=1)

    return output
