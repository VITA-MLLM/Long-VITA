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

import torch

from megatron.training import get_args
from megatron.inference.text_generation.communication import broadcast_int_list, broadcast_tensor
from long_vita_megatron.tasks.preprocess.templates import Template, get_model_template


def _encode_no_template(tokenizer, prompts):
    prompts_tokens = [[]]
    if isinstance(prompts, str):
        prompts_tokens = [tokenizer.encode(prompts)]
    elif torch.is_tensor(prompts):
        if len(prompts.shape) == 1:
            prompts_tokens = prompts.unsqueeze(0).numpy().tolist()
        elif len(prompts.shape) == 2:
            prompts_tokens = prompts.numpy().tolist()
    elif isinstance(prompts, (tuple, list)):
        if len(prompts) and isinstance(prompts[0], (tuple, list)):
            prompts_tokens = prompts
        elif len(prompts) and isinstance(prompts[0], int):
            prompts_tokens = [prompts]
        elif len(prompts) and isinstance(prompts[0], str):
            prompts_tokens = [tokenizer.encode(val) for val in prompts]
    else:
        raise TypeError("Please check input_ids in correct type.")

    return prompts_tokens


def _encode_by_template(template, tokenizer, prompts):
    prompts_tokens = []

    if prompts is None:
        return [[]]
    response_prompt = [{"role": "assistant", "content": ""}]
    if len(prompts) and isinstance(prompts, str):
        paired_messages = [{"role": "user", "content": "{}".format(prompts)}] + response_prompt
        tokens, _ = template.encode_oneturn(tokenizer=tokenizer, messages=paired_messages, tools="")
        prompts_tokens.append(tokens)
    elif len(prompts) and isinstance(prompts[0], (dict)):
        paired_messages = prompts + response_prompt
        tokens, _ = template.encode_oneturn(tokenizer=tokenizer, messages=paired_messages, tools="")
        prompts_tokens.append(tokens)
    elif len(prompts) and isinstance(prompts[0], (str)):
        for query in prompts:
            paired_messages = [{"role": "user", "content": "{}".format(query)}] + response_prompt
            tokens, _ = template.encode_oneturn(tokenizer=tokenizer, messages=paired_messages, tools="")
            prompts_tokens.append(tokens)
    elif len(prompts) and isinstance(prompts[0], (tuple, list)):
        for val in prompts:
            if len(val) and isinstance(val, (tuple, list)):
                paired_messages = val + response_prompt
                tokens, _ = template.encode_oneturn(tokenizer=tokenizer, messages=paired_messages, tools="")
                prompts_tokens.append(tokens)
    else:
        raise TypeError("Please check input_ids in correct type.")

    return prompts_tokens if len(prompts_tokens) > 0 else [prompts_tokens]


def tokenize_prompts(tokenizer=None, prompts=None, tokens_to_generate=None, max_generate_length=128,
                     add_BOS=None, rank=0):
    """Tokenize prompts and make them avaiable on all ranks."""

    # On all ranks set to None so we can pass them to functions
    sizes_list = None
    prompts_tokens_cuda_long_tensor = None
    prompts_length_cuda_long_tensor = None

    # On the specified rank, build the above.
    if torch.distributed.get_rank() == rank:
        assert prompts is not None
        assert tokens_to_generate is not None
        # Tensor of tokens padded and their unpadded length.
        prompts_tokens_cuda_long_tensor, prompts_length_cuda_long_tensor = \
            _tokenize_prompts_and_batch(tokenizer, prompts, tokens_to_generate, max_generate_length, add_BOS)
        # We need the sizes of these tensors for the boradcast
        sizes_list = [prompts_tokens_cuda_long_tensor.size(0), # Batch size
                      prompts_tokens_cuda_long_tensor.size(1)] # Sequence lenght

    # First, broadcast the sizes.
    sizes_tensor = broadcast_int_list(2, int_list=sizes_list, rank=rank)

    # Now that we have the sizes, we can boradcast the tokens
    # and length tensors.
    sizes = sizes_tensor.tolist()
    prompts_tokens_cuda_long_tensor = broadcast_tensor(
        sizes, torch.int64, tensor=prompts_tokens_cuda_long_tensor, rank=rank)
    prompts_length_cuda_long_tensor = broadcast_tensor(
        sizes[0], torch.int64, tensor=prompts_length_cuda_long_tensor,
        rank=rank)

    return prompts_tokens_cuda_long_tensor, prompts_length_cuda_long_tensor


def _tokenize_prompts_and_batch(tokenizer, prompts, tokens_to_generate, max_generate_length, add_BOS):
    """Given a set of prompts and number of tokens to generate:
        - tokenize prompts
        - set the sequence length to be the max of length of prompts
          plus the number of tokens we would like to generate
        - pad all the sequences to this length so we can convert them
          into a 2D tensor.
    """
    
    args = get_args()
    template = None

    # Tokenize all the prompts.
    if hasattr(args, "prompt_type") and args.prompt_type is not None:
        template = get_model_template(args.prompt_type.strip(), args.prompt_type_path.strip())

    if args.hf_chat_template:
        if not hasattr(tokenizer, "apply_chat_template"):
            raise AssertionError('The tokenizer has no Huggingface chat template, Please use chat model.')

        prompts_tokens = [tokenizer.apply_chat_template(
            prompts,
            tokenize=True,
            add_generation_prompt=True
        )]
    elif template is None:
        prompts_tokens = _encode_no_template(tokenizer, prompts)
    else:
        prompts_tokens = _encode_by_template(template, tokenizer, prompts)

    if add_BOS:
        prompts_tokens = [[tokenizer.eod] + prompt
                          for prompt in prompts_tokens]

    # Now we have a list of list of tokens which each list has a different
    # size. We want to extend this list to:
    #   - incorporate the tokens that need to be generated
    #   - make all the sequences equal length.
    # Get the prompts length.
    prompts_length = [len(prompt_tokens) for prompt_tokens in prompts_tokens]
    # Get the max prompts length.
    max_prompt_len = max(prompts_length)

    if tokens_to_generate > 0:
        max_samples_length = max_prompt_len + tokens_to_generate
    else:
        max_samples_length = max_generate_length

    # Now update the list of list to be of the same size: max_samples_length.
    for prompt_tokens, prompt_length in zip(prompts_tokens, prompts_length):
        padding_size = max_samples_length - prompt_length
        prompt_tokens.extend([tokenizer.pad_token_id] * padding_size)

    # Now we are in a structured format, we can convert to tensors.
    prompts_tokens_tensor = torch.tensor(prompts_tokens, dtype=torch.long, device='cuda')
    prompts_length_tensor = torch.tensor(prompts_length, dtype=torch.long, device='cuda')

    return prompts_tokens_tensor, prompts_length_tensor
