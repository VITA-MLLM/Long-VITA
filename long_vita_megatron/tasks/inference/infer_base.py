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

import sys
import time
import shutil
import logging
import subprocess
from copy import deepcopy

import torch
from torch import distributed as dist

logging.basicConfig(format="")
logging.getLogger().setLevel(logging.INFO)


def add_text_generate_args(parser):
    group = parser.add_argument_group(title='text generation')
    group.add_argument("--task",
                       nargs='*',
                       default=None, help='The task id to run.')
    group.add_argument("--top-p", type=float, default=0.95, help='Top p sampling.')
    group.add_argument("--top-k", type=int, default=50, help='Top k sampling.')
    group.add_argument("--temperature", type=float, default=0.7, help='Sampling temperature.')
    group.add_argument("--max-length", type=int, default=256, help='Total length of text.')
    group.add_argument("--max-new-tokens", type=int, default=128, help='Size of the output generated text.')
    group.add_argument('--hf-chat-template', action='store_true', default=False,
                        help="Using Huggingface chat template")
    group.add_argument('--add-eos-token', nargs='+', type=str, default=[],
                        help="Use additional eos tokens")
    group.add_argument('--use-kv-cache', action="store_true", default=False,
                       help="Use kv cache to accelerate inference")                                     
    group.add_argument('--history-turns', type=int, default=3, help='Chat turns of histories.')
    return parser


def print_flush(prev_str, curr_str):
    difference = ''.join([char2 for char1, char2 in zip(prev_str, curr_str) if char1 != char2])

    if len(prev_str) < len(curr_str):
        difference += curr_str[len(prev_str):]

    sys.stdout.write(difference)


def task_factory(args, model):
    task_map = {
        "greedy": task_greedy_search,
        "do_sample": task_do_sample,
        "beam_search": task_beam_search,
        "beam_search_with_sampling": task_beam_search_with_sampling,
        "chat": task_chat,
    }

    total_tasks = args.task

    if total_tasks is None:
        total_tasks = [
            "greedy",
            "do_sample",
            "beam_search",
            "beam_search_with_sampling",
            "chat"
        ]

    for task in total_tasks:
        if task not in task_map.keys():
            raise ValueError("Task name incorrect.")

        task_map.get(task)(
            args,
            model,
        )


def task_greedy_search(args, model):
    """Greedy Search"""
    instruction = "how are you?"

    t = time.time()
    output = model.generate(
        [instruction],
        do_sample=False,
        max_new_tokens=args.max_new_tokens,
        stream=False
    )

    if dist.get_rank() == 0:
        logging.info("\n=============== Greedy Search ================")
        logging.info("\nYou:\n%s\n\nModelLink:\n%s", instruction, output)
        logging.info("==============================================")
        logging.info("\nElapsed: %ss", round(time.time() - t, 2))

    dist.barrier()


def task_do_sample(args, model):
    """Do Sample"""
    instruction = "how are you?"

    t = time.time()
    output = model.generate(
        [instruction],
        do_sample=True,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        stream=False
    )

    if dist.get_rank() == 0:
        logging.info("\n================ Do Sample =================")
        logging.info("\nYou:\n%s\n\nModelLink:\n%s", instruction, output)
        logging.info("============================================")
        logging.info("\nElapsed: %ss", round(time.time() - t, 2))

    dist.barrier()


def task_beam_search(args, model):
    """Beam Search"""
    instruction = "how are you?"

    t = time.time()
    output = model.generate(
        [instruction],
        num_beams=2,
        do_sample=False,
        max_new_tokens=args.max_new_tokens,
        stream=False
    )

    if dist.get_rank() == 0:
        logging.info("\n=============== Beam Search =================")
        logging.info("\nYou:\n%s\n\nModelLink:\n%s", instruction, output)
        logging.info("=============================================")
        logging.info("\nElapsed: %ss", round(time.time() - t, 2))

    dist.barrier()


def task_beam_search_with_sampling(args, model):
    """Beam Search with sampling"""
    instruction = "how are you?"

    t = time.time()
    output = model.generate(
        [instruction],
        num_beams=2,
        do_sample=True,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        stream=False
    )

    if dist.get_rank() == 0:
        logging.info("\n======== Beam Search with sampling ==========")
        logging.info("\nYou:\n%s\n\nModelLink:\n%s", instruction, output)
        logging.info("=============================================")
        logging.info("\nElapsed: %ss", round(time.time() - t, 2))

    dist.barrier()


def chat_get_instruction(args, histories_no_template, histories_template, prompt):
    instruction = None

    def get_context(content):
        res = ""
        for q, r in content:
            if r is None:
                res += q
            else:
                res += q + r
        return res

    if args.hf_chat_template or args.prompt_type is not None:
        # Handle conversation history, there can be a better solution
        if len(histories_template) > 2 * args.history_turns:
            histories_template.pop(0)
            histories_template.pop(0)

        histories_template.append({"role": "user", "content": prompt})

        # use llamafactory template, We need to build the intermediate format ourselves 
        instruction = deepcopy(histories_template) 
    else:
        # not use llamafactory template,keep old process
        histories_no_template.append((prompt, None))
        instruction = get_context(histories_no_template)
        histories_no_template.pop()

    return instruction


def chat_print_and_update_histories(args, responses, histories_no_template, histories_template, prompt):
    response_template = "\nModelLink:\n"
    output = ""

    if dist.get_rank() == 0:
        sys.stdout.write(response_template)

    prev = ""
    for output in responses:
        if dist.get_rank() == 0:
            curr = output.replace("�", "")
            print_flush(prev, curr)
            prev = curr

    # old propress
    if args.hf_chat_template or args.prompt_type is not None:
        histories_template.append({"role": "assistant", "content": output})
    else:
        histories_no_template.append((prompt, output))
        if len(histories_no_template) > 3:
            histories_no_template.pop()

    return output


def task_chat(args, model):
    """Interactive dialog mode with multiple rounds of conversation"""

    histories_no_template = []
    histories_template = []
    instruction = None
    prompt = ""
    input_template = "\n\nYou >> "
    command_clear = ["clear"]

    while True:
        terminate_runs = torch.zeros(1, dtype=torch.int64, device=torch.cuda.current_device())

        if dist.get_rank() == 0:
            if not histories_no_template and not histories_template:
                logging.info("===========================================================")
                logging.info("1. If you want to quit, please entry one of [q, quit, exit]")
                logging.info("2. To create new title, please entry one of [clear, new]")
                logging.info("===========================================================")

            prompt = input(input_template)
            # remove non utf-8 characters
            prompt = prompt.encode('utf-8', errors='ignore').decode('utf-8')
            if prompt.strip() in ["q", "exit", "quit"]:
                terminate_runs += 1

            if prompt.strip() in ["clear", "new"]:
                subprocess.call(command_clear)
                histories_no_template = []
                histories_template = []
                continue

            if not prompt.strip():
                continue
            
            instruction = chat_get_instruction(args, histories_no_template, histories_template, prompt)


        dist.all_reduce(terminate_runs)
        dist.barrier()
        if terminate_runs > 0:
            break

        responses = model.generate(
            instruction,
            do_sample=True,
            top_k=args.top_k,
            top_p=args.top_p,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            stream=True
        )

        chat_print_and_update_histories(args, responses, histories_no_template, histories_template, prompt)
