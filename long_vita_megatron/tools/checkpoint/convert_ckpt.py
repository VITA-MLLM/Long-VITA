# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import argparse
import importlib
import os
import sys
from functools import wraps
import torch.multiprocessing as mp
try:
    import modellink
    import long_vita_modellink.megatron_adaptor
except Exception as e:
    print(e)


def load_plugin(plugin_type, name):
    module_name = f"{plugin_type}_{name}"
    # try:
    if True:
        plugin = importlib.import_module(module_name)
    # except ModuleNotFoundError:
    #     module_name = name
    #     try:
    #         plugin = importlib.import_module(module_name)
    #     except ModuleNotFoundError:
    #         sys.exit(f"Unable to load {plugin_type} plugin {name}. Exiting.")

    if not hasattr(plugin, 'add_arguments'):
        sys.exit(f"{module_name} module is not a plugin. Exiting.")

    print(f"Loaded {module_name} as the {plugin_type}.")
    return plugin


def main():

    parser = argparse.ArgumentParser(description="Megatron Checkpoint Utility Arguments",
                                     allow_abbrev=False, conflict_handler='resolve')

    parser.add_argument('--model-type', type=str, required=True,
                        choices=['GPT', 'BERT', "EVA"],
                        help='Type of the model')
    parser.add_argument('--loader', type=str, default='megatron',
                        help='Module name to load checkpoint, should be on python path')
    parser.add_argument('--saver', type=str, default='megatron',
                        help='Module name to save checkpoint, should be on python path')
    parser.add_argument('--load-dir', type=str, required=True,
                        help='Directory to load model checkpoint from')
    parser.add_argument('--save-dir', type=str, required=True,
                        help='Directory to save model checkpoint to')
    parser.add_argument('--max-queue-size', type=int, default=50,
                        help='Maximum number of tensors in the queue')
    parser.add_argument('--no-checking', action='store_false',
                        help='Do not perform checking on the name and ordering of weights',
                        dest='checking')
    parser.add_argument('--model-type-hf', type=str, default="llama2",
                        choices=['llama2', 'mixtral', 'chatglm3', 'gemma', 'gemma2', 'bloom', 'qwen', 'internlm2', 'deepseek2'], help='model-type')
    known_args, _ = parser.parse_known_args()
    loader = load_plugin('loader', known_args.loader)
    saver = load_plugin('saver', known_args.saver)

    loader.add_arguments(parser)
    saver.add_arguments(parser)

    args = parser.parse_args()

    queue = mp.Queue(maxsize=args.max_queue_size)

    print("Starting saver...")
    saver_proc = mp.Process(target=saver.save_model_checkpoint, args=(queue, args))
    saver_proc.start()

    print("Starting loader...")
    loader.load_checkpoint(queue, args)

    print("Waiting for saver to complete...")
    saver_proc.join()


if __name__ == '__main__':
    main()

