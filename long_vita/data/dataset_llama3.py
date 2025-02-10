import json
import logging
import os
import pdb
import random
import re
import time
import traceback
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np
import torch
import transformers
from transformers.trainer_pt_utils import LabelSmoother

from .dataset_base import BaseDataset

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


class Llama3Dataset(BaseDataset):
    def __init__(
        self,
        cfg_path,
        tokenizer,
        image_size=448,
        image_token_length=1024,
        model_max_length=32768,
        variable_length=False,
        output_dir="",
        training_args=None,
        add_task_symbol=True,
        change_path=False,
        shift_token=False,
        create_attention_mask=True,
        create_position_ids=False,
        create_attention_mask_2d=False,
        create_loss_mask=False,
    ):
        super().__init__(
            cfg_path,
            tokenizer,
            image_size=image_size,
            image_token_length=image_token_length,
            model_max_length=model_max_length,
            variable_length=variable_length,
            output_dir=output_dir,
            training_args=training_args,
            add_task_symbol=add_task_symbol,
            shift_token=shift_token,
            create_attention_mask=create_attention_mask,
            create_position_ids=create_position_ids,
            create_attention_mask_2d=create_attention_mask_2d,
            create_loss_mask=create_loss_mask,
        )

        self.system_message = "You are a helpful AI assistant."

    def __getitem__(self, index):
        while True:
            try:
                # if True:
                ret_batch = preprocess(
                    [self.raw_data[index]],
                    self.tokenizer,
                    self.max_len,
                    self.image_token_length,
                    variable_length=self.variable_length,
                    system_message=self.system_message,
                    shift_token=self.shift_token,
                    image_processor=self.processor["image"],
                )

                input_ids = ret_batch["input_ids"][0]
                labels = ret_batch["labels"][0]
                attention_mask = ret_batch["attention_mask"][0]

                ret = dict(
                    # input_ids=input_ids,
                    tokens=input_ids,
                    labels=labels,
                    # attention_mask=attention_mask,
                    # image_paths=image_paths,
                    # images=image_tensor,
                    # doclm_images=doclm_image_tensor,
                )

                if "images" in ret_batch and len(ret_batch["images"][0]) > 0:
                    images = ret_batch["images"][0]
                    # image_tensor, doclm_image_tensor = self.process_images(images)

                    # print("image_tensor", image_tensor.size(), flush=True)
                    # print("doclm_image_tensor", doclm_image_tensor.size(), flush=True)
                    # print("images", images.size(), flush=True)

                    image_paths = ret_batch["image_paths"][0]
                    image_indices = ret_batch["image_indices"][0]

                    # ret["images"] = image_tensor
                    ret["images"] = images
                    ret["image_indices"] = image_indices

                if self.create_position_ids:
                    position_ids = torch.arange(self.max_len, dtype=torch.int64)
                    ret["position_ids"] = position_ids

                # print("dataset_base_mixtral self.create_attention_mask", self.create_attention_mask)
                if self.create_attention_mask:
                    ret["attention_mask"] = attention_mask
                # print("ret", ret)

                # print("dataset_base_mixtral self.create_attention_mask_2d", self.create_attention_mask_2d)
                if self.create_attention_mask_2d:
                    # from cognitron_modellink.utils import get_tune_attention_mask
                    # attention_mask_2d = get_tune_attention_mask(attention_mask.unsqueeze(0)).squeeze(1)

                    attention_mask_2d = torch.tril(
                        torch.ones((1, self.max_len, self.max_len), dtype=torch.bool)
                    )
                    attention_mask_2d = attention_mask_2d.masked_fill(
                        (attention_mask < 0.5).view(1, 1, self.max_len), value=0
                    )
                    attention_mask_2d = attention_mask_2d < 0.5

                    ret["attention_mask"] = attention_mask_2d
                # print("ret", ret)

                if self.create_loss_mask:
                    loss_mask = torch.where(labels == -100, 0, 1)
                    ret["loss_mask"] = loss_mask.to(torch.float32)

                return ret
            except Exception as error:
                with open(os.path.join(self.output_dir, "dataset_error.log"), "a") as f:
                    print(error, file=f)
                    print([self.raw_data[index]], file=f)
                if index == 0:
                    index += 1
                else:
                    index -= 1


def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    max_len: int,
    image_token_length: int,
    variable_length: bool = False,
    system_message: str = "You are a helpful assistant.",
    shift_token: bool = False,
    image_processor=None,
) -> Dict:
    # <|begin_of_text|><|start_header_id|>system<|end_header_id|>

    # Cutting Knowledge Date: December 2023
    # Today Date: 23 Jul 2024

    # You are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>

    # What is the capital of France?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

    from ..constants import IMG_START_TOKEN, IMG_END_TOKEN, IMG_CONTEXT_TOKEN

    image_tokens = f"{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * image_token_length}{IMG_END_TOKEN}"

    IMG_CONTEXT_ID = tokenizer(IMG_CONTEXT_TOKEN, add_special_tokens=False).input_ids
    IMG_START_ID = tokenizer(IMG_START_TOKEN, add_special_tokens=False).input_ids
    IMG_END_ID = tokenizer(IMG_END_TOKEN, add_special_tokens=False).input_ids

    assert len(IMG_CONTEXT_ID) == 1
    assert len(IMG_START_ID) == 1
    assert len(IMG_END_ID) == 1

    IMG_CONTEXT_ID = IMG_CONTEXT_ID[0]
    IMG_START_ID = IMG_START_ID[0]
    IMG_END_ID = IMG_END_ID[0]

    begin_of_text_id = tokenizer.get_vocab()["<|begin_of_text|>"]
    start_header_id = tokenizer.get_vocab()["<|start_header_id|>"]
    end_header_id = tokenizer.get_vocab()["<|end_header_id|>"]
    eot_id = tokenizer.get_vocab()["<|eot_id|>"]
    nl_tokens = tokenizer("\n", add_special_tokens=False).input_ids
    _system = tokenizer("system", add_special_tokens=False).input_ids
    _user = tokenizer("user", add_special_tokens=False).input_ids
    _assistant = tokenizer("assistant", add_special_tokens=False).input_ids

    sources = [x["conversations"] for x in sources]

    # Apply prompt templates
    input_ids, targets = [], []
    attention_mask = []
    images = []
    image_paths = []
    image_indices = []
    for i, source in enumerate(sources):
        if len(source) > 0 and source[0]["from"] != "user":
            source = source[1:]

        input_id, target = [], []
        image = []
        image_path = []
        image_count = 0
        image_indice = []

        if source[0]["from"] == "system":
            custom_system = True
        else:
            custom_system = False

        if not custom_system and system_message is not None and len(system_message) > 0:
            system = (
                [begin_of_text_id]
                + [start_header_id]
                + _system
                + [end_header_id]
                + nl_tokens
                + tokenizer(system_message, add_special_tokens=False).input_ids
                + [eot_id]
            )
            input_id += system
            target += [IGNORE_TOKEN_ID] * len(system)
            assert len(input_id) == len(target)

        for j, sentence in enumerate(source):
            role = sentence["from"]
            value = sentence["value"]

            # regex to extract required strings
            # reg_str = IMG_START_TOKEN + "(.*?)" + IMG_END_TOKEN
            # res = re.findall(reg_str, sentence["value"])

            # for x in res:
            #     if not os.path.isfile(x):
            #         raise Exception(f"No such file or directory: '{x}'")

            # image += [Image.open(x) for x in res]
            # image += [cv2.imread(x) for x in res]
            # image_path += res

            _image = []
            _image_path = []

            bos_pos = [m.start() for m in re.finditer(IMG_START_TOKEN, value)]
            eos_pos = [m.start() for m in re.finditer(IMG_END_TOKEN, value)]
            # print(bos_pos, eos_pos)
            assert len(bos_pos) == len(eos_pos)
            new_value = ""
            st = 0
            for a, b in zip(bos_pos, eos_pos):
                # print(value[a+len(IMG_START_TOKEN):b])
                img_path = value[a + len(IMG_START_TOKEN) : b]
                # print("value", value)
                new_value += value[st:a] + image_tokens
                st = b + len(IMG_END_TOKEN)
                # print("value", value)

                # _image.append(Image.open(img_path))
                # _image.append(cv2.imread(img_path))
                _image.append(image_processor.process([img_path]))
                _image_path.append(img_path)

            new_value += value[st:]
            value = new_value

            if role == "user":
                _input_id = (
                    [start_header_id]
                    + _user
                    + [end_header_id]
                    + nl_tokens
                    + tokenizer(value, add_special_tokens=False).input_ids
                    + [eot_id]
                )
                _target = [IGNORE_TOKEN_ID] * len(_input_id)
            elif role == "assistant":
                _input_id = (
                    [start_header_id]
                    + _assistant
                    + [end_header_id]
                    + nl_tokens
                    + tokenizer(value, add_special_tokens=False).input_ids
                    + [eot_id]
                )
                _target = (
                    [IGNORE_TOKEN_ID]
                    + [IGNORE_TOKEN_ID] * len(_assistant)
                    + [IGNORE_TOKEN_ID]
                    + [IGNORE_TOKEN_ID] * len(nl_tokens)
                    + tokenizer(value, add_special_tokens=False).input_ids
                    + [eot_id]
                )

                if "type" in sentence:
                    if sentence["type"] == "wrong answer":
                        _target = [IGNORE_TOKEN_ID] * (len(_input_id) - 1) + [EOS_ID]
            elif role == "system":
                _input_id = (
                    [begin_of_text_id]
                    + [start_header_id]
                    + _system
                    + [end_header_id]
                    + nl_tokens
                    + tokenizer(system_message, add_special_tokens=False).input_ids
                    + [eot_id]
                )
                _target = [IGNORE_TOKEN_ID] * len(_input_id)
            else:
                raise NotImplementedError

            if len(input_id) + len(_input_id) > max_len:
                break

            input_id += _input_id
            target += _target

            image += _image
            image_path += _image_path

        # bos_pos = [m.start() for m in re.finditer(IMG_START_ID, input_id)]
        # eos_pos = [m.start() for m in re.finditer(IMG_END_ID, input_id)]

        bos_pos = [i for i, x in enumerate(input_id) if x == IMG_START_ID]
        eos_pos = [i for i, x in enumerate(input_id) if x == IMG_END_ID]

        assert (
            len(bos_pos) == len(eos_pos) == len(image) == len(image_path)
        ), f"{bos_pos} {eos_pos} {len(image)} {image_path} {IMG_START_TOKEN} {IMG_START_ID} {IMG_END_TOKEN} {IMG_END_ID}"
        for a, b in zip(bos_pos, eos_pos):
            image_indice_b = torch.zeros(
                1, image_token_length, dtype=torch.int64
            )  # This will change in collate_fn
            image_indice_s = torch.arange(a + 1, b).unsqueeze(0).repeat(1, 1)
            image_indice_b_s = torch.stack(
                [image_indice_b, image_indice_s], dim=0
            )  # 2, num_image, image_length
            image_indice.append(image_indice_b_s)

        if shift_token:
            input_id = input_id[:-1]
            target = target[1:]

        attn_mask = [1] * len(input_id)

        assert len(input_id) == len(target), f"{len(input_ids)} {len(target)}"
        if not variable_length and max_len > len(input_id):
            input_id += [tokenizer.pad_token_id] * (max_len - len(input_id))
            target += [IGNORE_TOKEN_ID] * (max_len - len(target))
            attn_mask += [0] * (max_len - len(attn_mask))

        input_ids.append(input_id[:max_len])
        targets.append(target[:max_len])
        attention_mask.append(attn_mask[:max_len])

        # images.append(image)
        if len(image) > 0:
            image = torch.cat(image, dim=0)
        images.append(image)
        image_paths.append(image_path)

        if len(image_indice) > 0:
            image_indice = torch.cat(image_indice, dim=1)
            image_indices.append(image_indice)

    input_ids = torch.tensor(input_ids, dtype=torch.int64)
    targets = torch.tensor(targets, dtype=torch.int64)
    attention_mask = torch.tensor(attention_mask, dtype=torch.int)
    if len(image_indices) > 0:
        image_indices = torch.stack(image_indices, dim=0)
    # print("image_indices", image_indices.size())

    # print("sources", sources, flush=True)
    # print("input_ids", input_ids, flush=True)
    # print("targets", targets, flush=True)
    # print("images", [xx.shape for x in images for xx in x], flush=True)
    # print("image_paths", image_paths, flush=True)

    return dict(
        input_ids=input_ids,
        labels=targets,
        # attention_mask=input_ids.ne(tokenizer.pad_token_id),
        attention_mask=attention_mask,
        images=images,
        image_paths=image_paths,
        image_indices=image_indices,
    )
