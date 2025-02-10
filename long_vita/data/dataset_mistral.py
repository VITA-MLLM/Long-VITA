import copy
import json
import logging
import os
import pdb
import re
import time
import traceback
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np
import torch
import transformers
from transformers.trainer_pt_utils import LabelSmoother

from .dataset_base import BaseDataset

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


class MistralDataset(BaseDataset):
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
        create_position_ids=True,
        create_attention_mask=True,
        create_attention_mask_2d=False,
        create_loss_mask=False,
        max_num_image=8,
        max_num_frame=8,
        max_fps=1,
        reset_position_ids=False,
        reset_attention_mask=False,
        min_patch_grid=1,
        max_patch_grid=6,
        process_type="anyres",
        normalize_type="imagenet",
        seed=42,
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
            shift_token=shift_token,
            add_task_symbol=add_task_symbol,
            create_position_ids=create_position_ids,
            create_attention_mask=create_attention_mask,
            create_attention_mask_2d=create_attention_mask_2d,
            create_loss_mask=create_loss_mask,
            max_num_image=max_num_image,
            max_num_frame=max_num_frame,
            max_fps=max_fps,
            reset_position_ids=reset_position_ids,
            reset_attention_mask=reset_attention_mask,
            min_patch_grid=min_patch_grid,
            max_patch_grid=max_patch_grid,
            process_type=process_type,
            normalize_type=normalize_type,
            seed=seed,
        )

        self.system_message = "You are a helpful AI assistant."
        self.system_message = None

        self.ret = defaultdict(dict)
        self.is_cat = True

    def maybe_init_ret(self, source, force=False):
        if source not in self.ret or force:
            self.ret[source] = {}

            self.ret[source]["tokens"] = []
            self.ret[source]["labels"] = []
            self.ret[source]["actual_seq_len"] = []

            if self.create_position_ids:
                self.ret[source]["position_ids"] = []

            if self.create_attention_mask:
                self.ret[source]["attention_mask"] = []

            if self.create_attention_mask_2d:
                self.ret[source]["attention_mask_2d"] = torch.tril(
                    torch.ones((1, self.max_len, self.max_len), dtype=torch.bool)
                )
        return len(self.ret[source]["tokens"]) == 0

    def __getitem__(self, index):
        while True:
            try:
                # if True:

                self.processed_samples += 1
                if self.processed_samples < self.skip_samples:
                    if self.processed_samples % 1e3 == 0:
                        print(
                            f"processed_samples {self.processed_samples} skip_samples {self.skip_samples}"
                        )
                    return {}

                source = self.raw_data[index]["source"]
                is_empty = self.maybe_init_ret(source)
                is_begin = is_empty or self.reset_position_ids or self.reset_attention_mask

                ret = preprocess(
                    self.raw_data[index],
                    self.tokenizer,
                    self.image_token_length,
                    system_message=self.system_message,
                    image_processor=self.processor["image"],
                    is_begin=is_begin,
                    max_num_frame=self.max_num_frame,
                    max_fps=self.max_fps,
                )

                if ret is None:
                    return {}

                if len(ret["input_ids"]) > self.max_len:
                    return {}

                if len(ret["images"]) > self.max_num_frame:
                    return {}

                self.unjoint_samples += 1

                if self.max_len > 2**17:
                    log_interval = 1e3
                elif self.max_len > 2**15:
                    log_interval = 1e4
                else:
                    log_interval = 1e4
                if self.unjoint_samples % log_interval == 0:
                    print(
                        f"processed_samples {self.processed_samples} unjoint_samples {self.unjoint_samples}"
                    )

                all_length = len(self.ret[source]["tokens"])
                cur_length = len(ret["input_ids"])

                # print("self.ret", self.ret.keys())
                # print("source", source)
                if all_length + cur_length > self.max_len or (
                    "images" in self.ret[source]
                    and len(self.ret[source]["images"]) + len(ret["images"]) > self.max_num_image
                ):
                    # if "tokens" in self.ret[source] and len(self.ret[source]["tokens"]) > 0:
                    to_ret = copy.deepcopy(self.ret[source])
                    self.maybe_init_ret(source, force=True)
                else:
                    to_ret = {}

                all_length = len(self.ret[source]["tokens"])
                cur_length = len(ret["input_ids"])

                if "images" in ret and len(ret["images"]) > 0:
                    if "images" in self.ret[source]:
                        self.ret[source]["images"] = torch.cat(
                            [self.ret[source]["images"], ret["images"]], dim=0
                        )
                        ret["image_indices"][1, :, :] += all_length
                        self.ret[source]["image_indices"] = torch.cat(
                            [self.ret[source]["image_indices"], ret["image_indices"]], dim=1
                        )
                    else:
                        self.ret[source]["images"] = ret["images"]
                        self.ret[source]["image_indices"] = ret["image_indices"]

                if self.create_attention_mask:
                    self.ret[source]["attention_mask"] += ret["attention_mask"]

                if self.create_attention_mask_2d:
                    self.ret[source]["attention_mask_2d"][:, all_length:, :all_length] = 0

                if self.create_position_ids:
                    self.ret[source]["position_ids"] += list(range(cur_length))

                self.ret[source]["tokens"] += ret["input_ids"]
                self.ret[source]["labels"] += ret["labels"]
                self.ret[source]["actual_seq_len"] += [all_length + cur_length]

                if "tokens" in to_ret:
                    if self.create_position_ids:
                        if self.reset_position_ids:
                            pass
                        else:
                            to_ret["position_ids"] = list(range(len(to_ret["tokens"])))

                    if self.create_attention_mask_2d:
                        if sefl.reset_attention_mask:
                            pass
                        else:
                            to_ret["attention_mask_2d"] = torch.tril(
                                torch.ones((1, self.max_len, self.max_len), dtype=torch.bool)
                            )

                    if self.shift_token:
                        to_ret["tokens"] = to_ret["tokens"][:-1]
                        to_ret["labels"] = to_ret["labels"][1:]
                        to_ret["actual_seq_len"][-1] -= 1
                        if self.create_position_ids:
                            to_ret["position_ids"] = to_ret["position_ids"][:-1]
                        if self.create_attention_mask:
                            to_ret["attention_mask"] = to_ret["attention_mask"][:-1]

                        if self.create_attention_mask_2d:
                            to_ret["attention_mask_2d"][:, :, -1] = 0
                            to_ret["attention_mask_2d"][:, -1, :] = 0

                    assert len(to_ret["tokens"]) == len(
                        to_ret["labels"]
                    ), f"{len(to_ret['tokens'])} {len(to_ret['labels'])}"
                    if not self.variable_length and self.max_len > len(to_ret["tokens"]):
                        to_ret["tokens"] += [self.tokenizer.pad_token_id] * (
                            self.max_len - len(to_ret["tokens"])
                        )
                        to_ret["labels"] += [IGNORE_TOKEN_ID] * (
                            self.max_len - len(to_ret["labels"])
                        )
                        to_ret["actual_seq_len"][-1] = self.max_len
                        if self.create_position_ids:
                            to_ret["position_ids"] += to_ret["position_ids"][-1:] * (
                                self.max_len - len(to_ret["position_ids"])
                            )
                        if self.create_attention_mask:
                            to_ret["attention_mask"] += [0] * (
                                self.max_len - len(to_ret["attention_mask"])
                            )

                    to_ret["tokens"] = to_ret["tokens"][: self.max_len]
                    to_ret["labels"] = to_ret["labels"][: self.max_len]
                    to_ret["actual_seq_len"][-1] = self.max_len
                    if self.create_position_ids:
                        to_ret["position_ids"] = to_ret["position_ids"][: self.max_len]
                    if self.create_attention_mask:
                        to_ret["attention_mask"] = to_ret["attention_mask"][: self.max_len]

                    to_ret["tokens"] = torch.tensor(to_ret["tokens"], dtype=torch.int64)
                    to_ret["labels"] = torch.tensor(to_ret["labels"], dtype=torch.int64)
                    to_ret["actual_seq_len"] = torch.tensor(
                        to_ret["actual_seq_len"], dtype=torch.int64
                    )
                    if self.create_position_ids:
                        to_ret["position_ids"] = torch.tensor(
                            to_ret["position_ids"], dtype=torch.int64
                        )
                    if self.create_attention_mask:
                        to_ret["attention_mask"] = torch.tensor(
                            to_ret["attention_mask"], dtype=torch.int64
                        )

                    if self.create_attention_mask_2d:
                        attention_mask_2d = to_ret.pop("attention_mask_2d")
                        attention_mask_2d = attention_mask_2d.masked_fill(
                            (to_ret["attention_mask"] < 0.5).view(1, 1, self.max_len), value=0
                        )
                        attention_mask_2d = attention_mask_2d < 0.5

                        to_ret["attention_mask"] = attention_mask_2d

                    if self.create_loss_mask:
                        loss_mask = torch.where(to_ret["labels"] == IGNORE_TOKEN_ID, 0, 1)
                        to_ret["loss_mask"] = loss_mask.to(torch.float32)

                    if not self.reset_position_ids and not self.reset_attention_mask:
                        to_ret.pop("actual_seq_len")

                    # print("to_ret[tokens]", to_ret["tokens"])
                    # print("to_ret[labels]", to_ret["labels"])

                return to_ret
            except Exception as error:
                with open(os.path.join(self.output_dir, "dataset_error.log"), "a") as f:
                    print(error, file=f)
                    print([self.raw_data[index]], file=f)
                if index == 0:
                    index += 1
                else:
                    index -= 1


def preprocess(
    source,
    tokenizer: transformers.PreTrainedTokenizer,
    image_token_length: int,
    system_message: str = "You are a helpful assistant.",
    image_processor=None,
    is_begin: bool = True,
    max_num_frame: int = 8,
    max_fps: int = 1,
) -> Dict:

    # <s>[INST]Hello, how are you?[/INST]I'm doing great. How can I help you today?</s>[INST]I'd like to show off how chat templating works![/INST]

    from ..constants import (
        IMG_START_TOKEN,
        IMG_END_TOKEN,
        IMG_CONTEXT_TOKEN,
        VID_START_TOKEN,
        VID_END_TOKEN,
        VID_CONTEXT_TOKEN,
        PATCH_START_TOKEN,
        PATCH_END_TOKEN,
        PATCH_CONTEXT_TOKEN,
    )

    image_tag = "<image>"
    video_tag = "<video>"
    human_roles = ["user", "human"]
    gpt_roles = ["assistant", "gpt"]
    system_roles = [
        "system",
    ]

    IMG_CONTEXT_ID = tokenizer(IMG_CONTEXT_TOKEN, add_special_tokens=False).input_ids
    IMG_START_ID = tokenizer(IMG_START_TOKEN, add_special_tokens=False).input_ids
    IMG_END_ID = tokenizer(IMG_END_TOKEN, add_special_tokens=False).input_ids

    VID_CONTEXT_ID = tokenizer(VID_CONTEXT_TOKEN, add_special_tokens=False).input_ids
    VID_START_ID = tokenizer(VID_START_TOKEN, add_special_tokens=False).input_ids
    VID_END_ID = tokenizer(VID_END_TOKEN, add_special_tokens=False).input_ids

    PATCH_CONTEXT_ID = tokenizer(PATCH_CONTEXT_TOKEN, add_special_tokens=False).input_ids
    PATCH_START_ID = tokenizer(PATCH_START_TOKEN, add_special_tokens=False).input_ids
    PATCH_END_ID = tokenizer(PATCH_END_TOKEN, add_special_tokens=False).input_ids

    assert len(IMG_CONTEXT_ID) == 1
    assert len(IMG_START_ID) == 1
    assert len(IMG_END_ID) == 1

    assert len(VID_CONTEXT_ID) == 1
    assert len(VID_START_ID) == 1
    assert len(VID_END_ID) == 1

    assert len(PATCH_CONTEXT_ID) == 1
    assert len(PATCH_START_ID) == 1
    assert len(PATCH_END_ID) == 1

    IMG_CONTEXT_ID = IMG_CONTEXT_ID[0]
    IMG_START_ID = IMG_START_ID[0]
    IMG_END_ID = IMG_END_ID[0]

    VID_CONTEXT_ID = VID_CONTEXT_ID[0]
    VID_START_ID = VID_START_ID[0]
    VID_END_ID = VID_END_ID[0]

    PATCH_CONTEXT_ID = PATCH_CONTEXT_ID[0]
    PATCH_START_ID = PATCH_START_ID[0]
    PATCH_END_ID = PATCH_END_ID[0]

    BOS_ID = tokenizer.bos_token_id
    EOS_ID = tokenizer.eos_token_id

    B_INST = "[INST]"
    E_INST = "[/INST]"
    B_SYS = "<<SYS>>\n"
    E_SYS = "\n<</SYS>>\n\n"

    nl_tokens = tokenizer("\n", add_special_tokens=False).input_ids
    B_SYS_IDS = tokenizer(B_SYS, add_special_tokens=False).input_ids
    E_SYS_IDS = tokenizer(E_SYS, add_special_tokens=False).input_ids
    B_INST_IDS = tokenizer(B_INST, add_special_tokens=False).input_ids
    E_INST_IDS = tokenizer(E_INST, add_special_tokens=False).input_ids

    input_ids, targets = [], []
    images = []
    image_indices = []

    conversations = source["conversations"]
    if is_begin:
        input_ids += [BOS_ID]
        targets += [IGNORE_TOKEN_ID]

        if conversations[0]["from"] == "system":
            custom_system = True
        else:
            custom_system = False

        if not custom_system and system_message is not None and len(system_message) > 0:
            for jj in range(0, len(conversations)):
                if conversations[jj]["from"] in human_roles:
                    conversations[jj]["value"] = (
                        system_message + "\n\n" + conversations[jj]["value"]
                    )

    # print(f"input_ids {input_ids}")
    # print(f"targets {targets}")
    for j, sentence in enumerate(conversations):
        role = sentence["from"]
        value = sentence["value"]

        # ----------------------------------------------------------------
        # image
        value = value.replace(image_tag, IMG_CONTEXT_TOKEN)

        if IMG_START_TOKEN in value:
            source["images"] = []
            bos_pos = [m.start() for m in re.finditer(IMG_START_TOKEN, value)]
            eos_pos = [m.start() for m in re.finditer(IMG_END_TOKEN, value)]

            # print(bos_pos, eos_pos)
            assert len(bos_pos) == len(eos_pos)
            new_value = ""
            st = 0
            for a, b in zip(bos_pos, eos_pos):
                img_path = value[a + len(IMG_START_TOKEN) : b]
                new_value += value[st:a] + IMG_CONTEXT_TOKEN
                st = b + len(IMG_END_TOKEN)

                source["images"].append(img_path)

            new_value += value[st:]
            value = new_value

        # ----------------------------------------------------------------
        # video
        value = value.replace(video_tag, VID_CONTEXT_TOKEN)

        if VID_START_TOKEN in value:
            source["videos"] = []
            bos_pos = [m.start() for m in re.finditer(VID_START_TOKEN, value)]
            eos_pos = [m.start() for m in re.finditer(VID_END_TOKEN, value)]

            # print(bos_pos, eos_pos)
            assert len(bos_pos) == len(eos_pos)
            new_value = ""
            st = 0
            for a, b in zip(bos_pos, eos_pos):
                vid_path = value[a + len(VID_START_TOKEN) : b]
                new_value += value[st:a] + VID_CONTEXT_TOKEN
                st = b + len(VID_END_TOKEN)

                source["videos"].append(vid_path)

            new_value += value[st:]
            value = new_value

        # ----------------------------------------------------------------
        # text
        if role in human_roles:
            _input_id = (
                B_INST_IDS + tokenizer(value, add_special_tokens=False).input_ids + E_INST_IDS
            )
            _target = [IGNORE_TOKEN_ID] * len(_input_id)
        elif role in gpt_roles:
            _input_id = tokenizer(value, add_special_tokens=False).input_ids + [EOS_ID]
            _target = tokenizer(value, add_special_tokens=False).input_ids + [EOS_ID]

            if "type" in sentence:
                if sentence["type"] == "wrong answer":
                    _target = [IGNORE_TOKEN_ID] * (len(_input_id) - 1) + [EOS_ID]
        elif role in system_roles:
            for jj in range(j + 1, len(conversations)):
                if conversations[jj]["from"] in human_roles:
                    conversations[jj]["value"] = value + "\n\n" + conversations[jj]["value"]
            _input_id = []
            _target = []
        else:
            raise NotImplementedError

        input_ids += _input_id
        targets += _target

    # ----------------------------------------------------------------
    # image
    if (
        "images" in source
        and isinstance(source["images"], list)
        and None not in source["images"]
        and len(source["images"])
    ):
        img_positions = [i for i, x in enumerate(input_ids) if x == IMG_CONTEXT_ID]
        assert len(img_positions) == len(source["images"]), source

        new_input_ids = []
        new_targets = []
        st = 0
        for img_idx, img_pos in enumerate(img_positions):
            image_patches, (best_width, best_height) = image_processor.process_images_with_subpatch(
                source["images"][img_idx]
            )
            images.append(image_patches)

            new_input_ids += input_ids[st:img_pos]
            new_targets += targets[st:img_pos]

            new_input_ids += [IMG_START_ID]
            new_targets += [IGNORE_TOKEN_ID]

            image_indice_b = torch.zeros(
                1, image_token_length, dtype=torch.int64
            )  # This will change in collate_fn
            image_indice_s = (
                torch.arange(len(new_input_ids), len(new_input_ids) + image_token_length)
                .unsqueeze(0)
                .repeat(1, 1)
            )
            image_indice_b_s = torch.stack(
                [image_indice_b, image_indice_s], dim=0
            )  # 2, num_image, image_length
            image_indices.append(image_indice_b_s)

            new_input_ids += [IMG_CONTEXT_ID] * image_token_length
            new_targets += [IGNORE_TOKEN_ID] * image_token_length

            new_input_ids += [IMG_END_ID]
            new_targets += [IGNORE_TOKEN_ID]

            if len(image_patches) > 1:
                for i in range(0, best_height, image_processor.patch_size):
                    new_input_ids += nl_tokens
                    new_targets += [IGNORE_TOKEN_ID] * len(nl_tokens)

                    for j in range(0, best_width, image_processor.patch_size):
                        new_input_ids += [PATCH_START_ID]
                        new_targets += [IGNORE_TOKEN_ID]

                        image_indice_b = torch.zeros(
                            1, image_token_length, dtype=torch.int64
                        )  # This will change in collate_fn
                        image_indice_s = (
                            torch.arange(
                                len(new_input_ids), len(new_input_ids) + image_token_length
                            )
                            .unsqueeze(0)
                            .repeat(1, 1)
                        )
                        image_indice_b_s = torch.stack(
                            [image_indice_b, image_indice_s], dim=0
                        )  # 2, num_image, image_length
                        image_indices.append(image_indice_b_s)

                        new_input_ids += [PATCH_CONTEXT_ID] * image_token_length
                        new_targets += [IGNORE_TOKEN_ID] * image_token_length

                        new_input_ids += [PATCH_END_ID]
                        new_targets += [IGNORE_TOKEN_ID]

            st = img_pos + 1

        new_input_ids += input_ids[st:]
        new_targets += targets[st:]

        input_ids = new_input_ids
        targets = new_targets

    # ----------------------------------------------------------------
    # video
    if (
        "videos" in source
        and isinstance(source["videos"], list)
        and None not in source["videos"]
        and len(source["videos"])
    ):
        vid_positions = [i for i, x in enumerate(input_ids) if x == VID_CONTEXT_ID]
        assert len(vid_positions) == len(source["videos"]), source

        new_input_ids = []
        new_targets = []
        st = 0
        for vid_idx, vid_pos in enumerate(vid_positions):
            vid_path = source["videos"][vid_idx]

            new_input_ids += input_ids[st:vid_pos]
            new_targets += targets[st:vid_pos]

            video_frames, _ = image_processor.process_video(vid_path, max_num_frame, max_fps)
            images.append(video_frames)

            for _ in video_frames:
                new_input_ids += [VID_START_ID]
                new_targets += [IGNORE_TOKEN_ID]

                image_indice_b = torch.zeros(
                    1, image_token_length, dtype=torch.int64
                )  # This will change in collate_fn
                image_indice_s = (
                    torch.arange(len(new_input_ids), len(new_input_ids) + image_token_length)
                    .unsqueeze(0)
                    .repeat(1, 1)
                )
                image_indice_b_s = torch.stack(
                    [image_indice_b, image_indice_s], dim=0
                )  # 2, num_image, image_length
                image_indices.append(image_indice_b_s)

                new_input_ids += [VID_CONTEXT_ID] * image_token_length
                new_targets += [IGNORE_TOKEN_ID] * image_token_length

                new_input_ids += [VID_END_ID]
                new_targets += [IGNORE_TOKEN_ID]

            st = vid_pos + 1

        new_input_ids += input_ids[st:]
        new_targets += targets[st:]

        input_ids = new_input_ids
        targets = new_targets

    if len(images) > 0:
        images = torch.cat(images, dim=0)

    if len(image_indices) > 0:
        image_indices = torch.cat(image_indices, dim=1)

    attention_mask = [1] * len(input_ids)

    # print("source", source, flush=True)
    # print("input_ids", input_ids[:100], flush=True)
    # print("targets", targets[:100], flush=True)
    # print("images", [xx.shape for x in images for xx in x], flush=True)

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=attention_mask,
        images=images,
        image_indices=image_indices,
    )
