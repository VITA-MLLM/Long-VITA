import json
import logging
import math
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch
import transformers
from torch.utils.data import default_collate
from transformers.trainer_pt_utils import LabelSmoother

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


@dataclass
class DataCollatorForSupervisedDataset(object):
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [instance["input_ids"] for instance in instances]
        labels = [instance["labels"] for instance in instances]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_TOKEN_ID,
        )

        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if "images" in instances[0]:
            images = [instance["images"] for instance in instances]
            batch["images"] = torch.cat(images, dim=0)

        if "doclm_images" in instances[0]:
            images = [instance["doclm_images"] for instance in instances]
            batch["doclm_images"] = torch.cat(images, dim=0)

        if "image_paths" in instances[0]:
            image_paths = [instance["image_paths"] for instance in instances]
            batch["image_paths"] = image_paths

        if "pixel_values" in instances[0]:
            pixel_values = torch.cat([instance["pixel_values"] for instance in instances])
            batch["pixel_values"] = pixel_values

        if "image_flags" in instances[0]:
            image_flags = torch.cat([instance["image_flags"] for instance in instances])
            batch["image_flags"] = image_flags

        return batch


def collate_fn_deepspeed(batch):

    keys = list(set().union(*[set(x.keys()) for x in batch]))
    new_batch = [{} for _ in range(len(batch))]
    if "actual_seq_len" in batch[0]:
        actual_seq_len = [x["actual_seq_len"] for x in batch]
    else:
        actual_seq_len = None

    for k in keys:
        if "images" in k or k == "image_indices":
            for x, y in zip(new_batch, batch):
                if k in y:
                    x[k] = y.pop(k)
                    # print("x[image_indices]", x["image_indices"].size())

    old_batch = default_collate(batch)

    for k in keys:
        if "images" in k or k == "image_indices":
            cat_dim = 0 if k != "image_indices" else 1
            if k == "image_indices":
                cnt = 0
                for sample in new_batch:
                    if k in sample:
                        sample[k][0] = cnt
                    cnt += 1
            old_batch[k] = torch.cat([x[k] for x in new_batch if k in x], dim=cat_dim)
    # print("old_batch[image_indices]", old_batch["image_indices"].size())

    if actual_seq_len is not None:
        seq_len = actual_seq_len[0][-1]
        actual_seq_len = [elem + i * seq_len for i, elem in enumerate(actual_seq_len)]
        old_batch["actual_seq_len"] = torch.cat(actual_seq_len)

    return old_batch
