import json
import logging
import os
import pdb
import re
import traceback
import uuid
import contextlib

import cv2
import numpy as np
import torch
import yaml
from PIL import Image

from torchvision import transforms
from torchvision.transforms import InterpolationMode

from .processor.image_processor import ImageProcessor
from .utils import draw_data

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BaseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        cfg_path,
        tokenizer,
        image_size=448,
        image_token_length=1024,
        model_max_length=32768,
        variable_length=False,
        output_dir="",
        add_task_symbol=True,
        training_args=None,
        shift_token=False,
        create_position_ids=True,
        create_attention_mask=True,
        create_attention_mask_2d=False,
        create_loss_mask=False,
        max_num_frame=8,
        max_fps=1,
        reset_position_ids=False,
        reset_attention_mask=False,
        min_patch_grid=1,
        max_patch_grid=6,
        process_type="anyres",
        normalize_type="imagenet",
        seed=42,
        cross_dataset_joint=False,
        dataset_joint=True,
    ):
        super(BaseDataset, self).__init__()

        self.cfg_path = cfg_path
        with open(self.cfg_path, "r", encoding="utf8") as cfg_file:
            cfg_data = cfg_file.read()

        self.cfg = yaml.load(cfg_data, Loader=yaml.CLoader)
        logger.info(f"cfg {self.cfg}")

        self.tokenizer = tokenizer
        self.max_len = model_max_length
        self.variable_length = variable_length
        self.output_dir = output_dir
        self.training_args = training_args
        self.shift_token = shift_token
        self.create_position_ids = create_position_ids
        self.create_attention_mask = create_attention_mask
        self.create_attention_mask_2d = create_attention_mask_2d
        self.create_loss_mask = create_loss_mask
        self.max_num_frame = max_num_frame
        self.max_fps = max_fps
        self.reset_position_ids = reset_position_ids
        self.reset_attention_mask = reset_attention_mask

        self.seed = seed
        self.cross_dataset_joint = cross_dataset_joint
        self.dataset_joint = dataset_joint

        self.image_size = image_size
        self.image_token_length = image_token_length

        self.do_dataset_format = self.cfg.get("do_dataset_format", False)
        self.do_dataset_cast = self.cfg.get("do_dataset_cast", False)
        self.xlsx_sample_num = self.cfg.get("xlsx_sample_num", 5)

        self.processor = {}
        self.processor["image"] = ImageProcessor(
            process_type,
            image_size=self.image_size,
            normalize_type=normalize_type,
            min_patch_grid=min_patch_grid,
            max_patch_grid=max_patch_grid,
        )

        # with main_process_first(local=True, desc="Loading data"):
        self.load_data()

        self.processed_samples = 0
        self.unjoint_samples = 0
        self.joint_samples = 0
        self.skip_samples = 0
        # self.skip_samples = 23000

    def load_data(self):
        from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset

        raw_data = None

        sampled_data = {}
        for data_name, data_info in self.cfg["dataset"].items():
            data_ratio = data_info.get("ratio", 1)
            data_num = data_info.get("num", 999999999)

            if data_ratio == 0:
                continue

            if data_num == 0:
                continue

            for data_path in data_info["data_paths"]:

                if not os.path.isfile(data_path) and not os.path.isdir(data_path):
                    logger.warning(f"Data file no found {data_path}")
                    continue

                this_data = load_json(data_path, self.output_dir)
                # this_data = load_data_one(data_path, self.outout_dir)
                if this_data is None:
                    logger.warning(f"Failed to load {data_path}")
                    continue
                # print(f"this_data {this_data}")

                column_names = list(this_data.features)
                if "id" in column_names:
                    this_data = this_data.remove_columns("id")

                sources = [data_path] * len(this_data)
                # sources = [data_name] * len(this_data)
                this_data = this_data.add_column("source", sources)

                if "images" not in column_names:
                    # images = [[]] * len(this_data)
                    images = [None] * len(this_data)
                    this_data = this_data.add_column("images", images)

                if "videos" not in column_names:
                    # videos = [[]] * len(this_data)
                    videos = [None] * len(this_data)
                    this_data = this_data.add_column("videos", videos)

                if False:
                    column_names = list(this_data.features)
                    this_data = this_data.map(
                        format_function_general,
                        batched=True,
                        batch_size=2560,
                        num_proc=1,
                        # batch_size=1,
                        # num_proc=1,
                        remove_columns=column_names,
                        keep_in_memory=False,
                        desc="Running format on dataset",
                    )

                this_data = this_data.shuffle(seed=self.seed)
                # this_data = this_data.flatten_indices()
                this_data = this_data.shuffle(seed=self.seed)
                # this_data = this_data.flatten_indices()

                data_ratio = float(data_ratio)
                total_num = len(this_data)
                used_num = min(int(total_num * data_ratio), data_num)
                logger.info(f"total_num {total_num}")
                logger.info(f"data_ratio {data_ratio}")
                logger.info(f"data_num {data_num}")
                logger.info(f"used_num {used_num}")

                indices = [x % total_num for x in range(used_num)]

                this_data = this_data.select(indices)

                if raw_data is None:
                    raw_data = this_data
                else:
                    if self.do_dataset_cast:
                        this_data = this_data.cast(raw_data.features)
                    raw_data = concatenate_datasets([raw_data, this_data])

                sampled_data[data_path] = {}
                sampled_data[data_path]["data"] = this_data.select(
                    range(min(self.xlsx_sample_num, used_num))
                )
                sampled_data[data_path]["total_num"] = total_num
                sampled_data[data_path]["used_num"] = used_num

                logger.info(f"this_data {this_data}")
                logger.info(f"raw_data {raw_data}")
                # logger.info(f"raw_data {raw_data[0]}")
                # logger.info(f"raw_data {raw_data[-1]}")
                logger.info(f"Successful load {data_path}")

        raw_data = raw_data.shuffle(seed=self.seed)
        # raw_data = raw_data.flatten_indices()
        raw_data = raw_data.shuffle(seed=self.seed)
        # raw_data = raw_data.flatten_indices()

        self.raw_data = raw_data

        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            output_xlsx = os.path.basename(self.cfg_path).replace("yaml", "xlsx")
            output_xlsx = os.path.join(self.output_dir, output_xlsx)
            logger.info(f"output_xlsx {output_xlsx}")
            draw_data(
                sampled_data,
                output_xlsx,
                tokenizer=self.tokenizer,
                image_processor=self.processor["image"],
            )

        logger.info(f"raw_data {raw_data}")

        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            logger.info(f"raw_data {raw_data[:10]}")
            logger.info(f"raw_data {raw_data[-10:]}")

    def __len__(self):
        return len(self.raw_data)


def format_function_general(examples):
    messages = [x for x in examples["messages"]]

    if "images" in examples:
        images = [x for x in examples["images"]]
    else:
        images = [None for _ in messages]

    if "videos" in examples:
        videos = [x for x in examples["videos"]]
    else:
        videos = [None for _ in messages]

    if "audios" in examples:
        audios = [x for x in examples["audios"]]
    else:
        audios = [None for _ in messages]

    return {
        "messages": messages,
        "images": images,
        "videos": videos,
        "audios": audios,
    }


def load_json_A(data_file):
    from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset

    with open(data_file, "r") as f:
        raw_data = json.load(f)
    this_data = Dataset.from_list(raw_data)
    return this_data


def load_json_B(data_file):
    from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset

    this_data = load_dataset("json", data_files=data_file, keep_in_memory=False)
    return this_data["train"]


def load_json_C(data_file):
    from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset

    raw_data = []
    with open(data_file, "r") as f:
        for line in f.readlines():
            d = json.loads(line)
            # raw_data.append({"conversations": d["conversations"], "id": d["id"]})
            if "conversations" in d:
                raw_data.append({"conversations": d["conversations"]})
            if "messages" in d:
                raw_data.append({"messages": d["messages"]})
    this_data = Dataset.from_list(raw_data)
    return this_data


def load_json(data_file, output_dir):
    for func in [load_json_B, load_json_A, load_json_C]:
        try:
            this_data = func(data_file)
            return this_data
        except Exception as error:
            with open(os.path.join(output_dir, "data_error.log"), "a") as f:
                print("-" * 100, file=f)
                # print(error, file=f)
                print(traceback.format_exc(), file=f)
            continue
    return None


def load_data_one(data_file, output_dir):
    if data_file.endswith("json") or data_file.endswith("jsonl"):
        return load_json(data_file, output_dir)
    from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset

    this_data = load_dataset(data_file, keep_in_memory=False)
    return this_data["train"]

@contextlib.contextmanager
def main_process_first(local=True, desc="work"):

    if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
        if local:
            rank = int(os.environ['LOCAL_RANK'])
        else:
            rank = torch.distributed.get_rank()
        is_main_process = rank == 0
        logger.info(f"rank {rank} is_main_process {is_main_process}")

        try:
            if not is_main_process:
                torch.distributed.barrier()
            yield
        finally:
            if is_main_process:
                torch.distributed.barrier()
    else:
        yield

