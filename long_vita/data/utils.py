import argparse
import copy
import hashlib
import io
import itertools
import json
import logging
import os
import re
from typing import Dict, Sequence, Tuple

import numpy as np
import xlsxwriter
from datasets import concatenate_datasets, load_dataset
from PIL import Image
from termcolor import colored

from tabulate import tabulate


def buffer_image(image: Image, format: str = "JPEG"):
    # Store image in buffer, so we don't have to write it to disk.
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return buffer, image


def resize(img_or_path: str, size: Tuple[int, int], format="JPEG"):
    if isinstance(img_or_path, str):
        image = Image.open(img_or_path)
    else:
        image = img_or_path
    # image = image.resize(size)
    image.thumbnail(size, Image.LANCZOS)
    image = image.convert("RGB")

    return buffer_image(image, format)


def calculate_scale(file_path, bound_size):
    # check the image size without loading it into memory
    im = Image.open(file_path)
    original_width, original_height = im.size

    # calculate the resize factor, keeping original aspect and staying within boundary
    bound_width, bound_height = bound_size
    ratios = (float(bound_width) / original_width, float(bound_height) / original_height)
    return min(ratios)


def draw_data(all_datasets, output_path, tokenizer=None, image_processor=None):
    if hasattr(tokenizer, "image_start_tag"):
        image_start_tag = tokenizer.image_start_tag
        image_end_tag = tokenizer.image_end_tag
    else:
        from ..constants import IMG_START_TOKEN, IMG_END_TOKEN

        image_start_tag = IMG_START_TOKEN
        image_end_tag = IMG_END_TOKEN

    workbook = xlsxwriter.Workbook(output_path)
    cell_format = workbook.add_format({"text_wrap": True, "font_size": 12})

    worksheet = workbook.add_worksheet("ALL")
    worksheet.set_column(1, 2, 20, cell_format)
    worksheet.set_column(3, 3, 240, cell_format)
    worksheet.write(0, 1, "total_num")
    worksheet.write(0, 2, "used_num")
    worksheet.write(0, 3, "name")
    row = 1
    for this_name, this_dataset in all_datasets.items():
        total_num = this_dataset["total_num"]
        used_num = this_dataset["used_num"]
        worksheet.write(row, 1, total_num)
        worksheet.write(row, 2, used_num)
        worksheet.write(row, 3, this_name)
        row += 1
    worksheet.write(
        row, 2, sum([this_dataset["used_num"] for this_dataset in all_datasets.values()])
    )

    all_base_name = []
    for this_name, this_dataset in all_datasets.items():
        base_name = os.path.basename(this_name)
        base_name = os.path.splitext(base_name)[0]

        base_name = base_name[:24]
        all_base_name.append(base_name)

        if all_base_name.count(base_name) > 1:
            base_name = base_name + "_" + str(all_base_name.count(base_name))

        worksheet = workbook.add_worksheet(base_name)
        worksheet.set_column(1, 2, 120, cell_format)
        worksheet.write(0, 1, "user")
        worksheet.write(0, 2, "assistant")
        row = 1

        data = this_dataset["data"]
        for this_data in data:
            # print(this_data)
            if isinstance(this_data, Dict):
                # print(this_data.keys())
                messages = this_data["messages"]
                if "images" in this_data:
                    images = this_data["images"]
                if "videos" in this_data:
                    videos = this_data["videos"]
            else:
                messages = this_data

            image_count = 0
            video_count = 0
            for message in messages:
                content = message["content"]
                role = message["role"]
                if role == "user" or role == "human":
                    col = 1
                else:
                    col = 2
                worksheet.write(row, col, content)
                row += 1

                bos_pos = [m.start() for m in re.finditer(image_start_tag, content)]
                eos_pos = [m.start() for m in re.finditer(image_end_tag, content)]
                # print(bos_pos, eos_pos)
                for a, b in zip(bos_pos, eos_pos):
                    # print(content[a+len(image_start_tag:b])
                    img_path = content[a + len(image_start_tag) : b]
                    # print(img_path)
                    worksheet.set_row(row, 200)

                    try:
                        image_buffer, image = resize(img_path, (512, 512), format="JPEG")
                    except:
                        continue

                    scale = min(256 / image.width, 256 / image.height)
                    data = {"x_scale": scale, "y_scale": scale, "object_position": 1}

                    worksheet.insert_image(row, col, img_path, {"image_data": image_buffer, **data})

                    row += 1

                for _ in range(content.count("<image>") + content.count("<|image|>")):
                    if images is None:
                        continue
                    img_path = images[image_count]
                    # print(img_path)
                    worksheet.set_row(row, 200)

                    try:
                        image_buffer, image = resize(img_path, (512, 512), format="JPEG")
                    except:
                        continue

                    scale = min(256 / image.width, 256 / image.height)
                    data = {"x_scale": scale, "y_scale": scale, "object_position": 1}

                    worksheet.insert_image(row, col, img_path, {"image_data": image_buffer, **data})

                    row += 1
                    image_count += 1

                for _ in range(content.count("<video>") + content.count("<|video|>")):
                    if videos is None:
                        continue
                    vid_path = videos[video_count]
                    try:
                        _, video_frames = image_processor.process_video(vid_path, max_num_frame=4)
                        # print(vid_path)
                    except:
                        continue

                    for video_frame in video_frames:
                        worksheet.set_row(row, 200)
                        try:
                            image_buffer, image = resize(video_frame, (512, 512), format="JPEG")
                        except:
                            continue

                        scale = min(256 / image.width, 256 / image.height)
                        data = {"x_scale": scale, "y_scale": scale, "object_position": 1}

                        if isinstance(video_frame, str):
                            video_path = video_frame
                        else:
                            video_file = hashlib.md5(video_frame.tobytes()).hexdigest() + ".png"
                            video_path = os.path.join("/tmp/", video_file)
                            video_frame.save(video_path)

                        worksheet.insert_image(
                            row, col, video_path, {"image_data": image_buffer, **data}
                        )

                        row += 1
                    video_count += 1

            row += 8

    workbook.close()
