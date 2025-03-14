# coding=utf-8
import os
import sys
import time
import logging
from typing import Union
import natsort
import json
import tqdm
import argparse

import requests
import numpy as np

import matplotlib.pyplot as plt
import os
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
from pathlib import Path
import random
import json

from datasets import load_dataset
import natsort


def inference_sigle_image():


    print("#" * 100)
    image_path_list = ['/data/data/LMUData/images/MMBench_V11/313.jpg']
    video_path_list = []
    prompt =  '<image>\nHint: The diagram below shows a solution with one solute. Each solute particle is represented by a yellow ball. The solution fills a closed container that is divided in half by a membrane. The membrane, represented by a dotted line, is permeable to the solute particles.\nThe diagram shows how the solution can change over time during the process of diffusion.\nQuestion: Complete the text to describe the diagram.\nSolute particles moved in both directions across the permeable membrane. But more solute particles moved across the membrane (). When there was an equal concentration on both sides, the particles reached equilibrium.\nOptions:\nA. to the right than to the left\nB. to the left than to the right\nPlease select the correct answer from the options above. \n'

    url = os.environ.get('LongVITA_URL', default='http://127.0.0.1:5001/api')

    headers = {
        'Content-Type': 'application/json',
        # 'Request-Id': 'remote-test',
        # 'Authorization': f'Bearer {self.key}'
    }
    payload = {
        'prompts': [prompt],
        'image_path_list': image_path_list if len(image_path_list) > 0 else None,
        'video_path_list': video_path_list if len(video_path_list) > 0 else None,
        # 'temperature': 0,
        'tokens_to_generate': 16,
        # 'max_num_frame': 1000,
    }
    print(f"payload {payload}")
    print(f"image_path_list {len(image_path_list)}")
    print(f"video_path_list {len(video_path_list)}")
    response = requests.put(url,
                            headers=headers,
                            data=json.dumps(payload),
                            verify=False)

    if response.status_code != 200:
        print(
            f"Error {response.status_code}: {response.json()['message']}"
        )
    else:
        answer = response.json()['text'][0]
        print(f"answer {answer}")


def inference_multi_image():


    image_path_list = []
    video_path_list = []

    image_dir = "/data/data/Comic/images/006-realm-of-kings-inhumans-02-of-5-2010/"
    for root, dirs, files in os.walk(image_dir):
        for filename in files:
            if (filename.endswith("png") or filename.endswith("jpeg")
                    or filename.endswith("jpg")):
                filepath = os.path.join(root, filename)
                image_path_list.append(filepath)

    image_path_list = natsort.natsorted(image_path_list)

    # prompt = "<image>" * len(image_path_list) + "\nProvide a full summary of the comic book."
    # prompt = "<image>" * len(image_path_list) + "\nHow many Avengers in the story? Provide their names."
    # prompt = "<image>" * len(image_path_list) + "\nThe inhumans donot join the noble battle, is that right?"
    prompt = "<image>" * len(image_path_list) + "\nWho is join the noble battle."

    print("#" * 100)

    url = os.environ.get('LongVITA_URL', default='http://127.0.0.1:5001/api')

    headers = {
        'Content-Type': 'application/json',
        # 'Request-Id': 'remote-test',
        # 'Authorization': f'Bearer {self.key}'
    }
    payload = {
        'prompts': [prompt],
        'image_path_list': image_path_list if len(image_path_list) > 0 else None,
        'video_path_list': video_path_list if len(video_path_list) > 0 else None,
        'tokens_to_generate': 8,
        'max_num_frame': 4096,
    }
    print(f"payload {payload}")
    print(f"image_path_list {len(image_path_list)}")
    print(f"video_path_list {len(video_path_list)}")
    response = requests.put(url,
                            headers=headers,
                            data=json.dumps(payload),
                            verify=False)

    if response.status_code != 200:
        print(
            f"Error {response.status_code}: {response.json()['message']}"
        )
    else:
        answer = response.json()['text'][0]
        print(f"answer {answer}")


def inference_video():


    image_path_list = []
    video_path_list = []

    video_path_list = ["/data/data/Kinetics/kinetics-dataset/k400_targz/train/00-bv7CYBwE_000031_000041.mp4"]
    prompt = "<video>" * len(video_path_list) + "\nProvide a full summary of the video."

    print("#" * 100)

    url = os.environ.get('LongVITA_URL', default='http://127.0.0.1:5001/api')

    headers = {
        'Content-Type': 'application/json',
        # 'Request-Id': 'remote-test',
        # 'Authorization': f'Bearer {self.key}'
    }
    payload = {
        'prompts': [prompt],
        'image_path_list': image_path_list if len(image_path_list) > 0 else None,
        'video_path_list': video_path_list if len(video_path_list) > 0 else None,
        'tokens_to_generate': 1024,
        'max_num_frame': 4096,
    }
    print(f"payload {payload}")
    print(f"image_path_list {len(image_path_list)}")
    print(f"video_path_list {len(video_path_list)}")
    response = requests.put(url,
                            headers=headers,
                            data=json.dumps(payload),
                            verify=False)

    if response.status_code != 200:
        print(
            f"Error {response.status_code}: {response.json()['message']}"
        )
    else:
        answer = response.json()['text'][0]
        print(f"answer {answer}")


def main():

    inference_sigle_image()
    inference_multi_image()
    inference_video()


if __name__ == "__main__":

    main()
