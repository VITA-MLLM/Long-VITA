# Data Preparation

## Long-VITA Training Data

The data configures are defined by the YAML files in [configs folder](https://github.com/VITA-MLLM/Long-VITA/tree/main/configs).

An example YAML file of the training data:
```
dataset:

  ...
  LLaVA-ReCap:
    ratio: 1
    data_paths:
      - datasets/jsonl/lmms-lab/LLaVA-ReCap-558K/data.jsonl
      - datasets/jsonl/lmms-lab/LLaVA-ReCap-118K/data.jsonl
      - datasets/jsonl/lmms-lab/LLaVA-ReCap-CC3M/data.jsonl
   ...

```

Our processed JSONL files can be downloaded from [Long-VITA-Training-Data](https://huggingface.co/datasets/VITA-MLLM/Long-VITA-Training-Data).

The images and videos can be downloaded by following the instructions from their original websites.

We list the data used in Long-VITA:
- LLaVA
  - https://huggingface.co/datasets/weizhiwang/llava_v15_instruction_images

- LLaVA-ReCap
  - https://huggingface.co/datasets/lmms-lab/LLaVA-ReCap-558K
  - https://huggingface.co/datasets/lmms-lab/LLaVA-ReCap-118K
  - https://huggingface.co/datasets/lmms-lab/LLaVA-ReCap-CC3M

- ALLaVA
  - https://huggingface.co/datasets/FreedomIntelligence/ALLaVA-4V

- LVIS
  - https://huggingface.co/datasets/X2FD/LVIS-Instruct4V

- ShareGPT4V
  - https://huggingface.co/datasets/Lin-Chen/ShareGPT4V

- the cauldron
  - https://huggingface.co/datasets/HuggingFaceM4/the_cauldron

- Docmatix
  - https://huggingface.co/datasets/HuggingFaceM4/Docmatix

- LLaVA-OneVision-Mid-Data
  - https://huggingface.co/datasets/lmms-lab/LLaVA-OneVision-Mid-Data

- LLaVA-OneVision-Data
  - https://huggingface.co/datasets/lmms-lab/LLaVA-OneVision-Data

- M4-Instruct-Data
  - https://huggingface.co/datasets/lmms-lab/M4-Instruct-Data

- OpenHermes
  - https://huggingface.co/datasets/teknium/OpenHermes-2.5

- lima
  - https://huggingface.co/datasets/GAIR/lima

- databricks-dolly-15k
  - https://huggingface.co/datasets/databricks/databricks-dolly-15k

- MetaMathQA
  - https://huggingface.co/datasets/meta-math/MetaMathQA

- MathInstruct
  - https://huggingface.co/datasets/TIGER-Lab/MathInstruct

- orca-math-word-problems-200k
  - https://huggingface.co/datasets/microsoft/orca-math-word-problems-200k

- atlas-math-sets
  - https://huggingface.co/datasets/AtlasUnified/atlas-math-sets

- goat
  - https://huggingface.co/datasets/tiedong/goat

- camel-ai
  - https://huggingface.co/datasets/camel-ai/math

- Long-Instruction-with-Paraphrasing
  - https://huggingface.co/datasets/yuyijiong/Long-Instruction-with-Paraphrasing

- Long
  - https://huggingface.co/datasets/akoksal/LongForm
  - https://huggingface.co/datasets/THUDM/LongAlign-10k
  - https://huggingface.co/datasets/THUDM/LongCite-45k
  - https://huggingface.co/datasets/THUDM/LongWriter-6k
  - https://huggingface.co/datasets/YeungNLP/LongQLoRA-Dataset
  - https://huggingface.co/datasets/Yukang/LongAlpaca-12k
  - https://huggingface.co/datasets/togethercomputer/Long-Data-Collections

- VideoGPT-plus_Training_Dataset
  - https://huggingface.co/datasets/MBZUAI/VideoGPT-plus_Training_Dataset

- ShareGemini
  - https://huggingface.co/datasets/Share14/ShareGemini

- Movie
  - annotations: https://huggingface.co/datasets/VITA-MLLM/MovieNet-Summary
  - images: https://opendatalab.com/OpenDataLab/MovieNet

- Comic
  - https://huggingface.co/datasets/VITA-MLLM/Comic-Summary

- lmms-lab/LLaVA-Video-178K
  - https://huggingface.co/datasets/lmms-lab/LLaVA-Video-178K

  
## Custom Data

- An example JSONL file of the training data:
```
[
    ...
    {
        "conversations": [
            {
                "from": "human",
                "value": "<image><image>..."
            },
            {
                "from": "gpt",
                "value": "..."
            }
        ],
        "images": ["path/to/first/image", "path/to/second/image", ...],
    },
    {
        "conversations": [
            {
                "from": "human",
                "value": "<video><video>..."
            },
            {
                "from": "gpt",
                "value": "..."
            }
        ],
        "videos": ["path/to/first/video", "path/to/second/video", ...],
    },
    ...
]
```
