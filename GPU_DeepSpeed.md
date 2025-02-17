
# ⭐ Training, Inference, and Evaluation on GPU with DeepSpeed
## Requirements and Installation

Long-VITA
```
git clone https://github.com/VITA-MLLM/Long-VITA.git
cd Long-VITA
git submodule update --init --recursive
pip install -r requirement.txt
pip install -e .
```

## Prepare Pre-trained Weight

### LLM
- Downloading LLM from https://huggingface.co/Qwen/Qwen2.5-14B-Instruct.

### VIT
- Downloading VIT from https://huggingface.co/OpenGVLab/InternViT-300M-448px.


## Training
### Stage-1 (Vision-Language Alignment)
This stage requires at least 8 GPUs, each with at least 96G memory.  (We only test on GPUs with 96G memory, and 80G GPUs may also work.)

```
bash scripts/deepspeed/qwen25/finetune_qwen25_14b_intern_300m_stage1.sh 16384 `date +'%Y%m%d_%H'`0000
```

The above script may need some adjustment.

- Set `MODEL_NAME_OR_PATH` to the LLM Huggingface weight folder
- Set `VISION_MODEL_NAME_OR_PATH` to the VIT Huggingface weight folder
- Modify other variables to suit the environment.

### Stage-2 (Long-VITA-16K)
This stage requires at least 8 GPUs, each with at least 96G memory. (We only test on GPUs with 96G memory, and 80G GPUs may also work.)

```
bash scripts/megatron/qwen25/finetune_qwen25_14b_intern_300m_stage2.sh 16384 `date +'%Y%m%d_%H'`0000
```

The above script may need some adjustment.

- Set `MODEL_NAME_OR_PATH` to the output folder of the Stage-1
- Modify other variables to suit the environment.


### Stage-3 (Long-VITA-128K)
Not Implemented Yet


### Stage-4 (Long-VITA-1M)
Not Implemented Yet


## Inference

We provide the converted Huggingface weights from MindSpeed weights in

- https://huggingface.co/VITA-MLLM/Long-VITA-16K_HF
- https://huggingface.co/VITA-MLLM/Long-VITA-128K_HF
- https://huggingface.co/VITA-MLLM/Long-VITA-1M_HF

Here we implement a simple script for inference
```
python tools/inference_long_vita.py VITA-MLLM/Long-VITA-16K_HF
python tools/inference_long_vita.py VITA-MLLM/Long-VITA-128K_HF
python tools/inference_long_vita.py VITA-MLLM/Long-VITA-1M_HF
```


## Evaluation

Evaluate with VLMEvalKit
```
bash VLMEvalKit/evaluation_long_vita_hf.sh
```


