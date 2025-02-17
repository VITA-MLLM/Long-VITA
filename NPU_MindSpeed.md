
# ⭐ Training, Inference, and Evaluation on NPU with MindSpeed
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

- Convert Huggingface weight to Megatron
  ```
  bash scripts/modellink/qwen25/ckpt_convert_qwen25.sh
  ```
### VIT
- Downloading VIT from https://huggingface.co/OpenGVLab/InternViT-300M-448px.

- Convert Huggingface weight to Megatron
  ```
  bash scripts/modellink/convert_model_intern_vit.sh
  ```

## Training
### Stage-1 (Vision-Language Alignment)
This stage requires at least 8 NPUs, each with at least 64G memory.

```
bash scripts/modellink/qwen25/finetune_qwen25_14b_intern_300m_ptd_tp8pp1_stage1.sh 32768 32768 `date +'%Y%m%d_%H'`0000
```

The above script may need some adjustment.

- Set `TOKENIZER_PATH` to the LLM Huggingface weight folder
- Set `CKPT_LOAD_DIR` to the LLM Megatron weight folder
- Set `VIT_CKPT_LOAD_DIR` to the VIT Megatron weight folder
- Modify other variables to suit the environment.

### Stage-2 (Long-VITA-16K)
This stage requires at least 8 NPUs, each with at least 64G memory.

```
bash scripts/modellink/qwen25/finetune_qwen25_14b_intern_300m_ptd_tp8pp1_stage2.sh 16384 16384 `date +'%Y%m%d_%H'`0000
```

The above script may need some adjustment.

- Set `TOKENIZER_PATH` to the LLM Huggingface weight folder
- Set `CKPT_LOAD_DIR` to the output folder of the Stage-1
- Set `VIT_CKPT_LOAD_DIR` to `"/"`
- Modify other variables to suit the environment.


### Stage-3 (Long-VITA-128K)
This stage requires at least 16 NPUs, each with at least 64G memory.

```
bash scripts/modellink/qwen25/finetune_qwen25_14b_intern_300m_ptd_tp8pp1cp2_stage3.sh 131072 131072 `date +'%Y%m%d_%H'`0000
```

The above script may need some adjustment.

- Set `TOKENIZER_PATH` to the LLM Huggingface weight folder
- Set `CKPT_LOAD_DIR` to the output folder of the Stage-2
- Set `VIT_CKPT_LOAD_DIR` to `"/"`
- Modify other variables to suit the environment.


### Stage-4 (Long-VITA-1M)
This stage requires at least 64 NPUs, each with at least 64G memory.

```
bash scripts/modellink/qwen25/finetune_qwen25_14b_intern_300m_ptd_tp8pp1cp8_stage4.sh 1048576 1048576 `date +'%Y%m%d_%H'`0000
```

The above script may need some adjustment.

- Set `TOKENIZER_PATH` to the LLM Huggingface weight folder
- Set `CKPT_LOAD_DIR` to the output folder of the Stage-3
- Set `VIT_CKPT_LOAD_DIR` to `"/"`
- Modify other variables to suit the environment.


## Inference
### Start the model server

```
bash scripts/modellink/qwen25/inference_qwen25_14b_intern_300m_server.sh 1048576 1048576 `date +'%Y%m%d_%H'`0000
```

Set up the model server for long context
```
bash scripts/modellink/qwen25/inference_qwen25_14b_intern_300m_server_cp.sh 1048576 1048576 `date +'%Y%m%d_%H'`0000
```
The above script may need some adjustment.

- Set `TOKENIZER_PATH` to the LLM Huggingface weight folder
- Set `CKPT_LOAD_DIR` to the output folder of any of the above stages.
  We release Long-VITA-16K, Long-VITA-128K, and Long-VITA-1M on https://huggingface.co/VITA-MLLM.
- Set `VIT_CKPT_LOAD_DIR` to `"/"`
- Modify other variables to suit the environment.

### Use the model API
Set `LongVITA_URL` to the server address, which is printed out when the above model server starts.
```
export LongVITA_URL=http://127.0.0.1:5001/api
```

Infer
```
python long_vita_modellink/inference_long_vita.py
```

## Evaluation

Evaluate with VLMEvalKit
```
bash VLMEvalKit/evaluation_long_vita.sh
```


