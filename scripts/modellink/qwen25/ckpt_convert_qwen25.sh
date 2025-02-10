#!/bin/bash

set -e
set -x


export ROOT_PATH=/data/
export CODE_PATH=${ROOT_PATH}/Long-VITA/

export LOCAL_ROOT_PATH=/data_local/
export LOCAL_CODE_PATH=${LOCAL_ROOT_PATH}/Long-VITA/

# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

MEGATRON_DIR=${LOCAL_CODE_PATH}/third_party/Megatron-LM_core_r0.6.0
MODELLINK_DIR=${LOCAL_CODE_PATH}/third_party/ModelLink/

export PYTHONPATH=${MEGATRON_DIR}/:${PYTHONPATH}
export PYTHONPATH=${LOCAL_CODE_PATH}/long_vita_modellink/:${PYTHONPATH}

apt install -y rsync
mkdir -p ${LOCAL_CODE_PATH}
rsync -a --exclude ".git" --exclude ".gitee" ${CODE_PATH}/ ${LOCAL_CODE_PATH}/


export CUDA_DEVICE_MAX_CONNECTIONS=1

######################################################################


# Huggingface to Megatron
if true
then

	#LOAD_DIR=${ROOT_PATH}/models/Qwen/Qwen2.5-72B-Instruct/
	#SAVE_DIR=${ROOT_PATH}/models/Qwen/Qwen2.5-72B-Instruct_tp8pp8/

	LOAD_DIR=${ROOT_PATH}/models/Qwen/Qwen2.5-14B-Instruct/
	SAVE_DIR=${ROOT_PATH}/models/Qwen/Qwen2.5-14B-Instruct_tp8pp1/

	#LOAD_DIR=${ROOT_PATH}/models/Qwen/Qwen2.5-7B/
	#SAVE_DIR=${ROOT_PATH}/models/Qwen/Qwen2.5-7B_tp4pp1/

	# 设置需要的权重转换参数
	python ${MODELLINK_DIR}/convert_ckpt.py \
		--use-mcore-models \
		--model-type GPT \
		--load-model-type hf \
		--save-model-type mg \
		--target-tensor-parallel-size 8 \
		--target-pipeline-parallel-size 1 \
		--add-qkv-bias \
		--load-dir ${LOAD_DIR} \
		--save-dir ${SAVE_DIR} \
		--tokenizer-model ${LOAD_DIR}/tokenizer.json \
		--model-type-hf llama2 \
		--params-dtype bf16

fi

# Mcore to Huggingface
if true
then

	#LOAD_DIR=${ROOT_PATH}/output/LM/long_vita_modellink/scripts/qwen25/pretrain_qwen25_7b_ptd_tp4pp1.sh/20241114_122341/

	#LOAD_DIR=${ROOT_PATH}/output/LM/long_vita_modellink/scripts/qwen25/pretrain_qwen25_7b_ptd_tp4pp1.sh/20241113_094150/
	LOAD_DIR=${ROOT_PATH}/output/LM/long_vita_modellink/scripts/qwen25/pretrain_qwen25_7b_ptd_tp4pp1.sh/20241115_020000/

	SAVE_DIR=${ROOT_PATH}/models/Qwen/Qwen2.5-7B-Instruct/

	rm -fr ${SAVE_DIR}/mg2hf/

	python ${MODELLINK_DIR}/convert_ckpt.py \
		--use-mcore-models \
		--model-type GPT \
		--model-type-hf llama2 \
		--load-model-type mg \
		--save-model-type hf \
		--target-tensor-parallel-size 1 \
		--target-pipeline-parallel-size 1 \
		--add-qkv-bias \
		--load-dir ${LOAD_DIR} \
		--save-dir ${SAVE_DIR}

	rsync -avh -P ${SAVE_DIR}/mg2hf/ ${LOAD_DIR}/mg2hf/
	rsync -avh -P ${SAVE_DIR}/tokenizer.json ${LOAD_DIR}/mg2hf/
	rsync -avh -P ${SAVE_DIR}/tokenizer_config.json ${LOAD_DIR}/mg2hf/
	rsync -avh -P ${SAVE_DIR}/vocab.json ${LOAD_DIR}/mg2hf/
	rsync -avh -P ${SAVE_DIR}/merges.txt ${LOAD_DIR}/mg2hf/

	rm -fr ${SAVE_DIR}/mg2hf/

fi


