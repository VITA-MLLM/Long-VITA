#!/bin/bash

set -e
set -x


export ROOT_PATH=/data/
export CODE_PATH=${ROOT_PATH}/Long-VITA/

export LOCAL_ROOT_PATH=/data_local/
export LOCAL_CODE_PATH=${LOCAL_ROOT_PATH}/Long-VITA/

#MEGATRON_DIR=${LOCAL_CODE_PATH}/third_party/Megatron-LM_core_r0.7.0

#export PYTHONPATH=${MEGATRON_DIR}/:${PYTHONPATH}

apt install -y rsync
mkdir -p ${LOCAL_CODE_PATH}
rsync -a --exclude ".git" --exclude ".gitee" ${CODE_PATH}/ ${LOCAL_CODE_PATH}/


export CUDA_DEVICE_MAX_CONNECTIONS=1

######################################################################


# Huggingface to Megatron
if true
then
	LOAD_DIR=${ROOT_PATH}/models/Qwen/Qwen2.5-14B-Instruct/
	SAVE_DIR=${ROOT_PATH}/models/Qwen/Qwen2.5-14B-Instruct_tp8pp1_te/
	cd third_party/Pai-Megatron-Patch/toolkits/model_checkpoints_convertor/qwen
	bash hf2mcore_qwen2.5_convertor.sh \
		14B \
		${LOAD_DIR} \
		${SAVE_DIR} \
		8 \
		1 \
		bf16 \
		true \
		false 
fi
