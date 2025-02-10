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
if false
then
	#load_dir="${ROOT_PATH}/models/mistralai/Mixtral-8x7B-v0.1/"
	#save_dir="${ROOT_PATH}/models/mistralai/Mixtral-8x7B-v0.1_tp1pp8ep2_nll25544444/"

	load_dir="${ROOT_PATH}/models/mistralai/Mixtral-8x7B-Instruct-v0.1/"
	save_dir="${ROOT_PATH}/models/mistralai/Mixtral-8x7B-Instruct-v0.1_tp4pp1ep1/"

	python long_vita_modellink/tools/checkpoint/convert_ckpt.py \
		--model-type GPT \
		--loader mixtral_hf \
		--saver mixtral \
		--load-dir ${load_dir} \
		--save-dir ${save_dir} \
		--tokenizer-model ${load_dir}/tokenizer.model \
		--target-tensor-parallel-size 8 \
		--target-pipeline-parallel-size 1 \
		--target-expert-parallel-size 1 \
		#--target-num-layer-list 2,5,5,4,4,4,4,4

fi

# Huggingface to Mcore
if false
then
	load_dir="${ROOT_PATH}/models/mistralai/Mixtral-8x7B-Instruct-v0.1/"
	save_dir="${ROOT_PATH}/models/mistralai/Mixtral-8x7B-Instruct-v0.1_tp8pp1ep1_mcore/"

	#load_dir="${ROOT_PATH}/models/mistralai/Mixtral-8x7B-Instruct-v0.1/"
	#save_dir="${ROOT_PATH}/models/mistralai/Mixtral-8x7B-Instruct-v0.1_tp8-pp2-ep1-fp14/"
	python ${MODELLINK_DIR}/tools/checkpoint/convert_ckpt.py \
		--model-type GPT \
		--loader hf_mcore \
		--saver mg_mcore \
		--load-dir ${load_dir} \
		--save-dir ${save_dir} \
		--tokenizer-model ${load_dir}/tokenizer.model \
		--target-tensor-parallel-size 8 \
		--target-pipeline-parallel-size 1 \
		--target-expert-parallel-size 1 \
		--use-mcore-models \
		--model-type-hf mixtral

fi

# Huggingface to Mcore
if false
then
	load_dir="${ROOT_PATH}/models/meta-llama/Llama-2-7b-chat-hf/"
	save_dir="${ROOT_PATH}/models/meta-llama/Llama-2-7b-chat-hf_tp1pp1_mcore/"

	python ${MODELLINK_DIR}/tools/checkpoint/convert_ckpt.py \
		--use-mcore-models \
		--model-type-hf llama2 \
		--model-type GPT \
		--loader hf_mcore \
		--saver mg_mcore \
		--params-dtype bf16 \
		--load-dir ${load_dir} \
		--save-dir ${save_dir} \
		--tokenizer-model ${load_dir}/tokenizer.json \
		--target-tensor-parallel-size 1 \
		--target-pipeline-parallel-size 1 \

fi

# Huggingface to Mcore
if false
then
	load_dir="${ROOT_PATH}/models/lmsys/vicuna-7b-v1.5/"
	save_dir="${ROOT_PATH}/models/lmsys/vicuna-7b-v1.5_tp2pp1_mcore/"

	python ${MODELLINK_DIR}/tools/checkpoint/convert_ckpt.py \
		--use-mcore-models \
		--model-type-hf llama2 \
		--model-type GPT \
		--loader hf_mcore \
		--saver mg_mcore \
		--params-dtype bf16 \
		--load-dir ${load_dir} \
		--save-dir ${save_dir} \
		--tokenizer-model ${load_dir}/tokenizer.json \
		--target-tensor-parallel-size 2 \
		--target-pipeline-parallel-size 1 \

fi

# Huggingface to Mcore
if false
then
	load_dir="${ROOT_PATH}/models/NousResearch/Meta-Llama-3.1-8B-Instruct/"
	save_dir="${ROOT_PATH}/models/NousResearch/Meta-Llama-3.1-8B-Instruct_tp2pp1_mcore/"

	python ${MODELLINK_DIR}/tools/checkpoint/convert_ckpt.py \
		--use-mcore-models \
		--model-type-hf llama2 \
		--model-type GPT \
		--loader hf_mcore \
		--saver mg_mcore \
		--params-dtype bf16 \
		--load-dir ${load_dir} \
		--save-dir ${save_dir} \
		--tokenizer-model ${load_dir}/tokenizer.json \
		--target-tensor-parallel-size 2 \
		--target-pipeline-parallel-size 1 \

fi


# Huggingface to Mcore
if false
then
	load_dir="${ROOT_PATH}/models/mistralai/Mistral-Nemo-Instruct-2407/"
	save_dir="${ROOT_PATH}/models/mistralai/Mistral-Nemo-Instruct-2407_tp4pp1_mcore/"

	python ${MODELLINK_DIR}/tools/checkpoint/convert_ckpt.py \
		--use-mcore-models \
		--model-type-hf llama2 \
		--model-type GPT \
		--loader hf_mcore \
		--saver mg_mcore \
		--params-dtype bf16 \
		--load-dir ${load_dir} \
		--save-dir ${save_dir} \
		--tokenizer-model ${load_dir}/tokenizer.json \
		--target-tensor-parallel-size 4 \
		--target-pipeline-parallel-size 1 \

fi


