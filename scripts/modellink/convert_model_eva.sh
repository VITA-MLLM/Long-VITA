#!/bin/bash

set -e
set -x


export ROOT_PATH=/data/
export CODE_PATH=${ROOT_PATH}/Long-VITA/

Megatron_path=${CODE_PATH}/third_party/THUDM/Megatron-LM/
#Megatron_path=${CODE_PATH}/third_party/Megatron-LM_core_r0.6.0/

export PYTHONPATH=${Megatron_path}/:${PYTHONPATH}


if true
then
	load_dir="${ROOT_PATH}/models/THUDM/SwissArmyTransformer/eva-clip-4b-14-x-drop-last-layer"
	save_dir="${ROOT_PATH}/models/THUDM/SwissArmyTransformer/eva-clip-4b-14-x-drop-last-layer-mcore_tp8pp1_imsz336"

	python3 long_vita_modellink/tools/checkpoint/convert_ckpt.py \
		--model-type EVA \
		--loader eva_sat \
		--saver eva_mcore \
		--load-dir ${load_dir} \
		--save-dir ${save_dir} \
		--target-tensor-parallel-size 8 \
		--target-pipeline-parallel-size 1 \
		--target-image-size 336

fi

set +x
