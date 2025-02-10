#!/bin/bash

set -e
set -x


export ROOT_PATH=/data/
export CODE_PATH=${ROOT_PATH}/Long-VITA/

Megatron_path=${CODE_PATH}/third_party/Megatron-LM/
#Megatron_path=${CODE_PATH}/third_party/Megatron-LM_core_r0.6.0/

export PYTHONPATH=${Megatron_path}/:${PYTHONPATH}


if true
then
	load_dir="${ROOT_PATH}/models/google/siglip-so400m-patch14-384/"
	save_dir="${ROOT_PATH}/models/google/siglip-so400m-patch14-384_tp8pp1"
	rm -fr ${save_dir}
	mkdir -p ${save_dir}
	python3 ${CODE_PATH}/long_vita_modellink/ckpt_converter_siglip.py --download-root ${load_dir} --output ${save_dir} --tensor-parallel-size 8
       	# -use-te-layernorm-linear

fi

