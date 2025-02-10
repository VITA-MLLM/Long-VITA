#!/bin/bash

set -e
set -x


export ROOT_PATH=/data/
export CODE_PATH=${ROOT_PATH}/Long-VITA/

Megatron_path=${CODE_PATH}/third_party/Megatron-LM_core_r0.6.0/

export PYTHONPATH=${Megatron_path}/:${PYTHONPATH}


if true
then
	load_dir="${ROOT_PATH}/models/OpenGVLab/InternViT-300M-448px/"
	save_dir="${ROOT_PATH}/models/OpenGVLab/InternViT-300M-448px_tp8pp1"
	rm -fr ${save_dir}
	mkdir -p ${save_dir}
	python3 ${CODE_PATH}/long_vita_modellink/ckpt_converter_intern_vit.py --download-root ${load_dir} --output ${save_dir} --tensor-parallel-size 8
       	# -use-te-layernorm-linear

fi

if false
then
	load_dir="${ROOT_PATH}/models/OpenGVLab/InternViT-6B-448px-V1-5/"
	save_dir="${ROOT_PATH}/models/OpenGVLab/InternViT-6B-448px-V1-5_tp1pp1"
	rm -fr ${save_dir}
	mkdir -p ${save_dir}
	python3 ${CODE_PATH}/long_vita_modellink/ckpt_converter_intern_vit.py --download-root ${load_dir} --output ${save_dir} --tensor-parallel-size 1
       	# -use-te-layernorm-linear

fi

if false
then
	load_dir="${ROOT_PATH}/models/OpenGVLab/InternViT-6B-448px-V2_5/"
	save_dir="${ROOT_PATH}/models/OpenGVLab/InternViT-6B-448px-V2_5_tp8pp1"
	rm -fr ${save_dir}
	mkdir -p ${save_dir}
	python3 ${CODE_PATH}/long_vita_modellink/ckpt_converter_intern_vit.py --download-root ${load_dir} --output ${save_dir} --tensor-parallel-size 8
       	# -use-te-layernorm-linear

fi

set +x
