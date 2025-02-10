#!/bin/bash

set -e
set -x

timestamp=`date +'%Y%m%d_%H%M%S'`

######################################################################
export ROOT_PATH=/data/
export CODE_PATH=${ROOT_PATH}/Long-VITA/

export LOCAL_ROOT_PATH=/data_local/
export LOCAL_CODE_PATH=${LOCAL_ROOT_PATH}/Long-VITA/
mkdir -p ${LOCAL_ROOT_PATH}
mkdir -p ${LOCAL_CODE_PATH}

apt install -y rsync
mkdir -p ${LOCAL_CODE_PATH}
rsync -a --exclude ".git" --exclude ".gitee" ${CODE_PATH}/ ${LOCAL_CODE_PATH}/

######################################################################
OUTPUT_DIR=${ROOT_PATH}/output/LM/"$0"/${timestamp}/

mkdir -p ${OUTPUT_DIR}
rsync -avh $0 ${OUTPUT_DIR}

export HF_HOME="${ROOT_PATH}/data/HF_HOME/"
mkdir -p ${HF_HOME}
#export HF_ENDPOINT=https://hf-mirror.com



######################################################################
LOG=${OUTPUT_DIR}/log.txt
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

######################################################################
VLMEvalKit_DIR=${LOCAL_CODE_PATH}/third_party/VLMEvalKit/
pip3 install --no-deps -e ${VLMEvalKit_DIR}
pip3 install moviepy==1.0.3

######################################################################
export ASCEND_RT_VISIBLE_DEVICES=0
unset RANK
unset WORLD_SIZE

export LMUData=${ROOT_PATH}/data/LMUData/
mkdir -p ${LMUData}

cd VLMEvalKit

######################################################################
HF_TOKEN=""

#huggingface-cli download --repo-type dataset --token ${HF_TOKEN} --resume-download lmms-lab/Video-MME
#huggingface-cli download --repo-type dataset --token ${HF_TOKEN} --resume-download OpenGVLab/MVBench
#huggingface-cli download --repo-type dataset --token ${HF_TOKEN} --resume-download opencompass/MMBench-Video
#huggingface-cli download --repo-type dataset --token ${HF_TOKEN} --resume-download MLVU/MVLU
#huggingface-cli download --repo-type dataset --token ${HF_TOKEN} --resume-download lmms-lab/TempCompass
#huggingface-cli download --repo-type dataset --token ${HF_TOKEN} --resume-download longvideobench/LongVideoBench
#huggingface-cli download --repo-type dataset --token ${HF_TOKEN} --resume-download yifanzhang114/MME-RealWorld-Base64
#exit 0

######################################################################
# judge / choice extractor
# lmdeploy serve api_server --backend pytorch --device ascend /data/models/Qwen/Qwen1.5-1.8B-Chat/ --server-port 23333
#export LOCAL_LLM=/data/models/Qwen/Qwen1.5-1.8B-Chat/


#export OPENAI_API_KEY=sk-123456
#export OPENAI_API_BASE=https://123456

export LongVITA_URL=http://127.0.0.1:5001/api


MODEL=LongVITA
NPROC=1


if true
then

	for NFRAME in 64 128 256 512 1024
	#for NFRAME in 2048 4096
	do

		export MAX_NUM_FRAME=${NFRAME}

		#python3 run.py --data Video-MME --model ${MODEL} --verbose --work-dir ${OUTPUT_DIR} --nframe ${NFRAME} --use-subtitle
		python3 run.py --data Video-MME --model ${MODEL} --verbose --work-dir ${OUTPUT_DIR} --nframe ${NFRAME}

		python3 run.py --data MVBench --model ${MODEL} --verbose --work-dir ${OUTPUT_DIR} --nframe ${NFRAME}

		#python3 run.py --data MMBench-Video --model ${MODEL} --verbose --work-dir ${OUTPUT_DIR} --nframe ${NFRAME}

		#python3 run.py --data MLVU --model ${MODEL} --verbose --work-dir ${OUTPUT_DIR} --nframe ${NFRAME}

		#python3 run.py --data TempCompass --model ${MODEL} --verbose --work-dir ${OUTPUT_DIR} --nframe ${NFRAME}

		python3 run.py --data LongVideoBench --model ${MODEL} --verbose --work-dir ${OUTPUT_DIR} --nframe ${NFRAME}

	done

	exit 0
fi


######################################################################
# OpenCompass

python3 run.py --data MMBench_DEV_EN_V11 --model ${MODEL} --verbose --work-dir ${OUTPUT_DIR}
python3 run.py --data MMBench_DEV_CN_V11 --model ${MODEL} --verbose --work-dir ${OUTPUT_DIR}
python3 run.py --data MMBench_TEST_EN_V11 --model ${MODEL} --verbose --work-dir ${OUTPUT_DIR}
python3 run.py --data MMBench_TEST_CN_V11 --model ${MODEL} --verbose --work-dir ${OUTPUT_DIR}

python3 run.py --data MMStar --model ${MODEL} --verbose --work-dir ${OUTPUT_DIR}

python3 run.py --data MMMU_DEV_VAL --model ${MODEL} --verbose --work-dir ${OUTPUT_DIR}

python3 run.py --data MMVet --model ${MODEL} --verbose --work-dir ${OUTPUT_DIR}

python3 run.py --data HallusionBench --model ${MODEL} --verbose --work-dir ${OUTPUT_DIR}

python3 run.py --data AI2D_TEST --model ${MODEL} --verbose --work-dir ${OUTPUT_DIR}
#python3 run.py --data AI2D_TEST_NO_MASK --model ${MODEL} --verbose --work-dir ${OUTPUT_DIR}

python3 run.py --data OCRBench --model ${MODEL} --verbose --work-dir ${OUTPUT_DIR}

python3 run.py --data MathVista_MINI --model ${MODEL} --verbose --work-dir ${OUTPUT_DIR}


######################################################################
# MCQ

python3 run.py --data SEEDBench_IMG --model ${MODEL} --verbose --work-dir ${OUTPUT_DIR}
#python3 run.py --data SEEDBench2 --model ${MODEL} --verbose --work-dir ${OUTPUT_DIR}
#python3 run.py --data SEEDBench2_Plus --model ${MODEL} --verbose --work-dir ${OUTPUT_DIR}

python3 run.py --data ScienceQA_TEST --model ${MODEL} --verbose --work-dir ${OUTPUT_DIR}

python3 run.py --data RealWorldQA --model ${MODEL} --verbose --work-dir ${OUTPUT_DIR}

python3 run.py --data MMT-Bench_VAL --model ${MODEL} --verbose --work-dir ${OUTPUT_DIR}

#python3 run.py --data AesBench_VAL --model ${MODEL} --verbose --work-dir ${OUTPUT_DIR}

python3 run.py --data BLINK --model ${MODEL} --verbose --work-dir ${OUTPUT_DIR}

#python3 run.py --data Q-Bench1_VAL --model ${MODEL} --verbose --work-dir ${OUTPUT_DIR}

#python3 run.py --data A-Bench_VAL --model ${MODEL} --verbose --work-dir ${OUTPUT_DIR}

python3 run.py --data MME-RealWorld --model ${MODEL} --verbose --work-dir ${OUTPUT_DIR}
#python3 run.py --data MME-RealWorld-CN --model ${MODEL} --verbose --work-dir ${OUTPUT_DIR}


######################################################################
# YON
python3 run.py --data MME --model ${MODEL} --verbose --work-dir ${OUTPUT_DIR}

python3 run.py --data POPE --model ${MODEL} --verbose --work-dir ${OUTPUT_DIR}



######################################################################
# VQA



#python3 run.py --data MMBench-Video --model ${MODEL} --verbose --work-dir ${OUTPUT_DIR} --nframe 128

#python3 run.py --data OCRVQA_TESTCORE --model ${MODEL} --verbose --work-dir ${OUTPUT_DIR}
#python3 run.py --data OCRVQA_TEST --model ${MODEL} --verbose --work-dir ${OUTPUT_DIR}

#python3 run.py --data TextVQA_VAL --model ${MODEL} --verbose --work-dir ${OUTPUT_DIR}

#python3 run.py --data ChartQA_TEST --model ${MODEL} --verbose --work-dir ${OUTPUT_DIR}

#python3 run.py --data LLaVABench --model ${MODEL} --verbose --work-dir ${OUTPUT_DIR}

#python3 run.py --data DocVQA_VAL --model ${MODEL} --verbose --work-dir ${OUTPUT_DIR}
#python3 run.py --data DocVQA_TEST --model ${MODEL} --verbose --work-dir ${OUTPUT_DIR}

#python3 run.py --data InfoVQA_VAL --model ${MODEL} --verbose --work-dir ${OUTPUT_DIR}

#python3 run.py --data CORE_MM (MTI) --model ${MODEL} --verbose --work-dir ${OUTPUT_DIR}

#python3 run.py --data MLLMGuard_DS --model ${MODEL} --verbose --work-dir ${OUTPUT_DIR}




set +x
