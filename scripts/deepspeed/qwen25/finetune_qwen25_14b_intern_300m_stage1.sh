#!/bin/bash

set -e
set -x

SEQ_LENGTH="$1"
if [ -z "$SEQ_LENGTH" ]
then
    SEQ_LENGTH=32768
fi

timestamp="$2"
if [ -z "$timestamp" ]
then
    timestamp=`date +'%Y%m%d_%H'`0000
fi

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

export HF_HOME="${ROOT_PATH}/data/HF_HOME_node${INDEX}/"
mkdir -p ${HF_HOME}

######################################################################
LOG=${OUTPUT_DIR}/log_node${INDEX}.txt
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"


######################################################################
DATA_PATH=${LOCAL_CODE_PATH}/configs/long_vita_finetune_stage1.yaml

MODEL_NAME_OR_PATH=${ROOT_PATH}/models/Qwen/Qwen2.5-14B-Instruct/

VISION_MODEL_NAME_OR_PATH=${ROOT_PATH}/models/OpenGVLab/InternViT-300M-448px/

CKPT_SAVE_DIR=${OUTPUT_DIR}/

rsync -avh ${DATA_PATH} ${OUTPUT_DIR}

######################################################################
cd ${LOCAL_CODE_PATH}
rm -fr datasets
ln -s ${ROOT_PATH}/data datasets

######################################################################
source ${LOCAL_CODE_PATH}/scripts/set_env_ds_gpu.sh
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

######################################################################
GPUS_PER_NODE=${NPROC_PER_NODE}
NNODES=${WORLD_SIZE}
NODE_RANK=${RANK}
MASTER_PORT=45678

if [ -z "$GPUS_PER_NODE" ]
then
    GPUS_PER_NODE=8
    NNODES=1
    NODE_RANK=0
    MASTER_ADDR=127.0.0.1
fi

######################################################################
#export ASCEND_LAUNCH_BLOCKING=1
#export WITHOUT_JIT_COMPILE=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

VISION_SEQ_LENGTH=1025
IMAGE_TOKEN_LENGTH=256
IMAGE_SIZE=448

VISION_MODEL_TYPE=intern_300m

######################################################################
DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

unset ASCEND_LAUNCH_BLOCKING
unset TASK_QUEUE_ENABLE

torchrun $DISTRIBUTED_ARGS tools/finetune_long_vita.py \
    --do_train \
    --overwrite_output_dir \
    --config_name long_vita/models/long_vita_qwen2_intern/config_14B.json \
    --tokenizer_name $MODEL_NAME_OR_PATH \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --dataset_name $DATA_PATH \
    --bf16 True \
    --tf32 True \
    --torch_dtype bfloat16 \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 1 \
    --max_steps 1000 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 0.05 \
    --save_total_limit 2 \
    --learning_rate 1.00e-3 \
    --max_grad_norm 1.0 \
    --weight_decay 0.0 \
    --adam_beta1 0.9 \
    --adam_beta2 0.999 \
    --adam_epsilon 1e-6 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to "tensorboard" \
    --model_max_length ${SEQ_LENGTH} \
    --gradient_checkpointing True \
    --deepspeed ${SCRIPT_DIR}/../ds_config_zero2.json \
    --trust_remote_code False \
    --ddp_timeout=7200 \
    --ddp_backend ${DISTRIBUTED_BACKEND} \
    --attn_implementation flash_attention_2 \
    --vision-model-freeze \
    --language-model-freeze \
    --vision_model_name_or_path ${VISION_MODEL_NAME_OR_PATH} \
    --vision_model_type ${VISION_MODEL_TYPE} \
    --vision_downsample_ratio 0.5 \
    --vision_projector_type mlp \
    --vision_projector_pre_norm \
    --vision_process_type dynamic \
    --vision_normalize_type imagenet \
    --image_token_length ${IMAGE_TOKEN_LENGTH} \
    --image_size ${IMAGE_SIZE} \
    --max_num_frame 64 \
    --max_fps 1 \
    --min_patch_grid 1 \
    --max_patch_grid 12 \
    --seed 42 \
    --data_seed 42 \
    --reset_attention_mask \
    --reset_position_ids \
    --create_attention_mask false \
    --create_attention_mask_2d false \
    --dataloader_num_workers 2 \

    #--dataset_joint false \
    #--tokenizer_name_or_path Qwen2Tokenizer \

    #--bf16 True \
    #--fp16 True \
    #--tf32 True \

set +x
