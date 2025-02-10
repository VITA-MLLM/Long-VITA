#!/bin/bash

set -e
set -x

SEQ_LENGTH="$1"
if [ -z "$SEQ_LENGTH" ]
then
    SEQ_LENGTH=32768
fi

DATA_SEQ_LENGTH="$2"
if [ -z "$DATA_SEQ_LENGTH" ]
then
    DATA_SEQ_LENGTH=${SEQ_LENGTH}
fi

timestamp="$3"
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

TOKENIZER_PATH=${ROOT_PATH}/models/Qwen/Qwen2.5-14B-Instruct/
CKPT_LOAD_DIR=${ROOT_PATH}/models/Qwen/Qwen2.5-14B-Instruct_tp8pp1_te/

VIT_CKPT_LOAD_DIR=${ROOT_PATH}/models/OpenGVLab/InternViT-300M-448px_tp8pp1/

CKPT_SAVE_DIR=${OUTPUT_DIR}/

rsync -avh ${DATA_PATH} ${OUTPUT_DIR}

######################################################################
cd ${LOCAL_CODE_PATH}
rm -fr datasets
ln -s ${ROOT_PATH}/data datasets

######################################################################
source ${LOCAL_CODE_PATH}/scripts/set_env_mg_gpu.sh

MEGATRON_DIR=${LOCAL_CODE_PATH}/third_party/Megatron-LM_core_r0.7.0/
#pip3 install --no-index --find-links=${ROOT_PATH}/software/ -e ${MEGATRON_DIR}
pip3 install -e ${MEGATRON_DIR}

export PYTHONPATH=${MEGATRON_DIR}/:${PYTHONPATH}

######################################################################
GPUS_PER_NODE=${NPROC_PER_NODE}
NNODES=${WORLD_SIZE}
NODE_RANK=${RANK}
MASTER_PORT=34567

if [ -z "$GPUS_PER_NODE" ]
then
    GPUS_PER_NODE=8
    NNODES=1
    NODE_RANK=0
    MASTER_ADDR=127.0.0.1
fi

######################################################################
export CUDA_DEVICE_MAX_CONNECTIONS=1
#export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

VISION_SEQ_LENGTH=1025
IMAGE_TOKEN_LENGTH=256
IMAGE_SIZE=448

VISION_MODEL_TYPE=intern_300m

TP=8
PP=1


DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"


GPT_ARGS="
    --use-mcore-models \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --sequence-parallel \
    --num-layers 48 \
    --hidden-size 5120 \
    --ffn-hidden-size 13824 \
    --num-attention-heads 40 \
    --group-query-attention \
    --num-query-groups 8 \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --seq-length ${SEQ_LENGTH} \
    --max-position-embeddings ${SEQ_LENGTH} \
    --micro-batch-size 1 \
    --global-batch-size 512 \
    --make-vocab-size-divisible-by 1 \
    --padded-vocab-size 152064 \
    --rotary-base 1000000.0 \
    --lr 1.00e-3 \
    --train-iters 1000 \
    --lr-decay-style cosine \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --attention-dropout 0.0 \
    --init-method-std 0.01 \
    --hidden-dropout 0.0 \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --use-fused-rmsnorm \
    --norm-epsilon 1e-6 \
    --swiglu \
    --use-flash-attn \
    --use-fused-rotary-pos-emb \
    --use-rotary-position-embeddings \
    --use-fused-swiglu \
    --use-mc2 \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --min-lr 1.00e-5 \
    --weight-decay 0.0 \
    --lr-warmup-fraction 0.03 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --add-qkv-bias \
    --initial-loss-scale 4096 \
    --no-gradient-accumulation-fusion \
    --use-distributed-optimizer \
    --bf16 \
    --overlap-grad-reduce \
    --finetune \
    --vision-model-freeze \
    --language-model-freeze \
    --vision-model-type ${VISION_MODEL_TYPE} \
    --vision-downsample-ratio 0.5 \
    --vision-projector-type mlp \
    --vision-projector-pre-norm \
    --vision-process-type dynamic \
    --vision-normalize-type imagenet \
    --vision-seq-length ${VISION_SEQ_LENGTH} \
    --image-token-length ${IMAGE_TOKEN_LENGTH} \
    --image-size ${IMAGE_SIZE} \
    --prompt-format "qwen2" \
    --is-instruction-dataset \
    --max-num-frame 128 \
    --max-fps 1 \
    --add-class-token \
    --reset-position-ids \
    --reset-attention-mask \
    --min-patch-grid 1 \
    --max-patch-grid 12 \
    --transformer-impl transformer_engine \
"

    #--dataloader-type cyclic \
    #--reuse-fp32-param \
    #--recompute-method block \
    #--recompute-granularity full \
    #--recompute-num-layers 40 \

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 100,0,0 \
    --data-seq-length ${DATA_SEQ_LENGTH} \
    --num-workers 8 \
"

CKPT_ARGS="
    --load ${CKPT_LOAD_DIR} \
    --vit-load ${VIT_CKPT_LOAD_DIR} \
    --no-load-optim \
    --no-load-rng \
    --seed 42 \
    --save ${CKPT_SAVE_DIR} \
"
    #--no-save-optim \
    #--no-save-rng \

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 500 \
    --eval-interval 500 \
    --eval-iters 0 \
    --log-throughput \
    --distributed-timeout-minutes 120 \
"

torchrun $DISTRIBUTED_ARGS ${LOCAL_CODE_PATH}/long_vita_megatron/pretrain_long_vita.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $CKPT_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \

set +x
