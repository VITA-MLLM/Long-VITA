#!/bin/bash

set -e
set -x

SEQ_LENGTH="$1"
if [ -z "$SEQ_LENGTH"  ]
then
    SEQ_LENGTH=16384
fi

DATA_SEQ_LENGTH="$2"
if [ -z "$DATA_SEQ_LENGTH"  ]
then
    DATA_SEQ_LENGTH=${SEQ_LENGTH}
fi

timestamp="$3"
if [ -z "$timestamp"  ]
then
    timestamp=`date +'%Y%m%d_%H'`0000
fi

######################################################################
export ROOT_PATH=/data/
export ROOT_PATH_2=/data_2/
export ROOT_PATH_4=/data_4/
export CODE_PATH=${ROOT_PATH_2}/Long-VITA/

export LOCAL_ROOT_PATH=/data_local/
export LOCAL_CODE_PATH=${LOCAL_ROOT_PATH}/Long-VITA/
mkdir -p ${LOCAL_ROOT_PATH}
mkdir -p ${LOCAL_CODE_PATH}

apt install -y rsync
mkdir -p ${LOCAL_CODE_PATH}
rsync -a --exclude ".git" --exclude ".gitee" ${CODE_PATH}/ ${LOCAL_CODE_PATH}/

######################################################################
OUTPUT_DIR=${ROOT_PATH_2}/output/LM/"$0"/${timestamp}/

mkdir -p ${OUTPUT_DIR}
rsync -avh $0 ${OUTPUT_DIR}

export HF_HOME="${ROOT_PATH_2}/data/HF_HOME_node${INDEX}/"
mkdir -p ${HF_HOME}

######################################################################
LOG=${OUTPUT_DIR}/log_node${INDEX}.txt
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"


######################################################################
DATA_PATH=${LOCAL_CODE_PATH}/configs/llm_sft_stage1.yaml
#DATA_PATH=${LOCAL_CODE_PATH}/configs/llm_sft_stage2.yaml
#DATA_PATH=${LOCAL_CODE_PATH}/configs/llm_sft.yaml

TOKENIZER_PATH=${ROOT_PATH}/models/Qwen/Qwen2.5-7B/
CKPT_LOAD_DIR=${ROOT_PATH}/models/Qwen/Qwen2.5-7B_tp4pp1/


CKPT_SAVE_DIR=${OUTPUT_DIR}/

rsync -avh ${DATA_PATH} ${OUTPUT_DIR}

######################################################################
cd ${LOCAL_CODE_PATH}
rm -fr datasets
mkdir -p datasets
ln -s ${ROOT_PATH}/data/ datasets/CV
ln -s ${ROOT_PATH}/data/LLM datasets/LLM
ln -s ${ROOT_PATH}/data/LMM datasets/LMM

######################################################################
if command -v nvcc 2>&1 >/dev/null
then
	source ${LOCAL_CODE_PATH}/scripts/set_env_mg_gpu.sh

	MEGATRON_DIR=${LOCAL_CODE_PATH}/third_party/Megatron-LM_core_r0.6.0/
	pip3 install --no-index --find-links=${ROOT_PATH}/software/ -e ${MEGATRON_DIR}
else
	source ${LOCAL_CODE_PATH}/scripts/set_env_mg_npu.sh

	MEGATRON_DIR=${LOCAL_CODE_PATH}/third_party/Megatron-LM_core_r0.6.0/
	MINDSPEED_DIR=${LOCAL_CODE_PATH}/third_party/MindSpeed_core_r0.6.0/
	MODELLINK_DIR=${LOCAL_CODE_PATH}/third_party/ModelLink/

	pip3 install --no-index --find-links=${ROOT_PATH}/software/ -e ${MEGATRON_DIR}
	pip3 install --no-index --find-links=${ROOT_PATH}/software/ -e ${MINDSPEED_DIR}
	pip3 install --no-index --find-links=${ROOT_PATH}/software/ -e ${MODELLINK_DIR}

	export ASCEND_PROCESS_LOG_PATH=${OUTPUT_DIR}/ascend/${INDEX}
	mkdir -p ${ASCEND_PROCESS_LOG_PATH}
fi

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

TP=4
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
    --num-layers 28 \
    --hidden-size 3584 \
    --ffn-hidden-size 18944 \
    --num-attention-heads 28  \
    --max-position-embeddings ${SEQ_LENGTH} \
    --seq-length ${SEQ_LENGTH} \
    --disable-bias-linear \
    --add-qkv-bias \
    --group-query-attention \
    --num-query-groups 4 \
    --use-flash-attn \
    --swiglu \
    --use-fused-swiglu \
    --normalization RMSNorm \
    --norm-epsilon 1e-6 \
    --use-fused-rmsnorm \
    --position-embedding-type rope \
    --rotary-base 1000000 \
    --use-fused-rotary-pos-emb \
    --untie-embeddings-and-output-weights \
    --micro-batch-size 1 \
    --global-batch-size 128 \
    --make-vocab-size-divisible-by 1 \
    --padded-vocab-size 152064 \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --train-iters 770 \
    --lr 1.25e-6 \
    --lr-decay-style cosine \
    --min-lr 1.25e-7 \
    --lr-warmup-fraction 0.01 \
    --init-method-std 0.01 \
    --weight-decay 1e-1 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --initial-loss-scale 4096 \
    --no-gradient-accumulation-fusion \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --bf16 \
    --use-distributed-optimizer \
    --overlap-grad-reduce \
    --finetune \
    --prompt-format "qwen2" \
    --reset-position-ids \
    --reset-attention-mask \
"
    #--transformer-impl transformer_engine \
    #--is-instruction-dataset \
    #--train-iters 230 \
    #--use-rotary-position-embeddings \

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 100,0,0 \
    --data-seq-length ${DATA_SEQ_LENGTH} \
"

CKPT_ARGS="
    --load ${CKPT_LOAD_DIR} \
    --no-load-optim \
    --no-load-rng \
    --seed 42 \
    --save ${CKPT_SAVE_DIR} \
"
    #--no-save-optim \
    #--no-save-rng \

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 100 \
    --eval-interval 100 \
    --eval-iters 0 \
    --log-throughput \
    --distributed-timeout-minutes 120 \
"

torchrun $DISTRIBUTED_ARGS ${LOCAL_CODE_PATH}/long_vita_megatron/pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $CKPT_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \

set +x
