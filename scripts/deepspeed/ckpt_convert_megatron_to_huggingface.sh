
set -e
set -x


export ROOT_PATH=/data/


TOKENIZER_PATH=${ROOT_PATH}/models/Qwen/Qwen2.5-14B-Instruct/
MODEL_NAME_OR_PATH=${ROOT_PATH}/models/Qwen/Qwen2.5-14B-Instruct/

MEGATRON_DIR=${ROOT_PATH}/long_vita/third_party/Megatron-LM_core_r0.7.0/
#pip3 install -e ${MEGATRON_DIR}

export PYTHONPATH=${MEGATRON_DIR}/:${PYTHONPATH}

export NVTE_FLASH_ATTN=1
export NVTE_FUSED_ATTN=0 

# Long-VITA-16K
CKPT_LOAD_DIR=/data/output/LM/scripts/modellink/qwen25/finetune_qwen25_14b_intern_300m_ptd_tp8pp1_stage2.sh/20241014_131952_mg/
CKPT_SAVE_DIR=/data/output/LM/scripts/modellink/qwen25/finetune_qwen25_14b_intern_300m_ptd_tp8pp1_stage2.sh/20241014_131952_hf/


# Long-VITA-128K
#CKPT_LOAD_DIR=/data/output/LM/scripts/modellink/qwen25/finetune_qwen25_14b_intern_300m_ptd_tp8pp1cp2_stage3.sh/20241127_204213_mg/
#CKPT_SAVE_DIR=/data/output/LM/scripts/modellink/qwen25/finetune_qwen25_14b_intern_300m_ptd_tp8pp1cp2_stage3.sh/20241127_204213_hf/


# Long-VITA-1M
#CKPT_LOAD_DIR=/data/output/LM/scripts/modellink/qwen25/finetune_qwen25_14b_intern_300m_ptd_tp8pp1cp8_stage4.sh/20241128_234743_mg/
#CKPT_SAVE_DIR=/data/output/LM/scripts/modellink/qwen25/finetune_qwen25_14b_intern_300m_ptd_tp8pp1cp8_stage4.sh/20241128_234743_hf/


export CUDA_DEVICE_MAX_CONNECTIONS=1

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=45678

VISION_SEQ_LENGTH=1025
IMAGE_TOKEN_LENGTH=256
IMAGE_SIZE=448

VISION_MODEL_TYPE=intern_300m

TP=8
PP=1
SEQ_LENGTH=1310720

python tools/hf2mcore_long_vita.py \
    --use-mcore-models \
    --target-tensor-model-parallel-size ${TP} \
    --target-pipeline-model-parallel-size ${PP} \
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
    --add-qkv-bias \
    --bf16 \
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
    --min-patch-grid 1 \
    --max-patch-grid 12 \
    --transformer-impl transformer_engine \
    --load ${CKPT_LOAD_DIR} \
    --save ${CKPT_SAVE_DIR} \
    --hf-ckpt-path ${MODEL_NAME_OR_PATH} \
    --convert-checkpoint-from-megatron-to-transformers \
    --mock-data \
    --config_name long_vita/models/long_vita_qwen2_intern/config_14B.json \
    --save-interval 1 \
    #--use-cpu-initialization \

python tools/inference_long_vita.py ${CKPT_SAVE_DIR}
