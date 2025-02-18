
set -e
set -x

# Long-VITA-16K
CKPT_LOAD_DIR=/data/output/LM/scripts/modellink/qwen25/finetune_qwen25_14b_intern_300m_ptd_tp8pp1_stage2.sh/20241014_131952/
CKPT_SAVE_DIR=/data/output/LM/scripts/modellink/qwen25/finetune_qwen25_14b_intern_300m_ptd_tp8pp1_stage2.sh/20241014_131952_mg/


# Long-VITA-128K
#CKPT_LOAD_DIR=/data/output/LM/scripts/modellink/qwen25/finetune_qwen25_14b_intern_300m_ptd_tp8pp1cp2_stage3.sh/20241127_204213/
#CKPT_SAVE_DIR=/data/output/LM/scripts/modellink/qwen25/finetune_qwen25_14b_intern_300m_ptd_tp8pp1cp2_stage3.sh/20241127_204213_mg/


# Long-VITA-1M
#CKPT_LOAD_DIR=/data/output/LM/scripts/modellink/qwen25/finetune_qwen25_14b_intern_300m_ptd_tp8pp1cp8_stage4.sh/20241128_234743/
#CKPT_SAVE_DIR=/data/output/LM/scripts/modellink/qwen25/finetune_qwen25_14b_intern_300m_ptd_tp8pp1cp8_stage4.sh/20241128_234743_mg/


python long_vita_megatron/ckpt_convert_modellink_to_megatron_with_te.py \
	--ckpt_load_dir ${CKPT_LOAD_DIR} \
	--ckpt_save_dir ${CKPT_SAVE_DIR} \

