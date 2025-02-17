#set -e
#set -x

######################################################################
#export NCCL_NET=IB

#export NCCL_SOCKET_IFNAME="bond1"
#export GLOO_SOCKET_IFNAME="bond1"
#export NCCL_DEBUG=INFO
#export NCCL_IB_QPS_PER_CONNECTION=2

#export GLOO_SOCKET_IFNAME=eth0
#export NCCL_DEBUG=INFO
#export NCCL_IB_QPS_PER_CONNECTION=2

#export NCCL_IB_DISABLE=1

#export GPU_NUM_PER_NODE=16
#export NODE_NUM=1
#export INDEX=0
#export MASTER_ADDR=127.0.0.1
#export WORLD_SIZE=16

export CUDA_DEVICE_MAX_CONNECTIONS=1

######################################################################
#export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
#export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
#export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:1024"
#export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:10240"

#export ASCEND_LAUNCH_BLOCKING=1
#export TASK_QUEUE_ENABLE=0

######################################################################
# MindSpeed
#export MEMORY_FRAGMENTATION=1


pip3 install --no-index --find-links=/data/software/ -r requirements.txt
#pip3 install -r requirements.txt

export NVTE_FLASH_ATTN=1
export NVTE_FUSED_ATTN=0 

