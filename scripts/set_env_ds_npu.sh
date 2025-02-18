#set -e
#set -x

######################################################################
source /usr/local/Ascend/driver/bin/setenv.bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

export HCCL_CONNECT_TIMEOUT=7200
export HCCL_EXEC_TIMEOUT=7200

export COMBINED_ENABLE=1
export MULTI_STREAM_MEMORY_REUSE=1

export HCCL_RDMA_TC=160
export HCCL_RDMA_SL=5
export HCCL_INTRA_PCIE_ENABLE=0
export HCCL_INTRA_ROCE_ENABLE=1
export HCCL_RDMA_TIMEOUT=20
#export HCCL_ALGO="level0:NA;level1:ring"

export INF_NAN_MODE_ENABLE=1

export DISTRIBUTED_BACKEND="hccl"


export ASCEND_LAUNCH_BLOCKING=0
#将Host日志输出到串口,0-关闭/1-开启
export ASCEND_SLOG_PRINT_TO_STDOUT=0
#设置默认日志级别,0-debug/1-info/2-warning/3-error
export ASCEND_GLOBAL_LOG_LEVEL=3
#设置Event日志开启标志,0-关闭/1-开启
export ASCEND_GLOBAL_EVENT_ENABLE=0
#设置是否开启taskque,0-关闭/1-开启
export TASK_QUEUE_ENABLE=1
#设置是否开启PTCopy,0-关闭/1-开启
export PTCOPY_ENABLE=1
#设置是否开启2个非连续combined标志,0-关闭/1-开启
export COMBINED_ENABLE=1
#设置特殊场景是否需要重新编译,不需要修改
export DYNAMIC_OP="ADD#MUL"
#HCCL白名单开关,1-关闭/0-开启
export HCCL_WHITELIST_DISABLE=1
#设置HCCL超时时间
export HCCL_CONNECT_TIMEOUT=7200
export ASCEND_WAIT_NO_PREP_PATH=0
#export HCCL_OP_BASE_FFTS_MODE_ENABLE=TRUE


export HCCL_WHITELIST_DISABLE=1

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

