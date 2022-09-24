#!/bin/bash
#
#SBATCH --exclude=p3-r52-a.g42cloud.net
#SBATCH --exclude=p4-r68-a.g42cloud.net
#SBATCH --exclude=p4-r67-b.g42cloud.net
#SBATCH --job-name=VL-LTR
#SBATCH --output=./slurm_out_multig/%J.out
#SBATCH --partition=multigpu
#SBATCH --account=mbzuai
#SBATCH --time=6-00:00:00
#SBATCH --ntasks=1
#SBATCH --gpus=8
#SBATCH --cpus-per-task=64

# #echo "Module = $(module avail)"

echo "Python Interpreter = $(which python)"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 #,8,9,10,11,12,13,14,15

NCCL_LL_THRESHOLD=0

echo "which nvcc = " $(which nvcc)

echo "nvcc --version = " $(nvcc --version)

#echo "NCCL_DEBUG = " $NCCL_DEBUG

echo "CUDA_VISIBLE_DEVICES = " $CUDA_VISIBLE_DEVICES

# echo "NCCL_LL_THRESHOLD = " $NCCL_LL_THRESHOLD


cd /nfs/users/ext_amaya.dharmasiri/repos/VL-LTR

set -x

export NCCL_LL_THRESHOLD=0
export MKL_SERVICE_FORCE_INTEL=1

CONFIG=$1
GPUS=$2
CPUS=$[GPUS*4]
PORT=${PORT:-8666}
if [ $GPUS -lt 8 ]; then
    GPUS_PER_NODE=${GPUS_PER_NODE:-$GPUS}
else
    GPUS_PER_NODE=${GPUS_PER_NODE:-8}
fi

CONFIG_NAME=${CONFIG##*/}
CONFIG_NAME=${CONFIG_NAME%.*}

OUTPUT_DIR="./checkpoints/${CONFIG_NAME}"
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p ${OUTPUT_DIR}
fi

/nfs/users/ext_amaya.dharmasiri/miniconda3/envs/VL-LTR/bin/python \
    -m torch.distributed.launch --nproc_per_node=$GPUS main.py \
    --port=$PORT \
    --num_workers 4 \
    --config $CONFIG ${@:3} \
    --resume "./checkpoints/${CONFIG_NAME}/checkpoint.pth" \
    2>&1 | tee -a ${OUTPUT_DIR}/train.log

