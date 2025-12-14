#!/bin/bash
# Training script for ALDEN model
# Usage: bash run.sh

export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=4

torchrun \
    --standalone \
    --nnodes=1 \
    --master_port 12345 \
    --nproc_per_node=2 \
    train.py \
    --config_path Config/config.yaml
