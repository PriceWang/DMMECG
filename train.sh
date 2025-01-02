###
# Author: Guoxin Wang
# Date: 2024-11-29 12:58:55
# LastEditors: Guoxin Wang
# LastEditTime: 2024-12-20 11:42:21
# FilePath: /workspace_3090/DNSECG/train.sh
# Description:
#
# Copyright (c) 2024 by Guoxin Wang, All Rights Reserved.
###

OMP_NUM_THREADS=20 torchrun --nnodes=1 --nproc-per-node=1 training.py \
    --batch_size 256 \
    --lr 1e-3 \
    --epochs 20 \
    --tau 0.05 \
    --lam 0 \
    --warmup_epochs 0 \
    --output_dir ./ckpts/lam0 \
    --log_dir ./ckpts/lam0

OMP_NUM_THREADS=20 torchrun --nnodes=1 --nproc-per-node=1 training.py \
    --batch_size 256 \
    --lr 1e-3 \
    --epochs 20 \
    --tau 0.05 \
    --lam 0.1 \
    --warmup_epochs 0 \
    --output_dir ./ckpts/lam0.1 \
    --log_dir ./ckpts/lam0.1

OMP_NUM_THREADS=20 torchrun --nnodes=1 --nproc-per-node=1 training.py \
    --batch_size 256 \
    --lr 1e-3 \
    --epochs 20 \
    --tau 0.05 \
    --lam 0.2 \
    --warmup_epochs 0 \
    --output_dir ./ckpts/lam0.2 \
    --log_dir ./ckpts/lam0.2

OMP_NUM_THREADS=20 torchrun --nnodes=1 --nproc-per-node=1 training.py \
    --batch_size 256 \
    --lr 1e-3 \
    --epochs 20 \
    --tau 0.05 \
    --lam 0.3 \
    --warmup_epochs 0 \
    --output_dir ./ckpts/lam0.3 \
    --log_dir ./ckpts/lam0.3

OMP_NUM_THREADS=20 torchrun --nnodes=1 --nproc-per-node=1 training.py \
    --batch_size 256 \
    --lr 1e-3 \
    --epochs 20 \
    --tau 0.05 \
    --lam 0.4 \
    --warmup_epochs 0 \
    --output_dir ./ckpts/lam0.4 \
    --log_dir ./ckpts/lam0.4

OMP_NUM_THREADS=20 torchrun --nnodes=1 --nproc-per-node=1 training.py \
    --batch_size 256 \
    --lr 1e-3 \
    --epochs 20 \
    --tau 0.05 \
    --lam 0.5 \
    --warmup_epochs 0 \
    --output_dir ./ckpts/lam0.5 \
    --log_dir ./ckpts/lam0.5

OMP_NUM_THREADS=20 torchrun --nnodes=1 --nproc-per-node=1 training.py \
    --batch_size 256 \
    --lr 1e-3 \
    --epochs 20 \
    --tau 0.05 \
    --lam 0.6 \
    --warmup_epochs 0 \
    --output_dir ./ckpts/lam0.6 \
    --log_dir ./ckpts/lam0.6

OMP_NUM_THREADS=20 torchrun --nnodes=1 --nproc-per-node=1 training.py \
    --batch_size 256 \
    --lr 1e-3 \
    --epochs 20 \
    --tau 0.05 \
    --lam 0.7 \
    --warmup_epochs 0 \
    --output_dir ./ckpts/lam0.7 \
    --log_dir ./ckpts/lam0.7

OMP_NUM_THREADS=20 torchrun --nnodes=1 --nproc-per-node=1 training.py \
    --batch_size 256 \
    --lr 1e-3 \
    --epochs 20 \
    --tau 0.05 \
    --lam 0.8 \
    --warmup_epochs 0 \
    --output_dir ./ckpts/lam0.8 \
    --log_dir ./ckpts/lam0.8

OMP_NUM_THREADS=20 torchrun --nnodes=1 --nproc-per-node=1 training.py \
    --batch_size 256 \
    --lr 1e-3 \
    --epochs 20 \
    --tau 0.05 \
    --lam 0.9 \
    --warmup_epochs 0 \
    --output_dir ./ckpts/lam0.9 \
    --log_dir ./ckpts/lam0.9

OMP_NUM_THREADS=20 torchrun --nnodes=1 --nproc-per-node=1 training.py \
    --batch_size 256 \
    --lr 1e-3 \
    --epochs 20 \
    --tau 0.05 \
    --lam 1 \
    --warmup_epochs 0 \
    --output_dir ./ckpts/lam1 \
    --log_dir ./ckpts/lam1
