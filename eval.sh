###
# Author: Guoxin Wang
# Date: 2024-12-05 15:22:04
# LastEditors: Guoxin Wang
# LastEditTime: 2025-01-02 15:13:39
# FilePath: /DNSECG/eval.sh
# Description:
#
# Copyright (c) 2024 by Guoxin Wang, All Rights Reserved.
###

python training.py \
    --test_path ./datasets/mitdb/af_beat_4_test.pth \
    --output_dir ./ckpts/lam0.4 \
    --eval # ./datasets/incartdb/af_beat_4_train.pth \
# ./datasets/incartdb/af_beat_4_valid.pth \
# ./datasets/incartdb/af_beat_4_test.pth \
