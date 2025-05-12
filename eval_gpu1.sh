#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

python data_test.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_3/AVL

# nohup bash eval_gpu1.sh > /share/nlp/tuwenming/projects/HAVIB/logs/eval_ola_unimodal_gpu1_$(date +%Y%m%d%H%M%S).log 2>&1 &
