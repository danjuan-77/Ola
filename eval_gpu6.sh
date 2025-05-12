#!/bin/bash
export CUDA_VISIBLE_DEVICES=2

python data_test.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_5/AVLG

# nohup bash eval_gpu6.sh > /share/nlp/tuwenming/projects/HAVIB/logs/eval_ola_unimodal_gpu6_$(date +%Y%m%d%H%M%S).log 2>&1 &

