#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

python inference/eval.py --task_path /home/tuwenming/Projects/HAVIB/data/levels/level_6/AVSQA_av

python inference/eval.py --task_path /home/tuwenming/Projects/HAVIB/data/levels/level_6/AVSQA_v

# nohup bash eval_gpu0.sh > /share/nlp/tuwenming/projects/HAVIB/logs/eval_ola_unimodal_gpu0_$(date +%Y%m%d%H%M%S).log 2>&1 &
