#!/bin/bash
export CUDA_VISIBLE_DEVICES=2

# python inference/eval.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_3/AVR

python inference/eval.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_3/VAH

python inference/eval.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_3/VAR

python inference/eval.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_4/AVC

# nohup bash eval_gpu2.sh > /share/nlp/tuwenming/projects/HAVIB/logs/eval_gpu2_$(date +%Y%m%d%H%M%S).log 2>&1 &

