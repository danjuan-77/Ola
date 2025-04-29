#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

python inference/eval.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_1/LAQA

python inference/eval.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_1/LIQA

python inference/eval.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_1/LVQA

python inference/eval.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_2/MAIC

# nohup bash eval_gpu0.sh > /share/nlp/tuwenming/projects/HAVIB/logs/eval_gpu0_$(date +%Y%m%d%H%M%S).log 2>&1 &
