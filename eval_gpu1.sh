#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

python inference/eval.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_2/MVIC

python inference/eval.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_3/AVH

python inference/eval.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_3/AVL

python inference/eval.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_3/AVM

# nohup bash eval_gpu1.sh > /share/nlp/tuwenming/projects/HAVIB/logs/eval_gpu1_$(date +%Y%m%d%H%M%S).log 2>&1 &
