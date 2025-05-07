#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

python inference/eval.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_3/AVL

python inference/eval.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_4/AVLG

python inference/eval.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_5/AVLG

# nohup bash eval.sh > /share/nlp/tuwenming/projects/HAVIB/logs/eval_gpu0_$(date +%Y%m%d%H%M%S).log 2>&1 &
