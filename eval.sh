#!/bin/bash
export CUDA_VISIBLE_DEVICES=3

python inference/eval.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_3/AVM

python inference/eval.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_3/AVR

python inference/eval.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_4/AVC

python inference/eval.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_4/AVQA

python inference/eval.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_5/AVLG

python inference/eval.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_5/AVQA

# nohup bash eval.sh > /share/nlp/tuwenming/projects/HAVIB/logs/eval_ola_gpu3_$(date +%Y%m%d%H%M%S).log 2>&1 &
