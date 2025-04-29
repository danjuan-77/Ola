#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

python inference/eval.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_1/LAQA

python inference/eval.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_1/LIQA

python inference/eval.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_1/LVQA

python inference/eval.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_2/MAIC

python inference/eval.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_2/MVIC

python inference/eval.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_3/AVH

python inference/eval.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_3/AVL

python inference/eval.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_3/AVM

python inference/eval.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_3/AVR

python inference/eval.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_3/VAH

python inference/eval.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_3/VAR

python inference/eval.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_4/AVC

python inference/eval.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_4/AVLG

python inference/eval.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_4/AVQA

python inference/eval.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_5/AVLG

python inference/eval.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_5/AVQA

# nohup bash eval_gpu1.sh > /share/nlp/tuwenming/projects/HAVIB/logs/eval_gpu1_$(date +%Y%m%d%H%M%S).log 2>&1 &
