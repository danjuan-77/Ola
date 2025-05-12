#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

python data_test.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_3/AVH

python data_test.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_3/AVL

python data_test.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_3/VAH

python data_test.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_4/AVC

python data_test.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_4/AVLG

python data_test.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_4/AVQA

python data_test.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_5/AVLG

python data_test.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_5/AVQA

# nohup bash eval.sh > /share/nlp/tuwenming/projects/HAVIB/logs/eval_ola_unimodal_gpu0_$(date +%Y%m%d%H%M%S).log 2>&1 &
