#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

python data_test.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_1/LAQA

python data_test.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_1/LIQA

python data_test.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_1/LVQA

python data_test.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_2/MAIC

python data_test.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_2/MVIC

python data_test.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_3/AVH

python data_test.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_3/AVL

python data_test.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_3/AVM

python data_test.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_3/AVR

python data_test.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_3/VAH

python data_test.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_3/VAR

python data_test.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_4/AVC

python data_test.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_4/AVLG

python data_test.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_4/AVQA

python data_test.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_5/AVLG

python data_test.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_5/AVQA