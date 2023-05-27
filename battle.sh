#!/bin/bash

MODEL_1="model/rl-model.bin"
MODEL_2="model/sl-model.bin"

for i in `seq 1 5` ; do
    gogui/bin/gogui-twogtp \
        -black 'python3 main.py --model $MODEL_1 --use-gpu true --time 3' \
        -white 'python3 main.py --model $MODEL_2 --use-gpu true --time 3' \
        -referee 'gnugo --mode gtp --chinese-rule' \
        -komi 7 -size 9 -force -auto -verbose 2>&1 | grep 'R>> final_score' -A 1
done

for i in `seq 1 5` ; do
    gogui/bin/gogui-twogtp \
        -black 'python3 main.py --model $MODEL_2 --use-gpu true --time 3' \
        -white 'python3 main.py --model $MODEL_1 --use-gpu true --time 3' \
        -referee 'gnugo --mode gtp --chinese-rule' \
        -komi 7 -size 9 -force -auto -verbose 2>&1 | grep 'R>> final_score' -A 1
done
