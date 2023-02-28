#!/bin/bash

# Train NOSMOG with SAGE teacher on "ogbn-arxiv"

for e in "tran" "ind"
do
    python train_student.py --exp_setting $e --teacher "SAGE" --student "MLP" --dataset "ogbn-arxiv" \
                        --num_exp 10 --max_epoch 200 --patience 50 --device 0 --dw --feat_distill --adv
done
