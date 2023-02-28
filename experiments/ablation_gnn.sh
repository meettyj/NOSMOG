#!/bin/bash

aggregated_result_file="ablation_gnn.txt"

printf "Teacher\n" >> $aggregated_result_file
for e in "tran" "ind"
do
    printf "%6s\n" $e >> $aggregated_result_file
    for t in "GCN" "GAT" "MLP" "APPNP"  # "SAGE"
    do
        printf "%6s\n" $t >> $aggregated_result_file
        for ds in "cora" "citeseer" "pubmed" "a-computer" "a-photo"
        do
            printf "%10s\t" $ds >> $aggregated_result_file
            python train_teacher.py --exp_setting $e --teacher $t --dataset $ds --num_exp 10 \
                                    --max_epoch 200 --patience 50 --device 3 >> $aggregated_result_file
        done
        printf "\n" >> $aggregated_result_file
    done
    printf "\n" >> $aggregated_result_file
done


printf "GLNN\n" >> $aggregated_result_file
for e in "tran" "ind"
do
    printf "%6s\n" $e >> $aggregated_result_file
    for t in "GCN" "GAT" "SAGE" "APPNP"
    do
        printf "%6s\n" $t >> $aggregated_result_file
        for ds in "cora" "citeseer" "pubmed" "a-computer" "a-photo"
        do
            printf "%10s\t" $ds >> $aggregated_result_file
            python train_student.py --exp_setting $e --teacher $t --dataset $ds --num_exp 10 \
                                    --max_epoch 200 --patience 50 --device 3 >> $aggregated_result_file
        done
        printf "\n" >> $aggregated_result_file
    done
    printf "\n" >> $aggregated_result_file
done


printf "SMOG\n" >> $aggregated_result_file
for e in "tran" "ind"
do
    printf "%6s\n" $e >> $aggregated_result_file
    for t in "GCN" "GAT" "SAGE" "APPNP"
    do
        printf "%6s\n" $t >> $aggregated_result_file
        for ds in "cora" "citeseer" "pubmed" "a-computer" "a-photo"
        do
            printf "%10s\t" $ds >> $aggregated_result_file
            python train_student.py --exp_setting $e --teacher $t --dataset $ds --num_exp 10 \
                                    --max_epoch 200 --patience 50 --device 3 \
                                    --dw --feat_distill --adv >> $aggregated_result_file
        done
        printf "\n" >> $aggregated_result_file
    done
    printf "\n" >> $aggregated_result_file
done
