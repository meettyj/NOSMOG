#!/bin/bash

aggregated_result_file="min_cut.txt"

printf "Teacher - SAGE\n" >> $aggregated_result_file
for ds in "cora" "citeseer" "pubmed" "a-computer" "a-photo" "ogbn-arxiv"
do
    printf "%10s\n" $ds >> $aggregated_result_file
    for cur_seed in "0" "1" "2" "3" "4" "5" "6" "7" "8" "9"
    do
        printf "%6s\t" $cur_seed >> $aggregated_result_file
        python train_teacher.py --exp_setting "tran" --teacher "SAGE" --dataset $ds --num_exp 1 \
                                --max_epoch 200 --patience 50 --device 3 --seed $cur_seed --compute_min_cut >> $aggregated_result_file
    done
    printf "\n" >> $aggregated_result_file
done
printf "\n" >> $aggregated_result_file

# products
for ds in "ogbn-products"
do
    printf "%10s\n" $ds >> $aggregated_result_file
    for cur_seed in "0" "1" "2" "3" "4" "5" "6" "7" "8" "9"
    do
        printf "%6s\t" $cur_seed >> $aggregated_result_file
        python train_teacher.py --exp_setting "tran" --teacher "SAGE" --dataset $ds --num_exp 1 \
                                --max_epoch 40 --patience 10 --device 3 --seed $cur_seed --compute_min_cut >> $aggregated_result_file
    done
    printf "\n" >> $aggregated_result_file
done
printf "\n" >> $aggregated_result_file



# -------- GLNN --------
printf "GLNN\n" >> $aggregated_result_file
for ds in "cora" "citeseer" "pubmed" "a-computer" "a-photo" "ogbn-arxiv" "ogbn-products"
do
    printf "%10s\n" $ds >> $aggregated_result_file
    for cur_seed in "0" "1" "2" "3" "4" "5" "6" "7" "8" "9"
    do
        printf "%6s\t" $cur_seed >> $aggregated_result_file
        python train_student.py --exp_setting "tran" --teacher "SAGE" --student "MLP" --dataset $ds \
                                --num_exp 1 --max_epoch 200 --patience 50 --device 0 --seed $cur_seed --compute_min_cut >> $aggregated_result_file
    done
    printf "\n" >> $aggregated_result_file
done
printf "\n" >> $aggregated_result_file



# -------- NOSMOG --------
printf "NOSMOG\n" >> $aggregated_result_file
for ds in "cora" "citeseer" "pubmed" "a-computer" "a-photo" "ogbn-arxiv" "ogbn-products"
do
    printf "%10s\n" $ds >> $aggregated_result_file
    for cur_seed in "0" "1" "2" "3" "4" "5" "6" "7" "8" "9"
    do
        printf "%6s\t" $cur_seed >> $aggregated_result_file
        python train_student.py --exp_setting "tran" --teacher "SAGE" --student "MLP" --dataset $ds \
                                --num_exp 1 --max_epoch 200 --patience 50 --device 0 --seed $cur_seed \
                                --dw --feat_distill --adv --compute_min_cut >> $aggregated_result_file
    done
    printf "\n" >> $aggregated_result_file
done
printf "\n" >> $aggregated_result_file



# -------- MLP --------
printf "Teacher - MLP\n" >> $aggregated_result_file
for ds in "cora" "citeseer" "pubmed" "a-computer" "a-photo" "ogbn-arxiv" "ogbn-products"
do
    printf "%10s\n" $ds >> $aggregated_result_file
    for cur_seed in "0" "1" "2" "3" "4" "5" "6" "7" "8" "9"
    do
        printf "%6s\t" $cur_seed >> $aggregated_result_file
        python train_teacher.py --exp_setting "tran" --teacher "MLP" --dataset $ds --num_exp 1 \
                                --max_epoch 200 --patience 50 --device 3 --seed $cur_seed --compute_min_cut >> $aggregated_result_file
    done
    printf "\n" >> $aggregated_result_file
done
printf "\n" >> $aggregated_result_file
