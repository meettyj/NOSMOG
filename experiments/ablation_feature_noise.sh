#!/bin/bash

aggregated_result_file="ablation_feature_noise.txt"

# Use origin teacher
#printf "Teacher - SAGE\n" >> $aggregated_result_file
#for n in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1
#do
#    printf "%3s\n" $n >> $aggregated_result_file
#    for ds in "cora" "citeseer" "pubmed" "a-computer" "a-photo"
#    do
#        printf "%10s\t" $ds >> $aggregated_result_file
#        python train_teacher.py --exp_setting "ind" --teacher "SAGE" --dataset $ds --feature_noise $n \
#                                --num_exp 10 --max_epoch 200 --patience 50 --device 0 >> $aggregated_result_file
#    done
#    printf "\n" >> $aggregated_result_file
#done

printf "Student - GLNN\n" >> $aggregated_result_file
for n in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1
do
    printf "%3s\n" $n >> $aggregated_result_file
    for ds in "cora" "citeseer" "pubmed" "a-computer" "a-photo"
    do
        printf "%10s\t" $ds >> $aggregated_result_file
        python train_student.py --exp_setting "ind" --teacher "SAGE" --dataset $ds --feature_noise $n \
                                --num_exp 10 --max_epoch 200 --patience 50 --device 2 >> $aggregated_result_file
    done
    printf "\n" >> $aggregated_result_file
done

printf "Student - NOSMOG\n" >> $aggregated_result_file
for n in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1
do
    printf "%3s\n" $n >> $aggregated_result_file
    for ds in "cora" "citeseer" "pubmed" "a-computer" "a-photo"
    do
        printf "%10s\t" $ds >> $aggregated_result_file
        python train_student.py --exp_setting "ind" --teacher "SAGE" --dataset $ds --feature_noise $n \
                                --num_exp 10 --max_epoch 200 --patience 50 --device 2 \
                                --dw --feat_distill --adv >> $aggregated_result_file
    done
    printf "\n" >> $aggregated_result_file
done


#printf "Teacher - MLP\n" >> $aggregated_result_file
#for n in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1
#do
#    printf "%3s\n" $n >> $aggregated_result_file
#    for ds in "cora" "citeseer" "pubmed" "a-computer" "a-photo"
#    do
#        printf "%10s\t" $ds >> $aggregated_result_file
#        python train_teacher.py --exp_setting "ind" --teacher "MLP" --dataset $ds --feature_noise $n \
#                                --num_exp 10 --max_epoch 200 --patience 50 --device 0 >> $aggregated_result_file
#    done
#    printf "\n" >> $aggregated_result_file
#done
