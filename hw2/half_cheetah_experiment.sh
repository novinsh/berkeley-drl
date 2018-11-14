#!/bin/bash

# obtained from half_cheetah_gridsearch.py
bs=10
lr=0.01
env="RoboschoolHalfCheetah-v1"
cmd="python train_pg_f18.py"

# TODO: make the command below shorter
eval "$cmd" ${env} -ep 150 --discount 0.95 -n 100 -e 3 -l 2 -s 32 -b ${bs} -lr ${lr} --exp_name ${experiment}_hc_b${bs}_r${lr} &
eval "$cmd" ${env} -ep 150 --discount 0.95 -n 100 -e 3 -l 2 -s 32 -b ${bs} -lr ${lr} --rtg --exp_name ${experiment}_hc_b${bs}_r${lr} &
eval "$cmd" ${env} -ep 150 --discount 0.95 -n 100 -e 3 -l 2 -s 32 -b ${bs} -lr ${lr} --nn_baseline --exp_name ${experiment}_hc_b${bs}_r${lr} &
eval "$cmd" ${env} -ep 150 --discount 0.95 -n 100 -e 3 -l 2 -s 32 -b ${bs} -lr ${lr}  -rtg --nn_baseline --exp_name ${experiment}_hc_b${bs}_r${lr} &
