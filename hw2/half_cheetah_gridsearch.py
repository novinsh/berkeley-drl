#!/usr/bin/python
import itertools
import os

batch_size = [10000, 30000, 50000]
learning_rate = [0.005, 0.01, 0.02]

for bs, lr in list(itertools.product(batch_size, learning_rate)):
    cmd = "python train_pg_f18.py RoboschoolHalfCheetah-v1 -ep 150 --discount 0.9 -n 100 -e 3 -l 2 -s 32 -b {0} -lr {1} -rtg --nn_baseline --exp_name problem8_hc_b{0}_r{1}".format(bs, lr)
    os.system(cmd)

