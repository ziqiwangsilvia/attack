#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 08:59:44 2021

@author: ziqi
"""

from utils import check_mkdir
import os
import numpy as np

path = 'jobs/tune_hps/'
check_mkdir(path)

conservative = 'center'
conservative_a = [round(i, 2) for i in np.arange(0.05, 1.1, 0.05)]
lrs = [1e-5, 1e-4, 5e-4, 1e-3, 5e-3]
tbs = [128, 256, 512]
wds = [5e-4, 5e-6, 5e-8]

for a in conservative_a:
    for lr in lrs:
        for tb in tbs:
            for wd in wds:
                for exp in range (0, 10):
                    with open(path + 'exp_a_%.2f_lr_%.5f_tb_%d_wd_%.8f.sh' %(a, lr, tb, wd), 'w') as f:
                        f.write("""#!/bin/sh
#SBATCH --partition=general --qos=short
#SBATCH --time=2:00:00
#SBATCH --gres=gpu
#SBATCH --mem=7000
#SBATCH --chdir=/tudelft.net/staff-bulk/ewi/insy/VisionLab/ziqiwang/attack/cifar
#SBATCH --job-name=tune hps""" + '\n'
"""#SBATCH --mail-type=END

module use /opt/insy/modulefiles
module load cuda/10.1 cudnn/10.1-7.6.0.64

echo "Starting at $(date)"
srun python cifar.py --tune_hps=True --conservative=center --conservative_a=""" + str(a) + ' --lr=' + str(lr)  \
    + ' --train_batch_size=' + str(tb) +' --weight_decay=' + str(wd) +'\n' +
"""echo "Finished at $(date)"
"""
)