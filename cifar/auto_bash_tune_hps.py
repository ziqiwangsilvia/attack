#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 08:59:44 2021

@author: ziqi
"""

from utils import check_mkdir
import os
import numpy as np

path = 'jobs/tune_hps_cifar100_resnet50_pretrain_marco_longjob/'
check_mkdir(path)
lrs = [1e-4, 5e-4, 5e-5]
tbs = [256, 512, 1024, 2048]
wds = [5e-4, 5e-6]

for lr in lrs:
    for tb in tbs:
        for wd in wds:
            for exp in range (0, 10):
                with open(path + 'exp_lr_%.5f_tb_%d_wd_%.8f.sh' %(lr, tb, wd), 'w') as f:
                    f.write("""#!/bin/sh
#SBATCH --partition=general --qos=long
#SBATCH --time=24:00:00
#SBATCH --gres=gpu
#SBATCH --mem=7000
#SBATCH --chdir=/tudelft.net/staff-bulk/ewi/insy/VisionLab/ziqiwang/attack/cifar
#SBATCH --job-name=tune_hps""" + '\n'
"""#SBATCH --mail-type=END

module use /opt/insy/modulefiles
module load cuda/10.1 cudnn/10.1-7.6.0.64

echo "Starting at $(date)"
srun python cifar.py --dataset=cifar100 --tune_hps=True --network=resnet50 --conservative=marco""" + ' --lr=' + str(lr)  \
    + ' --train_batch_size=' + str(tb) +' --weight_decay=' + str(wd) +'\n' +
"""echo "Finished at $(date)"
"""
)
                            
job_files = os.listdir(path)
with open(path + 'jobfile_all.sh', 'w') as f:
    for job in job_files:
        f.write('sbatch %s\n' % job)
