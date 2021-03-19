#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 18 12:11:28 2019

@author: ziqi
"""
from utils import check_mkdir
import os

path = 'jobs/conservative_'

conservative = ['False']
for item in conservative:
    path_item = path + item + '/'
    check_mkdir(path_item)
    for exp in range (0, 10):
        with open(path_item + 'exp_%d.sh' %(exp), 'w') as f:
            f.write("""#!/bin/sh
#SBATCH --partition=general --qos=short
#SBATCH --time=2:00:00
#SBATCH --gres=gpu
#SBATCH --mem=50000
#SBATCH --workdir=/tudelft.net/staff-bulk/ewi/insy/VisionLab/ziqiwang/attack/cifar
#SBATCH --job-name=attack""" + '\n'
"""#SBATCH --mail-type=END

module use /opt/insy/modulefiles
module load cuda/10.1 cudnn/10.1-7.6.0.64

echo "Starting at $(date)"
srun python cifar.py --conservative=""" + item + ' --exp=' + str(exp) +'\n' +
"""echo "Finished at $(date)"
"""
)
        
            
for item in conservative:
    path_item = path + item + '/'        
    job_files = os.listdir(path_item)
    with open(path_item + 'jobfile_all.sh', 'w') as f:
        for job in job_files:
            f.write('sbatch %s\n' % job)
