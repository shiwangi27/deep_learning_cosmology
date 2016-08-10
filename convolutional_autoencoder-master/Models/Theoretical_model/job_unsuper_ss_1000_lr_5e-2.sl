#!/bin/bash -l

#SBATCH -p regular 
#SBATCH -N 1 
#SBATCH -t 05:15:00 
#SBATCH -L SCRATCH   #note: specify license need for the file systems your job needs, such as SCRATCH,project

module load python
module load deeplearning

python Unsupervised_DCAE.py --epochs 800 --samples 1000 --batches 128 --lr 5e-2 --output 'Unsupervised_ss_1000_lr_5e-2'



