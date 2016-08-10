#!/bin/bash -l

#SBATCH -p regular 
#SBATCH -N 1 
#SBATCH -t 01:15:00 
#SBATCH -L SCRATCH   #note: specify license need for the file systems your job needs, such as SCRATCH,project

module load python
module load deeplearning

python Unsupervised_DCAE.py --epochs 800 --samples 100 --batches 80 --lr 5e-1 --output 'Unsupervised_ss_100_lr_5e-2'



