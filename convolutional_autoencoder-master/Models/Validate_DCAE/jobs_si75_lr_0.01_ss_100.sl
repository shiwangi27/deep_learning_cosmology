#!/bin/bash -l

#SBATCH -p regular 
#SBATCH -N 1 
#SBATCH -t 01:10:00 
#SBATCH -L SCRATCH   #note: specify license need for the file systems your job needs, such as SCRATCH,project

module load python
module load deeplearning

python Denoising_CAE_valid_si75.py --epochs 800 --samples 100 --batches 80 --lr 1e-2 --output 'Valid_si75_ss_100_lr_1e-2'



