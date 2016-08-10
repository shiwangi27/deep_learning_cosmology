#!/bin/bash -l

#SBATCH -p regular 
#SBATCH -N 1 
#SBATCH -t 08:10:00 
#SBATCH -L SCRATCH   #note: specify license need for the file systems your job needs, such as SCRATCH,project

module load python
module load deeplearning

python supervised_cnn.py --epochs 500 --samples 1000 --batches 128 --lr 3e-4 --output 'Supervised_ss_1000_lr_3e-4'



