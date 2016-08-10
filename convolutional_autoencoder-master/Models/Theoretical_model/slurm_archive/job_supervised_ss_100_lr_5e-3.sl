#!/bin/bash -l

#SBATCH -p regular 
#SBATCH -N 1 
#SBATCH -t 01:15:00 
#SBATCH -L SCRATCH   #note: specify license need for the file systems your job needs, such as SCRATCH,project

module load python
module load deeplearning

python supervised_cnn.py --epochs 500 --samples 100 --batches 80 --lr 5e-3 --output 'Supervised_ss_100_lr_5e-3'





