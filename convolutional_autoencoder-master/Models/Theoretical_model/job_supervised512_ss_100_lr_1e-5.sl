#!/bin/bash -l

#SBATCH -p regular 
#SBATCH -N 1 
#SBATCH -t 01:00:00 
#SBATCH -L SCRATCH   #note: specify license need for the file systems your job needs, such as SCRATCH,project

module load python
module load deeplearning

python supervised_cnn_2.py --epochs 500 --samples 100 --batches 80 --lr 1e-5 --output 'Supervised512_ss_100_lr_1e-5'






