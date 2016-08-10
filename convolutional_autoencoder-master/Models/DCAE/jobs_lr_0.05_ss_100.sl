#!/bin/bash -l

#SBATCH -p regular 
#SBATCH -N 1 
#SBATCH -t 01:30:00
#SBATCH -L SCRATCH   #note: specify license need for the file systems your job needs, such as SCRATCH,project

module load python
module load theano
module load h5py
module load scikit-learn

python Denoising_Conv_AE.py --epochs 700 --samples 100 --batches 80 --lr 5e-2 --output 'New_lr_0.05_ss_100' 




