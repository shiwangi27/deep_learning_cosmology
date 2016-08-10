#!/bin/bash -l

#SBATCH -p regular 
#SBATCH -N 1 
#SBATCH -t 05:30:00 
#SBATCH -L SCRATCH   #note: specify license need for the file systems your job needs, such as SCRATCH,project

module load python
module load theano
module load h5py
module load scikit-learn

python Classify_Denoising_Conv_AE.py --epochs 800 --samples 1000 --batches 128 --lr 5e-2 --output 'Classify_DCAE_lr_0.05_ss_1k'



