#!/bin/bash -l

#SBATCH -p regular 
#SBATCH -N 1 
#SBATCH -t 7:00:00
#SBATCH -L SCRATCH   #note: specify license need for the file systems your job needs, such as SCRATCH,project

module load python
module load deeplearning 
python wrapper_field_olsh_sparse.py --epoch 10000 --lambda 10 --basis 200 