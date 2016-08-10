#!/bin/bash -l

#SBATCH -p regular 
#SBATCH -N 1 
#SBATCH -t 3:29:99
#SBATCH -L SCRATCH   #note: specify license need for the file systems your job needs, such as SCRATCH,project

module load python 
module load h5py 
python wrapper_field_olsh_sparse.py 

salloc -N 1 -t 2:00:00 