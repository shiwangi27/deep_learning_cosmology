#!/bin/bash -l

#SBATCH -p regular 
#SBATCH -N 1 
#SBATCH -t 04:00:00
#SBATCH -L SCRATCH   #note: specify license need for the file systems your job needs, such as SCRATCH,project

module load python
source activate deeplearning 
python wrapper_field_olsh_sparse.py --lambda $1 --basis $2 




