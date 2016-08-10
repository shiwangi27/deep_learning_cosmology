#!/bin/bash -l

#SBATCH -p regular 
#SBATCH -N 1 
#SBATCH -t 03:30:00
#SBATCH -L SCRATCH   #note: specify license need for the file systems your job needs, such as SCRATCH,project

module load python
python segment_images.py 1 0 250 