#!/bin/bash -l
source /usr/Modules/init/sh


#SBATCH -p cortex
#SBATCH --time=150:00:00
#SBATCH --mem-per-cpu 4GB
#SBATCH --ntasks-per-node=4
#SBATCH --job-name=doorness_lists

cd $HOME/scene-sparse/makedata
#module load python/anaconda

echo "Going to create doorness lists"
/global/home/users/shiry/anaconda/bin/python doorness_lists.py
echo "Finished to create doorness lists"
