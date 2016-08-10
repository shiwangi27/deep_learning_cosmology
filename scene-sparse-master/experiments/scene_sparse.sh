#!/bin/bash -l
source /usr/Modules/init/sh


#SBATCH -p cortex
#SBATCH --time=150:00:00
#SBATCH --mem-per-cpu 2GB
#SBATCH --ntasks-per-node=4

cd $HOME/scene-sparse/experiments
module load matlab

#matlab -nodesktop -r "svm_test($lam,$overcomp,$cmpr,$start,$stop)"
matlab -nodesktop -r "scene_sparse"
