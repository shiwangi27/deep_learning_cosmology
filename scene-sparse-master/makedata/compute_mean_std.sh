#!/bin/bash -l
source /usr/Modules/init/sh


#SBATCH -p cortex
#SBATCH --time=150:00:00
#SBATCH --mem-per-cpu 5GB
#SBATCH --ntasks-per-node=4
#SBATCH --job-name=compute_mean_std
#SBATCH --output=/clusterfs/cortex/scratch/shiry/Logs/compute_mean_std-%j.out
#SBATCH --error=/clusterfs/cortex/scratch/shiry/Errors/compute_mean_std-%j.err

cd $HOME/scene-sparse/makedata
module load matlab/R2013a

img_dir='/clusterfs/cortex/scratch/shiry/places32_grayscale_vectors/'
result_dir='/clusterfs/cortex/scratch/shiry/'
img_sz=32

echo "Going to compute mean and std"
echo $(date)
matlab -nodesktop -nosplash -r "compute_mean_std $img_dir $result_dir $img_sz; exit"
echo $(date)
echo "Finished to compute mean and std"
