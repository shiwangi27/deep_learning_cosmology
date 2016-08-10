#!/bin/bash -l
source /usr/Modules/init/sh


#SBATCH -p cortex
#SBATCH --time=150:00:00
#SBATCH --mem-per-cpu 5GB
#SBATCH --ntasks-per-node=4
#SBATCH --job-name=whiten_images
#SBATCH --output=/clusterfs/cortex/scratch/shiry/Logs/whiten_images-%j.out
#SBATCH --error=/clusterfs/cortex/scratch/shiry/Errors/whiten_images-%j.err

cd $HOME/scene-sparse/makedata
module load matlab/R2013a

img_dir='/clusterfs/cortex/scratch/shiry/places32_grayscale_vectors/'
result_dir='/clusterfs/cortex/scratch/shiry/places32_whitened_grayscale_vectors/'
img_sz=32
mean_var_file='/clusterfs/cortex/scratch/shiry/data_mean_variance.mat'

echo "Going to whiten images"
matlab -nodesktop -nosplash -r "whiten_images $img_dir $result_dir $img_sz $mean_var_file; exit"
echo "Finished to whiten images"
