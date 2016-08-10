#!/bin/bash -l
source /usr/Modules/init/sh


#SBATCH -p cortex
#SBATCH --time=150:00:00
#SBATCH --mem-per-cpu 2GB
#SBATCH --ntasks-per-node=4
#SBATCH --job-name=tinify_images
#SBATCH --output=/clusterfs/cortex/scratch/shiry/Logs/tinify_images-%j.out
#SBATCH --error=/clusterfs/cortex/scratch/shiry/Errors/tinify_images-%j.err

cd $HOME/scene-sparse/makedata
module load matlab/R2013a

img_dir='/clusterfs/cortex/scratch/shiry/places256/'
result_dir='/clusterfs/cortex/scratch/shiry/places32_grayscale_vectors/'
img_sz=32

echo "Going to tinify images"
matlab -nodesktop -nosplash -r "tinify_images_recursive $img_dir $result_dir $img_sz; exit"
echo "Finished to tinify images"
