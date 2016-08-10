#!/bin/bash -l
source /usr/Modules/init/sh


#SBATCH -p cortex
#SBATCH --time=150:00:00
#SBATCH --mem-per-cpu 2GB
#SBATCH --ntasks-per-node=4
#SBATCH --error=download_imagenet.e
#SBATCH --output=download_imagenet.o

cd /clusterfs/cortex/scratch/shiry/
echo "Going to download sun"
tar xf imagesPlaces205_resize.tar
echo "Finished downloading sun"
