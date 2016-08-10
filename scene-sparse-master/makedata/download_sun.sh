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
wget http://vision.cs.princeton.edu/projects/2010/SUN/hierarchy_three_levels.zip
echo "Finished downloading sun"
