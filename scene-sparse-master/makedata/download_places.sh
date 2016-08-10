#!/bin/bash -l
source /usr/Modules/init/sh


#SBATCH -p cortex
#SBATCH --time=150:00:00
#SBATCH --mem-per-cpu 2GB
#SBATCH --ntasks-per-node=4
#SBATCH --error=download_imagenet.e
#SBATCH --output=download_imagenet.o

cd /clusterfs/cortex/scratch/shiry/
echo "Going to download places"
#wget http://places.csail.mit.edu/download_places/imagesPlaces205_resize.tar 
echo "Finished downloading places"
echo "Going to download training and validation splits"
wget http://places.csail.mit.edu/download_places/trainval_places205.zip
echo "Finished downloading training and validation splits"
