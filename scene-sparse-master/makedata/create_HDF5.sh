#!/bin/bash -l
source /usr/Modules/init/sh


#SBATCH -p cortex
#SBATCH --time=150:00:00
#SBATCH --mem-per-cpu 4GB
#SBATCH --ntasks-per-node=4
#SBATCH --job-name=create_HDF5
#SBATCH --output=/clusterfs/cortex/scratch/shiry/Logs/create_HDF5-%j.out
#SBATCH --error=/clusterfs/cortex/scratch/shiry/Errors/create_HDF5-%j.err

cd $HOME/scene-sparse/makedata
#module load python/anaconda

data_dir='/clusterfs/cortex/scratch/shiry/places32_grayscale_vectors/'

echo "Going to create HDF5 file"
/global/home/users/shiry/anaconda/bin/python create_HDF5.py $data_dir
echo "Finished to create HDF5 file"
