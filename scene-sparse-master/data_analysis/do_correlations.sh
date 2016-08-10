#!/bin/bash -l
source /usr/Modules/init/sh


#SBATCH -p cortex
#SBATCH --time=150:00:00
#SBATCH --mem-per-cpu 2GB
#SBATCH --ntasks-per-node=4
#SBATCH --error=man_made_pixelC.e
#SBATCH --output=man_made_pixelC.o

cd $HOME/scene-sparse/data_analysis
module load matlab/R2013a
img_dir='/clusterfs/cortex/scratch/shiry/image-net-tiny/man_made/'
num_images=12575
img_size=32
image_format='.JPEG'
results_file='/clusterfs/cortex/scratch/shiry/results/data_correlation/man_made'
echo "Going to start Matlab job"
matlab -nodesktop -r "get_data_correlation $img_dir $num_images $img_size $image_format $results_file; exit"
echo "Finished Matlab job"
