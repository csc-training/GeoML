#!/bin/bash
# Change to own project, if used outside of the course
#SBATCH --account=project_2002044
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=10G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:v100:1,nvme:20 #Local disk in Gb
# Reservation needed only during course
# SBATCH --reservation geoml-gpu

module load tensorflow
echo $LOCAL_SCRATCH

# Copy training and validation files from scratch to GPU local disk, unzip at the same time
tar xf trainingTilesBinary_1024.tar -C $LOCAL_SCRATCH

# Print out folders on GPU local disk
ls $LOCAL_SCRATCH

# Run the Python code, give data folder and number of classes in labels as arguments
srun python3 08_1_train.py $LOCAL_SCRATCH 2
