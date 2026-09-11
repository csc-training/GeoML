#!/bin/bash
#SBATCH --account=project_2019932 # Choose the project to be billed. Change to own project, if used outside of the course
#SBATCH --partition=gpumedium         # Which queue to use. Defines maximum time, memory, tasks, nodes and local storage for job
#SBATCH --ntasks=1                  # Number of tasks. Upper limit depends on partition.
#SBATCH --cpus-per-task=8           # How many processors work on one task. Upper limit depends on number of CPUs per GPU. In LUMI there are 7 CPU cores per one GPU. 
#SBATCH --time=00:30:00             # Maximum duration of the job. Upper limit depends on partition.
#SBATCH --gres=gpu:gh200:1          # Number of GPUs (Puhti version)

# Load Pytorch module
module load python-pytorch/2.13

mkdir -p /scratch/project_2019932/raster_data /scratch/project_2019932/students/$USER/GeoML/data/raster
test -f /scratch/project_2019932/students/$USER/GeoML/data/raster/cnn/train/labels/labels_training.tif || cp -R /scratch/project_2019932/raster_data/cnn /scratch/project_2019932/students/$USER/GeoML/data/raster/

# Run the Python code
srun python3 07A_1_train_model.py
