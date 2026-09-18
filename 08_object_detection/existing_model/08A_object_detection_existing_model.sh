#!/bin/bash
#SBATCH --account=project_2019932   # Choose the project to be billed. Change to own project, if used outside of the course
#SBATCH --partition=gpumedium       # Which queue to use. Defines maximum time, memory, tasks, nodes and local storage for job     
#SBATCH --ntasks=1                  # Number of tasks. Upper limit depends on partition.
#SBATCH --cpus-per-task=8           # How many processors work on one task. Upper limit depends on number of CPUs per GPU. In LUMI there are 7 CPU cores per one GPU.
#SBATCH --time=00:10:00             # Maximum duration of the job. Upper limit depends on partition.
#SBATCH --gres=gpu:gh200:1           # Number of GPUs (Roihu version)

# Load Pytorch module
module load python-pytorch/2.13

# Copy the input data from Roihu
# If you do this exercise outside of CSC course, remove next to lines and run 08_0_download_sentinel2_data.ipynb.
test -f /scratch/project_2019932/students/$USER/GeoML/08_object_detection/sentinel2-object-detection/T35VLG_20220626T095039_TCI.tif || cp -R /scratch/project_2019932/raster_data/sentinel2-object-detection /scratch/project_2019932/students/$USER/GeoML/08_object_detection/

# Run the Python code and give path to exercise folder
srun python3 08A_object_detection_existing_model.py
