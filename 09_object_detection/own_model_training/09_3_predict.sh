#!/bin/bash
#SBATCH --account=project_462001167 # Choose the project to be billed. Change to own project, if used outside of the course
#SBATCH --partition=small-g         # Which queue to use. Defines maximum time, memory, tasks, nodes and local storage for job                
#SBATCH --ntasks=1                  # Number of tasks. Upper limit depends on partition.
#SBATCH --cpus-per-task=8           # How many processors work on one task. Upper limit depends on number of CPUs per node.
#SBATCH --gpus=1                    # Number of GPUs
#SBATCH --time=01:00:00             # Maximum duration of the job. Upper limit depends on partition.
#SBATCH --mem=32G                   # Reserved memory

# Load the CSC module tree into use
module use /appl/local/csc/modulefiles/

# Load Pytorch module
module load pytorch/2.7

# Activate virtual environment containing special packages needed for GeoAI, inc TorchGeo
source /projappl/project_462001167/students/$USER/geoml/bin/activate

# Run the Python code and give path to the exercise directory
srun python3 09_3_predict.py /scratch/project_462001167/students/$USER/GeoML/09_object_detection/own_model_training/