#!/bin/bash
#SBATCH --account=project_2000599 # Choose the project to be billed. Change to own project, if used outside of the course
#SBATCH --partition=gputest         # Which queue to use. Defines maximum time, memory, tasks, nodes and local storage for job
#SBATCH --ntasks=1                  # Number of tasks. Upper limit depends on partition.
#SBATCH --time=00:10:00             # Maximum duration of the job. Upper limit depends on partition.
#SBATCH --gres=gpu:gh200:1          # Number of GPUs (Puhti version)
#SBATCH --cpus-per-task=10           # How many processors work on one task. Upper limit depends on number of CPUs per GPU. In LUMI there are 7 CPU cores per one GPU. 

# Load Pytorch module
module load python-pytorch/2.10

# Add paths for extra packages needed: terratorch and sahi
export PYTHONUSERBASE=/projappl/project_2019932/geoml
export PATH=/projappl/project_2019932/geoml/bin:$PATH

# Set cache directory for Huggingface
export HF_HOME=/projappl/project_2019932/huggingface

# Run the Python code
srun python3 07B_2_predict.py
