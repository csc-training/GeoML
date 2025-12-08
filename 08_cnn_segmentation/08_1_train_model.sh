#!/bin/bash
#SBATCH --account=project_462001167 # Choose the project to be billed. Change to own project, if used outside of the course
#SBATCH --partition=small-g           # Which queue to use. Defines maximum time, memory, tasks, nodes and local storage for job
#SBATCH --ntasks=1                # Number of tasks. Upper limit depends on partition.
#SBATCH --cpus-per-task=7       # How many processors work on one task. Upper limit depends on number of CPUs per GPU. In Lumi there are 7 CPU cores per one GPU. 
#SBATCH --mem=60G                 # Reserved memory
#SBATCH --time=00:30:00           # Maximum duration of the job. Upper limit depends on partition.
#SBATCH --gpus=1         # Number of GPUs (LUMI version)
# SBATCH --gres=gpu:v100:1         # Number of GPUs (Puhti version)

# Use course Python installation, inc TorchGeo
#export PATH="/projappl/project_462001167/geoml_tykky2/bin:$PATH"
module use /appl/local/csc/modulefiles/
module load pytorch

# Run the Python code, give data folder and number of classes in labels as arguments
srun python3 08_1_train_model.py
