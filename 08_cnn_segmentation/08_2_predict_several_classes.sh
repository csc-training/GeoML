#!/bin/bash
# Change to own project, if used outside of the course
#SBATCH --account=project_2002044
#SBATCH --partition=gputest
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=5G
#SBATCH --time=0:02:00
#SBATCH --gres=gpu:v100:1
# Reservation needed only during course
# SBATCH --reservation geoml-gpu

module load tensorflow
# Run the Python code, give model path and number of classes in labels as arguments

# During the course, use the previously trained model.
# srun python3 08_2_predict.py '/scratch/project_2002044/model_best_multiclass_kylli.h5' 5

# Run with own model
srun python3 08_2_predict.py 'model_best_multiclass.h5' 5