#!/bin/bash
#SBATCH --account=project_2002044
#SBATCH --partition=gputest
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=0:14:00
#SBATCH --gres=gpu:v100:1
#SBATCH --reservation geoml-gpu

module load tensorflow
# Run the Python code, give model path and number of classes in labels as arguments
# Use the previously trained model.
# If you want to see what the model can do after 2 minutest training, you ofcourse can change here your own model too.
srun python3 08_2_predict.py '/scratch/project_2002044/model_best_binary_kylli.h5' 2
