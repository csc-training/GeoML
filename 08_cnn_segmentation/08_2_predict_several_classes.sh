#!/bin/bash
#SBATCH --account=project_2002044
#SBATCH --partition=gputest
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=4G
#SBATCH --time=0:14:00
#SBATCH --gres=gpu:v100:1

module load tensorflow
srun python3 08_2_predict.py 'model_best_multiclass_rmsprop_sparse_sample_weights' 5
