#!/bin/bash
#SBATCH --account=project_2002044
#SBATCH --partition=gputest
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=0:14:00
#SBATCH --gres=gpu:v100:1

module load tensorflow
srun python3 08_2_predict.py 'model_best_binary_05_001' 2
