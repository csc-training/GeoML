#!/bin/bash
#SBATCH --account=project_2002044
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=20G
#SBATCH --time=24:14:00
#SBATCH --gres=gpu:v100:1,nvme:20 #Local disk in Gb

module load tensorflow
echo $LOCAL_SCRATCH

tar xf trainingTilesBinary_1024.tar -C $LOCAL_SCRATCH

ls $LOCAL_SCRATCH

srun python3 08_1_train.py $LOCAL_SCRATCH 2
#srun python3 09_1_train.py '/scratch/project_2000599/geoml/04_cnn_keras'
