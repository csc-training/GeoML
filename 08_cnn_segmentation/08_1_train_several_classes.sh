#!/bin/bash
#SBATCH --account=project_2002044
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=20G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:v100:1,nvme:20 #Local disk in Gb

module load tensorflow
export PYTHONPATH=/projappl/project_2002044/geoml_pip/lib/python3.8/site-packages
#module load tensorflow/nvidia-19.11-tf2-py3
#tar cvf forest.tar image_training_tiles_650 labels_all_classes_tiles_650
#TOFIX: set your own tiles folder
#tar xf /scratch/project_2002044/test/student_0000/tiles/forest.tar -C $LOCAL_SCRATCH

echo $LOCAL_SCRATCH

tar xf trainingTilesMulti_1024.tar -C $LOCAL_SCRATCH

ls $LOCAL_SCRATCH

srun python3 09_1_train.py $LOCAL_SCRATCH
#srun python3 09_1_train.py '/scratch/project_2000599/geoml/04_cnn_keras'
