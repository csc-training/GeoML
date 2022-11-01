# CNN instructions

(For nicer reading in Jupyter, righ-click and select `Show Markdown Preview`.) 

1. Tile data for CNN: [08_0_CNN_tiling.ipynb](08_0_CNN_tiling.ipynb). Must be opened with geoconda module as usual Notebook.
    * In Puhti web interface open the Jupyter app (not Jupyter for courses).
    * Select following settings:
        * Project: project_2002044 /
        * Partition: interactive
        * CPU cores: 1
        * Memory (Gb): 8 
        * Local disk: 0
        * Time: 1:00:00
        * Python: geoconda
        * Jupyter type: Lab
        * Working directory: /scratch/project_2002044
        
2. Train the model for binary classes. Open these files, we will go through it in details.
    * Python file with Tensorflow code: [08_1_train.py](08_1_train.py)
    * HPC batch job file: [08_1_train_binary.sh](08_1_train_binary.sh)
    * No modifications are needed to the files.
    * Open in another tab of web-browser, Puhti web interface -> Login node shell
    * A black window with SSH connection to Puhti opens, now Linux commands must be used.
    * The shell opens in everybody's home directory, to access the files, change working 
    directory: `cd /scratch/project_2002044/training_0xx/2022/GeoML/08_cnn_segmentation`
    * See that you are in the right folder: `ls -l`. It should list the files that you see also in Jupyter File panel.
    * Training proper CNN models takes hours even on GPU-machine, so during the course we run only a mini-test for 3 minutes (limit set in the batch job file).
    * Submit a batch job: `sbatch 08_1_train_binary.sh`
    * It prints back something like, exact number will be different: `Submitted batch job 1212121212`
    * To see the Python output file, open it with tail, the exact file name depends on the number printed previosly: `tail -f slurm-1212121212.out`. The output file includes:
        * Printout of used folders, just to double-check
        * Tensorflow warnings, should be ok.
        * Model description
        * Results of each epoch. If a better model was found, the model is saved:

```
Epoch 1: val_loss improved from inf to 0.08032, saving model to /scratch/project_2002044/ekkylli/2022/GeoML/08_cnn_segmentation/model_best_spruce_05_001.h5
4/4 - 18s - loss: 0.0817 - sparse_categorical_accuracy: 0.4033 - val_loss: 0.0803 - val_sparse_categorical_accuracy: 0.4802 - 18s/epoch - 5s/step
Epoch 2/5000
```

    * It is possible to see job's state (waiting, running, finished) and used resources with `seff 1212121212`
    * If you want later to run the model for hours, just change the time limit from batch job. The Python file inlcudes early stopping, so training will be stopped after 100 epochs without better model found.

3. Predict the whole image for binary classes. Open these files, we will go through it in details.
    * Python file with Tensorflow code: [08_2_predict.py](08_2_predict.py)
    * HPC batch job file: [08_2_predict.sh](08_2_predict.sh)
    * Submit the batch job again from login node shell: `sbatch 08_2_predict.sh`