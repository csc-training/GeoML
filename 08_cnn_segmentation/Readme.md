# CNN semantic segmentation exercise

(For nicer reading in Jupyter, righ-click and select `Show Markdown Preview`.) 

In this exercise, the land use classes are predicted with CNN semantic segmentation using [torchgeo](https://torchgeo.readthedocs.io/) and [Lightning](https://lightning.ai/docs/overview/getting-started) libraries.

The used data is the same as in shallow and deep classification exercises. 

**Before starting with this exercise, the general [raster data preparations exercise](../02_raster_data_preparation) must be done.**

Satellite images are usually too big for CNN models as such, se we need to tile them to smaller tiles for training the model and also later for prediction. Torchgeo has very nice functionality for tiling and sampling the data for training. Unfortunatelly similar functionality does not exist for inference.

This exercise includes two steps:
* Model training, incluing data loading and tiling with torchgeo. This part is run as batch job, because GPU-resoruces are needed.
* Inference and evaluation of the model visually and by calculating performance metrics.

## Data loading and CNN model training as a batch job.
* Open these files, we will go through it in details.
    * Python file with PyTorch code: [08_1_train.py](08_1_train.py)
    * HPC batch job file: [08_1_train_model.sh](08_1_train_model.sh)
    * No modifications are needed to the files.
* Submit Python script as SLURM batch job in a supercomputer:
    * Open Terminal to login-node: Open Apps -> Login node shell
    * A black window with SSH connection to Puhti opens, now Linux commands should be used.
    * The shell opens in home directory, to access the files, change working 
    directory:
        * `cd /scratch/project_462001167/students/$USER/GeoML/08_cnn_segmentation`
    * See that you are in the right folder:
        * `ls -l`.
        * It should list the files that you see also in Jupyter File panel.
    * Submit a batch job:
        * `sbatch 08_1_train_model.sh`
    * It outputs the job number, for example: `Submitted batch job 1212121212`
* To see the Python output file, open it with `tail`, the exact file name depends on the number printed previosly:
    * `tail -f slurm-1212121212.out`.
    * The output file includes:
        * Printout of used folders, just to double-check
        * Results of each epoch. 
        * This output file is also the first place to look for errors, when writing own scripts.
    * Optional, to see full output from beginning:
        * `less slurm-1212121212.out`
        * This does not update, if file gets more rows.
    * It is possible to see job's state (waiting, running, finished) and used resources with
        * `sacct -o jobid,partition,state,reqmem,maxrss,averss,elapsed`
        * (In CSC Puhti: `seff 1212121212`)
* There should be new files in the `08_cnn_segmentation` folder:
    * `best_model.ckpt` - the trained model in `checkpoints` folder. The best model has highest number. 
    *  Logs of training in `logs-<date>` folder that can be viewed using Tensorboard.

## Inference and evaluation of the model visually and by calculating performance metrics.
* Open Jupyter as decribed in [main Readme](../Readme.md)
* Open [08_2_evaluation_and_inference.ipynb](08_2_evaluation_and_inference.ipynb)


    
