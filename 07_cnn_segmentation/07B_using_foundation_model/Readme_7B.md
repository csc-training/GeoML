## Fine-tuning of Clay foundation model

In this exercise [Clay foundation](https://clay-foundation.github.io/model/) model is fine-tuned for our task. 
The foundation model is used as backbone, the decoder part is U-net CNN model trained from scratch. 

## Main steps

* Model training, including data loading and tiling with torchgeo. 
* Predicting the classification and class-wise probabilites.
* Evaluation of the model visually and by calculating performance metrics.

The first 2 steps are run as batch job, because GPU-resources are needed.
Read the `Readme`-files in both sub-directories for detailed instructions how to run the exercises.

## Model architecture

![model architecture](clay_unet_segmentation_architecture.png)

## Data loading and fine-tuning the model as a batch job.
* Open these files, we will go through it in details.
    * Python file with PyTorch code: [07B_1_train_model.py](07B_1_train_model.py)
    * HPC batch job file: [07B_1_train_model.sh](07B_1_train_model.sh)
    * No modifications are needed to the files.
* Submit Python script as SLURM batch job in a supercomputer:
    * Open Terminal to login-node: Open Tools -> Login node shell (Roihu-GPU)
    * A black window with SSH connection to Roihu opens, now Linux commands should be used.
    * The shell opens in home directory, to access the files, change working 
    directory:
        * `cd /scratch/project_462001167/students/$USER/GeoML/07_cnn_segmentation/07A_own_model_training`
    * See that you are in the right folder:
        * `ls -l`.
        * It should list the files that you see also in Jupyter File panel.
    * Submit a batch job:
        * `sbatch 07B_1_train_model.sh`
    * It outputs the job number, for example: `Submitted batch job 1212121212`
* To see the Python output file, open it with `tail`, the exact file name depends on the number printed previosly:
    * `tail -f slurm-1212121212.out`.
    * The output file includes:
        * Printout of used folders, just to double-check
        * This output file is also the first place to look for errors, when writing own scripts.
    * Optional, to see full output from beginning:
        * `less slurm-1212121212.out`
        * This does not update, if file gets more rows.
    * It is possible to see job's state (waiting, running, finished) and used resources with
        * `seff 1212121212`)
* Training takes about 10 minutes in Roihu.
* There should be new files in the `07A_own_model_training` folder:
    * `best_model.ckpt` - the trained model in `checkpoints` folder. The best model has highest number.
        *  It might be that Jupyter does not let to access the checkpoints folder, use Jupyter Terminal, Login-node shell or Files section in web interface to access it.
    *  Logs of training in `logs` folder that can be viewed using Tensorboard.

## Predict the classification and class-wise probabilites 
* Open these files, we will go through it in details.
    * Python file with PyTorch code: [07B_2_predict.py](07B_2_predict.py)
    * HPC batch job file: [07B_2_predict.sh](07B_2_predict.sh)
    * No modifications are needed to the files.
* Submit Python script as SLURM batch job in a supercomputer:
    * Submit a batch job:
        * `sbatch 07B_2_predict.sh`
* 3 new files are created:
    * The predicted classification and class probabilities .tif files to [../../classification_results](../../classification_results)-folder
    * `cnn_fm_model_description.txt`-file to the exercise folder, that shows the model architecture.

## Evaluate th model visually and by calculate performance metrics.
* Open Jupyter as described in [main Readme](../../Readme.md)
* Open [../07_3_segmentation_evaluation.ipynb](../07_3_segmentation_evaluation.ipynb) from the main folder of exerice 7.