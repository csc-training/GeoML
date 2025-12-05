# Object detection with a pre-trained model

(For nicer reading in Jupyter, righ-click and select `Show Markdown Preview`.) 

The goal of this exercise is to use pre-trained model from Huggingface to predict boats based on an Sentinel-2 1C RGB image. 

Main steps:
* Sentinel data download.
* Pre-trained model download.
* Ultralytics YOLO11 model creation from pre-trained weights
* Tiled prediction of Sentinel-2 1C RGB image.

We will use GPU for running the inference. In order to utilize GPUs we will run the script as a batch job. 

The main libraries of this exercise are:
* [Ultralytics](https://docs.ultralytics.com) model
* [SAHI](https://obss.github.io/sahi/) - tiled predicting

## Workflow

1. Open these files, we will go through it in details.
    * Python file: [09_object_detection_existing_model.py](09_object_detection_existing_model.py)
    * SLURM batch job file: [09_object_detection_batch_job.sh](09_object_detection_batch_job.sh)
    * No modifications are needed to the files.
    * Open in another tab of web-browser in the supercomputer web interface -> Login node shell
    * A black window with SSH connection to the supercomputer opens, now Linux commands must be used.
    * The shell opens in everybody's home directory, to access the files, change working 
      directory:
        * `cd /scratch/project_462001167/students/$USER/GeoML/09_object_detection/existing_model`
    * See that you are in the right folder:
        * `ls -l`.
        * It should list the files that you see also in Jupyter file panel.
    * Submit a batch job:
        * `sbatch 09_object_detection_batch_job.sh`
    * It prints back something like, exact number will be different:
        * `Submitted batch job 1212121212`
    * To see the Python output file, open it with `tail`, the exact file name depends on the previosly printed job number:
        * `tail -f slurm-1212121212.out`.
        * The output file includes:
            * Used model
            * Number of tiles in the prediction
            * This output file is also the first place to look for errors, when writing own scripts.
    * Optional, to see full output from beginning:
        * `less slurm-1212121212.out` (this does not update, if file gets more rows).
    * It is possible to see job's state (waiting, running, finished) and used resources with
        * `sacct -o jobid,partition,state,reqmem,maxrss,averss,elapsed`
        * (In CSC Puhti: `seff 1212121212`)
    * The produced annotated image is saved as a file to `preds/prediction_visual.png`
    * It is a rather big file, so opening it with Jupyter works not so well. To open the file, use rather `Files` section of the web interface:
        * Select `/scratch/project_462001167`
        * Navigate to exercise's folder, something like `/scratch/project_462001167/students/ekkylli/GeoML/09_object_detection/existing_model/preds`
        * Click the file name: `prediction_visual.png`
        * Alternatively, you can download the file to your local machine and see it with some local tool.
