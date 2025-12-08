# Object detection with self-trained model

(For nicer reading in Jupyter, righ-click and select `Show Markdown Preview`.) 

This exercise includes the full workflow of object detection: data preparations, model training, running object detection, saving results in GIS-format, estimating the model.

Main steps:
* Sentinel data download.
* Training data download.
* Data preparation for Ultralytics YOLO
* Model training
* Tiled prediction
* Saving the results as geo-referenced GeoJSON
* Plotting prediction results
* Model estimation

The main libraries of this exercise are:
* [geoml2](https://github.com/mayrajeo/geo2ml) - convert the data into YOLO compatible format.
* [Ultralytics](https://docs.ultralytics.com) model training
* [SAHI](https://obss.github.io/sahi/) - tiled predicting


## Workflow

1.  The Sentinel-2 data download. The data for this exercise is pre-downloaded, because it would require CDSE credentials. You can get familiarized with the downloading by going through the notebook: [09_0_download_sentinel2_data.ipynb](09_0_download_sentinel2_data.ipynb).
        
2. Prepare data for the Ultralyltics YOLO model. The notebook creates Ultralyltics YOLO compatible dataset and splits the data to training, validation, and test sets. 
    * Open the Jupyter notebook: [09_1_data_preparation.ipynb](09_1_data_preparation.ipynb) in the web interface. The instructions and specific settings on creating an interactive session are listed in the course Readme: [Readme.md](Readme.md). **Use geoconda module**
    * The notebook creates a subdirectory `yolo_data`.
      
3. Modify the `yolo.yaml` settings file, that sets the paths to data. 
    * Copy the `yolo.yaml` file from `yolo_data/test` to the `yolo_data` directory.
    * Modify all paths
        * Remove the last sub-folder `test` from `path`
        * Add correct folder for other three folders, note that paths are relative to `path`
        * The resulting file should look like this, but with your own username in the first row.

```
path: /scratch/project_462001167/students/YOUR_USER_NAME/GeoML/09_object_detection/own_model_training/yolo_data # dataset root dir 
train: train # train images (relative to 'path')
val: val # val images (relative to 'path')
test: test # test images (relative to 'path')

# Classes
names:
  0: boat
```
   
4. Train the YOLO model for object detection using the data prepared in the earlier steps. We are using a YOLO model from Ultralytics.
    * Open these files, we will go through it in details.
        * Python file: [09_2_train_model.py](09_2_train_model.py)
        * HPC batch job file: [09_2_train_model.sh](09_2_train_model.sh)
    * No modifications are needed to the files.
    * Open in another tab of web-browser in the supercomputer web interface -> Login node shell
    * A black window with SSH connection to the supercomputer opens, now Linux commands must be used.
    * The shell opens in everybody's home directory, to access the files, change working 
      directory:
        * `cd /scratch/project_462001167/students/$USER/GeoML/09_object_detection/own_model_training`
    * See that you are in the right folder:
        * `ls -l`.
        * It should list the files that you see also in Jupyter file panel.
    * Submit a batch job:
        * `sbatch 09_2_train_model.sh`
    * It prints back something like, exact number will be different: `Submitted batch job 1212121212`
    * To see the Python output file, open it with tail, the exact file name depends on the number printed previosly:
        * `tail -f slurm-1212121212.out`.
        * The output file includes:
            * Printout of used folders, just to double-check
            * Model description
            * Results of each epoch. 
            * This output file is also the first place to look for errors, when writing own scripts.
    * Optional, to see full output from beginning: `less slurm-1212121212.out` (this does not update, if file gets more rows).
    * It is possible to see job's state (waiting, running, finished) and used resources with
        * `sacct -o jobid,partition,state,reqmem,maxrss,averss,elapsed`
        * (In CSC Puhti: `seff 1212121212`)
    * There should be new files in the `yolo_project/train` folder:
        * `weights/best.pt` - the trained model       
        * The training log files    

5. Run inference using the trained model. The predictions are created by submitting batch job: [09_3_predict.sh](09_3_predict.sh). This script runs the Python file: [09_3_predict.py](09_3_predict.py) and predicts the vessels from the test image.
    * `sbatch 09_3_predict.sh`
    * Similarly as before, the output of the batch job can be inspected by opening the slurm output file. The output file includes information on the detected vessels. The polygons geometry, confidence score and the amount of detected vessels in total. 
    * The predictions are saved to a file to the `predictions` folder.
      
6. Evaluate model using the [09_4_evaluate.ipynb](09_4_evaluate.ipynb) Jupyter notebook. The script plots the results and computes performance metrics. 
    

