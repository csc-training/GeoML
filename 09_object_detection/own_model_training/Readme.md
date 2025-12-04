# Object detection

(For nicer reading in Jupyter, righ-click and select `Show Markdown Preview`.) 

The goal of this exercise is to detect marine vessels from the Baltic Sea, based on Sentinel-2 1C data. The model for object detection is a YOLO model. 

The main libraries of this exercise are:
* [geoml2](https://github.com/mayrajeo/geo2ml) - convert the data into YOLO compatible format.
* [Ultralytics](https://docs.ultralytics.com) model training
* [sahi](https://obss.github.io/sahi/) - tiled predicting

This exercise heavily relies on the work of [Janne Mäyrä](https://scholar.google.com/citations?user=xAT9080AAAAJ&hl=en) from SYKE (Finnish Environmental institute). He has prepared the used training data, `geo2ml`-library and example code for model training and predicting (TODO).

## Data sources
* ESA, [Sentinel-2 1C RGB data](https://sentinels.copernicus.eu/sentinel-data-access/sentinel-products/sentinel-2-data-products/collection-1-level-1c), (pre-)downloaded from [CDSE](https://dataspace.copernicus.eu/) via STAC
* Janne Mäyrä, SYKE, Dataset for marine vessel detection from Sentinel 2 images in the Finnish coast, downloaded from [Zenodo](https://doi.org/10.5281/zenodo.15019034). For keeping exercise durition reasonable, we use only part of the data.

Main steps:
* Sentinel data download.
* Training data download.
* Data preparation for Ultralytics YOLO
* Model training
* Tiled prediction 
* Model estimation
* TODO

## Workflow

1.  The Sentinel-2 data downloaded. The data for this exercise is pre-loaded, because it would require CDSE credentials. You can get familiarized with the downloading by going through the notebook: [09_0_download_sentinel2_data.ipynb](09_0_download_sentinel2_data.ipynb).
        
2. Prepare data for the YOLO model. The notebook creates a YOLO compatible dataset and splits the data to training, validation, and test sets. 
    * Open the Jupyter notebook: [09_1_data_preparation.ipynb](09_1_data_preparation.ipynb) in the web interface. The instructions and specific settings on creating an interactive session are listed in the course Readme: [Readme.md](Readme.md). **Use geoconda module**
    * The notebook creates a subdirectory `datasets` that includes the data for model training and evaluation.
      
3. Modify the `yolo.yaml` settings file, that sets the paths to data. 
    * Copy the `yolo.yaml` file from `datasets/test` to the `datasets` directory.
    * Modify all paths
        * Remove the last sub-folder `test` from `path`
        * Add correct folder for other three folders, note that paths are relative to `path`
        * The resulting file should look like this, but with your own username in the first row.

```
path: /scratch/project_462001167/students/YOUR_USER_NAME/GeoML/09_object_detection/own_model_training/datasets # dataset root dir 
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
    * It is possible to see job's state (waiting, running, finished) and used resources with `seff 1212121212`
    * There should be new files in the `09_object_detection/yolo_project/train` folder:
        * `weights/best.pt` - the trained model       
        * the training log files    

5. Run inference using the trained model. The predictions are created by submitting batch job: [09_3_predict.sh](09_3_predict.sh). This script runs the Python file: [09_3_predict.py](09_3_predict.py) and predicts the vessels from the test image.
    * `sbatch 09_3_predict.sh`
    * Similarly as before, the output of the batch job can be inspected by opening the slurm output file. The output file includes information on the detected vessels. The polygons geometry,           confidence score and the amount of detected vessels in total. 
    * The predictions are saved to a file to the `predictions` folder.
      
6. Evaluate model using the [09_4_evaluate.ipynb](09_4_evaluate.ipynb) Jupyter notebook. The script plots the results and computes performance metrics. 
    

