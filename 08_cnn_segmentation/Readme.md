# CNN instructions

(For nicer reading in Jupyter, righ-click and select `Show Markdown Preview`.) 

1. Tile data for CNN: [08_0_CNN_tiling.ipynb](08_0_CNN_tiling.ipynb). Must be opened with geoconda module as usual Notebook.
    * In Puhti web interface open the Jupyter app (not Jupyter for courses).
    * Select following settings:
        * Project: project_2002044 
		* Reservation: geoml
        * Partition: interactive
        * CPU cores: 1
        * Memory (Gb): 8 
        * Local disk: 0
        * Time: 5:00:00
        * Python: geoconda
        * Jupyter type: Lab
        * Working directory: /scratch/project_2002044
        
2. Train the model for binary classes. 
    * Open these files, we will go through it in details.
    	* Model definition file: [model_solaris.py](model_solaris.py)	
        * Python file with Tensorflow code: [08_1_train.py](08_1_train.py)
        * HPC batch job file: [08_1_train_binary.sh](08_1_train_binary.sh)
    * No modifications are needed to the files.
    * Open in another tab of web-browser, Puhti web interface -> Login node shell
    * A black window with SSH connection to Puhti opens, now Linux commands must be used.
    * The shell opens in everybody's home directory, to access the files, change working 
    directory: `cd /scratch/project_2002044/training_0xx/2022/GeoML/08_cnn_segmentation`
    * See that you are in the right folder: `ls -l`. It should list the files that you see also in Jupyter File panel.
    * Training proper CNN models takes hours even on GPU-machine, so during the course we run only a mini-test for 2 minutes (limit set in the batch job file).
    * Submit a batch job: `sbatch 08_1_train_binary.sh`
    * It prints back something like, exact number will be different: `Submitted batch job 1212121212`
    * To see the Python output file, open it with tail, the exact file name depends on the number printed previosly: `tail -f slurm-1212121212.out`. The output file includes:
        * Printout of used folders, just to double-check
        * Tensorflow warnings, should be ok.
        * Model description
        * Results of each epoch. 
        * This output file is also the first place to look for errors, when writing own scripts.
    * Optional, to see full output from beginning: `less slurm-1212121212.out` (this does not update, if file gets more rows).
    * It is possible to see job's state (waiting, running, finished) and used resources with `seff 1212121212`
    * There should be new files in the 08_cnn_segmentation folder:
        * `model_best_binary.h5` - the trained model       
        * `log_binary.csv` - logs of training    
    * If you want later to run the model for hours, just change the time limit from batch job. The Python file inlcudes early stopping, so training will be stopped after 100 epochs without better model found.

3. Predict the whole image for binary classes. 
     * Open these files, we will go through it in details.
        * Python file with Tensorflow code: [08_2_predict.py](08_2_predict.py)
        * HPC batch job file: [08_2_predict_binary.sh](08_2_predict_binary.sh)
    * Submit the batch job again from login node shell: `sbatch 08_2_predict_binary.sh`
    * Files created by this script:
        * Predicted tiles in folder `predictions512_2`
        * Merged big .tif file: `CNN_2.tif`

4. Evaluate the bianry class model visually.
    * Open the .tif file with QGIS.
   * If you want to try different prediction threshold values, we can restyle the layer:
       * Properites -> Symbology
       * Render type: Single band gray
       * Min:0, Max: threshold value
       * Contrast enhancement: Clip to MinMax
       * Compare to labels and Sentinel image.
       * If you want you can add also orthoimage WMS: `https://tiles.kartat.kapsi.fi/ortokuva?`

5. Evaluate the binary model with scikit-learn
    * Open [08_3_evaluate.ipynb](08_3_evaluate.ipynb)
    * Do not run the multi-class cells yet.
    
6. Train the model for multiple classes. 
    * Open this file, we will go through it in details.
        * HPC batch job file: [08_1_train_several_classes.sh](08_1_train_several_classes.sh)
    * Submit a batch job: `sbatch 08_1_train_several_classes.sh`
    * See instruction above for binary class training.
    * New files:
        * `model_best_multiclass_rmsprop_sparse_sample_weights.h5` - the trained model
        * `log_multiclass_rmsprop_sparse_sample_weights.csv` - logs of training
        
7. Predict the whole image for multiple classes.
     * Open these files, we will go through it in details.
        * HPC batch job file: [08_2_predict_several_classes.sh](08_2_predict_several_classes.sh)            
    * Submit the batch job again from login node shell: `sbatch 08_2_predict_several_classes.sh`
    * Files created by this script:
        * Predicted tiles in folder `predictions512_5`
        * Merged big .tif file: `CNN_5.tif`    
    
8. Evaluate the multi-class model visually with QGIS.
    
9. Evaluate the multi-class model with scikit-learn
    * Open again [08_3_evaluate.ipynb](08_3_evaluate.ipynb)
    * Run also the multi-class cells.
