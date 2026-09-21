"""
Script for training a YOLO model from Ultralytics for detecting marine vessels from Sentinel-2 imagery.
Training data for the model is created using the 08_1_data_preparation notebook. The created YOLO compatible training data is can be found from the yolo_data folder. In this script we train the model and save the output containing the log files and weights to the yolo_project folder. 

Created on Fri Oct 10 2025

@author: ihakulin
Ideas and codesnippets from: 
* https://docs.ultralytics.com/modes/train/#introduction

"""

import os, sys, time, datetime

# Load model
from ultralytics import YOLO

def main():

    exercise_folder =  os.path.join(os.sep, 'scratch', 'project_2019932', 'students', os.environ.get('USER'), 'GeoML', '08_object_detection', '08B_own_model_training')     

    # Set path to the dataset yolo.yaml configuration file. This file contains dataset-specific parameters, including paths to training and validation data, class names, and number of classes.
    yolo_yml = os.path.join(exercise_folder, 'yolo_data', 'yolo.yaml')
    print("yolo.yml:", yolo_yml)

    # Set path to project directory where the training outputs are saved
    yolo_project = os.path.join(exercise_folder, 'yolo_project')
    print("YOLO project:", yolo_project)    

    # Initialize YOLOv26 model from Ultralytics
    model = YOLO('yolo26n')

    # Train the model using the set hyperparameters and data paths
    results = model.train(
        data=yolo_yml, 
        epochs=200,                            # Number of epochs
        patience=20,                           # Set patience as the number of epochs the model considers before stopping when validation loss doesn't decrease anymore
        imgsz=320,                             # Image size 
        batch=16,                              # Batch size
        optimizer='SGD',                       # Optimizer. SGD (Stochastic Gradient Descent) is a simple and effective choice. Another suitable choice is for example ADAM.
        project=yolo_project,
        workers=len(os.sched_getaffinity(0))   # Number of workers, based on available cores 
    )


if __name__ == '__main__':
    ### This part just runs the main method and times it
    start = time.time()
    main()
    end = time.time()
    print("Script completed in " + str(round(((end - start)/60),0)) + " minutes") 
