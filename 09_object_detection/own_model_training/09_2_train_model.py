"""
Script for training a YOLO model from Ultralytics for detecting marine vessels from Sentinel-2 imagery.
Training data for the model is created using the 09_1_data_preparation notebook. The created YOLO compatible training data is can be found from the yolo_data folder. In this script we train the model and save the output containing the log files and weights to the yolo_project folder. 

Created on Fri Oct 10 2025

@author: ihakulin
Ideas and codesnippets from: 
* https://docs.ultralytics.com/modes/train/#introduction

"""

import os, sys, time, datetime

# Load model
from ultralytics import YOLO

# Torch
import torch


def get_args():
    #SETTINGS
    # Check that Python is given exactly two arguments:
    #  - first is script name, has index 0
    #  - second is the path to training data, has index 1
    if len(sys.argv) != 2:
        print('Please give the exercise directory')
        sys.exit()

    # Set exercise directory
    exercise_folder=sys.argv[1]

    # Set path to the dataset yolo.yaml configuration file. This file contains dataset-specific parameters, including paths to training and validation data, class names, and number of classes.
    yolo_yml = os.path.join(exercise_folder, 'yolo_data', 'yolo.yaml')
    print("yolo.yml:", yolo_yml)

    # Set path to project directory where the training outputs are saved
    yolo_project = os.path.join(exercise_folder, 'yolo_project')
    print("YOLO project:", yolo_project)    
    return yolo_yml, yolo_project

def main():
    # Parse args
    yolo_yml, yolo_project = get_args()

    # Set computing device: GPU or CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Number of workers, based on available cores
    # Max recommended for Ultralytics YOLO is 8
    cores = len(os.sched_getaffinity(0))
    
    # Number of epochs
    no_of_epochs = 200

    # Batch size
    batch_size = 16

    # Image size 
    image_size = 320

    # Optimizer
    # SGD (Stochastic Gradient Descent) is a simple and effective choice. Another suitable choice is for example ADAM. 
    optimizer = 'SGD'

    # Set patience as the number of epochs the model considers before stopping when validation loss doesn't decrease anymore
    # Usually a rather small amount is good so that the model doesn't overfit
    patience = 20

    # Initialize YOLOv8 model from Ultralytics
    model = YOLO('yolov8n')

    # Train the model using the set hyperparameters and data paths
    results = model.train(data=yolo_yml, epochs=no_of_epochs, patience=patience, imgsz=image_size, batch=batch_size, 
                      optimizer=optimizer, project=yolo_project, workers=cores)


if __name__ == '__main__':
    ### This part just runs the main method and times it
    start = time.time()
    main()
    end = time.time()
    print("Script completed in " + str(round(((end - start)/60),0)) + " minutes") 
