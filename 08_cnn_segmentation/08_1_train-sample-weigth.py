# -*- coding: utf-8 -*-
"""
Script for training a CNN segmentation model based on GeoTiff tiles and labels.
Labels may have one class or several, set no_of_classes accordingly.
The main Python libraries are Keras, rasterio and numpy.

Created on Thu Mar  5 13:17:30 2020

@author: ekkylli
Ideas and codesnippets from: 
* https://towardsdatascience.com/a-keras-pipeline-for-image-segmentation-part-1-6515a421157d
* https://jkjung-avt.github.io/keras-image-cropping/
* solaris: https://github.com/CosmiQ/solaris

"""

import os, sys, time, glob
import random
import datetime
import numpy as np
import pandas as pd
import rasterio



from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
#from tensorflow_addons.optimizers import AdamW

#from tensorflow.keras.optimizers.experimental import AdamW

from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import Recall
from tensorflow.keras.metrics import Precision
from tensorflow.keras.metrics import binary_crossentropy

from tensorflow import one_hot
import keras.backend as K

# The CNN model architecture is in anouther local file:
import model_solaris

#SETTINGS

# In Puhti the training data is moved to local disk of GPU, 
# so the path to data has to be given as argument from batch job.
# Check that Python is given exactly two arguments:
#  - first is script name, has index 0
#  - second is the path to training data, has index 1
if len(sys.argv) != 2:
   print('Please give the data directory')
   sys.exit()

data_dir=sys.argv[1]

data_dir='/scratch/project_2000599/geoml/04_cnn_keras'

# The results are written to Puhti scratch disk
# TOFIX: Change the path according to your own username
#results_dir='/scratch/project_2002044/students/ekkylli'
results_dir='/scratch/project_2000599/geoml/04_cnn_keras'
logs_dir= os.path.join(results_dir, 'logs', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

# The number of classes in labels
# TOFIX: Change the number according to the used data
#no_of_classes=2 #For binary classification
no_of_classes=5 # n for multiclass
no_of_image_bands=8

# Folders with data and labels and for results and log
# Model name includes learning rate and epsilon for optimizer
train_data_dir = os.path.join(data_dir, 'imageTiles_1024')
train_data_file_name='image_'
if no_of_classes == 2: 
    labels_data_dir = os.path.join(data_dir, 'label_tiles_650')  
    label_file_name = 'forest_spruce_clip_'
    model_best = os.path.join(results_dir, 'model_best_spruce_05_001.h5')
    training_log_file = os.path.join(results_dir, 'log_spruce_05_001.csv')
else:
    labels_data_dir = os.path.join(data_dir, 'labelTiles_1024')
    label_file_name = 'labels_'
    model_best = os.path.join(results_dir, 'model_best_multiclass_adam_01_001.h5')
    training_log_file = os.path.join(results_dir, 'log_multiclass_adam_01_001.csv')    

#Image sizes
# Training data size after tiling
# This size may well be changed, but change your data preparation accordingly.
trainingTileSize = 1024
# Training data size after crop, feeded to the model.
# Would not recommend changing this.
modelTileSize = 512

#Column names for training data dataframe
data_col='tile'
label_col='label'

#Trainig settings
# 16 or 32 might be better for bigger datasets 
batch_size=12
# Number of epochs depends a lot on amount of data
# In the exercise we have little data, so big amount of epochs goes fast.
no_of_epochs = 5000
# Changing optimizer or its settings could be the first option for trying different models
# By default Adam epsilon is much smaller, but for image segmentation tasks bigger epsilon like here could work better.
# Lower learning rate could also be often better. 
optimizer = Adam(learning_rate=0.1, epsilon=0.001)
#optimizer = AdamW(weight_decay=1e-4)

# Class weithts used during model training.
# Because in example data the classes have very different number of pixels,
# we use class weights to compensate.
# Without class_weights no pixels from class 3 were found and very few of class 2.
# But even with weights the model does not find class 3 well..
# The other option could be to use some other loss function.
# In binary classification the spruce class is simply given 10 time more importance than background.
if no_of_classes == 2: 
    class_weight = {0: 0.1, 1: 0.9}
else:
    # Weights are dependent on how many pixels certain class has in the data, smaller classes are given bigger weight:
    class_weight = {0: 1.,1: 10., 2: 200., 3: 5000., 4:1.}

# Set loss according to the number of classes.
# For very unbalanced cases it might make sense to use some other loss function.
# More suggestions for suitable loss functions: https://neptune.ai/blog/image-segmentation-in-2020
# Code examples available: https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch

# BCE-Dice Loss
# https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch/notebook
def DiceBCELoss(targets, inputs, smooth=1e-6):    
       
    #flatten label and prediction tensors
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)
    
    BCE =  binary_crossentropy(targets, inputs)
    intersection = K.sum(K.dot(targets, inputs))    
    dice_loss = 1 - (2*intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)
    Dice_BCE = BCE + dice_loss
    
    return Dice_BCE  

def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) +
                                           smooth)


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def tversky(y_true, y_pred, smooth=1, alpha=0.7):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
    return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)


def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true, y_pred)

def focal_tversky_loss(y_true, y_pred, gamma=0.75):
    tv = tversky(y_true, y_pred)
    return K.pow((1 - tv), gamma)

if no_of_classes == 2: 
    loss='binary_crossentropy'  
else:
    loss=focal_tversky_loss
    #loss=DiceBCELoss
    #loss='categorical_crossentropy'

# Select metrics shown during training. 
# Accuracy + recall and precision for each class separately for multiclass
if no_of_classes == 2: 
    metrics=['accuracy']

if no_of_classes == 5:
    p0 = Precision(top_k=1, class_id=0, name='precision_0')
    p1 = Precision(top_k=1, class_id=1, name='precision_1')
    p2 = Precision(top_k=1, class_id=2, name='precision_2')
    p3 = Precision(top_k=1, class_id=3, name='precision_3')
    p4 = Precision(top_k=1, class_id=4, name='precision_4')
    r0 = Recall(top_k=1, class_id=0, name='recall_0')
    r1 = Recall(top_k=1, class_id=1, name='recall_1')  
    r2 = Recall(top_k=1, class_id=2, name='recall_2')
    r3 = Recall(top_k=1, class_id=3, name='recall_3')
    r4 = Recall(top_k=1, class_id=4, name='recall_4')
    metrics=['accuracy', p0, p1, p2, p3, p4, r0, r1, r2, r3, r4] #, MeanIoU(num_classes=no_of_classes)   

# Read all the training files and randomly assign to training and validation sets
def prepareData():
    #List all .tif-files in training data folder.
    #Take only .tif files, for example if you open a .tif file in QGIS, 
    # it automatically creates the .tif.aux.xml file, which we do not want to include here.
    all_frames = glob.glob(train_data_dir+"/*.tif")
    
    # Arrange to random order
    random.shuffle(all_frames)
    
    # Change to Pandas dataframe
    all_frames_df = pd.DataFrame(all_frames, columns =[data_col]) 
    
    # Add labels files, labels are expected to have similar numbering than the data tiles.
    all_frames_df[label_col] = all_frames_df[data_col].str.replace(train_data_file_name, label_file_name, case = False)
    all_frames_df[label_col] = all_frames_df[label_col].str.replace(train_data_dir, labels_data_dir, case = False) 
    
    # Generate train, val, and test sets for frames
    # In the exercies we have so little data, so we skip the test set.
    # Here we use 70% of frames for training and 30% for validation.
    train_split = int(0.7*len(all_frames_df))
    train_frames = all_frames_df[:train_split]
    val_frames = all_frames_df[train_split:]
        
    return train_frames, val_frames

# Custom data generator for training, using rasterio.
# Keras ImageDataGenerator cann't be used because it uses PIL and PIL does not support multi-channel images bigger than 8-bit.
# Rasterio should support reading also other data spatial raster formats via GDAL.
# The data generator reads images from disk and crops and makes augmentations for each image.
# It returns the data in batches, therefore yield, not return.
def data_gen(img_df, augment):
  # Just a number for itereting the files in order.
  c = 0
  
  # Create data for one batch
  while (True):
    # Initialize the numpy arrays for results in advance, just for performance.
    # For now filled with zeros.
    img = np.zeros((batch_size, modelTileSize, modelTileSize, no_of_image_bands)).astype('float')
    mask = np.zeros((batch_size, modelTileSize, modelTileSize, no_of_classes)).astype('float')
    
    #Read images on by one, the number of images depends on batch size
    for i in range(c, c+batch_size): #initially from 0 to 16, c = 0. 

      # Read training data and labels with rasterio
      # Transpose is needed because Keras requires different axis order than rasterio
      train_img_file = rasterio.open(img_df[data_col].iloc[i])
      train_img = train_img_file.read().transpose(1, 2, 0)

      train_mask_file = rasterio.open(img_df[label_col].iloc[i])
      train_mask = train_mask_file.read().transpose(1, 2, 0)
      # If multiclass training, create one-hot-encoded channels for all classes
      if no_of_classes > 2: 
          # First drop the third dimension created by rasterio: (650, 650, 1) -> (650, 650)
          train_mask = train_mask.reshape(trainingTileSize, trainingTileSize)
          # One-hot encode with tensorflow: (650, 650) -> (650, 650, 4)
          train_mask = one_hot(train_mask, no_of_classes)
      
      # Crop images randomly
      # Select randombly a crop location used both for the data image and label
      x = np.random.randint(0, trainingTileSize - modelTileSize + 1)
      y = np.random.randint(0, trainingTileSize - modelTileSize + 1)    
      # Crop from same locations
      train_img_cropped = train_img[y:(y+modelTileSize), x:(x+modelTileSize), :]
      train_mask_cropped = train_mask[y:(y+modelTileSize), x:(x+modelTileSize), :]
      
      # Augment the data: flip horizontally, vertically and rotate 90 degrees.
      # Not used for validation data.
      if augment:
          if random.choice([True, False]):
              train_img_cropped = np.flipud(train_img_cropped)
              train_mask_cropped = np.flipud(train_mask_cropped)
          if random.choice([True, False]):
              train_img_cropped = np.fliplr(train_img_cropped) 
              train_mask_cropped = np.fliplr(train_mask_cropped) 
          t = random.choice([0, 3])
          if t > 0:
            train_img_cropped = np.rot90(train_img_cropped, t)    
            train_mask_cropped = np.rot90(train_mask_cropped, t) 
            #print (train_mask_cropped.shape)
              
      # Stack all images of the batch
      img[i-c] = train_img_cropped #add to array - img[0], img[1], and so on.
      mask[i-c] = train_mask_cropped

    c+=batch_size
    # If not enough tiles for next batch, shuffle the images list and start from beginning again.
    if (c+batch_size) >= len(img_df):
      c=0
      img_df = img_df.sample(frac=1).reset_index(drop=True)
    yield img, mask

  
    
# Train the model
def trainModel(train_gen, val_gen, no_of_training_tiles, no_of_validation_tiles):
   
    # If CNN model already exist continue training
    if os.path.exists(model_best):
        m = load_model(model_best, custom_objects={'focal_tversky_loss': focal_tversky_loss})

    # Create new CNN model
    else:
        # Get the model archtecture from the external file
        m= model_solaris.cosmiq_sn4_baseline(no_of_classes=no_of_classes)
        # Compile it with custom settings
        m.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=metrics)        
    
    # Add checkpoints to the training, save only best model
    checkpoint = ModelCheckpoint(model_best, monitor='val_loss', 
                                 verbose=1, save_best_only=True, mode='min')
    
    # Add logging to training, the log file can be with Excel to visualize the training results by epoch.
    # At least I had to replace all . with , in Excel.
    csv_logger = CSVLogger(training_log_file, append=True, separator=';')
    
    # Stop training if model does not get better in patience number of epochs.
    earlystopping = EarlyStopping(monitor = 'val_loss', verbose = 1,
                                  min_delta = 0.0001, patience = 100, mode = 'min')
    
    tensorboard_callback = TensorBoard(log_dir=logs_dir, histogram_freq=1)

    callbacks_list = [checkpoint, csv_logger, earlystopping, tensorboard_callback] #

    # Train the model
    m.fit(train_gen, epochs=no_of_epochs, 
                              steps_per_epoch = (no_of_training_tiles//batch_size),
                              verbose=2,
                              validation_data=val_gen, 
                              validation_steps=(no_of_validation_tiles//batch_size),
                              #class_weight=class_weight,
                              callbacks=callbacks_list) #
    
def main():
    # Read the files from data folders and divide between traininga, validataion (and testing).
    train_frames, val_frames = prepareData()
       
    # Genarators for training and validation. No augmentation for validation, otherwise the same.
    train_gen = data_gen(train_frames, augment=True)
    val_gen = data_gen(val_frames, augment=False)
    
    # Save how many images there is on both sets
    no_of_training_tiles = len(train_frames)
    no_of_validation_tiles = len(val_frames)      
    
    trainModel(train_gen, val_gen, no_of_training_tiles, no_of_validation_tiles)
    
if __name__ == '__main__':
    ### This part just runs the main method and times it
    start = time.time()
    main()
    end = time.time()
    print("Script completed in " + str(round(((end - start)/60),0)) + " minutes") 
