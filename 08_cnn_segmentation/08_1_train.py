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
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import Recall
from tensorflow.keras.metrics import Precision
from tensorflow.keras.metrics import binary_crossentropy

from tensorflow import one_hot
import tensorflow as tf
import keras.backend as K

# The CNN model architecture is in anouther local file:
import model_solaris

#SETTINGS

# In Puhti the training data is moved to local disk of GPU, 
# so the path to data has to be given as argument from batch job.
# Check that Python is given exactly two arguments:
#  - first is script name, has index 0
#  - second is the path to training data, has index 1
if len(sys.argv) != 3:
   print('Please give the data directory and number of classes')
   sys.exit()

data_dir=sys.argv[1]
# The number of classes in labels
no_of_classes=int(sys.argv[2])
print(no_of_classes)

# The results are written to Puhti scratch disk
user = os.environ.get('USER')
results_dir = os.path.join('/scratch/project_2002044', user, '2022/GeoML/08_cnn_segmentation')
logs_dir= os.path.join(results_dir, 'logs', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

no_of_image_bands=8

# Folders with data and labels and for results and log
# Model name includes learning rate and epsilon for optimizer
train_data_dir = os.path.join(data_dir, 'imageTiles_1024')
train_data_file_name='image_'
if no_of_classes == 2: 
    labels_data_dir = os.path.join(data_dir, 'binaryLabelTiles_1024')  
    label_file_name = 'labels_forest_'
    model_best = os.path.join(results_dir, 'model_best_binary.h5')
    training_log_file = os.path.join(results_dir, 'log_binary.csv')
else:
    labels_data_dir = os.path.join(data_dir, 'multiclassLabelTiles_1024')
    label_file_name = 'labels_multiclass_'
    model_best = os.path.join(results_dir, 'model_best_multiclass.h5')
    training_log_file = os.path.join(results_dir, 'log_multiclass.csv')    

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
batch_size=8
# Number of epochs depends a lot on amount of data
# In the exercise we have little data, so big amount of epochs goes fast.
no_of_epochs = 5000
# Changing optimizer or its settings could be the first option for trying different models
# By default Adam epsilon is much smaller, but for image segmentation tasks bigger epsilon like here could work better.
# Lower learning rate could also be often better. 
#optimizer = Adam(learning_rate=0.0001, epsilon=1.0)
optimizer = "rmsprop"

# Set loss and metrics shown during training according to the number of classes.
if no_of_classes == 2: 
    loss='binary_crossentropy'  
    metrics=['accuracy']
else:
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)   
    metrics=['sparse_categorical_accuracy']    

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
    # Because of tile overlap this is far from ideal. 
    train_split = int(0.7*len(all_frames_df))
    train_frames = all_frames_df[:train_split]
    val_frames = all_frames_df[train_split:]
        
    return train_frames, val_frames

# Custom data generator for training, using rasterio.
# Keras ImageDataGenerator cann't be used because it uses PIL and PIL does not support multi-channel images bigger than 8-bit.
# Rasterio should support reading also other data spatial raster formats via GDAL.
# The data generator reads images from disk and crops and makes augmentations for each image.
# It returns the data in batches, therefore yield, not return.
# https://www.tensorflow.org/tutorials/images/segmentation
def data_gen(img_df, augment):
  # Just a number for itereting the files in order.
  c = 0
  
  # Create data for one batch
  while (True):
    # Initialize the numpy arrays for results in advance, just for performance.
    # For now filled with zeros.
    img = np.zeros((batch_size, modelTileSize, modelTileSize, no_of_image_bands)).astype('float')
    mask = np.zeros((batch_size, modelTileSize, modelTileSize, 1)).astype('float') #
    train_weights = np.zeros((batch_size, modelTileSize, modelTileSize, 1)).astype('float') # 
    #print(train_weights.shape) 
    
    #Read images on by one, the number of images depends on batch size
    for i in range(c, c+batch_size): #initially from 0 to 16, c = 0. 

      # Read training data and labels with rasterio
      # Transpose is needed because Keras requires different axis order than rasterio
      train_img_file = rasterio.open(img_df[data_col].iloc[i])
      train_img = train_img_file.read().transpose(1, 2, 0)

      train_mask_file = rasterio.open(img_df[label_col].iloc[i])
      train_mask = train_mask_file.read().transpose(1, 2, 0)
      # If multiclass training, create one-hot-encoded channels for all classes
      #if no_of_classes > 2: 
      #    # First drop the third dimension created by rasterio: (650, 650, 1) -> (650, 650)
      #    train_mask = train_mask.reshape(trainingTileSize, trainingTileSize)
      #    # One-hot encode with tensorflow: (650, 650) -> (650, 650, 4)
      #    train_mask = one_hot(train_mask, no_of_classes)
      
      # Crop images randomly
      # Select randombly a crop location used both for the data image and label
      x = np.random.randint(0, trainingTileSize - modelTileSize + 1)
      y = np.random.randint(0, trainingTileSize - modelTileSize + 1)    
      # Crop from same locations
      train_img_cropped = train_img[y:(y+modelTileSize), x:(x+modelTileSize), :]
      train_mask_cropped = train_mask[y:(y+modelTileSize), x:(x+modelTileSize), :] #
      
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
           
      class_weights = tf.constant([5.0, 2.0, 1.0, 10.0, 10.0])
      class_weights = class_weights/tf.reduce_sum(class_weights)
      #print(class_weights)
      train_weights_image = tf.gather(class_weights, indices=tf.cast(train_mask_cropped, tf.int32))
      #print(train_weights_image.shape)
              
      # Stack all images of the batch
      img[i-c] = train_img_cropped #add to array - img[0], img[1], and so on.
      mask[i-c] = train_mask_cropped
      train_weights[i-c] = train_weights_image
      

    c+=batch_size
    # If not enough tiles for next batch, shuffle the images list and start from beginning again.
    if (c+batch_size) >= len(img_df):
      c=0
      img_df = img_df.sample(frac=1).reset_index(drop=True)
    #print(img.shape)
    #print(mask.shape)
    yield img, mask, train_weights
    #return img, mask, train_weights
  
    
# Train the model
def trainModel(train_gen, val_gen, no_of_training_tiles, no_of_validation_tiles):
   
    # If CNN model already exist continue training
    if os.path.exists(model_best):
        #m = load_model(model_best, custom_objects={'focal_tversky_loss': focal_tversky_loss})
        m = load_model(model_best)

    # Create new CNN model
    else:
        # Get the model archtecture from the external file
        m= model_solaris.cosmiq_sn4_baseline(no_of_classes=no_of_classes)
        m.summary()
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
    
    # Enable writing logs suitable fro TensorBoard
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
