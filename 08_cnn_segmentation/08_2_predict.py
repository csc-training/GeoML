# -*- coding: utf-8 -*-
"""
Script for predicting/interference based on a CNN segmentation model based on GeoTiff tiles.
As last the tiles are merged to one big image.
The model may be binary or multi-class, set no_of_classes accordingly.
The main Python libraries are Keras, rasterio and numpy.

Created on Fri Mar  6 12:46:58 2020

@author: ekkylli
"""
import os, time, glob, sys
import numpy as np
import rasterio
import rasterio.merge
from tensorflow.keras.models import load_model

#SETTINGS

# In Puhti the training data is moved to local disk of GPU, 
# so the path to data has to be given as argument from batch job.
# Check that Python is given exactly two arguments:
#  - first is script name, has index 0
#  - second is the path to training data, has index 1
if len(sys.argv) != 3:
   print('Please give the data directory and number of classes')
   sys.exit()

model_path=sys.argv[1]

no_of_classes=int(sys.argv[2])
print(no_of_classes)

user = os.environ.get('USER')
cnn_dir = os.path.join('/scratch/project_2002044', user, '2022/GeoML/08_cnn_segmentation')

# Paths for INPUTS: data and model
prediction_data_dir = os.path.join(cnn_dir, 'imageTiles_512')

#Paths for RESULTS
predicted_tiles_folder = os.path.join(cnn_dir, 'predictions512_'+str(no_of_classes))
prediction_image_file = os.path.join(cnn_dir,'CNN_'+str(no_of_classes)+'.tif')

#Setting of the data
img_size = 512
img_channels = 8

outputMeta = None

# Predict a tile and save it as .tif file
def predictTile(model, dataImage):
    # Set the file paths    
    predictedImageFile = os.path.join(predicted_tiles_folder, os.path.basename(dataImage))
    
    
    with rasterio.open(dataImage, 'r') as image_dataset:    
        # Read the data image
        image_data = image_dataset.read()
        
        #Reorder axis for Keras, channel last
        image_data2 = np.transpose(image_data, (1, 2, 0)) 
        
        #Reshape data for Keras, add extra dimension,  
        image_data3 = image_data2.reshape(1, img_size, img_size, img_channels)
        
        # predicting the probability of each pixel
        prediction = model.predict(image_data3, verbose=0)

        # If multi-class, find the class with best probability
        if no_of_classes > 2: 
            prediction = np.argmax(prediction, 3)
        
		# Reshape for rasterio       
        prediction2 = prediction.reshape(img_size, img_size)     
		
		# Save the results as .tif file.
		# Copy the coorindate system information, image size and other metadata from the satellite image 
        global outputMeta
        outputMeta = image_dataset.meta
		# Change the data type in file meta.
        if no_of_classes == 2: 
            dtype='float32'  
        else:
            #For multi-class change also data type, argmax output is in int64 not supported by rasterio.
            prediction2 = prediction2.astype(np.uint8)
            dtype=rasterio.uint8
        outputMeta.update(count=1, dtype=dtype,nodata=255)
        # Writing the image on the disk
        with rasterio.open(predictedImageFile, 'w', **outputMeta) as dst:
            dst.write(prediction2, 1)

               
# Merge all tiles to one big .tif-image
def  mergeTiles():
    
    #Find all .tif files in the predicted tiles folder
    tile_files = glob.glob(predicted_tiles_folder+"/*.tif")
    
    #Create a mosaic of all files           
    mosaic, out_trans = rasterio.merge.merge(tile_files)
        
    #Set output files metadata correctly
    outputMeta.update({
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans
    })
    
    #Write the output file to disk
    with rasterio.open(prediction_image_file, "w", **outputMeta) as dest:
        dest.write(mosaic)            
        

def main():    
    
    # Load the previously trained model
    model = load_model(model_path)
    
    # Find all data tiles for prediction
    all_frames = glob.glob(prediction_data_dir+"/*.tif")
    # Make a folder for the predicted tiles
    os.makedirs(predicted_tiles_folder, exist_ok=True)
    
    # Predict for all tiles
    for tile in all_frames:
        predictTile(model, tile)
    
    #Merge tiles to one GeoTiff    
    mergeTiles()
       
if __name__ == '__main__':
    ### This part just runs the main method and times it
    start = time.time()
    main()
    end = time.time()
    print("Script completed in " + str(round((end - start),0)) + " seconds")    