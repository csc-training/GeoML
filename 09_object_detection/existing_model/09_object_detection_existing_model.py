"""
Script for predicting marine vessels from Baltic Sea using an existing YOLO model from Ultralytics with pre-trained weigths. The weights meant for detecting marine vessels from L1C-TCI Sentinel-2 imagery with YOLO11 model are downloaded from the Hugging Face Platform (https://huggingface.co/mayrajeo/marine-vessel-detection-yolo). The used model is originally from the paper: http://dx.doi.org/10.2139/ssrn.4827287. 
"""

import os, shutil, time
import urllib.request

import torch

# YOLO
from ultralytics import YOLO

# Sahi
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction


def main():
    # Set path to the exercise directory
    exercise_folder = os.path.join(os.sep, 'scratch', 'project_462001167', 'students', \
                                   os.environ.get('USER'), 'GeoML', '09_object_detection', 'existing_model') 
    sentinel_image_pre_downloaded = '/scratch/project_462001167/09_sentinel_images/T34VEN_20210714T100029_TCI.tif'
    sentinel_image = os.path.join(exercise_folder, 'T34VEN_20210714T100029_TCI.tif')
    
    # Set image size used for inference
    image_size = 320
    
    # Pre-trained model weights
    model_url = 'https://huggingface.co/mayrajeo/marine-vessel-yolo/resolve/main/yolo11s_tci.pt'
    model_path = os.path.join(exercise_folder, 'yolo11s_sentinel2_rgb_marine_vessel_detection.pt')
    
    # Set confidence threshold for a prediction to be considered 
    # (the probability of a predicted bounding box containing the predicted object)
    # This is rather row threshold. In other project higher could be better.
    confidence = 0.1
    
    # Set computing device: GPU or CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} for predictions...')

    # Copy Sentinel-2 image to your own folder
    if not os.path.exists(sentinel_image):
        shutil.copy(sentinel_image_pre_downloaded, sentinel_image)
    
    # Copy model weights from HuggingFace.
    urllib.request.urlretrieve(model_url, model_path)
    
    print(f'Model weights from: {model_url}')

    # Instantiate a YOLO11 model for object detection using SAHI wrapper with the pre-trained weights
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='ultralytics',
        model=YOLO(model_path),
        confidence_threshold=confidence,
        device=device,
    )
    
    # Use SAHI sliced prediction to automatically slice the image into tiles with a size of 320. 
    # The inference is ran using the tiles and an overlap of 0.2. 
    # Lastly the function combines the tiles producing the combined annotation for whole image
    result = get_sliced_prediction(
        sentinel_image,
        detection_model,
        slice_height = image_size,
        slice_width = image_size,
        overlap_height_ratio = 0.2,
        overlap_width_ratio = 0.2
    )
    
    # Store the annotation image into 'preds/prediction_visual.png'
    result.export_visuals(export_dir="preds", text_size=0.25, rect_th=1, hide_conf=False)

if __name__ == '__main__':
    ### This part just runs the main method and times it
    start = time.time()
    main()
    end = time.time()
    print("Script completed in " + str(round(((end - start)/60),0)) + " minutes") 
