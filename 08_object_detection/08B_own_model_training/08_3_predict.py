"""
Script for prediction/inference using the trained YOLO model.
The inference is done using the tiled inference from sahi enabling the predictions for large mosaics.
The script runs the inference to a test image, georeferences the predictions, plots the predictions laid over the test image, and saves the predictions as a GeoJSON. 

Created on Fri Oct 17 2025

@author: ihakulin
Ideas and codesnippets from: 
* https://mayrajeo.github.io/ship-detection/
* https://docs.ultralytics.com/guides/sahi-tiled-inference/

"""

# General libraries
import os, sys, time

# Torch
import torch

# Pandas
import geopandas as gpd
import pandas as pd

# Rasterio
import rasterio as rio
import rasterio.plot as rioplot
from rasterio.plot import show
from rasterio.windows import from_bounds
 
# Coordinates script from mayrajeo/geo2ml for converting predictions to a GeoDataFrame
from coordinates import * 

# Ultralytics
from ultralytics import YOLO

# Shapely
from shapely.geometry import box

# Matplotlib for plotting
import matplotlib.pyplot as plt

# Sahi for tiled inference and model load
from sahi.models.ultralytics import UltralyticsDetectionModel
from sahi.predict import get_sliced_prediction

# Json
import json


def get_args():
    #SETTINGS
    # Check that Python is given exactly two arguments:
    #  - first is script name, has index 0
    #  - second is the path to training data, has index 1
    if len(sys.argv) != 2:
        print('Please give the exercise directory')
        sys.exit()

    # Set data directory
    exercise_dir=sys.argv[1]
    print("Exercise directory is:", exercise_dir)

    # Set path to test image for interference
    image = os.path.join(exercise_dir, 'sentinel2', 'T34VEN_20210714T100029_TCI.tif')

    # Set path to the trained model
    # Note, if you train model several times, Ultralytics creates new folders each time.
    model_path = os.path.join(exercise_dir, 'yolo_project', 'train', 'weights', 'best.pt')

    # yolo.yml settings file location
    yolo_yml = os.path.join(exercise_dir, 'yolo_data', 'yolo.yaml')

    # Set path for the outputs
    outpath = os.path.join(exercise_dir, 'predictions')
    os.makedirs(outpath, exist_ok=True)
    return image, model_path, outpath, yolo_yml


def main():
    # Parse args
    image, model_path, outpath, data_path = get_args()
    
    # Set computing device: GPU or CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} for predictions...')
    
    # Set confidence threshold for a prediction to be considered (the probability of a predicted bounding box containing the predicted object)
    # This is rather row threshold. In other project higher could be better.
    confidence = 0.1
    
    # Set image size used for inference
    image_size = 320
    
    # Load trained model using UltralyticsDetectionModel from SAHI for sliced interference
    model = UltralyticsDetectionModel(model_path=model_path,
                                            device=device,
                                            confidence_threshold=confidence,
                                            image_size=image_size)
    
    # Use half precision (FP16) to run inference faster and with less memory
    model.model.overrides.update({
        'half': True
    })
        
    # SAHI automatically tiles the data, and merges the predictions. 
    # Run inference on tiles with a size of 320 and overlap of 0.2.    
    # Overlap is used to avoid missing objects on the border 
    # 'perform_standard_pred' = True, would Perform a standard prediction on top of sliced predictions 
    # to increase large object detection accuracy. Here set to false, to predict only the tiles. 
    sliced_pred_results = get_sliced_prediction(image, 
                                                model, 
                                                slice_width=320,
                                                slice_height=320,
                                                overlap_height_ratio=0.2,
                                                overlap_width_ratio=0.2,
                                                perform_standard_pred=False,
                                                verbose=2)
    
    # Function to convert a list of ObjectPredictions to a Geopandas dataframe 
    # From mayrajeo/ship-detection/src/sahi_utils.py 
    def georef_sahi_preds(preds, path_to_ref_img, result_type='bbox') -> gpd.GeoDataFrame:
        "Converts a list of `ObjectPredictions` to a geodataframe, georeferenced according to reference image"
        labels = [p.category.id for p in preds]
    
        if result_type == 'bbox': 
            polys = [box(*p.bbox.to_xyxy()) for p in preds]
        elif result_type == 'mask': 
            polys = []
            for p in preds:
                segmentation = p.mask.segmentation
                temp_polys = []
                for segm in segmentation:
                    xy_coords = [(segm[i], segm[i+1]) for i in range(0, len(segm), 2)]
                    xy_coords.append(xy_coords[-1])
                    temp_polys.append(Polygon(xy_coords))
                polys.append(MultiPolygon(temp_polys))
        else:
            print(f'Unknown result type {result_type}, defaulting to bbox')
            polys = [box(*p.bbox.to_xyxy()) for p in preds]
        scores = [p.score.value for p in preds]
        gdf = gpd.GeoDataFrame({'label':labels, 'geometry':polys, 'score':scores})
        tfmd_gdf = georegister_px_df(gdf, path_to_ref_img)
        print(tfmd_gdf)
        return tfmd_gdf

    ##
    # Convert the prediction objects to a GeoDataFrame  
    # by georeferencing the predictions using the reference image and the list of the predictions.
    boats_gdf = georef_sahi_preds(preds=sliced_pred_results.object_prediction_list, path_to_ref_img=image)
    # Add label for the predictions                            
    boats_gdf['type'] = 'boat'
    
    # Extract name of the tile
    tile_fn = image.split('/')[-1].split('.')[0]
    
    print(f'Found {len(boats_gdf)} objects.')
    # Save georeferenced objects to a GeoJSON for visualization
    boats_gdf.to_file(os.path.join(outpath, f'{tile_fn}.geojson'), driver='GeoJSON')


    ##
    # Plot predictions
    fig, ax = plt.subplots(dpi=100, figsize=(30, 22))
    
    # Plot test image
    with rio.open(image) as src:
        rioplot.show(src, ax=ax)

    # Plot predictions
    boats_gdf.plot(ax=ax, column='type', linewidth=1.5, facecolor='none', legend=True)
    ax.set_title('Predicted vessels')
    
    # Save plot to a file
    filepath = os.path.join(outpath, f"{tile_fn}_predictions.png")
    plt.savefig(filepath)

    
    ##
    # Compute perfomance metrics and save as a JSON for later inspection
    results = model.model.val(data=data_path, split='test')
    metrics = results.results_dict
    
    with open("metrics.json", "w") as f:
     json.dump(metrics, f, indent=4)


if __name__ == '__main__':
    ### This part just runs the main method and times it
    start = time.time()
    main()
    end = time.time()
    print("Script completed in " + str(round(((end - start)/60),0)) + " minutes") 

