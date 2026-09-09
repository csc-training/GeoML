#!/usr/bin/env python
# coding: utf-8
# # CNN inference and evaluation
# 
# This script:
# * Predicts classes based on test data and the trained CNN model.
#     * For prediction the test data is tiled with overlap
#     * Data is fed to GPU in batches for optimal performance
#     * Tile edge predictions get lower importance in the mosaicing, so less model errors are visible in the result.
# * Saves the most likely class and probabilities of all classes as GeoTiff files.
#     * Predicted classes in 'segmentation_results.tif'
#     * Probabilities for each class an pixel in 'segmentation_results_all_classes.tif'
# * Computes the accuracy, IoU and F1 of the predictions by comparing the predictions to the ground truth.

# 
# Classes
# ```
# 1 - forest
# 2 - fields
# 3 - water
# 0 - everything else
# ```

import os
import numpy as np
import math, time
from typing import Optional, Any, Tuple
# Reading and writing raster data
import rasterio
# Torchgeo model
import torch
#from torchgeo.trainers import SemanticSegmentationTask CHANGED
from terratorch.tasks import SemanticSegmentationTask
# Model plotting; CHANGED
# from torchinfo import summary

# ## Settings
# Define folders and files.
base_folder = os.path.join(os.sep, 'scratch', 'project_2019932', 'students', os.environ.get('USER'), 'GeoML_old2') 
exercise_folder = os.path.join(base_folder, '07_cnn_segmentation', '07B_using_foundation_model') 
cnn_test_data_folder = os.path.join(base_folder,'data', 'raster', 'pixel-wise')
data_test = os.path.join(cnn_test_data_folder, 'data_sentinel2.tif')
labels_test = os.path.join(cnn_test_data_folder, 'labels.tif')
results_folder = os.path.join(base_folder, 'classification_results')
prediction_output = os.path.join(results_folder, 'classification_sentinel2_cnn_fm.tif') 
prediction_output_all_classes = os.path.join(results_folder, 'class_probabilities_sentinel2_cnn_fm_all_classes.tif') 

if not os.path.exists(results_folder):
    os.makedirs(results_folder)

# The final name depends on how long the training goes on.
# Check to correct name from `checkpoints` folder, the biggest number is the best checkpoint.
checkpoint_path = os.path.join(exercise_folder, 'checkpoints', 'best_model.ckpt') 

# Settings for prediction
num_classes = 4
TILE_SIZE = 224 # Use the same as for model training, must be smaller than data height/width.
BATCH_SIZE = 8
OVERLAP = 20
NO_OF_BANDS = 10


# ## Tiled inference to predict the classes
# 
# During inference the model should be given similar tiles as during model training, so again the big raster has to be tiled. The prediction quality on tile edges is often weak, so therefore we use overlapping tiles and use predictions the model is more confident of.
# 
# The steps of tiled inference:
# * Calculate importances for each pixel in the tile, the pixels on the edge get lower importance, because usually there the model makes more mistakes.
# * Tile the raster into overlapping tiles.
# * Run inference on each tile. Each pixel gets probability value for each class - how likely this pixel belongs to any of the classes.
#     * Inference is practically run in batches, because so the GPU can be better utilized and the total time of prediction is smaller. 
# * Merge the tiles, keep the estimation with highest probability, counting also with importance (distance to tile edge).
# 
# 
# Calculate importances for each pixel in the tile, the pixels on the edge get lower importance, because usually there the model makes more mistakes. Pixels in the center of the tile have higher importance. This helps with smooth blending at boundaries. Practically only pixels that overlap get reduced importance. 
# 
# Code modified from: https://github.com/opengeos/geoai/blob/main/geoai/train.py

def inference_on_geotiff(
    model: torch.nn.Module,
    data,
    tile_size: int = 512,
    overlap: int = 0,
    batch_size: int = 4,
    num_channels: int = 3,
    device: [torch.device] = "cpu",
    **kwargs: Any,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform inference on a large GeoTIFF using a sliding window approach with improved blending.
    Args:
        model (torch.nn.Module): Trained model for inference.
        data (numpy Array): Data of a GeoTIFF file.
        tile_size (int): Size of sliding window for inference.
        overlap (int): Overlap between adjacent tiles.
        batch_size (int): Batch size for inference.
        num_channels (int): Number of channels to use from the input image.
        device (torch.device, optional): Device to run inference on. If None, uses CUDA if available.
        **kwargs: Additional arguments.
    Returns:
        tuple: Tuple containing output path and inference time in seconds.
    """

    # Create importance matrix for each predicted tile
    h = TILE_SIZE
    w = TILE_SIZE
    y_grid, x_grid = np.mgrid[0:h, 0:w]
    # Calculate distance from each edge
    dist_from_left = x_grid
    dist_from_right = w - x_grid - 1
    dist_from_top = y_grid
    dist_from_bottom = h - y_grid - 1
    # Combine distances (minimum distance to any edge)
    edge_distance = np.minimum.reduce(
        [
            dist_from_left,
            dist_from_right,
            dist_from_top,
            dist_from_bottom,
        ]
    )
    # Convert to weight (higher weight for center pixels)
    # Scale to [-5, 0]
    edge_distance = np.minimum(edge_distance, OVERLAP / 2)
    importance = edge_distance * (5 / edge_distance.max()) - 5
    
    # Set same importances to all bands
    #importances = torch.from_numpy(np.repeat(importance[np.newaxis, :, :], num_classes, axis=0))
    importances = torch.from_numpy(np.repeat(importance[np.newaxis, :, :], num_classes, axis=0)).to(device)  # CHANGED: move to same device as model/data

    # Put model in evaluation mode
    model.to(device)
    model.eval()
    height = data.shape[1]
    width = data.shape[2]
    # Initialize predictions array with very small numbers
    pixel_predictions = torch.full((num_classes, height, width), -float('inf'), device=device)
    # Calculate the number of windows needed to cover the entire image
    steps_y = math.floor((height - overlap) / (tile_size - overlap))
    steps_x = math.floor((width - overlap) / (tile_size - overlap))
    # Ensure we cover the entire image
    last_y = height - tile_size
    last_x = width - tile_size
    total_windows = steps_y * steps_x
    print(
        f"Processing {steps_y * steps_x} tiles with size {tile_size}x{tile_size} and overlap {overlap}..."
    )
    # Process in batches, the calculation goes faster, if data is fed to GPU in batches.
    batch_inputs = []
    batch_positions = []
    batch_count = 0
    # Change data type to Float as required by the model.
    image = data.astype(np.float32) 
    # Convert to tensor
    image_tensor = torch.tensor(image, device=device)
    # Slide window over the image - make sure we cover the entire image
    for i in range(steps_y + 1):  # +1 to ensure we reach the edge
        y = i * (tile_size - overlap)
        y = min(i * (tile_size - overlap), last_y)
        for j in range(steps_x + 1):  # +1 to ensure we reach the edge
            x = j * (tile_size - overlap)
            x = min(j * (tile_size - overlap), last_x)
            # Add to batch
            batch_inputs.append(image_tensor[:, y:y+tile_size, x:x+tile_size])
            # Keep track where each tile is located
            batch_positions.append((y, x))
            batch_count += 1
            # Process batch when it reaches the batch size or at the end
            if batch_count == batch_size or (i == steps_y and j == steps_x):
                batch_inputs_tensor = torch.stack(batch_inputs)
                # Forward pass, give model a batch of data.
                with torch.no_grad():
                    outputs = model(batch_inputs_tensor).output #CHANGED from outputs = model(batch_inputs_tensor)
                # Process each output in the batch.
                for idx, output in enumerate(outputs):
                    y_pos, x_pos, = batch_positions[idx]
                    # Multiply with the importances based on pixel's distance to tile edge.
                    weighted_scores = output + importances #output #+ importances #TODO 
                    # Save predictions for the pixels/classes that have heigher score than previously saved.
                    pixel_predictions[:, y_pos:y_pos+tile_size, x_pos:x_pos+tile_size] = torch.max(pixel_predictions[:, y_pos:y_pos+tile_size, x_pos:x_pos+tile_size], weighted_scores)
                # Reset batch
                batch_inputs = []
                batch_positions = []
                batch_count = 0
    # Calculate most probable class for each pixel
    #class_predictions = pixel_predictions.argmax(dim=0).numpy().astype(np.uint8)

    class_predictions = pixel_predictions.argmax(dim=0).cpu().numpy().astype(np.uint8)   # CHANGED
    pixel_predictions = pixel_predictions.cpu().numpy()   # NEW: also move this to CPU/numpy before returning

    return pixel_predictions, class_predictions

def main():
    # Set computing device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set working directory.
    os.chdir(exercise_folder)
    # ## Model
    # Load the trained model from checkpoint.
    
    model = SemanticSegmentationTask.load_from_checkpoint(checkpoint_path, strict=False)
    # See the model architecture.
    
    #summary(model, input_size=(BATCH_SIZE, NO_OF_BANDS, TILE_SIZE, TILE_SIZE)) DOES NOT WORK with terratorch
    
    # Show model details
    print(model) # CHANGED NEW
    print(model.forward)

    # Read test data from file, calculate predicted classes and save as GeoTiff, save also probabilities of all classes for each pixel (might be interesting to check).
    with rasterio.open(data_test) as src:
        data = src.read()
        pixel_predictions, class_predictions = inference_on_geotiff(model, data, TILE_SIZE, OVERLAP, BATCH_SIZE, NO_OF_BANDS, device)
        # Save predition raster with most likely class
        out_meta = src.meta.copy()
        out_meta.update(
            {"count": 1, "dtype": "uint8"}  # Single band for mask  # Binary mask
        )
        with rasterio.open(prediction_output, "w", **out_meta) as dst:
            dst.write(class_predictions, 1)
        
        # Save predition raster with with probabilities for all classes
        out_meta2 = src.meta.copy()
        out_meta2.update(
            {"count": num_classes} 
        )   
        with rasterio.open(prediction_output_all_classes, "w", **out_meta2) as dst:
            dst.write(pixel_predictions)
        print(f"Saved prediction to {prediction_output}")

if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print("Script completed in " + str(round(((end - start) / 60), 0)) + " minutes")