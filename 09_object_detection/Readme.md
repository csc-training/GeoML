# Object detection

(For nicer reading in Jupyter, righ-click and select `Show Markdown Preview`.) 

The goal is to detect marine vessels from the Baltic Sea, based on Sentinel-2 1C RGB data. The model for object detection is a YOLO-11 model. 

This folder includes two exercises:

1. [existing_model](existing_model) - use pre-trained model from Huggingface to predict boats based on an Sentinel-2 image. 
2. [own_model_training](own_model_training) - full workflow for own model training: data preparations, model training, running object detection, saving results in GIS-format, estimating the model.

These exercises heavily rely on the work of [Janne Mäyrä](https://scholar.google.com/citations?user=xAT9080AAAAJ&hl=en) from SYKE (Finnish Environmental institute). He has prepared the used pre-trained model, training data, `geo2ml`-library for data preparation and example code for model training and predicting.

## Data sources
* ESA, [Sentinel-2 1C RGB data](https://sentinels.copernicus.eu/sentinel-data-access/sentinel-products/sentinel-2-data-products/collection-1-level-1c). 
* Janne Mäyrä, SYKE, Dataset for marine vessel detection from Sentinel 2 images in the Finnish coast, downloaded from [Zenodo](https://doi.org/10.5281/zenodo.15019034). For keeping the exercise duration reasonable, we use only part of the training data.

