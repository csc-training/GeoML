# Semantic segmentation exercise

(For nicer reading in Jupyter, righ-click and select `Show Markdown Preview`.) 

In this exercise, the land use classes are predicted with semantic segmentation using 2 models:

* 7A: own CNN model from scratch. 
* 7B: fine-tuning of [Clay foundation](https://clay-foundation.github.io/model/) model. 

Main libraries are: [torchgeo](https://torchgeo.readthedocs.io/), [PyTorch](https://pytorch.org/) and [Lightning](https://lightning.ai/docs/overview/getting-started), for fine-tuning exercise also [Terratorch](https://torchgeo.org/terratorch/stable/).

## Input data

The used data is similar to the data in shallow and deep classification exercises, but the files are different. For these exercise there is separate files for training, validation and test and the covered area is bigger than for previous exercises. The test file is the same as for 

6 raster files (3 for labels and 3 for data) with:

* Coordinate system: Finnish ETRS-TM35FIN, EPSG:3067
* Resolution: 20m

### Labels

Multiclass classification raster: 
* 1 - forest
* 2 - fields
* 3 - water
* 0 - everything else

### Data 

**Sentinel2 mosaic**
* Date: 2021-05-22- 2021-05-31
* 10 bands: 'b02', 'b03', 'b04', 'b05', 'b06', 'b07', 'b08', 'b8a', 'b11', 'b12'.
* The reflection values scaled to [0 ... 1].

If you do this exercise outside CSC course, the general [raster data preparations exercise](../02_raster_data_preparation) must be done.

### Tiling

Satellite images are usually too big for CNN models as such, se we need to tile them to smaller tiles for training the model and also later for prediction. Torchgeo has very nice functionality for tiling and sampling the data for training. Unfortunatelly similar functionality does not exist for inference.

## Main steps

Both 7A and 7B exercises include three steps:
* Model training, including data loading and tiling with torchgeo. 
* Predicting the classification and class-wise probabilites.
* Evaluation of the model visually and by calculating performance metrics.

The first 2 steps are run as batch job, because GPU-resources are needed.
Read the `Readme`-files in both sub-directories for detailed instructions how to run the exercises.


    
