# Optional tools
## Widely used general ML tools

* [PyTorch](https://pytorch.org/), deep learning framework
* [R, caret](http://topepo.github.io/caret/index.html ) - general ML library, similar to scikit-learn. Supports also Keras as back-end for deep learning. Supports parallel computing.
* [Dask-ML](https://ml.dask.org/), scalable machine learning with Scikit-Learn, XGBoost, and others.
   * [Dask+ scikit-learn example](https://examples.dask.org/machine-learning/scale-scikit-learn.html)

## Spatial data specific ML tools
### Python
* [Torchgeo](https://torchgeo.readthedocs.io/)
* [RasterVision](https://docs.rastervision.io/)
* [geoai](https://github.com/opengeos/geoai)
* [segment-geospatial](https://samgeo.gishub.org/)
* More options:
  * [Torchgeo related libraries table](https://torchgeo.readthedocs.io/en/stable/user/alternatives.html)
  * [Awesome-Geospatial listing](https://github.com/sacridini/Awesome-Geospatial#deep-learning)

### R

* [CAST](https://rdrr.io/cran/CAST/src/R/CAST-package.R) - improve spatial-temporal modelling tasks using 'caret'. 
   * OpenGeoHub, [Spatial machine learning for GIS with R course materials](http://www.opengeohub.org/machine-learning-spatial-data)
* [SITS](https://e-sensing.github.io/sitsbook/)

### ArcGIS

* Options: 
   * ArcGIS Pro, very easy to use with default settings / existing models
   * ArcGIS Python API, ArcGIS Notebooks, model traiing, more advanced options.
   * ArcGIS Image server, Trained models at scale in production
* [Shallow learning](https://pro.arcgis.com/en/pro-app/tool-reference/spatial-analyst/an-overview-of-the-segmentation-and-classification-tools.htm ): K-means, SVM, random forest, maximum likelihood  classifications and ISO clustering
* [Deep learning](https://www.esri.com/arcgis-blog/products/api-python/analytics/deep-learning-models-in-arcgis-learn/):  
   * object detection: find bbox of the objects
   * pixel classification
   * object classification: classify features or tiles
   * Based on PyTorch and Keras
* Export Training Data For Deep Learning, 5 different formats, “Classified tiles” similar to our exercise. Very easy to use, but clearly slower than GDAL
* Training serious models requires GPU, either power-PC or GPU server, not suitable for CSC GPU resources:
   * ArcGIS Pro only as Windows software
   * ArcGIS Python API could be installed, but no access to local data

Tip: See [ESRI virtual campus machine learning materials](https://www.esri.com/training/catalog/5eb18cf2a7a78b65b7e26134/deep-learning-using-arcgis/)

### QGIS
Plugins:
* [Semi-Automatic Classification Plugin](https://fromgistors.blogspot.com/p/semi-automatic-classification-plugin.html), allows for the supervised classification of remote sensing images, providing tools for the download, the preprocessing and postprocessing of images.
https://fromgistors.blogspot.com/ 
* A few new / little used ones, also for data preparation: [machine-learning](https://plugins.qgis.org/plugins/tags/machine-learning/), [deep-learning](https://plugins.qgis.org/plugins/tags/deep-learning/)

### Other

* Point cloud voxelizer example: https://github.com/Eayrey/3D-Convolutional-Neural-Networks-with-LiDAR/blob/master/point_cloud_voxelizer.py 

## GIS ML tools in CSC Puhti supercomputer

* [geoconda module](https://docs.csc.fi/apps/geoconda/): scikit-learn + a lot of Python GIS packages
* [pytorch module](https://docs.csc.fi/apps/pytorch/): pytorch, geopandas, rasterio
* [tensorflow module](https://docs.csc.fi/apps/tensorflow/): keras, tensorflow, geopandas, rasterio
* [r-env module](https://docs.csc.fi/apps/r-env/): caret, CAST + a lot of R GIS packages

Puhti documentation: 
* [machine-learning](https://docs.csc.fi/apps/#data-analytics-and-machine-learning)
* [Tools for spatial data analysis](https://docs.csc.fi/apps/#geosciences), inc QGIS, GRASS, OrfeoToolbox





