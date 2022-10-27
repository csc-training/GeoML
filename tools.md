# Optional tools
## Widely used mainstream general ML tools

* [R, caret](http://topepo.github.io/caret/index.html ) - general ML library, similar to scikit-learn. Supports also Keras as back-end for deep learning. Supports parallel computing.
* [PyTorch](https://pytorch.org/), deep learning framework
* [Dask-ML](https://ml.dask.org/), scalable machine learning with Scikit-Learn, XGBoost, and others.
   * [Dask+ scikit-learn example](https://examples.dask.org/machine-learning/scale-scikit-learn.html)

## Spatial data specific ML tools
### R

* [CAST](https://rdrr.io/cran/CAST/src/R/CAST-package.R) - improve spatial-temporal modelling tasks using 'caret'. 
   * OpenGeoHub, [Spatial machine learning for GIS with R course materials](http://www.opengeohub.org/machine-learning-spatial-data)

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
* [ERDAS imagine](https://bynder.hexagon.com/m/5d441e34a685b634/original/Hexagon_GSP_Machine_-Learning_Deep_Learning_white_paper.pdf)
* [Google Earth Engine](https://developers.google.com/earth-engine/guides/machine-learning)
* GRASS, [r.learn.ml](https://grass.osgeo.org/grass82/manuals/addons/r.learn.ml.html), supervised classification and regression of GRASS rasters using the python scikit-learn package
* [McFly](https://blog.esciencecenter.nl/mcfly-an-easy-to-use-tool-for-deep-learning-for-time-series-classification-b2ee6b9419c2), build and compare different DL models for timeseries classification task
* [PyspatialML](https://github.com/stevenpawley/Pyspatialml), for applying scikit-learn machine learning models to 'stacks' of raster datasets. 
* [Orfeo Toolbox](https://www.orfeo-toolbox.org/CookBook/Applications/Learning.html). Looks good from documentation, but in practice had several problems...
* [OTBTF: Orfeo ToolBox meets TensorFlow](https://github.com/remicres/otbtf)
* [Torchgeo](https://www.microsoft.com/en-us/research/publication/torchgeo-deep-learning-with-geospatial-data/)
* More links: https://github.com/sacridini/Awesome-Geospatial#deep-learning

## GIS ML tools in CSC Puhti HPC

* geoconda: scikit-learn + a lot of Python GIS packages
* tensorflow: keras, tensorflow, geopandas, rasterio
* pytorch: pytorch, geopandas, rasterio
* r-env: caret, CAST + a lot of Python GIS packages
* GRASS
* QGIS 
* OrteoToolBox 

Puhti documentation: 
* [machine-learning](https://docs.csc.fi/apps/#data-analytics-and-machine-learning)
* [tools for spatial data](https://docs.csc.fi/apps/#geosciences)





