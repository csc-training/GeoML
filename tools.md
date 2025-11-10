# Optional tools
## General ML tools
* [R, caret](http://topepo.github.io/caret/index.html ) - general ML library, similar to scikit-learn. Supports also Keras as back-end for deep learning. Supports parallel computing.
* [Dask-ML](https://ml.dask.org/), scalable machine learning with Scikit-Learn, XGBoost, and others.
   * [Dask+ scikit-learn example](https://examples.dask.org/machine-learning/scale-scikit-learn.html)

## Spatial data specific ML tools
### Python
* [geoai](https://github.com/opengeos/geoai) - very nice documentation, nice short examples, but some settings are hard-coded and can not be adjusted
* Tree segmentation:
  * [Detectree2](https://github.com/PatBall1/detectree2)
  * [SegmentAnyTree](https://github.com/SmartForest-no/SegmentAnyTree)
* Listings of with more options:
  * [Torchgeo related libraries table](https://torchgeo.readthedocs.io/en/stable/user/alternatives.html)
  * [Awesome-Geospatial listing](https://github.com/sacridini/Awesome-Geospatial#deep-learning)

### R

* [CAST](https://rdrr.io/cran/CAST/src/R/CAST-package.R) - improve spatial-temporal modelling tasks using 'caret'. 
   * OpenGeoHub, [Spatial machine learning for GIS with R course materials](http://www.opengeohub.org/machine-learning-spatial-data)
* [SITS](https://e-sensing.github.io/sitsbook/) - time-series analysis, inc deep learning

### ArcGIS

* Options: 
   * ArcGIS Pro, training and used ML/DL modesls, very easy to use (toolboxes, no coding) 
   * ArcGIS Python API, ArcGIS Notebooks, model training, more advanced options.
   * ArcGIS Image server, using trained models at scale in production
* [Segmentation and Classification toolset for shallow learning](https://pro.arcgis.com/en/pro-app/tool-reference/spatial-analyst/an-overview-of-the-segmentation-and-classification-tools.htm ): K-means, SVM, random forest, maximum likelihood  classifications and ISO clustering
* [Deep Learning geoprocessing functions[(https://pro.arcgis.com/en/pro-app/3.3/arcpy/image-analyst/deep-learning-geoprocessing-functions.htm)
  * Tools for classifying pixels and objects, detecting objects.
  * Compute Accuracy
  * [Export Training Data For Deep Learning](https://pro.arcgis.com/en/pro-app/3.3/tool-reference/image-analyst/export-training-data-for-deep-learning.htm), for many different models. 
  * [Train Deep Learning Model](https://pro.arcgis.com/en/pro-app/latest/tool-reference/image-analyst/train-deep-learning-model.htm)
     * Object detection: find bbox of the objects
     * Object tracking from videos
     * Pixel classification
     * Object classification: classify features or tiles
     * Image translation (GAN)
* Training serious models requires Nvidia GPU

Tip: See [ESRI virtual campus machine learning materials](https://www.esri.com/training/catalog/5eb18cf2a7a78b65b7e26134/deep-learning-using-arcgis/)

### QGIS
Plugins:
* [Semi-Automatic Classification Plugin](https://fromgistors.blogspot.com/p/semi-automatic-classification-plugin.html), allows for the supervised classification of remote sensing images, providing tools for the download, the preprocessing and postprocessing of images. Mainly shallow learning models, but also MLP.
* A few new / little used ones, also for data preparation:
  * [machine-learning](https://plugins.qgis.org/plugins/tags/machine-learning/)
  * [deep-learning](https://plugins.qgis.org/plugins/tags/deep-learning/)


## GIS ML tools in CSC Puhti supercomputer

* [geoconda module](https://docs.csc.fi/apps/geoconda/): scikit-learn, dask-ml, ArcGIS Python API + a lot of Python GIS packages
* [pytorch module](https://docs.csc.fi/apps/pytorch/): pytorch, geopandas, rasterio
* [tensorflow module](https://docs.csc.fi/apps/tensorflow/): keras, tensorflow, geopandas, rasterio
* [r-env module](https://docs.csc.fi/apps/r-env/): caret, CAST + a lot of R GIS packages
* Not available in CSC provided modules, but tested/used as user-installations in Puhti:
  * Detectree2
  * SegmentAnyTree
  * SITS: r-sits and pysits
  * ArcGIS Python API with deep learning libraries
  * If interested in using these, ask for help via CSC Servicedesk. All of these had some challenges with installation.
* Not possible:
   * ArcGIS Pro, only for Windows
 
Puhti documentation: 
* [Machine-learning tools](https://docs.csc.fi/apps/#data-analytics-and-machine-learning)
* [Spatial data analysis tools](https://docs.csc.fi/apps/#geosciences), inc QGIS





