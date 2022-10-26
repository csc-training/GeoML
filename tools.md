# Tools

* scikit-learn with Dask → https://examples.dask.org/machine-learning/scale-scikit-learn.html
* Torchgeo:  https://www.microsoft.com/en-us/research/publication/torchgeo-deep-learning-with-geospatial-data/
* PyspatialML: https://github.com/stevenpawley/Pyspatialml
* McFly, build and compare different DL models for timeseries classification task: https://blog.esciencecenter.nl/mcfly-an-easy-to-use-tool-for-deep-learning-for-time-series-classification-b2ee6b9419c2
* ArcGIS (not in CSC environments!)
* Google Earth Engine
* ERDAS IMAGINE (not in CSC environments!)
* https://github.com/sacridini/Awesome-Geospatial#deep-learning
* available on Puhti:
    * Orfeo (standalone, QGIS → https://www.orfeo-toolbox.org/features-2/)
    * GRASS (r.learn.ml - Supervised classification and regression of GRASS rasters using the python scikit-learn package →> https://grass.osgeo.org/grass82/manuals/addons/r.learn.ml.html)
    * QGIS (SCP: The Semi-Automatic Classification Plugin (SCP) allows for the supervised classification of remote sensing images, providing tools for the download, the preprocessing and postprocessing of images. → https://fromgistors.blogspot.com/p/semi-automatic-classification-plugin.html

## R - caret and CAST

caret - general ML library, similar to scikit-learn. 
http://topepo.github.io/caret/index.html 
Supports also Keras as back-end for deep learning
Supports parallel computing
CAST - support to run 'caret' with spatial or spatial-temporal data. 
R is in Puhti: https://docs.csc.fi/apps/r-env/ 
Has also GIS-packages: https://docs.csc.fi/apps/r-env-for-gis/ 
Machine learning for GIS with R: http://www.opengeohub.org/machine-learning-spatial-data 


## Mainstream general ML tools
Python/scikit-learn and R/caret for shallow learning
Keras+Tensorflow or PyTorch for deep learning
(+ several others...)

## GIS specific ML tools
ESRI: ArcGIS Pro and ArcGIS Python API
Orfeo Toolbox (QGIS) + OTBTF
GRASS
eo-learn


## ArcGIS

ArcGIS Python API, ArcGIS Pro, ArcGIS Notebooks or ArcGIS Image server
Shallow learning: K-means, SVM, random forest, maximum likelihood  classifications and ISO clustering
https://pro.arcgis.com/en/pro-app/tool-reference/spatial-analyst/an-overview-of-the-segmentation-and-classification-tools.htm 

Deep learning: 
object detection: find bbox of the objects
pixel classification
object classification: classify features or tiles
Based on PyTorch and Keras
https://www.esri.com/arcgis-blog/products/api-python/analytics/deep-learning-models-in-arcgis-learn/ 

Tip: See ESRI virtual campus machine learning materials

Export Training Data For Deep Learning
5 different formats, “Classified tiles” similar to our exercise
Very easy to use, but clearly slower than GDAL

Very easy to use with default settings / existing models, ArcGIS Python API enables enables more advanced options.
Trained models can be used in ArcGIS Server at scale in production

Training serious models requires GPU, either power-PC or GPU server
Not suitable for CSC GPU resources:
ArcGIS Pro only as Windows software
ArcGIS Python API could be installed, but no access to local data

## QGIS

Orfeo Toolbox
Plugins: 
Semi-Automatic Classification Plugin: 
https://fromgistors.blogspot.com/ 
A few new / little used ones, also for data preparations:
https://plugins.qgis.org/plugins/tags/machine-learning/
https://plugins.qgis.org/plugins/tags/deep-learning/ 

## Orfeo Toolbox

Documentation: https://www.orfeo-toolbox.org/CookBook/Applications/Learning.html
Demonstartion video of landuse classification: https://www.youtube.com/watch?v=emoGMibsgv0 
Object-based classification tutorial - segmentation, feature extraction with zonal statistics, SVM training, classification: http://wiki.awf.forst.uni-goettingen.de/wiki/index.php/Object-based_classification_(Tutorial) 
FOSS4G 2019 OTB workshop guideline: https://github.com/savmickael/foss4G-2019/raw/master/FOSS4G-2019_SNAP%26OTB-Crop-Classification.pdf and materials: https://drive.google.com/open?id=1lWcRXMhNegYFs3Cpc2ULsHHFeYy5lZHm

For raster and vector
Classifiers and regressions
Shallow learning models: SVM, rf, boost, decision tree, normal bayes, KNN, k-means
Fully connected deep learning models
Based on LibSVM, OpenCV and Shark ML
Available from CLI, Mapla GUI, QGIS or Python
In a way easy to use
Very easy to install
Orfeo Toolbox is in Puhti

One OTB function wraps the whole process of preparing data and training, so: 
No possibility to repeat only parts. For example, training forest stands with random forest, took 2h:50min, of which 2:40 was preparing data. If interested in training again with different settings, have to start from scratch. 
Hard to understand what exactly is happening.
Classification requires polygons also for the “No”-class (= not forest), which has to be computed first. Slow.

Regression requires integer values for labels.
QGIS plug-in can do the job, but has some bugs, and for example does not offer column names as selectable lists. 
Mapla has a lot of options to select, so if interested in repeating training with different options easier to use the OTB CLI.

## Puhti GIS ML

geoconda: scikit-learn + a lot of Python GIS packages
tensorflow: keras, tensorflow, geopandas, rasterio
pytorch: pytorch, geopandas, rasterio
r-env: caret, CAST, sf, terra other R GIS packages
OrteoToolBox
grass

https://docs.csc.fi/apps/alpha/ 

### Acknowledgement

Please acknowledge CSC and Geoportti in your publications, 
it is important for project continuation and funding reports. 
As an example, you can write 

"The authors wish to thank CSC - IT Center for Science, Finland (urn:nbn:fi:research-infras-2016072531) and the Open Geospatial Information Infrastructure for Research (Geoportti, urn:nbn:fi:research-infras-2016072513) for computational resources and support".

Available also on docs.csc.fi application pages.
