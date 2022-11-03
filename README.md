# Course material for "Practical machine learning for spatial data" at CSC

## Course goals: 
 * General overview
 * Give vocabulary
 * Practical examples

### Out of scope
 * Feature engineering (geospatial analysis; problem dependent),
 * Result interpretation in depth(problem and model dependent),
 * Working with pointclouds

## Schedule
https://ssl.eventilla.com/event/VDK2b

## Course computing enviroment

First 2 days the exercises are done in "Jupyter notebooks for courses" in the [Puhti webinterface](https://www.puhti.csc.fi) . 
Puhti is one of CSCs supercomputers providing researchers in Finland with High Performance Computing resources.
Course participants will be provided with a temporary account (training_xxx); it is not possible to use your own CSC account. 

### Puhti webinterface
* Open https://www.puhti.csc.fi
* Enter provided username (training_xxx) and password 
    
### Jupyter for courses
* Click "Jupyter for courses" on dashboard
* Select:
   * Project: project_2002044
   * Module: GeoML22
   * Reservation: geoml
   * Working directory: "/scratch/project_2002044"
* Click launch and wait until granted resources
* Click "Connect to Jupyter" 
* ... to be continued within intro.ipynb

### QGIS
* Click "Desktop" on dashboard
* Select:
   * Project: project_2002044
   * Partition: small
   * Reservation: geoml
   * Number of CPU cores: 1
   * Memory (GB): 8 Gb
   * Local disk: 0
   * Time: 4:00:00 (or adjust to reasonable)
   * Desktop: single application
   * App: QGIS
* Click launch and wait until granted resources
* Click "Launch Desktop" 

## Content of this repository

This repository contains all Jupyter Notebooks and other code used in the course. Data is not inlcuded here, data download links are provided in data preparations Notebooks. Each exercise has its own folder:

* [01_clustering](01_clustering)
* [02_vector_data_preparation](02_vector_data_preparation)
* [03_raster_data_preparation](03_raster_data_preparation)
* [04_shallow_regression](04_shallow_regression)
* [05_shallow_classification](05_shallow_classification)
* [06_deep_regression](06_deep_regression)
* [07_deep_classification](07_deep_classification)
* [08_cnn_segmentation](08_cnn_segmentation)

Extra material:
* [Links to further resources](links.md)
* [Optional tools for machine learning with spatial data](tools.md)
* [Links to machine learning with point clouds](point_cloud.md)
* [Links to machine learning with time series](timeseries.md)


## Authors
Kylli Ek, Samantha Wittke, Johannes Nyman, Ziya Yektay
