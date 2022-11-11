# Exercise materials for "Practical machine learning for spatial data" at CSC

## Course goals
 * General overview of machine learning for spatial data
 * Give vocabulary to continue with own experiments
 * Show and run some practical examples

## Out of scope
 * Feature engineering 
 * In depth result interpretation
 * Working with pointclouds

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


## Course exercise enviroment

During the course exercises are done in Puhti, which is one of CSCs supercomputers providing researchers in Finland with High Performance Computing resources.


### Puhti webinterface
* Open https://www.puhti.csc.fi
* Enter [CSC username](https://docs.csc.fi/accounts/) and password 
* For course use temporary training accounts are provided.
    
#### Jupyter for courses (only available during the course)

* Click "Jupyter for courses" on dashboard
* Select:
   * Project: project_2002044 during course, own project later
   * Module: GeoML22
   * Working directory: "/scratch/project_2002044"
* Click launch and wait until granted resources
* Click "Connect to Jupyter" 
* ... to be continued within [Introduction notebook](intro.ipynb)

#### Jupyter 
* Click "Jupyter" on dashboard
* Select following settings:
	* Project: project_2002044 during course, own project later 
	* Partition: interactive
	* CPU cores: 1
	* Memory (Gb): 8 
	* Local disk: 0
	* Time: 4:00:00 (or adjust to reasonable)
	* Python: geoconda OR tensorflow depending on the exercise
		* Exercises 1, 2, 4 and 5: geoconda OR tensorflow
		* Exercises 3 and 8 notebooks: geoconda
		* Exercises 6 and 7: tensorflow
	* Jupyter type: Lab
	* Working directory: /scratch/project_2002044 during course, own project scratch later
* Click launch and wait until granted resources 
* Click "Connect to Jupyter" 

#### QGIS
* Click "Desktop" on dashboard
* Select:
   * Project: project_2002044 during course, own project later
   * Partition: interactive
   * Number of CPU cores: 1
   * Memory (GB): 10
   * Local disk: 0
   * Time: 4:00:00 (or adjust to reasonable)
   * Desktop: single application
   * App: QGIS
* Click launch and wait until granted resources
* Click "Launch Desktop" 

## Exercises on own laptop

Exercises 1-7 could be done on any laptop, required installations are described in [requirements.txt](requirements.txt), required data is downloaded in Notebooks. For exercise 8 (CNN) GPU resources are good to have, the batch jobs here are suitable for Puhti only.

Extra steps for own laptop
* Download the exercise material from Github
	* Clone this Github repository: `git clone https://github.com/csc-training/GeoML.git` 
	* OR if unfamiliar with Github, download the files as a [zip-file](https://github.com/csc-training/GeoML/archive/refs/heads/main.zip)
* Change the main path in beginning of each notebook.

## Extra material

* [Links to further resources](links.md)
* [Optional tools for machine learning with spatial data](tools.md)
* [Hints for machine learning with point clouds; Work in progress](point_cloud.md)
* [Hints for machine learning with time series; Work in progress](timeseries.md)


## Authors
Kylli Ek, Samantha Wittke, Johannes Nyman, Ziya Yektay

## Acknowledgement

Please acknowledge CSC and Geoportti in your publications, it is important for project continuation and funding reports. As an example, you can write "The authors wish to thank CSC - IT Center for Science, Finland (urn:nbn:fi:research-infras-2016072531) and the Open Geospatial Information Infrastructure for Research (Geoportti, urn:nbn:fi:research-infras-2016072513) for computational resources and support".
