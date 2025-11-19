# Exercise materials for "Practical machine learning for spatial data" at CSC

## Course goals
 * General overview of machine learning for spatial data
 * Give vocabulary to continue with own experiments
 * Learn to train GeoAI models at practical level

## Out of scope
 * Feature engineering 
 * In depth result interpretation
 * Working with vector geometries, point clouds and time-series data.

## Content of this repository

This repository contains all Jupyter Notebooks and other code used in the course. Data is not inlcuded here, data download links are provided in data preparations Notebooks. Each exercise has its own folder:

* [01_vector_data_preparation](01_vector_data_preparation)
* [02_raster_data_preparation](02_raster_data_preparation)
* [03_shallow_regression](03_shallow_regression)
* [04_shallow_classification](04_shallow_classification)
* [05_deep_regression](06_deep_regression)
* [06_deep_classification](06_deep_classification)
* [07_samgeo](07_samgeo)
* [08_cnn_segmentation](08_cnn_segmentation)
* [09_object_detection](09_object_detection)


## Course exercise enviroment
During the course exercises are done in LUMI, which is EuroHPC supercomputer. Accessing LUMi requires LUMI project. Finnish users get access to LUMI via CSC. For course the course participants are added to course project.

### LUMI webinterface
* Open https://www.lumi.csc.fi
* Enter [CSC username](https://docs.csc.fi/accounts/) and password 
* For course use temporary training accounts are provided.
    
#### Jupyter for courses (only available during the course)

* Click "Jupyter for courses" on dashboard
* Select:
   * Project: project_462001167 during course, own project later
   * Module: GeoML22
   * Working directory: "/scratch/project_2002044"
* Click launch and wait until granted resources
* Click "Connect to Jupyter" 
* ... to be continued within [Introduction notebook](intro.ipynb)

#### Jupyter 
* Click "Jupyter" on dashboard
* Select following settings:
	* Project: project_462001167 during course, own project later 
	* Partition: interactive
	* CPU cores: 1
	* Memory (Gb): 8 
	* Local disk: 0
	* Time: 4:00:00 (or adjust to reasonable)
	* Python: geoconda OR pytorch depending on the exercise
		* Exercises 1 - 6, 8 data preparation: geoconda
		* Exercises 7 - 9: pytorch
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

## Exercises on own computer

Exercises 1-7 Jupyter notebooks can be run as is on any computer. Exercise 8 (CNN), batch job scripts are Puhti specific as GPU resources are good to have for the exercise to run in reasonable time. However, the Python scripts can also be run on your own computer with some path adjustments.

To get started:
* Get the exercise material from Github
	* Clone this Github repository: `git clone https://github.com/csc-training/GeoML.git` 
	* OR download the repository as a [zip-file](https://github.com/csc-training/GeoML/archive/refs/heads/main.zip)
* Install all needed packages for running the notebooks:
	* For pip use the [requirements.txt](requirements.txt) with `pip install -e requirements.txt`
	* OR for conda, use the [environment.yml](environment.yml) with `conda create --name geoml --file environment.yml` which also creates a conda environment; see [conda homepage](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html#managing-envs) on how to use it). 
	* Package versions in comments in these files are versions used for GeoML course 2022 on Puhti.
* Adapt the main path in beginning of each notebook to your environment.
* Have fun going through the notebooks and add an issue to this repository if something is not working.

## Extra material

* [Links to further resources](links.md)
* [Optional tools for machine learning with spatial data](tools.md)


## Authors
Iida Hakulinen, Kylli Ek, Samantha Wittke, Johannes Nyman, Ziya Yektay

## Acknowledgement

Please acknowledge CSC and Geoportti in your publications, it is important for project continuation and funding reports. As an example, you can write "The authors wish to thank CSC - IT Center for Science, Finland (urn:nbn:fi:research-infras-2016072531) and the Open Geospatial Information Infrastructure for Research (Geoportti, urn:nbn:fi:research-infras-2016072513) for computational resources and support".
