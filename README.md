# Exercise materials for "Practical machine learning for spatial data" at CSC
## Content of this repository

This repository contains all Jupyter Notebooks and other code used in the course. Data is not inlcuded here, data download links are provided in data preparations Notebooks. Each exercise has its own folder:

* [01_vector_data_preparation](01_vector_data_preparation)
* [02_raster_data_preparation](02_raster_data_preparation)
* [03_shallow_regression](03_shallow_regression)
* [04_shallow_classification](04_shallow_classification)
* [05_deep_regression](06_deep_regression)
* [06_deep_classification](06_deep_classification)
* [07_cnn_segmentation](07_cnn_segmentation)
* [08_object_detection](08_object_detection)


## Course exercise enviroment
During the course exercises are done in Roihu, which is CSC supercomputer. Accessing Roihu requires CSC project with Roihu service enabled. Finnish academic users get access to Roihu via CSC. For course the course participants are added to the course project.

### Roihu webinterface
* Open https://roihu.csc.fi
* Log in with any of the following:
 	* [CSC account](https://docs.csc.fi/accounts/), you need your CSC username and password 
	* HAKA (Finnish universities and some research institutes)
 	* Virtu (Finnish governmental organizatons)
* You need to use MFA (a code from phone to log in)   
	
#### Jupyter 
* Click "Jupyter" on dashboard
* Select following settings:
	* Project: project_2019932 during course, own project later 
	* Partition: interactive
	* CPU cores: 1
 	* Memory: 8 GiB
	* Time: 4:00:00 (or adjust to reasonable)
	* Python: `python-geo`
 		* `python-pytorch` for Exercise 8, object detection 
 	* Working directory: /scratch/project_2019932 during course, own project's scratch later* 
   	* (Do not select any of the check-boxes below.)
	
* Click launch and wait until granted resources 
* Click "Connect to Jupyter"
* Open Terminal and clone exercise materials
```
cd /scratch/project_2019932/students/
mkdir $USER
cd $USER
git clone https://github.com/csc-training/GeoML.git
```
* Open in JupyterLab folder `students/<your_username>/GeoML`

#### Optional, QGIS
[CSC Dosc: QGIS](https://docs.csc.fi/apps/qgis/)

## Exercises on own computer

Exercises 1-6 Jupyter notebooks can be run as is on any computer. Exercises 7- 8 (CNN and object detection) require GPU availability for execution in reasonable time. 

To get started:
* Get the exercise materials from Github
	* If you have `git` intalled: `git clone https://github.com/csc-training/GeoML.git` 
	* OR download the repository as a [zip-file](https://github.com/csc-training/GeoML/archive/refs/heads/main.zip)
* Install all needed packages for running the notebooks:
	* Install [mini-conda](https://conda-forge.org/download/) or some other tool supporting conda .yml files.
 	* Open Miniforge promt
  	* Go to the folder where you saved the downloaded Github materials
  	* Create new conda environment based on the `environmet.yml`
  		* `conda env create --name geo-ml --file environment.yml` 
		* See [conda docs, envs](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html#managing-envs) for more information.
* Activate the created environment:
	* `conda activate geo-ml`
* Launch JupyterLab:
	* `jupyter lab` 
* Adapt the main path in beginning of each notebook to your environment.
	* On Windows, it would look something like: `base_directory= os.path.join('C:\\temp\\geoml-course','GeoML')`
* Have fun going through the notebooks and add an issue to this repository if something is not working.

## Extra material

* [Links to further resources](links.md)
* [Optional tools for machine learning with spatial data](tools.md)


## Authors
Iida Hakulinen, Kylli Ek, Samantha Wittke, Johannes Nyman

## Acknowledgement

These materials have been developed with Location Innovation Hub (LIH) and Geoportti funding.

Please acknowledge CSC, Location Innovation Hub (LIH) and Geoportti in your publications, it is important for project continuation and funding reports. As an example, you can write "The authors wish to thank CSC - IT Center for Science, Finland (urn:nbn:fi:research-infras-2016072531), Location Innovation Hub and the Open Geospatial Information Infrastructure for Research (Geoportti, urn:nbn:fi:research-infras-2016072513) for computational resources and support".
