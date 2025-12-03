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
During the course exercises are done in LUMI, which is EuroHPC supercomputer. Accessing LUMi requires LUMI project. Finnish users get access to LUMI via CSC. For course the course participants are added to the course project.

### LUMI webinterface
* Open https://www.lumi.csc.fi
* Log in with:
	* HAKA, if you have (Finnish universities and some research institutes)
 	* [CSC account](https://docs.csc.fi/accounts/), you need your CSC username and password 

### Copy exercise materials
Open Login node shell
```
cd /scratch/project_462001167/students/
mkdir $USER
cd $USER
git clone https://github.com/csc-training/GeoML.git
```
	
#### Jupyter 
* Click "Jupyter" on dashboard
* Select following settings:
	* Project: project_462001167 during course, own project later 
	* Partition: interactive
	* CPU cores: 4
	* Local disk: 0
	* Time: 4:00:00 (or adjust to reasonable)
 	* Working directory: /scratch/project_462001167 during course, own project scratch later* 
	* Python: geoconda OR custom depending on the exercise
		* Exercises 1 - 6, 9 data preparation: geoconda
  			* No virtual environment  	
		* Exercises 7 - 9: pytorch
  			* Before opening Jupyter the first time, you need to create virtual environment with some extra packages, see below.
      		* Check, `Enable virtual environment`
			* Virtual environment path: `/projappl/project_462001167/students/$USER/geoml`
   			* Check, `Enable packages under ~/.local/lib on venv start`
   	* (Do not select any of the check-boxes below.)
	
* Click launch and wait until granted resources 
* Click "Connect to Jupyter"
* Open Terminal and clone exercise materials
* Open in JupyterLab folder `students/<your_username>/GeoML`

#### Adding deep learning librares to pytorch module
The Pytorch module does not include all Python packages required by these exercises. To add custom packages, the best option is to use [venv](https://docs.csc.fi/support/tutorials/python-usage-guide/#using-venv) (virtual environment).

Open Login node shell and add the venv to `projappl`:
```
cd /projappl/project_462001167/students/
mkdir $USER
cd $USER
module use /appl/local/csc/modulefiles/
ml pytorch
python3 -m venv --system-site-packages geoml
source geoml/bin/activate
pip install torchgeo # CNN exercise
pip install sahi ultralytics folium==0.13 # Object detection exercise
pip install segment-geospatial[samgeo] addict yapf pycocotools supervision groundingdino-py # SAM
```

#### Optional, QGIS
[CSC Dosc: QGIS](https://docs.csc.fi/apps/qgis/)

## Exercises on own computer

Exercises 1-7 Jupyter notebooks can be run as is on any computer. Exercises 8 - 9 (CNN and object detection), batch job scripts are supercomputer (LUMI, Puhti etc) specific as GPU resources are good to have for the exercise to run in reasonable time. However, the Python scripts can also be run on your own computer with some path adjustments.

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
Iida Hakulinen, Kylli Ek, Samantha Wittke, Johannes Nyman

## Acknowledgement

These materials have been developed with Location Innovation Hub (LIH) and Geoportti funding.

Please acknowledge CSC, Location Innovation Hub (LIH) and Geoportti in your publications, it is important for project continuation and funding reports. As an example, you can write "The authors wish to thank CSC - IT Center for Science, Finland (urn:nbn:fi:research-infras-2016072531), Location Innovation Hub and the Open Geospatial Information Infrastructure for Research (Geoportti, urn:nbn:fi:research-infras-2016072513) for computational resources and support".
