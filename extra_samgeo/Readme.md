# Segment Anything Model (SAM)
How to run the notebook in Roihu?

## Install `samgeo`
The packages required by this notebook are not installed to any of the existing modules. 
So before opening Jupyter and running the notebook, some installations are needed.

Many of the required libraries are included to `python-pytorch` modules, so we add the missing ones to that. 
`python-pytorch` is available as GPU and CPU version, for this notebook the CPU version suits better, 
but for running bigger areas the GPU version should be preferred.

Open Login-node shell (Roihu-CPU) under Apps in Roihu web interface.

```
# Load the starting poin module
module load python-pytorch

# Create folder to store the installation files
mkdir -p /projappl/project_2019932/students/$USER/samgeo-install-cpu

# Add this folder to Python, for user installations
export PYTHONUSERBASE=/projappl/project_2019932/students/$USER/samgeo-install-cpu

# Install samgeo
pip install --user segment-geospatial
```

## Add the installed packages to Jupyter notebook

* Click "Jupyter" on dashboard
* Select following settings:
	* Reservation: select the reservation during the course 
	* Project: project_2019932 during course, own project later 
	* Partition: interactive
	* CPU cores: 1
 	* Memory: 10 GiB
	* Time: 0:30:00 (or adjust to reasonable)
	* **Python: `python-pytorch`**
 	* Working directory: /scratch/project_2019932 during course, own project's scratch later* 
  * **Enable advanced settings**
  * Select `User packages`
    * **User packages path: /projappl/project_2019932/students/$USER/samgeo-install-cpuy**
	
* Click launch and wait until granted resources 
* Click "Connect to Jupyter"
