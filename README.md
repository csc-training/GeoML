# Course material for "Practical machine learning for spatial data" at CSC

# Course introduction

## Welcome 

* Introduction round; course expectations
    * in class: knowledge level geospatial data analysis, vector, raster, ML, DL

## Important links


**This document:** https://github.com/csc-training/GeoML/README.md

**[Slides](https://drive.google.com/drive/folders/1q0-eSCFKcApzTql828Z2ZfDe8xFeFjXd?usp=sharing)**

**[Exercise material](https://github.com/csc-training/GeoML)**

**Computing environment:** https://puhti.csc.fi

## Practicalities Dogmi

* Visitor badge
* parking
* WC
* list on wall
* Wifi/Ethernet cable
    * eduroam 
        * temporary access via SMS 
    * grey ethernet cable
* Dogmi computers, CentOS, web browser, QGIS
    *** DO NOT RESTART!**

## Code of conduct

We strive to follow the [Code of Conduct developed by The Carpentries organisation](https://docs.carpentries.org/topic_folders/policies/code-of-conduct.html) to foster a welcoming environment for everyone. In short:
- Use welcoming and inclusive language
- Be respectful of different viewpoints and experiences
- Gracefully accept constructive criticism
- Focus on what is best for the community
- Show courtesy and respect towards other community members

And please let us know if there is any accessibility issues; fontsize, sound, etc.
Terminology can be confusing , e.g. Cartographic label != ML label, please ask if unclear :)
-> **How to ask questions:** by hand-raising (physically/in zoom) and ask by voice (we have a backup solution in case there is too many questions)

## This course

### Course goals: 
    * general overview
    * give vocabulary
    * practical examples

### Out of scope
    * feature engineering (geospatial analysis; problem dependent),
    * result interpretation in depth(problem and model dependent),
    * working with pointclouds

### Schedule

**Monday 7.11.**
9:00-12:00
* Practicalities and Introduction 
* Lecture 1: Introduction to machine learning
* Exercise 1: Image segmentation using k-means with scikit-learn

12.00-13:00 - Lunch break

13:00-16:15
* Lecture 2: Shallow machine learning models
* Lecture 3: Preparing spatial data for machine learning
* Exercise 2: Preparing vector data for regression 

**Tuesday 8.11.**
9:00-12:00
* Exercise 3: Shallow regression with scikit-learn
* Exercise 4: Image classification using shallow classifiers, grid search with scikit-learn
* Lecture 4: Introduction to deep learning models

12.00-13:00 - Lunch break

13:00-16:15
* Lecture 5: Fully connected neural networks
* Lecture 6: Puhti GPUs and batch jobs
* Exercise 5: Fully connected regressor with keras

**Wednesday 9.11.**
9:00-12:00
* Exercise 6: Fully connected classifier with keras
* Lecture 7: Convolutional neural networks (CNN)
* Exercise 7: Data preparations for CNN

12.00-13:00 - Lunch break

13:00-16:00
* Exercise 8: CNN based image segmentation with keras
* Lecture 8: GIS software supporting machine learning for spatial data. 
* Wrap-up and where to go from here

## Course computing enviroment

Almost all exercises will be done in "Jupyter notebooks for courses" in the [Puhti webinterface](https://www.puhti.csc.fi) .
Puhti is one of CSCs supercomputers providing researchers in Finland with High Performance Computing resources.
Course participants will be provided with a temporary account (training_xxx); it is not possible to use your own CSC account. 

* Puhti webinterface & accounts
    1. connect to https://www.puhti.csc.fi
    2. enter provided username (training_xxx) and password 
    3. find and click "Jupyter for courses" on dashboard
    4. choose Project "project_2002044" first!
    5. then choose Module "GeoML22"
    6. and set workingdirectory to "/scratch/project_2002044"
    7. click launch and wait until granted resources
    8. then click "Connect to Jupyter" 

We will use the Jupyter for courses environment on Day 1 and 2, on day 3 we will use Jupyter and the batch job submission system "the normal way" (the same way that you can also use Puhti after the course).

... to be continued within intro.ipynb

## Authors
Kylli Ek, Samantha Wittke, Johannes Nyman

## Content of this repository

This repository contains all Jupyter notebooks used in the course.

**01_clustering**
**02_vector_data_preparation**
* Preparation of Paavo zip code dataset for machine learning
* Input:
    * Paavo zip code dataset
    * counties dataset
* Output:
    * scaled and unscaled train, test validation datasets and labels
**03_raster_data_preparation**
**04_shallow_regression**
* Runnning some shallow regression models on Paavo zip code dataset
* Input:
    *  scaled and unscaled train, test validation datasets and labels
* Output:
    * Regression error metrics for validation dataset
**05_shallow_classification**
**06_deep_regression**
* Building and Runnning some deep regression models on Paavo zip code dataset
* Input:
    *  scaled train, test validation datasets and labels
**07_deep_classification**
**08_cnn_segmentation** 
