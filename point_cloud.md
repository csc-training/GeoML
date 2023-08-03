# Point clouds

**This document is work in progress**

## Point cloud characteristics

* Uniform representation - unstructured, unordered set of 3D points
* Discrete representation - discrete samples of shapes without restrictions regarding topology and geometry
* Irregularity - expose irregular spatial distribution and varying spatial density
* Incompleteness - due to discrete sampling, representations are incomplete by nature
* Ambiguity - semantics of a point can generally not be determined without considering its neighborhood
* Per-point attributes - each point can be attributed by additional per-point data, such as color or surface normal
* Massiveness - depending on the capturing technology, 3D point clouds may consist of millions or billions of points

## Applications

* Classification: labels are attached to each point
* Segmentation: based on identifying 3D geometry features such as edges, planar facets or corners
* Object detection: extension of classification to localize objects by bounding box

## Data structure

Structured data: 

* Consistent storage and appearance, 
* Indices can be used to find neighbors; 
* Examples: images, text, tabular data

3D pointclouds are unstructured data (set): 

* Vector data stored in text file
* Storage not consistent with appearance 
* N! possible storage options for N points

> Point cloud stored as pointA, pointB, pointC has same appearance as point cloud stored as pointB, pointA, pointC.


## Machine learning

Pointwise classification with ML:
3D point cloud -> neighborhood selection -> feature extraction -> feature selection -> supervised classification -> labeled 3D point cloud

Contextual classification; statistical models of context

* Label smoothing
    * Kernel filtering in sliding window fashion
* MRF (markov random field)
    * Take into account feature vector of a point and the labels corresponding to neighboring points
* CRF (conditional random field)
    * Take into account feature vector of a point and the feature vectors and labels corresponding to neighboring points

### Challenges

Single scale vs multi scale representation

* Single scale
    * Spherical neighborhood with fixed radius
    * Cylindrical neighborhood with fixed radius
    * K-closest neighbors in 3D or 2D -> scale parameter?
* Multi scale
    * Collection of cylindrical neighborhoods (Niemeyer et al, 2014)
    * Collection of spherical neighborhoods (Brodu & Lague, 2012)
    * Multiscale multi type neighborhood (Blomley & Weinmann, 2017)

Interpretable features vs complex features

* Interpretable
    * Eigenvalue/-vector based 
    * Linearity, planarity, spericity, omnivariance, aniotropy, eigenentropy, sum of eigenvalues, change of curvature
    * Geometric properties of local neighborhood, scanner agnostic
    * Relfectance and other scanner specific features are tricky but can help with noise filtering
    * Scaling for example by distance
    * 2D projection based
    * Additional attributes (e.g. intensity, ...)
* Complex
    * 3D shape context descriptor (Frome et al, 2004)
    * SHOT descriptor (Tombari et al, 2010)
    * PFHs (Rusu et al, ICRA 2009)
    * Shape distributions (Osada et al 2002)

Selection of suitable features among relevant, irrelevant and redundant features:

* Filter based methods
* Wrapper based methods
* Embedded methods

### Models

For example:

* Nearest Neighbor (ninstance based learning) 
* Decision tree (rule-based learning)
* Naive Bayesian classifier (probabilistic learning)
* Random forest, xdg boosting, extratrees (ensemble learning)
* Support vector machines (max-margin learning)
* ...

## Deep learning

3D point cloud challenges for deep learning:

* Irregularity: points are not evenly sampled; contain dense and sparse regions
* Unstructured: independent points without fixed distances to neighbors
* Unordered: set of points -> invariant to permutation

### Main approaches

Main approaches for dealing with point cloud data for deep learning:

* Structured grid 
    * Multiview: project point cloud to image
    * Volumetric: convert point cloud to volumetric representation (e.g. voxels)
* Raw point cloud
    * PointNet
    * Local region computation; capturing local structures -> sampling, grouping, mapping function
        * No local correlation: POintNet++,...
    * Local correlation : PointCNN, ...
        * Graph based

#### Structuring the point cloud 

##### Multiview

Projection / Structured grid based approaches -> adaptation to 2D data for e.g. 2D CNN based approaches

1. Point cloud to "images" from different angles
2. Use 2D methods
3. Back project results to 3D point cloud 


Examples:

* MVCNN (multi-view CNN), Su et al, 2015 
* SLCAE (Stacked local convolutional autoencoder), Leng et al, 2015
* GIFT, Bai et al, 2016
* 3D ShapeNet, Wu et al, 2015 
* MV-SpericalProject, Cao et al, 2017

##### Voxel

Voxel based approaches -> adaptation of CNN to 3D data

1. Define local neighborhood
2. Derive voxel occupancy grid
3. use 3D-CNN

Examples:

* VoxNet, Maturana et al, 2015: 3DCNN for object recognition based on 3D binary occupancy grid
* VMCNN (volumentric and multi-view CNN) Qui et al, 2016
* NormalNet, Wang et al, 2019
* MRCNN (multi-resolution CNN), Ghadai et al, 2018

Voxelisation challenges:

* High memory consumption (often need to reduce resolution)
* Many voxels empty -> inefficient resource usage
-> OctNet as more efficient memory solution (OctNet: Learning Deep 3D Representations at High Resolutions, Riegler et. al)


##### Higher dimensional lattices

-> Converting the point cloud into higher dimensional regular lattice

* SplatNet, Su et al, 2018
* Spherical fractal CNN, Rao et al, 2019

#### Raw point cloud

-> PointNet (Qi et. al, 2017) addresses those problems.

* First method for DL on pointclouds directly
* Can be used for classification and segmentation
* Each point is processed individually with same transform, finally a symmetric function (e.g. max pooling) is applied -> for permutation invariance
* Employs spatial transformer network for transformation invariance
* Interaction is not addressed but PointNet still works well

#### Local region - no local correlation

* PointNet++, Qi et al, 2017 -> iterative farthest sampling ; restricted to small point clouds
* VoxelNet, Zhou et al, 2018: voxel feature encoding -> sparse convolution: dealing with sparsity
* SO-Net, Li et al, 2018: using self-organizing map 
* Pointwise Convolution, Hua et al, 2018: Convolution operation is done on all input points
* 3D PointCapsNet, Zhao et al 2019: region correlation by dynamic routing procedure
* RandLA-Net (2020) -> random sampling -> agnostic to number of points -> scalable!

#### Local region - local correlation

* PointCNN, Li et al, 2018 -> improvement on PointNet++
* PointWeb, Zhao et al, 2019 -> "local web ob points" 
* PointConv, Wu et al, 2019 
* Relation-shape-CNN, Liu et al, 2019 
* GeoCNN, Lan et al, 2019
* Annularly-CNN, Komarichev et al, 2019
* SpiderCNN, Xu et al, 2018
* Point attention transformers, Yang et al, 2019


#### Graph based

Locality (interaction among local points in a neighborhood) can be exploited if point cloud is represented as graph. Graph Neural Network takes inspiration from regular convolution in CNN.

* Kd-Network, Klokov et al, 2017
* Dynamic graph CNN, Wang et al, 2018
* Point2Node, Han et al, 2020


## Labeled point clouds

* Manual labeling: 
    * e.g. CloudCompare semi manual classification; noise filtering based on intensity; manual labeling based on relectance plus geometry (creating polygons spanning points in 3D space)
    * Expesive and laborious
* Synthetic data generation, 
    * e.g. [Helios++](https://github.com/3dgeo-heidelberg/helios)
    * possibility to rescan same scene
    * Various scan setups can be tested
    * Intensity cannot be simulated
* Labeled datasets
    * Segmented from LiDAR
        * Semantic3D, Hackel et al, 2017
        * MIMP, Wang et al 2018
        * KITTI, Geiger et al 2012
        * Semantic KITTI, Behley et al, 2019
        * ASL Dataset, Pomerleau et al, 2012
        * iQmulus, Bredif et al, 2014
        * Oxford Robotcar, Maddern et al, 2017
        * NPM3D, Roynard et al, 2018
    * Objects from LiDAR
        * Apollo, Lu et al, 2019
        * Whu-TLS, Dong et al, 2020

## Recent trends

* Large scale classification
    * Improved effectiveness
        * Information sampling
        * Variations in scale
        * Scene complexity
    * Improved efficiency
        * Lightweight models
* Small amount of training data (weakly- self- semi-supervised classification)
* Explainability 


## References and further reading

* PointNet++ : https://towardsdatascience.com/understanding-machine-learning-on-point-clouds-through-pointnet-f8f3f2d53cc3
* Point cloud completion with DL - Review: https://arxiv.org/abs/2203.03311
* Papers with code: https://paperswithcode.com/task/3d-point-cloud-classification
* Review article for point cloud deep learning: https://www.mdpi.com/2072-4292/12/11/1729 -> includes almost all references to the different methods and datasets mentioned here

Thanks to Anna Shcherbacheva , FGI/NLS for sharing some insights. Also thanks to insights gained from presentations at the International Workshop on Point cloud processing 2023.