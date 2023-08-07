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

* MVCNN (multi-view CNN), [Su et al, 2015](https://www.cv-foundation.org/openaccess/content_iccv_2015/html/Su_Multi-View_Convolutional_Neural_ICCV_2015_paper.html) 
* SLCAE (Stacked local convolutional autoencoder), [Leng et al, 2015](https://www-sciencedirect-com.libproxy.aalto.fi/science/article/pii/S0165168414004150)
* GIFT, [Bai et al, 2016](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Bai_GIFT_A_Real-Time_CVPR_2016_paper.html)
* 3D ShapeNet, [Chang et al, 2015](https://arxiv.org/abs/1512.03012)
* MV-SpericalProject, [Cao et al, 2017](https://ieeexplore-ieee-org.libproxy.aalto.fi/abstract/document/8374611/)

##### Voxel

Voxel based approaches -> adaptation of CNN to 3D data

1. Define local neighborhood
2. Derive voxel occupancy grid
3. use 3D-CNN

Examples:

* VoxNet, [Maturana et al, 2015](https://ieeexplore-ieee-org.libproxy.aalto.fi/abstract/document/7353481/)
* VMCNN (volumentric and multi-view CNN) [Qi et al, 2016](http://openaccess.thecvf.com/content_cvpr_2016/html/Qi_Volumetric_and_Multi-View_CVPR_2016_paper.html)
* NormalNet, [Wang et al, 2019](https://www-sciencedirect-com.libproxy.aalto.fi/science/article/pii/S0925231218311561)
* MRCNN (multi-resolution CNN), [Ghadai et al, 2018](https://www-sciencedirect-com.libproxy.aalto.fi/science/article/pii/S0167839621000832)

Voxelisation challenges:

* High memory consumption (often need to reduce resolution)
* Many voxels empty -> inefficient resource usage
-> OctNet as more efficient memory solution ([Riegler et. al, 2017](http://openaccess.thecvf.com/content_cvpr_2017/html/Riegler_OctNet_Learning_Deep_CVPR_2017_paper.html))


##### Higher dimensional lattices

-> Converting the point cloud into higher dimensional regular lattice

* SplatNet, [Su et al, 2018](http://openaccess.thecvf.com/content_cvpr_2018/html/Su_SPLATNet_Sparse_Lattice_CVPR_2018_paper.html)
* Spherical fractal CNN, [Rao et al, 2019](http://openaccess.thecvf.com/content_CVPR_2019/html/Rao_Spherical_Fractal_Convolutional_Neural_Networks_for_Point_Cloud_Recognition_CVPR_2019_paper.html)

#### Raw point cloud

-> PointNet ([Qi et. al, 2017](http://openaccess.thecvf.com/content_cvpr_2017/html/Qi_PointNet_Deep_Learning_CVPR_2017_paper.html)) addresses those problems.

* First method for DL on pointclouds directly
* Can be used for classification and segmentation
* Each point is processed individually with same transform, finally a symmetric function (e.g. max pooling) is applied -> for permutation invariance
* Employs spatial transformer network for transformation invariance
* Interaction is not addressed but PointNet still works well

#### Local region - no local correlation

* PointNet++, [Qi et al, 2017](https://proceedings.neurips.cc/paper_files/paper/2017/hash/d8bf84be3800d12f74d8b05e9b89836f-Abstract.html) -> iterative farthest sampling ; restricted to small point clouds
* VoxelNet, [Zhou et al, 2018](http://openaccess.thecvf.com/content_cvpr_2018/html/Zhou_VoxelNet_End-to-End_Learning_CVPR_2018_paper.html): voxel feature encoding -> sparse convolution: dealing with sparsity
* SO-Net, [Li et al, 2018](http://openaccess.thecvf.com/content_cvpr_2018/html/Li_SO-Net_Self-Organizing_Network_CVPR_2018_paper.html): using self-organizing map 
* Pointwise Convolution, [Hua et al, 2018](http://openaccess.thecvf.com/content_cvpr_2018/html/Hua_Pointwise_Convolutional_Neural_CVPR_2018_paper.html): Convolution operation is done on all input points
* 3D PointCapsNet, [Zhao et al 2019](http://openaccess.thecvf.com/content_CVPR_2019/html/Zhao_3D_Point_Capsule_Networks_CVPR_2019_paper.html): region correlation by dynamic routing procedure


#### Local region - local correlation

* PointCNN, [Li et al, 2018](https://proceedings.neurips.cc/paper/2018/hash/f5f8590cd58a54e94377e6ae2eded4d9-Abstract.html) -> improvement on PointNet++
* PointWeb, [Zhao et al, 2019](http://openaccess.thecvf.com/content_CVPR_2019/html/Zhao_PointWeb_Enhancing_Local_Neighborhood_Features_for_Point_Cloud_Processing_CVPR_2019_paper.html) -> "local web ob points" 
* PointConv, [Wu et al, 2019](http://openaccess.thecvf.com/content_CVPR_2019/html/Wu_PointConv_Deep_Convolutional_Networks_on_3D_Point_Clouds_CVPR_2019_paper.html) 
* Relation-shape-CNN, [Liu et al, 2019](http://openaccess.thecvf.com/content_CVPR_2019/html/Liu_Relation-Shape_Convolutional_Neural_Network_for_Point_Cloud_Analysis_CVPR_2019_paper.html) 
* GeoCNN, [Lan et al, 2019](http://openaccess.thecvf.com/content_CVPR_2019/html/Lan_Modeling_Local_Geometric_Structure_of_3D_Point_Clouds_Using_Geo-CNN_CVPR_2019_paper.html)
* Annularly-CNN, [Komarichev et al, 2019](http://openaccess.thecvf.com/content_CVPR_2019/html/Komarichev_A-CNN_Annularly_Convolutional_Neural_Networks_on_Point_Clouds_CVPR_2019_paper.html)
* SpiderCNN, [Xu et al, 2018](http://openaccess.thecvf.com/content_ECCV_2018/html/Yifan_Xu_SpiderCNN_Deep_Learning_ECCV_2018_paper.html)
* Point attention transformers, [Yang et al, 2019](http://openaccess.thecvf.com/content_CVPR_2019/html/Yang_Modeling_Point_Clouds_With_Self-Attention_and_Gumbel_Subset_Sampling_CVPR_2019_paper.html)


#### Graph based

Locality (interaction among local points in a neighborhood) can be exploited if point cloud is represented as graph. Graph Neural Network takes inspiration from regular convolution in CNN.

* Kd-Network, [Klokov et al, 2017](http://openaccess.thecvf.com/content_iccv_2017/html/Klokov_Escape_From_Cells_ICCV_2017_paper.html)
* Dynamic graph CNN, [Wang et al, 2018](https://dl-acm-org.libproxy.aalto.fi/doi/abs/10.1145/3326362)
* Point2Node, [Han et al, 2020](https://ojs.aaai.org/index.php/AAAI/article/view/6725)


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
        * Semantic3D, [Hackel et al, 2017](https://arxiv.org/abs/1704.03847)
        * KITTI, [Geiger et al 2013](https://journals-sagepub-com.libproxy.aalto.fi/doi/abs/10.1177/0278364913491297)
        * Semantic KITTI, [Behley et al, 2021](https://journals-sagepub-com.libproxy.aalto.fi/doi/abs/10.1177/02783649211006735)
        * iQmulus, [Bredif et al, 2014](https://hal.science/hal-01101621/)
        * Oxford Robotcar, [Maddern et al, 2017](https://journals-sagepub-com.libproxy.aalto.fi/doi/abs/10.1177/0278364916679498)
    * Objects from LiDAR
        * Apollo, [Lu et al, 2019](http://openaccess.thecvf.com/content_CVPR_2019/html/Lu_L3-Net_Towards_Learning_Based_LiDAR_Localization_for_Autonomous_Driving_CVPR_2019_paper.html)
        * Whu-TLS, [Dong et al, 2020](https://www-sciencedirect-com.libproxy.aalto.fi/science/article/pii/S0924271620300836)

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

### Models

* RandLA-Net for semantic segmentation ([Hu et al, 2020](http://openaccess.thecvf.com/content_CVPR_2020/html/Hu_RandLA-Net_Efficient_Semantic_Segmentation_of_Large-Scale_Point_Clouds_CVPR_2020_paper.html)) -> random sampling -> agnostic to number of points -> scalable!
* PointNext: Revisiting pointnet++ with improved training and scaling strategies, [Qian et al, 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/9318763d049edf9a1f2779b2a59911d3-Abstract-Conference.html)
* Siamese KPConv: 3D multiple change detection from raw point clouds using deep learning, [Gélis et al, 2023](https://www.sciencedirect.com/science/article/abs/pii/S0924271623000394)
* Multi Point-Voxel Convolution, [Zhou et al, 2023](https://www.sciencedirect.com/science/article/abs/pii/S0097849323000377)


## References and further reading

* PointNet++ : https://towardsdatascience.com/understanding-machine-learning-on-point-clouds-through-pointnet-f8f3f2d53cc3
* Point cloud completion with DL - Review: https://arxiv.org/abs/2203.03311
* Papers with code: https://paperswithcode.com/task/3d-point-cloud-classification
* Review article for point cloud deep learning: https://www.mdpi.com/2072-4292/12/11/1729 -> includes almost all references to the different methods and datasets mentioned here
* Deep Learning for 3D Point Clouds: A survey, [Guo et al, 2020](https://ieeexplore-ieee-org.libproxy.aalto.fi/abstract/document/9127813)
* Deep learning-based 3D point cloud classification: A systematic survey and outlook, [Zhang et al, 2023](https://www.sciencedirect.com/science/article/abs/pii/S0141938223000896)


Thanks to Anna Shcherbacheva , FGI/NLS for sharing some insights. Also thanks to insights gained from presentations at the International Workshop on Point cloud processing 2023.