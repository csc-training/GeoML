# Pointcloud ML 

**This document is work in progress**

Structured data: 
* consistent storage and appearance, 
* indices can be used to find neighbors; 
* examples: images, text, tabular data

3D pointclouds are unstructured data: 
* vector data stored in text file
* storage not consistent with appearance 
* N! possible storage options for N points

> Point cloud stored as pointA, pointB, pointC has same appearance as point cloud stored as pointB, pointA, pointC.

3D point cloud challenges:
* small scale
* high dimensionality
* unstructured data

Main approaches for dealing with point cloud data for Deep Learning:
* multiview: project point cloud to image
* volumetric: convert point cloud to volumetric representation (e.g. voxels)
* pointbased: no information loss (!), but expensive neighborhood search
* segmentation: imbalanced data
* semantic segmentation: pointbased (every point has label), class aware (group of points has label); no instances (one class car but no instances car1, car2)
* represent as graph and use suitable network (graph NN -> superpoint graph (SPGraph),...)


Porjection based approaches -> adaptation to 2D data
1. POint cloud to "images"
2. Use 2D methods
3. Back project results to 3D point cloud 


Voxel based approaches -> adaptation of CNN to 3D data
1. Define local neighborhood
2. Derive voxel occupancy grid
3. use 3D-CNN

Voxelisation problems:
* high memory consumption (often need to reduce resolution)
* many voxels empty -> inefficient resource usage
-> OctNet as more efficient memory solution (OctNet: Learning Deep 3D Representations at High Resolutions, Riegler et. al)
-> VoxelNet, sparse convolution: dealing with sparsity



Pointwise classification with ML: 
3D point cloud -> neighborhood selection -> feature extraction -> feature selection -> supervised classification -> labeled 3D point cloud

Main challenges: 
* single scale vs multi scale representation
    * single scale 
        * spherical neighborhood with fixed radius
        * cylindrical neighborhood with fixed radius
        * k closest neighbors in 3D or 2D -> scale parameter?
    * multi scale
        * collection of cylindrical neighborhoods (Niemeyer et al, 2014)
        * collection of spherical neighborhoods (Brodu & Lague, 2012)
        * Multiscale multi type neighborhood (Blomley & Weinmann, 2017)
* interpretable features vs complex features
    * interpretable
        * eigenvalue based (linearity, planarity, spericity, omnivariance, aniotropy, eigenentropy, sum of eigenvalues, change of curvature)
        * geometric properties of local neighborhood
        * 2D projection based
        * additional attributes (e.g. intensity, ...)
    * complex
        * 3D shape context descriptor (Frome et al, 2004)
        * SHOT descriptor (Tombari et al, 2010)
        * PFHs (Rusu et al, ICRA 2009)
        * Shape distributions (Osada et al 2002)
* all features vs relevant features
    * selection of suitable features among relevant, irrelevant and redundant features:
        * filter based methods
        * wrapper based methods
        * embedded methods

Contextual classification; statistical models of context
* label smoothing
    * kernel filtering in sliding window fashion
* MRF (markov random field)
    * take into accoutn feature vector of a point and the labels corresponding to neighboring points
* CRF (conditional random field)
    * take into accoutn feature vector of a point and the feature vectors and labels corresponding to neighboring points






Problems of using point clouds directly:
* point clouds are unstructured, we need permutation invariance
* need to model interaction between points
* need to model tranformation invariance (e.g. rotation,...)

-> PointNet (PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation, Qi et. al, 2017) addresses those problems.
* first method for DL on pointclouds directly
* can be used for classification and segmentation
* each point is processed individually with same transform, finally a symmetric function (e.g. max pooling) is applied -> for permutation invariance
* employs spatial transformer network for transformation invariance
* interaction is not addressed but PointNet still works well

PointNet++ -> iterative farthest sampling ; restricted to small point clouds

RandLA-Net (2020) -> random sampling -> agnostic to number of points -> scalable!

Locality (interaction among local points in a neighborhood) can be exploited if point cloud is represented as graph. Graph Neural Network takes inspiration from regular convolution in CNN.






If not using pointcloud specific models, pointcloud can be turned into features.
* geometric features → scanner agnostic
* reflectance and other scanner specific features are tricky but can help filtering noise
* scaling for example by distance
* feature engineering: neighborhood kNN, eigenvalues and eigenvectors, do they change with increased neighborhood etc
* think also about necessary point cloud density (5pts/m2 much cheaper to produce and process)
* Tree classification specifics:
    * Treebased algorithms for wood(branch,stem)/leave classification
        * random forest
        * xdg boosting
        * extratrees (much faster training than random forest)
    * reusing labeled data from different forest is tricky (even different birches can look quite differently)
    * possibility: separate models for different species
    * tree segments by height to separate noise
* labeling pointcloud data 
    * → CloudCompare semi manual classification; 
        * noise filtering based on intensity, 
        * manual labeling based on reflectance plus geometry (by creating polygons spanning points in 3D space)


Thanks to Anna Shcherbacheva for sharing some insights :)

