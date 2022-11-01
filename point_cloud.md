# Pointcloud ML 

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

