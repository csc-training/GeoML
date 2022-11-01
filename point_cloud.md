# Pointcloud ML 


* geometric features → scanner agnostic (SW will get some examples from FGI)

* scaled by distance
* neighborhood kNN, eigenvalues and eigenvectors, do they change with increased neighborhood etc
* Treebased algorithms for wood(branch,stem)/leave classification
    * random forest
    * xdg boosting
    * extratrees (much faster training than random forest)
* timeseries of reflectance without geometry for tree species classification
* think also about necessary point cloud density (5pts/m2 much cheaper to produce and process)
* reusing labeled data from different forest is tricky (even different birches can look quite differently)
* separate models for different species
* tree segments by height to separate noise
* labeling pointcloud data → CloudCompare semi manual classification; noise filtering based on intensity, manual labeling based on reflectance plus geometry (polygon spanning points in 3D space)
* reflectance and other scanner specific features are tricky but can help filtering noise

