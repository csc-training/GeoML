# Further reading material

## Other GIS or AI courses

* CSC:
	* Fundamentals of Machine Learning
	* [Practical Deep Learning](https://e-learn.csc.fi/course/view.php?id=14)
	* [Geocomputing on supercomputers](https://csc-training.github.io/geocomputing_course/index.html)
	* [Introduction to Python GIS](https://e-learn.csc.fi/course/view.php?id=122)
	* [Spatial data analysis with R](https://e-learn.csc.fi/course/view.php?id=120)
 	* To get infor about CSC instructor-led intensive courses:
  		* Follow [CSC training calendar]
    	* Subscribe to [CSC Customer Training newsletter](https://csc.fi/en/subscribe-to-newsletters/)
     	* Subscribe to [CSC gis-hpc mailing list](https://postit.csc.fi/sympa/subscribe/gis-hpc)
* Location Innovation Hub (LIH) courses in [OGC academy](https://academy.ogc.org/)
	* Location and AI: The Basics
 	* Location & AI – Deep Dive into Practical Solutions
* University of Helsinki:
	* [Elements of AI](https://www.elementsofai.com/)
	* [Advanced remote sensing of environment](https://studies.helsinki.fi/courses/course-implementation/hy-opt-cur-2526-fed2206d-989a-4124-95fc-d2666df33db7/GEOG-322)
* [SYKE, ML course](https://github.com/esiivola/syke-machine-learning-course)
* [Huggingface Learn](https://huggingface.co/learn)

## Data
### Labeled spatial ML data

* [Hugginface Torchgeo datasets](https://huggingface.co/datasets?sort=downloads&search=torchgeo), use search to find other datasets

#### Finnish spatial labeled ML data
* [Buildings from aerial images](https://tiedostopalvelu.maanmittauslaitos.fi/tp/julkinen/lataus/tuotteet/TrainingDataForBuildings_ATMU), NLS/FGI, ATMU project
* [Marine vessels detection from Sentinel-2 RGB](https://zenodo.org/records/15019034), Janne Mäyrä, SYKE
* [Forest variables from VNIR hyperspectral image](https://doi.org/10.23729/fe7ce882-8125-44e7-b0cf-ae652d7ed0d5), Matti Mõttus, VTT
* [Semantic segmentation of mobile laser scanning point clouds](https://doi.org/10.23729/8d2d3765-b5a0-4998-82c1-13a6f8bc9de3), Harri Kaartinen, Antero Kukko, NLS/FGI
* [Semantic segmentation of aerial LiDAR](https://sharpershape.com/eclair-a-high-fidelity-aerial-lidar-dataset-for-semantic-segmentation/), SharperShape
* [Land cover and use points, according to Eurostat LUCAS](https://ckan.ymparisto.fi/organization/syke-geoinformatics?q=LUCAS), data from Finland, available for several years, SYKE
* [10 000 random 512x512 pixel Sentinel 2 Level-1C RGB satellite images](https://doi.org/10.23729/32a321ac-9012-4f17-a849-a4e7ed6b6c8c), from Finland, years 2015–2022, Olli Niemitalo & Elias Junior Anzini & Vinicius Hermann D. Liczkoski, HAMK

* https://github.com/Seyed-Ali-Ahmadi/Awesome_Satellite_Benchmark_Datasets
* https://github.com/chrieke/awesome-satellite-imagery-datasets
* https://github.com/robmarkcole/satellite-image-deep-learning/blob/master/assets/datasets.md
* Public tree pointcloud dataset: https://data.mendeley.com/datasets/4gbzk9sy24/1 , publication: https://www.sciencedirect.com/science/article/pii/S0924271620302094
* [AI datasets by NASA](https://search.earthdata.nasa.gov/search?portal=ai-ml&lat=-0.140625)
* [FLAIR: a Country-Scale Land Cover Semantic Segmentation Dataset From Multi-Source Optical Imagery](https://arxiv.org/abs/2310.13336) and [code](https://github.com/IGNF/FLAIR-2-AI-Challenge)
* [OpenForest catalog for machine learning in forest monitoring](https://github.com/RolnickLab/OpenForest)
* GLanCE - Global land cover training dataset from 1984 to 2020, [publication](https://www.nature.com/articles/s41597-023-02798-5) and [dataset](https://beta.source.coop/repositories/boston-university/bu-glance/)
* Schmitt et al / Department of Aerospace Engineering, University of the Bundeswehr Munich, Neubiberg, Germany ,[There Are No Data Like More Data: Datasets for deep learning in Earth observation](https://ieeexplore.ieee.org/document/10213439)

### Spatial data

* [Paituli STAC](https://paituli.csc.fi/stac.html) - Finnish raster datasets via STAC
* [Geoportti Geocubes](https://vm0160.kaj.pouta.csc.fi/geocubes/) - Finnish harmonized data, easy to use for ML
* [Open spatial data](https://www.geoportti.fi/services/data/), Geoportti listin of open global and Finnish spatial datasets
* https://github.com/sacridini/Awesome-Geospatial#data-sources

## Models
* [Huggingface Torchgeo models](https://huggingface.co/models?sort=downloads&search=torchgeo), use search to find other models

## Books

* Gwanggil Jeon: [Advanced Machine Learning and Deep Learning Approaches for Remote Sensing](https://www.mdpi.com/books/book/7482)

## Publications
* [Huggingface papers](https://huggingface.co/papers)
* Ava Vali / Politecnico di Milano, [Deep Learning for Land Use and Land Cover Classification Based on Hyperspectral and Multispectral Earth Observation Data: A Review](https://www.researchgate.net/publication/343419901_Deep_Learning_for_Land_Use_and_Land_Cover_Classification_Based_on_Hyperspectral_and_Multispectral_Earth_Observation_Data_A_Review)
* Monia Digra / Monia Digra Shri Mata Vaishno Devi University, [Land use land cover classification of remote sensing images based on the deep learning approaches: a statistical analysis and review](https://www.researchgate.net/publication/360662937_Land_use_land_cover_classification_of_remote_sensing_images_based_on_the_deep_learning_approaches_a_statistical_analysis_and_review)
* Jürgen Döllner / University of Potsdam, [Geospatial Artificial Intelligence: Potentials of Machine Learning for 3D Point Clouds and Geospatial Digital Twins](https://link.springer.com/article/10.1007/s41064-020-00102-3)
* Marvin Mc Cutchan / TU Wien, [Encoding Geospatial Vector Data for Deep Learning: LULC as a Use Case](https://www2.mdpi.com/2072-4292/14/12/2812/htm)
* Younes Charfaoui, [Working with Geospatial Data in Machine Learning](https://heartbeat.comet.ml/working-with-geospatial-data-in-machine-learning-ad4097c7228d) , feature extration from geospatial data for ML
* Behnam Nikparvar / University of North Carolina at Charlotte, [Machine Learning of Spatial Data](https://www.mdpi.com/2220-9964/10/9/600/htm)
* Aaron E. Maxwell /  West Virginia University, [Implementation of machine-learning classification
in remote sensing: an applied review](https://www.tandfonline.com/doi/pdf/10.1080/01431161.2018.1433343)
* Safonova et al / Leibniz Centre for Agricultural Landscape Research (ZALF), Müncheberg, Germany, [Deep Learning techniques for adressing small data problems in remote sensing](https://www.sciencedirect.com/science/article/pii/S156984322300393X)
* Calyan Chen  et al ,[Using time-series imagery and 3DLSTM model to classify individual tree species](https://www.tandfonline.com/doi/full/10.1080/17538947.2024.2308728)

## Point clouds

* A comprehensive overview of deep learning techniques for 3D point cloud classification and semantic segmentation, [Sarker et al, 2023](https://arxiv.org/abs/2405.11903)
* Deep learning-based 3D point cloud classification: A systematic survey and outlook, [Zhang et al, 2023](https://www.sciencedirect.com/science/article/abs/pii/S0141938223000896)
* Point cloud completion with DL - [Review, Fei et al, 2022](https://arxiv.org/abs/2203.03311)

## GeoML public code
* https://huggingface.co/papers
* [SpaceNet challenges](https://spacenet.ai/), [baseline and winning codes](https://github.com/SpaceNetChallenge/)
* [The Environmental Data Science book](https://edsbook.org/gallery) several GeoAI projects presented as Jupyter notebooks.


## Links to more links
* https://github.com/satellite-image-deep-learning
* https://github.com/deepVector/geospatial-machine-learning
* https://github.com/robmarkcole/satellite-image-deep-learning
* https://github.com/sacridini/Awesome-Geospatial#deep-learning
* https://github.com/sshuair/awesome-gis#deep-learning
* https://github.com/wenhwu/awesome-remote-sensing-change-detection
