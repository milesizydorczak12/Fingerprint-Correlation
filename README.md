# REU-Biometrics-1F

Analyzing the pattern correlation among ten fingerprint patterns - **first in the world to prove existence of intra-person fingerprint correlation**. By Gabe Guo, Miles Izydorczak, Wenyao Xu.

# Installation

colorama==0.3.7
imutils==0.5.4
llvmlite==0.36.0
matplotlib==3.3.4
networkx==2.5.1
numba==0.53.1
numpy==1.19.5
opencv-python==4.6.0.66
pandas==1.1.5
pickleshare==0.7.4
Pillow==8.4.0
protobuf==3.7.1
pyimage==0.1.1
PyWavelets==1.1.1
scikit-image==0.17.2
scikit-learn==0.24.2
scipy==1.5.4
shap==0.41.0
six==1.16.0
torch==1.10.2
torchvision==0.11.2

**Incomplete**

# How to Run

Data Split:
- Get the dataset from NIST.
- Run ```gather_nist_data.py --train_src <string of training_src_dirs separated by spaces> --test_src <string of testing_src_dirs separated by spaces> --dest <dest_dir> [--summary]```, which does a train-val split on the data.
  - Will need to change ```root_dir``` on line 5 of ```run_all_fgrp.py```

Deep Learning Analysis:
- Run ```MainHypothesis/run_all_fgrp.py```. 
  - Will need to change ```data_dir``` on line 46 of ```transfer_learning.py```, ```MainHypothesis/fingerprint_data/features_examined.txt``` (this one just for logging purposes, according to the images we pass in).

Conventional Feature Extraction:
- For Level 2, run ```level2FeatureExtraction/extract_minutiae.py --src \[name of your images folder\]```
  - Saves in folder at same level as the input folder, called ```l2_feature_extractions```; will preserve subdirectory structure
  - Subfolders in output are ```enhance```, ```minutiae```
- For Level 1, run ```level1FeatureExtraction/main.py --src \[name of the enhanced images folder\]```
  - Saves in folder at same level as the input folder, called ```l1_feature_extractions```; will preserve subdirectory structure
  - Note: Input images should already be enhanced/binarized: we can get these from the output of the Level 2 feature extractions, under the folder ```\[l2_feature_extractions\]/enhance```

# Data

Dataset not included, due to NIST restrictions. Can be requested from [NIST](https://nigos.nist.gov/datasets/sd302/request). 
