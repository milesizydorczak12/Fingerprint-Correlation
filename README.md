# REU-Biometrics-1F

Analyzing the pattern correlation among ten fingerprint patterns - **first in the world to prove existence of intra-person fingerprint correlation**. By Gabe Guo, Miles Izydorczak, Wenyao Xu.

# Installation

imutils==0.5.4
matplotlib==3.3.4
numpy==1.16.2
opencv_python==4.6.0.66
Pillow==9.3.0
scikit_image==0.17.2
scipy==1.5.4
shap==0.41.0
skimage==0.0
torch==1.10.1
torchvision==0.11.2

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
  - Saves in folder at same level as the input folder, called ```img_l2_feature_extractions```; will preserve subdirectory structure
  - Subfolders in output are ```enhance```, ```minutiae```
- For Level 1, run ```level1FeatureExtraction/main.py --src \[name of the enhanced images folder\]```
  - Saves in folder at same level as the input folder, called ```img_l1_feature_extractions```; will preserve subdirectory structure
  - Note: Input images should already be enhanced/binarized: we can get these from the output of the Level 2 feature extractions, under the folder ```img_l2_feature_extractions/enhance```

# Data

Dataset not included, due to NIST restrictions. Can be requested from [NIST](https://nigos.nist.gov/datasets/sd302/request). 
