# REU-Biometrics-1F
Summer research project to analyze the pattern correlation among ten fingerprint patterns. Dataset not included, due to NIST restrictions; can be requested from [NIST](https://nigos.nist.gov/datasets/sd302/request). By Miles Izydorczak, Gabe Guo, Wenyao Xu.

# Some Info on What This Code Does

Data Split:
- Get the dataset from NIST.
- Run ```gather_nist_data.py```, which does a train-val split on the data.
- Will need to change root_dir on line 5 of ```run_all_fgrp.py```, data_dir on line 46 of ```transfer_learning.py```, ```MainHypothesis/fingerprint_data/features_examined.txt``` (this one just for logging purposes, according to the images we pass in).

Deep Learning Analysis:
- Run ```MainHypothesis/run_all_fgrp.py```. 

Conventional Feature Extraction:
- For Level 2, run ```level2FeatureExtraction/extract_minutiae.py --src \[name of your images folder\]```
  - Saves in folder at same level as the input folder, called ```l2_feature_extractions```; will preserve subdirectory structure
  - Subfolders in output are ```enhance```, ```minutiae```
- For Level 1, run ```level1FeatureExtraction/main.py --src \[name of the enhanced images folder\]```
  - Saves in folder at same level as the input folder, called ```l1_feature_extractions```; will preserve subdirectory structure
  - Note: Input images should already be enhanced/binarized: we can get these from the output of the Level 2 feature extractions, under the folder ```\[l2_feature_extractions\]/enhance```
