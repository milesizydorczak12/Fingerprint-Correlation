# REU-Biometrics-1F
Summer research project to analyze the pattern correlation among ten fingerprint patterns. Dataset not included, due to NIST restrictions; can be requested from [NIST](https://nigos.nist.gov/datasets/sd302/request). By Miles Izydorczak, Gabe Guo, Wenyao Xu.

# Some Info on What This Code Does

Data Split:
- Get the dataset from NIST.
- Run gather_nist_data.py, which does a train-val split on the data.
- (Optional, since run_all_fgrp.py does it automatically) ~~Run general_train_test_split.py on the directory that we previously put the train-val split into, so that training and testing use mutually exclusive finger types (e.g., left pinky, right thumb). The used samples will stay in 'train' and 'val', respectively; the unused samples (that would cause overlapping finger types between sets) go into 'train_unused' and 'val_unused', respectively. Now, the training and testing (i.e., validation) sets should have no overlap in finger types used or sensors used to gather the data, which should mean that any discovered correlation would be due to intrinsic fingerprint patterns common to all of a person's fingerprints.~~

Edits to Make:
- Will need to change root_dir on line 5 of run_all_fgrp.py, data_dir on line 46 of transfer_learning.py, MainHypothesis/fingerprint_data/features_examined.txt.

Deep Learning Analysis:
- Run MainHypothesis/run_all_fgrp.py. 

Conventional Feature Extraction:
- For Level 1, run iFeel_FeatureExtraction/main.py.
- For Level 2, run ConventionalFeatureExtraction/level2/generate_minutiae_diagram.py.
