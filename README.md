# REU-Biometrics-1F
Summer research project to analyze the pattern correlation among ten fingerprint patterns. Dataset not included, due to NIST restrictions; can be requested from [NIST](https://nigos.nist.gov/datasets/sd302/request). By Miles Izydorczak, Gabe Guo, Wenyao Xu.

# Some Info on What This Code Does

Data Split:
- Get the dataset from NIST.
- Run helpful_scripts/split_files_by_sensor_letter.py to separate the fingerprint images into two mutually exclusive sets by sensors used. The first set should be fingerprints that can be used for training (subdirectory called 'train'), and the second set should be fingerprints that can be used for testing (subdirectory called 'val'). Both sets should contain samples of all ten fingerprint types, but the sensors that were used to gather the data for the two sets should be mutually exclusive. This way, in the experiments, the discovered correlation would not be due to factors related to sensors, but rather, intrinsic to fingerprints.
- Run general_train_test_split.py on the directory that we previously put the train-test split into, so that training and testing use mutually exclusive finger types (e.g., left pinky, right thumb). The used samples will stay in 'train' and 'val', respectively; the unused samples (that would cause overlapping finger types between sets) go into 'train_unused' and 'val_unused', respectively. Now, the training and testing (i.e., validation) sets should have no overlap in finger types used or sensors used to gather the data, which should mean that any discovered correlation would be due to intrinsic fingerprint patterns common to all of a person's fingerprints.

Deep Learning Analysis:
- Run MainHypothesis/run_all_fgrp.py.

Conventional Feature Extraction:
- For Level 1, run iFeel_FeatureExtraction/main.py.
- For Level 2, run ConventionalFeatureExtraction/level2/generate_minutiae_diagram.py.
