import os
import sys
import shutil
import random

src_dir = '/home/gabeguo/Biometric_Research/Fingerprint/REU-Biometrics-1F/ModifiedDataset'

dest_dir = '/home/gabeguo/Biometric_Research/Fingerprint/REU-Biometrics-1F/SampleModified'
    
SAMPLE_SIZE = 600

src_and_dest = []
for subdir in ['minutiae', 'enhance', 'level1_Output']:
    src_and_dest.append((os.path.join(src_dir, subdir), os.path.join(dest_dir, subdir)))

for src_dest_tuple in src_and_dest:
    src_sub, dest_sub = src_dest_tuple[0], src_dest_tuple[1]
    
    assert os.path.exists(src_sub)
    os.makedirs(dest_sub, exist_ok=True)

    count = 0
    for root, dirs, files in os.walk(src_sub):
        for file in files:
            if random.random() < 0.25:
                src_filepath = os.path.join(root, file)
                shutil.copy(src_filepath, dest_sub)
                count += 1
        if count > SAMPLE_SIZE:
            break

