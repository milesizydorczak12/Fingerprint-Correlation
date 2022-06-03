# FingerprintExtraction
To run there is an included yml for anaconda containing the environment used for python libraries.

Use this command for the environment.yml

conda env create -f environment.yml

Install that environment or install these things

opencv
imutils
scipy
matplotlib

Run main.py with two flags
--inputPath <dest to input path folder of grayscale fingerprint images (png,jpg,jpeg,bmp)
--outputPath <dest to output path for all extracted info>

HOW TO RUN FROM COMMANDLINE:
# Step 1: To Get Fingerprint Images
python main.py --inputPath ./inputPathImages --outputPath ./outputPathData

# Step 2: To Get GLCM Features (Make sure to edit path in Line 176 of features.py)
python features.py

# Outside Stuff used
<b>NBIS MINDTCT Algorithm</b>

This software was developed at the National Institute of Standards and Technology (NIST) by employees of the Federal Government in the course of their official duties. Pursuant to Title 17 Section 105 of the United States Code, this software is not subject to copyright protection and is in the public domain. NIST assumes no responsibility whatsoever for use by other parties of its source code or open source server, and makes no guarantees, expressed or implied, about its quality, reliability, or any other characteristic.
