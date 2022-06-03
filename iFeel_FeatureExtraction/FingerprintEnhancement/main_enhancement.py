# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 11:42:58 2016
https://github.com/Utkarsh-Deshmukh/Fingerprint-Enhancement-Python
@author: utkarsh
"""
import os
import numpy as np
#import cv2
#import numpy as np;
import matplotlib.pylab as plt;
import scipy.ndimage
import sys
import cv2

from FingerprintEnhancement.image_enhance import image_enhance


def main_enhancement(img):
    # img = cv2.imread(image)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.bitwise_not(img)

    """rows,cols = np.shape(img);
    aspect_ratio = np.double(rows)/np.double(cols);

    new_rows = 350;             # randomly selected number
    new_cols = new_rows/aspect_ratio;

    img = cv2.resize(img,(int(new_rows),int(new_cols)));
    # img = scipy.misc.imresize(img,(np.int(new_rows),np.int(new_cols)));
"""
    enhanced_img = image_enhance(img);

    return enhanced_img
