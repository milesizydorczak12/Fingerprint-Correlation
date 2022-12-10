# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 22:50:30 2016

@author: utkarsh
"""
import cv2
import numpy as np

from FingerprintEnhancement.ridge_filter import ridge_filter
from FingerprintEnhancement.ridge_freq import ridge_freq
from FingerprintEnhancement.ridge_orient import ridge_orient
from FingerprintEnhancement.ridge_segment import ridge_segment


def image_enhance(img):
    blksze = 16;
    thresh = 0.1;
    normim,mask = ridge_segment(img,blksze,thresh);             # normalise the image and find a ROI


    gradientsigma = 1;
    blocksigma = 7;
    orientsmoothsigma = 7;
    orientim = ridge_orient(normim, gradientsigma, blocksigma, orientsmoothsigma);              # find orientation of every pixel


    blksze = 38;
    windsze = 5;
    minWaveLength = 5;
    maxWaveLength = 15;
    freq,medfreq = ridge_freq(normim, mask, orientim, blksze, windsze, minWaveLength,maxWaveLength);    #find the overall frequency of ridges
    
    
    freq = medfreq*mask;
    kx = 0.65;ky = 0.65;
    newim = ridge_filter(normim, orientim, freq, kx, ky);       # create gabor filter and do the actual filtering

    ret, thresh1 = cv2.threshold(newim, 1, 255, cv2.THRESH_BINARY)
    return thresh1


def image_extract(img):
    blksze = 16;
    thresh = 0.1;
    normim, mask = ridge_segment(img, blksze, thresh);  # normalise the image and find a ROI

    gradientsigma = 1;
    blocksigma = 7;
    orientsmoothsigma = 7;
    orient = ridge_orient(normim, gradientsigma, blocksigma, orientsmoothsigma);  # find orientation of every pixel
    for y in range(orient.shape[0]):
        for x in range(orient.shape[1]):
            orient[y, x] = (orient[y, x] / np.pi) * 255
    return orient
