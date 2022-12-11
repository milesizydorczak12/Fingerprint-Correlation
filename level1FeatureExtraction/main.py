
import argparse
from multiprocessing.pool import ThreadPool
import levelOneExtraction
import levelTwoExtraction
import cv2
import os
import csv


def editImage(inputImg):
    try:
        imgName = inputImg.split('/')[-1]
        print('processing {}'.format(inputImg))
        enhancedPath = inputImg # NOTE: Should have passed enhanced images
        orientPath = os.path.join(orientDir, imgName)
        ridgePath = os.path.join(freqDir, imgName)
        orient, orientList = levelOneExtraction.findOrientationPhase(enhancedPath)
        freq, ridgeCount = levelOneExtraction.findRidgeFlowCount(enhancedPath, orientList)
        cv2.imwrite(ridgePath, freq)
        cv2.imwrite(orientPath, orient)
        print("{} Finished Level One Extraction".format(enhancedPath))
    except Exception as e:
        print(inputImg, "doesn't work")
        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Master Run File')
    parser.add_argument('--src', dest='input', help='Path to folder containing images (should have been enhanced beforehand)', type=str)
    args = parser.parse_args()
    inputPath = args.input
    outputPath = os.path.join(inputPath, '../l1_feature_extractions')
    os.makedirs(outputPath, exist_ok=True)
    orientDir = os.path.join(outputPath, 'orient')
    os.makedirs(orientDir, exist_ok=True)
    freqDir = os.path.join(outputPath, 'freq')
    os.makedirs(freqDir, exist_ok=True)

    imageFiles = []
    for root, dirs, files in os.walk(inputPath, topdown=False):
        for name in files:
            relPath = os.path.join(root, name)  
            if relPath.endswith('.png') or relPath.endswith('.jpg') or relPath.endswith('.jpeg') or relPath.endswith('.pneg'):
                imageFiles.append(os.path.join(inputPath, relPath))
    """
    pool = ThreadPool(20)
    pool.map(editImage, imageFiles)
    """
    for img in imageFiles:
        editImage(img)
    
