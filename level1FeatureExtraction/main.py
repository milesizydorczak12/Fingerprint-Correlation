
import argparse
from multiprocessing.pool import ThreadPool
import levelOneExtraction
import levelTwoExtraction
import cv2
import os
import csv


def editImage(inputImg):
    try:
        print('processing {}'.format(inputImg))
        imgBase = inputImg[inputImg.rfind("/"):inputImg.rfind(".")]
       
        maskPath = outputPath + imgBase + "_mask.png"
        enhancedPath = outputPath + imgBase + "_enhanced.png"
        orientPath = outputPath + imgBase + "_orient.png"
        ridgePath = outputPath + imgBase + "_ridge.png"
        #print('generating enhanced img')
        enhancedImg = levelOneExtraction.imageFilter1(inputImg, outputPath)
        #print('generating orientation')
        orient, orientList = levelOneExtraction.findOrientationPhase(enhancedPath)
        mask = cv2.imread(maskPath, 0)
        #print('generating ridge count')
        freq, ridgeCount = levelOneExtraction.findRidgeFlowCount(enhancedPath, orientList)
        # print(ridgeCount)
        #print(enhancedImg.shape, 'enhacned img shape')
        #print(mask.shape, 'mask shape')
        for y in range(enhancedImg.shape[0]):
            for x in range(enhancedImg.shape[1]):
                if mask[y, x] != 255:
                    orient[y, x] = 255
                    freq[y, x] = 0
        cv2.imwrite(ridgePath, freq)
        cv2.imwrite(orientPath, orient)
        print("{} Finished Level One Extraction".format(enhancedPath))
    except Exception as e:
        print(inputImg, "doesn't work")
        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Master Run File')
    parser.add_argument('--src', dest='input', help='Path to folder containing images', type=str)
    args = parser.parse_args()
    inputPath = args.input
    outputPath = os.path.join(inputPath, '../l1_feature_extractions')
    os.makedirs(outputPath, exist_ok=True)

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
    
