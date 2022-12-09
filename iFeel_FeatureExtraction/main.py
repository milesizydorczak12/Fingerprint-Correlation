
import argparse
from multiprocessing.pool import ThreadPool
import levelOneExtraction
import levelTwoExtraction
import cv2
import os
import csv


def editImage(inputImg):
    try:
        imgBase = inputImg[inputImg.rfind("/"):inputImg.rfind(".")]
        maskPath = outputPath + imgBase + "_mask.png"
        enhancedPath = outputPath + imgBase + "_enhanced.png"
        orientPath = outputPath + imgBase + "_orient.png"
        ridgePath = outputPath + imgBase + "_ridge.png"
        enhancedImg = levelOneExtraction.imageFilter1(inputImg, outputPath)
        orient, orientList = levelOneExtraction.findOrientationPhase(enhancedPath)
        mask = cv2.imread(maskPath, 0)
        freq, ridgeCount = levelOneExtraction.findRidgeFlowCount(enhancedPath, orientList)
        # print(ridgeCount)
        for y in range(enhancedImg.shape[0]):
            for x in range(enhancedImg.shape[1]):
                if mask[y, x] != 255:
                    orient[y, x] = 255
                    freq[y, x] = 0
        cv2.imwrite(ridgePath, freq)
        cv2.imwrite(orientPath, orient)
        print("{} Finished Level One Extraction".format(enhancedPath))
        #levelTwoExtraction.levelTwoExtraction(inputImg, outputPath)
        # print(len(x))
        # with open(outputPath+imgBase+'_orientF.csv','w',newline='') as f:
        #    writer = csv.writer(f)
        #    writer.writerows(orient)
        # with open(outputPath+imgBase+'_ridgeF.csv','w',newline='') as f:
        #    writer = csv.writer(f)
        #    writer.writerows(freq)
        # with open(outputPath+imgBase+'_minutiaeF.csv','w',newline='') as f:
        #    writer = csv.writer(f)
        #    writer.writerows(x,y,theta)
    except Exception:
        print(inputImg, "doesn't work")
        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Master Run File')
    parser.add_argument('--inputPath', dest='input', help='Path to folder containing images', type=str)
    parser.add_argument('--outputPath', dest='output', help='Path to folder to save images', type=str)
    args = parser.parse_args()
    inputPath = args.input
    outputPath = args.output
    outputPath = outputPath if outputPath[-1] != "/" else outputPath[:-1]
    
    imageFiles = []
    for root, dirs, files in os.walk(inputPath, topdown=False):
        for name in files:
            relPath = os.path.join(root, name)  
            if relPath.endswith('.png') or relPath.endswith('.jpg') or relPath.endswith('.jpeg') or relPath.endswith('.pneg'):
                imageFiles.append(os.path.join(inputPath, relPath))

    pool = ThreadPool(20)
    pool.map(editImage, imageFiles)
