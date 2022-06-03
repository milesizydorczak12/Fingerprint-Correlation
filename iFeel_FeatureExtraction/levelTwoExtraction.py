import cv2
import csv
import os
import numpy as np


def levelTwoExtraction(inputPath, outputPath):
    """this uses the NBIS MINDTCT algorithm to extract minutiae from the original image
    once extracted the minutiae are laid on a blank canvas they could also be overlayed onto the original image
    if so chosen"""
    img = cv2.imread(inputPath, 0)
    imgBase = inputPath[inputPath.rfind("/"):inputPath.rfind(".")]
    outputImageName = outputPath+imgBase
    outputImagePath = outputImageName + ".png"
    cv2.imwrite(outputImagePath, img)
    # call mindtct object with outputImagePath and outputImageName
    cmd = "./mindtct " + outputImagePath + " " + outputImageName
    os.system(cmd)
    outputXYT = outputImageName + ".xyt"
    data = []
    with open(outputXYT) as minutiae:
        reader = csv.reader(minutiae, delimiter=' ')
        for row in reader:
            x,y,t,q = row
            data.append([int(x),int(y),int(t)])
    # put all minutiae on empty numpy array
    canvas = np.zeros(img.shape)
    canvas.fill(255)
    #x_arr = []
    #y_arr = []
    #theta_arr = []
    for minut in data:
        x, y, theta = minut
        #x_arr.append(x)
        #y_arr.append(y)
        #theta_arr.append(theta)
        cv2.rectangle(canvas, (x - 3, y - 3), (x + 3, y + 3), 0, 1)
        cv2.line(canvas, (x, y), (int(x + (8 * np.cos(theta))), int(y + (8 * np.sin(theta)))), 0, 1)
    cv2.imwrite(outputImageName+"_minutiae.png", canvas)
    print("{} Finished Level Two Extraction".format(outputImagePath))
    #return x_arr, y_arr, theta_arr
