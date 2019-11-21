import cv2
import numpy as np
import os
from ctypes import cdll, c_byte, c_float, c_uint8, c_int
lib = cdll.LoadLibrary("./HogVis.dll")

class HogVisual(object):
    def __init__(self):
        self.obj = lib.HogVis_new()

    def test(self):
        lib.HogVis_testDLL(self.obj)

    def visualize(self,s1,s2,s3,fv):
        lib.HogVis_hog(self.obj,s1,s2,s3,fv)

imageDir = './CompassImages'
output = np.zeros((1764,1))
count = 0
for filename in os.listdir(imageDir):
    image = cv2.imread(imageDir + '\\' + filename)
    grey_image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    width = 64
    height = 64
    small_img_grey = cv2.resize(grey_image,(width,height))
    small_img = cv2.resize(image,(width,height))

    # Display image
    # cv2.imshow('image',small_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Changable HOG Parameters
    winSize = (width,height)
    blockSize = (16,16)
    blockStride = (8,8)
    cellSize = (8,8)
    nbins = 9
    # Non-changable HOG Parameters
    derivAperture = 1
    winSigma = -1.
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = 1
    nlevels = 64

    # Calculate descriptor
    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                            histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
    descriptor = hog.compute(small_img_grey)
    img_desc = np.asarray(descriptor[0:1764])

    # Prepare planes for input
    oneRow = c_byte * 64
    aSlice = oneRow * 64
    slice1 = aSlice()
    slice2 = aSlice()
    slice3 = aSlice()
    plane1 = cv2.split(small_img)[0]
    plane2 = cv2.split(small_img)[1]
    plane3 = cv2.split(small_img)[2]

    # Copy the R, B, & G planes to C-acceptable arrays
    for ind1 in range(0,64):
        for ind2 in range(0,64):
            slice1[ind1][ind2] = plane1[ind1][ind2] 
            slice2[ind1][ind2] = plane2[ind1][ind2]  
            slice3[ind1][ind2] = plane3[ind1][ind2]

    # Prepare descriptor for input
    featureVecType = c_float * 1764
    featureVec = featureVecType()
    for ind in range(0,1764):
        featureVec[ind] = descriptor[ind]

    h = HogVisual()
    h.visualize(slice1,slice2,slice3,featureVec)
    # h.test()

    # Add to the array to be saved
    if count == 0:
        output = img_desc
    else: 
        output = np.append(output,img_desc,axis=1)
    count += 1
np.savetxt("HogFeatures.csv",output, delimiter=",")

print("Done")
