import cv2
import numpy as np
import os
from ctypes import cdll, c_void_p
lib = cdll.LoadLibrary("C:\\Users\\mconrad2\\Documents\\Compass_HogVisualization\\HogVis.dll")

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
    plane1 = c_void_p(cv2.split(small_img)[0].ctypes.data)
    plane2 = c_void_p(cv2.split(small_img)[1].ctypes.data)
    plane3 = c_void_p(cv2.split(small_img)[2].ctypes.data)

    featureVec = c_void_p(descriptor.ctypes.data)

    h = HogVisual()
    h.visualize(plane1,plane2,plane3,featureVec)
    # h.test()

    # Add to the array to be saved
    if count == 0:
        output = img_desc
    else: 
        output = np.append(output,img_desc,axis=1)
    count += 1
np.savetxt("HogFeatures.csv",output, delimiter=",")

print("Done")
