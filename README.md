# A Compass Should Always Point North

In the CompassImages folder you will see a collection of 16 images. There is an arrow and the letter "N" in each image. The arrow points either up, right, down, or left. Additionally, the placement of "N" is also in those same directions. This image set contains all the possible combinations of the configurations. As a follow-up approach to the [introduction approach](https://github.com/Matt-Conrad/Compass_NeuralNetwork) that utilized manually created features and a Neural Network classifier, I wanted to calculate the Histogram Of Gradients (HOG) descriptor for each of these compass images to see if those features could be used to classify the images where the arrow was correctly pointing to the "N". 

In addition to calculating the HOG descriptor, I wanted to visualize the output to get a simple representation of the features in one place. At the time, I couldn't find a way to visualize this output using Python; the HOG descriptor was calculated using opencv's Python wrapper function cv2.HOGDescriptor() and there was no way to visualize that. So I found this to be a good time to delve into running C/C++ code from Python using the ctypes Python library.

### Dependencies

I'm using the following software in this project:
* Python 3.7.3 
* opencv 4.1.2
* Numpy 1.17.3

### Explanation of Files

* HogFeatureExtraction.py: The script that prepares the images, calculates the HOG Descriptor, and displays the descriptor using the DLL
* HogVis.dll: The DLL compiled from the HogVis.cpp and opencv_world341d.lib
* HogVis.cpp: The source code that carries out the visualization of the HOG Descriptor
* hog.o: Intermediary object file compiled from HogVis.cpp and integrated into HogVis.dll
* HogFeatures.csv: Feature matrix containing all HOG descriptors from all the images
* opencv_world341d.lib: A static library that is used in compiling HogVis.dll
* OutputImageExample.png: An example output image where the HOG descriptor is overlaid on one of the compass images

### Results

I was successfully able to get a visualization of the HOG descriptor using this approach. 