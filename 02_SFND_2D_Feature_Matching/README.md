# SFND 2D Feature Tracking

<img src="images/keypoints.png" width="820" height="248" />

The idea of the camera course is to build a collision detection system - that's the overall goal for the Final Project. As a preparation for this, you will now build the feature tracking part and test various detector / descriptor combinations to see which ones perform best. This mid-term project consists of four parts:

* First, you will focus on loading images, setting up data structures and putting everything into a ring buffer to optimize memory load. 
* Then, you will integrate several keypoint detectors such as HARRIS, FAST, BRISK and SIFT and compare them with regard to number of keypoints and speed. 
* In the next part, you will then focus on descriptor extraction and matching using brute force and also the FLANN approach we discussed in the previous lesson. 
* In the last part, once the code framework is complete, you will test the various algorithms in different combinations and compare them with regard to some performance measures. 

See the classroom instruction and code comments for more details on each of these parts. Once you are finished with this project, the keypoint matching part will be set up and you can proceed to the next lesson, where the focus is on integrating Lidar points and on object detection using deep-learning. 

## Dependencies for Running Locally
* cmake >= 2.8
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* OpenCV >= 4.1
  * This must be compiled from source using the `-D OPENCV_ENABLE_NONFREE=ON` cmake flag for testing the SIFT and SURF detectors.
  * The OpenCV 4.1.0 source code can be found [here](https://github.com/opencv/opencv/tree/4.1.0)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory in the top level directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./2D_feature_tracking`.

## MidTerm Report : Project Rubric Points (For Att: Reviewer) - ASHOK RAJ

### MP.1 Data Buffer optimization
* implimented a data buffer of size = dataBufferSize {in this case its 2}
  * Implemented as a std::vector
  * As soon as the vector size reaches dataBufferSize , the image at the begining of the vector is removed. 
  * Then the new image is pushed into the vector, at the vector end
  * Alternative Implementation is to add the image at first, then remove the image at begining of the vector as soon as the vector size is  greater than dataBufferSize

### MP.2 KeyPoint Detection
* implemented using if else and standard open CV function for FAST BRISK ORB AKAZE SIFT Detector
* For HARRIS used the implementation from a previous exerscise
* Additional variable tDetector is used to bring computation inforamation back to the main function. This is used in MP.9

### MP.3 KeyPoints Removal
* Iterate through every Keypoint in keypoints
* if the keypoint is inside the Rectangle {using contains function}, add the keypoint to a tempory vector
* After iteration move the contents of the temperory vector to the keypoints vector

### MP.4 Keypoints Descriptors
* Used open CV library function  for BRIEF, ORB, FREAK, AKAZE and SIFT to generate the descriptors as well as refered the exerscise related to this topic
* addtional variable tDescriptor to share the computation time with the main function

### MP.5 Descriptor Matching
* Implemented the Descriptor matching referring the exercise on this topic. BF method was already implemeted
* FLANN Method is used - Both Nearest Neighbour(NN) and kNN
* In order to avoid openCV bug on FLANN, binary descriptors need to be converted to float, This need to be done on both source and ref frames and so we should individually check both frames for datatype

### MP.6 Descriptor Distance Ratio
* Iterate through the knn matches and choose the match that satisfies the Distance Ratio condition,


### MP.7 MP.8 MP.9 : Performance Evaluation
* Refer the Project2_Camera_Detectors_Descriptors.xlsx
* There are seperate sheet for each task and a detailed analysis and conclusion


### Conclusion
* THe following are my top 3 choice for detector /descriptor combination
  * BRISK - BRIEF
  * FAST - BRIEF
  * FAST - ORB
* I have determined this based on Total no of matches, total speed and the distribution of keypoints over various areas of the Vehicle