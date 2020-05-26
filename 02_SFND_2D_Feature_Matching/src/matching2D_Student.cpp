#include <numeric>
#include "matching2D.hpp"

using namespace std;

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
    // configure matcher
    bool crossCheck = true;
    
    cv::Ptr<cv::DescriptorMatcher> matcher;
    

    if (matcherType.compare("MAT_BF") == 0)
    {
        int normType = descriptorType.compare("DES_BINARY") == 0 ? cv::NORM_HAMMING : cv::NORM_L2;
        //int normType = cv::NORM_HAMMING;
        matcher = cv::BFMatcher::create(normType, crossCheck);
        cout << "BF matching with crosscheck = "<<crossCheck<<endl;
           
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        // ...
        
        if (descSource.type() != CV_32F)
        { // OpenCV bug workaround : convert binary descriptors to floating point due to a bug in current OpenCV implementation
            
            descSource.convertTo(descSource, CV_32F);
            //cout<<"converted Source"<<endl;
            
        }
        if (descRef.type() != CV_32F)
        { // OpenCV bug workaround : convert binary descriptors to floating point due to a bug in current OpenCV implementation
                        
            descRef.convertTo(descRef, CV_32F);
            //cout<<"converted Ref"<<endl;
        }

        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
        cout << "FLANN matching";

    }

    // perform matching task
    
    
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)
        double t = (double)cv::getTickCount();
        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        cout << " (NN) with n=" << matches.size() << " matches in " << 1000 * t / 1.0 << " ms" << endl;
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)

        // ...
        vector<vector<cv::DMatch>> knn_matches;
        double t = (double)cv::getTickCount();
        matcher->knnMatch(descSource, descRef, knn_matches, 2); // finds the 2 best matches
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        cout << " (KNN) with n=" << knn_matches.size() << " matches in " << 1000 * t / 1.0 << " ms" << endl;

        // STUDENT TASK
        // filter matches using descriptor distance ratio test
        double minDescDistRatio = 0.8;
        for (auto it = knn_matches.begin(); it != knn_matches.end(); ++it)
        {

            if ((*it)[0].distance < minDescDistRatio * (*it)[1].distance)
            {
                matches.push_back((*it)[0]);
            }
        }
        cout << "# keypoints removed = " << knn_matches.size() - matches.size() << endl;
    }
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType,  double &tDescriptor)
{
    // select appropriate descriptor
  
    cv::Ptr<cv::DescriptorExtractor> extractor;
    // BRIEF, ORB, FREAK, AKAZE, SIFT BRISK
    if (descriptorType.compare("BRISK") == 0)
    {
         cout<<"Choosen Descriptor Type : BRISK"<<endl;

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else if(descriptorType.compare("BRIEF") == 0)
    {
        cout<<"Choosen Descriptor Type : BRIEF"<<endl; //not using the variable descriptorType, to ensure there is no confusion because of empty spaces or typos in string
        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
    }
    else if(descriptorType.compare("ORB") == 0)
    {
        cout<<"Choosen Descriptor Type : ORB"<<endl; //not using a variable
        extractor = cv::ORB::create();
    }
    else if(descriptorType.compare("FREAK") == 0)
    {
        cout<<"Choosen Descriptor Type : FREAK"<<endl; //not using a variable
        extractor = cv::xfeatures2d::FREAK::create();
    }
    else if(descriptorType.compare("AKAZE") == 0)
    {
        cout<<"Choosen Descriptor Type : AKAZE"<<endl; //not using a variable
        extractor = cv::AKAZE::create();
    }
    else // SIFT default
    {
        cout<<"Choosen Descriptor Type : SIFT"<<endl; //not using a variable
        extractor = cv::xfeatures2d::SiftDescriptorExtractor::create();

        //...
    }

    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
    tDescriptor =t;
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img,  double &tDetector, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
    tDetector=t;
    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img,  double &tDetector, bool bVis)
{
    
    // Detector parameters
    int blockSize = 2;     // for every pixel, a blockSize × blockSize neighborhood is considered
    int apertureSize = 3;  // aperture parameter for Sobel operator (must be odd)
    int minResponse = 100; // minimum value for a corner in the 8bit scaled response matrix
    double k = 0.04;       // Harris parameter 

    // Detect Harris corners and normalize output
    cv::Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);
    double t = (double)cv::getTickCount();
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    
  
   //Non maxima supression
    
    double maxOverlap=0;
    for (size_t i=0; i < dst_norm.rows; i++)
    {
        for (size_t j =0; j < dst_norm.cols; j++)
        {
            int response = (int) dst_norm.at<float>(i,j);
            if (response>minResponse)
            {
                cv::KeyPoint newKeyPoint;
                newKeyPoint.pt= cv::Point2f(j,i);
                newKeyPoint.size = 2 * apertureSize;
                newKeyPoint.response = response;

                //perform NMS
                bool bOverlap = false;
                //iterate thru the list of keypoints to check overlap
                for (auto it = keypoints.begin(); it != keypoints.end(); it++)
                {
                    //calculate the overlap with the new keypoint
                    double kptOverlap = cv::KeyPoint::overlap(newKeyPoint,*it);

                    // check if the overlap is above the permitted max
                    if (kptOverlap > maxOverlap)
                    {
                        bOverlap = true;
                        // check if the new point has a higher response
                        if (newKeyPoint.response > (*it).response)
                        {
                            //replace old key point in the list with new one
                            *it = newKeyPoint;
                            break; // exit the loop
                        }


                    }
                }

                //if no overlap add the new point to list
                if(!bOverlap)
                    keypoints.push_back(newKeyPoint);

            }
        } // end of cols
    }//end of rows
     t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "HARRIS detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
    tDetector=t;
    //visualize kkeypoints
    if (bVis)
    {
        string windowName = "Harris Corner detection";
        cv::namedWindow(windowName, 1);
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints,visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::imshow(windowName,visImage);
        cv::waitKey(0);
    }




}



void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType,double &tDetector,  bool bVis)
{
    cv::Ptr<cv::FeatureDetector> detector;
    if(detectorType.compare("FAST") == 0)
    {            
        //FAST
        int threshold = 30;                                                              // difference between intensity of the central pixel and pixels of a circle around this pixel
        bool bNMS = true;                                                                // perform non-maxima suppression on keypoints
        cv::FastFeatureDetector::DetectorType type = cv::FastFeatureDetector::TYPE_9_16; // TYPE_9_16, TYPE_7_12, TYPE_5_8
        detector = cv::FastFeatureDetector::create(threshold, bNMS, type);
        
    }
    else if(detectorType.compare("BRISK") == 0)
    {
        //BRISK
        cout<<"Choosen Detector Type : BRISK"<<endl;
        detector = cv::BRISK::create();        

    }
    else if(detectorType.compare("ORB") == 0)
    {
        //ORB

        cout<<"Choosen Detector Type : ORB"<<endl;
        detector = cv::ORB::create();
        

    }
    else if(detectorType.compare("AKAZE") == 0)
    {
        //AKAZE

        cout<<"Choosen Detector Type : AKAZE"<<endl;
        detector = cv::AKAZE::create();
        
    }
    else //Assuming SIFT as default  
    {
        //SIFT

        cout<<"Choosen Detector Type : SIFT"<<endl;
        detector = cv::xfeatures2d::SIFT::create();
        
    }
    double t = (double)cv::getTickCount();
    detector->detect(img, keypoints);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << detectorType<<" detector with n= " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
    tDetector=t;
    if(bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = detectorType + " Results";
        cv::namedWindow(windowName, 1);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }

}