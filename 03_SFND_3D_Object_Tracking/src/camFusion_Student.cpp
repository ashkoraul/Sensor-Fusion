
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2); // pixel coordinates
        pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}


void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
  	
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));
    //cout<<worldSize.height<<"\t"<<worldSize.width<<"\t"<<imageSize.height<<"\t"<<imageSize.width<<endl;

    //int count =0;
    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
      	
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));
        //cv::Scalar currColor = cv::Scalar(0,255,0);

        
        //cout<<count++; 
        // plot Lidar points into top view image
        float top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        //cout<<"enter";
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
                    	
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;
            

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;
            //cout<<y<<"\t"<<"\t"<<x<<endl;
            //cout<<left<<"\t"<<top<<"\t"<<right<<"\t"<<bottom<<endl;

            // draw individual point
            //cv::circle(topviewImg, cv::Point(y, x), 4, currColor, -1);
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }
        //cout<<"exit"<<endl;
        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);
       // cv::rectangle(topviewImg, cv::Point(440, 1159), cv::Point(586, 1202),cv::Scalar(0,0,0), 2);
       // cv::rectangle(topviewImg, cv::Point(1159, 440), cv::Point(1202, 586),cv::Scalar(0,0,0), 2);
       //cout<<left<<"\t"<<top<<"\t"<<right<<"\t"<<bottom<<endl;

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        //cout<<test++<<endl;
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }
    
    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 4);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
//void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
//{
    // ...
//}

void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    // ...
    std::vector<cv::DMatch> kptMatchesWithROI;
    std::vector<float> vEuclideanDist;

    float shrinkFactor =0.2, stdFactor = 1.7;
    cv::Rect smallerBox;
    smallerBox.x = boundingBox.roi.x + shrinkFactor * boundingBox.roi.width / 2.0;
    smallerBox.y = boundingBox.roi.y + shrinkFactor * boundingBox.roi.height / 2.0;
    smallerBox.width = boundingBox.roi.width * (1 - shrinkFactor);
    smallerBox.height = boundingBox.roi.height * (1 - shrinkFactor);
    float distSum =0;
    // Generate the list of keypoint matches and euclidean dist contained inside the bounding box
    for (auto itrMatch = kptMatches.begin(); itrMatch!=kptMatches.end();itrMatch++)
    {
        cv::KeyPoint keypointPrev = kptsPrev[itrMatch->queryIdx];
        cv::KeyPoint keypointCurr = kptsCurr[itrMatch->trainIdx];
        if(smallerBox.contains(keypointCurr.pt))
        {
            float euclideanDist = std::sqrt((keypointCurr.pt.x - keypointPrev.pt.x) * (keypointCurr.pt.x - keypointPrev.pt.x) + 
                                            (keypointCurr.pt.y - keypointPrev.pt.y) * (keypointCurr.pt.y - keypointPrev.pt.y));
            distSum = distSum + euclideanDist;
            //cout<<euclideanDist<<endl;
            vEuclideanDist.push_back(euclideanDist);
            kptMatchesWithROI.push_back(*itrMatch);

        }
    }
    //calculate the mean distance
    float meanEuclideanDist = distSum / vEuclideanDist.size();
    float varSumEuclideanDist = 0;
  	//calculate the std Deviation
    for (float distance : vEuclideanDist )
    {
        varSumEuclideanDist = varSumEuclideanDist + (distance - meanEuclideanDist) * (distance - meanEuclideanDist);
    }
    float stdEuclideanDist = std::sqrt(varSumEuclideanDist/vEuclideanDist.size());
    //cout<< meanEuclideanDist<<"\t"<<stdEuclideanDist<< "\t"<<kptMatchesWithROI.size() <<endl;
  	//filter out the outliers.
    for (auto itrMatch = kptMatchesWithROI.begin(); itrMatch!=kptMatchesWithROI.end();itrMatch++)
    {
        cv::KeyPoint keypointPrev = kptsPrev[itrMatch->queryIdx];
        cv::KeyPoint keypointCurr = kptsCurr[itrMatch->trainIdx];
        float euclideanDist = std::sqrt((keypointCurr.pt.x - keypointPrev.pt.x) * (keypointCurr.pt.x - keypointPrev.pt.x) + 
                                        (keypointCurr.pt.y - keypointPrev.pt.y) * (keypointCurr.pt.y - keypointPrev.pt.y));
        if ( std::fabs(euclideanDist - meanEuclideanDist)  < ( stdFactor * stdEuclideanDist) )
        {
            boundingBox.kptMatches.push_back(*itrMatch);
            boundingBox.keypoints.push_back(keypointCurr);
        }

    }




}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // compute distance ratios between all matched keypoints
    vector<double> distRatios; // stores the distance ratios for all keypoints between curr. and prev. frame
    //float ratioSum=0,ratioSquareSum=0;
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1)
    { // outer kpt. loop

        // get current keypoint and its matched partner in the prev. frame
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);
        

        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2)
        { // inner kpt.-loop

            double minDist = 100.0; // min. required distance

            // get next keypoint and its matched partner in the prev. frame
            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

            // compute distances and distance ratios
            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
            { // avoid division by zero

                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
               // ratioSum =ratioSum + distRatio;
                //ratioSquareSum = ratioSquareSum + distRatio*distRatio; 
            }

        } // eof inner loop over all matched kpts
    }     // eof outer loop over all matched kpts

    // only continue if list of distance ratios is not empty
    if (distRatios.size() == 0)
    {
        TTC = NAN;
        return;
    }
    /*
    float avgRatio = ratioSum/distRatios.size();
    float stdRatio = std::fabs(std::sqrt(ratioSquareSum/distRatios.size()) - avgRatio); // approximate std deviation
    float factor =2;
    cout<<avgRatio<<"\t"<<stdRatio<<"\t";

    std::vector<double> temp;
    for (auto itr = distRatios.begin(); itr!=distRatios.end();itr++)
    {
        if (std::fabs(*itr - avgRatio) < factor* stdRatio)
        { 
            temp.push_back(*itr);
        }

    }
    distRatios = temp;
    */

    // compute camera-based TTC from distance ratios
    //double meanDistRatio = std::accumulate(distRatios.begin(), distRatios.end(), 0.0) / distRatios.size();
    int distRatiosSize = distRatios.size();
    int medianIndex = distRatiosSize/2 + distRatiosSize%2;
    std::sort(distRatios.begin(),distRatios.end());
    double medianDistRatio = distRatios [medianIndex];

    double dT = 1 / frameRate;
    //TTC = -dT / (1 - meanDistRatio);
    cout<<medianDistRatio;
    TTC = -dT / (1 - medianDistRatio);

    // STUDENT TASK (replacement for meanDistRatio)
}

/*
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // ...

    
}
*/

void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    // ...
     // auxiliary variables
    double dT = 1/frameRate;        // time between two measurements in seconds
    double laneWidth = 4.0; // assumed width of the ego lane
 
    // Method1
    double xPrevMean, xPrevStd, xCurrMean, xCurrStd;
    double xStdFactor = 1.8; //  omit noise using normal distribution xStdFactor times sigma{std deviation}
    
    //calculate mean variance for prev frame x distance. To remove outliers
    double sum=0, sqSum=0;
    for (auto it = lidarPointsPrev.begin(); it != lidarPointsPrev.end(); ++it)
    {
        sum = sum + it->x;
        
    }
    xPrevMean = sum/(lidarPointsPrev.size());
    for (auto it = lidarPointsPrev.begin(); it != lidarPointsPrev.end(); ++it)
    {
        
        sqSum= sqSum + ((it->x) - xPrevMean) * ((it->x) - xPrevMean);
    }
    xPrevStd = std::sqrt(sqSum/lidarPointsPrev.size());
    

    //calculate mean variance for curr frame x distance. To remove outliers
    sum=0, sqSum=0;
    for (auto it = lidarPointsCurr.begin(); it != lidarPointsCurr.end(); ++it)
    {
        sum = sum + it->x;
        
    }
    xCurrMean = sum/(lidarPointsCurr.size());
    for (auto it = lidarPointsCurr.begin(); it != lidarPointsCurr.end(); ++it)
    {
        sqSum= sqSum + ((it->x)-xCurrMean) * ((it->x)-xCurrMean);
    }
    
    xCurrStd = std::sqrt(sqSum/lidarPointsCurr.size());

    //std::vector<float> x_listPrev,x_listCurr;
    
    // find closest distance to Lidar points within ego lane
    double minXPrev = 1e9, minXCurr = 1e9;
    for (auto it = lidarPointsPrev.begin(); it != lidarPointsPrev.end(); ++it)
    {
        
        if (abs(it->y) <= laneWidth / 2.0 && std::fabs(it->x - xPrevMean) < (xStdFactor*xPrevStd)  )
        { // 3D point within ego lane? & Within noise tolerance
            minXPrev = minXPrev > it->x ? it->x : minXPrev;
            //x_listPrev.push_back(it->x);
            
        }
    }

    for (auto it = lidarPointsCurr.begin(); it != lidarPointsCurr.end(); ++it)
    {

        if (abs(it->y) <= laneWidth / 2.0 && std::fabs(it->x - xCurrMean) < (xStdFactor*xCurrStd) )
        { // 3D point within ego lane? & Within noise tolerance
            minXCurr = minXCurr > it->x ? it->x : minXCurr;
            //x_listCurr.push_back(it->x);
            
        }
    }

    
   // Method 2 
   /*
    std::vector<float> x_listPrev,x_listCurr;
    double minXPrev = 1e9, minXCurr = 1e9;
    for (auto it = lidarPointsPrev.begin(); it != lidarPointsPrev.end(); ++it)
    {
        
        if (abs(it->y) <= laneWidth / 2.0   )
        { // 3D point within ego lane? & Within noise tolerance
           
            x_listPrev.push_back(it->x);
        }
    }

    for (auto it = lidarPointsCurr.begin(); it != lidarPointsCurr.end(); ++it)
    {

        if (abs(it->y) <= laneWidth / 2.0  )
        { // 3D point within ego lane? & Within noise tolerance
            
            x_listCurr.push_back(it->x);
        }
    }
   
    
    std::sort(x_listCurr.begin(),x_listCurr.end());

    std::sort(x_listPrev.begin(),x_listPrev.end());
    minXCurr = x_listCurr[x_listCurr.size()/4];
    minXPrev = x_listPrev[x_listPrev.size()/4];
     */


    //cout<<minXCurr<<"\t"<<minXPrev<<"\t"<<xCurrThresh<<"\t"<<xPrevThresh<<"\t"<<xCurrMean<<"\t"<<xCurrStd<<"\t"<<xPrevMean<<"\t"<<xPrevStd<<endl;
    // compute TTC from both measurements
    TTC = minXCurr * dT / (minXPrev - minXCurr);
    //cout<<TTC<<"\t"<<minXCurr<< "\t"<<xCurrMean << "\t"<<xCurrStd<<endl;
}


void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    float shrinkFactor = 0.10; 
    std::map<std::pair<int,int>,int> bbMatchesWCount;
    for (auto itrMatch = matches.begin(); itrMatch!=matches.end();itrMatch++) // iterate thru keypoint match pairs
    {
        cv::KeyPoint keypointPrev = prevFrame.keypoints[itrMatch->queryIdx];
        cv::KeyPoint keypointCurr = currFrame.keypoints[itrMatch->trainIdx];
               
        // find the bounding box in previous frame containing the keypoint prev
        for (auto itrBoxPrev =( prevFrame.boundingBoxes).begin();itrBoxPrev!=( prevFrame.boundingBoxes).end(); itrBoxPrev++) 
        {          
            //using shrinked box to neglect points on the boundary
            cv::Rect smallerBox;
            smallerBox.x = (*itrBoxPrev).roi.x + shrinkFactor * (*itrBoxPrev).roi.width / 2.0;
            smallerBox.y = (*itrBoxPrev).roi.y + shrinkFactor * (*itrBoxPrev).roi.height / 2.0;
            smallerBox.width = (*itrBoxPrev).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*itrBoxPrev).roi.height * (1 - shrinkFactor);

            if(smallerBox.contains(keypointPrev.pt))
            {
                // finding bounding box in curr frame containing keypoint curr. using nested for loop to cover same keypoint presence in multiple Bounting boxes
                for (auto itrBoxCurr =(currFrame.boundingBoxes).begin();itrBoxCurr!=( currFrame.boundingBoxes).end(); itrBoxCurr++)
                {              
                    cv::Rect smallerBox;
                    smallerBox.x = (*itrBoxCurr).roi.x + shrinkFactor * (*itrBoxCurr).roi.width / 2.0;
                    smallerBox.y = (*itrBoxCurr).roi.y + shrinkFactor * (*itrBoxCurr).roi.height / 2.0;
                    smallerBox.width = (*itrBoxCurr).roi.width * (1 - shrinkFactor);
                    smallerBox.height = (*itrBoxCurr).roi.height * (1 - shrinkFactor);
                    if(smallerBox.contains(keypointCurr.pt))
                    {  
                        std::pair<int,int> bbMatch (itrBoxPrev->boxID, itrBoxCurr->boxID );
                        // if a bb match exist , increment the count. else insert the new match
                        if(bbMatchesWCount.count(bbMatch)>0)
                        {
                            bbMatchesWCount[bbMatch] = bbMatchesWCount[bbMatch]+1;
                        }
                        else
                        {
                            bbMatchesWCount[bbMatch] = 1;
                        } 
                    }
                }
            }

        }

      
    
    }

    //Displaying BB match and count for testing
    /*/
    for (auto itr : bbMatchesWCount)
    {
        cout<<(itr.first).first<<"\t"<<(itr.first).second<<"\t"<<itr.second<<endl;
    }
    *///
   
    int threshKeypointsMatch = 40; // threshold to remove boxes with low match pair  
    // first round of filtering to choose between pairs like (0,1) and (2,1)     
    for (auto itrBoxCurr =(currFrame.boundingBoxes).begin();itrBoxCurr!=( currFrame.boundingBoxes).end(); itrBoxCurr++)
    {
        std::pair <int,int> bbBestMatch (-1,-1);
        for (auto itrBoxPrev =( prevFrame.boundingBoxes).begin();itrBoxPrev!=( prevFrame.boundingBoxes).end(); itrBoxPrev++)
        {
            std::pair<int,int> bbMatch (itrBoxPrev->boxID, itrBoxCurr->boxID );
            if(bbMatchesWCount[bbMatch] > threshKeypointsMatch && bbMatchesWCount[bbMatch] > bbMatchesWCount[bbBestMatch])
                bbBestMatch = bbMatch;
            else
            {
                bbMatchesWCount.erase(bbMatch);
            }

        }
    }

    // second round of filtering to choose between pairs like (1,0) and (1,2) 
    for (auto itrBoxPrev =( prevFrame.boundingBoxes).begin();itrBoxPrev!=( prevFrame.boundingBoxes).end(); itrBoxPrev++)
    {
        std::pair <int,int> bbBestMatch (-1,-1);
        
        for (auto itrBoxCurr =(currFrame.boundingBoxes).begin();itrBoxCurr!=( currFrame.boundingBoxes).end(); itrBoxCurr++)
        {
            std::pair<int,int> bbMatch (itrBoxPrev->boxID, itrBoxCurr->boxID );
            if(bbMatchesWCount[bbMatch] > threshKeypointsMatch && bbMatchesWCount[bbMatch] > bbMatchesWCount[bbBestMatch])
                bbBestMatch = bbMatch;
                
        }

        if(bbBestMatch != std::make_pair(-1,-1))
        {
            
            bbBestMatches.insert(bbBestMatch);
            //cout<<bbBestMatch.first <<"\t"<<bbBestMatch.second<<endl;

        }
            

    }
        


}
