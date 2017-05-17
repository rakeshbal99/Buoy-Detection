#include <iostream>
#include <opencv2/opencv.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace cv;
using namespace std;

int main() {

  VideoCapture cap("/home/ayu/Videos/buoy.avi");
  VideoWriter writer;

  Mat image, hsv, grey;
  cap.read(image);

  writer.open("video.avi", CV_FOURCC('M', 'J', 'P', 'G'), cap.get(CV_CAP_PROP_FPS), image.size(), CV_8UC3); 

  while(cap.read(image)) {
     cvtColor(image, hsv, COLOR_BGR2HSV);

     for(int i=0; i<hsv.rows; i++)
         for(int j=0; j<hsv.cols; j++) 
	      if(hsv.at<Vec3b>(i,j)[0] >80 && hsv.at<Vec3b>(i,j)[0] < 95)
                       hsv.at<Vec3b>(i,j)[2] = 0;
          
     imshow("Frame", hsv);   

  // MIN-ENCLOSING CIRCLE

     cvtColor(hsv, hsv, COLOR_HSV2BGR);
     cvtColor(hsv, grey, COLOR_BGR2GRAY);

    // erode(grey, grey, 0, Point(-1, -1), 5, 1, 1);
    // dilate(grey, grey, 0, Point(-1, -1), 2, 1, 1);
    // erode(grey, grey, 0, Point(-1, -1), 5, 1, 1);
    // dilate(grey, grey, 0, Point(-1, -1), 2, 1, 1);

     imshow("grayscale", grey);
     vector<vector<Point> > contours;
     vector<Vec4i> hierarchy;

     findContours( grey, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0) );

     vector<vector<Point> > contours_poly( contours.size() );
     vector<Point2f>center( contours.size() );
     vector<float>radius( contours.size() );

     for( size_t i = 0; i < contours.size(); i++ )
    {
      approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
      minEnclosingCircle( contours_poly[i], center[i], radius[i] );
    }

     for( size_t i = 0; i< contours.size(); i++ )
        if((int)radius[i]>50)
          circle( image, center[i], (int)radius[i], Scalar(0,255,0), 2, 8, 0 );
    
  
    imshow( "Contours", image );
    writer.write(image);

    if(waitKey(100)=='q') 
	break;
 }

}
