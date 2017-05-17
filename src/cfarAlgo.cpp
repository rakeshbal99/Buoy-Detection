#include <iostream>
#include <math.h>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace cv;
using namespace std;

Mat cfar(Mat& frame, Mat img, int num_train, int num_guard, float rate1, float rate2, int block_size) {

  Mat a(img.rows, img.cols, CV_8UC3, Scalar(0,0,0));
  
  int num_rows = a.rows - (a.rows%num_train) ;
  int num_cols = a.cols - (a.cols%num_train) ;
  int num_side = num_train/2;
  
  double alpha1 = num_train * ( pow(rate1, -1.00/num_train) -1 );	
  double alpha2 = num_train * ( pow(rate2, -1.00/num_train) -1 );  

  for(int i = num_side; i <= num_rows; i += num_side*2) 
    for(int j = num_side; j <= num_cols; j += num_side*2) {
       
       int sum1 = 0, sum2 = 0;
       double thresh, p_noise;
      
	//CASO-CFAR
       for(int x = i-num_side; x <= i-num_guard; x++) 
         for(int y = j-num_side; y <= j-num_guard; y++)
           sum1 += img.at<Vec3b>(x,y)[0];

       for(int x = i+num_guard; x <= i+num_side; x++)
         for(int y = j+num_guard; y <= j+num_side; y++)
           sum2 += img.at<Vec3b>(x,y)[0];
  	
       p_noise=min(sum1,sum2)/(num_train*num_train);
       thresh = alpha1*p_noise;

       if( img.at<Vec3b>(i,j)[0] < thresh) {

         for(int k = i-block_size/2; k <= i+block_size/2; k++)
           for(int l = j-block_size/2; l <= j+block_size/2; l++)
             a.at<Vec3b>(k,l)[0] = 255;	
	}

	//CA-CFAR 
       for(int x = i-(num_guard/2); x <= i+(num_guard/2)+1; x++)
         for(int y = j-(num_guard/2); y <= j+(num_guard/2)+1; y++)
           sum1 += img.at<Vec3b>(x,y)[0];

       for(int x = i-num_side; x <= i+num_side+1; x++)
         for(int y = j-num_side; y <= j+num_side+1; y++)
           sum2 += img.at<Vec3b>(x,y)[0];
  
       p_noise = fabs(sum1-sum2)/(num_train*num_train);
       thresh = alpha2*p_noise;
  
       if( img.at<Vec3b>(i,j)[0] > thresh) {

         for(int k = i-block_size/2; k <= i+block_size/2; k++)
           for(int l = j-block_size/2; l <= j+block_size/2; l++)
             a.at<Vec3b>(k,l)[0] = 255;
	}   
 
  }

  return a;
}

int main() {

  VideoCapture cap("/home/ayu/Videos/buoy.avi");
  
  if(!cap.isOpened()) {
    cout<<"Can't open video file.";
    return -1;
  }
    Mat frame, hsv, grey,res;
  
    while(cap.read(frame)) {
  
      cvtColor(frame, hsv, COLOR_BGR2HSV);

      Mat image = cfar(frame, hsv, 6, 3, 0.5, 0.2, 3);
      cvtColor(image, grey, COLOR_BGR2GRAY);
   
      imshow("cfar_output", image);    
      imshow("object", frame);
 	   
      if(waitKey(1)=='q')
      	break;
   
  }

  return 0;
}
