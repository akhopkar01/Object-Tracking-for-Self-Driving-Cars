/**
 * Copyright 2020 Mahmoud Dahmani, Aditya Khopkar
 */

#include "tracker.h"
#include <opencv2/highgui.hpp>
#include <iostream>

int main(int argc, char* argv[]) {
  //cv::VideoCapture stream("ourVideo.mp4");
  std::unordered_set<std::string> objectClasses{"person", "car"};
  cv::Mat frame;
  cv::Matx34f extP{0, 0, 1, -1, 1, 0, 0, 0, 0, 1, 0, 1};
  cv::Matx33f intP{0.5, 0, 160, 0, 0.5, 160, 0, 0, 1};
  ENPM808X::ObjectTracker tracker(objectClasses, extP, intP);
  std::vector<cv::Point3f> objectLocations;  // system output
/*
  while (stream.read(frame)) {
    objectLocations = tracker.localizeObjects(frame);
    // tracker.detectObjectKeypoints(frame);

    cv::namedWindow("Live", cv::WINDOW_AUTOSIZE);
    cv::imshow("Live", frame);
    if (cv::waitKey(1) >= 0) break;
  }
*/
  cv::Mat img = cv::imread("../testImage.png");
  if(img.empty()){
    std::cout << "[Err] Couldn't read image" << std::endl;
    return -1;
  }
  tracker.localizeObjects(img);
  cv::imshow("Display", img);
  cv::waitKey(0);

  return 0;
}
