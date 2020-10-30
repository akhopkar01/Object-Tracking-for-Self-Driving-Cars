/**
 * Copyright 2020 Mahmoud Dahmani, Aditya Khopkar
 */
/**
 * @file: tracker.cpp
 * @brief: This file contains the definition of the tracker class which is responsible for detecting and tracking the humans
 * @author: Mahmoud Dahmani (driver), Aditya Khopkar (navigator)
 * */

#include "tracker.h"
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <opencv2/imgproc.hpp>

ENPM808X::ObjectTracker::ObjectTracker(
    const std::unordered_set<std::string>& objectClasses,
    const cv::Matx34f& extP, const cv::Matx33f& intP,
    const std::string& detectionModel, float minConfidence, float minOverlap)
    : objectClasses_{objectClasses},
      minConfidence_{minConfidence},
      minOverlap_{minOverlap} {
  datasetLabels_ = {parseFile("../" + detectionModel + "/coco.names")};
  cv::Size inputResolution{320, 320};
  // network_ = cv::dnn::DetectionModel(detectionModel + "/yolov4.weights",
  // detectionModel + "/yolov4.cfg");
  network_.setInputSize(inputResolution);
  network_.setInputScale(1.0 / 255.0);
  network_.setInputSwapRB(true);
  P_ = intP * extP;
  colors_ = {{"safe", {0, 255, 0}}, {"unsafe", {0, 0, 255}}};
}

std::vector<cv::Point3f> ENPM808X::ObjectTracker::localizeObjects(
    cv::Mat frame) {
  std::vector<cv::Point3f> dummyOutput{{0, 0, 0}};

  return dummyOutput;
}

cv::Point3f ENPM808X::ObjectTracker::localizeObjectKeypoint(
    const cv::Point2i& object) const {
  cv::Point3f dummyOutput{-2, -0.975, 0};

  return dummyOutput;
}

std::vector<cv::Point2i> ENPM808X::ObjectTracker::detectObjectKeypoints(
    cv::Mat frame) {
  std::vector<cv::Point2i> objectKeypoints;

  return objectKeypoints;
}

std::vector<std::string> ENPM808X::ObjectTracker::parseFile(
    const std::string& fileName) const {
  std::vector<std::string> classLabels;
  std::ifstream cocoNames(fileName);
  std::string line;

  if(cocoNames){
    while (getline(cocoNames, line)) classLabels.push_back(line);
    cocoNames.close();
  } else {
    std::cout << "File cannot be opened!" << std::endl;
    std::cin.get();
    exit(0);
  }

  return classLabels;
}
