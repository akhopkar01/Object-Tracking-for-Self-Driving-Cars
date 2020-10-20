/**
 * Copyright 2020 Mahmoud Dahmani, Aditya Khopkar
 */

/**
 * @file: tracker.h
 * @brief: This file contains the declaration of the tracker class which is responsible for detecting and tracking the humans
 * @author: Mahmoud Dahmani (driver), Aditya Khopkar (navigator)
 * */

#pragma once

#include <string>
#include <vector>
#include <unordered_set>
#include <map>
#include <opencv2/dnn.hpp>

namespace ENPM808X {
class ObjectTracker {
 public:
 /**
  * @brief: Declaration of the constructor
  * @param: Object classes, Extrinsic Parameter, Intrinsic Parameter, Detection Model, Minimum Confidence, Minimum Overlap
  * @return: None
  * */
  ObjectTracker(const std::unordered_set<std::string>& objectClasses,
                const cv::Matx34f& extP, const cv::Matx33f& intP,
                const std::string& detectionModel = "yolo",
                float minConfidence = 0.5, float minOverlap = 0.3);

  /**
   * @brief: Responsible for tracking the objects through every frame
   * @param: frame 
   * @return: Vector of 3D coordinates
   * */
  std::vector<cv::Point3f> localizeObjects(cv::Mat frame);

  /**
   * @brief: Responsible for getting the 3D coordinates of the object
   * @param: const reference to 2D image coordinates of the Object
   * @return: 3D coordinate 
   * */
  cv::Point3f localizeObjectKeypoint(const cv::Point2i& object) const;
  
  /**
   * @brief: Detects the objects in the frame by employing a DNN
   * @param: frame
   * @return: vector of 2D image points of the bounding boxes
   * */
  std::vector<cv::Point2i> detectObjectKeypoints(cv::Mat frame);

  std::vector<std::string> datasetLabels_;

 private:
  std::vector<std::string> parseFile(const std::string& fileName) const;
  std::unordered_set<std::string> objectClasses_;
  float minConfidence_, minOverlap_;
  cv::dnn::DetectionModel network_{"../yolo/yolov4.weights", "../yolo/yolov4.cfg"};
  cv::Matx34f P_;  // camera calibration matrix
  std::map<std::string, cv::Scalar> colors_;
};
}  // namespace ENPM808X
