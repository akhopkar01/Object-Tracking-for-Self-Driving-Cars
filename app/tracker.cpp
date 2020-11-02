/*******************************************************************************
 MIT License

 Copyright (c) 2020 Mahmoud Dahmani, Aditya Khopkar

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.
*******************************************************************************/

/**
 * @file      tracker.cpp
 * @author    Mahmoud Dahmani (Driver)
 * @author    Aditya Khopkar (Navigator)
 * @copyright MIT License
 * @brief     ObjectTracker class implementation
 */

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
  datasetLabels_ = {parseFile("../models/" +
                              detectionModel + "/coco.names")};
  cv::Size inputResolution{320, 320};
  // network_ = cv::dnn::DetectionModel(detectionModel + "/yolov4.weights",
  //                                    detectionModel + "/yolov4.cfg");
  network_.setInputSize(inputResolution);
  network_.setInputScale(1.0 / 255.0);
  network_.setInputSwapRB(true);
  network_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
  network_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
  P_ = intP * extP;
  colors_ = {{"safe", {0, 255, 0}}, {"unsafe", {0, 0, 255}}};
}

std::vector<cv::Point3f> ENPM808X::ObjectTracker::localizeObjects(
    cv::Mat frame) {
  std::vector<cv::Point3f> objectLocations;
  std::vector<cv::Point2i> objectKeypoints;

  objectKeypoints = detectObjectKeypoints(frame);
  objectLocations.reserve(objectKeypoints.size());
  for (const auto& p : objectKeypoints)
    objectLocations.emplace_back(localizeObjectKeypoint(p));

  return objectLocations;
}

// Stub Implementation
cv::Point3f ENPM808X::ObjectTracker::localizeObjectKeypoint(
    const cv::Point2i& object) const {
  // cv::Point3f dummyOutput{-2, -0.975, 0};
  cv::Point3f dummyOutput{0, 0, 0};
  cv::Matx31f imgpoints{(float)object.x, (float)object.y, 1};
  cv::Mat b, c;
  cv::Mat Pmatrix = cv::Mat(P_);
  Pmatrix(cv::Range(0, Pmatrix.rows), cv::Range(0, 2)).copyTo(b);
  Pmatrix(cv::Range(0, Pmatrix.rows), cv::Range(Pmatrix.cols - 1, Pmatrix.cols))
      .copyTo(c);
  cv::hconcat(b, c, b);
  cv::Mat coords = b.inv() * imgpoints;
  float lastElement = coords.at<float>(2);
  coords /= lastElement;
  dummyOutput.x = coords.at<float>(0);
  dummyOutput.y = coords.at<float>(1);
  return dummyOutput;
}

std::vector<cv::Point2i> ENPM808X::ObjectTracker::detectObjectKeypoints(
    cv::Mat frame) {
  std::vector<cv::Point2i> objectKeypoints;
  std::vector<int> classIds;
  std::vector<float> confidences;
  std::vector<cv::Rect> boxes;

  objectKeypoints.reserve(50);  // guess: there won't be more than 50 detections
  network_.detect(frame, classIds, confidences, boxes, minConfidence_,
                  minOverlap_);

  char caption[50];  // buffer of size 50
  cv::Point robot{frame.cols / 2, frame.rows}, groundMidpoint;
  for (int i = 0; i < boxes.size(); ++i)
    if (objectClasses_.find(datasetLabels_[classIds[i]]) !=
        objectClasses_.end()) {
      auto [x, y, w, h] = boxes[i];
      objectKeypoints.emplace_back(x + w / 2, y + h);
      groundMidpoint = {x + w / 2, y + h};
      snprintf(caption, sizeof(caption), "%s: %.2f",
               datasetLabels_[classIds[i]].c_str(), confidences[i]);
      // auto yo = cv::getTextSize(caption, cv::LINE_AA, 0.5, 2);
      cv::rectangle(frame, boxes[i], colors_["safe"], 2);
      cv::circle(frame, groundMidpoint, 5, colors_["safe"], cv::FILLED);
      cv::line(frame, robot, groundMidpoint, colors_["safe"], 2);
      cv::putText(frame, caption, cv::Point(x, y + 15), cv::LINE_AA, 0.5,
                  colors_["safe"], 2);
    }

  return objectKeypoints;
}

std::vector<std::string> ENPM808X::ObjectTracker::parseFile(
    const std::string& fileName) const {
  std::vector<std::string> classLabels;
  std::ifstream cocoNames(fileName);
  std::string line;

  classLabels.reserve(80);  // there are 80 classes in COCO
  if (cocoNames) {
    while (getline(cocoNames, line)) classLabels.emplace_back(line);
    cocoNames.close();
  } else {
    std::cout << "file could not be opened !\n";
    std::cin.get();
    exit(0);
  }

  return classLabels;
}

std::vector<std::string> ENPM808X::ObjectTracker::datasetLabels() const {
  return datasetLabels_;
}
