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

#include <cstdio>
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
  network_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
  network_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
  network_.setInputSize(inputResolution);
  network_.setInputScale(1.0 / 255.0);
  network_.setInputSwapRB(true);
  P_ = intP * extP;
  colors_ = {{"safe", {0, 255, 0}}, {"unsafe", {0, 0, 255}}};
}

std::vector<cv::Point3f> ENPM808X::ObjectTracker::localizeObjects(
    cv::Mat frame) {
  std::vector<cv::Point3f> locations;
  cv::Point3f objectLocation;
  cv::Point2i objectKeypoint;
  cv::Point2f robotXY{5, 50}, objectXY;
  float distance;

  auto [classIds, confidences, boxes] = detectObjects(frame);

  locations.reserve(boxes.size());
  for (int i = 0; i < boxes.size(); ++i) {
    auto [x, y, w, h] = boxes[i];
    objectKeypoint = {x + w / 2, y + h};  // ground midpoint
    objectLocation = localize(objectKeypoint);
    objectXY = {objectLocation.x, objectLocation.y};
    distance = cv::norm(objectXY - robotXY);
    locations.emplace_back(objectLocation);
    visualize(frame, classIds[i], confidences[i], boxes[i],
              objectKeypoint, objectLocation, distance);
  }

  return locations;
}

cv::Point3f ENPM808X::ObjectTracker::localize(
    const cv::Point2i& objectKeypoint) const {
  cv::Matx31f worldPoint;

  // camera projection matrix P_ simplifies to a planar hoography H
  // as a result of restricting the 3D world points of interest to
  // the plane Z = 0 (i.e. road plane in the robot frame)
  cv::Matx33f H{P_(0, 0), P_(0, 1), P_(0, 3),
                P_(1, 0), P_(1, 1), P_(1, 3),
                P_(2, 0), P_(2, 1), P_(2, 3)};
  cv::Matx31f pixel{static_cast<float>(objectKeypoint.x),
                    static_cast<float>(objectKeypoint.y),
                    1};
  worldPoint = H.inv() * pixel;
  worldPoint /= worldPoint(2);

  return {worldPoint(0), worldPoint(1), 0};
}

std::tuple<std::vector<int>, std::vector<float>, std::vector<cv::Rect>>
ENPM808X::ObjectTracker::detectObjects(cv::Mat frame) {
  std::vector<int> classIds;
  std::vector<float> confidences;
  std::vector<cv::Rect> boxes;

  network_.detect(frame, classIds, confidences, boxes,
                  minConfidence_, minOverlap_);

  return {classIds, confidences, boxes};
}

std::vector<std::string> ENPM808X::ObjectTracker::datasetLabels() const {
  return datasetLabels_;
}

void ENPM808X::ObjectTracker::visualize(cv::Mat frame, int classId,
                                        float confidence, const cv::Rect& box,
                                        const cv::Point2i& keypoint,
                                        const cv::Point3f& location,
                                        float distance) {
  float safetyMargin {5};
  cv::Point robotKeypoint{frame.cols / 2, frame.rows};
  cv::Scalar color {(distance >= safetyMargin) ? colors_["safe"]
                                               : colors_["unsafe"]};
  char caption[50];  // buffer of size 50
  if (objectClasses_.find(datasetLabels_[classId]) !=
      objectClasses_.end()) {
    cv::rectangle(frame, box, color, 2);
    cv::circle(frame, keypoint, 5, color, cv::FILLED);
    cv::line(frame, robotKeypoint, keypoint, color, 2);

    snprintf(caption, sizeof(caption), "%s(%.1fm,%.1fm,0m)",
             datasetLabels_[classId].c_str(), location.x, location.y);
    cv::putText(frame, caption, cv::Point(box.x, box.y + 15),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1.5);
    // auto yo = cv::getTextSize(caption, cv::LINE_AA, 0.5, 2);
    snprintf(caption, sizeof(caption), "%.1fm", distance);
    cv::putText(frame, caption, cv::Point(box.x + box.width/2, box.y - 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1.5);
  }
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
