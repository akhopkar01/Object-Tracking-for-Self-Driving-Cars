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
 * @file      tracker.h
 * @author    Mahmoud Dahmani (Driver)
 * @author    Aditya Khopkar (Navigator)
 * @copyright MIT License
 * @brief     ObjectTracker class specification
 */

#pragma once

#include <string>
#include <vector>
#include <unordered_set>
#include <map>
#include <tuple>
#include <opencv2/dnn.hpp>

/**
 * @brief Class for monocular multiple object tracking for a mobile robot
 */
namespace ENPM808X {
class ObjectTracker {
 public:
  /**
   * @brief Constructor
   * @param object categories, camera extrinsics, camera intrinsics, 
   * detection model, minimum confidence, minimum overlap
   * @return None
   */
  ObjectTracker(const std::unordered_set<std::string>& objectClasses,
                const cv::Matx34f& extP, const cv::Matx33f& intP,
                const std::string& detectionModel = "yolo",
                float minConfidence = 0.5, float minOverlap = 0.3);

  /**
   * @brief Function for tracking all specified objects through every frame
   * @param frame
   * @return vector of the objects' 3D locations in the robot coordinate system
   */
  std::vector<cv::Point3f> localizeObjects(cv::Mat frame);

  /**
   * @brief Function for estimating the 3D location of a given pixel
   * @param pixel in the image
   * @return 3D location in the robot coordinate system
   */
  cv::Point3f localize(const cv::Point2i& objectKeypoint) const;

  /**
   * @brief Function for detecting objects in a given frame
   * @param frame
   * @return classIds, confidences, bounding boxes
   */
  std::tuple<std::vector<int>, std::vector<float>, std::vector<cv::Rect>>
  detectObjects(cv::Mat frame);

  /**
   * @brief Getter method for the classes of the pre-trained detection model
   * @param void
   * @return vector of the the dataset classes
   */
  std::vector<std::string> datasetLabels() const;

 private:
  void visualize(cv::Mat frame, int classId, float confidence,
                 const cv::Rect& box, const cv::Point2i& objectKeypoint,
                 const cv::Point3f& objectLocation, float distance);
  std::vector<std::string> parseFile(const std::string& fileName) const;

  std::unordered_set<std::string> objectClasses_;
  float minConfidence_, minOverlap_;
  std::vector<std::string> datasetLabels_;
  cv::dnn::DetectionModel network_{
      "../models/yolo/yolov4.weights",
      "../models/yolo/yolov4.cfg"};
  cv::Matx34f P_;  // camera calibration matrix
  std::map<std::string, cv::Scalar> colors_;
};
}  // namespace ENPM808X
