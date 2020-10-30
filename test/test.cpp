/**
 * Copyright 2020 Mahmoud Dahmani, Aditya Khopkar
 */
/**
 * @file: test.cpp
 * @brief: This file contains the unit tests of the software for detecting and tracking the humans
 * @author: Mahmoud Dahmani (driver), Aditya Khopkar (navigator)
 * */

#include <gtest/gtest.h>

#include "tracker.h"
#include <opencv2/highgui.hpp>

class ObjectTrackerTest : public ::testing::Test {
 protected:
 /**
  * @brief: create override method for SetUp of extrinsic and intrinsic parameter for the derived test class along with instantiation
  * @param: None
  * @return: None
  * */
  void SetUp() override {
    std::unordered_set<std::string> objectClasses{"person"};
    cv::Matx34f extP{0, 0, 1, -1, 1, 0, 0, 0, 0, 1, 0, 1};
    cv::Matx33f intP{0.5, 0, 160, 0, 0.5, 160, 0, 0, 1};
    tracker = new ENPM808X::ObjectTracker(objectClasses, extP, intP);
    P = intP * extP;
  }

  /**
   * @brief: Responsible for cleaning up, deallocating memmory
   * @param: None
   * @return: None
   * */
  void TearDown() override { delete tracker; }

  ENPM808X::ObjectTracker* tracker;
  cv::Matx34f P;
};

/**
 * @brief: tests localization method (tracking) of the tracker class. By arbitrary setting pixel coordinate,
 * 3D coordinates of the point in real world is obtained which forms the ground truth. This ground truth is fed back
 * to reconstruct pixel in image frame. The two pixel values must match to pass this test
 * */
TEST_F(ObjectTrackerTest, LocalizationWorks) {
  cv::Point2f pixel{140.00014, 120.00011};
  cv::Point3f worldPoint{tracker->localizeObjectKeypoint(
      pixel)};  // ground thruth = [-2, -0.975, 0]
  cv::Matx31f pixel_true{static_cast<float>(pixel.x),
                         static_cast<float>(pixel.y), 1},
      pixel_reconstructed;
  cv::Matx41f X{worldPoint.x, worldPoint.y, worldPoint.z, 1};
  float abs_error = 0.001;
  pixel_reconstructed = P * X;
  pixel_reconstructed /= pixel_reconstructed(2);
  EXPECT_EQ(pixel_reconstructed, pixel_true);
}

/**
 * @brief: Checks the detection method of the tracker class - Multiple detection
 * Feeds a test image and asserts the number of detections to the number of humans in the image
 * */
/*
TEST_F(ObjectTrackerTest, MultipleHumanDetectionWorks) {
  cv::Mat frame{cv::imread("../testImage.png")};
  std::vector<cv::Point2i> detections = tracker->detectObjectKeypoints(frame);
  EXPECT_EQ(detections.size(), 2);  // there are 2 humans in the test image
}*/

/**
 * @brief: test if the cocolabels are correctly loaded. COCO dataset contians 80 labels. This is compared with the
 * dataset labels loaded in the software. 
 * */
TEST_F(ObjectTrackerTest, CocoLabelsAreRead) {
  ASSERT_EQ(tracker->datasetLabels_.size(), 80);  // there are 80 classes in COCO
  for (const auto& label : tracker->datasetLabels_) EXPECT_FALSE(label.empty());
}
