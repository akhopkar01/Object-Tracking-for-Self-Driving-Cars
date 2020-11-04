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
 * @file      config.h
 * @author    Mahmoud Dahmani (Driver)
 * @author    Aditya Khopkar (Navigator)
 * @copyright MIT License
 * @brief     System configuration
 */

#pragma once

#include <string>
#include <unordered_set>
#include <opencv2/highgui.hpp>

static cv::VideoCapture stream(
    "../data/testVideo.mp4");  // alternatively a camera device index
                               // can be passed
static std::unordered_set<std::string> objectClasses{"person", "car"};
static cv::Matx34f extP{0, 0, 1, -1, 1, 0, 0, 0, 0, 1, 0, 1};
static cv::Matx33f intP{0.5, 0, 160, 0, 0.5, 160, 0, 0, 1};
