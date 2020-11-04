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
 * @file      profiler.cpp
 * @author    Mahmoud Dahmani (Driver)
 * @author    Aditya Khopkar (Navigator)
 * @copyright MIT License
 * @brief     Profiler class specification
 */

#include "profiler.h"

#include <cstdio>

ENPM808X::ProfileResult ENPM808X::Timer::profileResult;

ENPM808X::Timer::Timer(const char* name) : name_(name), stopped_(false) {
  startTimepoint_ = std::chrono::steady_clock::now();
}

void ENPM808X::Timer::Stop() {
  auto endTimepoint = std::chrono::steady_clock::now();
  auto elapsedTime =
      std::chrono::time_point_cast<std::chrono::milliseconds>(endTimepoint)
          .time_since_epoch() -
      std::chrono::time_point_cast<std::chrono::milliseconds>(startTimepoint_)
          .time_since_epoch();
  profileResult = {name_, elapsedTime.count()};
  if (name_ != "unit test")
    printf("%s: time = %lims,  FPS = %.1f\n", profileResult.name.c_str(),
            profileResult.elapsedTime, 1000.0f / profileResult.elapsedTime);
  stopped_ = true;
}

ENPM808X::Timer::~Timer() {
  if (!stopped_)
    Stop();
}
