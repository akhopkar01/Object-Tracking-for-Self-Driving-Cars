#!/bin/sh

opencv_version=4.5.0
opencv_dir=~

## Step 1: Update Packages
sudo apt -y update
sudo apt -y upgrade

## Step 2: Install OS Libraries
sudo apt -y remove x264 libx264-dev
sudo apt -y autoremove
# Install dependencies
sudo apt -y install build-essential checkinstall cmake pkg-config yasm
sudo apt -y install git gfortran
sudo apt -y install software-properties-common
sudo add-apt-repository -y "deb http://security.ubuntu.com/ubuntu xenial-security main"
sudo apt -y install libjpeg8-dev libjasper-dev libpng12-dev
sudo apt -y install libtiff5-dev
sudo apt -y install libavcodec-dev libavformat-dev libswscale-dev libdc1394-22-dev
sudo apt -y install libxine2-dev libv4l-dev
#cd /usr/include/linux
#sudo ln -s -f ../libv4l1-videodev.h videodev.h
sudo apt -y install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
sudo apt -y install libgtk2.0-dev libtbb-dev qt5-default
sudo apt -y install libatlas-base-dev
sudo apt -y install libfaac-dev libmp3lame-dev libtheora-dev
sudo apt -y install libvorbis-dev libxvidcore-dev
sudo apt -y install libopencore-amrnb-dev libopencore-amrwb-dev
sudo apt -y install libavresample-dev
sudo apt -y install x264 v4l-utils
# Optional dependencies
sudo apt -y install libprotobuf-dev protobuf-compiler
sudo apt -y install libgoogle-glog-dev libgflags-dev
sudo apt -y install libgphoto2-dev libeigen3-dev libhdf5-dev doxygen

## Step 3: Download opencv and opencv_contrib
cd ~/Desktop/
# Clean build directories
rm -rf opencv/build
rm -rf opencv_contrib/build

git clone https://github.com/opencv/opencv.git
cd opencv/
git checkout ${opencv_version}
cd ../

#git clone https://github.com/opencv/opencv_contrib.git
#cd opencv_contrib/
#git checkout ${opencv_version}
#cd ../

## Step 4: Compile and install OpenCV with contrib modules
cd opencv/
mkdir build/
cd build/

cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=${opencv_dir}/OpenCV-${opencv_version} \
      -D INSTALL_C_EXAMPLES=ON \
      -D INSTALL_PYTHON_EXAMPLES=ON \
      -D WITH_TBB=ON \
      -D WITH_V4L=ON \
      -D WITH_QT=ON \
      -D WITH_OPENGL=ON \
      -D BUILD_EXAMPLES=ON \
      -D OPENCV_GENERATE_PKGCONFIG=YES ..

#make -j${nproc}
make -j2
sudo make install
sudo sh -c "echo '${opencv_dir}/OpenCV-${opencv_version}/lib' > /etc/ld.so.conf.d/opencv.conf"
sudo ldconfig
