#!/bin/sh

# default build configuration
build_config=Debug

if [ $# -ge 1 ]
then build_config=$1
fi
mkdir -p ../build/
cd ../build/
cmake -DCMAKE_BUILD_TYPE=${build_config} ../ && make -j${nproc}
cd ../scripts/
