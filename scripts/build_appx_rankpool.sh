#! /bin/bash
# Author: Daniel Kurniadi

## Define directories
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" > /dev/null 2>&1 && pwd )"
PROJ_DIR="$( dirname "${SCRIP_DIR}")"

## Opencv installation directory
OPENCV_INSTALL_PREFIX="/home/saa2/saa-workplace/opencv_build_files/installation/opencv-3.4.4/share/OpenCV"

## C++ main file
CPP_MAIN_FILE="${PROJ_DIR}/appxrankpool.cpp"

echo ".. creating build/ directory at $PROJ_DIR/build"

## Delete and recreate build/ dir
if [ -d build ]; then rm -rf build; fi
mkdir build && cd build

echo ".. cmake compiling cpp code: $CPP_MAIN_FILE"
echo ".. cmake using opencv installation build: $OPENCV_INSTALL_PREFIX"

## CMake compile
cmake -DEXEC_CPP_FILE=$CPP_MAIN_FILE \
-DOPENCV_CMAKE_INSTALL_PREFIX=$OPENCV_INSTALL_PREFIX ..

## Make install
make -j8

echo ".. done"

