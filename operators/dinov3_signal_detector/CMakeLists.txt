# SPDX-FileCopyrightText: 2026 National Instruments Corporation
#
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.20)
project(dinov3_signal_detector CXX)

set(CMAKE_CUDA_ARCHITECTURES "70;80;90")
enable_language(CUDA)

find_package(holoscan 2.5.0 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")
find_package(matx CONFIG REQUIRED)

add_library(dinov3_signal_detector
  dinov3_signal_detector.cu
  dinov3_signal_detector.hpp
)
add_library(holoscan::ops::dinov3_signal_detector ALIAS dinov3_signal_detector)
target_include_directories(dinov3_signal_detector INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(dinov3_signal_detector
  PRIVATE
    holoscan::core
    matx::matx
)

install(TARGETS dinov3_signal_detector)
