# SPDX-FileCopyrightText: 2026 The University of Texas at Austin
#
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.20)
project(power_detection_tuned LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES "70;80;90")

find_package(holoscan 2.5.0 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")
find_package(CUDAToolkit REQUIRED)
find_package(matx CONFIG REQUIRED)

add_library(power_detection_tuned
  power_detection_tuned.cu
  power_detection_tuned.hpp
)
add_library(holoscan::ops::power_detection_tuned ALIAS power_detection_tuned)
target_include_directories(power_detection_tuned INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(power_detection_tuned
  PRIVATE
    CUDA::cudart
    CUDA::cufft
    holoscan::core
    matx::matx
)

target_compile_features(power_detection_tuned PRIVATE cxx_std_17)

install(TARGETS power_detection_tuned)
