# SPDX-FileCopyrightText: 2026 National Instruments Corporation
#
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.20)
project(coherent_power_signal_detector CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES "70;80;90")
enable_language(CUDA)

find_package(holoscan 2.5.0 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")
find_package(matx CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_library(coherent_power_signal_detector
  coherent_power_signal_detector.cu
  coherent_power_signal_detector.hpp
)

add_library(holoscan::ops::coherent_power_signal_detector ALIAS coherent_power_signal_detector)
target_include_directories(coherent_power_signal_detector INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(coherent_power_signal_detector
  PRIVATE
    CUDA::cudart
    holoscan::core
    matx::matx
)

target_compile_features(coherent_power_signal_detector PRIVATE cxx_std_17)

install(TARGETS coherent_power_signal_detector)