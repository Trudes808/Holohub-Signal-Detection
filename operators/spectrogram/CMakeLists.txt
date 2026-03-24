# SPDX-FileCopyrightText: 2026 National Instruments Corporation
#
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.20)
project(spectrogram CXX)

set(CMAKE_CUDA_ARCHITECTURES "70;80;90")
enable_language(CUDA)

find_package(holoscan 2.5.0 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")
find_package(matx CONFIG REQUIRED)

add_library(spectrogram
  spectrogram.cu
  spectrogram.hpp
)
add_library(holoscan::ops::spectrogram ALIAS spectrogram)
target_include_directories(spectrogram INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(spectrogram
  PRIVATE
    holoscan::core
    matx::matx
)

install(TARGETS spectrogram)
