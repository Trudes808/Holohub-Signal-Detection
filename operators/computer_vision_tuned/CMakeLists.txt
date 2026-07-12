# SPDX-FileCopyrightText: 2026 The University of Texas at Austin
#
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.20)
project(computer_vision_tuned LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES "70;80;90")

find_package(holoscan 2.5.0 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")
find_package(CUDAToolkit REQUIRED)
find_package(matx CONFIG REQUIRED)

add_library(computer_vision_tuned
  computer_vision_tuned.cu
  computer_vision_tuned.hpp
)
add_library(holoscan::ops::computer_vision_tuned ALIAS computer_vision_tuned)
target_include_directories(computer_vision_tuned INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(computer_vision_tuned
  PRIVATE
    CUDA::cudart
    holoscan::core
    matx::matx
)

target_compile_features(computer_vision_tuned PRIVATE cxx_std_17)

install(TARGETS computer_vision_tuned)
