# SPDX-FileCopyrightText: 2025 National Instruments Corporation
#
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.20)
project(usrp_freq_detection CXX CUDA)

find_package(holoscan 2.5.0 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")
find_package(matx CONFIG REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES "70;80;90")
enable_language(CUDA)

add_subdirectory(CHDR_converter)

add_executable(usrp_freq_detection
  main.cpp
)

target_link_libraries(usrp_freq_detection PRIVATE
  matx::matx
  holoscan::core
  holoscan::advanced_network
  holoscan::ops::fft
  CHDR_converter
)
target_include_directories(usrp_freq_detection PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)

add_custom_target(config_yaml
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/config.yaml ${CMAKE_CURRENT_BINARY_DIR}/config.yaml
)

add_dependencies(usrp_freq_detection config_yaml)
