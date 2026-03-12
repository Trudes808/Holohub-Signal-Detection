# SPDX-FileCopyrightText: 2026 National Instruments Corporation
#
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.20)
project(usrp_freq_detection CXX CUDA)

# Dependencies
find_package(holoscan 2.6 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")
# MatX to be installed globally to work with find_package()
find_package(matx CONFIG REQUIRED)
# Alternatively, fetch the sources directly as MatX is header only
#include(FetchContent)
#FetchContent_Declare(
#  MatX
#  GIT_REPOSITORY https://github.com/NVIDIA/MatX.git
#  GIT_TAG v0.9.0
#)
#FetchContent_MakeAvailable(MatX)

# CUDA settings
set(CMAKE_CUDA_ARCHITECTURES "70;80;90")
enable_language(CUDA)

# Custom operator dependency
add_subdirectory(CHDR_converter)

# Create the executable
add_executable(usrp_freq_detection
  main.cpp
)
target_link_libraries(usrp_freq_detection
  PRIVATE
  matx::matx
  holoscan::core
  holoscan::advanced_network
  holoscan::ops::fft
  CHDR_converter
)
target_include_directories(usrp_freq_detection
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)

# Configure manager support through compile definitions
if(NOT DEFINED ANO_MGR)
    # Initialize ANO_MGR with a default if not provided
    # Note: rivermax not added to the default list yet since it requires licensing
    set(ANO_MGR "dpdk gpunetio" CACHE STRING "Manager type(s) list")
endif()
separate_arguments(ANO_MGR_LIST UNIX_COMMAND ${ANO_MGR})
foreach(MGR IN LISTS ANO_MGR_LIST)
    message(STATUS "Adding manager ${MGR} to the configuration")
    target_compile_definitions(${PROJECT_NAME} PUBLIC "ANO_MGR_${MGR}=1")
endforeach()

# Copy config file
add_custom_target(config_yaml
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/config.yaml ${CMAKE_CURRENT_BINARY_DIR}/config.yaml
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/config.yaml"
)
add_dependencies(usrp_freq_detection config_yaml)
