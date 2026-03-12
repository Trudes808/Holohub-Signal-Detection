# SPDX-FileCopyrightText: 2026 National Instruments Corporation
#
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.20)

# Dependencies
find_package(holoscan 3.1.0 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")
# MatX to be installed globally to work with find_package()
find_package(matx CONFIG REQUIRED)
# Alternatively, fetch the sources directly as MatX is header only
include(FetchContent)
#FetchContent_Declare(
#  MatX
#  GIT_REPOSITORY https://github.com/NVIDIA/MatX.git
#  GIT_TAG v0.9.0
#)
#FetchContent_MakeAvailable(MatX)

# CUDA settings
set(CMAKE_CUDA_ARCHITECTURES "70;80;90")
enable_language(CUDA)

# Create the library
add_library(CHDR_converter
  chdr_rx.cu
)
target_link_libraries(CHDR_converter
  PRIVATE
  matx::matx
  holoscan::core
  holoscan::advanced_network
)
target_include_directories(CHDR_converter
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/..
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../operators/advanced_network
)

# Configure manager support through compile definitions
#if(NOT DEFINED ANO_MGR)
#    # Initialize ANO_MGR with a default if not provided
#    # Note: rivermax not added to the default list yet since it requires licensing
#    set(ANO_MGR "dpdk gpunetio" CACHE STRING "Manager type(s) list")
#endif()
#separate_arguments(ANO_MGR_LIST UNIX_COMMAND ${ANO_MGR})
#foreach(MGR IN LISTS ANO_MGR_LIST)
#    message(STATUS "Adding manager ${MGR} to the configuration")
#    target_compile_definitions(${PROJECT_NAME} PUBLIC "ANO_MGR_${MGR}=1")
#endforeach()
