# SPDX-FileCopyrightText: 2026 National Instruments Corporation
#
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.20)
project(usrp_wideband_signal_detection CXX CUDA)

find_package(holoscan 2.6 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")
find_package(matx CONFIG REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES "70;80;90")
enable_language(CUDA)

set(CHDR_CONVERTER_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../usrp_freq_detection/CHDR_converter)
add_library(CHDR_converter_wideband
  ${CHDR_CONVERTER_SRC_DIR}/chdr_rx.cu
)
target_link_libraries(CHDR_converter_wideband
  PRIVATE
  matx::matx
  holoscan::core
  holoscan::advanced_network
)
target_include_directories(CHDR_converter_wideband
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../usrp_freq_detection
  ${CMAKE_CURRENT_SOURCE_DIR}/../../operators/advanced_network
)

add_executable(usrp_wideband_signal_detection
  main.cpp
)
target_link_libraries(usrp_wideband_signal_detection
  PRIVATE
  matx::matx
  holoscan::core
  holoscan::advanced_network
  holoscan::ops::fft
  holoscan::ops::spectrogram
  holoscan::ops::dinov3_signal_detector
  CHDR_converter_wideband
)
target_include_directories(usrp_wideband_signal_detection
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)

if(NOT DEFINED ANO_MGR)
    set(ANO_MGR "dpdk gpunetio" CACHE STRING "Manager type(s) list")
endif()
separate_arguments(ANO_MGR_LIST UNIX_COMMAND ${ANO_MGR})
foreach(MGR IN LISTS ANO_MGR_LIST)
    message(STATUS "Adding manager ${MGR} to the configuration")
    target_compile_definitions(${PROJECT_NAME} PUBLIC "ANO_MGR_${MGR}=1")
endforeach()

add_custom_target(config_yaml
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/config.yaml ${CMAKE_CURRENT_BINARY_DIR}/config.yaml
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/config.yaml"
)
add_dependencies(usrp_wideband_signal_detection config_yaml)
