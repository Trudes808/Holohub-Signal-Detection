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
find_package(CUDAToolkit REQUIRED)
find_package(Python3 COMPONENTS Interpreter REQUIRED)

execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import torch.utils; print(torch.utils.cmake_prefix_path)"
  OUTPUT_VARIABLE PYTORCH_CMAKE_PREFIX_PATH
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET
)

if(PYTORCH_CMAKE_PREFIX_PATH)
  list(PREPEND CMAKE_PREFIX_PATH ${PYTORCH_CMAKE_PREFIX_PATH})
endif()

find_package(Torch QUIET)

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

if(Torch_FOUND)
  target_link_libraries(dinov3_signal_detector PRIVATE ${TORCH_LIBRARIES})

  if(TARGET CUDA::nvJitLink)
    target_link_libraries(dinov3_signal_detector PRIVATE CUDA::nvJitLink)
  else()
    get_filename_component(CUDAToolkit_ROOT_DIR "${CUDAToolkit_BIN_DIR}" DIRECTORY)
    set(NVJITLINK_LIBRARY "")
    foreach(NVJITLINK_CANDIDATE
      "${CUDAToolkit_ROOT_DIR}/targets/x86_64-linux/lib/libnvJitLink.so.12"
      "${CUDAToolkit_ROOT_DIR}/lib64/libnvJitLink.so.12"
      "/usr/local/cuda-12.6/targets/x86_64-linux/lib/libnvJitLink.so.12"
      "/usr/local/cuda-12.6/lib64/libnvJitLink.so.12")
      if(EXISTS "${NVJITLINK_CANDIDATE}")
        set(NVJITLINK_LIBRARY "${NVJITLINK_CANDIDATE}")
        break()
      endif()
    endforeach()

    if(NVJITLINK_LIBRARY)
      target_link_libraries(dinov3_signal_detector PRIVATE ${NVJITLINK_LIBRARY})
    else()
      message(WARNING "dinov3_signal_detector: nvJitLink library not found; Torch linkage may fail")
    endif()
  endif()

  target_compile_options(dinov3_signal_detector PRIVATE ${TORCH_CXX_FLAGS})
  target_compile_definitions(dinov3_signal_detector PRIVATE HOLOHUB_HAS_TORCH=1)
  target_compile_features(dinov3_signal_detector PRIVATE cxx_std_17)
  message(STATUS "dinov3_signal_detector: Torch found, enabling PyTorch backend path")
else()
  message(STATUS "dinov3_signal_detector: Torch not found, building CUDA fallback path only")
endif()

install(TARGETS dinov3_signal_detector)
