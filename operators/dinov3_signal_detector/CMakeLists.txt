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
  COMMAND ${Python3_EXECUTABLE} -c "from pathlib import Path; import sys; hints=[];\ntry:\n import torch, torch.utils\n cmake_prefix=getattr(torch.utils, 'cmake_prefix_path', '')\n torch_file=Path(torch.__file__).resolve()\n torch_dir=torch_file.parent\n candidates=[cmake_prefix, str(torch_dir), str(torch_dir / 'share' / 'cmake'), str(torch_dir / 'share' / 'cmake' / 'Torch'), str(torch_dir.parent), str(torch_dir.parent / 'share' / 'cmake'), str(torch_dir.parent / 'share' / 'cmake' / 'Torch')]\n for item in candidates:\n  if item and item not in hints:\n   hints.append(item)\n print(';'.join(hints))\nexcept Exception:\n print('')"
  OUTPUT_VARIABLE PYTORCH_CMAKE_HINTS
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET
)

set(TORCH_SEARCH_HINTS "")

if(PYTORCH_CMAKE_HINTS)
  string(REPLACE ";" ";" PYTORCH_CMAKE_HINTS_LIST "${PYTORCH_CMAKE_HINTS}")
  foreach(TORCH_HINT IN LISTS PYTORCH_CMAKE_HINTS_LIST)
    if(TORCH_HINT AND EXISTS "${TORCH_HINT}")
      list(APPEND TORCH_SEARCH_HINTS "${TORCH_HINT}")
    endif()
  endforeach()
endif()

list(REMOVE_DUPLICATES TORCH_SEARCH_HINTS)

if(TORCH_SEARCH_HINTS)
  list(PREPEND CMAKE_PREFIX_PATH ${TORCH_SEARCH_HINTS})
  message(STATUS "dinov3_signal_detector: Torch search hints=${TORCH_SEARCH_HINTS}")
else()
  message(STATUS "dinov3_signal_detector: No Python Torch CMake hints detected from ${Python3_EXECUTABLE}")
endif()

find_package(Torch QUIET PATHS ${TORCH_SEARCH_HINTS})

if(NOT Torch_FOUND)
  find_package(Torch QUIET)
endif()

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
