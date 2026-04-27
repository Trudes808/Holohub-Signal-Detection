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

set(CUDA_LINK_DIRS
  "/usr/local/cuda/targets/x86_64-linux/lib"
  "/usr/local/cuda/lib64"
  "/usr/local/cuda-12.6/targets/x86_64-linux/lib"
  "/usr/local/cuda-12.6/lib64"
)

execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import importlib.util, pathlib; spec = importlib.util.find_spec('nvidia.nvjitlink'); print(pathlib.Path(spec.submodule_search_locations[0]) / 'lib' if spec and spec.submodule_search_locations else '')"
  OUTPUT_VARIABLE PY_NVJITLINK_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET
)

if(PY_NVJITLINK_DIR AND EXISTS "${PY_NVJITLINK_DIR}/libnvJitLink.so.12")
  list(APPEND CUDA_LINK_DIRS ${PY_NVJITLINK_DIR})
endif()

execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "from pathlib import Path; import sys; hints=[];\ntry:\n import torch, torch.utils\n cmake_prefix=getattr(torch.utils, 'cmake_prefix_path', '')\n torch_file=Path(torch.__file__).resolve()\n torch_dir=torch_file.parent\n candidates=[cmake_prefix, str(torch_dir), str(torch_dir / 'share' / 'cmake'), str(torch_dir / 'share' / 'cmake' / 'Torch'), str(torch_dir.parent), str(torch_dir.parent / 'share' / 'cmake'), str(torch_dir.parent / 'share' / 'cmake' / 'Torch')]\n for item in candidates:\n  if item and item not in hints:\n   hints.append(item)\n print(';'.join(hints))\nexcept Exception:\n print('')"
  OUTPUT_VARIABLE PYTORCH_CMAKE_HINTS
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET
)

execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "from pathlib import Path; import sys; includes=[];\ntry:\n import torch\n torch_dir=Path(torch.__file__).resolve().parent\n candidates=[torch_dir / 'include', torch_dir / 'include' / 'torch' / 'csrc' / 'api' / 'include', torch_dir.parent / 'include', torch_dir.parent / 'include' / 'torch' / 'csrc' / 'api' / 'include']\n for item in candidates:\n  if item.exists():\n   includes.append(str(item))\n print(';'.join(dict.fromkeys(includes)))\nexcept Exception:\n print('')"
  OUTPUT_VARIABLE PYTORCH_INCLUDE_HINTS
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

set(TORCH_INCLUDE_HINTS "")
if(PYTORCH_INCLUDE_HINTS)
  string(REPLACE ";" ";" PYTORCH_INCLUDE_HINTS_LIST "${PYTORCH_INCLUDE_HINTS}")
  foreach(TORCH_INCLUDE_HINT IN LISTS PYTORCH_INCLUDE_HINTS_LIST)
    if(TORCH_INCLUDE_HINT AND EXISTS "${TORCH_INCLUDE_HINT}")
      list(APPEND TORCH_INCLUDE_HINTS "${TORCH_INCLUDE_HINT}")
    endif()
  endforeach()
endif()
list(REMOVE_DUPLICATES TORCH_INCLUDE_HINTS)

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

find_library(DINOV3_TENSORRT_LIB
  NAMES nvinfer
  PATHS ${CUDA_LINK_DIRS} /usr/lib/x86_64-linux-gnu /usr/lib64 /usr/local/lib /usr/local/TensorRT/lib
)

set(DINOV3_TENSORRT_FOUND FALSE)
if(DINOV3_TENSORRT_LIB)
  set(DINOV3_TENSORRT_FOUND TRUE)
endif()

set(DINOV3_SIGNAL_DETECTOR_SOURCES
  dinov3_signal_detector.cu
  dinov3_signal_detector.hpp
  dinov3_runtime_cuda_preprocess.cu
  dinov3_runtime_cuda_preprocess.hpp
)

if(Torch_FOUND)
  add_library(dinov3_torch_runtime_obj OBJECT dinov3_torch_runtime.cpp)
  target_include_directories(dinov3_torch_runtime_obj
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
  )
  target_link_libraries(dinov3_torch_runtime_obj
    PRIVATE
      ${TORCH_LIBRARIES}
  )
  if(TORCH_INCLUDE_DIRS)
    target_include_directories(dinov3_torch_runtime_obj PRIVATE ${TORCH_INCLUDE_DIRS})
  endif()
  if(DINOV3_TENSORRT_FOUND)
    target_compile_definitions(dinov3_torch_runtime_obj PRIVATE HOLOHUB_HAS_TENSORRT=1)
  endif()
  if(TORCH_INCLUDE_HINTS)
    target_include_directories(dinov3_torch_runtime_obj PRIVATE ${TORCH_INCLUDE_HINTS})
    message(STATUS "dinov3_signal_detector: Torch include hints=${TORCH_INCLUDE_HINTS}")
  endif()
else()
  list(APPEND DINOV3_SIGNAL_DETECTOR_SOURCES dinov3_torch_runtime_stub.cpp)
endif()

add_library(dinov3_signal_detector
  ${DINOV3_SIGNAL_DETECTOR_SOURCES}
)

if(Torch_FOUND)
  target_sources(dinov3_signal_detector PRIVATE $<TARGET_OBJECTS:dinov3_torch_runtime_obj>)
endif()

add_library(holoscan::ops::dinov3_signal_detector ALIAS dinov3_signal_detector)
target_include_directories(dinov3_signal_detector INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(dinov3_signal_detector
  PRIVATE
    holoscan::core
    matx::matx
)

if(Torch_FOUND)
  target_link_libraries(dinov3_signal_detector PRIVATE ${TORCH_LIBRARIES})
  if(DINOV3_TENSORRT_FOUND)
    target_link_libraries(dinov3_signal_detector PRIVATE ${DINOV3_TENSORRT_LIB})
    target_compile_definitions(dinov3_signal_detector PRIVATE HOLOHUB_HAS_TENSORRT=1)
  endif()

  set(OP_NVJITLINK_LINK_ITEM "")
  if(TARGET CUDA::nvJitLink)
    set(OP_NVJITLINK_LINK_ITEM CUDA::nvJitLink)
  else()
    unset(OP_NVJITLINK_LINK_ITEM CACHE)
    find_library(OP_NVJITLINK_LINK_ITEM
      NAMES libnvJitLink.so.12 nvJitLink
      PATHS ${CUDA_LINK_DIRS}
      NO_DEFAULT_PATH
    )

    if(OP_NVJITLINK_LINK_ITEM)
      get_filename_component(OP_NVJITLINK_LINK_DIR "${OP_NVJITLINK_LINK_ITEM}" DIRECTORY)
      if(EXISTS "${OP_NVJITLINK_LINK_DIR}")
        list(APPEND CUDA_LINK_DIRS ${OP_NVJITLINK_LINK_DIR})
      endif()
    else()
      message(WARNING "dinov3_signal_detector: nvJitLink library not found; Torch linkage may fail")
    endif()
  endif()

  if(OP_NVJITLINK_LINK_ITEM)
    target_link_libraries(dinov3_signal_detector PRIVATE ${OP_NVJITLINK_LINK_ITEM})
  endif()

  list(REMOVE_DUPLICATES CUDA_LINK_DIRS)
  target_link_directories(dinov3_signal_detector PRIVATE ${CUDA_LINK_DIRS})
  foreach(CUDA_LINK_DIR IN LISTS CUDA_LINK_DIRS)
    if(EXISTS "${CUDA_LINK_DIR}")
      target_link_options(dinov3_signal_detector PRIVATE "LINKER:-rpath-link,${CUDA_LINK_DIR}")
    endif()
  endforeach()

  target_compile_definitions(dinov3_signal_detector PRIVATE HOLOHUB_HAS_TORCH=1)
  target_compile_features(dinov3_signal_detector PRIVATE cxx_std_17)

  target_compile_options(dinov3_torch_runtime_obj PRIVATE ${TORCH_CXX_FLAGS})
  target_compile_definitions(dinov3_torch_runtime_obj PRIVATE HOLOHUB_HAS_TORCH=1)
  target_compile_features(dinov3_torch_runtime_obj PRIVATE cxx_std_17)

  add_executable(dinov3_libtorch_sandbox
    dinov3_libtorch_sandbox.cpp
  )
  target_link_libraries(dinov3_libtorch_sandbox PRIVATE ${TORCH_LIBRARIES})
  if(TORCH_INCLUDE_DIRS)
    target_include_directories(dinov3_libtorch_sandbox PRIVATE ${TORCH_INCLUDE_DIRS})
  endif()
  if(TORCH_INCLUDE_HINTS)
    target_include_directories(dinov3_libtorch_sandbox PRIVATE ${TORCH_INCLUDE_HINTS})
  endif()
  if(OP_NVJITLINK_LINK_ITEM)
    target_link_libraries(dinov3_libtorch_sandbox PRIVATE ${OP_NVJITLINK_LINK_ITEM})
  endif()
  target_link_directories(dinov3_libtorch_sandbox PRIVATE ${CUDA_LINK_DIRS})
  foreach(CUDA_LINK_DIR IN LISTS CUDA_LINK_DIRS)
    if(EXISTS "${CUDA_LINK_DIR}")
      target_link_options(dinov3_libtorch_sandbox PRIVATE "LINKER:-rpath-link,${CUDA_LINK_DIR}")
    endif()
  endforeach()
  target_compile_options(dinov3_libtorch_sandbox PRIVATE ${TORCH_CXX_FLAGS})
  target_compile_features(dinov3_libtorch_sandbox PRIVATE cxx_std_17)

  message(STATUS "dinov3_signal_detector: Torch found, enabling PyTorch backend path")
  if(DINOV3_TENSORRT_FOUND)
    message(STATUS "dinov3_signal_detector: TensorRT found, enabling TensorRT engine backend path")
  else()
    message(STATUS "dinov3_signal_detector: TensorRT not found, TensorRT backend will be unavailable")
  endif()
else()
  message(STATUS "dinov3_signal_detector: Torch not found, building CUDA fallback path only")
endif()

install(TARGETS dinov3_signal_detector)
