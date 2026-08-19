# SPDX-FileCopyrightText: 2026 National Instruments Corporation
#
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.20)
project(finetuned_dino_detector LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES "90;121")

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

# --- Torch discovery (mirrors operators/cuda_dino_detector/CMakeLists.txt) ---
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "from pathlib import Path; import sys; hints=[];\ntry:\n import torch, torch.utils\n cmake_prefix=getattr(torch.utils, 'cmake_prefix_path', '')\n torch_file=Path(torch.__file__).resolve()\n torch_dir=torch_file.parent\n candidates=[cmake_prefix, str(torch_dir), str(torch_dir / 'share' / 'cmake'), str(torch_dir / 'share' / 'cmake' / 'Torch'), str(torch_dir.parent), str(torch_dir.parent / 'share' / 'cmake'), str(torch_dir.parent / 'share' / 'cmake' / 'Torch')]\n for item in candidates:\n  if item and item not in hints:\n   hints.append(item)\n print(';'.join(hints))\nexcept Exception:\n print('')"
  OUTPUT_VARIABLE PYTORCH_CMAKE_HINTS OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "from pathlib import Path; import sys; includes=[];\ntry:\n import torch\n torch_dir=Path(torch.__file__).resolve().parent\n candidates=[torch_dir / 'include', torch_dir / 'include' / 'torch' / 'csrc' / 'api' / 'include', torch_dir.parent / 'include', torch_dir.parent / 'include' / 'torch' / 'csrc' / 'api' / 'include']\n for item in candidates:\n  if item.exists():\n   includes.append(str(item))\n print(';'.join(dict.fromkeys(includes)))\nexcept Exception:\n print('')"
  OUTPUT_VARIABLE PYTORCH_INCLUDE_HINTS OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)

set(TORCH_SEARCH_HINTS "")
foreach(TORCH_HINT IN LISTS PYTORCH_CMAKE_HINTS)
  if(TORCH_HINT AND EXISTS "${TORCH_HINT}")
    list(APPEND TORCH_SEARCH_HINTS "${TORCH_HINT}")
  endif()
endforeach()
list(REMOVE_DUPLICATES TORCH_SEARCH_HINTS)
set(TORCH_INCLUDE_HINTS "")
foreach(H IN LISTS PYTORCH_INCLUDE_HINTS)
  if(H AND EXISTS "${H}")
    list(APPEND TORCH_INCLUDE_HINTS "${H}")
  endif()
endforeach()
list(REMOVE_DUPLICATES TORCH_INCLUDE_HINTS)
if(TORCH_SEARCH_HINTS)
  list(PREPEND CMAKE_PREFIX_PATH ${TORCH_SEARCH_HINTS})
endif()
# GB10 (sm_121 / capability 12.1): Torch's Caffe2 CUDA autodetect misparses 12.1 and injects a
# bogus `-gencode arch=compute_20,code=sm_121` (nvcc fatal: Unsupported gpu architecture). Pin
# TORCH_CUDA_ARCH_LIST before find_package(Torch) so the flags are valid without relying on the
# TORCH_CUDA_ARCH_LIST environment variable being exported by the build wrapper.
set(TORCH_CUDA_ARCH_LIST "9.0;12.1")
find_package(Torch QUIET PATHS ${TORCH_SEARCH_HINTS})
if(NOT Torch_FOUND)
  find_package(Torch QUIET)
endif()

add_library(finetuned_dino_detector
  finetuned_dino_detector.cu
  finetuned_dino_torch_helpers.cpp
  finetuned_dino_detector.hpp
  finetuned_dino_torch_helpers.hpp
)
add_library(holoscan::ops::finetuned_dino_detector ALIAS finetuned_dino_detector)
target_include_directories(finetuned_dino_detector INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(finetuned_dino_detector
  PRIVATE
    holoscan::core
    matx::matx
    CUDA::cudart
)

if(Torch_FOUND)
  target_link_libraries(finetuned_dino_detector PRIVATE ${TORCH_LIBRARIES})
  foreach(D IN LISTS TORCH_INCLUDE_DIRS TORCH_INCLUDE_HINTS)
    target_include_directories(finetuned_dino_detector PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${D}>)
  endforeach()

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
  endif()
  if(OP_NVJITLINK_LINK_ITEM)
    target_link_libraries(finetuned_dino_detector PRIVATE ${OP_NVJITLINK_LINK_ITEM})
  endif()

  list(REMOVE_DUPLICATES CUDA_LINK_DIRS)
  target_link_directories(finetuned_dino_detector PRIVATE ${CUDA_LINK_DIRS})
  foreach(CUDA_LINK_DIR IN LISTS CUDA_LINK_DIRS)
    if(EXISTS "${CUDA_LINK_DIR}")
      target_link_options(finetuned_dino_detector PRIVATE "LINKER:-rpath-link,${CUDA_LINK_DIR}")
    endif()
  endforeach()

  target_compile_options(finetuned_dino_detector PRIVATE ${TORCH_CXX_FLAGS})
  target_compile_definitions(finetuned_dino_detector PRIVATE HOLOHUB_HAS_TORCH=1)
endif()

target_compile_features(finetuned_dino_detector PRIVATE cxx_std_17)
install(TARGETS finetuned_dino_detector)
