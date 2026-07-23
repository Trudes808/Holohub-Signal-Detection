# SPDX-FileCopyrightText: 2026 National Instruments Corporation
#
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.20)
project(mask_replay_detector LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES "70;80;90")

find_package(holoscan 2.5.0 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")
find_package(matx CONFIG REQUIRED)   # only for the spectrogram-tensor input port type
find_package(CUDAToolkit REQUIRED)

# No torch / no dinov3 runtime: this operator only reads .npy files + emits a
# DetectorMaskMessage. matx is pulled in solely so the "in" port type matches the
# spectrogram tuple the graph delivers (payload is ignored).
add_library(mask_replay_detector
  mask_replay_detector.cu
  mask_replay_detector.hpp
)

add_library(holoscan::ops::mask_replay_detector ALIAS mask_replay_detector)
target_include_directories(mask_replay_detector INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(mask_replay_detector
  PRIVATE
    holoscan::core
    matx::matx
    CUDA::cudart
)

# std::filesystem: implicit on gcc>=9 / clang libc++. TODO(lab-admin): if the
# build errors on filesystem symbols, add `stdc++fs` to the link libraries above.

target_compile_features(mask_replay_detector PRIVATE cxx_std_17)

install(TARGETS mask_replay_detector)
