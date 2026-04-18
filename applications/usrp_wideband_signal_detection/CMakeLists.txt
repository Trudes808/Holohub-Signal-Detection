# SPDX-FileCopyrightText: 2026 National Instruments Corporation
#
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.20)
project(usrp_wideband_signal_detection CXX CUDA)

find_package(holoscan 2.6 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")
find_package(matx CONFIG REQUIRED
             PATHS "/usr/local/lib/cmake/matx" "/usr/local")
find_package(CUDAToolkit REQUIRED)
find_package(Python3 COMPONENTS Interpreter REQUIRED)

find_library(APP_VULKAN_LINK_ITEM
  NAMES vulkan libvulkan.so.1
  PATHS
    /usr/lib/x86_64-linux-gnu
    /lib/x86_64-linux-gnu
    /usr/local/lib
    /usr/local/lib64
)

set(APP_SYSTEM_LINK_DIRS
  /usr/lib/x86_64-linux-gnu
  /lib/x86_64-linux-gnu
  /usr/local/lib
  /usr/local/lib64
)

if(NOT APP_VULKAN_LINK_ITEM)
  message(WARNING "usrp_wideband_signal_detection: Vulkan loader library not found; visualization binaries may fail to link")
endif()

set(CMAKE_CUDA_ARCHITECTURES "70;80;90")
enable_language(CUDA)

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

set(APP_NVJITLINK_LINK_ITEM "")
if(TARGET CUDA::nvJitLink)
  set(APP_NVJITLINK_LINK_ITEM CUDA::nvJitLink)
else()
  unset(APP_NVJITLINK_LINK_ITEM CACHE)
  find_library(APP_NVJITLINK_LINK_ITEM
    NAMES libnvJitLink.so.12 nvJitLink
    PATHS ${CUDA_LINK_DIRS}
    NO_DEFAULT_PATH
  )

  if(APP_NVJITLINK_LINK_ITEM)
    get_filename_component(APP_NVJITLINK_LINK_DIR "${APP_NVJITLINK_LINK_ITEM}" DIRECTORY)
    if(EXISTS "${APP_NVJITLINK_LINK_DIR}")
      list(APPEND CUDA_LINK_DIRS ${APP_NVJITLINK_LINK_DIR})
    endif()
  else()
    message(WARNING "usrp_wideband_signal_detection: libnvJitLink.so.12 not found for direct app linkage; relying on transitive linkage if required")
  endif()
endif()

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
  spectrogram_visualization.cu
  screenshot_sink.cpp
)
target_link_libraries(usrp_wideband_signal_detection
  PRIVATE
  matx::matx
  holoscan::ops::coherent_power_signal_detector
  CUDA::cudart
  holoscan::core
  holoscan::advanced_network
  holoscan::ops::fft
  holoscan::ops::holoviz
  holoscan::ops::spectrogram
  holoscan::ops::dinov3_signal_detector
  CHDR_converter_wideband
)

add_executable(offline_spectrogram_visualizer
  offline_spectrogram_replay.cpp
  spectrogram_visualization.cu
  screenshot_sink.cpp
)

add_executable(offline_coherent_power_validator
  offline_coherent_power_validator.cpp
)

target_link_libraries(offline_spectrogram_visualizer
  PRIVATE
  CUDA::cudart
  matx::matx
  holoscan::core
  holoscan::ops::holoviz
)

target_link_libraries(offline_coherent_power_validator
  PRIVATE
  holoscan::ops::coherent_power_signal_detector
)

if(APP_VULKAN_LINK_ITEM)
  target_link_libraries(usrp_wideband_signal_detection PRIVATE ${APP_VULKAN_LINK_ITEM})
  target_link_libraries(offline_spectrogram_visualizer PRIVATE ${APP_VULKAN_LINK_ITEM})
endif()

target_link_directories(offline_spectrogram_visualizer PRIVATE ${APP_SYSTEM_LINK_DIRS})
foreach(APP_LINK_DIR IN LISTS APP_SYSTEM_LINK_DIRS)
  if(EXISTS "${APP_LINK_DIR}")
    target_link_options(offline_spectrogram_visualizer PRIVATE "LINKER:-rpath-link,${APP_LINK_DIR}")
  endif()
endforeach()

if(APP_NVJITLINK_LINK_ITEM)
  target_link_libraries(usrp_wideband_signal_detection PRIVATE ${APP_NVJITLINK_LINK_ITEM})
endif()

target_link_directories(usrp_wideband_signal_detection PRIVATE ${CUDA_LINK_DIRS})
foreach(CUDA_LINK_DIR IN LISTS CUDA_LINK_DIRS)
  if(EXISTS "${CUDA_LINK_DIR}")
    target_link_options(usrp_wideband_signal_detection PRIVATE "LINKER:-rpath-link,${CUDA_LINK_DIR}")
  endif()
endforeach()

target_include_directories(usrp_wideband_signal_detection
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_include_directories(offline_spectrogram_visualizer
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_include_directories(offline_coherent_power_validator
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../../operators/coherent_power_signal_detector
)

target_compile_definitions(usrp_wideband_signal_detection
  PRIVATE
  USRP_WIDEBAND_APP_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

target_compile_definitions(offline_spectrogram_visualizer
  PRIVATE
  USRP_WIDEBAND_APP_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

target_compile_features(offline_coherent_power_validator PRIVATE cxx_std_17)

if(NOT DEFINED ANO_MGR)
    set(ANO_MGR "dpdk gpunetio" CACHE STRING "Manager type(s) list")
endif()
separate_arguments(ANO_MGR_LIST UNIX_COMMAND ${ANO_MGR})
foreach(MGR IN LISTS ANO_MGR_LIST)
    message(STATUS "Adding manager ${MGR} to the configuration")
    target_compile_definitions(${PROJECT_NAME} PUBLIC "ANO_MGR_${MGR}=1")
endforeach()

  file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/old_configs)

  set(USRP_WIDEBAND_CONFIG_FILES
    old_configs/config.yaml
    old_configs/config_coherent_power_debug_capture.yaml
    config_coherent_power_performance.yaml
    old_configs/config_coherent_power_realtime_guarded.yaml
    config_coherent_power_validation.yaml
    old_configs/config_offline_replay.yaml
    old_configs/config_cuda_fallback.yaml
    config_torchscript_performance.yaml
    config_torchscript_validation_spectrogram_only.yaml
    old_configs/config_torchscript_performance_timing_debug.yaml
    old_configs/config_torchscript_performance_timing_debug_fast_post.yaml
    old_configs/config_torchscript_performance_timing_debug_small_input.yaml
    old_configs/config_torchscript_performance_timing_debug_fp16.yaml
    old_configs/config_torchscript_realtime_guarded.yaml
    old_configs/config_torchscript_performance_fft_only_matched.yaml
    old_configs/config_torchscript_performance_fft_only.yaml
    old_configs/config_torchscript_performance_small_batches.yaml
    old_configs/config_torchscript_performance_spectrogram_only.yaml
    old_configs/config_torchscript_validation_single_channel.yaml
    old_configs/config_torchscript_cpu_eval.yaml
    old_configs/config_torchscript_cuda_no_eval.yaml
    old_configs/config_torchscript_validation.yaml
    old_configs/config_torchscript_load_only.yaml
  )

  set(USRP_WIDEBAND_CONFIG_OUTPUTS)
  foreach(CONFIG_FILE IN LISTS USRP_WIDEBAND_CONFIG_FILES)
    add_custom_command(
      OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_FILE}
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${CMAKE_CURRENT_SOURCE_DIR}/${CONFIG_FILE}
          ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_FILE}
      DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${CONFIG_FILE}
    )
    list(APPEND USRP_WIDEBAND_CONFIG_OUTPUTS ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_FILE})
  endforeach()

  add_custom_target(usrp_wideband_configs DEPENDS ${USRP_WIDEBAND_CONFIG_OUTPUTS})
  add_dependencies(usrp_wideband_signal_detection usrp_wideband_configs)
  add_dependencies(offline_coherent_power_validator usrp_wideband_configs)
