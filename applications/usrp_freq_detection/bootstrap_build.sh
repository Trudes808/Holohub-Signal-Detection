#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR=/workspace/holohub/build/usrp_freq_detection
SRC_DIR=/workspace/holohub
BIN_PATH=${BUILD_DIR}/applications/usrp_freq_detection/usrp_freq_detection

cmake -B "${BUILD_DIR}" \
  -S "${SRC_DIR}" \
  --no-warn-unused-cli \
  -DPython3_EXECUTABLE=/usr/bin/python3 \
  -DPython3_ROOT_DIR=/usr \
  -DPython3_INCLUDE_DIR=/usr/include/python3.10 \
  -DPython3_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.10.so \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/opt/nvidia/holoscan/lib \
  -DHOLOHUB_DATA_DIR:PATH=/workspace/holohub/data \
  -DAPP_usrp_freq_detection=ON \
  -DBUILD_PYTHON_BINDINGS=OFF \
  -G Ninja

cmake --build "${BUILD_DIR}" -j
ls -l "${BIN_PATH}"

grep -E "address:|num_bufs:|batch_size:|num_ffts_per_batch:|num_bursts:" \
  /workspace/holohub/applications/usrp_freq_detection/config.yaml
