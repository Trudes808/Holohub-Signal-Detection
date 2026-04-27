// SPDX-FileCopyrightText: 2026 National Instruments Corporation
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime_api.h>

#include <memory>
#include <string>

namespace holoscan::ops {

struct DinoCudaGrayBatch {
  const float* data = nullptr;
  std::shared_ptr<void> owner;
};

bool prepare_tensorrt_grayscale_batch_cuda(const float* resized_batch_device,
                                           int batch_size,
                                           int rows,
                                           int cols,
                                           bool legacy_fast_gray_preprocess,
                                           cudaStream_t cuda_stream,
                                           DinoCudaGrayBatch* output,
                                           std::string* error_message);

}  // namespace holoscan::ops