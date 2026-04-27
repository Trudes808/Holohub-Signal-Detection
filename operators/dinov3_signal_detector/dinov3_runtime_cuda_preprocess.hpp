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

struct DinoCudaScoreBatch {
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

bool project_tensorrt_raw_score_batch_cuda(const float* patch_features_batch_device,
                                           int batch_size,
                                           int patch_rows,
                                           int patch_cols,
                                           int feature_dim,
                                           int aligned_rows,
                                           int aligned_cols,
                                           int output_rows,
                                           int output_cols,
                                           float positional_suppression,
                                           bool resized_full_chunk,
                                           cudaStream_t cuda_stream,
                                           float* output_score_device,
                                           DinoCudaScoreBatch* output,
                                           std::string* error_message);

}  // namespace holoscan::ops