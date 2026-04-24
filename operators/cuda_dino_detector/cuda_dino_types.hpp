// SPDX-FileCopyrightText: 2026 National Instruments Corporation
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda/std/__algorithm/max.h>
#include <cuda/std/complex>
#include <cuda_runtime.h>
#include <matx.h>

#include <tuple>

namespace holoscan::ops {

using cuda_dino_complex = cuda::std::complex<float>;
using cuda_dino_in_t = std::tuple<matx::tensor_t<cuda_dino_complex, 2>, cudaStream_t>;

}  // namespace holoscan::ops
