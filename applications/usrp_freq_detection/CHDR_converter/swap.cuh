/*
 * SPDX-FileCopyrightText: 2025 Valley Tech Systems, Inc.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <cuda_runtime.h>
#include <cuda/std/complex>

inline __device__ uint16_t bswap_16(const uint16_t val) {
    return (uint16_t)__byte_perm((uint32_t)val, 0, 0x01);
}

inline __device__ int16_t bswap_16(const int16_t val) {
    const uint16_t v = bswap_16(*((uint16_t*)&val));
    return *((int16_t*)&v);
}

inline __device__ uint32_t bswap_32(uint32_t val) {
    return __byte_perm(val, 0, 0x0123);
}

inline __device__ int32_t bswap_32(int32_t val) {
    uint32_t v = bswap_32(*((uint32_t*)&val));
    return *((int32_t*)&v);
}

inline __device__ float bswap_32(float val) {
    uint32_t v = bswap_32(*((uint32_t*)&val));
    return *((float*)&v);
}

inline __device__ uint64_t bswap_64(uint64_t val) {
    uint32_t hi = __byte_perm((uint32_t)(val & 0xFFFFFFFF), 0, 0x0123);
    uint32_t lo = __byte_perm((uint32_t)(val >> 32), 0, 0x0123);
    return ((uint64_t)hi << 32) | lo;
}

inline __device__ int64_t bswap_64(int64_t val) {
    uint64_t v = bswap_64(*((uint64_t*)&val));
    return *((int64_t*)&v);
}

inline __device__ double bswap_64(double val) {
    uint64_t v = bswap_64(*((uint64_t*)&val));
    return *((double*)&v);
}
