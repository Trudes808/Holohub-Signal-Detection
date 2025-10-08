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

#ifndef CHDR_CONVERTER_SWAP_H
#define CHDR_CONVERTER_SWAP_H

// Swapping logic borrowed from RedhawkSDR:
// https://github.com/RedhawkSDR/VITA49/blob/master/cpp/include/VRTMath.h#L77
inline uint16_t bswap_16_h(const uint16_t val) {
    return ((val & 0xff00) >> 8)
         | ((val & 0x00ff) << 8);
}

inline int16_t bswap_16_h(const int16_t val) {
    const uint16_t v = bswap_16_h(*((uint16_t*)&val));
    return *((int16_t*)&v);
}

inline uint32_t bswap_32_h(uint32_t val) {
    return ((val & 0xff000000) >> 24)
         | ((val & 0x00ff0000) >>  8)
         | ((val & 0x0000ff00) <<  8)
         | ((val & 0x000000ff) << 24);
}

inline int32_t bswap_32_h(int32_t val) {
    uint32_t v = bswap_32_h(*((uint32_t*)&val));
    return *((int32_t*)&v);
}

inline float bswap_32_h(float val) {
    uint32_t v = bswap_32_h(*((uint32_t*)&val));
    return *((float*)&v);
}

inline uint64_t bswap_64_h(uint64_t val) {
    return ((val & __UINT64_C(0xff00000000000000)) >> 56)
         | ((val & __UINT64_C(0x00ff000000000000)) >> 40)
         | ((val & __UINT64_C(0x0000ff0000000000)) >> 24)
         | ((val & __UINT64_C(0x000000ff00000000)) >>  8)
         | ((val & __UINT64_C(0x00000000ff000000)) <<  8)
         | ((val & __UINT64_C(0x0000000000ff0000)) << 24)
         | ((val & __UINT64_C(0x000000000000ff00)) << 40)
         | ((val & __UINT64_C(0x00000000000000ff)) << 56);
}

inline int64_t bswap_64_h(int64_t val) {
    uint64_t v = bswap_64_h(*((uint64_t*)&val));
    return *((int64_t*)&v);
}

inline double bswap_64_h(double val) {
    uint64_t v = bswap_64_h(*((uint64_t*)&val));
    return *((double*)&v);
}

#endif /* CHDR_CONVERTER_SWAP_H */
