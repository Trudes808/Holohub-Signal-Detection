// SPDX-FileCopyrightText: 2026 National Instruments Corporation
//
// SPDX-License-Identifier: Apache-2.0
#include "snippet_compression.hpp"

#include <sys/stat.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>
#include <utility>
#include <vector>

namespace holoscan::ops {

namespace {

// ------------------------------------------------------------------ kernels ----

// Grid-stride max|x| over a float array, reduced into *out_bits with atomicMax on the raw bits.
// Valid because |x| >= 0: for non-negative IEEE-754 floats, unsigned bit ordering == value ordering.
__global__ void max_abs_f32_kernel(const float* __restrict__ x, size_t n, unsigned int* out_bits) {
  __shared__ float sh[256];
  float m = 0.0f;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    m = fmaxf(m, fabsf(x[i]));
  }
  sh[threadIdx.x] = m;
  __syncthreads();
  for (int o = blockDim.x / 2; o > 0; o >>= 1) {
    if (threadIdx.x < static_cast<unsigned>(o)) sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + o]);
    __syncthreads();
  }
  if (threadIdx.x == 0) atomicMax(out_bits, __float_as_uint(sh[0]));
}

// Quantize floats to int16 with a single scale (x_hat = q * scale).
__global__ void quant_sc16_kernel(const float* __restrict__ x, size_t n, float inv_scale,
                                  int16_t* __restrict__ out) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    const float q = rintf(x[i] * inv_scale);
    out[i] = static_cast<int16_t>(fmaxf(-32767.0f, fminf(32767.0f, q)));
  }
}

// Block floating point encode: one CUDA block per BFP block of `bfp_block` complex samples
// (= 2*bfp_block scalars). Output layout per block: [int8 exponent][mantissas...], blocks
// contiguous at stride `block_bytes`. Mantissas are two's-complement MANT_BITS values; for
// MANT_BITS=12 each scalar pair packs into 3 bytes (m0[11:4], m0[3:0]|m1[11:8], m1[7:0]).
// Exponent e is the smallest power of two with max|x| / 2^e <= qmax; x_hat = m * 2^e.
template <int MANT_BITS>
__global__ void bfp_encode_kernel(const float2* __restrict__ iq, uint64_t n_complex, int bfp_block,
                                  uint8_t* __restrict__ out, uint64_t block_bytes) {
  extern __shared__ unsigned char smem[];
  float* vals = reinterpret_cast<float*>(smem);                       // 2*bfp_block floats
  int* quants = reinterpret_cast<int*>(vals + 2 * bfp_block);         // 2*bfp_block ints
  __shared__ float sh_max[256];
  __shared__ int sh_e;

  const int scalars = 2 * bfp_block;
  const uint64_t c0 = static_cast<uint64_t>(blockIdx.x) * bfp_block;
  constexpr float qmax = static_cast<float>((1 << (MANT_BITS - 1)) - 1);

  float m = 0.0f;
  for (int i = threadIdx.x; i < scalars; i += blockDim.x) {
    const uint64_t c = c0 + (i >> 1);
    const float v = (c < n_complex) ? ((i & 1) ? iq[c].y : iq[c].x) : 0.0f;
    vals[i] = v;
    m = fmaxf(m, fabsf(v));
  }
  sh_max[threadIdx.x] = m;
  __syncthreads();
  for (int o = blockDim.x / 2; o > 0; o >>= 1) {
    if (threadIdx.x < static_cast<unsigned>(o)) sh_max[threadIdx.x] = fmaxf(sh_max[threadIdx.x], sh_max[threadIdx.x + o]);
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    const float maxabs = sh_max[0];
    int e = -127;
    if (maxabs > 0.0f) {
      e = static_cast<int>(ceilf(log2f(maxabs / qmax)));
      // float-edge guard: ensure the quantized max actually fits
      while (maxabs * exp2f(static_cast<float>(-e)) > qmax) ++e;
      e = max(-127, min(127, e));
    }
    sh_e = e;
  }
  __syncthreads();

  const int e = sh_e;
  const float dq = exp2f(static_cast<float>(-e));
  for (int i = threadIdx.x; i < scalars; i += blockDim.x) {
    const float q = rintf(vals[i] * dq);
    quants[i] = static_cast<int>(fmaxf(-qmax, fminf(qmax, q)));
  }
  __syncthreads();

  uint8_t* blk = out + static_cast<uint64_t>(blockIdx.x) * block_bytes;
  if (threadIdx.x == 0) blk[0] = static_cast<uint8_t>(static_cast<int8_t>(e));
  uint8_t* payload = blk + 1;
  if (MANT_BITS == 8) {
    for (int i = threadIdx.x; i < scalars; i += blockDim.x) {
      payload[i] = static_cast<uint8_t>(static_cast<int8_t>(quants[i]));
    }
  } else {  // 12-bit: pack scalar pairs into 3 bytes
    for (int j = threadIdx.x; j < scalars / 2; j += blockDim.x) {
      const unsigned m0 = static_cast<unsigned>(quants[2 * j]) & 0xFFFu;
      const unsigned m1 = static_cast<unsigned>(quants[2 * j + 1]) & 0xFFFu;
      payload[3 * j + 0] = static_cast<uint8_t>(m0 >> 4);
      payload[3 * j + 1] = static_cast<uint8_t>(((m0 & 0xFu) << 4) | (m1 >> 8));
      payload[3 * j + 2] = static_cast<uint8_t>(m1 & 0xFFu);
    }
  }
}

// ------------------------------------------------------------- host helpers ----

int launch_blocks_for(size_t n, int threads) {
  const size_t b = (n + threads - 1) / threads;
  return static_cast<int>(std::min<size_t>(b, 1024));
}

bool codec_is_known(const std::string& c) {
  return c == "none" || c == "off" || c == "sc16" || c == "bfp8" || c == "bfp12";
}

double now_s() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// Minimal JSON string-value lookup ("key": "value") -- same spirit as the visualizer's control
// reader; the control file is machine-written so this stays trivially parseable.
bool json_find_string(const std::string& text, const std::string& key, std::string& out) {
  const std::string needle = "\"" + key + "\"";
  size_t p = text.find(needle);
  if (p == std::string::npos) return false;
  p = text.find(':', p + needle.size());
  if (p == std::string::npos) return false;
  p = text.find('"', p);
  if (p == std::string::npos) return false;
  const size_t q = text.find('"', p + 1);
  if (q == std::string::npos) return false;
  out = text.substr(p + 1, q - p - 1);
  return true;
}

// Alias a pooled complex slab as a byte pointer, keeping the pool's control block alive.
std::shared_ptr<uint8_t> alias_bytes(std::shared_ptr<SnipComplex> slab) {
  auto* raw = reinterpret_cast<uint8_t*>(slab.get());
  return std::shared_ptr<uint8_t>(std::move(slab), raw);
}

}  // namespace

// ------------------------------------------------------------------ operator ----

void SnippetCompressionOp::setup(OperatorSpec& spec) {
  // A sized input does NOT get a default scheduling condition -- without an explicit
  // MessageAvailableCondition the operator is never woken and batches queue forever
  // (mirrors sigmf_file_sink's port setup).
  auto& input_port = spec.input<SnippetBatchMessage>("in", IOSpec::IOSize{16});
  input_port.conditions().emplace_back(
      ConditionType::kMessageAvailable,
      std::make_shared<MessageAvailableCondition>(size_t{1}));
  spec.output<SnippetBatchMessage>("out", IOSpec::IOSize{16});
  spec.param(codec_, "codec", "Codec",
             "none | sc16 | bfp12 | bfp8 (decode-grade quantization codecs; see header).",
             std::string("none"));
  spec.param(bfp_block_, "bfp_block", "BFP block",
             "Complex samples per shared-exponent block for the bfp codecs.", 64);
  spec.param(control_json_path_, "control_json_path", "Control JSON path",
             "Optional demo-control file; its \"codec\" value switches the codec live.",
             std::string(""));
}

void SnippetCompressionOp::initialize() {
  Operator::initialize();
  cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
  pool_ = std::make_shared<DeviceBufferPool>();
  active_codec_ = codec_.get();
  if (!codec_is_known(active_codec_)) {
    HOLOSCAN_LOG_WARN("snippet_compression: unknown codec '{}' -> using 'none'", active_codec_);
    active_codec_ = "none";
  }
  HOLOSCAN_LOG_INFO("snippet_compression: codec={} bfp_block={} control_json={}",
                    active_codec_, bfp_block_.get(),
                    control_json_path_.get().empty() ? "(static)" : control_json_path_.get());
}

void SnippetCompressionOp::stop() {
  if (d_max_bits_ != nullptr) { cudaFree(d_max_bits_); d_max_bits_ = nullptr; }
  if (h_max_bits_ != nullptr) { cudaFreeHost(h_max_bits_); h_max_bits_ = nullptr; }
  if (stream_ != nullptr) {
    cudaStreamSynchronize(stream_);
    cudaStreamDestroy(stream_);
    stream_ = nullptr;
  }
  Operator::stop();
}

void SnippetCompressionOp::ensure_max_scratch(size_t n) {
  if (n <= max_scratch_capacity_) return;
  const size_t cap = std::max<size_t>(n, std::max<size_t>(64, max_scratch_capacity_ * 2));
  if (d_max_bits_ != nullptr) cudaFree(d_max_bits_);
  if (h_max_bits_ != nullptr) cudaFreeHost(h_max_bits_);
  cudaMalloc(&d_max_bits_, cap * sizeof(unsigned int));
  cudaMallocHost(&h_max_bits_, cap * sizeof(unsigned int));
  max_scratch_capacity_ = cap;
}

void SnippetCompressionOp::maybe_refresh_codec() {
  const std::string path = control_json_path_.get();
  if (path.empty()) return;
  const double now = now_s();
  if (now - last_control_check_s_ < 0.5) return;
  last_control_check_s_ = now;
  struct stat st;
  if (stat(path.c_str(), &st) != 0) return;
  const int64_t mtime_ns = static_cast<int64_t>(st.st_mtim.tv_sec) * 1000000000 + st.st_mtim.tv_nsec;
  if (mtime_ns == control_mtime_ns_) return;
  control_mtime_ns_ = mtime_ns;
  std::ifstream in(path);
  if (!in.is_open()) return;
  std::stringstream ss;
  ss << in.rdbuf();
  std::string requested;
  if (!json_find_string(ss.str(), "codec", requested)) return;  // key absent: keep current codec
  if (requested == "off") requested = "none";
  if (!codec_is_known(requested) || requested == active_codec_) return;
  HOLOSCAN_LOG_INFO("snippet_compression: DEMO CONTROL codec {} -> {}", active_codec_, requested);
  active_codec_ = requested;
}

void SnippetCompressionOp::compute(InputContext& op_input, OutputContext& op_output,
                                   ExecutionContext&) {
  auto input = op_input.receive<SnippetBatchMessage>("in");
  if (!input) return;
  auto batch = std::move(input.value());

  maybe_refresh_codec();
  if (active_codec_ == "none" || batch.snippets.empty()) {
    op_output.emit(std::move(batch), "out");
    return;
  }

  const bool sc16 = active_codec_ == "sc16";
  const int mant_bits = active_codec_ == "bfp12" ? 12 : 8;
  const int B = std::max(8, bfp_block_.get());

  if (sc16) {
    // Pass 1: batch-wide per-snippet max|x| (one scratch slot each), single sync.
    ensure_max_scratch(batch.snippets.size());
    cudaMemsetAsync(d_max_bits_, 0, batch.snippets.size() * sizeof(unsigned int), stream_);
    for (size_t s = 0; s < batch.snippets.size(); ++s) {
      const auto& snip = batch.snippets[s];
      if (snip.n_iq == 0 || !snip.device_iq) continue;
      const size_t n = snip.n_iq * 2;
      max_abs_f32_kernel<<<launch_blocks_for(n, 256), 256, 0, stream_>>>(
          reinterpret_cast<const float*>(snip.device_iq.get()), n, d_max_bits_ + s);
    }
    cudaMemcpyAsync(h_max_bits_, d_max_bits_, batch.snippets.size() * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);
  }

  for (size_t s = 0; s < batch.snippets.size(); ++s) {
    auto& snip = batch.snippets[s];
    if (snip.n_iq == 0 || !snip.device_iq || !snip.codec.empty()) continue;
    const uint64_t logical = snip.n_iq * sizeof(SnipComplex);

    if (sc16) {
      float maxabs = 0.0f;
      const unsigned int bits = h_max_bits_[s];
      std::memcpy(&maxabs, &bits, sizeof(float));
      const float scale = maxabs > 0.0f ? maxabs / 32767.0f : 1.0f;
      const uint64_t nbytes = snip.n_iq * 2 * sizeof(int16_t);
      auto slab = pool_->acquire((nbytes + sizeof(SnipComplex) - 1) / sizeof(SnipComplex));
      const size_t n = snip.n_iq * 2;
      quant_sc16_kernel<<<launch_blocks_for(n, 256), 256, 0, stream_>>>(
          reinterpret_cast<const float*>(snip.device_iq.get()), n, 1.0f / scale,
          reinterpret_cast<int16_t*>(slab.get()));
      snip.codec = "sc16";
      snip.comp_scale = scale;
      snip.comp_bytes = nbytes;
      snip.device_comp = alias_bytes(std::move(slab));
    } else {
      const uint64_t nblocks = (snip.n_iq + B - 1) / B;
      const uint64_t block_bytes = 1 + (static_cast<uint64_t>(2 * B) * mant_bits) / 8;
      const uint64_t nbytes = nblocks * block_bytes;
      auto slab = pool_->acquire((nbytes + sizeof(SnipComplex) - 1) / sizeof(SnipComplex));
      const int threads = std::min(256, 2 * B);
      const size_t shmem = static_cast<size_t>(2 * B) * (sizeof(float) + sizeof(int));
      if (mant_bits == 12) {
        bfp_encode_kernel<12><<<static_cast<unsigned>(nblocks), threads, shmem, stream_>>>(
            reinterpret_cast<const float2*>(snip.device_iq.get()), snip.n_iq, B,
            reinterpret_cast<uint8_t*>(slab.get()), block_bytes);
      } else {
        bfp_encode_kernel<8><<<static_cast<unsigned>(nblocks), threads, shmem, stream_>>>(
            reinterpret_cast<const float2*>(snip.device_iq.get()), snip.n_iq, B,
            reinterpret_cast<uint8_t*>(slab.get()), block_bytes);
      }
      snip.codec = active_codec_;
      snip.comp_block = B;
      snip.comp_bytes = nbytes;
      snip.device_comp = alias_bytes(std::move(slab));
    }

    logical_bytes_ += logical;
    stored_bytes_ += snip.comp_bytes;
    ++snippets_compressed_;
  }

  // The sink stages with plain cudaMemcpy on the default stream; make the compressed payloads
  // visible before the message leaves, then release the fat cf32 buffers back to the snipper pool.
  cudaStreamSynchronize(stream_);
  for (auto& snip : batch.snippets) {
    if (!snip.codec.empty()) snip.device_iq.reset();
  }

  const double now = now_s();
  if (now - last_log_s_ > 5.0 && stored_bytes_ > 0) {
    last_log_s_ = now;
    HOLOSCAN_LOG_INFO("snippet_compression: codec={} snippets={} ratio={:.2f}x ({} -> {} MB)",
                      active_codec_, snippets_compressed_,
                      static_cast<double>(logical_bytes_) / static_cast<double>(stored_bytes_),
                      logical_bytes_ / 1000000, stored_bytes_ / 1000000);
  }

  op_output.emit(std::move(batch), "out");
}

}  // namespace holoscan::ops
