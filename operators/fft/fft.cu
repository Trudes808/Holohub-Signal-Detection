// SPDX-FileCopyrightText: 2024 Valley Tech Systems, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include "fft.hpp"

#include <chrono>
#include <cmath>

using in_t = std::tuple<tensor_t<complex, 2>, cudaStream_t>;
using out_t = std::tuple<tensor_t<complex, 2>, cudaStream_t>;

namespace holoscan::ops {

namespace {

uint64_t steady_time_ns() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                     std::chrono::steady_clock::now().time_since_epoch())
                                     .count());
}

double elapsed_ms(uint64_t start_ns, uint64_t end_ns) {
    if (start_ns == 0 || end_ns <= start_ns) {
        return 0.0;
    }
    return static_cast<double>(end_ns - start_ns) / 1.0e6;
}

}  // namespace

void FFT::setup(OperatorSpec& spec) {
    spec.input<in_t>("in", holoscan::IOSpec::IOSize{16});
    spec.output<out_t>("out", holoscan::IOSpec::IOSize{16});
    spec.param(burst_size,
        "burst_size",
        "Burst size"
        "Number of samples to process in each burst");
    spec.param(emit_stride,
        "emit_stride",
        "Emit stride",
        "Emit one FFT output every N input batches.",
        1);
    spec.param(num_bursts,
        "num_bursts",
        "Number of bursts"
        "Number of sample bursts to process at once");
    spec.param(num_channels,
        "num_channels",
        "Number of channels",
        "Number of channels to allocate memory for");
    spec.param(spectrum_type,
        "spectrum_type",
        "Spectrum type",
        "VITA 49.2 spectrum type to pass along in metadata");
    spec.param(spectrum_type,
        "spectrum_type",
        "Spectrum type",
        "VITA 49.2 spectrum type to pass along in metadata");
    spec.param(averaging_type,
        "averaging_type",
        "Averaging type",
        "VITA 49.2 averaging type to pass along in metadata");
    spec.param(window_time,
        "window_time",
        "Window time",
        "VITA 49.2 window time to pass along in metadata");
    spec.param(window_type,
        "window_type",
        "Window type",
        "VITA 49.2 window type to pass along in metadata");
    spec.param(transform_points,
        "transform_points",
        "Transform points",
        "Number of FFT points to take and VITA 49.2 transform points to pass along in metadata");
    spec.param(window_points,
        "window_points",
        "Window points",
        "VITA 49.2 window points to pass along in metadata");
    spec.param(resolution,
        "resolution",
        "Resolution",
        "VITA 49.2 resolution to pass along in metadata");
    spec.param(span,
        "span",
        "Span",
        "VITA 49.2 span to pass along in metadata");
    spec.param(weighting_factor,
        "weighting_factor",
        "Weighting factory",
        "VITA 49.2 weighting factor to pass along in metadata");
    spec.param(f1_index,
        "f1_index",
        "F1 index",
        "VITA 49.2 F1 index to pass along in metadata");
    spec.param(f2_index,
        "f2_index",
        "F2 index",
        "VITA 49.2 F2 index to pass along in metadata");
    spec.param(window_time_delta,
        "window_time_delta",
        "Window time delta",
        "VITA 49.2 window time delta to pass along in metadata");
    spec.param(timing_summary_every_n,
        "timing_summary_every_n",
        "Timing Summary Every N",
        "Emit live FFT ingress timing summaries every N emitted frames per channel.",
        128);
}

void FFT::initialize() {
    holoscan::Operator::initialize();
    make_tensor(outputs,
                {num_channels.get(), num_bursts.get(), burst_size.get()},
                MATX_DEVICE_MEMORY);
    ingress_stats.assign(static_cast<size_t>(std::max<uint16_t>(1, num_channels.get())), ChannelIngressStats {});
    output_frame_count.assign(static_cast<size_t>(std::max<uint16_t>(1, num_channels.get())), 0);
}

void FFT::compute(InputContext& op_input, OutputContext& op_output, ExecutionContext& context) {
    auto input = op_input.receive<in_t>("in").value();
    auto meta = metadata();
    auto channel_num = meta->get<uint16_t>("channel_number", 0);
    const uint64_t fft_enter_ns = steady_time_ns();
    if (channel_num < ingress_stats.size()) {
        const uint64_t chdr_emit_ns = meta->get<uint64_t>("chdr_emit_ts_ns", 0);
        auto& stats = ingress_stats[channel_num];
        const double chdr_to_fft_ms = elapsed_ms(chdr_emit_ns, fft_enter_ns);
        stats.samples++;
        stats.window_samples++;
        stats.total_chdr_to_fft_ms += chdr_to_fft_ms;
        stats.window_total_chdr_to_fft_ms += chdr_to_fft_ms;
        stats.max_chdr_to_fft_ms = std::max(stats.max_chdr_to_fft_ms, chdr_to_fft_ms);
        stats.window_max_chdr_to_fft_ms = std::max(stats.window_max_chdr_to_fft_ms, chdr_to_fft_ms);
    }
    meta->set("fft_enter_ts_ns", fft_enter_ns);
    auto out = slice<2>(outputs, {static_cast<index_t>(channel_num), 0, 0},
            {matxDropDim, matxEnd, matxEnd});

    (out = fftshift1D(fft(std::get<0>(input)))).run(std::get<1>(input));

    const int configured_emit_stride = std::max(1, emit_stride.get());
    uint64_t frame_number = 0;
    if (channel_num < output_frame_count.size()) {
        frame_number = ++output_frame_count[channel_num];
    }
    if (configured_emit_stride > 1 &&
        (frame_number % static_cast<uint64_t>(configured_emit_stride)) != 0) {
        return;
    }

    if (channel_num < ingress_stats.size()) {
        auto& stats = ingress_stats[channel_num];
        stats.emitted_frames++;
        stats.window_emitted_frames++;
        const uint64_t summary_every = static_cast<uint64_t>(std::max(1, timing_summary_every_n.get()));
        if (stats.window_emitted_frames >= summary_every) {
            const double window_samples = static_cast<double>(std::max<uint64_t>(1, stats.window_samples));
            HOLOSCAN_LOG_INFO(
                "FFT ingress live ch={} window_samples={} window_emitted={} mean_chdr_to_fft_ms={:.3f} max_chdr_to_fft_ms={:.3f}",
                channel_num,
                stats.window_samples,
                stats.window_emitted_frames,
                stats.window_total_chdr_to_fft_ms / window_samples,
                stats.window_max_chdr_to_fft_ms);
            stats.window_samples = 0;
            stats.window_emitted_frames = 0;
            stats.window_total_chdr_to_fft_ms = 0.0;
            stats.window_max_chdr_to_fft_ms = 0.0;
        }
    }

    if (spectrum_type.has_value())
        meta->set("spectrum_type", spectrum_type.get());
    if (averaging_type.has_value())
        meta->set("averaging_type", averaging_type.get());
    if (window_time.has_value())
        meta->set("window_time_delta_interpretation", window_time.get());
    if (window_type.has_value())
        meta->set("window_type", window_type.get());
    const uint32_t transform_count = transform_points.has_value() ? transform_points.get()
                                                                  : static_cast<uint32_t>(out.Size(1));
    if (transform_points.has_value())
        meta->set("num_transform_points", transform_count);
    if (window_points.has_value())
        meta->set("num_window_points", window_points.get());
    double derived_span_hz = 0.0;
    if (meta->has_key("sample_rate_hz")) {
        derived_span_hz = meta->get<double>("sample_rate_hz");
    } else if (meta->has_key("bandwidth_hz")) {
        derived_span_hz = meta->get<double>("bandwidth_hz");
    } else if (span.has_value()) {
        derived_span_hz = static_cast<double>(span.get());
    }
    if (!std::isfinite(derived_span_hz) || derived_span_hz <= 0.0) {
        derived_span_hz = 0.0;
    }

    uint64_t metadata_span_hz = span.has_value() ? span.get() : 0;
    if (derived_span_hz > 0.0) {
        metadata_span_hz = static_cast<uint64_t>(std::llround(derived_span_hz));
    }
    if (metadata_span_hz > 0) {
        meta->set("span", metadata_span_hz);
    }

    uint64_t metadata_resolution_hz = resolution.has_value() ? resolution.get() : 0;
    if (derived_span_hz > 0.0 && transform_count > 0) {
        metadata_resolution_hz = static_cast<uint64_t>(std::llround(derived_span_hz / static_cast<double>(transform_count)));
    }
    if (metadata_resolution_hz > 0) {
        meta->set("resolution", metadata_resolution_hz);
    }
    if (weighting_factor.has_value())
        meta->set("weighting_factor", weighting_factor.get());
    if (f1_index.has_value())
        meta->set("f1_index", f1_index.get());
    if (f2_index.has_value())
        meta->set("f2_index", f2_index.get());
    if (window_time_delta.has_value())
        meta->set("window_time_delta", window_time_delta.get());
    meta->set("fft_emit_stride", configured_emit_stride);
    meta->set("fft_emitted_frame_number", frame_number);
    meta->set("fft_emit_ts_ns", steady_time_ns());

    op_output.emit(
        out_t {
            out,
            std::get<1>(input)
        },
        "out");
}

void FFT::stop() {
    for (size_t channel_index = 0; channel_index < ingress_stats.size(); ++channel_index) {
        const auto& stats = ingress_stats[channel_index];
        HOLOSCAN_LOG_INFO(
            "FFT ingress latency ch={} samples={} emitted_frames={} mean_chdr_to_fft_ms={:.3f} max_chdr_to_fft_ms={:.3f}",
            channel_index,
            stats.samples,
            stats.emitted_frames,
            stats.samples == 0 ? 0.0 : stats.total_chdr_to_fft_ms / static_cast<double>(stats.samples),
            stats.max_chdr_to_fft_ms);
    }
    holoscan::Operator::stop();
}
}  // namespace holoscan::ops
