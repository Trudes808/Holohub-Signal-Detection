#!/usr/bin/env python3

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate realtime headroom for wideband DINO runs.")
    parser.add_argument("--sample-rate", type=float, default=500e6, help="Configured per-channel sample rate in samples/s")
    parser.add_argument("--channels", type=int, default=2, help="Number of active channels sharing the detector GPU")
    parser.add_argument("--samples-per-packet", type=int, default=1024, help="Complex samples per CHDR packet")
    parser.add_argument("--packets-per-fft", type=int, default=20, help="Packets accumulated per FFT")
    parser.add_argument("--num-ffts-per-batch", type=int, default=64, help="FFTs processed per downstream batch")
    parser.add_argument("--service-ms", type=float, required=True, help="Measured mean DINO service time per emitted frame in ms")
    parser.add_argument("--emit-stride", type=int, default=1, help="Current emit stride")
    parser.add_argument("--observed-packets", type=float, default=0.0, help="Optional packets processed per channel from a run log")
    parser.add_argument("--duration-s", type=float, default=0.0, help="Optional run duration for observed packet count")
    args = parser.parse_args()

    samples_per_fft = args.samples_per_packet * args.packets_per_fft
    samples_per_batch = samples_per_fft * args.num_ffts_per_batch
    service_s = args.service_ms / 1000.0

    configured_batch_rate = args.sample_rate / samples_per_batch
    detector_batch_rate_budget_per_channel = (1.0 / service_s) / max(args.channels, 1)
    max_sample_rate_per_channel = detector_batch_rate_budget_per_channel * samples_per_batch
    utilization_at_configured_rate = configured_batch_rate / detector_batch_rate_budget_per_channel
    required_stride = max(1.0, utilization_at_configured_rate)
    batch_fill_ms = samples_per_batch / args.sample_rate * 1000.0

    print(f"samples_per_fft: {samples_per_fft}")
    print(f"samples_per_batch: {samples_per_batch}")
    print(f"batch_fill_ms_at_configured_rate: {batch_fill_ms:.6f}")
    print(f"configured_batch_rate_per_channel_hz: {configured_batch_rate:.6f}")
    print(f"detector_batch_rate_budget_per_channel_hz: {detector_batch_rate_budget_per_channel:.6f}")
    print(f"max_sample_rate_per_channel_realtime: {max_sample_rate_per_channel:.3f}")
    print(f"utilization_at_configured_rate: {utilization_at_configured_rate:.6f}")
    print(f"minimum_emit_stride_for_configured_rate: {required_stride:.6f}")
    print(f"current_emit_stride_margin: {args.emit_stride / required_stride:.6f}")

    if args.observed_packets > 0.0 and args.duration_s > 0.0:
      observed_sample_rate = args.observed_packets * args.samples_per_packet / args.duration_s
      observed_batch_rate = observed_sample_rate / samples_per_batch
      observed_utilization = observed_batch_rate / detector_batch_rate_budget_per_channel
      print(f"observed_sample_rate_per_channel: {observed_sample_rate:.3f}")
      print(f"observed_batch_rate_per_channel_hz: {observed_batch_rate:.6f}")
      print(f"observed_utilization: {observed_utilization:.6f}")


if __name__ == "__main__":
    main()