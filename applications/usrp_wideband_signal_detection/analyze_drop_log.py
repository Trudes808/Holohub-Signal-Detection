#!/usr/bin/env python3

import argparse
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path


THROUGHPUT_RE = re.compile(
    r"Processed \d+ samples from channel (?P<channel>\d+) at (?P<msps>[0-9.]+) MSps"
)
CHDR_SUMMARY_RE = re.compile(
    r"CHDR summary ch=(?P<channel>\d+).*?queued=(?P<queued>\d+).*?emitted=(?P<emitted>\d+)"
    r".*?backlog_events=(?P<backlog>\d+).*?out_q_depth=(?P<out_q_depth>\d+)"
    r".*?max_out_q_depth=(?P<max_out_q_depth>\d+).*?aggr_pkts_recv=(?P<aggr>\d+)"
    r".*?partial_drops=(?P<partial>\d+)"
)
CHDR_BACKLOG_RE = re.compile(r"CHDR backlog state ch=(?P<channel>\d+)")
FFT_RE = re.compile(
    r"FFT ingress (?:live|latency) op=.*? ch=(?P<channel>\d+).*?mean_chdr_to_fft_ms=(?P<to_fft>[0-9.]+)"
    r".*?mean_chdr_emit_gap_ms=(?P<emit_gap>[0-9.]+).*?mean_fft_enter_gap_ms=(?P<enter_gap>[0-9.]+)"
    r".*?mean_fft_compute_ms=(?P<compute>[0-9.]+)"
)
COHERENT_RE = re.compile(
    r"Coherent power timing summary ch=(?P<channel>\d+).*?fft_to_detector_enter_mean=(?P<enter>[0-9.]+)"
    r".*?fft_to_detector_done_mean=(?P<done>[0-9.]+)"
)
VISUALIZER_RE = re.compile(
    r"Visualizer timing: .*?frames_seen=(?P<seen>\d+).*?processed=(?P<processed>\d+)"
    r".*?rendered=(?P<rendered>\d+).*?drop_vis_busy=(?P<drop_vis>\d+)"
    r".*?drop_render_busy=(?P<drop_render>\d+).*?drop_rate=(?P<drop_rate>[0-9.]+)"
)
DPDK_MISSED_RE = re.compile(
    r"'(?P<interface>[^']+)' interface \((?P<port>\d+)\), Rx: Dropped (?P<delta>\d+) packets"
)
DPDK_QUEUE_RE = re.compile(
    r"'(?P<interface>[^']+)' interface \((?P<port>\d+)\), Rx '(?P<queue_name>[^']+)' queue \((?P<queue>\d+)\):"
)
DPDK_ALLOC_RE = re.compile(
    r"Port (?P<interface>[^:]+): Buffer allocation errors since last poll .* total (?P<total>\d+)"
)
RX_WORKER_RE = re.compile(
    r"RX (?P<worker_kind>\w+) summary port=(?P<port>\d+) queue=(?P<queue>\d+) .*?packets=(?P<packets>\d+)"
    r".*?full_flushes=(?P<full_flushes>\d+) timeout_flushes=(?P<timeout_flushes>\d+)"
    r".*?malformed=(?P<malformed>\d+) ring_full_drops=(?P<ring_full_drops>\d+)"
    r".*?idle_polls=(?P<idle_polls>\d+) max_burst=(?P<max_burst>\d+) max_ring_depth=(?P<max_ring_depth>\d+)"
)
DPDK_PORT_HEADER_RE = re.compile(r"Port (?P<port>\d+):")
DPDK_PORT_MISSED_RE = re.compile(r"- Missed packets:\s+(?P<value>\d+)")
DPDK_PORT_RX_NOMBUF_RE = re.compile(r"- RX out of buffers:\s+(?P<value>\d+)")
COHERENT_PARTIAL_SKIP_RE = re.compile(
    r"Skipping partial CHDR batch in coherent detector ch=(?P<channel>\d+)"
)
SPECTROGRAM_PARTIAL_SKIP_RE = re.compile(
    r"Skipping partial CHDR spectrogram preview for channel (?P<channel>\d+)"
)


@dataclass
class ChannelStats:
    latest_msps: float | None = None
    chdr_queued: int = 0
    chdr_emitted: int = 0
    chdr_backlog_events: int = 0
    chdr_partial_drops: int = 0
    chdr_out_q_depth: int = 0
    chdr_max_out_q_depth: int = 0
    chdr_aggr_pkts_recv: int = 0
    backlog_log_hits: int = 0
    fft_mean_chdr_to_fft_ms: float | None = None
    fft_mean_chdr_emit_gap_ms: float | None = None
    fft_mean_fft_enter_gap_ms: float | None = None
    fft_mean_compute_ms: float | None = None
    coherent_enter_mean_ms: float | None = None
    coherent_done_mean_ms: float | None = None
    coherent_partial_skips: int = 0
    spectrogram_partial_skips: int = 0


@dataclass
class RxWorkerStats:
    port: int
    queue: int
    worker_kind: str
    packets: int = 0
    full_flushes: int = 0
    timeout_flushes: int = 0
    malformed: int = 0
    ring_full_drops: int = 0
    idle_polls: int = 0
    max_burst: int = 0
    max_ring_depth: int = 0


@dataclass
class PortStats:
    port: int
    missed_packets: int | None = None
    rx_out_of_buffers: int | None = None


@dataclass
class GlobalStats:
    channels: dict[int, ChannelStats] = field(default_factory=dict)
    dpdk_missed_events: list[str] = field(default_factory=list)
    dpdk_queue_events: list[str] = field(default_factory=list)
    dpdk_alloc_events: list[str] = field(default_factory=list)
    rx_workers: dict[tuple[int, int], RxWorkerStats] = field(default_factory=dict)
    ports: dict[int, PortStats] = field(default_factory=dict)
    visualizer_seen: int | None = None
    visualizer_processed: int | None = None
    visualizer_rendered: int | None = None
    visualizer_drop_vis_busy: int = 0
    visualizer_drop_render_busy: int = 0
    visualizer_drop_rate: float | None = None

    def channel(self, channel: int) -> ChannelStats:
        if channel not in self.channels:
            self.channels[channel] = ChannelStats()
        return self.channels[channel]

    def rx_worker(self, port: int, queue: int, worker_kind: str) -> RxWorkerStats:
        key = (port, queue)
        if key not in self.rx_workers:
            self.rx_workers[key] = RxWorkerStats(port=port, queue=queue, worker_kind=worker_kind)
        return self.rx_workers[key]

    def port(self, port: int) -> PortStats:
        if port not in self.ports:
            self.ports[port] = PortStats(port=port)
        return self.ports[port]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze usrp_wideband_signal_detection logs and rank the most likely drop boundary "
            "between NIC, CHDR batching, FFT ingress, coherent detector, and visualization."
        )
    )
    parser.add_argument("logfile", nargs="?", help="Path to a saved app log. Reads stdin if omitted.")
    parser.add_argument("--target-msps", type=float, default=500.0)
    parser.add_argument("--sample-rate-hz", type=float, default=500e6)
    parser.add_argument("--fft-size", type=int, default=20480)
    parser.add_argument("--ffts-per-batch", type=int, default=512)
    return parser.parse_args()


def read_lines(logfile: str | None) -> list[str]:
    if logfile:
        return Path(logfile).read_text(encoding="utf-8", errors="replace").splitlines()
    return sys.stdin.read().splitlines()


def parse_log(lines: list[str]) -> GlobalStats:
    stats = GlobalStats()
    current_port: int | None = None
    for line in lines:
        match = THROUGHPUT_RE.search(line)
        if match:
            channel = int(match.group("channel"))
            stats.channel(channel).latest_msps = float(match.group("msps"))
            continue

        match = CHDR_SUMMARY_RE.search(line)
        if match:
            channel = int(match.group("channel"))
            channel_stats = stats.channel(channel)
            channel_stats.chdr_queued = int(match.group("queued"))
            channel_stats.chdr_emitted = int(match.group("emitted"))
            channel_stats.chdr_backlog_events = int(match.group("backlog"))
            channel_stats.chdr_partial_drops = int(match.group("partial"))
            channel_stats.chdr_out_q_depth = int(match.group("out_q_depth"))
            channel_stats.chdr_max_out_q_depth = int(match.group("max_out_q_depth"))
            channel_stats.chdr_aggr_pkts_recv = int(match.group("aggr"))
            continue

        match = CHDR_BACKLOG_RE.search(line)
        if match:
            stats.channel(int(match.group("channel"))).backlog_log_hits += 1
            continue

        match = FFT_RE.search(line)
        if match:
            channel = int(match.group("channel"))
            channel_stats = stats.channel(channel)
            channel_stats.fft_mean_chdr_to_fft_ms = float(match.group("to_fft"))
            channel_stats.fft_mean_chdr_emit_gap_ms = float(match.group("emit_gap"))
            channel_stats.fft_mean_fft_enter_gap_ms = float(match.group("enter_gap"))
            channel_stats.fft_mean_compute_ms = float(match.group("compute"))
            continue

        match = COHERENT_RE.search(line)
        if match:
            channel = int(match.group("channel"))
            channel_stats = stats.channel(channel)
            channel_stats.coherent_enter_mean_ms = float(match.group("enter"))
            channel_stats.coherent_done_mean_ms = float(match.group("done"))
            continue

        match = VISUALIZER_RE.search(line)
        if match:
            stats.visualizer_seen = int(match.group("seen"))
            stats.visualizer_processed = int(match.group("processed"))
            stats.visualizer_rendered = int(match.group("rendered"))
            stats.visualizer_drop_vis_busy = int(match.group("drop_vis"))
            stats.visualizer_drop_render_busy = int(match.group("drop_render"))
            stats.visualizer_drop_rate = float(match.group("drop_rate"))
            continue

        match = DPDK_MISSED_RE.search(line)
        if match:
            stats.dpdk_missed_events.append(line.strip())
            continue

        match = DPDK_QUEUE_RE.search(line)
        if match:
            stats.dpdk_queue_events.append(line.strip())
            continue

        match = DPDK_ALLOC_RE.search(line)
        if match:
            stats.dpdk_alloc_events.append(line.strip())
            continue

        match = RX_WORKER_RE.search(line)
        if match:
            port = int(match.group("port"))
            queue = int(match.group("queue"))
            worker_stats = stats.rx_worker(port, queue, match.group("worker_kind"))
            worker_stats.packets = int(match.group("packets"))
            worker_stats.full_flushes = int(match.group("full_flushes"))
            worker_stats.timeout_flushes = int(match.group("timeout_flushes"))
            worker_stats.malformed = int(match.group("malformed"))
            worker_stats.ring_full_drops = int(match.group("ring_full_drops"))
            worker_stats.idle_polls = int(match.group("idle_polls"))
            worker_stats.max_burst = int(match.group("max_burst"))
            worker_stats.max_ring_depth = int(match.group("max_ring_depth"))
            continue

        match = DPDK_PORT_HEADER_RE.search(line)
        if match:
            current_port = int(match.group("port"))
            stats.port(current_port)
            continue

        match = DPDK_PORT_MISSED_RE.search(line)
        if match and current_port is not None:
            stats.port(current_port).missed_packets = int(match.group("value"))
            continue

        match = DPDK_PORT_RX_NOMBUF_RE.search(line)
        if match and current_port is not None:
            stats.port(current_port).rx_out_of_buffers = int(match.group("value"))
            continue

        match = COHERENT_PARTIAL_SKIP_RE.search(line)
        if match:
            stats.channel(int(match.group("channel"))).coherent_partial_skips += 1
            continue

        match = SPECTROGRAM_PARTIAL_SKIP_RE.search(line)
        if match:
            stats.channel(int(match.group("channel"))).spectrogram_partial_skips += 1
            continue

    return stats


def ratio(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    denom = max(abs(a), abs(b), 1e-6)
    return abs(a - b) / denom


def channel_diagnosis(channel: int,
                      stats: ChannelStats,
                      global_stats: GlobalStats,
                      target_msps: float,
                      frame_budget_ms: float) -> list[tuple[str, int, str]]:
    findings: list[tuple[str, int, str]] = []

    latest_port_stats = global_stats.ports.get(0)
    total_ring_full_drops = sum(worker.ring_full_drops for worker in global_stats.rx_workers.values())
    total_timeout_flushes = sum(worker.timeout_flushes for worker in global_stats.rx_workers.values())

    if global_stats.dpdk_missed_events or global_stats.dpdk_queue_events or global_stats.dpdk_alloc_events:
        detail = "DPDK/NIC counters increased"
        if global_stats.dpdk_missed_events:
            detail = "DPDK rx_missed increased"
        elif global_stats.dpdk_queue_events:
            detail = "queue-specific DPDK errors increased"
        elif global_stats.dpdk_alloc_events:
            detail = "DPDK mbuf allocation errors increased"
        findings.append(("NIC or DPDK ingress loss", 10, detail))

    if latest_port_stats and latest_port_stats.rx_out_of_buffers and latest_port_stats.rx_out_of_buffers > 0:
        findings.append((
            "NIC buffer exhaustion",
            9,
            f"port {latest_port_stats.port} rx_out_of_buffers={latest_port_stats.rx_out_of_buffers}",
        ))

    if total_ring_full_drops > 0:
        findings.append((
            "DPDK to application ring backpressure",
            9,
            f"ring_full_drops={total_ring_full_drops}",
        ))

    if stats.chdr_backlog_events > 0 or stats.backlog_log_hits > 0:
        findings.append((
            "CHDR converter GPU backlog",
            9,
            f"backlog_events={stats.chdr_backlog_events}, backlog_logs={stats.backlog_log_hits}, max_out_q_depth={stats.chdr_max_out_q_depth}",
        ))

    gap_delta = ratio(stats.fft_mean_chdr_emit_gap_ms, stats.fft_mean_fft_enter_gap_ms)
    if stats.chdr_partial_drops > 0 or stats.coherent_partial_skips > 0 or stats.spectrogram_partial_skips > 0:
        findings.append((
            "CHDR batching burstiness",
            8,
            f"partial_drops={stats.chdr_partial_drops}, coherent_partial_skips={stats.coherent_partial_skips}, spectrogram_partial_skips={stats.spectrogram_partial_skips}",
        ))
    elif total_timeout_flushes > 0:
        findings.append((
            "DPDK timeout-driven batching",
            7,
            f"rx_worker_timeout_flushes={total_timeout_flushes}",
        ))
    elif gap_delta is not None and gap_delta < 0.2 and (stats.fft_mean_chdr_emit_gap_ms or 0.0) > 0.0:
        findings.append((
            "CHDR ingress cadence already bursty before FFT",
            6,
            f"mean_chdr_emit_gap_ms={stats.fft_mean_chdr_emit_gap_ms:.3f} ~= mean_fft_enter_gap_ms={stats.fft_mean_fft_enter_gap_ms:.3f}",
        ))

    if (
        stats.coherent_done_mean_ms is not None
        and stats.latest_msps is not None
        and stats.latest_msps < target_msps * 0.95
        and stats.coherent_done_mean_ms > frame_budget_ms * 1.05
    ):
        findings.append((
            "Coherent detector compute bound",
            9,
            f"coherent_done_mean_ms={stats.coherent_done_mean_ms:.3f} exceeds frame_budget_ms={frame_budget_ms:.3f}",
        ))
    elif (
        stats.coherent_enter_mean_ms is not None
        and stats.coherent_done_mean_ms is not None
        and stats.fft_mean_compute_ms is not None
        and stats.latest_msps is not None
        and stats.latest_msps < target_msps * 0.95
        and stats.coherent_enter_mean_ms < 2.0
        and stats.fft_mean_compute_ms < frame_budget_ms * 0.25
        and not global_stats.dpdk_missed_events
        and not global_stats.dpdk_queue_events
        and stats.chdr_backlog_events == 0
    ):
        findings.append((
            "Downstream of FFT, likely detector or detector-adjacent scheduling",
            6,
            f"fft_compute_ms={stats.fft_mean_compute_ms:.3f}, coherent_done_mean_ms={stats.coherent_done_mean_ms:.3f}, throughput={stats.latest_msps:.2f} MSps",
        ))

    if global_stats.visualizer_drop_rate is not None and global_stats.visualizer_drop_rate > 0.05:
        findings.append((
            "Visualization is dropping preview frames",
            4,
            f"drop_rate={global_stats.visualizer_drop_rate:.3f}, drop_vis_busy={global_stats.visualizer_drop_vis_busy}, drop_render_busy={global_stats.visualizer_drop_render_busy}",
        ))

    if not findings:
        findings.append((
            "No strong drop boundary detected from current log families",
            1,
            "Collect a run with FFT ingress, coherent timing, and DPDK warnings enabled in the same log.",
        ))

    findings.sort(key=lambda item: item[1], reverse=True)
    return findings


def render_report(stats: GlobalStats,
                  target_msps: float,
                  frame_budget_ms: float,
                  sample_rate_hz: float,
                  fft_size: int,
                  ffts_per_batch: int) -> str:
    lines: list[str] = []
    lines.append("USRP Wideband Drop Analysis")
    lines.append(f"Target throughput: {target_msps:.2f} MSps")
    lines.append(
        f"Assumed frame budget: {frame_budget_ms:.3f} ms from sample_rate={sample_rate_hz:.0f} Hz, fft_size={fft_size}, ffts_per_batch={ffts_per_batch}"
    )
    lines.append("")

    if stats.dpdk_missed_events or stats.dpdk_queue_events or stats.dpdk_alloc_events:
      lines.append("DPDK ingress warnings:")
      for line in stats.dpdk_missed_events[-3:]:
          lines.append(f"  - {line}")
      for line in stats.dpdk_queue_events[-3:]:
          lines.append(f"  - {line}")
      for line in stats.dpdk_alloc_events[-3:]:
          lines.append(f"  - {line}")
      lines.append("")

    if stats.ports:
        lines.append("DPDK port stats:")
        for port in sorted(stats.ports):
            port_stats = stats.ports[port]
            lines.append(
                f"  Port {port}: missed_packets={port_stats.missed_packets} rx_out_of_buffers={port_stats.rx_out_of_buffers}"
            )
        lines.append("")

    if stats.rx_workers:
        lines.append("RX worker summaries:")
        for key in sorted(stats.rx_workers):
            worker = stats.rx_workers[key]
            lines.append(
                f"  Port {worker.port} queue {worker.queue}: packets={worker.packets} full_flushes={worker.full_flushes} "
                f"timeout_flushes={worker.timeout_flushes} ring_full_drops={worker.ring_full_drops} "
                f"idle_polls={worker.idle_polls} max_burst={worker.max_burst} max_ring_depth={worker.max_ring_depth}"
            )
        lines.append("")

    if stats.visualizer_drop_rate is not None:
        lines.append(
            "Visualizer summary: "
            f"frames_seen={stats.visualizer_seen} processed={stats.visualizer_processed} rendered={stats.visualizer_rendered} "
            f"drop_vis_busy={stats.visualizer_drop_vis_busy} drop_render_busy={stats.visualizer_drop_render_busy} "
            f"drop_rate={stats.visualizer_drop_rate:.3f}"
        )
        lines.append("")

    for channel in sorted(stats.channels):
        channel_stats = stats.channels[channel]
        lines.append(f"Channel {channel}")
        lines.append(
            f"  Throughput: {channel_stats.latest_msps:.2f} MSps" if channel_stats.latest_msps is not None else "  Throughput: unavailable"
        )
        lines.append(
            "  CHDR: "
            f"queued={channel_stats.chdr_queued} emitted={channel_stats.chdr_emitted} "
            f"backlog_events={channel_stats.chdr_backlog_events} partial_drops={channel_stats.chdr_partial_drops} "
            f"out_q_depth={channel_stats.chdr_out_q_depth} max_out_q_depth={channel_stats.chdr_max_out_q_depth} "
            f"aggr_pkts_recv={channel_stats.chdr_aggr_pkts_recv}"
        )
        if channel_stats.fft_mean_chdr_to_fft_ms is not None:
            lines.append(
                "  FFT ingress: "
                f"chdr_to_fft={channel_stats.fft_mean_chdr_to_fft_ms:.3f} ms "
                f"chdr_emit_gap={channel_stats.fft_mean_chdr_emit_gap_ms:.3f} ms "
                f"fft_enter_gap={channel_stats.fft_mean_fft_enter_gap_ms:.3f} ms "
                f"fft_compute={channel_stats.fft_mean_compute_ms:.3f} ms"
            )
        if channel_stats.coherent_enter_mean_ms is not None:
            lines.append(
                "  Coherent timing: "
                f"enter={channel_stats.coherent_enter_mean_ms:.3f} ms "
                f"done={channel_stats.coherent_done_mean_ms:.3f} ms"
            )
        findings = channel_diagnosis(channel,
                                     channel_stats,
                                     stats,
                                     target_msps,
                                     frame_budget_ms)
        lines.append("  Likely boundaries:")
        for title, score, detail in findings[:3]:
            lines.append(f"    [{score}] {title}: {detail}")
        lines.append("")

    if not stats.channels and not stats.rx_workers and not stats.ports and not (
        stats.dpdk_missed_events or stats.dpdk_queue_events or stats.dpdk_alloc_events
    ):
        lines.append("No recognized metrics were found in the provided log.")

    lines.append("Interpretation notes:")
    lines.append("  - Visualizer drops are preview/render drops only. They do not prove RF packet loss by themselves.")
    lines.append("  - NIC or DPDK warnings outrank later stages because those packets never reach FFT intact.")
    lines.append("  - If mean_chdr_emit_gap_ms and mean_fft_enter_gap_ms track closely, burstiness is already present before FFT scheduling.")
    lines.append("  - If coherent_done_mean exceeds the batch time budget with clean ingress, the detector path is not realtime for that config.")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    lines = read_lines(args.logfile)
    stats = parse_log(lines)
    frame_budget_ms = (args.fft_size * args.ffts_per_batch) / args.sample_rate_hz * 1000.0
    report = render_report(stats,
                           args.target_msps,
                           frame_budget_ms,
                           args.sample_rate_hz,
                           args.fft_size,
                           args.ffts_per_batch)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())