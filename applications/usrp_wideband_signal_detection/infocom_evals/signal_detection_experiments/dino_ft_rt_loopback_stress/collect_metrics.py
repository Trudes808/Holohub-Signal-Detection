#!/usr/bin/env python3
"""Turn one loopback leg's results dir (run_leg.sh) into metrics.json + a one-line summary.

Sources: app.log (DPDK final stats, CHDR exit report + per-second summaries, 'Fell behind',
FFT ingress latency, DINO-FT inference log, coherent timing summaries, visualizer timing,
snipper/sink/mask-dump totals), masks/mask_dump_manifest.csv (per-frame detector_ms, host
cadence, CHDR per-batch fill, occupancy, missing batch indices), tcpreplay.log (sent packets,
achieved pps), gpu.csv (nvidia-smi samples during the replay window), leg_meta.json.

  python3 collect_metrics.py /tmp/usrp_spectrograms/rt_stress/<run> [--json-only]
"""
from __future__ import annotations

import csv
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml


def auto_fft_size(rate_hz: float, ref_span=500e6, ref_fft=20480, packet=1024) -> int:
    ratio = rate_hz / ref_span
    snapped = 2.0 ** round(math.log2(ratio)) if ratio > 0 else 1.0
    packets = max(1, int(round(ref_fft * snapped / packet)))
    return max(packet, packets * packet)


def _last_int(pat, text):
    m = re.findall(pat, text)
    return int(m[-1]) if m else None


def _last_float(pat, text):
    m = re.findall(pat, text)
    return float(m[-1]) if m else None


def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else None


def stats(a):
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        return None
    return {"n": int(a.size), "mean": float(a.mean()), "p50": pct(a, 50), "p95": pct(a, 95),
            "p99": pct(a, 99), "max": float(a.max()), "min": float(a.min())}


def parse_app_log(text: str) -> dict:
    m: dict = {}
    m["dpdk"] = {
        "received_packets": _last_int(r"- Received packets:\s+(\d+)", text),
        "missed_packets": _last_int(r"- Missed packets:\s+(\d+)", text),
        "rx_out_of_buffers": _last_int(r"- RX out of buffers:\s+(\d+)", text),
        "rx_phy_packets": _last_int(r"rx_phy_packets:\s+(\d+)", text),
        "rx_q0_errors": _last_int(r"rx_q0_errors:\s+(\d+)", text),
        "rx_q0_packets": _last_int(r"rx_q0_packets:\s+(\d+)", text),
    }
    m["chdr"] = {
        "processed_packets": _last_int(r"Processed packets:\s+(\d+)", text),
        "active_receive_s": _last_float(r"Active receive sec:\s+([\d.]+)", text),
        "avg_msps": _last_float(r"Average throughput:\s+([\d.]+) MSps", text),
        "batches_queued": _last_int(r"Completed batches queued:\s+(\d+)", text),
        "batches_emitted": _last_int(r"Completed batches emitted:\s+(\d+)", text),
        "partial_batch_drops": _last_int(r"Partial batch drops:\s+(\d+)", text),
        "panic_resets": _last_int(r"Panic resets:\s+(\d+)", text),
        "backlog_events": _last_int(r"Backlog events:\s+(\d+)", text),
        "ring_full_shed_packets": _last_int(r"Ring-full shed packets:\s+(\d+)", text),
        "max_out_q_depth": _last_int(r"Max out_q depth:\s+(\d+)", text),
        "mean_release_dwell_ms": _last_float(r"Mean release dwell ms:\s+([\d.]+)", text),
        "max_release_dwell_ms": _last_float(r"Max release dwell ms:\s+([\d.]+)", text),
    }
    m["fell_behind_count"] = text.count("Fell behind in processing on GPU!")
    m["dpdk_might_get_dropped_warnings"] = text.count("might get dropped")
    m["derived_fft"] = dict(re.findall(r"Derived runtime FFT config (\w+)=([\w.e+]+)", text)[:12]) if "Derived runtime FFT config" in text else {}
    fl = re.findall(r"FFT ingress latency ch=0 samples=(\d+) emitted_frames=(\d+) mean_chdr_to_fft_ms=([\d.]+) max_chdr_to_fft_ms=([\d.]+)", text)
    fl = [f for f in fl if int(f[0]) > 0]
    if fl:
        m["fft_ingress"] = {"windows": len(fl), "emitted_frames_last": int(fl[-1][1]),
                            "mean_chdr_to_fft_ms_last": float(fl[-1][2]),
                            "mean_chdr_to_fft_ms_windows_mean": float(np.mean([float(f[2]) for f in fl])),
                            "max_chdr_to_fft_ms": max(float(f[3]) for f in fl)}
    tg = [(int(f), float(t), float(mp)) for f, t, mp in
          re.findall(r"\[finetuned_dino_detector\] frame (\d+): total GPU ([\d.]+) ms \(front-end \+ model path ([\d.]+) ms\)", text)]
    if tg:
        steady = [t for f, t, _ in tg if f > 32] or [t for _, t, _ in tg]
        m["dino_total_gpu_sampled"] = {"samples": len(tg), "last_frame": tg[-1][0],
                                       "p50_ms": float(np.median(steady)), "max_ms": float(np.max(steady))}
    dl = re.findall(r"\] (downsample|native): (.*?)inference ([\d.]+) ms \(mean ([\d.]+) ms over (\d+) frames\)", text)
    if dl:
        m["dino_inference_log"] = {"path": dl[-1][0], "geometry": dl[-1][1].strip().rstrip(';'),
                                   "mean_ms_final": float(dl[-1][3]), "frames_timed": int(dl[-1][4])}
    cl = [dict((k, float(v)) for k, v in re.findall(r"(\w+)=([-\d.eE+]+)", l))
          for l in re.findall(r"Coherent power timing summary ch=0 (.*)", text)]
    if cl:
        def agg(key):
            vals = [c[key] for c in cl if key in c]
            return {"mean_of_window_means": float(np.mean(vals)), "max_window": float(np.max(vals))} if vals else None
        m["coherent_timing"] = {"windows": len(cl),
                                "fft_to_detector_done_ms": agg("fft_to_detector_done_mean"),
                                "fft_to_detector_done_max_ms": max(c.get("fft_to_detector_done_max", 0) for c in cl),
                                "pipeline_ms": agg("pipeline_ms_mean"), "total_ms": agg("total_ms_mean")}
    vl = [dict(re.findall(r"(\w+)=([-\d.eE+]+)", l)) for l in re.findall(r"Visualizer timing: (.*)", text)]
    if vl:
        last = vl[-1]
        m["visualizer_last"] = {k: float(v) for k, v in last.items() if k in (
            "frames_seen", "drop_render_busy", "backlog_folded", "drop_rate", "masks_received",
            "masks_backfilled", "render_ms_mean", "render_ms_max", "fft_to_render_ms_mean")}
        m["visualizer_last_raw"] = last
    sn = re.findall(r"signal_snipper: processed (\d+) masks, emitted (\d+) snippets", text)
    if sn:
        m["snipper"] = {"masks_processed": int(sn[-1][0]), "snippets_emitted": int(sn[-1][1])}
    sk = re.findall(r"sigmf_file_sink: total files written (\d+), (.*?)\((\d+) dropped\)", text)
    if sk:
        m["sink"] = {"files_written": int(sk[-1][0], ), "dropped": int(sk[-1][2])}
    md = re.findall(r"\[mask_dump\] stopped: (\d+) written, (\d+) submitted, (\d+) dropped", text)
    if md:
        m["mask_dump"] = {"written": int(md[-1][0]), "submitted": int(md[-1][1]), "dropped": int(md[-1][2])}
    g = re.findall(r"invalid-frame guard: suppressed .*? total suppressed=(\d+)", text)
    m["dino_guard_suppressed"] = int(g[-1]) if g else 0
    rx = re.findall(r"RX worker summary port=0 queue=0 window_s=([\d.]+) bursts=(\d+) packets=(\d+) rate=([\d.]+) Mpps .*? ring_full_drops=(\d+)", text)
    rates = [float(r[3]) for r in rx if int(r[2]) > 0]
    if rates:
        m["rx_worker"] = {"active_windows": len(rates), "rate_mpps_mean": float(np.mean(rates)),
                          "rate_mpps_max": float(np.max(rates)), "ring_full_drops_last": int(rx[-1][4])}
    # Sustainability: the RX data pool (seg2) is the buffer between the NIC and the pipeline. If the
    # pipeline is slower than the arrival rate, free buffers fall every second until the NIC starts
    # dropping (rx_out_of_buffers). A finite replay can therefore end with ZERO loss while the
    # pipeline is not real-time; the drain rate exposes that. Fit free-buffers vs time over the
    # full-rate windows (first/last partial windows excluded) while the pool is not yet exhausted.
    pool = re.findall(r"RX worker summary port=0 queue=0 window_s=([\d.]+) bursts=\d+ packets=(\d+) rate=([\d.]+) Mpps .*?seg2\(avail/in_use\)=(\d+)/(\d+)", text)
    act = [(float(w), int(n), float(r), int(a), int(u)) for w, n, r, a, u in pool if int(n) > 0]
    if len(act) >= 4:
        core = act[1:-1]
        cap = max(a + u for _, _, _, a, u in act)
        pts = [(i, a) for i, (_, _, _, a, _) in enumerate(core) if a > 0.02 * cap]
        drain_bps = None
        if len(pts) >= 2:
            # Theil-Sen (median of pairwise slopes): a steady deficit declines in every window, while a
            # single recovered dip (GC / end-of-replay hiccup) must not read as a sustained drain.
            slopes = [(pts[j][1] - pts[i][1]) / (pts[j][0] - pts[i][0])
                      for i in range(len(pts)) for j in range(i + 1, len(pts))]
            slope = float(np.median(slopes))  # buffers per ~1 s window
            drain_bps = max(0.0, -slope)
        pps = float(np.mean([n / max(w, 1e-9) for w, n, _, _, _ in core]))
        m["rx_pool"] = {
            "capacity": cap, "free_min": min(a for _, _, _, a, _ in act), "free_first": core[0][3],
            "drain_buffers_per_s": drain_bps, "arrival_pps": pps,
            "drain_frac_of_arrival": (drain_bps / pps) if (drain_bps is not None and pps > 0) else None,
            "exhausted": min(a for _, _, _, a, _ in act) < 0.02 * cap,
        }
    cs = re.findall(r"CHDR summary ch=0 window_s=([\d.]+) bursts=(\d+) packets=(\d+) rate=([\d.]+) Mpps queued=(\d+) emitted=(\d+) empty_polls=(\d+) backlog_events=(\d+) out_q_depth=(\d+) max_out_q_depth=(\d+)", text)
    cs = [c for c in cs if int(c[2]) > 0]
    if cs:
        m["chdr_summaries"] = {"active_windows": len(cs), "rate_mpps_mean": float(np.mean([float(c[3]) for c in cs])),
                               "emitted_per_window_mean": float(np.mean([int(c[5]) for c in cs])),
                               "max_out_q_depth": max(int(c[9]) for c in cs)}
    m["errors"] = len(re.findall(r"\[error\]", text))
    m["cuda_errors"] = len(re.findall(r"CUDA error|cudaError|out of memory", text))
    return m


def parse_manifest(path: Path, expected_frames: int | None, frame_period_ms: float | None) -> dict:
    rows = list(csv.DictReader(open(path)))
    if not rows:
        return {"frames_dumped": 0}
    frames = np.array(sorted(int(r["frame_number"]) for r in rows))
    det = np.array([float(r.get("detector_ms", -1) or -1) for r in rows])
    det = det[det >= 0]
    host = np.array(sorted(int(r.get("host_mono_ns", 0) or 0) for r in rows))
    host = host[host > 0]
    gaps_ms = np.diff(host) / 1e6 if host.size > 1 else np.array([])
    partial = sum(int(r.get("chdr_partial", 0) or 0) for r in rows)
    short = sum(1 for r in rows if int(r.get("chdr_expected_pkts", 0) or 0) > 0
                and int(r.get("chdr_pkts", 0) or 0) < int(r.get("chdr_expected_pkts", 0) or 0))
    occ = np.array([int(r["nonzero"]) / (int(r["rows"]) * int(r["cols"])) for r in rows])
    span = int(frames[-1] - frames[0] + 1)
    missing = span - frames.size
    out = {
        "frames_dumped": int(frames.size), "first_frame": int(frames[0]), "last_frame": int(frames[-1]),
        "frame_span": span, "missing_in_span": int(missing),
        "coverage_of_span": float(frames.size / span) if span else None,
        "expected_frames": expected_frames,
        "coverage_of_expected": (float(frames.size / expected_frames) if expected_frames else None),
        "detector_ms": stats(det), "cadence_ms": stats(gaps_ms),
        "partial_batches": int(partial), "short_batches": int(short),
        "occupancy": stats(occ), "grid": f"{rows[0]['rows']}x{rows[0]['cols']}",
    }
    if frame_period_ms and det.size:
        out["detector_over_budget_frac"] = float((det > frame_period_ms).mean())
    if frame_period_ms and gaps_ms.size:
        out["cadence_over_2x_period_frac"] = float((gaps_ms > 2 * frame_period_ms).mean())
        out["realtime_rate_x"] = float(frame_period_ms / np.median(gaps_ms)) if np.median(gaps_ms) > 0 else None
    return out


def parse_tcpreplay(path: Path) -> dict:
    t = path.read_text() if path.exists() else ""
    out = {}
    m = re.search(r"Actual: (\d+) packets \((\d+) bytes\) sent in ([\d.]+) seconds", t)
    if m:
        out.update(sent_packets=int(m.group(1)), sent_bytes=int(m.group(2)), send_seconds=float(m.group(3)))
    m = re.search(r"Rated: ([\d.]+) Bps, ([\d.]+) Mbps, ([\d.]+) pps", t)
    if m:
        out.update(rated_mbps=float(m.group(2)), rated_pps=float(m.group(3)))
    m = re.search(r"Successful packets:\s+(\d+)", t)
    if m:
        out["successful_packets"] = int(m.group(1))
    m = re.search(r"Failed packets:\s+(\d+)", t)
    if m:
        out["failed_packets"] = int(m.group(1))
    return out


def parse_gpu(path: Path, t_end_epoch: float | None, send_seconds: float | None) -> dict:
    if not path.exists():
        return {}
    samples = []
    for line in path.read_text().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            ts = datetime.strptime(parts[0], "%Y/%m/%d %H:%M:%S.%f").timestamp()
            util = float(parts[1].replace("%", ""))
            power = float(parts[3].replace("W", ""))
        except ValueError:
            continue
        samples.append((ts, util, power))
    if not samples:
        return {}
    arr = np.array(samples)
    sel = arr
    if t_end_epoch and send_seconds:
        lo, hi = t_end_epoch - send_seconds - 0.5, t_end_epoch + 1.0
        w = arr[(arr[:, 0] >= lo) & (arr[:, 0] <= hi)]
        if w.shape[0] >= 4:
            sel = w
    return {"samples_all": int(arr.shape[0]), "samples_window": int(sel.shape[0]),
            "util_mean": float(sel[:, 1].mean()), "util_p95": pct(sel[:, 1], 95), "util_max": float(sel[:, 1].max()),
            "power_w_mean": float(sel[:, 2].mean()), "power_w_max": float(sel[:, 2].max())}


def collect(run_dir: Path) -> dict:
    meta = json.loads((run_dir / "leg_meta.json").read_text()) if (run_dir / "leg_meta.json").exists() else {}
    cfg_path = None
    if meta.get("config"):
        # container path -> host path through the repo mount
        rel = meta["config"].split("/workspace/holohub/", 1)[-1]
        cand = Path(__file__).resolve().parents[5] / rel
        cfg_path = cand if cand.exists() else None
    ffts_per_batch = None
    detector_type = None
    if cfg_path:
        cfg = yaml.safe_load(cfg_path.read_text())
        ffts_per_batch = int(cfg["chdr_converter"]["num_ffts_per_batch"])
        detector_type = cfg["pipeline"]["detector_type"]
    rate = float(meta.get("rate_hz", 0) or 0)
    fft_size = auto_fft_size(rate) if rate else None
    samples_per_frame = fft_size * ffts_per_batch if (fft_size and ffts_per_batch) else None
    frame_period_ms = samples_per_frame / rate * 1e3 if samples_per_frame else None
    pcap_packets = int(meta.get("pcap_packets", 0) or 0)
    loops = int(meta.get("loops", 1) or 1)
    expected_frames = (pcap_packets * loops * 1024) // samples_per_frame if (samples_per_frame and pcap_packets) else None

    log_text = (run_dir / "app.log").read_text(errors="ignore") if (run_dir / "app.log").exists() else ""
    out = {
        "run": run_dir.name, "leg": meta.get("leg"), "detector_type": detector_type,
        "rate_hz": rate, "pps": meta.get("pps"), "loops": loops, "pcap_packets": pcap_packets,
        "fft_size": fft_size, "ffts_per_batch": ffts_per_batch, "samples_per_frame": samples_per_frame,
        "frame_period_ms": frame_period_ms, "expected_frames": expected_frames,
        "sent_samples": pcap_packets * loops * 1024 if pcap_packets else None,
        "app_log": parse_app_log(log_text),
        "tcpreplay": parse_tcpreplay(run_dir / "tcpreplay.log"),
    }
    man = run_dir / "masks" / "mask_dump_manifest.csv"
    out["manifest"] = parse_manifest(man, expected_frames, frame_period_ms) if man.exists() else {"frames_dumped": 0}
    out["gpu"] = parse_gpu(run_dir / "gpu.csv", meta.get("replay_end_epoch"), out["tcpreplay"].get("send_seconds"))
    # headline derived numbers
    d = out["app_log"]["dpdk"]
    sent = out["tcpreplay"].get("sent_packets") or (pcap_packets * loops)
    recv = d.get("received_packets")
    out["headline"] = {
        "packets_sent": sent,
        "packets_received_dpdk": recv,
        "packet_loss_frac": (1.0 - recv / sent) if (sent and recv is not None) else None,
        "dpdk_missed_packets": d.get("missed_packets"),
        "chdr_partial_batch_drops": out["app_log"]["chdr"].get("partial_batch_drops"),
        "chdr_backlog_events": out["app_log"]["chdr"].get("backlog_events"),
        "fell_behind": out["app_log"]["fell_behind_count"],
        "frames_expected": expected_frames,
        # without the mask dump, fall back to the converter's emitted-batch count / sampled detector times
        "frames_emitted": out["manifest"].get("frames_dumped") or out["app_log"]["chdr"].get("batches_emitted"),
        "frames_emitted_frac": (out["manifest"].get("coverage_of_expected")
                                or ((out["app_log"]["chdr"].get("batches_emitted") or 0) / expected_frames
                                    if expected_frames else None)),
        "detector_ms_p50": ((out["manifest"].get("detector_ms") or {}).get("p50")
                            or (out["app_log"].get("dino_total_gpu_sampled") or {}).get("p50_ms")),
        "detector_ms_p95": (out["manifest"].get("detector_ms") or {}).get("p95"),
        "detector_ms_max": (out["manifest"].get("detector_ms") or {}).get("max"),
        "frame_period_ms": frame_period_ms,
        "fft_ingress_ms_mean": (out["app_log"].get("fft_ingress") or {}).get("mean_chdr_to_fft_ms_windows_mean"),
        "fft_ingress_ms_max": (out["app_log"].get("fft_ingress") or {}).get("max_chdr_to_fft_ms"),
        "gpu_util_mean": out["gpu"].get("util_mean"), "gpu_util_max": out["gpu"].get("util_max"),
        "achieved_pps": out["tcpreplay"].get("rated_pps"),
        "pool_drain_frac": (out["app_log"].get("rx_pool") or {}).get("drain_frac_of_arrival"),
        "pool_exhausted": (out["app_log"].get("rx_pool") or {}).get("exhausted"),
        "ring_full_shed_packets": out["app_log"]["chdr"].get("ring_full_shed_packets"),
    }
    # "Keeps up" = nothing lost AND the input buffer is not filling (drain < 1% of the arrival rate).
    drain = out["headline"]["pool_drain_frac"]
    out["headline"]["sustained_realtime"] = bool(
        (out["headline"]["packet_loss_frac"] or 0.0) == 0.0 and drain is not None and drain < 0.01
        and not out["headline"]["pool_exhausted"])
    return out


def fmt(v, nd=2):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def main() -> int:
    run_dir = Path(sys.argv[1]).resolve()
    out = collect(run_dir)
    (run_dir / "metrics.json").write_text(json.dumps(out, indent=2))
    h = out["headline"]
    print(f"{out['run']:<22} {out['leg'] or '-':<20} rate={out['rate_hz']/1e6:.2f}M "
          f"frames {fmt(h['frames_emitted'])}/{fmt(h['frames_expected'])} "
          f"({fmt((h['frames_emitted_frac'] or 0)*100,1)}%) "
          f"loss={fmt((h['packet_loss_frac'] or 0)*100,3)}% missed={fmt(h['dpdk_missed_packets'])} "
          f"partial={fmt(h['chdr_partial_batch_drops'])} backlog={fmt(h['chdr_backlog_events'])} "
          f"fell_behind={h['fell_behind']} det_ms p50/p95/max={fmt(h['detector_ms_p50'])}/{fmt(h['detector_ms_p95'])}/{fmt(h['detector_ms_max'])} "
          f"(budget {fmt(h['frame_period_ms'],1)}) fft_lat mean/max={fmt(h['fft_ingress_ms_mean'],1)}/{fmt(h['fft_ingress_ms_max'],1)} "
          f"gpu={fmt(h['gpu_util_mean'],0)}%/{fmt(h['gpu_util_max'],0)}% "
          f"pool_drain={fmt((h['pool_drain_frac'] or 0)*100,1)}%{' EXHAUSTED' if h['pool_exhausted'] else ''} "
          f"sustained={'yes' if h['sustained_realtime'] else 'NO'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
