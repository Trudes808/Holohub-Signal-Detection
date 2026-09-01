#!/usr/bin/env python3
"""Offline rate-distortion-task eval for the snippet_compression operator.

For each (codec x SNR capture): run the compiled offline eval binary with the
snipper + compression stage enabled (the REAL CUDA codec kernels, same as the
live graph), then run the AMC decode daemon over the emitted snippet packs and
harvest: compression ratio, whole/attempted PN9 BER, oracle channel BER
(chBER), and per-model classification accuracy.

Output: results.csv in this directory (one row per codec x SNR) + a printed
table. Snippet scratch lives under /tmp/usrp_spectrograms/rd_eval/.

Usage (from the app root, host side):
    python3 infocom_evals/signal_detection_experiments/compression_rd/run_rd_eval.py \
        [--codecs none sc16 bfp12 bfp8] [--snrs 30 20 12 6] [--keep-scratch]

Requires the app rebuilt with snippet_compression (rebuild_demo_container_app.sh)
and the .venv-ml python for the daemon.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import signal
import shutil
import subprocess
import sys
import time
from pathlib import Path

THIS = Path(__file__).resolve().parent
APP_DIR = THIS.parents[2]
BASE_CONFIG = APP_DIR / "config_snipper_viz_demo.yaml"
CAPTURES = Path.home() / "Documents/holoscan_waveform_generation/composition/composites"
VENV_PY = Path.home() / "Documents/holoscan_waveform_generation/.venv-ml/bin/python"
SCRATCH = Path("/tmp/usrp_spectrograms/rd_eval")

CODECS = ["none", "sc16", "bfp12", "bfp8"]
SNRS = ["30", "20", "12", "6"]


def make_base_config(codec: str) -> Path:
    """Per-codec offline base config: the demo snipper config with the codec pinned
    and live control-file switching disabled (an eval must not race the dashboard)."""
    text = BASE_CONFIG.read_text()
    text = re.sub(r'(^\s*codec:\s*)"[a-z0-9]*"', rf'\g<1>"{codec}"', text, count=1, flags=re.M)
    text = re.sub(r'(^\s*control_json_path:\s*)"[^"]*"', r'\g<1>""', text, count=1, flags=re.M)
    # The offline captures are only ~4 frames long; flush a pack per frame or the sink's
    # pack_frames threshold (16 in the live demo config) never trips and nothing is written.
    text = re.sub(r"(^\s*pack_frames:\s*)\d+", r"\g<1>1", text, count=1, flags=re.M)
    out_dir = APP_DIR / "generated_configs"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"rd_base_{codec}.yaml"
    out.write_text(text)
    return out


def run_offline(codec: str, snr: str, capture: Path, out_root: Path) -> None:
    cfg = make_base_config(codec)
    cmd = [sys.executable, str(APP_DIR / "run_cuda_dino_offline_file.py"), str(capture),
           "--detector", "coherent_power", "--config", str(cfg),
           "--snippets-only", "--output-root", str(out_root)]
    print(f"[rd] offline eval: codec={codec} snr={snr} dB", flush=True)
    subprocess.run(cmd, check=True, cwd=APP_DIR,
                   stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def run_daemon(snips_dir: Path, run_dir: Path, snr: str, timeout_s: float = 900.0) -> dict:
    """Run the decode daemon over a static snippet dir until it drains, then harvest."""
    metrics_path = run_dir / "rt_metrics.json"
    control_path = run_dir / "demo_control.json"
    control_path.write_text(json.dumps(
        {"gate": "tprime", "snr": snr, "detector": "coherent_power", "seq": 1}) + "\n")
    cmd = [str(VENV_PY), str(APP_DIR / "infocom_evals/pycodec_e2e/rt_decode_daemon.py"),
           "--snips", str(snips_dir), "--metrics-out", str(metrics_path),
           "--control-json", str(control_path)]
    log = open(run_dir / "daemon.log", "w")
    proc = subprocess.Popen(cmd, cwd=APP_DIR / "infocom_evals/pycodec_e2e",
                            stdout=log, stderr=subprocess.STDOUT)
    t0 = time.time()
    last_seen, stable, metrics = -1, 0, {}
    try:
        while time.time() - t0 < timeout_s:
            time.sleep(4.0)
            try:
                metrics = json.loads(metrics_path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            seen = int(metrics.get("snippets_seen", 0))
            stable = stable + 1 if seen == last_seen and seen > 0 else 0
            last_seen = seen
            if stable >= 3:  # no new snippets across ~12 s: pack dir is drained
                break
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()
    return metrics


def harvest(codec: str, snr: str, metrics: dict, snips_dir: Path) -> dict:
    row = {"codec": codec, "snr_db": snr}
    row["ratio"] = metrics.get("comp_ratio", 1.0)
    row["stored_mb"] = round(sum(f.stat().st_size for f in snips_dir.glob("*.sigmf-data")) / 1e6, 2)
    row["snips"] = metrics.get("snippets_seen", 0)
    row["frames"] = metrics.get("frames_decoded", 0)
    row["crc_ok"] = metrics.get("frames_crc_ok", 0)
    row["ber_attempted"] = metrics.get("ber_attempted")
    row["ber_whole"] = metrics.get("ber_whole", metrics.get("pn9_ber"))
    models = metrics.get("classifier", {}).get("models", {})
    for name in ("vtcnn2", "resnet1d", "tprime"):
        row[f"acc_{name}"] = models.get(name, {}).get("acc")
    b = metrics.get("by_snr", {}).get(snr, {})
    row["chber"] = b.get("chber")
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--codecs", nargs="+", default=CODECS)
    ap.add_argument("--snrs", nargs="+", default=SNRS)
    ap.add_argument("--keep-scratch", action="store_true")
    args = ap.parse_args()

    rows = []
    for codec in args.codecs:
        for snr in args.snrs:
            capture = CAPTURES / f"snr_single_{snr}db.sigmf-data"
            if not capture.exists():
                print(f"[rd] MISSING capture {capture}, skipping", flush=True)
                continue
            run_dir = SCRATCH / f"{codec}_{snr}db"
            if run_dir.exists():  # container-side writes are root-owned
                subprocess.run(["sudo", "rm", "-rf", str(run_dir)], check=True)
            run_dir.mkdir(parents=True)
            run_offline(codec, snr, capture, run_dir)
            snips_dir = run_dir / "snippets"
            if not snips_dir.exists():
                print(f"[rd] no snippets emitted for codec={codec} snr={snr}", flush=True)
                continue
            # daemon artifacts live in a harness-owned dir (the container re-owns run_dir as root)
            daemon_dir = SCRATCH / f"{codec}_{snr}db_daemon"
            if daemon_dir.exists():
                subprocess.run(["sudo", "rm", "-rf", str(daemon_dir)], check=True)
            daemon_dir.mkdir(parents=True)
            metrics = run_daemon(snips_dir, daemon_dir, snr)
            row = harvest(codec, snr, metrics, snips_dir)
            rows.append(row)
            print(f"[rd]   ratio={row['ratio']}x stored={row['stored_mb']}MB "
                  f"wBER={row['ber_whole']} chBER={row['chber']} "
                  f"acc(T-PRIME)={row['acc_tprime']}", flush=True)
            if not args.keep_scratch:
                subprocess.run(["sudo", "rm", "-rf", str(snips_dir)], check=False)

    if rows:
        out = THIS / "results.csv"
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[rd] wrote {out} ({len(rows)} rows)")
        hdr = f"{'codec':>6} {'snr':>4} {'ratio':>6} {'MB':>8} {'wBER':>10} {'chBER':>10} {'accT':>6}"
        print(hdr)
        for r in rows:
            print(f"{r['codec']:>6} {r['snr_db']:>4} {r['ratio']!s:>6} {r['stored_mb']:>8} "
                  f"{r['ber_whole']!s:>10} {r['chber']!s:>10} {r['acc_tprime']!s:>6}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
