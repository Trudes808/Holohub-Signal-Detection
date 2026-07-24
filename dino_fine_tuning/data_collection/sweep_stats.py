#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 National Instruments Corporation
# SPDX-License-Identifier: Apache-2.0
"""
Turn one or more sweep_capture runs into the augmentation sidecar the dataset builder consumes.

Reads each run dir's psd.f32 (+ manifest.jsonl + sweep_meta.json), merging phases (low/high) and
passes. Produces:
  - envelopes.npz : per-rate receiver ENVELOPE template (nfft, zero-mean dB shape) from the TERMINATED
                    channel via median-across-bursts (level-normalized first -> pure shape, gain-agnostic).
  - floor_stats.json : per-(rate, role) noise-floor dB distribution (min/median/max/pcts). Terminated =
                    clean floor lower bound; antenna = occupied/upper range. Sets the LEVEL aug range.
  - backgrounds.json : index of saved antenna raw-IQ bursts (for cut-paste backgrounds), by rate.

Usage:
  ./sweep_stats.py --run-dirs <dir_low> [<dir_high> ...] --out <sidecar_dir>
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dirs", nargs="+", required=True, help="One or more sweep_capture output dirs.")
    p.add_argument("--out", required=True, help="Output sidecar dir.")
    p.add_argument("--floor-percentile", type=float, default=20.0,
                   help="Percentile of per-bin dB used as the per-burst floor scalar (robust to signals).")
    return p.parse_args()


def load_run(run_dir: str):
    meta_path = os.path.join(run_dir, "sweep_meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    nfft = int(meta["nfft"])
    recs = []
    with open(os.path.join(run_dir, "manifest.jsonl")) as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    psd_path = os.path.join(run_dir, "psd.f32")
    n_rows = os.path.getsize(psd_path) // (nfft * 4)
    psd = np.memmap(psd_path, dtype=np.float32, mode="r", shape=(n_rows, nfft))
    return meta, nfft, recs, psd


def main() -> int:
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    # role -> rate_hz -> list of per-burst PSD rows (np arrays)
    by_role_rate: dict[str, dict[float, list[np.ndarray]]] = {}
    backgrounds: dict[str, list[dict]] = {}
    nfft_seen = None
    total = 0

    for run_dir in args.run_dirs:
        meta, nfft, recs, psd = load_run(run_dir)
        if nfft_seen is None:
            nfft_seen = nfft
        elif nfft != nfft_seen:
            print(f"WARN: nfft mismatch ({nfft} vs {nfft_seen}) in {run_dir}; skipping.", file=sys.stderr)
            continue
        for r in recs:
            row = r.get("row")
            if row is None or row >= psd.shape[0]:
                continue
            role = r.get("role", "?")
            rate = float(r["rate_hz"])
            by_role_rate.setdefault(role, {}).setdefault(rate, []).append(np.asarray(psd[row]))
            if r.get("iq_file"):
                backgrounds.setdefault(f"{rate:.0f}", []).append(
                    {"iq_file": os.path.join(run_dir, r["iq_file"]), "center_hz": r["center_hz"],
                     "gain_db": r["gain_db"], "rate_hz": rate})
            total += 1
        print(f"loaded {run_dir}: {len(recs)} rows")

    if not by_role_rate:
        print("No data loaded.", file=sys.stderr); return 2

    # ---- per-rate envelope templates from the terminated channel ------------------------------
    envelopes = {}          # rate -> zero-mean dB shape [nfft]
    envelope_std = {}
    term = by_role_rate.get("terminated", {})
    for rate, rows in sorted(term.items()):
        stack = np.stack(rows, axis=0)                        # [n, nfft], dB
        shape = stack - np.median(stack, axis=1, keepdims=True)   # level-normalize each burst -> pure shape
        env = np.median(shape, axis=0).astype(np.float32)     # median across bursts rejects any residual signal
        envelopes[f"{rate:.0f}"] = env
        envelope_std[f"{rate:.0f}"] = float(np.median(np.std(shape, axis=0)))
        print(f"  envelope @ {rate/1e6:.3f} MS/s: {len(rows)} bursts, edge droop "
              f"{env[0]:.1f}/{env[-1]:.1f} dB vs center, jitter~{envelope_std[f'{rate:.0f}']:.2f} dB")
    if not envelopes:
        print("WARN: no terminated-channel data -> no clean envelope templates. "
              "Falling back to antenna median (less clean).", file=sys.stderr)
        for rate, rows in sorted(by_role_rate.get("antenna", {}).items()):
            stack = np.stack(rows, axis=0)
            shape = stack - np.median(stack, axis=1, keepdims=True)
            envelopes[f"{rate:.0f}"] = np.median(shape, axis=0).astype(np.float32)

    # ---- per-(rate, role) floor-level distribution -> level aug range -------------------------
    floor_stats = {}
    for role, rate_map in by_role_rate.items():
        floor_stats[role] = {}
        for rate, rows in sorted(rate_map.items()):
            floors = np.array([np.percentile(row, args.floor_percentile) for row in rows])
            floor_stats[role][f"{rate:.0f}"] = {
                "n": int(floors.size), "min": float(floors.min()), "p10": float(np.percentile(floors, 10)),
                "median": float(np.median(floors)), "p90": float(np.percentile(floors, 90)),
                "max": float(floors.max())}
    # global level-offset range = span of observed floors (terminated low .. antenna high), for the builder.
    all_floors = [v["median"] for role in floor_stats for v in floor_stats[role].values()]
    lo = min(v["min"] for role in floor_stats for v in floor_stats[role].values())
    hi = max(v["max"] for role in floor_stats for v in floor_stats[role].values())
    level_offset_range_db = [round(lo - float(np.median(all_floors)), 2),
                             round(hi - float(np.median(all_floors)), 2)]

    # ---- write sidecar -------------------------------------------------------------------------
    np.savez(os.path.join(args.out, "envelopes.npz"),
             **{f"rate_{k}": v for k, v in envelopes.items()},
             rates=np.array([float(k) for k in envelopes.keys()]))
    with open(os.path.join(args.out, "floor_stats.json"), "w") as f:
        json.dump({"floor_percentile": args.floor_percentile, "floor_stats": floor_stats,
                   "level_offset_range_db": level_offset_range_db, "nfft": nfft_seen}, f, indent=2)
    with open(os.path.join(args.out, "backgrounds.json"), "w") as f:
        json.dump({"by_rate": backgrounds, "n": sum(len(v) for v in backgrounds.values())}, f, indent=2)

    print(f"\n{total} rows over {len(args.run_dirs)} run(s). Envelopes: {len(envelopes)} rates. "
          f"Level-offset aug range (dB): {level_offset_range_db}. Background IQ: "
          f"{sum(len(v) for v in backgrounds.values())}. -> {args.out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
