#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 National Instruments Corporation
# SPDX-License-Identifier: Apache-2.0
"""
Automated UNLABELED deployment-range sweep for the band/rate-invariant DINO retrain (USRP X410).

Captures channels 0 (antenna) and 1 (terminated/null) SIMULTANEOUSLY over center x rate x gain in
short bursts, so one sweep yields both: the antenna PSD (real backgrounds + level upper range) and the
terminated PSD (clean receiver envelope + noise-floor lower bound) under perfectly matched conditions.
Occupancy is uncontrolled -> this data is for CHARACTERIZATION, never labels. See
applications/usrp_wideband_signal_detection/notes/retrain_band_rate_invariant_plan.md.

Per successful (channel, center, rate, gain) cell we append a per-frequency PSD (dB) row to psd.f32
and a JSON line to manifest.jsonl. Failed cells go to failures.jsonl and print FAILED. Raw IQ is saved
for a subset of antenna-channel bursts (cut-paste backgrounds). Rates are grouped by master clock
(491.52 vs 500 MHz); the device is (re)created per clock (no runtime clock switch). Rates >245.76 MS/s
need a different X410 FPGA image -> run --phase low first, reimage, then --phase high, and merge dirs.

  # preflight sanity check (radio ready? PSD sane? enough disk?) -- does NOT run the full sweep:
  ./sweep_capture.py --device-args "addr=192.168.10.2" --out-dir <dir> --preflight
  # full low-phase run (both channels):
  ./sweep_capture.py --device-args "addr=192.168.10.2" --out-dir <dir> --phase low
  # resume an interrupted run / retry just the failed cells:
  ./sweep_capture.py ... --out-dir <dir> --resume
  ./sweep_capture.py ... --out-dir <dir> --retry-failed <dir>/failures.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

import numpy as np

try:
    from tqdm import tqdm as _tqdm
except Exception:
    _tqdm = None

try:
    import uhd
except Exception as exc:  # pragma: no cover - bench dependency
    print(f"ERROR: could not import uhd ({exc}). Run on the X410 host with the UHD python bindings.",
          file=sys.stderr)
    raise

# The usable sample rates on the X410 are integer decimations of the MASTER CLOCK, and which master
# clocks are legal depends on the loaded FPGA image:
#   200 MHz-bandwidth image  -> master clocks 245.76 / 250 MHz   (this is the STOCK image; rates <=~200)
#   400 MHz-bandwidth image  -> master clocks 491.52 / 500 MHz   (needs a reimage; rates up to ~491.52/500)
# So pass --master-clocks-hz matching the CURRENTLY LOADED image; rates are auto-derived as decimations
# within [--rate-min-hz, --rate-max-hz]. Two-image workflow: sweep the 200-image clocks, reimage, sweep
# the 400-image clocks, then merge the run dirs in sweep_stats.py.
STOCK_200_CLOCKS_HZ = [245_760_000.0, 250_000_000.0]
WIDEBAND_400_CLOCKS_HZ = [491_520_000.0, 500_000_000.0]


def build_rate_clock_pairs(clocks_hz, rate_min_hz, rate_max_hz, decims, dedup_rel_tol):
    """(rate, master_clock) for the given integer decimations of each clock within [min,max].

    The two clocks per image (e.g. 245.76 & 250) produce near-identical rates/envelopes, so rates within
    dedup_rel_tol of an already-kept rate are dropped (first clock wins). This keeps the sweep to a clean
    representative set (~5-6 rates/image); the training capture-chain emulation covers the continuum, so
    the sweep only needs to characterize the envelope/level trend across the range."""
    kept = []
    for mcr in clocks_hz:
        for d in decims:
            rate = mcr / d
            if not (rate_min_hz - 1 <= rate <= rate_max_hz + 1):
                continue
            if any(abs(rate - r) <= dedup_rel_tol * r for r, _ in kept):
                continue
            kept.append((rate, mcr))
    return sorted(kept)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device-args", default="", help="UHD device args, e.g. 'addr=192.168.10.2'.")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--channels", type=int, nargs="+", default=[0, 1], help="Capture channels (X410).")
    p.add_argument("--channel-roles", default="0=antenna,1=terminated",
                   help="Comma list mapping channel->role; roles: 'antenna' or 'terminated'.")
    p.add_argument("--nfft", type=int, default=1024)
    p.add_argument("--frames-per-burst", type=int, default=4, help="burst samples = frames*nfft.")
    p.add_argument("--center-start-hz", type=float, default=50e6)
    p.add_argument("--center-stop-hz", type=float, default=5900e6)
    p.add_argument("--center-count", type=int, default=100)
    p.add_argument("--centers-hz", type=float, nargs="+", default=None,
                   help="Explicit center list (Hz), overriding --center-start/stop/count. Use for "
                        "targeted captures at specific bands, e.g. a finetune_val set: "
                        "--centers-hz 100e6 500e6 1200e6 2400e6. Pair with a large --frames-per-burst "
                        "and --save-iq-every 1 to save full model-frame IQ at every cell.")
    p.add_argument("--gains-db", type=float, nargs="+", default=None)
    p.add_argument("--gain-levels", type=int, default=6)
    p.add_argument("--save-iq-every", type=int, default=20, help="Save antenna raw IQ every Nth burst; 0=never.")
    p.add_argument("--settle-s", type=float, default=0.05)
    p.add_argument("--master-clocks-hz", type=float, nargs="+", default=STOCK_200_CLOCKS_HZ,
                   help="Master clock(s) the CURRENTLY LOADED FPGA image supports. Default = the stock "
                        "200 MHz image clocks (245.76/250). After reimaging to the 400 MHz image, pass "
                        "491.52e6 500e6. An unsupported clock is skipped with a message (not fatal).")
    p.add_argument("--rate-min-hz", type=float, default=20e6)
    p.add_argument("--rate-max-hz", type=float, default=500e6)
    p.add_argument("--decims", type=int, nargs="+", default=[1, 2, 4, 8, 12, 16, 24],
                   help="Integer decimations of each master clock to sweep (clean representative set).")
    p.add_argument("--dedup-rel-tol", type=float, default=0.02,
                   help="Drop rates within this relative tolerance of an already-kept rate (collapses the "
                        "near-identical 245.76/250 and 491.52/500 families). 0 = keep all.")
    p.add_argument("--min-free-gb", type=float, default=5.0, help="Abort if the run would leave less than this free.")
    p.add_argument("--preflight", action="store_true", help="Sanity-check only: one burst/channel + disk check, then exit.")
    p.add_argument("--resume", action="store_true", help="Skip cells already recorded in manifest.jsonl.")
    p.add_argument("--retry-failed", default=None, help="Path to a failures.jsonl; run ONLY those cells.")
    return p.parse_args()


def psd_db(samps: np.ndarray, nfft: int, frames: int) -> np.ndarray:
    """Per-bin power (dB), fftshifted, averaged over `frames` FFT blocks -- matches the training front-end."""
    need = frames * nfft
    x = samps[:need]
    if x.size < need:
        x = np.concatenate([x, np.zeros(need - x.size, dtype=x.dtype)])
    blk = x.reshape(frames, nfft)
    spec = np.fft.fftshift(np.fft.fft(blk, axis=-1), axes=-1)
    power = spec.real.astype(np.float64) ** 2 + spec.imag.astype(np.float64) ** 2 + 1e-12
    return (10.0 * np.log10(power)).mean(axis=0).astype(np.float32)


def cell_key(rate, gain, center, chan) -> str:
    return f"{rate:.0f}_{gain:.3f}_{center:.0f}_ch{chan}"


def plot_preflight(data, nfft, out_dir):
    """Sanity plots from the preflight bursts: (1) per-channel envelope shapes overlaid across rates
    (median-subtracted, normalized freq -> should overlay if the envelope is rate-stable), (2) absolute
    antenna-vs-terminated PSD at a mid rate. Returns saved paths (or [] if matplotlib is unavailable)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"  [preflight] matplotlib unavailable ({exc}) -> skipping plots (pip install matplotlib).")
        return []
    if not data:
        return []
    roles = sorted({d[4] for d in data})
    rates = sorted({d[0] for d in data})
    norm_f = np.linspace(-0.5, 0.5, nfft)
    saved = []

    # (1) envelope shapes per role, overlaid across rates
    fig, axes = plt.subplots(1, len(roles), figsize=(7 * len(roles), 4.5), squeeze=False)
    for ax, role in zip(axes[0], roles):
        for rate in rates:
            rows = [p for (r, g, c, ch, ro, p) in data if ro == role and r == rate]
            if not rows:
                continue
            p = rows[0]
            ax.plot(norm_f, p - np.median(p), lw=1.0, label=f"{rate/1e6:.2f} MS/s")
        ax.set_title(f"{role}: envelope shape vs rate (median-subtracted)")
        ax.set_xlabel("normalized frequency"); ax.set_ylabel("dB above median")
        ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    p1 = os.path.join(out_dir, "preflight_envelope.png")
    fig.savefig(p1, dpi=110); plt.close(fig); saved.append(p1)

    # (2) absolute PSD (antenna vs terminated) at the mid rate
    mid_rate = rates[len(rates) // 2]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    baseband_mhz = np.linspace(-mid_rate / 2, mid_rate / 2, nfft) / 1e6
    for role in roles:
        rows = [p for (r, g, c, ch, ro, p) in data if ro == role and r == mid_rate]
        if rows:
            ax.plot(baseband_mhz, rows[0], lw=1.0, label=role)
    ax.set_title(f"absolute PSD @ {mid_rate/1e6:.2f} MS/s (antenna vs terminated)")
    ax.set_xlabel("baseband frequency (MHz)"); ax.set_ylabel("power (dB)")
    ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout()
    p2 = os.path.join(out_dir, "preflight_psd.png")
    fig.savefig(p2, dpi=110); plt.close(fig); saved.append(p2)
    return saved


def parse_roles(spec: str) -> dict:
    roles = {}
    for tok in spec.split(","):
        if "=" in tok:
            ch, role = tok.split("=", 1)
            roles[int(ch)] = role.strip()
    return roles


def estimate_disk_bytes(n_cells, n_chan, nfft, frames, save_iq_every) -> int:
    psd_bytes = n_cells * n_chan * nfft * 4                      # float32 per (cell,channel)
    n_iq = (n_cells // save_iq_every) if save_iq_every else 0    # antenna IQ subset (1 antenna channel)
    iq_bytes = n_iq * frames * nfft * 8                          # complex64
    return psd_bytes + iq_bytes


def main() -> int:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    iq_dir = os.path.join(args.out_dir, "iq")
    if args.save_iq_every:
        os.makedirs(iq_dir, exist_ok=True)
    roles = parse_roles(args.channel_roles)
    rates = build_rate_clock_pairs(args.master_clocks_hz, args.rate_min_hz, args.rate_max_hz,
                                   args.decims, args.dedup_rel_tol)
    if not rates:
        print("No rates derived. Check --master-clocks-hz / --rate-min-hz / --rate-max-hz / --max-decim.",
              file=sys.stderr); return 2
    centers = (np.asarray(args.centers_hz, dtype=float) if args.centers_hz
               else np.linspace(args.center_start_hz, args.center_stop_hz, args.center_count))
    burst_samps = args.frames_per_burst * args.nfft

    # ---- disk preflight ------------------------------------------------------------------------
    n_cells_full = len(rates) * (len(args.gains_db) if args.gains_db else args.gain_levels) * len(centers)
    est = estimate_disk_bytes(n_cells_full, len(args.channels), args.nfft, args.frames_per_burst,
                              args.save_iq_every)
    free = shutil.disk_usage(args.out_dir).free
    print(f"[preflight] master_clocks(MHz)={[round(m/1e6,2) for m in args.master_clocks_hz]} "
          f"rates(MS/s)={[round(r/1e6,3) for r, _ in rates]}")
    print(f"[preflight] centers={len(centers)} gains={len(args.gains_db) if args.gains_db else args.gain_levels} "
          f"channels={args.channels} roles={roles}")
    print(f"[preflight] estimated disk: {est/1e9:.2f} GB  (PSD tiny + IQ every {args.save_iq_every}); "
          f"free on '{args.out_dir}': {free/1e9:.2f} GB")
    if est > free - args.min_free_gb * 1e9:
        print(f"[preflight] ABORT: estimated {est/1e9:.2f} GB would leave < {args.min_free_gb} GB free. "
              f"Lower --center-count / --gain-levels, raise --save-iq-every, or free space.", file=sys.stderr)
        return 3

    # ---- resume / retry bookkeeping ------------------------------------------------------------
    manifest_path = os.path.join(args.out_dir, "manifest.jsonl")
    failures_path = os.path.join(args.out_dir, "failures.jsonl")
    psd_path = os.path.join(args.out_dir, "psd.f32")
    done_keys = set()
    if args.resume and os.path.exists(manifest_path):
        with open(manifest_path) as f:
            for line in f:
                try:
                    done_keys.add(json.loads(line)["key"])
                except Exception:
                    pass
        print(f"[resume] {len(done_keys)} cells already done -> skipping them.")
    retry_cells = None
    if args.retry_failed:
        retry_cells = []
        with open(args.retry_failed) as f:
            for line in f:
                try:
                    retry_cells.append(json.loads(line))
                except Exception:
                    pass
        print(f"[retry] re-running {len(retry_cells)} failed cells from {args.retry_failed}")

    # next psd row index = current file size / (nfft*4)
    row = os.path.getsize(psd_path) // (args.nfft * 4) if os.path.exists(psd_path) else 0
    psd_f = open(psd_path, "ab")
    man_f = open(manifest_path, "a")
    fail_f = open(failures_path, "a")
    n_ok = n_fail = 0
    t0 = time.monotonic()

    groups: dict[float, list[float]] = {}
    for r, m in rates:
        groups.setdefault(m, []).append(r)

    preflight_data = []  # (rate, gain, center, channel, role, psd) collected in --preflight

    def do_cell(usrp, rate, mcr, gain, center, preflight=False):
        nonlocal row, n_ok, n_fail
        try:
            data = usrp.recv_num_samps(burst_samps, float(center), float(rate), list(args.channels), float(gain))
        except Exception as exc:
            for ch in args.channels:
                rec = {"key": cell_key(rate, gain, center, ch), "center_hz": float(center),
                       "rate_hz": float(rate), "master_clock_hz": float(mcr), "gain_db": float(gain),
                       "channel": ch, "role": roles.get(ch, "?"), "error": str(exc)}
                if not preflight:
                    fail_f.write(json.dumps(rec) + "\n")
            n_fail += 1
            print(f"  FAILED c={center/1e6:.0f}MHz r={rate/1e6:.3f} g={gain:.1f}: {exc}")
            return False
        if args.settle_s:
            time.sleep(args.settle_s)
        for ci, ch in enumerate(args.channels):
            p = psd_db(data[ci], args.nfft, args.frames_per_burst)
            if preflight:
                dyn = float(p.max() - p.min())
                ok = np.isfinite(p).all() and dyn > 1.0
                print(f"  [preflight] r={rate/1e6:.3f}MS/s ch{ch}({roles.get(ch,'?')}): "
                      f"floor~{np.median(p):.1f} dB, dyn-range={dyn:.1f} dB -> "
                      f"{'OK' if ok else 'SUSPECT (flat/NaN)'}")
                preflight_data.append((float(rate), float(gain), float(center), ch, roles.get(ch, "?"), p))
                continue
            p.tofile(psd_f)
            rec = {"key": cell_key(rate, gain, center, ch), "row": row, "center_hz": float(center),
                   "rate_hz": float(rate), "master_clock_hz": float(mcr), "gain_db": float(gain),
                   "channel": ch, "role": roles.get(ch, "?"), "nfft": args.nfft}
            if roles.get(ch) == "antenna" and args.save_iq_every and (n_ok % args.save_iq_every == 0):
                iq_path = os.path.join(iq_dir, f"burst_r{int(rate)}_g{gain:.0f}_c{int(center)}_ch{ch}.npy")
                np.save(iq_path, data[ci].astype(np.complex64))
                rec["iq_file"] = os.path.relpath(iq_path, args.out_dir)
            man_f.write(json.dumps(rec) + "\n")
            row += 1
        n_ok += 1
        return True

    # ---- run --------------------------------------------------------------------------------------
    bar = _tqdm(total=n_cells_full, desc="sweep", unit="cell", dynamic_ncols=True) \
        if (_tqdm is not None and not args.preflight) else None
    for mcr, group_rates in groups.items():
        dev = args.device_args + (f",master_clock_rate={mcr:.0f}" if args.device_args else f"master_clock_rate={mcr:.0f}")
        print(f"\n=== master clock {mcr/1e6:.2f} MHz : rates {[r/1e6 for r in group_rates]} MS/s ===")
        try:
            usrp = uhd.usrp.MultiUSRP(dev)
        except RuntimeError as exc:
            print(f"  SKIP master clock {mcr/1e6:.2f} MHz -- not supported by the loaded FPGA image "
                  f"({exc}). Load the matching image (245.76/250 -> 200 MHz image; 491.52/500 -> 400 MHz "
                  f"image) or drop it from --master-clocks-hz.", file=sys.stderr)
            continue
        gr = usrp.get_rx_gain_range(args.channels[0])
        gains = list(args.gains_db) if args.gains_db is not None else \
            list(np.linspace(gr.start(), gr.stop(), max(1, args.gain_levels)))
        print(f"  gain range {gr.start():.1f}..{gr.stop():.1f} dB -> {[round(g,1) for g in gains]}")

        if args.preflight:
            # one burst per rate at mid gain/center -> sanity plots of the envelope trend vs rate.
            for rate in group_rates:
                do_cell(usrp, rate, mcr, gains[len(gains)//2], centers[len(centers)//2], preflight=True)
            del usrp
            continue

        if retry_cells is not None:
            for c in retry_cells:
                if abs(c["master_clock_hz"] - mcr) > 1:
                    continue
                do_cell(usrp, c["rate_hz"], mcr, c["gain_db"], c["center_hz"])
                if bar: bar.update(1)
        else:
            for rate in group_rates:
                for gain in gains:
                    for center in centers:
                        if args.resume and all(cell_key(rate, gain, center, ch) in done_keys for ch in args.channels):
                            if bar: bar.update(1)
                            continue
                        do_cell(usrp, rate, mcr, gain, center)
                        if bar:
                            bar.set_postfix(ok=n_ok, failed=n_fail, refresh=False); bar.update(1)
                        elif (n_ok + n_fail) % 50 == 0 and (n_ok + n_fail) > 0:
                            print(f"  {n_ok} ok / {n_fail} failed ({time.monotonic()-t0:.0f}s)", flush=True)
        del usrp

    if bar: bar.close()
    psd_f.close(); man_f.close(); fail_f.close()
    with open(os.path.join(args.out_dir, "sweep_meta.json"), "w") as f:
        json.dump({"nfft": args.nfft, "frames_per_burst": args.frames_per_burst, "burst_samps": burst_samps,
                   "channels": args.channels, "roles": roles,
                   "master_clocks_hz": args.master_clocks_hz,
                   "rates_hz": [r for r, _ in rates], "centers_hz": centers.tolist(),
                   "n_ok": n_ok, "n_fail": n_fail}, f, indent=2)
    if args.preflight:
        saved = plot_preflight(preflight_data, args.nfft, args.out_dir)
        for p in saved:
            print(f"  [preflight] wrote {p}")
        print("\n[preflight] radio + disk check complete. Eyeball the plots above; re-run without "
              "--preflight for the full sweep.")
        return 0
    print(f"\nDone: {n_ok} ok, {n_fail} failed in {time.monotonic()-t0:.0f}s -> {args.out_dir}")
    if n_fail:
        print(f"  Retry the failed cells with:  --retry-failed {failures_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
