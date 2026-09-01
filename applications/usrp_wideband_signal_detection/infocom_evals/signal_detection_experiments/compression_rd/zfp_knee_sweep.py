#!/usr/bin/env python3
"""ZFP fixed-rate knee sweep: where does compression start costing the task?

For each capture (clean/30/20/12 dB) x zfp rate (bits per scalar), take the
UNCOMPRESSED offline run's snippet packs, re-encode every snippet with zfp
fixed-rate (planar I/Q, zfpy streams -- bitstream-identical to what the CUDA
backend would produce), measure reconstruction SNR inline, then run the decode
daemon over the reconstructions and harvest chBER / whole BER / accuracy.

zfp rates 1-2 crash zfpy 1.0.1 on aarch64; the sweep floor is rate 3 (10.7x).

Run (host, app root):  python3 .../zfp_knee_sweep.py [--rates 12 10 8 6 5 4 3]
Outputs: zfp_knee_results.csv + zfp_knee.png in this directory.
"""
from __future__ import annotations

import argparse
import csv
import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

THIS = Path(__file__).resolve().parent
APP_DIR = THIS.parents[2]
sys.path.insert(0, str(THIS))
from run_rd_eval import CAPTURES, SCRATCH, VENV_PY, harvest, make_base_config, run_daemon  # noqa: E402

RATES = [12, 10, 8, 6, 5, 4, 3]
CAPS = {"clean": "snr_clean_burst.sigmf-data",
        "30": "snr_single_30db.sigmf-data",
        "20": "snr_single_20db.sigmf-data",
        "12": "snr_single_12db.sigmf-data"}


def ensure_raw_pack(label: str) -> Path:
    """One uncompressed offline run per capture (cached across sweep invocations)."""
    out_root = SCRATCH / f"zfpraw_{label}"
    snips = out_root / "snippets"
    if snips.exists() and any(snips.glob("*.sigmf-meta")):
        return snips
    cfg = make_base_config("none")
    cap = CAPTURES / CAPS[label]
    print(f"[zfp] raw offline run for {label}", flush=True)
    subprocess.run([sys.executable, str(APP_DIR / "run_cuda_dino_offline_file.py"), str(cap),
                    "--detector", "coherent_power", "--config", str(cfg),
                    "--snippets-only", "--output-root", str(out_root)],
                   check=True, cwd=APP_DIR, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    return snips


def repack_zfp(raw_dir: Path, out_dir: Path, rate: int) -> float:
    """Re-encode every snippet chunk with zfp fixed-rate; returns mean recon SNR (dB).
    Runs in THIS process (zfpy is safe for rates >= 3)."""
    import zfpy
    out_dir.mkdir(parents=True, exist_ok=True)
    snr_num, snr_den = 0.0, 0.0
    for mp in sorted(raw_dir.glob("*.sigmf-meta")):
        meta = json.load(open(mp))
        raw = np.fromfile(str(mp).replace(".sigmf-meta", ".sigmf-data"), np.complex64)
        payload = bytearray()
        for a in meta["annotations"]:
            s, n = int(a["core:sample_start"]), int(a["core:sample_count"])
            iq = raw[s:s + n]
            ci = zfpy.compress_numpy(np.ascontiguousarray(iq.real, np.float32), rate=rate)
            cq = zfpy.compress_numpy(np.ascontiguousarray(iq.imag, np.float32), rate=rate)
            chunk = struct.pack("<I", len(ci)) + ci + cq
            a["wfgt:compression"] = f"zfp{rate}"
            a["wfgt:comp_byte_offset"] = len(payload)
            a["wfgt:comp_byte_count"] = len(chunk)
            payload += chunk
            ri = zfpy.decompress_numpy(ci)[:n]
            rq = zfpy.decompress_numpy(cq)[:n]
            err = (iq.real - ri) ** 2 + (iq.imag - rq) ** 2
            snr_num += float(np.sum(np.abs(iq) ** 2))
            snr_den += float(np.sum(err))
        meta["global"]["core:datatype"] = "u8"
        meta["global"]["wfgt:compressed_container"] = True
        (out_dir / mp.name).write_text(json.dumps(meta, indent=1))
        (out_dir / mp.name.replace(".sigmf-meta", ".sigmf-data")).write_bytes(bytes(payload))
    return 10.0 * np.log10(snr_num / max(snr_den, 1e-30))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rates", nargs="+", type=int, default=RATES)
    ap.add_argument("--caps", nargs="+", default=list(CAPS))
    args = ap.parse_args()

    rows = []
    for label in args.caps:
        raw_dir = ensure_raw_pack(label)
        for rate in args.rates:
            if rate < 3:
                print(f"[zfp] rate {rate} skipped (zfpy aarch64 crash below rate 3)")
                continue
            run_dir = SCRATCH / f"zfp{rate}_{label}"
            subprocess.run(["sudo", "rm", "-rf", str(run_dir)], check=False)
            run_dir.mkdir(parents=True)
            recon_snr = repack_zfp(raw_dir, run_dir / "snippets", rate)
            metrics = run_daemon(run_dir / "snippets", run_dir, label)
            row = harvest(f"zfp{rate}", label, metrics, run_dir / "snippets")
            row["bits_per_scalar"] = rate
            row["recon_snr_db"] = round(recon_snr, 1)
            rows.append(row)
            print(f"[zfp] {label:>5} dB rate {rate:>2}: ratio={row['ratio']}x "
                  f"recon={recon_snr:.1f}dB chBER={row['chber']} accT={row['acc_tprime']}",
                  flush=True)

    out = THIS / "zfp_knee_results.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[zfp] wrote {out} ({len(rows)} rows)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        for label in args.caps:
            sub = [r for r in rows if r["snr_db"] == label]
            x = [32.0 / r["bits_per_scalar"] for r in sub]  # ratio vs cf32
            axes[0].semilogy(x, [max(float(r["chber"] or 0), 1e-6) for r in sub], "o-", label=f"{label} dB")
            axes[1].plot(x, [r["acc_tprime"] for r in sub], "o-", label=f"{label} dB")
            axes[2].plot(x, [r["recon_snr_db"] for r in sub], "o-", label=f"{label} dB")
        for ax, t, yl in zip(axes, ("oracle channel BER", "T-PRIME accuracy", "reconstruction SNR"),
                             ("chBER", "accuracy", "dB")):
            ax.set_title(t); ax.set_xlabel("compression ratio vs cf32"); ax.set_ylabel(yl)
            ax.grid(alpha=0.3); ax.legend()
        fig.tight_layout()
        fig.savefig(THIS / "zfp_knee.png", dpi=130)
        print(f"[zfp] wrote {THIS / 'zfp_knee.png'}")
    except Exception as exc:  # plot is best-effort
        print(f"[zfp] plot skipped: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
