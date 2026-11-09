#!/usr/bin/env python3
"""Detector masks from the TorchScript DINO-FT exports (M7-S, M12-S, M13-S, any dino_fine_tuning .ts), no container.

The released fine-tuned models are plain TorchScript: they need PyTorch and the Python twin of the operator's front
end (dino_fine_tuning/src/frontend.py + rates.py), not the Holoscan container. This module reproduces what the C++
finetuned_dino_detector emits for an offline file and writes it in the soft-label pipeline's mask contract, so
snip_annotations.py turns it into <stem>_softlabel.sigmf-meta exactly as for the container detectors:

  <run_dir>/mask_arrays/mask_ch0_f{N}_{rows}x{cols}.packed.npz      (frames numbered from 1, like the container)
  <run_dir>/frame_manifest.csv                                      (frame_number, global_sample_start, ...)
  <run_dir>/ts_model.json                                           (model, threshold, guard, front end, counts)

Matching the operator: the per-model live config (config_live_v3_dino_ft_<model>.yaml) supplies the model file,
threshold, front-end settings, invalid-frame guard and snipper gates; each frame is 512 rows x the app's FFT size for
the sample rate; the 1024-column model mask is widened to that FFT size with the operator's nearest-column index
(src = min(floor(dst * in/out), in-1) in float32); a frame denser than max(min_occupancy, k x running baseline) is
emitted empty. Masks are bit-identical to dino_fine_tuning/src/run_ts_on_sigmf.py before the widening.

Recordings that do not fill whole frames: a clip shorter than one frame is repeated cyclically to fill it, and a
recording's tail is covered by one last frame aligned to its end; mask rows that are not new real samples are zeroed,
so no detection is reported twice or outside the recording.

  python3 dino_ts_masks.py <capture.sigmf-meta> --detector dino_finetuned_m12s --run-dir /tmp/run   # standalone
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
APP_DIR = HERE.parent.parent.parent                       # .../applications/usrp_wideband_signal_detection
REPO_ROOT = APP_DIR.parent.parent
FT_SRC = REPO_ROOT / "dino_fine_tuning" / "src"
WEIGHTS_DIR = REPO_ROOT / "dino_fine_tuning" / "weights"
RELEASES = "https://github.com/Trudes808/Holohub-Signal-Detection/releases/download"
sys.path.insert(0, str(HERE))
import to_sigmf as tsg  # noqa: E402

# detector name -> the live config that defines it (model file, threshold, guard, snipper gates) + its release tag
TS_DETECTORS = {
    "dino_finetuned_m12s": {"config": "config_live_v3_dino_ft_m12s.yaml", "release": "v1.6.0",
                            "about": "M12-S: best recall of weak / low-SNR emissions"},
    "dino_finetuned_m13s": {"config": "config_live_v3_dino_ft_m13s.yaml", "release": "v1.6.0",
                            "about": "M13-S: fewest false positives, drops some weak flat carriers"},
    "dino_finetuned_m7s": {"config": "config_live_v3_dino_ft_m7s.yaml", "release": "v1.5.1",
                           "about": "M7-S: the live demo's deployed model"},
}
GENERIC = "dino_finetuned_ts"     # any .ts export via --ts-model; settings of the M12-S config (shared front end)
TRAINED_RATE_HZ = (5e6, 500e6)
CUDA_LITERAL = b'torch.device("cuda:0")'
CPU_LITERAL = b'torch.device("cpu")   '                  # same length: keeps the archive's debug source ranges valid


def is_ts_detector(name: str) -> bool:
    return name in TS_DETECTORS or name == GENERIC


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _download(url: str, dst: Path):
    tmp = dst.with_name(dst.name + ".part")
    print(f"  downloading {url}", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "holohub-soft-label-pipeline"})
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    tmp.rename(dst)


def cpu_variant(ts_path: Path) -> Path:
    """The released exports were traced on a GPU, which baked device="cuda:0" into the RoPE position-encoding
    arange() calls. For CPU-only machines write <stem>.cpu.ts with those literals pointed at the CPU."""
    with zipfile.ZipFile(ts_path) as z:
        if not any(CUDA_LITERAL in z.read(n) for n in z.namelist() if n.endswith(".py")):
            return ts_path
        dst = ts_path.with_name(ts_path.name[: -len(".ts")] + ".cpu.ts")
        if dst.is_file() and dst.stat().st_mtime >= ts_path.stat().st_mtime:
            return dst
        tmp = dst.with_name(dst.name + ".part")
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED) as out:
            for info in z.infolist():
                data = z.read(info)
                if info.filename.endswith(".py"):
                    data = data.replace(CUDA_LITERAL, CPU_LITERAL)
                out.writestr(info, data)
    tmp.rename(dst)
    return dst


@dataclass
class TsSettings:
    name: str
    config: Path
    ts_path: Path
    threshold: float
    guard: bool
    guard_min_occ: float
    guard_k: float
    guard_alpha: float
    downsample_fft_size: int
    frontend: dict                  # FrontEndCfg overrides from the config
    snipper: dict                   # min_box_pixels, merge_gap_rows, merge_gap_cols


def load_settings(detector: str, ts_model: str | None = None, threshold: float | None = None,
                  offline: bool = False) -> TsSettings:
    """Read the detector's live config; fetch the release weights into dino_fine_tuning/weights/ if missing."""
    import yaml
    entry = TS_DETECTORS.get(detector) or TS_DETECTORS["dino_finetuned_m12s"]
    if detector == GENERIC and not ts_model:
        raise SystemExit(f"--detector {GENERIC} needs --ts-model <file>.ts")
    cfg_path = APP_DIR / entry["config"]
    cfg = yaml.safe_load(cfg_path.read_text())
    d = cfg.get("finetuned_dino_detector") or {}
    sn = cfg.get("signal_snipper") or {}
    if ts_model:
        ts = Path(ts_model).expanduser().resolve()
        if not ts.is_file():
            raise SystemExit(f"--ts-model {ts}: no such file")
    else:
        ts = Path(str(d["model_script_path"]).replace("/workspace/holohub", str(REPO_ROOT)))
        if not ts.is_file():
            if offline:
                raise SystemExit(f"{ts.name} missing (dino_fine_tuning/README.md 'Weights'); --offline set")
            WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
            base = f"{RELEASES}/{entry['release']}/{ts.name[: -len('.ts')]}"
            _download(base + ".meta.json", WEIGHTS_DIR / (ts.name[: -len(".ts")] + ".meta.json"))
            _download(base + ".ts", ts)
    thr = threshold
    if thr is None:
        meta_p = ts.with_name(ts.name[: -len(".ts")] + ".meta.json")
        thr = float(d.get("threshold", 0.6)) if not ts_model else \
            float(json.loads(meta_p.read_text()).get("threshold", 0.6)) if meta_p.is_file() else 0.6
    fe_map = {"nfft": "nfft", "tile_rows": "tile_rows", "fft_window": "fft_window", "db_vmin": "db_vmin",
              "db_vmax": "db_vmax", "flatten_noise_floor": "flatten", "adaptive_normalization": "adaptive",
              "adaptive_span_db": "adaptive_span_db", "adaptive_floor_frac": "adaptive_floor_frac",
              "adaptive_robust_floor": "adaptive_robust_floor"}
    frontend = {fe_map[k]: v for k, v in d.items() if k in fe_map}
    for k in ("short_fft_size", "short_fft", "time_pool_rows"):
        if d.get(k):
            raise SystemExit(f"{cfg_path.name}: '{k}' (short-FFT front end) is not reproduced here")
    if not d.get("real_time_downsample", True):
        raise SystemExit(f"{cfg_path.name}: real_time_downsample=false (native path) is not reproduced here")
    return TsSettings(
        name=detector, config=cfg_path, ts_path=ts, threshold=float(thr),
        guard=bool(d.get("invalid_frame_guard", True)),
        guard_min_occ=float(d.get("invalid_frame_min_occupancy", 0.03)),
        guard_k=float(d.get("invalid_frame_occupancy_k", 6.0)),
        guard_alpha=float(d.get("invalid_frame_baseline_alpha", 0.05)),
        downsample_fft_size=int(d.get("downsample_fft_size", 0) or 0),
        frontend=frontend,
        snipper={k: sn[k] for k in ("min_box_pixels", "merge_gap_rows", "merge_gap_cols") if k in sn})


class TsDetector:
    def __init__(self, s: TsSettings, device: str | None = None):
        import torch
        self.torch, self.s = torch, s
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.R = _load_module("holohub_ft_rates", FT_SRC / "rates.py")
        self.fe = _load_module("holohub_ft_frontend", FT_SRC / "frontend.py")
        self.cfg = self.fe.FrontEndCfg(**s.frontend)
        ts = cpu_variant(s.ts_path) if self.device == "cpu" else s.ts_path
        self.model = torch.jit.load(str(ts), map_location=self.device).eval()
        self.frontend_sha = hashlib.sha256((FT_SRC / "frontend.py").read_bytes()
                                           + (FT_SRC / "rates.py").read_bytes()).hexdigest()[:12]

    def fft_size(self, rate: float) -> int:
        return self.s.downsample_fft_size or self.R.app_fft_size(rate)

    def compact_mask(self, iq: np.ndarray, rate: float, F: int) -> np.ndarray:
        torch = self.torch
        with torch.no_grad():
            o = self.fe.compute_front_end(torch.from_numpy(iq).to(self.device).unsqueeze(0), rate, self.cfg, fft_size=F)
            prob = torch.sigmoid(self.model(o.tiles.to(self.device).float()).float())
        return (prob >= self.s.threshold).to(torch.uint8).cpu().numpy().reshape(-1, self.cfg.nfft)[: self.R.ROWS_WIDE]

    @staticmethod
    def widen_index(ncols: int, wide: int) -> np.ndarray:
        """The operator's nearest-column index: src = min(floor(dst * (in/out)), in-1), float32 math."""
        scale = np.float32(ncols) / np.float32(wide)
        return np.minimum(np.floor(np.arange(wide, dtype=np.float32) * scale).astype(np.int64), ncols - 1)

    @staticmethod
    def plan(s0: int, L: int, F: int, FS: int, rows: int) -> list[tuple[int, int, int, bool]]:
        """(frame start, first valid row, end valid row, wrapped) covering samples [s0, s0+L) of one segment."""
        if L < F:
            return []
        if L < FS:                                       # short clip: repeat it cyclically to one frame
            return [(s0, 0, L // F, True)]
        n_full = L // FS
        out = [(s0 + k * FS, 0, rows, False) for k in range(n_full)]
        if L - n_full * FS >= F:                         # tail: one frame aligned to the end, keep only new rows
            start = s0 + L - FS
            out.append((start, -(-(s0 + n_full * FS - start) // F), rows, False))
        return out

    def run(self, meta_path: Path, run_dir: Path, max_seconds: float | None = None, force: bool = False) -> dict:
        info_p = run_dir / "ts_model.json"
        key = {"model": str(self.s.ts_path), "threshold": self.s.threshold, "max_seconds": max_seconds,
               "frame_guard": self.s.guard}
        if not force and info_p.is_file() and (run_dir / "frame_manifest.csv").is_file():
            prev = json.loads(info_p.read_text())
            if all(prev.get(k) == v for k, v in key.items()):
                prev["skipped"] = True
                return prev
        t0 = time.time()
        meta = json.loads(meta_path.read_text())
        g = meta["global"]
        rate = float(g["core:sample_rate"])
        dtype = g.get("core:datatype", "cf32_le")
        data = meta_path.with_name(meta_path.name[: -len(".sigmf-meta")] + ".sigmf-data")
        n_total = data.stat().st_size // tsg.bytes_per_sample(dtype)
        caps = meta.get("captures") or [{"core:sample_start": 0}]
        F = self.fft_size(rate)
        rows = self.R.ROWS_WIDE
        FS = rows * F
        nfft = self.cfg.nfft
        widx = self.widen_index(nfft, F)
        budget = int(max_seconds * rate) if max_seconds else None
        mask_dir = run_dir / "mask_arrays"
        if mask_dir.exists():
            shutil.rmtree(mask_dir)
        mask_dir.mkdir(parents=True)
        # On unified-memory machines (DGX Spark / GB10) a page cache full of recording data makes CUDA allocations
        # fail although the memory is reclaimable, so drop what has been read (Linux; a no-op elsewhere).
        bps = tsg.bytes_per_sample(dtype)
        cache_fd = os.open(data.resolve(), os.O_RDONLY) if hasattr(os, "posix_fadvise") else None
        manifest, base = [], -1.0
        n_frame = suppressed = short = wrapped = used = 0
        for i, c in enumerate(caps):
            s0 = int(c.get("core:sample_start", 0))
            s1 = int(caps[i + 1]["core:sample_start"]) if i + 1 < len(caps) else n_total
            L = s1 - s0
            if budget is not None:
                if used >= budget:
                    break
                L = min(L, budget - used)
            used += L
            pl = self.plan(s0, L, F, FS, rows)
            if not pl:
                short += 1
                continue
            for start, r_lo, r_hi, wrap in pl:
                n_read = L if wrap else FS
                iq = tsg.read_raw(data, dtype, start, n_read)
                if cache_fd is not None:
                    os.posix_fadvise(cache_fd, start * bps, n_read * bps, os.POSIX_FADV_DONTNEED)
                if wrap:
                    iq = np.resize(iq, FS)
                    wrapped += 1
                m = self.compact_mask(iq, rate, F)
                occ = float(m.mean())
                bad = False
                if self.s.guard:
                    thresh = max(self.s.guard_min_occ, self.s.guard_k * base if base >= 0 else self.s.guard_min_occ)
                    if occ > thresh:
                        bad = True
                        suppressed += 1
                    else:
                        base = occ if base < 0 else (1 - self.s.guard_alpha) * base + self.s.guard_alpha * occ
                wide = np.zeros((rows, F), dtype=np.uint8) if bad else m[:, widx]
                wide[:r_lo] = 0
                wide[r_hi:] = 0
                n_frame += 1
                name = f"mask_ch0_f{n_frame}_{rows}x{F}.packed.npz"
                np.savez_compressed(mask_dir / name, packed=np.packbits(wide.ravel()), rows=rows, cols=F)
                manifest.append({
                    "channel": 0, "frame_number": n_frame, "file_offset_complex": start, "complex_samples_read": FS,
                    "complex_samples_padded": FS - n_read if wrap else 0, "partial_frame": str(wrap or r_lo > 0).lower(),
                    "fft_rows": rows, "fft_cols": F, "mask_npy": f"mask_arrays/{name}", "global_sample_start": start,
                    "global_frame_end_sample": start + FS, "samples_per_row": F, "valid_row_start": r_lo,
                    "valid_row_end": r_hi, "occupancy": round(occ, 6), "guard_suppressed": str(bad).lower()})
        if cache_fd is not None:
            os.close(cache_fd)
        with open(run_dir / "frame_manifest.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(manifest[0].keys()) if manifest else ["frame_number"])
            w.writeheader()
            w.writerows(manifest)
        notes = []
        if not (TRAINED_RATE_HZ[0] <= rate <= TRAINED_RATE_HZ[1]):
            notes.append(f"sample rate {rate / 1e6:.4g} MS/s is outside the models' training range "
                         f"({TRAINED_RATE_HZ[0] / 1e6:.0f}-{TRAINED_RATE_HZ[1] / 1e6:.0f} MS/s): detections are less reliable")
        if short:
            notes.append(f"{short} segment(s) shorter than one FFT row ({F} samples) skipped")
        if wrapped:
            notes.append(f"{wrapped} clip(s) shorter than a frame ({FS} samples) repeated cyclically to fill one; "
                         "only rows of real samples are annotated")
        if suppressed:
            notes.append(f"{suppressed} frame(s) emptied by the invalid-frame guard (occupancy above "
                         f"max({self.s.guard_min_occ}, {self.s.guard_k} x baseline))")
        info = dict(key, detector=self.s.name, config=self.s.config.name, device=self.device,
                    front_end=f"dino_fine_tuning/src frontend+rates sha256 {self.frontend_sha}",
                    capture=str(meta_path), rate_hz=rate, fft_size=F, frame_samples=FS, frames=n_frame,
                    guard_suppressed=suppressed, samples_used=used, seconds=round(time.time() - t0, 2), notes=notes)
        info_p.write_text(json.dumps(info, indent=1) + "\n")
        return info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("capture", help="<capture>.sigmf-meta (any SigMF sample format)")
    ap.add_argument("--detector", default="dino_finetuned_m12s", choices=list(TS_DETECTORS) + [GENERIC])
    ap.add_argument("--ts-model", default=None, help="a dino_fine_tuning .ts export (overrides the config's model)")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    det = TsDetector(load_settings(a.detector, a.ts_model, a.threshold), a.device)
    run_dir = Path(a.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    info = det.run(Path(a.capture), run_dir, a.max_seconds, a.force)
    print(json.dumps(info, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
