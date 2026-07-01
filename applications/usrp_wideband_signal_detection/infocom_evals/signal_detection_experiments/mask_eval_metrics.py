#!/usr/bin/env python3
"""Modular metrics layer for offline detector-mask evaluation.

Consumes the per-(detector, file) artifacts written by ``run_offline_cuda_detector_eval``
(``frame_manifest.csv`` + ``mask_arrays/*.npy`` + ``gt_masks/*.npy`` + ``gt_annotations/*.json``)
and produces two tidy "fact" tables that downstream notebooks roll up at read time:

  * ``frame_pixel_metrics``  — one row per (detector, file, frame): pixel TP/FP/FN,
    precision / recall / F1 / IoU, and false-positive area on empty spectrum.
  * ``region_metrics``       — one row per (detector, file, frame, annotation): box
    coverage, detection at a threshold, box IoU, plus the breakdown attributes
    (signal class, occupied bandwidth, pulse length, power_db, time_group,
    attenuation level) re-joined from the source ``.sigmf-meta``.

Design goals (see REPLAY_INGEST_PLAN.md / the eval plan):
  * GROUND TRUTH = the binary-side ``gt_masks/*.npy`` (rasterised on the exact
    detector FFT grid). The Python GT builder in ``signal_detection_eval.py`` is a
    different grid and is intentionally NOT used here.
  * Adding a breakdown dimension = one entry in ``BUCKETERS``.
  * Adding a metric = one more column in the fact tables (rollups are read-time).
  * Detector-agnostic: nothing here assumes DINO vs coherent-power; new detectors
    drop in for free.

Pure-Python + numpy; pandas is optional (used only for the tidy-table convenience
wrappers). No GPU, no Holoscan.
"""
from __future__ import annotations

import csv
import glob
import json
import math
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np

# --------------------------------------------------------------------------- #
# Attenuation level (power axis) parsed from the capture file stem
# --------------------------------------------------------------------------- #
_ATTEN_RE = re.compile(r"attenuation_dB_(\d+)(?:_v(\d+))?")


def parse_attenuation_db(stem: str) -> Optional[int]:
    """Return the attenuation level in dB parsed from a capture stem, or None."""
    match = _ATTEN_RE.search(stem)
    if not match:
        return None
    return int(match.group(1))


# --------------------------------------------------------------------------- #
# Loaders (self-contained; do not depend on the container or holoscan)
# --------------------------------------------------------------------------- #
def _find_artifact(run_dir: Path, subdir: str, prefix: str, channel: int, frame: int) -> str:
    """Relative path to a per-frame artifact, raw .npy or packed .npz, or ''."""
    d = Path(run_dir) / subdir
    for pat in (f"{prefix}_ch{channel}_f{frame}_*.npy", f"{prefix}_ch{channel}_f{frame}_*.packed.npz"):
        hits = sorted(d.glob(pat))
        if hits:
            return str(hits[0].relative_to(run_dir))
    return ""


def reconstruct_manifest(run_dir: Path) -> list[dict]:
    """Rebuild manifest rows from the surviving ``gt_annotations/*.json`` sidecars.

    Used when ``frame_manifest.csv`` is absent (e.g. the binary's coverage check
    rejected a run whose detector dropped a few tail frames, deleting the manifest
    but leaving the per-frame artifacts). The GT JSONs carry every offset/geometry
    field; artifact paths are recovered by globbing (raw .npy or packed .npz).
    """
    run_dir = Path(run_dir)
    gt_dir = run_dir / "gt_annotations"
    rows: list[dict] = []
    if not gt_dir.is_dir():
        return rows
    for jf in sorted(gt_dir.glob("*.json")):
        p = json.loads(jf.read_text())
        channel = int(p.get("channel", 0))
        frame = int(p.get("frame_number", 0))
        row = {
            "channel": channel,
            "frame_number": frame,
            "file_offset_complex": int(p.get("file_offset_complex", 0)),
            "data_end_complex": int(p.get("data_end_sample", 0)),
            "frame_end_complex": int(p.get("frame_end_sample", 0)),
            "complex_samples_read": int(p.get("complex_samples_read", 0)),
            "complex_samples_padded": int(p.get("complex_samples_padded", 0)),
            "partial_frame": bool(p.get("complex_samples_padded", 0)),
            "fft_rows": int(p.get("fft_rows", 0)),
            "fft_cols": int(p.get("fft_cols", 0)),
            "local_file_offset_complex": int(p.get("local_file_offset_complex", 0)),
            "global_sample_start": int(p.get("global_sample_start", 0)),
            "samples_per_row": int(p.get("samples_per_row", 0)),
            "gt_annotations_json": str(jf.relative_to(run_dir)),
            "gt_mask_npy": _find_artifact(run_dir, "gt_masks", "ground_truth_mask", channel, frame),
            "mask_npy": _find_artifact(run_dir, "mask_arrays", "mask", channel, frame),
        }
        rows.append(row)
    rows.sort(key=lambda r: r["frame_number"])
    return rows


def load_manifest(run_dir: Path) -> list[dict]:
    """Parse ``frame_manifest.csv``; if absent, reconstruct from gt_annotations JSONs."""
    manifest_path = Path(run_dir) / "frame_manifest.csv"
    if not manifest_path.exists():
        return reconstruct_manifest(run_dir)
    rows: list[dict] = []
    with open(manifest_path, newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            row = dict(raw)
            for int_key in (
                "channel",
                "frame_number",
                "file_offset_complex",
                "data_end_complex",
                "frame_end_complex",
                "complex_samples_read",
                "complex_samples_padded",
                "fft_rows",
                "fft_cols",
                "preview_rows",
                "preview_cols",
                "global_sample_start",
                "global_data_end_sample",
                "global_frame_end_sample",
                "samples_per_row",
            ):
                if int_key in row and row[int_key] != "":
                    row[int_key] = int(row[int_key])
            row["partial_frame"] = str(row.get("partial_frame", "false")).lower() == "true"
            rows.append(row)
    return rows


def load_summary(run_dir: Path) -> dict:
    summary_path = Path(run_dir) / "offline_eval_summary.json"
    if not summary_path.exists():
        return {}
    return json.loads(summary_path.read_text())


def _load_npy_u8(path: Path) -> np.ndarray:
    arr = np.load(path)
    return (arr != 0).astype(np.uint8)


def _load_packed_npz(path: Path) -> np.ndarray:
    data = np.load(path)
    rows = int(data["rows"])
    cols = int(data["cols"])
    flat = np.unpackbits(data["packed"])[: rows * cols]
    return flat.reshape(rows, cols).astype(np.uint8)


def load_mask_any(path: Path) -> Optional[np.ndarray]:
    """Load a binary mask from a raw ``.npy`` or a packbits ``.packed.npz``.

    Accepts either path form directly, and falls back from a ``.npy`` logical path
    to its ``.packed.npz`` sibling (the batch orchestrator can repack masks to save
    disk), so evaluation/visualisation works regardless of on-disk form.
    """
    path = Path(path)
    if path.name.endswith(".packed.npz"):
        return _load_packed_npz(path) if path.exists() else None
    if path.exists():
        return _load_npy_u8(path)
    packed = path.with_suffix(".packed.npz")
    if packed.exists():
        return _load_packed_npz(packed)
    return None


def resize_mask_nearest(mask: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Nearest-neighbour resize of a 2D binary mask to (rows, cols).

    Detector masks and GT masks can have different native grids (e.g. DINO 1024x1024
    vs the FFT-grid GT 512x10240). We resample the detector mask onto the GT grid so
    pixel metrics are computed on a common geometry. Nearest-neighbour preserves the
    binary nature; any-pooling would inflate small detections.
    """
    src_rows, src_cols = mask.shape
    if (src_rows, src_cols) == (rows, cols):
        return mask
    row_idx = (np.arange(rows) * src_rows // max(1, rows)).clip(0, src_rows - 1)
    col_idx = (np.arange(cols) * src_cols // max(1, cols)).clip(0, src_cols - 1)
    return mask[np.ix_(row_idx, col_idx)]


# --------------------------------------------------------------------------- #
# Source-meta attribute rejoin
# --------------------------------------------------------------------------- #
@dataclass
class SourceAnnotation:
    sample_start: int
    sample_count: int
    freq_lower_hz: float
    freq_upper_hz: float
    label: str
    wfgt_class: str
    kind: str
    occupied_bw_hz: Optional[float]
    power_db: Optional[float]
    length_samples: Optional[int]
    time_group: Optional[int]


def load_source_annotations(sigmf_meta_path: Path) -> list[SourceAnnotation]:
    """Load full annotation attributes from the source ``.sigmf-meta``."""
    meta = json.loads(Path(sigmf_meta_path).read_text())
    out: list[SourceAnnotation] = []
    for ann in meta.get("annotations", []):
        out.append(
            SourceAnnotation(
                sample_start=int(ann.get("core:sample_start", 0)),
                sample_count=int(ann.get("core:sample_count", 0)),
                freq_lower_hz=float(ann.get("core:freq_lower_edge", 0.0)),
                freq_upper_hz=float(ann.get("core:freq_upper_edge", 0.0)),
                label=str(ann.get("core:label", "UNLABELED")),
                wfgt_class=str(ann.get("wfgt:class", ann.get("core:label", "UNLABELED"))),
                kind=str(ann.get("wfgt:kind", "annotation")),
                occupied_bw_hz=_opt_float(ann.get("wfgt:occupied_bw_hz")),
                power_db=_opt_float(ann.get("wfgt:power_db")),
                length_samples=_opt_int(ann.get("wfgt:length_samples")),
                time_group=_opt_int(ann.get("wfgt:time_group")),
            )
        )
    return out


def _opt_float(value) -> Optional[float]:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _opt_int(value) -> Optional[int]:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _freq_overlap(lo_a: float, hi_a: float, lo_b: float, hi_b: float) -> float:
    return max(0.0, min(hi_a, hi_b) - max(lo_a, lo_b))


def match_source_annotation(
    item: dict, source: list[SourceAnnotation]
) -> Optional[SourceAnnotation]:
    """Match a binary GT item back to its full-attribute source annotation.

    The binary's annotation parser drops wfgt:* attributes and clips the frequency
    edges to baseband, so we match on (sample_start, sample_count) and then pick the
    source annotation whose frequency window best overlaps the (clipped) item window.
    """
    s_start = int(item.get("sample_start", 0))
    s_count = int(item.get("sample_count", 0))
    lo = float(item.get("freq_lower_hz", 0.0))
    hi = float(item.get("freq_upper_hz", 0.0))
    candidates = [a for a in source if a.sample_start == s_start and a.sample_count == s_count]
    if not candidates:
        candidates = [a for a in source if a.sample_start == s_start]
    if not candidates:
        return None
    best = None
    best_overlap = -1.0
    for cand in candidates:
        ov = _freq_overlap(lo, hi, cand.freq_lower_hz, cand.freq_upper_hz)
        # tie-break toward matching label
        score = ov + (1.0 if cand.label == item.get("label") else 0.0)
        if score > best_overlap:
            best_overlap = score
            best = cand
    return best


# --------------------------------------------------------------------------- #
# Breakdown bucketers — add a dimension by adding one entry here
# --------------------------------------------------------------------------- #
def bucket_bandwidth(attrs: dict) -> str:
    bw = attrs.get("occupied_bw_hz")
    if bw is None:
        return "unknown"
    bw_mhz = bw / 1e6
    edges = [(0, 2, "<2MHz"), (2, 10, "2-10MHz"), (10, 25, "10-25MHz"),
             (25, 60, "25-60MHz"), (60, 1e9, ">=60MHz")]
    for lo, hi, name in edges:
        if lo <= bw_mhz < hi:
            return name
    return "unknown"


def bucket_length(attrs: dict) -> str:
    length = attrs.get("length_samples")
    if length is None:
        length = attrs.get("sample_count")
    if length is None:
        return "unknown"
    edges = [(0, 1e4, "<10k"), (1e4, 1e5, "10k-100k"), (1e5, 1e6, "100k-1M"),
             (1e6, 5e6, "1M-5M"), (5e6, 1e12, ">=5M")]
    for lo, hi, name in edges:
        if lo <= length < hi:
            return name
    return "unknown"


def bucket_power_db(attrs: dict) -> str:
    power = attrs.get("power_db")
    if power is None:
        return "unknown"
    return f"{int(round(power))}dB"


# Registry: dimension name -> function(attrs) -> bucket label.
# `attrs` is the merged region attribute dict (source annotation + file-level fields).
BUCKETERS: dict[str, Callable[[dict], str]] = {
    "signal_class": lambda a: str(a.get("wfgt_class") or a.get("label") or "unknown"),
    "bandwidth": bucket_bandwidth,
    "pulse_length": bucket_length,
    "power_db": bucket_power_db,
    "attenuation_db": lambda a: ("unknown" if a.get("attenuation_db") is None
                                 else f"{a['attenuation_db']}dB"),
    "time_group": lambda a: ("unknown" if a.get("time_group") is None
                             else str(a["time_group"])),
}


# --------------------------------------------------------------------------- #
# Metric primitives
# --------------------------------------------------------------------------- #
@dataclass
class PixelMetrics:
    tp: int
    fp: int
    fn: int
    gt_pixels: int
    pred_pixels: int
    total_pixels: int

    @property
    def precision(self) -> float:
        return self.tp / self.pred_pixels if self.pred_pixels else math.nan

    @property
    def recall(self) -> float:
        return self.tp / self.gt_pixels if self.gt_pixels else math.nan

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        if math.isnan(p) or math.isnan(r) or (p + r) == 0:
            return math.nan
        return 2 * p * r / (p + r)

    @property
    def iou(self) -> float:
        union = self.tp + self.fp + self.fn
        return self.tp / union if union else math.nan

    @property
    def fp_area_fraction(self) -> float:
        """Fraction of empty (non-GT) spectrum that the detector falsely flagged."""
        empty = self.total_pixels - self.gt_pixels
        return self.fp / empty if empty else math.nan


def pixel_metrics(pred: np.ndarray, gt: np.ndarray) -> PixelMetrics:
    """Pixel-level confusion between a binary detector mask and the GT mask.

    The detector mask is resampled to the GT grid first.
    """
    pred_r = resize_mask_nearest(pred, gt.shape[0], gt.shape[1]).astype(bool)
    gt_b = gt.astype(bool)
    tp = int((pred_r & gt_b).sum())
    fp = int((pred_r & ~gt_b).sum())
    fn = int((~pred_r & gt_b).sum())
    return PixelMetrics(
        tp=tp, fp=fp, fn=fn,
        gt_pixels=int(gt_b.sum()),
        pred_pixels=int(pred_r.sum()),
        total_pixels=int(gt_b.size),
    )


@dataclass
class RegionResult:
    box_pixels: int
    covered_pixels: int
    coverage: float
    box_iou: float


def region_coverage(pred_on_gtgrid: np.ndarray, item: dict, gt_rows: int, gt_cols: int) -> RegionResult:
    """Coverage of a single GT annotation box by the detector mask.

    Uses the box's row/col rectangle (already on the FFT/GT grid) from the binary's
    ``gt_annotations`` JSON, so no re-rasterisation is needed.
    """
    r0 = max(0, int(item.get("row_start", 0)))
    r1 = min(gt_rows, int(item.get("row_stop", 0)))
    c0 = max(0, int(item.get("col_start", 0)))
    c1 = min(gt_cols, int(item.get("col_stop", 0)))
    if r1 <= r0 or c1 <= c0:
        return RegionResult(0, 0, math.nan, math.nan)
    sub = pred_on_gtgrid[r0:r1, c0:c1].astype(bool)
    box_pixels = int(sub.size)
    covered = int(sub.sum())
    coverage = covered / box_pixels if box_pixels else math.nan
    # box_iou: detector-on within box vs the full box (box is all-GT here)
    box_iou = covered / box_pixels if box_pixels else math.nan
    return RegionResult(box_pixels, covered, coverage, box_iou)


# --------------------------------------------------------------------------- #
# Per-run evaluation
# --------------------------------------------------------------------------- #
@dataclass
class EvalConfig:
    region_coverage_threshold: float = 0.5  # box "detected" if coverage >= threshold


def _artifact_path(run_dir: Path, rel: str) -> Path:
    return Path(run_dir) / rel


def evaluate_run(
    run_dir: Path,
    detector: str,
    file_stem: str,
    sigmf_meta_path: Optional[Path] = None,
    config: EvalConfig = EvalConfig(),
    frame_limit: Optional[int] = None,
) -> tuple[list[dict], list[dict]]:
    """Evaluate one (detector, file) run directory.

    Returns ``(frame_pixel_rows, region_rows)`` — lists of flat dicts ready for a
    DataFrame. ``sigmf_meta_path`` supplies the breakdown attributes; if omitted, the
    region rows fall back to the (attribute-poor) binary GT items.
    """
    run_dir = Path(run_dir)
    manifest = load_manifest(run_dir)
    attenuation_db = parse_attenuation_db(file_stem)
    source_annotations = (
        load_source_annotations(sigmf_meta_path) if sigmf_meta_path and Path(sigmf_meta_path).exists() else []
    )

    frame_rows: list[dict] = []
    region_rows: list[dict] = []

    for record in manifest:
        if frame_limit is not None and record["frame_number"] > frame_limit:
            continue
        gt_rel = record.get("gt_mask_npy")
        if not gt_rel:
            continue
        gt = load_mask_any(_artifact_path(run_dir, gt_rel))
        if gt is None:
            continue
        mask_rel = record.get("mask_npy")
        pred = load_mask_any(_artifact_path(run_dir, mask_rel)) if mask_rel else None

        if pred is None:
            # Detector emitted no mask for this frame (e.g. a tail-frame pipeline-drain
            # artifact). Record it for visibility but leave metrics NaN so aggregates
            # exclude it, rather than silently scoring it as a miss (which would bias the
            # detector comparison). Region rows are skipped for the same reason.
            frame_rows.append({
                "detector": detector, "file_stem": file_stem,
                "attenuation_db": attenuation_db, "frame_number": record["frame_number"],
                "fft_rows": record.get("fft_rows"), "fft_cols": record.get("fft_cols"),
                "tp": 0, "fp": 0, "fn": int((gt != 0).sum()),
                "gt_pixels": int((gt != 0).sum()), "pred_pixels": 0,
                "total_pixels": int(gt.size),
                "precision": math.nan, "recall": math.nan, "f1": math.nan,
                "iou": math.nan, "fp_area_fraction": math.nan,
                "mask_present": False,
            })
            continue

        pm = pixel_metrics(pred, gt)
        pred_on_gtgrid = resize_mask_nearest(pred, gt.shape[0], gt.shape[1])

        frame_rows.append({
            "detector": detector,
            "file_stem": file_stem,
            "attenuation_db": attenuation_db,
            "frame_number": record["frame_number"],
            "fft_rows": record.get("fft_rows"),
            "fft_cols": record.get("fft_cols"),
            "tp": pm.tp, "fp": pm.fp, "fn": pm.fn,
            "gt_pixels": pm.gt_pixels, "pred_pixels": pm.pred_pixels,
            "total_pixels": pm.total_pixels,
            "precision": pm.precision, "recall": pm.recall,
            "f1": pm.f1, "iou": pm.iou,
            "fp_area_fraction": pm.fp_area_fraction,
            "mask_present": True,
        })

        # region-level rows from the binary GT annotations JSON
        gt_ann_rel = record.get("gt_annotations_json")
        if not gt_ann_rel:
            continue
        gt_ann_path = _artifact_path(run_dir, gt_ann_rel)
        if not gt_ann_path.exists():
            continue
        payload = json.loads(gt_ann_path.read_text())
        for ann_index, item in enumerate(payload.get("items", [])):
            rr = region_coverage(pred_on_gtgrid, item, gt.shape[0], gt.shape[1])
            src = match_source_annotation(item, source_annotations)
            attrs = {
                "label": item.get("label"),
                "wfgt_class": (src.wfgt_class if src else item.get("label")),
                "occupied_bw_hz": (src.occupied_bw_hz if src else None),
                "power_db": (src.power_db if src else None),
                "length_samples": (src.length_samples if src else None),
                "sample_count": item.get("sample_count"),
                "time_group": (src.time_group if src else None),
                "attenuation_db": attenuation_db,
            }
            row = {
                "detector": detector,
                "file_stem": file_stem,
                "frame_number": record["frame_number"],
                "annotation_index": ann_index,
                "kind": item.get("kind"),
                "box_pixels": rr.box_pixels,
                "covered_pixels": rr.covered_pixels,
                "coverage": rr.coverage,
                "box_iou": rr.box_iou,
                "detected": (not math.isnan(rr.coverage)) and rr.coverage >= config.region_coverage_threshold,
                "matched_source": src is not None,
                **attrs,
            }
            # precompute breakdown bucket labels so notebooks can group directly
            for dim, fn in BUCKETERS.items():
                row[f"bucket_{dim}"] = fn(row)
            region_rows.append(row)

    return frame_rows, region_rows


# --------------------------------------------------------------------------- #
# Convenience: build tidy DataFrames / write Parquet
# --------------------------------------------------------------------------- #
def write_rows_csv(rows: list[dict], path: Path) -> None:
    """Write a list of flat dicts to CSV (stdlib only, no pandas).

    Column union is taken across all rows so heterogeneous rows still serialise.
    This is the durable fact-table format; Parquet is written too when pandas is
    available (see ``write_tables``).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_tables(frame_rows: list[dict], region_rows: list[dict], out_dir: Path) -> dict:
    """Persist both fact tables as CSV (always) and Parquet (if pandas available)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "frame_pixel_metrics_csv": out_dir / "frame_pixel_metrics.csv",
        "region_metrics_csv": out_dir / "region_metrics.csv",
    }
    write_rows_csv(frame_rows, paths["frame_pixel_metrics_csv"])
    write_rows_csv(region_rows, paths["region_metrics_csv"])
    try:
        import pandas as pd

        pd.DataFrame(frame_rows).to_parquet(out_dir / "frame_pixel_metrics.parquet")
        pd.DataFrame(region_rows).to_parquet(out_dir / "region_metrics.parquet")
        paths["frame_pixel_metrics_parquet"] = out_dir / "frame_pixel_metrics.parquet"
        paths["region_metrics_parquet"] = out_dir / "region_metrics.parquet"
    except Exception:  # pandas/pyarrow not available — CSV is the source of truth
        pass
    return paths


def to_frames(frame_rows: list[dict], region_rows: list[dict]):
    import pandas as pd  # local import so the module is usable without pandas

    return pd.DataFrame(frame_rows), pd.DataFrame(region_rows)


def detection_rate_by(region_df, dimension: str):
    """Read-time rollup: detection rate per bucket of ``dimension``.

    Example: ``detection_rate_by(region_df, "signal_class")``.
    """
    col = f"bucket_{dimension}"
    grouped = region_df.groupby(["detector", col])
    out = grouped.agg(
        n_regions=("detected", "size"),
        n_detected=("detected", "sum"),
        mean_coverage=("coverage", "mean"),
        mean_box_iou=("box_iou", "mean"),
    ).reset_index()
    out["detection_rate"] = out["n_detected"] / out["n_regions"].clip(lower=1)
    return out
