#!/usr/bin/env python3
"""Visualization helpers for the offline batch detector evaluation.

Ingests the saved per-(detector, file) artifacts produced by
``run_offline_cuda_detector_eval`` / ``run_batch_offline_eval.py`` and the SigMF
ground truth, and renders comparison panels. The flagship view is an N+1 panel
figure for one frame: panel 0 = spectrogram + ground-truth boxes, panels 1..N =
the same spectrogram with each detector's mask overlaid.

Because the batch sweep runs masks-only (``--no-tensors``), the spectrogram
background is reconstructed from the source SigMF (faithfully matching the binary's
FFT grid: ``fft_rows`` time bins x ``fft_cols`` frequency bins). When a saved
spectrogram tensor is present (non-``--no-tensors`` runs) it is used directly.

Depends only on numpy + matplotlib + :mod:`mask_eval_metrics`.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

import mask_eval_metrics as mem

DEFAULT_CAPTURE_DIRS = [
    Path(__file__).resolve().parents[2] / "generated_inputs",
    Path("/home/bqn82/captures"),
]
_BYTES_PER_COMPLEX = {"cf32_le": 8, "ci16_le": 4}
_NP_DTYPE = {"cf32_le": "<c8", "ci16_le": "<i2"}


# --------------------------------------------------------------------------- #
# Source capture lookup + IQ reading
# --------------------------------------------------------------------------- #
def find_capture_data(file_stem: str, search_dirs: Optional[list[Path]] = None) -> Optional[Path]:
    """Locate the source ``.sigmf-data`` for a capture stem (exact then prefix)."""
    search_dirs = search_dirs or DEFAULT_CAPTURE_DIRS
    candidates = [file_stem]
    match = re.match(r"(attenuation_dB_\d+(?:_v\d+)?)", file_stem)
    if match and match.group(1) != file_stem:
        candidates.append(match.group(1))
    for directory in [Path(d) for d in search_dirs if Path(d).exists()]:
        for cand in candidates:
            data = directory / f"{cand}.sigmf-data"
            if data.exists():
                return data
    return None


def read_frame_iq(data_path: Path, local_offset_complex: int, n_complex: int,
                  datatype: str = "cf32_le") -> np.ndarray:
    """Read ``n_complex`` complex samples starting at ``local_offset_complex``.

    Returns complex64. Integer SigMF types are decoded and scaled to ~[-1, 1].
    Uses a memmap so only the requested slice is touched (files are ~14 GB).
    """
    if datatype == "cf32_le":
        mm = np.memmap(data_path, dtype=np.complex64, mode="r")
        return np.asarray(mm[local_offset_complex: local_offset_complex + n_complex], dtype=np.complex64)
    if datatype == "ci16_le":
        mm = np.memmap(data_path, dtype=np.int16, mode="r")
        start = local_offset_complex * 2
        flat = np.asarray(mm[start: start + n_complex * 2], dtype=np.float32) / 32767.0
        return (flat[0::2] + 1j * flat[1::2]).astype(np.complex64)
    raise ValueError(f"unsupported datatype for viz: {datatype}")


def spectrogram_db_from_iq(iq: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Reconstruct the binary's spectrogram grid: per-row fftshift(fft) magnitude in dB.

    Row r is the ``cols``-point FFT of samples [r*cols : (r+1)*cols], matching the
    FFT operator (num_bursts=rows, transform_points=cols, fftshifted to f1=-N/2).
    """
    usable = rows * cols
    if iq.size < usable:
        iq = np.concatenate([iq, np.zeros(usable - iq.size, dtype=np.complex64)])
    block = iq[:usable].reshape(rows, cols)
    spectrum = np.fft.fftshift(np.fft.fft(block, axis=1), axes=1)
    power = np.abs(spectrum) ** 2 + 1e-12
    return (10.0 * np.log10(power)).astype(np.float32)


# --------------------------------------------------------------------------- #
# Frame bundle: spectrogram + GT + per-detector masks for one frame
# --------------------------------------------------------------------------- #
@dataclass
class FrameBundle:
    file_stem: str
    frame_number: int
    fft_rows: int
    fft_cols: int
    span_hz: float
    sample_rate_hz: float
    center_frequency_hz: float
    samples_per_row: int
    spectrogram_db: np.ndarray
    gt_mask: np.ndarray
    gt_items: list[dict]
    detector_masks: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def freq_extent_mhz(self) -> tuple[float, float]:
        half = 0.5 * self.span_hz / 1e6
        return (-half, half)

    @property
    def time_extent_ms(self) -> tuple[float, float]:
        total_ms = (self.fft_rows * self.samples_per_row) / self.sample_rate_hz * 1e3
        return (0.0, total_ms)

    @property
    def imshow_extent(self) -> list[float]:
        f0, f1 = self.freq_extent_mhz
        t0, t1 = self.time_extent_ms
        return [f0, f1, t1, t0]  # origin upper: time increases downward


def resolve_layout(path: Path, file_stem: Optional[str] = None) -> dict:
    """Resolve ``{batch_root, file_stem, detectors}`` from a path at any of the three
    natural depths, located by probing for ``frame_manifest.csv``:

    * **run dir**      ``.../<detector>/<file_stem>/``  (has frame_manifest.csv)
    * **detector dir** ``.../<detector>/``              (children are <file_stem> runs)
    * **batch root**   ``.../`` containing ``<detector>/<file_stem>/``

    ``batch_root`` in the returned dict is always the proper batch root
    (``<root>/<detector>/<file_stem>/``) that :func:`load_frame_bundle` expects.
    """
    path = Path(path)

    def is_run_dir(p: Path) -> bool:
        # a completed run has a manifest; a coverage-rejected run still has gt_annotations
        return (p / "frame_manifest.csv").exists() or (p / "gt_annotations").is_dir()

    # 1) single run dir
    if is_run_dir(path):
        return {"batch_root": path.parent.parent, "file_stem": path.name,
                "detectors": [path.parent.name]}

    # 2) detector dir: <detector>/<file_stem>/
    run_children = [c for c in path.iterdir() if c.is_dir() and is_run_dir(c)]
    if run_children:
        stems = sorted(c.name for c in run_children)
        stem = file_stem if (file_stem in stems) else stems[0]
        return {"batch_root": path.parent, "file_stem": stem, "detectors": [path.name]}

    # 3) batch root: <detector>/<file_stem>/
    stems: set[str] = set()
    for det in path.iterdir():
        if not det.is_dir():
            continue
        for stem_dir in det.iterdir():
            if stem_dir.is_dir() and is_run_dir(stem_dir):
                stems.add(stem_dir.name)
    stem = file_stem if (file_stem in stems) else (sorted(stems)[0] if stems else None)
    detectors = sorted(
        d.name for d in path.iterdir()
        if d.is_dir() and stem and is_run_dir(d / stem)
    ) if stem else []
    return {"batch_root": path, "file_stem": stem, "detectors": detectors}


def load_frame_bundle_smart(path, frame_number, file_stem=None, detectors=None,
                            capture_dirs=None, gt_from=None) -> "FrameBundle":
    """``load_frame_bundle`` that tolerates being pointed at a batch root OR a single
    run dir. Resolves the layout, then delegates."""
    layout = resolve_layout(Path(path), file_stem)
    if not layout["file_stem"]:
        raise FileNotFoundError(f"No runs found under {path}")
    return load_frame_bundle(
        layout["batch_root"], layout["file_stem"], frame_number,
        detectors=detectors or layout["detectors"],
        capture_dirs=capture_dirs, gt_from=gt_from,
    )


def _manifest_row_for_frame(run_dir: Path, frame_number: int) -> Optional[dict]:
    for row in mem.load_manifest(run_dir):
        if int(row["frame_number"]) == frame_number:
            return row
    return None


def load_frame_bundle(
    batch_root: Path,
    file_stem: str,
    frame_number: int,
    detectors: Optional[list[str]] = None,
    capture_dirs: Optional[list[Path]] = None,
    gt_from: Optional[str] = None,
) -> FrameBundle:
    """Load everything needed to render one frame across detectors.

    ``batch_root`` is laid out ``<batch_root>/<detector>/<file_stem>/``. GT is
    identical across detectors (derived from the SigMF on the shared FFT grid), so
    it is loaded from ``gt_from`` (default: the first available detector).
    """
    batch_root = Path(batch_root)
    if detectors is None:
        detectors = sorted(p.name for p in batch_root.iterdir() if (p / file_stem).is_dir())
    if not detectors:
        raise FileNotFoundError(f"No detector runs for {file_stem} under {batch_root}")

    # GT + geometry from a reference detector run
    gt_detector = gt_from or detectors[0]
    ref_dir = batch_root / gt_detector / file_stem
    ref_row = _manifest_row_for_frame(ref_dir, frame_number)
    if ref_row is None:
        raise FileNotFoundError(f"frame {frame_number} not in {ref_dir}/frame_manifest.csv")

    gt_mask = mem.load_mask_any(ref_dir / ref_row["gt_mask_npy"])
    if gt_mask is None:
        raise FileNotFoundError(f"GT mask missing for frame {frame_number} in {ref_dir}")
    fft_rows = int(ref_row["fft_rows"])
    fft_cols = int(ref_row["fft_cols"])

    gt_payload = json.loads((ref_dir / ref_row["gt_annotations_json"]).read_text())
    span_hz = float(gt_payload.get("span_hz") or 0.0)
    sample_rate_hz = float(gt_payload.get("sample_rate_hz") or span_hz or 1.0)
    center = gt_payload.get("center_frequency_hz")
    center_hz = float(center) if center is not None else 0.0
    samples_per_row = int(ref_row.get("samples_per_row") or (gt_payload.get("samples_per_row") or 1))

    # spectrogram: prefer a saved tensor, else reconstruct from SigMF
    spectrogram_db = _load_or_reconstruct_spectrogram(
        ref_dir, ref_row, file_stem, fft_rows, fft_cols, capture_dirs)

    bundle = FrameBundle(
        file_stem=file_stem, frame_number=frame_number,
        fft_rows=fft_rows, fft_cols=fft_cols, span_hz=span_hz,
        sample_rate_hz=sample_rate_hz, center_frequency_hz=center_hz,
        samples_per_row=samples_per_row,
        spectrogram_db=spectrogram_db, gt_mask=gt_mask,
        gt_items=gt_payload.get("items", []),
    )

    for detector in detectors:
        run_dir = batch_root / detector / file_stem
        row = _manifest_row_for_frame(run_dir, frame_number)
        if row is None or not row.get("mask_npy"):
            continue
        mask = mem.load_mask_any(run_dir / row["mask_npy"])
        if mask is not None:
            bundle.detector_masks[detector] = mem.resize_mask_nearest(mask, fft_rows, fft_cols)
    return bundle


def _load_or_reconstruct_spectrogram(run_dir, row, file_stem, fft_rows, fft_cols, capture_dirs):
    tensor_rel = row.get("spectrogram_tensor_npy")
    if tensor_rel:
        tensor_path = run_dir / tensor_rel
        if tensor_path.exists():
            tensor = np.load(tensor_path)
            power = np.abs(tensor) ** 2 + 1e-12
            return (10.0 * np.log10(power)).astype(np.float32)
    # reconstruct from SigMF
    data_path = find_capture_data(file_stem, capture_dirs)
    if data_path is None:
        raise FileNotFoundError(
            f"No saved spectrogram tensor and no source .sigmf-data found for {file_stem}; "
            f"pass capture_dirs= or run without --no-tensors.")
    meta_path = data_path.with_name(data_path.name.replace(".sigmf-data", ".sigmf-meta"))
    datatype = "cf32_le"
    if meta_path.exists():
        datatype = json.loads(meta_path.read_text()).get("global", {}).get("core:datatype", "cf32_le")
    local_offset = int(row.get("local_file_offset_complex") or 0)
    n_complex = int(row.get("complex_samples_read") or (fft_rows * fft_cols))
    iq = read_frame_iq(data_path, local_offset, n_complex, datatype)
    return spectrogram_db_from_iq(iq, fft_rows, fft_cols)


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
# display labels for panels/plots (internal run-dir names stay as-is everywhere else).
# Keep in sync with plot_snr_results.DETECTOR_LABELS + plot_eval_results.DETECTOR_LABELS.
DETECTOR_LABELS = {"cuda_dino": "zero_shot_dino"}


def label_for(det) -> str:
    return DETECTOR_LABELS.get(det, det)


def _db_limits(spectrogram_db: np.ndarray, lo_q=5.0, hi_q=99.5) -> tuple[float, float]:
    finite = spectrogram_db[np.isfinite(spectrogram_db)]
    if finite.size == 0:
        return (-120.0, 0.0)
    return (float(np.percentile(finite, lo_q)), float(np.percentile(finite, hi_q)))


def plot_frame_panels(
    bundle: FrameBundle,
    detectors: Optional[list[str]] = None,
    db_limits: Optional[tuple[float, float]] = None,
    mask_color=(1.0, 0.2, 0.2),
    gt_color=(0.1, 1.0, 0.3),
    box_color=(1.0, 0.0, 0.0),
    figsize_per_panel=(6.0, 5.0),
    show_gt_boxes: bool = True,
):
    """Render [GT | detector_1 | ... | detector_N] panels for one frame.

    Panel 0: spectrogram + ground-truth boxes (and filled-box GT mask, faint).
    Panels 1..N: the same spectrogram with each detector's mask overlaid.
    Returns the matplotlib Figure.
    """
    import matplotlib.pyplot as plt
    from matplotlib import colors as mcolors
    from matplotlib.patches import Rectangle

    detectors = detectors or list(bundle.detector_masks.keys())
    n_panels = 1 + len(detectors)
    extent = bundle.imshow_extent
    vmin, vmax = db_limits or _db_limits(bundle.spectrogram_db)

    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(figsize_per_panel[0] * n_panels, figsize_per_panel[1]),
        squeeze=False,
        layout="constrained",
    )
    axes = axes[0]

    def draw_spectrogram(ax, title):
        ax.imshow(bundle.spectrogram_db, aspect="auto", extent=extent,
                  origin="upper", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("frequency (MHz, baseband)")
        ax.set_ylabel("time (ms)")

    def overlay_mask(ax, mask, rgb):
        rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
        rgba[..., 0], rgba[..., 1], rgba[..., 2] = rgb
        rgba[..., 3] = (mask != 0).astype(np.float32) * 0.45
        ax.imshow(rgba, aspect="auto", extent=extent, origin="upper")

    # Panel 0: GT
    draw_spectrogram(axes[0], f"{bundle.file_stem}\nframe {bundle.frame_number} — ground truth")
    overlay_mask(axes[0], bundle.gt_mask, gt_color)
    if show_gt_boxes:
        # Draw boxes from the row/col grid indices (frame-relative, same grid as the
        # mask) — NOT from x_ms/y_mhz: x_ms is absolute capture time, which would push
        # boxes far outside the frame's time window.
        f0, f1 = bundle.freq_extent_mhz
        t0, t1 = bundle.time_extent_ms
        rows = max(1, bundle.fft_rows)
        cols = max(1, bundle.fft_cols)
        for item in bundle.gt_items:
            cs, ce = int(item.get("col_start", 0)), int(item.get("col_stop", 0))
            rs, re = int(item.get("row_start", 0)), int(item.get("row_stop", 0))
            if ce <= cs or re <= rs:
                continue
            freq_left = f0 + (cs / cols) * (f1 - f0)
            freq_w = ((ce - cs) / cols) * (f1 - f0)
            time_top = t0 + (rs / rows) * (t1 - t0)
            time_h = ((re - rs) / rows) * (t1 - t0)
            axes[0].add_patch(Rectangle((freq_left, time_top), freq_w, time_h,
                                        fill=False, edgecolor=box_color, linewidth=1.0))
            # Label just ABOVE the box's top edge (smaller time = higher on the
            # inverted time axis), so it doesn't sit on top of the box/signal.
            axes[0].text(freq_left, time_top, item.get("label", ""), color=box_color,
                         fontsize=7, va="bottom", ha="left", clip_on=True)

    # Detector panels
    for idx, detector in enumerate(detectors, start=1):
        draw_spectrogram(axes[idx], f"{label_for(detector)} mask")
        mask = bundle.detector_masks.get(detector)
        if mask is not None:
            overlay_mask(axes[idx], mask, mask_color)
            cov = float((mask != 0).sum()) / mask.size
            axes[idx].set_title(f"{label_for(detector)} mask  (on={cov*100:.2f}% of grid)")
        else:
            axes[idx].set_title(f"{label_for(detector)} mask (MISSING)")

    # Pin identical limits on every panel so an overlay/box can never autoscale one
    # panel out of sync with the others (this is what made masks look 'misaligned').
    for ax in axes:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

    return fig


def available_frames(batch_root: Path, detector: str, file_stem: str) -> list[int]:
    run_dir = Path(batch_root) / detector / file_stem
    return [int(r["frame_number"]) for r in mem.load_manifest(run_dir)]


def frame_gt_counts(run_dir: Path) -> dict:
    """{frame_number -> number of GT annotation items} from gt_annotations/*.json.

    GT is identical across detectors (rasterised from the SigMF), so any detector's
    run dir works. Cheap — only reads the small JSON sidecars, not the masks.
    """
    run_dir = Path(run_dir)
    counts: dict = {}
    for jf in sorted((run_dir / "gt_annotations").glob("*.json")):
        try:
            p = json.loads(jf.read_text())
        except Exception:
            continue
        counts[int(p.get("frame_number", 0))] = len(p.get("items", []))
    return counts


def classify_frames(batch_root: Path, file_stem: str, gt_from: Optional[str] = None):
    """Split a file's frames into (annotated, noise_only) by GT item count.

    Returns ``(annotated_frames, noise_frames)`` as sorted lists of frame numbers.
    """
    layout = resolve_layout(Path(batch_root), file_stem)
    if not layout["detectors"]:
        raise FileNotFoundError(f"No detector runs found under {batch_root} for {file_stem}")
    detector = gt_from or layout["detectors"][0]
    counts = frame_gt_counts(layout["batch_root"] / detector / layout["file_stem"])
    annotated = sorted(f for f, n in counts.items() if n > 0)
    noise = sorted(f for f, n in counts.items() if n == 0)
    return annotated, noise


def sample_review_frames(batch_root: Path, file_stem: str, n_annotated: int = 5,
                         n_noise: int = 1, seed: int = 0) -> dict:
    """Pick ``n_annotated`` random annotated frames + ``n_noise`` random noise-only
    frames (reproducible via ``seed``). Returns a dict with both lists + the picks.
    """
    import random
    annotated, noise = classify_frames(batch_root, file_stem)
    rng = random.Random(seed)
    picked_ann = sorted(rng.sample(annotated, min(n_annotated, len(annotated)))) if annotated else []
    picked_noise = sorted(rng.sample(noise, min(n_noise, len(noise)))) if noise else []
    return {
        "annotated_available": len(annotated),
        "noise_available": len(noise),
        "annotated_frames": picked_ann,
        "noise_frames": picked_noise,
        "review_frames": picked_ann + picked_noise,
    }


# --------------------------------------------------------------------------- #
# Filtered frame selection (by signal class / bandwidth / pulse-length buckets)
# Uses the region_metrics.csv table (has the re-joined per-annotation attributes),
# so it can pick example frames containing a given signal type / bandwidth / duration.
# --------------------------------------------------------------------------- #
def load_region_table(region_csv: Path) -> list[dict]:
    """Load region_metrics.csv (from eval_detector_masks.py) as a list of dict rows."""
    import csv
    with open(Path(region_csv), newline="") as fh:
        return list(csv.DictReader(fh))


def filter_options(region_rows: list[dict], file_stem: str) -> dict:
    """Available {signal_class, bandwidth, pulse_length} bucket values (with counts)
    for a given capture stem — the menu of what you can filter on."""
    from collections import Counter
    rows = [r for r in region_rows if r.get("file_stem") == file_stem]

    def counts(col):
        return dict(Counter(r.get(col, "") for r in rows))

    return {
        "file_stem": file_stem,
        "n_annotation_rows": len(rows),
        "signal_class": counts("bucket_signal_class"),
        "bandwidth": counts("bucket_bandwidth"),
        "pulse_length": counts("bucket_pulse_length"),
    }


def frames_matching(region_rows: list[dict], file_stem: str,
                    signal_class: Optional[str] = None,
                    bandwidth: Optional[str] = None,
                    pulse_length: Optional[str] = None,
                    detector: Optional[str] = None) -> list[int]:
    """Sorted, de-duplicated frame numbers for `file_stem` whose annotations match the
    given bucket filters (any of which may be None = 'any'). GT is detector-independent,
    so `detector` only limits which run's rows are scanned (defaults to any)."""
    out = set()
    for r in region_rows:
        if r.get("file_stem") != file_stem:
            continue
        if detector and r.get("detector") != detector:
            continue
        if signal_class and r.get("bucket_signal_class") != signal_class:
            continue
        if bandwidth and r.get("bucket_bandwidth") != bandwidth:
            continue
        if pulse_length and r.get("bucket_pulse_length") != pulse_length:
            continue
        try:
            out.add(int(r["frame_number"]))
        except (KeyError, ValueError):
            continue
    return sorted(out)


def pick_spread(items: list, n: int) -> list:
    """Evenly-spaced subset of up to `n` items (first/…/last) for varied examples."""
    if len(items) <= n:
        return list(items)
    idx = [round(i * (len(items) - 1) / (n - 1)) for i in range(n)] if n > 1 else [0]
    return [items[i] for i in sorted(set(idx))]


def select_frames(batch_root: Path, region_rows: list[dict], file_stem: str,
                  signal_class: Optional[str] = None, bandwidth: Optional[str] = None,
                  pulse_length: Optional[str] = None) -> list[int]:
    """Frames to visualize for the filter cell.

    The special class ``'noise'`` returns **blank / noise-only** frames (no GT annotations),
    ignoring the bandwidth/duration filters. Any other class delegates to ``frames_matching``.
    """
    if signal_class and signal_class.lower() == "noise":
        _annotated, noise = classify_frames(batch_root, file_stem)
        return noise
    return frames_matching(region_rows, file_stem, signal_class=signal_class,
                           bandwidth=bandwidth, pulse_length=pulse_length)
