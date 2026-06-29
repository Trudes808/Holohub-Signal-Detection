from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any

import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

KNOWN_RELATIVE_DIR = Path("applications/usrp_wideband_signal_detection/infocom_evals/signal_detection_experiments")
DEFAULT_INPUT_DATA_PATH = Path("/home/bqn82/captures/attenuation_dB_0.sigmf-data")
DEFAULT_ANNOTATION_INDEX = None
DEFAULT_FFT_SIZE = 2048
DEFAULT_HOP_SIZE = 512
DEFAULT_DYNAMIC_RANGE_DB = 70.0
DEFAULT_FIGSIZE = (18.0, 7.0)
DEFAULT_REPLAY_OUTPUT_DIR = Path("/tmp/signal_detection_eval_cuda_dino")
DEFAULT_REPLAY_CONFIG_PATH = Path(
    "/home/sat3737/holohub-dev/applications/usrp_wideband_signal_detection/config_cuda_dino_performance_single_channel.yaml"
)
DEFAULT_REPLAY_BINARY_PATH = Path(
    "/home/sat3737/holohub-dev/build/applications/usrp_wideband_signal_detection/offline_cuda_dino_operator_replay"
)
DEFAULT_OFFLINE_WRAPPER_PATH = Path(
    "/home/sat3737/holohub-dev/applications/usrp_wideband_signal_detection/run_cuda_dino_offline_file.py"
)
DEFAULT_OFFLINE_OUTPUT_ROOT = Path("/tmp/usrp_spectrograms/offline_cuda_dino")
HOST_SCRATCH_ROOT = Path("/tmp/usrp_spectrograms")
CONTAINER_SCRATCH_ROOT = Path("/workspace/spectrograms")
DEFAULT_WINDOWED_SIGMF_ROOT = Path(
    "/home/sat3737/holohub-dev/applications/usrp_wideband_signal_detection/generated_inputs"
)
DEFAULT_OFFLINE_FRAME_SAMPLE_COUNT = 512 * 10240
KIND_COLORS = {
    "waveform": "lime",
    "metadata": "deepskyblue",
    "zadoff_chu": "gold",
    "annotation": "magenta",
}


@dataclass(frozen=True)
class SigMFBundle:
    data_path: Path
    meta_path: Path
    meta: dict[str, Any]
    sample_rate_hz: float
    center_frequency_hz: float
    datatype: str
    scalar_dtype: np.dtype
    total_complex_samples: int
    capture_sample_start: int
    annotations: list[dict[str, Any]]


@dataclass(frozen=True)
class SampleWindow:
    start_sample: int
    sample_count: int
    stop_sample: int
    annotation_index: int | None


@dataclass(frozen=True)
class SpectrogramFrame:
    analysis_tensor: np.ndarray
    power_db: np.ndarray
    absolute_sample_axis: np.ndarray
    absolute_frequency_axis_hz: np.ndarray
    extent: list[float]


@dataclass(frozen=True)
class ReplayArtifacts:
    output_dir: Path
    tensor_path: Path
    gt_mask_path: Path
    summary_path: Path
    replay_command: str


@dataclass(frozen=True)
class OfflineFrameMatch:
    frame_number: int
    overlap_samples: int
    frame_start_sample: int
    frame_stop_sample: int


@dataclass(frozen=True)
class OfflineDetectorDebugArtifacts:
    output_root: Path
    chunk_summary: dict[str, Any]
    validation_summary: dict[str, Any]
    chunk_arrays: dict[str, np.ndarray]
    global_arrays: dict[str, np.ndarray]


def resolve_notebook_dir(start_dir: Path | None = None) -> Path:
    cwd = (start_dir or Path.cwd()).resolve()
    candidate = (cwd / KNOWN_RELATIVE_DIR).resolve()
    if candidate.exists():
        return candidate
    return cwd


def sigmf_meta_for_data_path(data_path: Path) -> Path:
    if data_path.name.endswith(".sigmf-data"):
        return data_path.with_name(data_path.name[: -len(".sigmf-data")] + ".sigmf-meta")
    return data_path.with_suffix(".sigmf-meta")


def parse_sigmf_complex_dtype(datatype: str) -> np.dtype:
    base = datatype[:-3] if datatype.endswith(("_le", "_be")) else datatype
    endian = "<" if datatype.endswith("_le") else ">" if datatype.endswith("_be") else "="
    if not base.startswith("c") or len(base) < 3:
        raise ValueError(f"Expected a complex SigMF datatype, got {datatype!r}")

    scalar_kind = base[1]
    scalar_bits = int(base[2:])
    if scalar_bits % 8 != 0:
        raise ValueError(f"Unsupported datatype bit width in {datatype!r}")

    dtype_map = {
        "i": {8: "i1", 16: "i2", 32: "i4", 64: "i8"},
        "u": {8: "u1", 16: "u2", 32: "u4", 64: "u8"},
        "f": {16: "f2", 32: "f4", 64: "f8"},
    }
    code = dtype_map.get(scalar_kind, {}).get(scalar_bits)
    if code is None:
        raise ValueError(f"Unsupported SigMF datatype {datatype!r}")
    if code.endswith("1"):
        return np.dtype(code)
    return np.dtype(endian + code)


def load_sigmf_bundle(data_path: str | Path) -> SigMFBundle:
    resolved_data_path = Path(data_path).expanduser().resolve()
    if not resolved_data_path.is_file():
        raise FileNotFoundError(f"Missing SigMF data file: {resolved_data_path}")

    meta_path = sigmf_meta_for_data_path(resolved_data_path)
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing SigMF metadata sidecar: {meta_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    global_info = meta.get("global", {})
    captures = meta.get("captures", [])
    capture_info = captures[0] if captures else {}
    datatype = str(global_info.get("core:datatype", "")).strip()
    if not datatype:
        raise ValueError(f"{meta_path} is missing global.core:datatype")

    sample_rate_hz = float(global_info.get("core:sample_rate", 0.0))
    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0.0:
        raise ValueError(f"{meta_path} has an invalid global.core:sample_rate")

    scalar_dtype = parse_sigmf_complex_dtype(datatype)
    bytes_per_complex = scalar_dtype.itemsize * 2
    file_size = resolved_data_path.stat().st_size
    if file_size == 0:
        raise ValueError(f"{resolved_data_path} is empty")
    if file_size % bytes_per_complex:
        raise ValueError(
            f"{resolved_data_path} size {file_size} is not aligned to {bytes_per_complex} bytes/complex sample"
        )

    return SigMFBundle(
        data_path=resolved_data_path,
        meta_path=meta_path,
        meta=meta,
        sample_rate_hz=sample_rate_hz,
        center_frequency_hz=float(capture_info.get("core:frequency", 0.0) or 0.0),
        datatype=datatype,
        scalar_dtype=scalar_dtype,
        total_complex_samples=file_size // bytes_per_complex,
        capture_sample_start=int(capture_info.get("core:sample_start", 0) or 0),
        annotations=list(meta.get("annotations", [])),
    )


def show_offline_saved_mask_comparison(
    bundle: SigMFBundle,
    window: SampleWindow,
    saved_detector_mask: np.ndarray | None = None,
    saved_gt_mask: np.ndarray | None = None,
    saved_detector_mask_path: str | Path | None = None,
    saved_gt_mask_path: str | Path | None = None,
    fft_size: int = DEFAULT_FFT_SIZE,
    hop_size: int = DEFAULT_HOP_SIZE,
    dynamic_range_db: float = DEFAULT_DYNAMIC_RANGE_DB,
    figsize: tuple[float, float] = (20.0, 6.0),
) -> tuple[Any, Any, dict[str, Any]]:
    spectrogram_frame = build_spectrogram_frame(bundle, window, fft_size=fft_size, hop_size=hop_size)
    overlays = build_annotation_overlays(bundle, window)

    resolved_gt_mask = saved_gt_mask
    if resolved_gt_mask is None and saved_gt_mask_path is not None:
        resolved_gt_mask = load_detector_mask(saved_gt_mask_path)

    resolved_detector_mask = saved_detector_mask
    if resolved_detector_mask is None and saved_detector_mask_path is not None:
        resolved_detector_mask = load_detector_mask(saved_detector_mask_path)

    display_gt_mask = None if resolved_gt_mask is None else np.asarray(resolved_gt_mask).T
    display_detector_mask = None if resolved_detector_mask is None else np.asarray(resolved_detector_mask).T

    vmax = float(np.nanmax(spectrogram_frame.power_db))
    vmin = vmax - float(dynamic_range_db)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
        vmin, vmax = np.percentile(spectrogram_frame.power_db, [5.0, 99.5])

    fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True, sharex=True, sharey=True)
    titles = ["SigMF GT Boxes", "Saved Offline GT Mask", "Saved Offline Detector vs GT"]
    for ax, title in zip(axes, titles):
        ax.imshow(
            spectrogram_frame.power_db,
            origin="lower",
            aspect="auto",
            extent=spectrogram_frame.extent,
            cmap="magma",
            vmin=float(vmin),
            vmax=float(vmax),
        )
        ax.set_title(title)
        ax.set_xlabel("Sample Index")
    axes[0].set_ylabel("Frequency (MHz)")

    for overlay in overlays:
        color = KIND_COLORS.get(overlay["kind"], "magenta")
        rect = Rectangle(
            (float(overlay["overlap_start"]), float(overlay["freq_lower_hz"] / 1.0e6)),
            max(1.0, float(overlay["overlap_stop"] - overlay["overlap_start"])),
            float((overlay["freq_upper_hz"] - overlay["freq_lower_hz"]) / 1.0e6),
            fill=False,
            linewidth=2.0,
            edgecolor=color,
        )
        axes[0].add_patch(rect)

    offline_extent = [
        float(window.start_sample),
        float(window.stop_sample),
        float((bundle.center_frequency_hz - 0.5 * bundle.sample_rate_hz) / 1.0e6),
        float((bundle.center_frequency_hz + 0.5 * bundle.sample_rate_hz) / 1.0e6),
    ]

    if display_gt_mask is not None:
        gt_overlay = np.where(display_gt_mask > 0, 1.0, np.nan)
        axes[1].imshow(
            gt_overlay,
            origin="lower",
            aspect="auto",
            extent=offline_extent,
            cmap="Blues",
            alpha=0.45,
            interpolation="nearest",
        )
        axes[2].imshow(
            gt_overlay,
            origin="lower",
            aspect="auto",
            extent=offline_extent,
            cmap="Blues",
            alpha=0.30,
            interpolation="nearest",
        )

    if display_detector_mask is not None:
        detector_overlay = np.where(display_detector_mask > 0, 1.0, np.nan)
        axes[2].imshow(
            detector_overlay,
            origin="lower",
            aspect="auto",
            extent=offline_extent,
            cmap="Greens",
            alpha=0.35,
            interpolation="nearest",
        )

    saved_mask_iou = None
    if resolved_gt_mask is not None and resolved_detector_mask is not None:
        saved_mask_iou = compute_mask_iou(
            np.asarray(resolved_gt_mask, dtype=np.uint8),
            np.asarray(resolved_detector_mask, dtype=np.uint8),
        )

    comparison_context = {
        "window_start_sample": window.start_sample,
        "window_stop_sample": window.stop_sample,
        "saved_gt_annotation_count": len(overlays),
        "saved_gt_mask_shape": None if resolved_gt_mask is None else list(np.asarray(resolved_gt_mask).shape),
        "saved_detector_mask_shape": None
        if resolved_detector_mask is None
        else list(np.asarray(resolved_detector_mask).shape),
        "saved_mask_iou": saved_mask_iou,
    }
    if resolved_detector_mask is not None:
        print(f"Saved offline detector mask pixels: {int(np.count_nonzero(np.asarray(resolved_detector_mask) > 0))}")
    if resolved_gt_mask is not None:
        print(f"Saved offline GT mask pixels: {int(np.count_nonzero(np.asarray(resolved_gt_mask) > 0))}")
    return fig, axes, comparison_context


def show_offline_saved_binary_masks(
    bundle: SigMFBundle,
    window: SampleWindow,
    saved_detector_mask: np.ndarray | None = None,
    saved_gt_mask: np.ndarray | None = None,
    saved_detector_mask_path: str | Path | None = None,
    saved_gt_mask_path: str | Path | None = None,
    figsize: tuple[float, float] = (16.0, 6.0),
) -> tuple[Any, Any, dict[str, Any]]:
    resolved_gt_mask = saved_gt_mask
    if resolved_gt_mask is None and saved_gt_mask_path is not None:
        resolved_gt_mask = load_detector_mask(saved_gt_mask_path)

    resolved_detector_mask = saved_detector_mask
    if resolved_detector_mask is None and saved_detector_mask_path is not None:
        resolved_detector_mask = load_detector_mask(saved_detector_mask_path)

    display_gt_mask = None if resolved_gt_mask is None else np.asarray(resolved_gt_mask).T
    display_detector_mask = None if resolved_detector_mask is None else np.asarray(resolved_detector_mask).T
    offline_extent = [
        float(window.start_sample),
        float(window.stop_sample),
        float((bundle.center_frequency_hz - 0.5 * bundle.sample_rate_hz) / 1.0e6),
        float((bundle.center_frequency_hz + 0.5 * bundle.sample_rate_hz) / 1.0e6),
    ]

    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True, sharex=True, sharey=True)
    titles = ["Saved Offline GT Binary Mask", "Saved Offline Detector Binary Mask"]
    masks = [display_gt_mask, display_detector_mask]
    cmaps = ["Blues", "Greens"]
    for ax, title, mask, cmap in zip(axes, titles, masks, cmaps):
        ax.set_title(title)
        ax.set_xlabel("Sample Index")
        if mask is None:
            ax.text(0.5, 0.5, "No saved mask", transform=ax.transAxes, ha="center", va="center")
            continue
        binary_mask = (np.asarray(mask) > 0).astype(np.uint8)
        ax.imshow(
            binary_mask,
            origin="lower",
            aspect="auto",
            extent=offline_extent,
            cmap=cmap,
            interpolation="nearest",
            vmin=0,
            vmax=1,
        )
    axes[0].set_ylabel("Frequency (MHz)")

    context = {
        "saved_gt_mask_shape": None if resolved_gt_mask is None else list(np.asarray(resolved_gt_mask).shape),
        "saved_detector_mask_shape": None
        if resolved_detector_mask is None
        else list(np.asarray(resolved_detector_mask).shape),
        "saved_gt_mask_pixels": 0 if resolved_gt_mask is None else int(np.count_nonzero(np.asarray(resolved_gt_mask) > 0)),
        "saved_detector_mask_pixels": 0
        if resolved_detector_mask is None
        else int(np.count_nonzero(np.asarray(resolved_detector_mask) > 0)),
    }
    return fig, axes, context


def resolve_offline_artifact_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    if resolved.exists():
        return resolved.resolve()
    try:
        relative = resolved.relative_to(CONTAINER_SCRATCH_ROOT)
    except ValueError:
        return resolved.resolve()
    candidate = HOST_SCRATCH_ROOT / relative
    return candidate.resolve()


def load_json_artifact(path: str | Path) -> dict[str, Any]:
    resolved_path = resolve_offline_artifact_path(path)
    return json.loads(resolved_path.read_text(encoding="utf-8"))


def build_missing_debug_artifacts_message(output_root: Path) -> str:
    validation_summary_path = output_root / "offline_validation_summary.json"
    return (
        "Offline detector debug artifacts were not found under "
        f"{output_root}. Expected file: {validation_summary_path}. "
        "This usually means the offline run was executed before the debug-artifact config update was regenerated. "
        "Rerun notebook cell 8 so the wrapper regenerates the offline config and prints fresh manual commands, "
        "then rerun the printed sudo/docker command, and finally rerun notebook cell 10."
    )


def load_offline_detector_debug_artifacts(output_root: str | Path) -> OfflineDetectorDebugArtifacts:
    resolved_output_root = Path(output_root).expanduser().resolve()
    validation_summary_path = resolved_output_root / "offline_validation_summary.json"
    if not validation_summary_path.exists():
        raise FileNotFoundError(build_missing_debug_artifacts_message(resolved_output_root))

    validation_summary = load_json_artifact(validation_summary_path)
    chunk_summary = load_json_artifact(validation_summary["chunk_debug_summary_json"])

    chunk_array_keys = [
        "corrected_resized_npy",
        "dino_score_raw_npy",
        "dino_score_raw_deweighted_npy",
        "coherence_gate_npy",
        "combined_score_npy",
        "hybrid_keep_freq_npy",
        "hybrid_keep_res_npy",
        "hybrid_seed_mask_npy",
        "hybrid_closed_mask_npy",
        "hybrid_filled_mask_npy",
        "hybrid_component_filtered_mask_npy",
        "grouped_mask_npy",
        "final_mask_npy",
        "final_mask_source_npy",
        "final_mask_projected_npy",
    ]
    global_array_keys = [
        "projected_grouped_mask_npy",
        "projected_grouped_score_npy",
        "merged_box_mask_npy",
        "final_mask_npy",
    ]

    chunk_arrays = {
        key: np.asarray(np.load(resolve_offline_artifact_path(chunk_summary[key]))) for key in chunk_array_keys
    }
    global_arrays = {
        key: np.asarray(np.load(resolve_offline_artifact_path(validation_summary[key]))) for key in global_array_keys
    }
    return OfflineDetectorDebugArtifacts(
        output_root=resolved_output_root,
        chunk_summary=chunk_summary,
        validation_summary=validation_summary,
        chunk_arrays=chunk_arrays,
        global_arrays=global_arrays,
    )


def _show_stage_image(
    ax: Any,
    image: np.ndarray,
    title: str,
    cmap: str,
    binary: bool = False,
) -> None:
    data = np.asarray(image)
    if binary:
        display = (data > 0).astype(np.uint8)
        ax.imshow(display.T if display.shape[0] > display.shape[1] else display, origin="lower", aspect="auto", cmap=cmap, vmin=0, vmax=1)
    else:
        finite = np.asarray(data, dtype=np.float32)
        vmin, vmax = np.percentile(finite, [5.0, 99.5]) if finite.size else (0.0, 1.0)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
            vmin, vmax = float(np.nanmin(finite)), float(np.nanmax(finite))
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
                vmin, vmax = 0.0, 1.0
        ax.imshow(finite.T if finite.shape[0] > finite.shape[1] else finite, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("Time bins")
    ax.set_ylabel("Freq bins")


def show_offline_detector_debug_pathways(
    debug_artifacts: OfflineDetectorDebugArtifacts,
    figsize: tuple[float, float] = (18.0, 10.0),
) -> tuple[Any, Any, dict[str, Any]]:
    fig, axes = plt.subplots(2, 4, figsize=figsize, constrained_layout=True)
    panels = [
        ("corrected_resized_npy", "Corrected Spectrogram", "magma", False),
        ("dino_score_raw_npy", "DINO Raw Score", "viridis", False),
        ("dino_score_raw_deweighted_npy", "DINO Deweighted Score", "viridis", False),
        ("coherence_gate_npy", "Coherence Gate", "cividis", False),
        ("hybrid_keep_freq_npy", "Pathway A: Keep Freq", "plasma", False),
        ("hybrid_keep_res_npy", "Pathway B: Keep Residual", "plasma", False),
        ("combined_score_npy", "Combined Score", "inferno", False),
        ("projected_grouped_score_npy", "Projected Combined Score", "inferno", False),
    ]
    for ax, (key, title, cmap, binary) in zip(axes.flat, panels):
        source = debug_artifacts.chunk_arrays if key in debug_artifacts.chunk_arrays else debug_artifacts.global_arrays
        _show_stage_image(ax, source[key], title, cmap, binary=binary)

    context = {
        "selected_chunk_index": int(debug_artifacts.validation_summary["selected_chunk_index"]),
        "chunk_count": int(debug_artifacts.validation_summary["chunk_count"]),
        "chunk_thresholds": dict(debug_artifacts.chunk_summary.get("hybrid_thresholds", {})),
        "operator_timing_ms": dict(debug_artifacts.validation_summary.get("operator_timing_ms", {})),
    }
    return fig, axes, context


def show_offline_detector_debug_postprocess(
    debug_artifacts: OfflineDetectorDebugArtifacts,
    saved_detector_mask: np.ndarray | None = None,
    saved_gt_mask: np.ndarray | None = None,
    figsize: tuple[float, float] = (18.0, 10.0),
) -> tuple[Any, Any, dict[str, Any]]:
    fig, axes = plt.subplots(2, 4, figsize=figsize, constrained_layout=True)
    panels: list[tuple[str, str, str, bool, str]] = [
        ("hybrid_seed_mask_npy", "Seed Mask", "Blues", True, "chunk"),
        ("hybrid_closed_mask_npy", "Closed Mask", "Blues", True, "chunk"),
        ("hybrid_filled_mask_npy", "Filled Mask", "Blues", True, "chunk"),
        ("hybrid_component_filtered_mask_npy", "Component Filtered", "Blues", True, "chunk"),
        ("grouped_mask_npy", "Grouped Mask", "Greens", True, "chunk"),
        ("final_mask_npy", "Global Final Mask", "Greens", True, "global"),
        ("merged_box_mask_npy", "Merged Box Mask", "Greens", True, "global"),
        ("projected_grouped_mask_npy", "Projected Grouped Mask", "Greens", True, "global"),
    ]
    for ax, (key, title, cmap, binary, source_name) in zip(axes.flat, panels):
        source = debug_artifacts.chunk_arrays if source_name == "chunk" else debug_artifacts.global_arrays
        _show_stage_image(ax, source[key], title, cmap, binary=binary)

    detector_pixels = None if saved_detector_mask is None else int(np.count_nonzero(np.asarray(saved_detector_mask) > 0))
    gt_pixels = None if saved_gt_mask is None else int(np.count_nonzero(np.asarray(saved_gt_mask) > 0))
    context = {
        "selected_chunk_index": int(debug_artifacts.validation_summary["selected_chunk_index"]),
        "detector_mask_pixels": detector_pixels,
        "ground_truth_mask_pixels": gt_pixels,
        "postprocess_timing_ms": {
            key: value
            for key, value in dict(debug_artifacts.validation_summary.get("operator_timing_ms", {})).items()
            if key.startswith("hybrid_") or key in {"global_merge", "artifact_serialization"}
        },
    }
    return fig, axes, context
def normalize_annotation_index(bundle: SigMFBundle, annotation_index: int | None) -> int | None:
    if annotation_index is None:
        return None
    if annotation_index < 0 or annotation_index >= len(bundle.annotations):
        raise IndexError(
            f"annotation_index={annotation_index} is out of range for {len(bundle.annotations)} annotations"
        )
    return annotation_index


def choose_default_annotation_index(bundle: SigMFBundle) -> int | None:
    if not bundle.annotations:
        return None

    for preferred_kind in ("waveform", "metadata", "zadoff_chu"):
        candidates = [
            (index, annotation)
            for index, annotation in enumerate(bundle.annotations)
            if str(annotation.get("wfgt:kind", "")).strip().lower() == preferred_kind
        ]
        if candidates:
            return max(
                candidates,
                key=lambda item: int(item[1].get("core:sample_count", 0) or 0),
            )[0]

    return max(
        range(len(bundle.annotations)),
        key=lambda index: int(bundle.annotations[index].get("core:sample_count", 0) or 0),
    )


def choose_sample_window(
    bundle: SigMFBundle,
    sample_start: int | None = None,
    sample_count: int | None = None,
    annotation_index: int | None = DEFAULT_ANNOTATION_INDEX,
    margin_ratio: float = 0.15,
    min_sample_count: int = 1 << 20,
) -> SampleWindow:
    if sample_start is not None and sample_count is not None:
        start_sample = max(0, int(sample_start))
        bounded_count = max(1, int(sample_count))
        stop_sample = min(bundle.total_complex_samples, start_sample + bounded_count)
        return SampleWindow(
            start_sample=start_sample,
            sample_count=stop_sample - start_sample,
            stop_sample=stop_sample,
            annotation_index=normalize_annotation_index(bundle, annotation_index),
        )

    selected_annotation_index = normalize_annotation_index(bundle, annotation_index)
    annotation: dict[str, Any] | None = None
    if selected_annotation_index is None:
        selected_annotation_index = choose_default_annotation_index(bundle)
    if selected_annotation_index is not None and bundle.annotations:
        annotation = bundle.annotations[selected_annotation_index]

    if annotation is None:
        fallback_count = min(bundle.total_complex_samples, max(min_sample_count, DEFAULT_FFT_SIZE * 256))
        return SampleWindow(
            start_sample=0,
            sample_count=fallback_count,
            stop_sample=fallback_count,
            annotation_index=None,
        )

    annotation_start = int(annotation.get("core:sample_start", 0) or 0)
    annotation_count = max(1, int(annotation.get("core:sample_count", 1) or 1))
    context_margin = max(DEFAULT_FFT_SIZE * 8, int(round(annotation_count * margin_ratio)))
    requested_start = annotation_start - context_margin if sample_start is None else int(sample_start)
    requested_count = (
        annotation_count + 2 * context_margin if sample_count is None else max(1, int(sample_count))
    )
    bounded_count = max(min_sample_count, requested_count)
    start_sample = max(0, requested_start)
    if start_sample + bounded_count > bundle.total_complex_samples:
        start_sample = max(0, bundle.total_complex_samples - bounded_count)
    stop_sample = min(bundle.total_complex_samples, start_sample + bounded_count)

    return SampleWindow(
        start_sample=start_sample,
        sample_count=stop_sample - start_sample,
        stop_sample=stop_sample,
        annotation_index=selected_annotation_index,
    )


def choose_offline_compatible_window(
    bundle: SigMFBundle,
    sample_start: int | None = None,
    sample_count: int | None = None,
    annotation_index: int | None = DEFAULT_ANNOTATION_INDEX,
    frame_sample_count: int = DEFAULT_OFFLINE_FRAME_SAMPLE_COUNT,
) -> SampleWindow:
    if sample_start is not None and sample_count is not None:
        start_sample = max(0, int(sample_start))
        resolved_count = max(1, int(sample_count))
        stop_sample = min(bundle.total_complex_samples, start_sample + resolved_count)
        return SampleWindow(
            start_sample=start_sample,
            sample_count=stop_sample - start_sample,
            stop_sample=stop_sample,
            annotation_index=normalize_annotation_index(bundle, annotation_index),
        )

    selected_annotation_index = normalize_annotation_index(bundle, annotation_index)
    if selected_annotation_index is None:
        selected_annotation_index = choose_default_annotation_index(bundle)

    if selected_annotation_index is None or not bundle.annotations:
        fallback_count = min(bundle.total_complex_samples, frame_sample_count)
        return SampleWindow(
            start_sample=0,
            sample_count=fallback_count,
            stop_sample=fallback_count,
            annotation_index=None,
        )

    annotation = bundle.annotations[selected_annotation_index]
    annotation_start = int(annotation.get("core:sample_start", 0) or 0)
    annotation_count = max(1, int(annotation.get("core:sample_count", 1) or 1))
    resolved_count = max(frame_sample_count, annotation_count)
    if resolved_count > frame_sample_count:
        resolved_count = pad_sample_count_to_frame_multiple(resolved_count, frame_sample_count=frame_sample_count)

    slack = max(0, resolved_count - annotation_count)
    start_sample = max(0, annotation_start - (slack // 2))
    stop_sample = start_sample + resolved_count
    if stop_sample > bundle.total_complex_samples:
        stop_sample = bundle.total_complex_samples
        start_sample = max(0, stop_sample - resolved_count)

    return SampleWindow(
        start_sample=start_sample,
        sample_count=stop_sample - start_sample,
        stop_sample=stop_sample,
        annotation_index=selected_annotation_index,
    )


def read_complex_samples(bundle: SigMFBundle, window: SampleWindow) -> np.ndarray:
    raw = np.memmap(
        bundle.data_path,
        dtype=bundle.scalar_dtype,
        mode="r",
        offset=window.start_sample * bundle.scalar_dtype.itemsize * 2,
        shape=(window.sample_count, 2),
    )
    i_samples = np.asarray(raw[:, 0], dtype=np.float32)
    q_samples = np.asarray(raw[:, 1], dtype=np.float32)
    return i_samples + 1j * q_samples


def compute_spectrogram(
    iq_samples: np.ndarray,
    sample_rate_hz: float,
    fft_size: int = DEFAULT_FFT_SIZE,
    hop_size: int = DEFAULT_HOP_SIZE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if fft_size <= 0:
        raise ValueError("fft_size must be positive")
    if hop_size <= 0:
        raise ValueError("hop_size must be positive")
    if iq_samples.size < fft_size:
        raise ValueError(
            f"Need at least fft_size={fft_size} complex samples, received {iq_samples.size}"
        )

    frame_count = 1 + (iq_samples.size - fft_size) // hop_size
    fft_window = np.hanning(fft_size).astype(np.float32)
    analysis_tensor = np.empty((fft_size, frame_count), dtype=np.complex64)
    power_db = np.empty((fft_size, frame_count), dtype=np.float32)
    batch_size = 256

    for batch_start in range(0, frame_count, batch_size):
        batch_stop = min(frame_count, batch_start + batch_size)
        batch_indices = np.arange(batch_start, batch_stop, dtype=np.int64) * hop_size
        batch_frames = np.stack([iq_samples[index : index + fft_size] for index in batch_indices], axis=0)
        batch_spectrum = np.fft.fftshift(np.fft.fft(batch_frames * fft_window[None, :], axis=1), axes=1)
        batch_power = np.abs(batch_spectrum) ** 2
        analysis_tensor[:, batch_start:batch_stop] = batch_spectrum.T.astype(np.complex64)
        power_db[:, batch_start:batch_stop] = (10.0 * np.log10(batch_power + 1.0e-12)).T.astype(np.float32)

    frequency_axis_hz = np.fft.fftshift(np.fft.fftfreq(fft_size, d=1.0 / sample_rate_hz))
    frame_center_offsets = np.arange(frame_count, dtype=np.float64) * hop_size + (fft_size / 2.0)
    return analysis_tensor, power_db, frame_center_offsets, frequency_axis_hz


def build_spectrogram_frame(
    bundle: SigMFBundle,
    window: SampleWindow,
    fft_size: int = DEFAULT_FFT_SIZE,
    hop_size: int = DEFAULT_HOP_SIZE,
) -> SpectrogramFrame:
    iq_samples = read_complex_samples(bundle, window)
    analysis_tensor, power_db, frame_offsets, frequency_axis_hz = compute_spectrogram(
        iq_samples,
        sample_rate_hz=bundle.sample_rate_hz,
        fft_size=fft_size,
        hop_size=hop_size,
    )
    absolute_sample_axis = frame_offsets + float(window.start_sample)
    absolute_frequency_axis_hz = frequency_axis_hz + bundle.center_frequency_hz
    extent = [
        float(absolute_sample_axis[0]),
        float(absolute_sample_axis[-1]),
        float(absolute_frequency_axis_hz[0] / 1.0e6),
        float(absolute_frequency_axis_hz[-1] / 1.0e6),
    ]
    return SpectrogramFrame(
        analysis_tensor=analysis_tensor,
        power_db=power_db,
        absolute_sample_axis=absolute_sample_axis,
        absolute_frequency_axis_hz=absolute_frequency_axis_hz,
        extent=extent,
    )


def annotation_overlaps_window(annotation: dict[str, Any], window: SampleWindow) -> bool:
    annotation_start = int(annotation.get("core:sample_start", 0) or 0)
    annotation_stop = annotation_start + int(annotation.get("core:sample_count", 0) or 0)
    return annotation_stop > window.start_sample and annotation_start < window.stop_sample


def build_annotation_overlays(bundle: SigMFBundle, window: SampleWindow) -> list[dict[str, Any]]:
    overlays: list[dict[str, Any]] = []
    for annotation_index, annotation in enumerate(bundle.annotations):
        if not annotation_overlaps_window(annotation, window):
            continue

        annotation_start = int(annotation.get("core:sample_start", 0) or 0)
        annotation_stop = annotation_start + int(annotation.get("core:sample_count", 0) or 0)
        overlap_start = max(window.start_sample, annotation_start)
        overlap_stop = min(window.stop_sample, annotation_stop)
        overlays.append(
            {
                "annotation_index": annotation_index,
                "sample_start": annotation_start,
                "sample_stop": annotation_stop,
                "overlap_start": overlap_start,
                "overlap_stop": overlap_stop,
                "freq_lower_hz": float(annotation.get("core:freq_lower_edge", 0.0) or 0.0),
                "freq_upper_hz": float(annotation.get("core:freq_upper_edge", 0.0) or 0.0),
                "kind": str(annotation.get("wfgt:kind", "annotation")),
                "label": str(annotation.get("core:label", annotation.get("wfgt:kind", "annotation"))),
            }
        )
    return overlays


def axis_edges_from_centers(centers: np.ndarray) -> np.ndarray:
    centers = np.asarray(centers, dtype=np.float64)
    if centers.ndim != 1 or centers.size == 0:
        raise ValueError("Expected a non-empty 1D centers array")
    if centers.size == 1:
        half_step = 0.5
        return np.asarray([centers[0] - half_step, centers[0] + half_step], dtype=np.float64)
    deltas = np.diff(centers)
    edges = np.empty(centers.size + 1, dtype=np.float64)
    edges[1:-1] = centers[:-1] + 0.5 * deltas
    edges[0] = centers[0] - 0.5 * deltas[0]
    edges[-1] = centers[-1] + 0.5 * deltas[-1]
    return edges


def build_ground_truth_mask(
    bundle: SigMFBundle,
    window: SampleWindow,
    spectrogram_frame: SpectrogramFrame,
) -> np.ndarray:
    time_edges = axis_edges_from_centers(spectrogram_frame.absolute_sample_axis)
    freq_edges = axis_edges_from_centers(spectrogram_frame.absolute_frequency_axis_hz)
    mask = np.zeros(spectrogram_frame.power_db.shape, dtype=np.uint8)

    for overlay in build_annotation_overlays(bundle, window):
        time_active = (time_edges[:-1] < float(overlay["overlap_stop"])) & (
            time_edges[1:] > float(overlay["overlap_start"])
        )
        freq_active = (freq_edges[:-1] < float(overlay["freq_upper_hz"])) & (
            freq_edges[1:] > float(overlay["freq_lower_hz"])
        )
        if not np.any(time_active) or not np.any(freq_active):
            continue
        mask[np.ix_(freq_active, time_active)] = 1
    return mask


def compute_mask_iou(reference_mask: np.ndarray, candidate_mask: np.ndarray) -> float:
    ref = np.asarray(reference_mask) > 0
    cand = np.asarray(candidate_mask) > 0
    if ref.shape != cand.shape:
        raise ValueError(f"Mask shape mismatch: {ref.shape} vs {cand.shape}")
    intersection = np.count_nonzero(ref & cand)
    union = np.count_nonzero(ref | cand)
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection / union)


def load_detector_mask(mask_path: str | Path) -> np.ndarray:
    mask = np.asarray(np.load(Path(mask_path).expanduser().resolve()))
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D detector mask, got shape {mask.shape}")
    return (mask > 0).astype(np.uint8)


def build_detector_replay_command(
    tensor_path: str | Path,
    output_dir: str | Path,
    span_hz: float,
    config_path: str | Path = DEFAULT_REPLAY_CONFIG_PATH,
    replay_binary_path: str | Path = DEFAULT_REPLAY_BINARY_PATH,
    channel_number: int = 0,
    debug_chunk_index: int = 13,
) -> str:
    command_parts = [
        str(Path(replay_binary_path).expanduser()),
        "--tensor-npy",
        str(Path(tensor_path).expanduser()),
        "--output-dir",
        str(Path(output_dir).expanduser()),
        "--config",
        str(Path(config_path).expanduser()),
        "--span-hz",
        f"{float(span_hz):.12g}",
        "--channel",
        str(int(channel_number)),
        "--debug-chunk-index",
        str(int(debug_chunk_index)),
        "--tensor-axis-order",
        "frequency_time",
    ]
    return " ".join(shlex.quote(part) for part in command_parts)


def export_detector_replay_inputs(
    bundle: SigMFBundle,
    window: SampleWindow,
    output_dir: str | Path = DEFAULT_REPLAY_OUTPUT_DIR,
    fft_size: int = DEFAULT_FFT_SIZE,
    hop_size: int = DEFAULT_HOP_SIZE,
    config_path: str | Path = DEFAULT_REPLAY_CONFIG_PATH,
    replay_binary_path: str | Path = DEFAULT_REPLAY_BINARY_PATH,
    channel_number: int = 0,
    debug_chunk_index: int = 13,
) -> ReplayArtifacts:
    resolved_output_dir = Path(output_dir).expanduser().resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    spectrogram_frame = build_spectrogram_frame(bundle, window, fft_size=fft_size, hop_size=hop_size)
    gt_mask = build_ground_truth_mask(bundle, window, spectrogram_frame)

    tensor_path = resolved_output_dir / "detector_input_tensor.npy"
    gt_mask_path = resolved_output_dir / "ground_truth_mask.npy"
    summary_path = resolved_output_dir / "detector_replay_input_summary.json"
    np.save(tensor_path, np.asarray(spectrogram_frame.analysis_tensor, dtype=np.complex64))
    np.save(gt_mask_path, np.asarray(gt_mask, dtype=np.uint8))

    replay_command = build_detector_replay_command(
        tensor_path=tensor_path,
        output_dir=resolved_output_dir,
        span_hz=bundle.sample_rate_hz,
        config_path=config_path,
        replay_binary_path=replay_binary_path,
        channel_number=channel_number,
        debug_chunk_index=debug_chunk_index,
    )
    summary_payload = {
        "input_data_path": str(bundle.data_path),
        "input_meta_path": str(bundle.meta_path),
        "sample_rate_hz": bundle.sample_rate_hz,
        "center_frequency_hz": bundle.center_frequency_hz,
        "window_start_sample": window.start_sample,
        "window_stop_sample": window.stop_sample,
        "window_sample_count": window.sample_count,
        "window_annotation_index": window.annotation_index,
        "fft_size": int(fft_size),
        "hop_size": int(hop_size),
        "analysis_tensor_path": str(tensor_path),
        "analysis_tensor_shape": list(spectrogram_frame.analysis_tensor.shape),
        "ground_truth_mask_path": str(gt_mask_path),
        "ground_truth_mask_shape": list(gt_mask.shape),
        "replay_output_dir": str(resolved_output_dir),
        "replay_command": replay_command,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    print(f"Exported detector replay tensor: {tensor_path}")
    print(f"Exported ground-truth mask: {gt_mask_path}")
    return ReplayArtifacts(
        output_dir=resolved_output_dir,
        tensor_path=tensor_path,
        gt_mask_path=gt_mask_path,
        summary_path=summary_path,
        replay_command=replay_command,
    )


def find_single_detector_mask(output_dir: str | Path) -> Path:
    resolved_output_dir = Path(output_dir).expanduser().resolve()
    candidates = sorted((resolved_output_dir / "mask_arrays").glob("*.npy"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected exactly one detector mask under {resolved_output_dir / 'mask_arrays'}, found {len(candidates)}"
        )
    return candidates[0]


def default_offline_output_root_for_input(data_path: str | Path) -> Path:
    resolved_data_path = Path(data_path).expanduser().resolve()
    stem = resolved_data_path.name
    if stem.endswith(".sigmf-data"):
        stem = stem[: -len(".sigmf-data")]
    else:
        stem = resolved_data_path.stem
    return DEFAULT_OFFLINE_OUTPUT_ROOT / stem


def default_windowed_sigmf_root() -> Path:
    return DEFAULT_WINDOWED_SIGMF_ROOT


def build_windowed_sigmf_stem(data_path: str | Path, window: SampleWindow) -> str:
    resolved_data_path = Path(data_path).expanduser().resolve()
    base_name = resolved_data_path.name
    if base_name.endswith(".sigmf-data"):
        base_name = base_name[: -len(".sigmf-data")]
    else:
        base_name = resolved_data_path.stem
    return f"{base_name}_samples_{window.start_sample}_{window.stop_sample}"


def pad_sample_count_to_frame_multiple(
    sample_count: int,
    frame_sample_count: int = DEFAULT_OFFLINE_FRAME_SAMPLE_COUNT,
) -> int:
    if frame_sample_count <= 0:
        raise ValueError("frame_sample_count must be positive")
    resolved_sample_count = max(1, int(sample_count))
    remainder = resolved_sample_count % frame_sample_count
    if remainder == 0:
        return resolved_sample_count
    return resolved_sample_count + (frame_sample_count - remainder)


def write_sigmf_window(
    bundle: SigMFBundle,
    window: SampleWindow,
    output_root: str | Path | None = None,
    pad_to_frame_multiple: bool = True,
    frame_sample_count: int = DEFAULT_OFFLINE_FRAME_SAMPLE_COUNT,
) -> tuple[Path, Path]:
    resolved_root = (
        Path(output_root).expanduser().resolve() if output_root is not None else default_windowed_sigmf_root().resolve()
    )
    resolved_root.mkdir(parents=True, exist_ok=True)

    stem = build_windowed_sigmf_stem(bundle.data_path, window)
    data_path = resolved_root / f"{stem}.sigmf-data"
    meta_path = resolved_root / f"{stem}.sigmf-meta"

    bytes_per_complex = bundle.scalar_dtype.itemsize * 2
    byte_offset = window.start_sample * bytes_per_complex
    byte_count = window.sample_count * bytes_per_complex
    padded_sample_count = (
        pad_sample_count_to_frame_multiple(window.sample_count, frame_sample_count=frame_sample_count)
        if pad_to_frame_multiple
        else window.sample_count
    )
    with bundle.data_path.open("rb") as source_handle:
        source_handle.seek(byte_offset)
        with data_path.open("wb") as target_handle:
            remaining_bytes = byte_count
            while remaining_bytes > 0:
                chunk = source_handle.read(min(1024 * 1024, remaining_bytes))
                if not chunk:
                    raise ValueError(
                        f"Reached end of file while extracting samples [{window.start_sample}, {window.stop_sample})"
                    )
                target_handle.write(chunk)
                remaining_bytes -= len(chunk)
            padding_samples = max(0, padded_sample_count - window.sample_count)
            if padding_samples > 0:
                target_handle.write(b"\x00" * (padding_samples * bytes_per_complex))

    meta = json.loads(json.dumps(bundle.meta))
    captures = list(meta.get("captures", []))
    if captures:
        captures[0] = dict(captures[0])
    else:
        captures = [{}]
    captures[0]["core:sample_start"] = int(window.start_sample)
    captures[0]["core:frequency"] = float(bundle.center_frequency_hz)
    meta["captures"] = captures

    filtered_annotations: list[dict[str, Any]] = []
    for annotation in bundle.annotations:
        annotation_start = int(annotation.get("core:sample_start", 0) or 0)
        annotation_stop = annotation_start + int(annotation.get("core:sample_count", 0) or 0)
        if annotation_stop <= window.start_sample or annotation_start >= window.stop_sample:
            continue
        filtered_annotations.append(dict(annotation))
    meta["annotations"] = filtered_annotations

    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return data_path, meta_path


def build_offline_wrapper_command(
    data_path: str | Path,
    config_path: str | Path = DEFAULT_REPLAY_CONFIG_PATH,
    output_root: str | Path | None = None,
    wrapper_path: str | Path = DEFAULT_OFFLINE_WRAPPER_PATH,
    target_chunk_count: int | None = None,
    debug_chunk_index: int | None = None,
    dry_run: bool = False,
) -> list[str]:
    resolved_data_path = Path(data_path).expanduser().resolve()
    resolved_wrapper_path = Path(wrapper_path).expanduser().resolve()
    resolved_config_path = Path(config_path).expanduser().resolve()
    resolved_output_root = (
        Path(output_root).expanduser().resolve() if output_root is not None else default_offline_output_root_for_input(resolved_data_path)
    )
    command = [
        sys.executable,
        str(resolved_wrapper_path),
        str(resolved_data_path),
        "--config",
        str(resolved_config_path),
        "--output-root",
        str(resolved_output_root),
    ]
    if target_chunk_count is not None:
        command.extend(["--target-chunk-count", str(int(target_chunk_count))])
    if debug_chunk_index is not None:
        command.extend(["--debug-chunk-index", str(int(debug_chunk_index))])
    if dry_run:
        command.append("--dry-run")
    return command


def run_offline_cuda_dino_file(
    data_path: str | Path,
    config_path: str | Path = DEFAULT_REPLAY_CONFIG_PATH,
    output_root: str | Path | None = None,
    wrapper_path: str | Path = DEFAULT_OFFLINE_WRAPPER_PATH,
    bundle: SigMFBundle | None = None,
    window: SampleWindow | None = None,
    windowed_sigmf_root: str | Path | None = None,
    target_chunk_count: int | None = None,
    debug_chunk_index: int | None = None,
    dry_run: bool = False,
) -> tuple[subprocess.CompletedProcess[str], Path]:
    resolved_data_path = Path(data_path).expanduser().resolve()
    effective_data_path = resolved_data_path
    if window is not None:
        effective_bundle = bundle if bundle is not None else load_sigmf_bundle(resolved_data_path)
        effective_data_path, _ = write_sigmf_window(
            effective_bundle,
            window,
            output_root=windowed_sigmf_root,
        )

    command = build_offline_wrapper_command(
        data_path=effective_data_path,
        config_path=config_path,
        output_root=output_root,
        wrapper_path=wrapper_path,
        target_chunk_count=target_chunk_count,
        debug_chunk_index=debug_chunk_index,
        dry_run=dry_run,
    )
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    resolved_output_root = (
        Path(output_root).expanduser().resolve()
        if output_root is not None
        else default_offline_output_root_for_input(effective_data_path)
    )
    return completed, resolved_output_root


def find_best_matching_offline_frame(output_root: str | Path, window: SampleWindow) -> OfflineFrameMatch:
    try:
        from applications.usrp_wideband_signal_detection.infocom_evals.signal_detection_experiments.offline_cuda_detector_eval_review_helpers import (
            build_run_context,
        )
    except ModuleNotFoundError:
        from offline_cuda_detector_eval_review_helpers import build_run_context

    run_context = build_run_context(Path(output_root).expanduser().resolve())
    best_match: OfflineFrameMatch | None = None
    for frame_row in run_context["manifest_rows"]:
        frame_start = int(frame_row.get("global_sample_start", frame_row["file_offset_complex"]))
        frame_stop = int(frame_row.get("global_frame_end_sample", frame_row["frame_end_complex"]))
        overlap_start = max(frame_start, window.start_sample)
        overlap_stop = min(frame_stop, window.stop_sample)
        overlap_samples = max(0, overlap_stop - overlap_start)
        candidate = OfflineFrameMatch(
            frame_number=int(frame_row["frame_number"]),
            overlap_samples=overlap_samples,
            frame_start_sample=frame_start,
            frame_stop_sample=frame_stop,
        )
        if best_match is None or candidate.overlap_samples > best_match.overlap_samples:
            best_match = candidate

    if best_match is None:
        raise ValueError(f"No offline frames found under {output_root}")
    return best_match


def summarize_bundle(bundle: SigMFBundle, window: SampleWindow | None = None) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "data_path": str(bundle.data_path),
        "meta_path": str(bundle.meta_path),
        "datatype": bundle.datatype,
        "sample_rate_hz": bundle.sample_rate_hz,
        "center_frequency_hz": bundle.center_frequency_hz,
        "capture_sample_start": bundle.capture_sample_start,
        "total_complex_samples": bundle.total_complex_samples,
        "annotation_count": len(bundle.annotations),
    }
    if window is not None:
        summary.update(
            {
                "window_start_sample": window.start_sample,
                "window_stop_sample": window.stop_sample,
                "window_sample_count": window.sample_count,
                "window_annotation_index": window.annotation_index,
            }
        )
    return summary


def print_sigmf_summary(bundle: SigMFBundle, window: SampleWindow | None = None) -> dict[str, Any]:
    summary = summarize_bundle(bundle, window=window)
    for key, value in summary.items():
        print(f"{key}: {value}")
    return summary


def show_sigmf_spectrogram(
    bundle: SigMFBundle,
    window: SampleWindow,
    fft_size: int = DEFAULT_FFT_SIZE,
    hop_size: int = DEFAULT_HOP_SIZE,
    dynamic_range_db: float = DEFAULT_DYNAMIC_RANGE_DB,
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
) -> tuple[Any, Any, dict[str, Any]]:
    spectrogram_frame = build_spectrogram_frame(bundle, window, fft_size=fft_size, hop_size=hop_size)
    overlays = build_annotation_overlays(bundle, window)

    vmax = float(np.nanmax(spectrogram_frame.power_db))
    vmin = vmax - float(dynamic_range_db)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
        vmin, vmax = np.percentile(spectrogram_frame.power_db, [5.0, 99.5])

    fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
    ax.imshow(
        spectrogram_frame.power_db,
        origin="lower",
        aspect="auto",
        extent=spectrogram_frame.extent,
        cmap="magma",
        vmin=float(vmin),
        vmax=float(vmax),
    )
    ax.set_title(
        f"SigMF Spectrogram | {bundle.data_path.name} | samples [{window.start_sample}, {window.stop_sample})"
    )
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Frequency (MHz)")

    for overlay in overlays:
        color = KIND_COLORS.get(overlay["kind"], "magenta")
        rect = Rectangle(
            (float(overlay["overlap_start"]), float(overlay["freq_lower_hz"] / 1.0e6)),
            max(1.0, float(overlay["overlap_stop"] - overlay["overlap_start"])),
            float((overlay["freq_upper_hz"] - overlay["freq_lower_hz"]) / 1.0e6),
            fill=False,
            linewidth=2.0,
            edgecolor=color,
        )
        ax.add_patch(rect)
        label_text = f"{overlay['label']} #{overlay['annotation_index']}"
        text = ax.text(
            float(overlay["overlap_start"]),
            float(overlay["freq_upper_hz"] / 1.0e6),
            label_text,
            color=color,
            fontsize=8,
            va="bottom",
            ha="left",
        )
        text.set_path_effects([patheffects.withStroke(linewidth=3, foreground="black")])

    render_context = {
        "summary": summarize_bundle(bundle, window=window),
        "overlay_count": len(overlays),
        "overlays": overlays,
        "fft_size": fft_size,
        "hop_size": hop_size,
        "dynamic_range_db": dynamic_range_db,
        "spectrogram_frame": spectrogram_frame,
    }
    print(f"Rendered overlays: {len(overlays)}")
    return fig, ax, render_context


def show_detector_mask_comparison(
    bundle: SigMFBundle,
    window: SampleWindow,
    detector_mask: np.ndarray | None = None,
    detector_mask_path: str | Path | None = None,
    fft_size: int = DEFAULT_FFT_SIZE,
    hop_size: int = DEFAULT_HOP_SIZE,
    dynamic_range_db: float = DEFAULT_DYNAMIC_RANGE_DB,
    figsize: tuple[float, float] = (20.0, 6.0),
) -> tuple[Any, Any, dict[str, Any]]:
    spectrogram_frame = build_spectrogram_frame(bundle, window, fft_size=fft_size, hop_size=hop_size)
    gt_mask = build_ground_truth_mask(bundle, window, spectrogram_frame)
    overlays = build_annotation_overlays(bundle, window)
    resolved_detector_mask = detector_mask
    if resolved_detector_mask is None and detector_mask_path is not None:
        resolved_detector_mask = load_detector_mask(detector_mask_path)
    if resolved_detector_mask is not None and resolved_detector_mask.shape != gt_mask.shape:
        raise ValueError(
            f"Detector mask shape {resolved_detector_mask.shape} does not match GT grid {gt_mask.shape}"
        )

    vmax = float(np.nanmax(spectrogram_frame.power_db))
    vmin = vmax - float(dynamic_range_db)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
        vmin, vmax = np.percentile(spectrogram_frame.power_db, [5.0, 99.5])

    fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True, sharex=True, sharey=True)
    titles = ["GT Boxes", "Ground Truth Mask", "Detector vs GT"]
    for ax, title in zip(axes, titles):
        ax.imshow(
            spectrogram_frame.power_db,
            origin="lower",
            aspect="auto",
            extent=spectrogram_frame.extent,
            cmap="magma",
            vmin=float(vmin),
            vmax=float(vmax),
        )
        ax.set_title(title)
        ax.set_xlabel("Sample Index")
    axes[0].set_ylabel("Frequency (MHz)")

    for overlay in overlays:
        color = KIND_COLORS.get(overlay["kind"], "magenta")
        rect = Rectangle(
            (float(overlay["overlap_start"]), float(overlay["freq_lower_hz"] / 1.0e6)),
            max(1.0, float(overlay["overlap_stop"] - overlay["overlap_start"])),
            float((overlay["freq_upper_hz"] - overlay["freq_lower_hz"]) / 1.0e6),
            fill=False,
            linewidth=2.0,
            edgecolor=color,
        )
        axes[0].add_patch(rect)

    gt_overlay = np.where(gt_mask > 0, 1.0, np.nan)
    axes[1].imshow(
        gt_overlay,
        origin="lower",
        aspect="auto",
        extent=spectrogram_frame.extent,
        cmap="Blues",
        alpha=0.45,
        interpolation="nearest",
    )

    axes[2].imshow(
        gt_overlay,
        origin="lower",
        aspect="auto",
        extent=spectrogram_frame.extent,
        cmap="Blues",
        alpha=0.35,
        interpolation="nearest",
    )
    iou = None
    if resolved_detector_mask is not None:
        detector_overlay = np.where(np.asarray(resolved_detector_mask) > 0, 1.0, np.nan)
        axes[2].imshow(
            detector_overlay,
            origin="lower",
            aspect="auto",
            extent=spectrogram_frame.extent,
            cmap="Greens",
            alpha=0.35,
            interpolation="nearest",
        )
        iou = compute_mask_iou(gt_mask, resolved_detector_mask)
        axes[2].set_title(f"Detector vs GT | IoU={iou:.4f}")
    else:
        axes[2].text(0.5, 0.5, "No detector mask loaded", transform=axes[2].transAxes, ha="center", va="center", color="white")

    context = {
        "spectrogram_frame": spectrogram_frame,
        "ground_truth_mask": gt_mask,
        "detector_mask": resolved_detector_mask,
        "iou": iou,
        "overlays": overlays,
        "fft_size": fft_size,
        "hop_size": hop_size,
    }
    return fig, axes, context