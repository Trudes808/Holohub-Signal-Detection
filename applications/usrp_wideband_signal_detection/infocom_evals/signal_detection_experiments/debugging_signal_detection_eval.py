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
DEFAULT_COHERENT_OFFLINE_OUTPUT_ROOT = Path(
    "/home/sat3737/holohub-dev/applications/usrp_wideband_signal_detection/generated_inputs/coherent_power_offline"
)
DEFAULT_RUN_DEMO_CONTAINER_PATH = Path(
    "/home/sat3737/holohub-dev/applications/usrp_wideband_signal_detection/run_demo_container.sh"
)
DEFAULT_CONTAINER_NAME = "usrp_x410_signal_detection_demo"
DEFAULT_CONTAINER_BUILD_APP_DIR = (
    "/workspace/holohub/build/usrp_wideband_signal_detection/applications/usrp_wideband_signal_detection"
)
DEFAULT_CONTAINER_BUILD_APP_DIR_FALLBACK = (
    "/workspace/holohub-dev/build/usrp_wideband_signal_detection/applications/usrp_wideband_signal_detection"
)
DEFAULT_COHERENT_VALIDATOR_NAME = "offline_coherent_power_validator"
HOST_SCRATCH_ROOT = Path("/tmp/usrp_spectrograms")
HOST_COHERENT_SNAPSHOT_ROOT = Path("/tmp/coherent_power_snapshots")
CONTAINER_SCRATCH_ROOT = Path("/workspace/spectrograms")
CONTAINER_COHERENT_SNAPSHOT_ROOT = Path("/workspace/coherent_power_snapshots")
DEFAULT_WINDOWED_SIGMF_ROOT = Path(
    "/home/sat3737/holohub-dev/applications/usrp_wideband_signal_detection/generated_inputs"
)
DEFAULT_OFFLINE_FRAME_SAMPLE_COUNT = 512 * 10240
REPO_ROOT = Path(__file__).resolve().parents[4]
CONTAINER_REPO_ROOT = Path("/workspace/holohub")
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
    detector_type: str
    output_root: Path
    chunk_summary: dict[str, Any]
    validation_summary: dict[str, Any]
    chunk_arrays: dict[str, np.ndarray]
    global_arrays: dict[str, np.ndarray]


@dataclass(frozen=True)
class OfflineSavedMasks:
    detector_type: str
    output_root: Path
    detector_mask: np.ndarray | None
    ground_truth_mask: np.ndarray | None
    context: dict[str, Any]


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

    # Axes convention: frequency on X, time (sample index) on Y. Saved masks are native
    # (time, freq) == (rows, cols), which is already what imshow wants for freq-on-X / time-on-Y,
    # so no transpose here; the spectrogram (power_db is (freq, time)) is transposed instead.
    display_gt_mask = None if resolved_gt_mask is None else np.asarray(resolved_gt_mask)
    display_detector_mask = None if resolved_detector_mask is None else np.asarray(resolved_detector_mask)

    vmax = float(np.nanmax(spectrogram_frame.power_db))
    vmin = vmax - float(dynamic_range_db)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
        vmin, vmax = np.percentile(spectrogram_frame.power_db, [5.0, 99.5])

    # spectrogram_frame.extent is [sample_start, sample_stop, freq_low_mhz, freq_high_mhz];
    # swap it to [freq_low, freq_high, sample_start, sample_stop] for freq-on-X / time-on-Y.
    se = spectrogram_frame.extent
    spectrogram_extent_fx = [se[2], se[3], se[0], se[1]]

    fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True, sharex=True, sharey=True)
    titles = ["SigMF GT Boxes", "Saved Offline GT Mask", "Saved Offline Detector vs GT"]
    for ax, title in zip(axes, titles):
        ax.imshow(
            spectrogram_frame.power_db.T,
            origin="lower",
            aspect="auto",
            extent=spectrogram_extent_fx,
            cmap="magma",
            vmin=float(vmin),
            vmax=float(vmax),
        )
        ax.set_title(title)
        ax.set_xlabel("Frequency (MHz)")
    axes[0].set_ylabel("Sample Index")

    for overlay in overlays:
        color = KIND_COLORS.get(overlay["kind"], "magenta")
        rect = Rectangle(
            (float(overlay["freq_lower_hz"] / 1.0e6), float(overlay["overlap_start"])),
            float((overlay["freq_upper_hz"] - overlay["freq_lower_hz"]) / 1.0e6),
            max(1.0, float(overlay["overlap_stop"] - overlay["overlap_start"])),
            fill=False,
            linewidth=2.0,
            edgecolor=color,
        )
        axes[0].add_patch(rect)

    offline_extent = [
        float((bundle.center_frequency_hz - 0.5 * bundle.sample_rate_hz) / 1.0e6),
        float((bundle.center_frequency_hz + 0.5 * bundle.sample_rate_hz) / 1.0e6),
        float(window.start_sample),
        float(window.stop_sample),
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

    # Frequency on X, time (sample index) on Y. Masks are native (time, freq) == (rows, cols),
    # which is already imshow-ready for freq-on-X / time-on-Y, so no transpose.
    display_gt_mask = None if resolved_gt_mask is None else np.asarray(resolved_gt_mask)
    display_detector_mask = None if resolved_detector_mask is None else np.asarray(resolved_detector_mask)
    offline_extent = [
        float((bundle.center_frequency_hz - 0.5 * bundle.sample_rate_hz) / 1.0e6),
        float((bundle.center_frequency_hz + 0.5 * bundle.sample_rate_hz) / 1.0e6),
        float(window.start_sample),
        float(window.stop_sample),
    ]

    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True, sharex=True, sharey=True)
    titles = ["Saved Offline GT Binary Mask", "Saved Offline Detector Binary Mask"]
    masks = [display_gt_mask, display_detector_mask]
    cmaps = ["Blues", "Greens"]
    for ax, title, mask, cmap in zip(axes, titles, masks, cmaps):
        ax.set_title(title)
        ax.set_xlabel("Frequency (MHz)")
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
    axes[0].set_ylabel("Sample Index")

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
        relative = None
    if relative is not None:
        candidate = HOST_SCRATCH_ROOT / relative
        return candidate.resolve()

    try:
        coherent_relative = resolved.relative_to(CONTAINER_COHERENT_SNAPSHOT_ROOT)
    except ValueError:
        coherent_relative = None
    if coherent_relative is not None:
        candidate = HOST_COHERENT_SNAPSHOT_ROOT / coherent_relative
        return candidate.resolve()

    try:
        repo_relative = resolved.relative_to(CONTAINER_REPO_ROOT)
    except ValueError:
        return resolved.resolve()
    candidate = REPO_ROOT / repo_relative
    return candidate.resolve()


def load_json_artifact(path: str | Path) -> dict[str, Any]:
    resolved_path = resolve_offline_artifact_path(path)
    return json.loads(resolved_path.read_text(encoding="utf-8"))


def build_missing_debug_artifacts_message(output_root: Path) -> str:
    validation_summary_path = output_root / "offline_validation_summary.json"
    return (
        "Offline detector debug artifacts were not found under "
        f"{output_root}. Expected file: {validation_summary_path}. "
        "Rerun notebook cell 8 so the helper regenerates the offline inputs and prints fresh manual commands, "
        "then rerun the printed command and finally rerun notebook cell 10."
    )


def load_offline_detector_debug_artifacts(output_root: str | Path) -> OfflineDetectorDebugArtifacts:
    resolved_output_root = Path(output_root).expanduser().resolve()
    validation_summary_path = resolved_output_root / "offline_validation_summary.json"
    if not validation_summary_path.exists():
        # The offline run now goes through the batch-eval binary (run_offline_cuda_detector_eval),
        # which writes offline_eval_summary.json + per-frame artifacts, NOT the reference-validator's
        # offline_validation_summary.json with its corrected_full/chunk debug arrays. So these
        # validator-only debug/diagnostic cells no longer apply. Use the mask-comparison and
        # cell-6-style panel cells above for the faithful batch-equivalent spot check.
        if (resolved_output_root / "offline_eval_summary.json").exists():
            raise FileNotFoundError(
                "This is a batch-eval-pathway run (offline_eval_summary.json present), which does not "
                "produce the reference-validator debug arrays these cells expect. Skip these two "
                "coherent debug cells; the mask-comparison and panel cells above already show the "
                f"faithful batch-equivalent result. (looked under {resolved_output_root})"
            )
        raise FileNotFoundError(build_missing_debug_artifacts_message(resolved_output_root))

    validation_summary = load_json_artifact(validation_summary_path)
    if "chunk_debug_summary_json" in validation_summary:
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
            key: np.asarray(np.load(resolve_offline_artifact_path(chunk_summary[key])))
            for key in chunk_array_keys
        }
        # Optional keys introduced later; tolerate older artifact dirs that lack them.
        for optional_key in ("dino_enhanced_input_npy",):
            if optional_key in chunk_summary:
                try:
                    chunk_arrays[optional_key] = np.asarray(
                        np.load(resolve_offline_artifact_path(chunk_summary[optional_key]))
                    )
                except (OSError, ValueError):
                    pass
        global_arrays = {
            key: np.asarray(np.load(resolve_offline_artifact_path(validation_summary[key])))
            for key in global_array_keys
        }
        return OfflineDetectorDebugArtifacts(
            detector_type="cuda_dino",
            output_root=resolved_output_root,
            chunk_summary=chunk_summary,
            validation_summary=validation_summary,
            chunk_arrays=chunk_arrays,
            global_arrays=global_arrays,
        )

    chunk_arrays: dict[str, np.ndarray] = {}
    global_arrays: dict[str, np.ndarray] = {}
    final_mask_path = validation_summary.get("final_mask_npy")
    final_mask = None
    if final_mask_path:
        final_mask = np.asarray(np.load(resolve_offline_artifact_path(final_mask_path)))
        global_arrays["final_mask_npy"] = final_mask
    power_db_path = validation_summary.get("power_db_npy")
    if power_db_path:
        chunk_arrays["power_db_full_npy"] = np.asarray(np.load(resolve_offline_artifact_path(power_db_path)))
    else:
        tensor_snapshot_path = validation_summary.get("tensor_snapshot_path")
        if tensor_snapshot_path:
            tensor_snapshot = np.asarray(np.load(resolve_offline_artifact_path(tensor_snapshot_path)))
            chunk_arrays["power_db_full_npy"] = (
                10.0 * np.log10(np.abs(tensor_snapshot).astype(np.float32) ** 2 + 1.0e-12)
            ).astype(np.float32)
    corrected_path = validation_summary.get("corrected_sxx_db_npy")
    if corrected_path:
        corrected_full = np.asarray(np.load(resolve_offline_artifact_path(corrected_path)))
        chunk_arrays["corrected_full_npy"] = corrected_full
        snapshot_metadata = None
        metadata_path = validation_summary.get("metadata_path")
        if metadata_path:
            snapshot_metadata = load_json_artifact(metadata_path)
        snapshot_metadata = snapshot_metadata or {}
        target_rows = int(snapshot_metadata.get("input_height", corrected_full.shape[0]) or corrected_full.shape[0])
        target_cols = int(snapshot_metadata.get("input_width", corrected_full.shape[1]) or corrected_full.shape[1])
        chunk_plan = build_coherent_chunk_plan(snapshot_metadata)
        selected_chunk_index = select_coherent_chunk_index(
            chunk_plan,
            final_mask,
            validation_summary.get("selected_chunk_index"),
        )
        selected_chunk = (
            chunk_plan[selected_chunk_index]
            if chunk_plan
            else {"chunk_index": 0, "row_start": 0, "row_stop": corrected_full.shape[0]}
        )
        row_start = int(selected_chunk["row_start"])
        row_stop = int(selected_chunk["row_stop"])
        corrected_chunk = corrected_full[row_start:row_stop, :]
        if corrected_chunk.size == 0:
            corrected_chunk = corrected_full
            row_start = 0
            row_stop = corrected_full.shape[0]
        chunk_arrays["corrected_chunk_npy"] = corrected_chunk
        chunk_arrays["corrected_resized_npy"] = resize_array_bilinear(
            corrected_chunk,
            output_rows=target_rows,
            output_cols=target_cols,
        )
        validation_summary["selected_chunk_index"] = int(selected_chunk_index)
        validation_summary["chunk_count"] = int(len(chunk_plan)) if chunk_plan else 1
        validation_summary["selected_chunk_row_start"] = int(row_start)
        validation_summary["selected_chunk_row_stop"] = int(row_stop)
    merged_score_path = validation_summary.get("merged_score_npy")
    if merged_score_path:
        global_arrays["projected_grouped_score_npy"] = np.asarray(
            np.load(resolve_offline_artifact_path(merged_score_path))
        )
    merged_power_path = validation_summary.get("merged_power_npy")
    if merged_power_path:
        global_arrays["merged_power_npy"] = np.asarray(
            np.load(resolve_offline_artifact_path(merged_power_path))
        )
    merged_coherence_path = validation_summary.get("merged_coherence_npy")
    if merged_coherence_path:
        global_arrays["merged_coherence_npy"] = np.asarray(
            np.load(resolve_offline_artifact_path(merged_coherence_path))
        )
    raw_projected_path = validation_summary.get("raw_projected_mask_npy")
    if raw_projected_path:
        global_arrays["projected_grouped_mask_npy"] = np.asarray(
            np.load(resolve_offline_artifact_path(raw_projected_path))
        )
    return OfflineDetectorDebugArtifacts(
        detector_type="coherent_power",
        output_root=resolved_output_root,
        chunk_summary={},
        validation_summary=validation_summary,
        chunk_arrays=chunk_arrays,
        global_arrays=global_arrays,
    )


def _show_stage_image(
    ax: Any,
    image: np.ndarray | None,
    title: str,
    cmap: str,
    binary: bool = False,
    transpose_if_tall: bool = True,
) -> None:
    if image is None:
        ax.set_title(title)
        ax.set_xlabel("Time bins")
        ax.set_ylabel("Freq bins")
        ax.text(0.5, 0.5, "Not available", transform=ax.transAxes, ha="center", va="center")
        return
    data = np.asarray(image)
    display = data.T if transpose_if_tall and data.shape[0] > data.shape[1] else data
    if binary:
        ax.imshow((display > 0).astype(np.uint8), origin="lower", aspect="auto", cmap=cmap, vmin=0, vmax=1)
    else:
        finite = np.asarray(data, dtype=np.float32)
        vmin, vmax = np.percentile(finite, [5.0, 99.5]) if finite.size else (0.0, 1.0)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
            vmin, vmax = float(np.nanmin(finite)), float(np.nanmax(finite))
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
                vmin, vmax = 0.0, 1.0
        ax.imshow(np.asarray(display, dtype=np.float32), origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("Time bins")
    ax.set_ylabel("Freq bins")


def show_offline_detector_debug_pathways(
    debug_artifacts: OfflineDetectorDebugArtifacts,
    figsize: tuple[float, float] = (18.0, 10.0),
) -> tuple[Any, Any, dict[str, Any]]:
    transpose_if_tall = debug_artifacts.detector_type != "coherent_power"
    if debug_artifacts.detector_type == "coherent_power":
        panels = [
            ("power_db_full_npy", "Pre-Chunk Power Spectrogram", "magma", False),
            ("corrected_full_npy", "Pre-Chunk Corrected Spectrogram", "magma", False),
            ("merged_coherence_npy", "Merged Coherence", "cividis", False),
            ("merged_power_npy", "Merged Power", "viridis", False),
            ("projected_grouped_score_npy", "Merged Score", "inferno", False),
            ("projected_grouped_mask_npy", "Projected Raw Mask", "gray", True),
            ("final_mask_npy", "Final Mask", "gray", True),
        ]
    else:
        panels = [
            ("corrected_resized_npy", "Corrected Spectrogram", "magma", False),
            ("dino_enhanced_input_npy", "DINO Enhanced Input", "magma", False),
            ("dino_score_raw_npy", "DINO Raw Score", "viridis", False),
            ("dino_score_raw_deweighted_npy", "DINO Deweighted Score", "viridis", False),
            ("coherence_gate_npy", "Coherence Gate", "cividis", False),
            ("hybrid_keep_freq_npy", "Pathway A: Keep Freq", "plasma", False),
            ("hybrid_keep_res_npy", "Pathway B: Keep Residual", "plasma", False),
            ("combined_score_npy", "Combined Score", "inferno", False),
            ("projected_grouped_score_npy", "Projected Combined Score", "inferno", False),
        ]
    ncols = 4
    nrows = max(1, (len(panels) + ncols - 1) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize[0], figsize[1] / 2.0 * nrows),
                             constrained_layout=True)
    flat_axes = list(np.atleast_1d(axes).flat)
    for ax, (key, title, cmap, binary) in zip(flat_axes, panels):
        source = debug_artifacts.chunk_arrays if key in debug_artifacts.chunk_arrays else debug_artifacts.global_arrays
        _show_stage_image(ax, source.get(key), title, cmap, binary=binary, transpose_if_tall=transpose_if_tall)
    for ax in flat_axes[len(panels):]:
        ax.axis("off")

    context = {
        "detector_type": debug_artifacts.detector_type,
        "selected_chunk_index": debug_artifacts.validation_summary.get("selected_chunk_index"),
        "chunk_count": debug_artifacts.validation_summary.get("chunk_count", 1),
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
    transpose_if_tall = debug_artifacts.detector_type != "coherent_power"
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
        _show_stage_image(ax, source.get(key), title, cmap, binary=binary, transpose_if_tall=transpose_if_tall)

    detector_pixels = None if saved_detector_mask is None else int(np.count_nonzero(np.asarray(saved_detector_mask) > 0))
    gt_pixels = None if saved_gt_mask is None else int(np.count_nonzero(np.asarray(saved_gt_mask) > 0))
    context = {
        "detector_type": debug_artifacts.detector_type,
        "selected_chunk_index": debug_artifacts.validation_summary.get("selected_chunk_index"),
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
    sample_edges = axis_edges_from_centers(absolute_sample_axis)
    frequency_edges_hz = axis_edges_from_centers(absolute_frequency_axis_hz)
    extent = [
        float(sample_edges[0]),
        float(sample_edges[-1]),
        float(frequency_edges_hz[0] / 1.0e6),
        float(frequency_edges_hz[-1] / 1.0e6),
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


def normalize_annotation_frequency_edges_hz(
    bundle: SigMFBundle,
    freq_lower_hz: float,
    freq_upper_hz: float,
) -> tuple[float, float]:
    lower = float(freq_lower_hz)
    upper = float(freq_upper_hz)
    if lower > upper:
        lower, upper = upper, lower

    passband_lower = bundle.center_frequency_hz - 0.5 * bundle.sample_rate_hz
    passband_upper = bundle.center_frequency_hz + 0.5 * bundle.sample_rate_hz
    direct_overlaps = upper > passband_lower and lower < passband_upper

    shifted_lower = lower + bundle.center_frequency_hz
    shifted_upper = upper + bundle.center_frequency_hz
    shifted_overlaps = shifted_upper > passband_lower and shifted_lower < passband_upper

    if not direct_overlaps and shifted_overlaps:
        return shifted_lower, shifted_upper
    return lower, upper


def build_annotation_overlays(bundle: SigMFBundle, window: SampleWindow) -> list[dict[str, Any]]:
    overlays: list[dict[str, Any]] = []
    for annotation_index, annotation in enumerate(bundle.annotations):
        if not annotation_overlaps_window(annotation, window):
            continue

        annotation_start = int(annotation.get("core:sample_start", 0) or 0)
        annotation_stop = annotation_start + int(annotation.get("core:sample_count", 0) or 0)
        overlap_start = max(window.start_sample, annotation_start)
        overlap_stop = min(window.stop_sample, annotation_stop)
        freq_lower_hz, freq_upper_hz = normalize_annotation_frequency_edges_hz(
            bundle,
            float(annotation.get("core:freq_lower_edge", 0.0) or 0.0),
            float(annotation.get("core:freq_upper_edge", 0.0) or 0.0),
        )
        overlays.append(
            {
                "annotation_index": annotation_index,
                "sample_start": annotation_start,
                "sample_stop": annotation_stop,
                "overlap_start": overlap_start,
                "overlap_stop": overlap_stop,
                "freq_lower_hz": freq_lower_hz,
                "freq_upper_hz": freq_upper_hz,
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


def resize_array_bilinear(image: np.ndarray, output_rows: int, output_cols: int) -> np.ndarray:
    source = np.asarray(image, dtype=np.float32)
    if source.ndim != 2:
        raise ValueError(f"Expected a 2D array to resize, got shape {source.shape}")

    input_rows, input_cols = source.shape
    target_rows = max(1, int(output_rows))
    target_cols = max(1, int(output_cols))
    if input_rows == target_rows and input_cols == target_cols:
        return source.copy()

    row_positions = ((np.arange(target_rows, dtype=np.float32) + 0.5) * input_rows / target_rows) - 0.5
    col_positions = ((np.arange(target_cols, dtype=np.float32) + 0.5) * input_cols / target_cols) - 0.5

    row0 = np.clip(np.floor(row_positions).astype(np.int64), 0, input_rows - 1)
    col0 = np.clip(np.floor(col_positions).astype(np.int64), 0, input_cols - 1)
    row1 = np.clip(row0 + 1, 0, input_rows - 1)
    col1 = np.clip(col0 + 1, 0, input_cols - 1)
    row_lerp = np.clip(row_positions - row0, 0.0, 1.0).astype(np.float32)
    col_lerp = np.clip(col_positions - col0, 0.0, 1.0).astype(np.float32)

    top_left = source[row0[:, None], col0[None, :]]
    top_right = source[row0[:, None], col1[None, :]]
    bottom_left = source[row1[:, None], col0[None, :]]
    bottom_right = source[row1[:, None], col1[None, :]]
    top = top_left + (top_right - top_left) * col_lerp[None, :]
    bottom = bottom_left + (bottom_right - bottom_left) * col_lerp[None, :]
    return top + (bottom - top) * row_lerp[:, None]


def build_coherent_chunk_plan(snapshot_metadata: dict[str, Any]) -> list[dict[str, int]]:
    rows = int(snapshot_metadata.get("rows", 0) or 0)
    cols = int(snapshot_metadata.get("cols", 0) or 0)
    if rows <= 0 or cols <= 0:
        return []

    config = snapshot_metadata.get("config", {}) or {}
    resolution_hz = float(snapshot_metadata.get("resolution_hz", 0.0) or 0.0)
    ignore_bins_per_side = int(snapshot_metadata.get("ignore_bins_per_side", 0) or 0)
    ignore_bins_per_side = max(0, min(ignore_bins_per_side, max(0, (rows - 16) // 2)))
    valid_rows = list(range(ignore_bins_per_side, max(ignore_bins_per_side, rows - ignore_bins_per_side)))
    if len(valid_rows) < 16:
        return []

    chunk_bandwidth_hz = float(config.get("chunk_bandwidth_hz", 0.0) or 0.0)
    chunk_overlap_hz = float(config.get("chunk_overlap_hz", 0.0) or 0.0)
    uncalibrated_chunk_fraction = float(config.get("uncalibrated_chunk_fraction", 1.0) or 1.0)
    uncalibrated_overlap_fraction = float(config.get("uncalibrated_overlap_fraction", 0.0) or 0.0)
    if resolution_hz > 0.0 and chunk_bandwidth_hz > 0.0:
        chunk_rows = int(np.clip(np.rint(chunk_bandwidth_hz / resolution_hz), 16, rows))
        overlap_rows = int(np.clip(np.rint(chunk_overlap_hz / resolution_hz), 0, max(0, chunk_rows - 1)))
    else:
        chunk_rows = int(np.clip(np.rint(len(valid_rows) * uncalibrated_chunk_fraction), 16, len(valid_rows)))
        overlap_rows = int(np.clip(np.rint(chunk_rows * uncalibrated_overlap_fraction), 0, max(0, chunk_rows - 1)))

    step_rows = max(1, chunk_rows - overlap_rows)
    chunks: list[dict[str, int]] = []
    chunk_index = 0
    for start_index in range(0, len(valid_rows), step_rows):
        stop_index = min(start_index + chunk_rows, len(valid_rows))
        if stop_index - start_index < 16:
            if chunks:
                break
            continue
        row_start = valid_rows[start_index]
        row_stop = valid_rows[stop_index - 1] + 1
        chunks.append({"chunk_index": chunk_index, "row_start": int(row_start), "row_stop": int(row_stop)})
        chunk_index += 1
        if stop_index >= len(valid_rows):
            break
    return chunks


def select_coherent_chunk_index(
    chunk_plan: list[dict[str, int]],
    final_mask: np.ndarray | None,
    requested_chunk_index: int | None = None,
) -> int:
    if not chunk_plan:
        return 0
    if requested_chunk_index is not None:
        return max(0, min(int(requested_chunk_index), len(chunk_plan) - 1))

    if final_mask is not None and np.asarray(final_mask).ndim == 2:
        mask = np.asarray(final_mask)
        best_index = 0
        best_score = -1
        for index, chunk in enumerate(chunk_plan):
            row_start = int(chunk["row_start"])
            row_stop = int(chunk["row_stop"])
            score = int(np.count_nonzero(mask[row_start:row_stop, :] > 0))
            if score > best_score:
                best_score = score
                best_index = index
        if best_score > 0:
            return best_index

    return min(len(chunk_plan) // 2, len(chunk_plan) - 1)


def load_detector_mask(mask_path: str | Path) -> np.ndarray:
    mask = np.asarray(np.load(resolve_offline_artifact_path(mask_path)))
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D detector mask, got shape {mask.shape}")
    return (mask > 0).astype(np.uint8)


def parse_simple_yaml_scalar(raw_value: str) -> Any:
    text = raw_value.strip()
    if not text:
        return None
    if text.startswith(("\"", "'")) and text.endswith(("\"", "'")):
        return text[1:-1]
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        return float(text) if any(char in text for char in (".", "e", "E")) else int(text)
    except ValueError:
        return text


def parse_simple_yaml_sections(config_text: str, section_names: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    sections = {name: {} for name in section_names}
    current_section: str | None = None
    for raw_line in config_text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if not line.startswith(" "):
            current_section = line[:-1] if line.endswith(":") else None
            continue
        if current_section not in sections or not line.startswith("  ") or line.startswith("    "):
            continue
        stripped = line.strip()
        if stripped.startswith("- ") or ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        parsed_value = parse_simple_yaml_scalar(value)
        if parsed_value is not None:
            sections[current_section][key.strip()] = parsed_value
    return sections


def resolve_runtime_fft_config(
    fft_section: dict[str, Any],
    explicit_span_hz: float | None = None,
    packet_samples: int = 1024,
) -> dict[str, Any]:
    reference_span_hz = max(1.0, float(fft_section.get("reference_span_hz", fft_section.get("span", 500.0e6)) or 500.0e6))
    reference_fft_size = max(1, int(fft_section.get("reference_fft_size", fft_section.get("transform_points", 20480)) or 20480))
    override_fft_bin_size_hz = max(0.0, float(fft_section.get("override_fft_bin_size", 0.0) or 0.0))
    configured_span_hz = max(1.0, float(fft_section.get("span", reference_span_hz) or reference_span_hz))
    active_span_hz = configured_span_hz
    if explicit_span_hz is not None and np.isfinite(explicit_span_hz) and explicit_span_hz > 0.0:
        active_span_hz = float(explicit_span_hz)

    target_bin_size_hz = reference_span_hz / float(max(1, reference_fft_size))
    requested_fft_size = float(reference_fft_size)
    if override_fft_bin_size_hz > 0.0:
        target_bin_size_hz = override_fft_bin_size_hz
        requested_fft_size = active_span_hz / override_fft_bin_size_hz
    else:
        span_ratio = active_span_hz / reference_span_hz
        if np.isfinite(span_ratio) and span_ratio > 0.0:
            snapped_ratio = float(np.exp2(np.round(np.log2(span_ratio))))
            requested_fft_size = float(reference_fft_size) * snapped_ratio

    packet_samples = max(1, int(packet_samples))
    requested_fft_size_int = max(1, int(np.rint(requested_fft_size)))
    num_packets_per_fft = max(1, int(np.rint(float(requested_fft_size_int) / float(packet_samples))))
    actual_fft_size = max(packet_samples, num_packets_per_fft * packet_samples)
    resolution_hz = float(active_span_hz) / float(max(1, actual_fft_size))
    return {
        "reference_span_hz": reference_span_hz,
        "reference_fft_size": reference_fft_size,
        "active_span_hz": float(active_span_hz),
        "target_bin_size_hz": target_bin_size_hz,
        "override_fft_bin_size_hz": override_fft_bin_size_hz,
        "requested_fft_size": requested_fft_size_int,
        "actual_fft_size": actual_fft_size,
        "packet_samples": packet_samples,
        "num_packets_per_fft": num_packets_per_fft,
        "resolution_hz": resolution_hz,
    }


def read_detector_type_from_config(config_path: str | Path) -> str:
    resolved_config_path = Path(config_path).expanduser().resolve()
    config_text = resolved_config_path.read_text(encoding="utf-8")
    sections = parse_simple_yaml_sections(config_text, ("pipeline",))
    detector_type = str(sections.get("pipeline", {}).get("detector_type", "cuda_dino")).strip()
    return detector_type or "cuda_dino"


def map_host_path_to_container(path: str | Path) -> Path:
    resolved_path = Path(path).expanduser().resolve()
    try:
        relative = resolved_path.relative_to(HOST_SCRATCH_ROOT)
    except ValueError:
        relative = None
    if relative is not None:
        return CONTAINER_SCRATCH_ROOT / relative

    try:
        coherent_relative = resolved_path.relative_to(HOST_COHERENT_SNAPSHOT_ROOT)
    except ValueError:
        coherent_relative = None
    if coherent_relative is not None:
        return CONTAINER_COHERENT_SNAPSHOT_ROOT / coherent_relative

    try:
        repo_relative = resolved_path.relative_to(REPO_ROOT)
    except ValueError as error:
        raise ValueError(f"Path is not mounted into the container: {resolved_path}") from error
    return CONTAINER_REPO_ROOT / repo_relative


def detector_grid_frequency_edges_hz(bundle: SigMFBundle, frequency_bin_count: int) -> np.ndarray:
    return np.linspace(
        bundle.center_frequency_hz - 0.5 * bundle.sample_rate_hz,
        bundle.center_frequency_hz + 0.5 * bundle.sample_rate_hz,
        int(frequency_bin_count) + 1,
        dtype=np.float64,
    )


def detector_grid_time_edges(window: SampleWindow, time_bin_count: int) -> np.ndarray:
    return np.linspace(
        float(window.start_sample),
        float(window.stop_sample),
        int(time_bin_count) + 1,
        dtype=np.float64,
    )


def build_detector_grid_ground_truth_mask(
    bundle: SigMFBundle,
    window: SampleWindow,
    time_bin_count: int,
    frequency_bin_count: int,
) -> np.ndarray:
    time_edges = detector_grid_time_edges(window, time_bin_count)
    frequency_edges_hz = detector_grid_frequency_edges_hz(bundle, frequency_bin_count)
    mask = np.zeros((int(time_bin_count), int(frequency_bin_count)), dtype=np.uint8)

    for overlay in build_annotation_overlays(bundle, window):
        time_active = (time_edges[:-1] < float(overlay["overlap_stop"])) & (
            time_edges[1:] > float(overlay["overlap_start"])
        )
        frequency_active = (frequency_edges_hz[:-1] < float(overlay["freq_upper_hz"])) & (
            frequency_edges_hz[1:] > float(overlay["freq_lower_hz"])
        )
        if np.any(time_active) and np.any(frequency_active):
            mask[np.ix_(time_active, frequency_active)] = 1
    return mask


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


def default_coherent_offline_artifact_root(data_path: str | Path, window: SampleWindow | None = None) -> Path:
    if window is not None:
        stem = build_windowed_sigmf_stem(data_path, window)
    else:
        resolved_data_path = Path(data_path).expanduser().resolve()
        stem = resolved_data_path.name
        if stem.endswith(".sigmf-data"):
            stem = stem[: -len(".sigmf-data")]
        else:
            stem = resolved_data_path.stem
    return DEFAULT_COHERENT_OFFLINE_OUTPUT_ROOT / stem


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


def build_coherent_validator_input_tensor(
    bundle: SigMFBundle,
    window: SampleWindow,
    fft_size: int,
) -> np.ndarray:
    if fft_size <= 0:
        raise ValueError("fft_size must be positive")

    iq_samples = read_complex_samples(bundle, window)
    frame_count = int(np.ceil(iq_samples.size / float(fft_size)))
    if frame_count <= 0:
        raise ValueError("Window does not contain enough samples for coherent validation")

    padded_sample_count = frame_count * fft_size
    if padded_sample_count != iq_samples.size:
        padded_samples = np.zeros(padded_sample_count, dtype=np.complex64)
        padded_samples[: iq_samples.size] = iq_samples.astype(np.complex64, copy=False)
    else:
        padded_samples = iq_samples.astype(np.complex64, copy=False)

    frames = padded_samples.reshape(frame_count, fft_size)
    spectrum = np.fft.fftshift(np.fft.fft(frames, axis=1), axes=1)
    return np.asarray(spectrum.T, dtype=np.complex64)


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
    detector_type: str | None = None,
    dry_run: bool = False,
) -> list[str]:
    resolved_data_path = Path(data_path).expanduser().resolve()
    resolved_wrapper_path = Path(wrapper_path).expanduser().resolve()
    resolved_config_path = Path(config_path).expanduser().resolve()
    resolved_output_root = (
        Path(output_root).expanduser().resolve() if output_root is not None else default_offline_output_root_for_input(resolved_data_path)
    )
    # Mirror run_batch_offline_eval.run_one: pass --detector so the wrapper (and the real
    # run_offline_cuda_detector_eval binary) runs the matching operator. Default cuda_dino
    # in the wrapper would otherwise ignore a coherent_power config.
    resolved_detector_type = detector_type or read_detector_type_from_config(resolved_config_path)
    command = [
        sys.executable,
        str(resolved_wrapper_path),
        str(resolved_data_path),
        "--detector",
        str(resolved_detector_type),
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


def build_coherent_validator_inner_command(
    metadata_path: str | Path,
    output_root: str | Path,
) -> str:
    container_metadata_path = str(map_host_path_to_container(metadata_path))
    container_output_root = str(map_host_path_to_container(output_root))
    return (
        "set -euo pipefail; "
        f"validator_name={shlex.quote(DEFAULT_COHERENT_VALIDATOR_NAME)}; "
        "validator_bin=''; "
        f"for candidate in {shlex.quote(DEFAULT_CONTAINER_BUILD_APP_DIR)}/$validator_name {shlex.quote(DEFAULT_CONTAINER_BUILD_APP_DIR_FALLBACK)}/$validator_name; do "
        "if [[ -x \"$candidate\" ]]; then validator_bin=\"$candidate\"; break; fi; "
        "done; "
        "if [[ -z \"$validator_bin\" ]]; then echo \"offline_coherent_power_validator not found in the container.\" >&2; exit 1; fi; "
        f"rm -rf {shlex.quote(container_output_root)}; mkdir -p {shlex.quote(container_output_root)}; "
        f"\"$validator_bin\" --snapshot-json {shlex.quote(container_metadata_path)} --output-dir {shlex.quote(container_output_root)}"
    )


def build_coherent_validator_command(
    metadata_path: str | Path,
    output_root: str | Path,
) -> list[str]:
    return [
        "sudo",
        "docker",
        "exec",
        "-i",
        DEFAULT_CONTAINER_NAME,
        "bash",
        "-lc",
        build_coherent_validator_inner_command(metadata_path, output_root),
    ]


def build_coherent_validator_manual_commands(
    metadata_path: str | Path,
    output_root: str | Path,
) -> list[str]:
    prep_command = (
        f"cd {shlex.quote(str(DEFAULT_RUN_DEMO_CONTAINER_PATH.parent))} && "
        f"{shlex.quote(str(DEFAULT_RUN_DEMO_CONTAINER_PATH))}"
    )
    run_command = " ".join(
        shlex.quote(part) for part in build_coherent_validator_command(metadata_path, output_root)
    )
    return [prep_command, run_command]


def export_coherent_offline_inputs(
    bundle: SigMFBundle,
    window: SampleWindow,
    config_path: str | Path,
    output_root: str | Path | None = None,
    fft_size: int = DEFAULT_FFT_SIZE,
    hop_size: int = DEFAULT_HOP_SIZE,
) -> tuple[Path, Path, Path, Path]:
    resolved_config_path = Path(config_path).expanduser().resolve()
    config_text = resolved_config_path.read_text(encoding="utf-8")
    sections = parse_simple_yaml_sections(
        config_text,
        ("pipeline", "fft", "chdr_converter", "coherent_power_signal_detector"),
    )
    detector_type = str(sections.get("pipeline", {}).get("detector_type", "")).strip()
    if detector_type != "coherent_power":
        raise ValueError(f"Expected a coherent_power config, got detector_type={detector_type!r}")

    artifact_root = (
        Path(output_root).expanduser().resolve()
        if output_root is not None
        else default_coherent_offline_artifact_root(bundle.data_path, window=window).resolve()
    )
    artifact_root.mkdir(parents=True, exist_ok=True)
    validator_output_root = artifact_root / "operator_live_validator"

    fft_section = sections.get("fft", {})
    coherent_section = sections.get("coherent_power_signal_detector", {})
    packet_samples = int(
        sections.get("chdr_converter", {}).get("num_complex_samples_per_packet", 1024)
        if "chdr_converter" in sections
        else 1024
    )
    fft_runtime = resolve_runtime_fft_config(
        fft_section,
        explicit_span_hz=bundle.sample_rate_hz,
        packet_samples=packet_samples,
    )
    fft_size_config = int(fft_runtime["actual_fft_size"])
    if fft_size_config <= 0:
        raise ValueError(f"{resolved_config_path} is missing fft.transform_points or fft.burst_size")

    tensor = build_coherent_validator_input_tensor(bundle, window, fft_size=fft_size_config)
    tensor_path = artifact_root / "input_tensor.npy"
    np.save(tensor_path, tensor)

    input_height = int(coherent_section.get("input_height", tensor.shape[1]))
    input_width = int(coherent_section.get("input_width", tensor.shape[0]))
    span_hz = float(fft_runtime["active_span_hz"])
    resolution_hz = float(fft_runtime["resolution_hz"])

    gt_mask = build_detector_grid_ground_truth_mask(
        bundle,
        window,
        time_bin_count=tensor.shape[1],
        frequency_bin_count=tensor.shape[0],
    ).T
    gt_mask_path = artifact_root / "ground_truth_mask.npy"
    np.save(gt_mask_path, gt_mask)

    metadata = {
        "rows": int(tensor.shape[0]),
        "cols": int(tensor.shape[1]),
        "original_input_rows": int(tensor.shape[0]),
        "original_input_cols": int(tensor.shape[1]),
        "input_height": input_height,
        "input_width": input_width,
        "resolution_hz": resolution_hz,
        "sample_rate_hz": span_hz,
        "span_hz": span_hz,
        "center_frequency_hz": bundle.center_frequency_hz,
        "frequency_axis_calibrated": True,
        "ignore_bins_per_side": int(np.ceil(float(coherent_section.get("ignore_sideband_hz", 0.0) or 0.0) / resolution_hz)) if resolution_hz > 0.0 else 0,
        "fast_performance": True,
        "path_mode_effective": "fast_performance",
        "pipeline_variant": "notebook_offline_export",
        "tensor_axis_order": "frequency_time",
        "tensor_snapshot_path": str(map_host_path_to_container(tensor_path)),
        "power_db_snapshot_path": None,
        "mask_path": None,
        "reference_debug_artifacts": None,
        "config": {
            key: value
            for key, value in coherent_section.items()
            if key in {
                "fast_performance",
                "chunk_bandwidth_hz",
                "chunk_overlap_hz",
                "uncalibrated_chunk_fraction",
                "uncalibrated_overlap_fraction",
                "ignore_sideband_percent",
                "ignore_sideband_hz",
                "frontend_row_q",
                "frontend_reference_q",
                "frontend_smooth_sigma",
                "frontend_max_boost_db",
                "frontend_signal_cap_db",
                "coherence_weight",
                "power_weight",
                "power_assist_mode",
                "power_floor_time_q",
                "power_floor_global_q",
                "power_excess_start_db",
                "power_excess_full_db",
                "power_local_blend",
                "coherence_source_mode",
                "coherence_gate_start",
                "coherence_gate_full",
                "coherence_bridge_bias",
                "coherence_power_joint_weight",
                "score_threshold_mode",
                "fixed_score_threshold",
                "coherence_power_support_q",
                "coherence_power_q",
                "min_component_size",
                "filter_detection_mask",
                "grouping_seed_score_q",
                "grouping_bridge_freq_px",
                "grouping_bridge_time_px",
                "grouping_min_component_size",
                "grouping_min_freq_span_px",
                "grouping_min_time_span_px",
                "grouping_min_density",
                "grouping_time_continuity_ratio",
                # Fast-performance path + per-frequency floor fill. Required for the offline
                # validator to reproduce a fast/perfreq config (e.g.
                # config_coherent_power_perf_perfreq_single_channel.yaml); without these the
                # validator falls back to the slow reference path with zeroed thresholds.
                "fast_power_floor_db",
                "fast_power_span_db",
                "fast_score_threshold",
                "fast_background_freq_radius",
                "fast_background_time_radius",
                "fast_mask_smooth_iterations",
                "per_freq_threshold_enable",
                "per_freq_threshold_mode",
                "per_freq_threshold_path",
                "per_freq_threshold_offset_db",
            }
        },
    }
    metadata["config"].setdefault("power_assist_mode", "local")
    metadata["config"].setdefault("coherence_source_mode", "merged")
    metadata["config"]["input_height"] = input_height
    metadata["config"]["input_width"] = input_width
    metadata_path = artifact_root / "coherent_power_input_snapshot.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    compare_context = {
        "detector_type": "coherent_power",
        "ground_truth_mask_npy": str(gt_mask_path),
        "tensor_snapshot_path": str(tensor_path),
        "metadata_json": str(metadata_path),
        "validator_output_root": str(validator_output_root),
        "window_start_sample": window.start_sample,
        "window_stop_sample": window.stop_sample,
        "runtime_fft_size": fft_size_config,
        "runtime_span_hz": span_hz,
        "runtime_resolution_hz": resolution_hz,
    }
    (artifact_root / "offline_compare_context.json").write_text(
        json.dumps(compare_context, indent=2) + "\n",
        encoding="utf-8",
    )
    return tensor_path, metadata_path, gt_mask_path, validator_output_root


def run_offline_coherent_power_file(
    data_path: str | Path,
    config_path: str | Path,
    output_root: str | Path | None = None,
    bundle: SigMFBundle | None = None,
    window: SampleWindow | None = None,
    dry_run: bool = False,
) -> tuple[subprocess.CompletedProcess[str], Path]:
    if window is None:
        raise ValueError("A window is required for coherent offline validation from the notebook helper")

    effective_bundle = bundle if bundle is not None else load_sigmf_bundle(data_path)
    _, metadata_path, _, validator_output_root = export_coherent_offline_inputs(
        effective_bundle,
        window,
        config_path=config_path,
        output_root=output_root,
    )
    manual_commands = build_coherent_validator_manual_commands(metadata_path, validator_output_root)
    if dry_run:
        completed = subprocess.CompletedProcess(
            args=manual_commands,
            returncode=0,
            stdout=(
                "Prepared coherent-power offline inputs. Run these commands manually:\n\n"
                + "\n".join(manual_commands)
            ),
            stderr="",
        )
        return completed, validator_output_root

    prep_completed = subprocess.run(
        [str(DEFAULT_RUN_DEMO_CONTAINER_PATH)],
        capture_output=True,
        text=True,
        check=False,
    )
    if prep_completed.returncode != 0:
        return prep_completed, validator_output_root

    completed = subprocess.run(
        build_coherent_validator_command(metadata_path, validator_output_root),
        capture_output=True,
        text=True,
        check=False,
    )
    combined_stdout = (prep_completed.stdout or "") + (completed.stdout or "")
    combined_stderr = (prep_completed.stderr or "") + (completed.stderr or "")
    return (
        subprocess.CompletedProcess(
            args=completed.args,
            returncode=completed.returncode,
            stdout=combined_stdout,
            stderr=combined_stderr,
        ),
        validator_output_root,
    )


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
    # Both detectors (cuda_dino AND coherent_power) go through the SAME real-operator binary
    # (run_offline_cuda_detector_eval, via the wrapper) that run_batch_offline_eval.py drives,
    # just over a SigMF cropped to the requested section. This makes the offline run a faithful
    # single-file/single-section equivalent of the batch eval, instead of the separate
    # reference-only offline_coherent_power_validator (which does not implement the live/fast
    # coherent path and produced masks that did not match the batch).
    detector_type = read_detector_type_from_config(config_path)

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
        detector_type=detector_type,
        dry_run=dry_run,
    )
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    resolved_output_root = (
        Path(output_root).expanduser().resolve()
        if output_root is not None
        else default_offline_output_root_for_input(effective_data_path)
    )
    return completed, resolved_output_root


def run_offline_detector_file(
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
    return run_offline_cuda_dino_file(
        data_path,
        config_path=config_path,
        output_root=output_root,
        wrapper_path=wrapper_path,
        bundle=bundle,
        window=window,
        windowed_sigmf_root=windowed_sigmf_root,
        target_chunk_count=target_chunk_count,
        debug_chunk_index=debug_chunk_index,
        dry_run=dry_run,
    )


def load_offline_saved_masks(
    output_root: str | Path,
    window: SampleWindow,
    bundle: SigMFBundle | None = None,
) -> OfflineSavedMasks:
    resolved_output_root = Path(output_root).expanduser().resolve()
    cuda_summary_path = resolved_output_root / "offline_eval_summary.json"
    cuda_manifest_path = resolved_output_root / "frame_manifest.csv"
    if cuda_summary_path.exists() and cuda_manifest_path.exists():
        try:
            from applications.usrp_wideband_signal_detection.infocom_evals.signal_detection_experiments.offline_cuda_detector_eval_review_helpers import (
                build_run_context,
                load_detector_mask as load_saved_detector_mask,
                load_ground_truth_mask,
            )
        except ModuleNotFoundError:
            from offline_cuda_detector_eval_review_helpers import (
                build_run_context,
                load_detector_mask as load_saved_detector_mask,
                load_ground_truth_mask,
            )

        matching_frame = find_best_matching_offline_frame(resolved_output_root, window)
        run_context = build_run_context(resolved_output_root)
        frame_row = next(
            row for row in run_context["manifest_rows"] if int(row["frame_number"]) == matching_frame.frame_number
        )
        detector_mask = load_saved_detector_mask(frame_row)
        gt_mask = load_ground_truth_mask(frame_row)
        return OfflineSavedMasks(
            detector_type="cuda_dino",
            output_root=resolved_output_root,
            detector_mask=detector_mask,
            ground_truth_mask=gt_mask,
            context={
                "frame_number": matching_frame.frame_number,
                "overlap_samples": matching_frame.overlap_samples,
            },
        )

    coherent_summary_path = resolved_output_root / "offline_validation_summary.json"
    if not coherent_summary_path.exists():
        raise FileNotFoundError(f"No supported offline artifact summary was found under {resolved_output_root}")

    validation_summary = load_json_artifact(coherent_summary_path)
    detector_mask = None
    final_mask_path = validation_summary.get("final_mask_npy")
    if final_mask_path:
        detector_mask = load_detector_mask(resolve_offline_artifact_path(final_mask_path))

    compare_context_path = resolved_output_root.parent / "offline_compare_context.json"
    compare_context = (
        json.loads(compare_context_path.read_text(encoding="utf-8")) if compare_context_path.exists() else {}
    )
    gt_mask = None
    gt_mask_path = compare_context.get("ground_truth_mask_npy")
    if gt_mask_path:
        gt_mask = load_detector_mask(gt_mask_path)
    if (
        bundle is not None
        and detector_mask is not None
        and (gt_mask is None or gt_mask.shape != detector_mask.shape)
    ):
        gt_mask = build_detector_grid_ground_truth_mask(
            bundle,
            window,
            time_bin_count=detector_mask.shape[1],
            frequency_bin_count=detector_mask.shape[0],
        ).T

    return OfflineSavedMasks(
        detector_type="coherent_power",
        output_root=resolved_output_root,
        detector_mask=detector_mask,
        ground_truth_mask=gt_mask,
        context={
            **compare_context,
            "grouped_box_count": validation_summary.get("grouped_box_count"),
            "final_mask_fill_ratio": validation_summary.get("final_mask_fill_ratio"),
        },
    )


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