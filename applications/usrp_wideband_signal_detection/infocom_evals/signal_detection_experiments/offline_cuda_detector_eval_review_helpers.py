from __future__ import annotations

import csv
import json
from pathlib import Path
from pprint import pprint
from typing import Any

import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np

KNOWN_RELATIVE_DIR = Path("applications/usrp_wideband_signal_detection/infocom_evals/signal_detection_experiments")
DEFAULT_OUTPUT_ROOT = Path("/tmp/usrp_spectrograms/offline_cuda_detector_eval/attenuation_dB_0")
DEFAULT_FRAME_NUMBER = None
DEFAULT_FIRST_N_FRAMES = 5
DEFAULT_FIGSIZE = (24, 5.5)
GT_MASK_OVERLAY_CMAP = ListedColormap(["#00b7ff"])
DETECTOR_MASK_OVERLAY_CMAP = ListedColormap(["#00ff00"])
KIND_COLORS = {
    "waveform": "lime",
    "metadata": "deepskyblue",
    "zadoff_chu": "gold",
    "annotation": "magenta",
}

FrameRow = dict[str, Any]
RunContext = dict[str, Any]


def resolve_notebook_dir(start_dir: Path | None = None) -> Path:
    cwd = (start_dir or Path.cwd()).resolve()
    candidate = (cwd / KNOWN_RELATIVE_DIR).resolve()
    if candidate.exists():
        return candidate
    return cwd


def load_summary(output_root: Path) -> tuple[dict[str, Any], Path]:
    summary_path = output_root / "offline_eval_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing offline eval summary: {summary_path}")
    return json.loads(summary_path.read_text()), summary_path


def load_manifest(output_root: Path) -> tuple[list[FrameRow], Path]:
    manifest_path = output_root / "frame_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing frame manifest: {manifest_path}")

    rows: list[FrameRow] = []
    with manifest_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Manifest is missing a header row: {manifest_path}")

        required_columns = {
            "channel",
            "frame_number",
            "file_offset_complex",
            "data_end_complex",
            "frame_end_complex",
            "complex_samples_read",
            "complex_samples_padded",
            "partial_frame",
            "fft_rows",
            "fft_cols",
            "preview_rows",
            "preview_cols",
            "spectrogram_preview_pgm",
            "spectrogram_tensor_npy",
            "mask_preview_pgm",
            "mask_npy",
            "gt_annotations_json",
            "gt_mask_npy",
        }
        missing_columns = sorted(required_columns.difference(reader.fieldnames))
        if missing_columns:
            raise ValueError(
                "Manifest is missing required columns "
                f"{missing_columns}; clear the output root and rerun the offline eval."
            )

        for raw_row in reader:
            def artifact_path(key: str) -> Path | None:
                value = raw_row.get(key, "")
                value = value.strip() if value is not None else ""
                return output_root / value if value else None

            row: FrameRow = {
                "channel": int(raw_row["channel"]),
                "frame_number": int(raw_row["frame_number"]),
                "file_offset_complex": int(raw_row["file_offset_complex"]),
                "data_end_complex": int(raw_row["data_end_complex"]),
                "frame_end_complex": int(raw_row["frame_end_complex"]),
                "complex_samples_read": int(raw_row["complex_samples_read"]),
                "complex_samples_padded": int(raw_row["complex_samples_padded"]),
                "partial_frame": raw_row["partial_frame"].strip().lower() == "true",
                "fft_rows": int(raw_row["fft_rows"]),
                "fft_cols": int(raw_row["fft_cols"]),
                "preview_rows": int(raw_row["preview_rows"]),
                "preview_cols": int(raw_row["preview_cols"]),
                "spectrogram_preview_path": artifact_path("spectrogram_preview_pgm"),
                "spectrogram_tensor_path": artifact_path("spectrogram_tensor_npy"),
                "mask_preview_path": artifact_path("mask_preview_pgm"),
                "mask_npy_path": artifact_path("mask_npy"),
                "gt_annotations_path": artifact_path("gt_annotations_json"),
                "gt_mask_npy_path": artifact_path("gt_mask_npy"),
            }
            for optional_int_key in (
                "global_sample_start",
                "global_data_end_sample",
                "global_frame_end_sample",
                "local_file_offset_complex",
                "local_data_end_complex",
                "local_frame_end_complex",
                "capture_sample_start",
                "samples_per_row",
            ):
                value = raw_row.get(optional_int_key, "")
                if value is not None and value.strip():
                    row[optional_int_key] = int(value)
            row["frame_span_samples"] = row["complex_samples_read"] + row["complex_samples_padded"]
            rows.append(row)
    return rows, manifest_path


def load_spectrogram_background(frame_row: FrameRow) -> np.ndarray:
    tensor_path = frame_row.get("spectrogram_tensor_path")
    if tensor_path is not None and tensor_path.exists():
        tensor = np.asarray(np.load(tensor_path))
        if np.iscomplexobj(tensor):
            return 10.0 * np.log10(np.abs(tensor) ** 2 + 1.0e-12)
        return tensor.astype(np.float32)

    preview_path = frame_row.get("spectrogram_preview_path")
    if preview_path is not None and preview_path.exists():
        preview = np.asarray(plt.imread(preview_path), dtype=np.float32)
        if preview.ndim == 3:
            preview = preview[..., 0]
        return preview

    raise FileNotFoundError(f"No spectrogram artifact found for frame {frame_row['frame_number']}")


def load_binary_mask(mask_path: Path | None) -> np.ndarray | None:
    if mask_path is None or not mask_path.exists():
        return None
    mask = np.asarray(np.load(mask_path))
    return (mask > 0).astype(np.float32)


def load_detector_mask(frame_row: FrameRow) -> np.ndarray | None:
    mask = load_binary_mask(frame_row.get("mask_npy_path"))
    if mask is not None:
        return mask

    preview_path = frame_row.get("mask_preview_path")
    if preview_path is not None and preview_path.exists():
        preview = np.asarray(plt.imread(preview_path), dtype=np.float32)
        if preview.ndim == 3:
            preview = preview[..., 0]
        return (preview > 0).astype(np.float32)

    return None


def load_ground_truth_annotations(frame_row: FrameRow) -> list[dict[str, Any]]:
    gt_annotations_path = frame_row.get("gt_annotations_path")
    if gt_annotations_path is None or not gt_annotations_path.exists():
        return []
    payload = json.loads(gt_annotations_path.read_text())
    items = payload.get("items", [])
    return items if isinstance(items, list) else []


def load_ground_truth_mask(frame_row: FrameRow) -> np.ndarray | None:
    return load_binary_mask(frame_row.get("gt_mask_npy_path"))


def orient_panel_for_display(frame_row: FrameRow, panel: np.ndarray) -> np.ndarray:
    panel = np.asarray(panel)
    if panel.ndim != 2:
        raise ValueError(f"Expected a 2D panel, got shape {panel.shape}")

    candidate_shapes = [
        (int(frame_row.get("fft_rows", 0)), int(frame_row.get("fft_cols", 0))),
        (int(frame_row.get("preview_rows", 0)), int(frame_row.get("preview_cols", 0))),
    ]

    for time_bins, freq_bins in candidate_shapes:
        if time_bins <= 0 or freq_bins <= 0:
            continue
        if panel.shape == (time_bins, freq_bins):
            return panel.T
        if panel.shape == (freq_bins, time_bins):
            return panel

    return panel.T


def background_limits(background: np.ndarray) -> tuple[float, float]:
    vmin, vmax = np.percentile(background, [5.0, 99.5])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin = float(np.nanmin(background))
        vmax = float(np.nanmax(background))
    if vmax <= vmin:
        vmax = vmin + 1.0
    return vmin, vmax


def mask_support(mask: np.ndarray | None) -> dict[str, int] | None:
    if mask is None:
        return None

    binary = np.asarray(mask) > 0.0
    if binary.ndim != 2:
        raise ValueError(f"Expected a 2D binary mask, got shape {binary.shape}")

    time_bins, freq_bins = binary.shape
    active_pixels = int(binary.sum())
    active_time = np.flatnonzero(binary.any(axis=1))
    active_freq = np.flatnonzero(binary.any(axis=0))

    support = {
        "time_bins": int(time_bins),
        "freq_bins": int(freq_bins),
        "active_pixels": active_pixels,
        "active_time_bins": int(active_time.size),
        "active_freq_bins": int(active_freq.size),
    }
    if active_time.size:
        support["time_start"] = int(active_time[0])
        support["time_stop"] = int(active_time[-1] + 1)
    if active_freq.size:
        support["freq_start"] = int(active_freq[0])
        support["freq_stop"] = int(active_freq[-1] + 1)
    return support


def describe_mask_support(label: str, support: dict[str, int] | None) -> str:
    if support is None:
        return f"{label}: unavailable"
    if support["active_pixels"] == 0:
        return (
            f"{label}: empty mask in a {support['time_bins']} x {support['freq_bins']}"
            " time-by-frequency grid"
        )

    return (
        f"{label}: pixels={support['active_pixels']} | "
        f"time bins={support['active_time_bins']}/{support['time_bins']} "
        f"[{support['time_start']}, {support['time_stop']}) | "
        f"freq bins={support['active_freq_bins']}/{support['freq_bins']} "
        f"[{support['freq_start']}, {support['freq_stop']})"
    )


def draw_mask_edges(ax: Any,
                    display_mask: np.ndarray | None,
                    extent: list[float],
                    color: str,
                    linewidth: float,
                    label: str | None = None) -> None:
    if display_mask is None:
        return

    contour_mask = np.asarray(display_mask) > 0.0
    if contour_mask.ndim != 2 or not np.any(contour_mask):
        return

    contour = ax.contour(
        contour_mask.astype(np.float32),
        levels=[0.5],
        colors=[color],
        linewidths=linewidth,
        origin="lower",
        extent=extent,
    )
    if label is not None:
        if hasattr(contour, "collections"):
            for collection in contour.collections:
                collection.set_label(label)
        else:
            contour.set_label(label)


def draw_ground_truth_boxes(ax: Any, overlays: list[dict[str, Any]]) -> None:
    for overlay in overlays:
        color = KIND_COLORS.get(overlay.get("kind", "annotation"), "magenta")
        x_start = float(overlay["overlap_sample_start"])
        x_stop = float(overlay["overlap_sample_stop"])
        rect = Rectangle(
            (x_start, float(overlay["y_mhz"])),
            max(0.0, x_stop - x_start),
            float(overlay["height_mhz"]),
            fill=False,
            linewidth=2.0,
            edgecolor=color,
        )
        ax.add_patch(rect)
        text = ax.text(
            x_start,
            float(overlay["y_mhz"]) + float(overlay["height_mhz"]),
            str(overlay.get("label", "UNLABELED")),
            color=color,
            fontsize=8,
            va="bottom",
            ha="left",
        )
        text.set_path_effects([patheffects.withStroke(linewidth=3, foreground="black")])


def build_run_context(output_root: Path) -> RunContext:
    summary, summary_path = load_summary(output_root)
    manifest_rows, manifest_path = load_manifest(output_root)
    sample_rate_hz = float(summary.get("input_sample_rate_hz", 0.0))
    span_hz = float(summary.get("span_hz", 0.0))
    if sample_rate_hz <= 0.0:
        raise ValueError("offline_eval_summary.json is missing input_sample_rate_hz")
    if span_hz <= 0.0:
        span_hz = sample_rate_hz

    frames_with_saved_gt = [
        frame_row["frame_number"]
        for frame_row in manifest_rows
        if frame_row.get("gt_annotations_path") is not None and frame_row["gt_annotations_path"].exists()
    ]

    return {
        "output_root": output_root,
        "summary": summary,
        "summary_path": summary_path,
        "manifest_rows": manifest_rows,
        "manifest_path": manifest_path,
        "sample_rate_hz": sample_rate_hz,
        "span_hz": span_hz,
        "frames_with_saved_gt": frames_with_saved_gt,
    }


def frame_by_number(run_context: RunContext, frame_number: int) -> FrameRow:
    for frame_row in run_context["manifest_rows"]:
        if frame_row["frame_number"] == frame_number:
            return frame_row
    raise KeyError(f"Frame {frame_number} is not present in the manifest.")


def channel_frame_numbers(run_context: RunContext, channel: int = 0) -> list[int]:
    return [
        frame_row["frame_number"]
        for frame_row in run_context["manifest_rows"]
        if frame_row["channel"] == channel
    ]


def frame_has_visible_gt_content(frame_row: FrameRow) -> bool:
    if load_ground_truth_annotations(frame_row):
        return True

    gt_mask = load_ground_truth_mask(frame_row)
    if gt_mask is not None and bool(np.any(gt_mask > 0.0)):
        return True

    return False


def frames_with_saved_gt_content(run_context: RunContext, channel: int = 0) -> list[int]:
    frames: list[int] = []
    for frame_row in run_context["manifest_rows"]:
        if frame_row["channel"] != channel:
            continue
        if frame_has_visible_gt_content(frame_row):
            frames.append(frame_row["frame_number"])
    return frames


def choose_default_frame_number(run_context: RunContext,
                                frame_number: int | None = None,
                                channel: int = 0) -> int:
    if frame_number is not None:
        return frame_number

    gt_candidate_frames = frames_with_saved_gt_content(run_context, channel=channel)
    if gt_candidate_frames:
        return gt_candidate_frames[0]

    available_frames = channel_frame_numbers(run_context, channel=channel)
    if not available_frames:
        raise ValueError(f"No frames found for channel {channel}")
    return available_frames[0]


def choose_first_frame_numbers(run_context: RunContext,
                               first_n_frames: int = DEFAULT_FIRST_N_FRAMES,
                               channel: int = 0) -> list[int]:
    if first_n_frames <= 0:
        return []

    ordered_frames: list[int] = []
    seen_frames: set[int] = set()

    for frame_number in frames_with_saved_gt_content(run_context, channel=channel):
        if frame_number not in seen_frames:
            seen_frames.add(frame_number)
            ordered_frames.append(frame_number)
        if len(ordered_frames) >= first_n_frames:
            return ordered_frames

    for frame_number in channel_frame_numbers(run_context, channel=channel):
        if frame_number not in seen_frames:
            seen_frames.add(frame_number)
            ordered_frames.append(frame_number)
        if len(ordered_frames) >= first_n_frames:
            break

    return ordered_frames


def plot_frame(frame_number: int,
               run_context: RunContext,
               figsize: tuple[float, float] = DEFAULT_FIGSIZE) -> tuple[Any, Any, FrameRow, list[dict[str, Any]]]:
    frame_row = frame_by_number(run_context, frame_number)
    background = orient_panel_for_display(frame_row, load_spectrogram_background(frame_row))
    detector_mask = load_detector_mask(frame_row)
    gt_mask = load_ground_truth_mask(frame_row)
    gt_overlays = load_ground_truth_annotations(frame_row)

    display_detector_mask = None if detector_mask is None else orient_panel_for_display(frame_row, detector_mask)
    display_gt_mask = None if gt_mask is None else orient_panel_for_display(frame_row, gt_mask)

    span_hz = run_context["span_hz"]
    uses_global_samples = "global_sample_start" in frame_row
    frame_start_sample = int(frame_row.get("global_sample_start", frame_row["file_offset_complex"]))
    data_end_sample = int(frame_row.get("global_data_end_sample", frame_row["data_end_complex"]))
    frame_end_sample = int(frame_row.get("global_frame_end_sample", frame_row["frame_end_complex"]))
    sample_axis_label = "Global SigMF Sample" if uses_global_samples else "Absolute Sample From File"
    freq_min_mhz = -0.5 * span_hz / 1.0e6
    freq_max_mhz = 0.5 * span_hz / 1.0e6
    extent = [frame_start_sample, frame_end_sample, freq_min_mhz, freq_max_mhz]

    vmin, vmax = background_limits(background)
    titles = [
        "Saved Spectrogram",
        "Saved GT Boxes",
        "Saved GT Mask",
        "Detector Final Mask (Merged Boxes)",
        "GT vs Detector Edges",
    ]
    fig, axes = plt.subplots(
        1,
        len(titles),
        figsize=(max(figsize[0], 6.0 * len(titles)), figsize[1]),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    for ax, title in zip(axes, titles):
        ax.imshow(
            background,
            aspect="auto",
            origin="lower",
            extent=extent,
            cmap="magma",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(title)
        ax.set_xlabel(sample_axis_label)
        if frame_row["partial_frame"]:
            ax.axvline(data_end_sample, color="white", linestyle="--", linewidth=1.0, alpha=0.9)

    axes[0].set_ylabel("Frequency (MHz)")
    draw_ground_truth_boxes(axes[1], gt_overlays)

    if display_gt_mask is not None:
        gt_overlay = np.where(display_gt_mask > 0.0, 1.0, np.nan)
        axes[2].imshow(
            gt_overlay,
            aspect="auto",
            origin="lower",
            extent=extent,
            cmap=GT_MASK_OVERLAY_CMAP,
            alpha=0.50,
            interpolation="nearest",
        )
    else:
        axes[2].text(0.5, 0.5, "No saved GT mask", transform=axes[2].transAxes, ha="center", va="center", color="white")

    if display_detector_mask is not None:
        detector_overlay = np.where(display_detector_mask > 0.0, 1.0, np.nan)
        axes[3].imshow(
            detector_overlay,
            aspect="auto",
            origin="lower",
            extent=extent,
            cmap=DETECTOR_MASK_OVERLAY_CMAP,
            alpha=0.55,
            interpolation="nearest",
        )
    else:
        axes[3].text(0.5, 0.5, "No saved detector mask", transform=axes[3].transAxes, ha="center", va="center", color="white")

    draw_ground_truth_boxes(axes[4], gt_overlays)
    draw_mask_edges(axes[4], display_gt_mask, extent, color="#00b7ff", linewidth=1.5, label="GT mask edge")
    draw_mask_edges(axes[4], display_detector_mask, extent, color="#00ff00", linewidth=1.1, label="Detector mask edge")
    axes[4].legend(
        handles=[
            Line2D([0], [0], color="#00b7ff", linewidth=1.5, label="GT mask edge"),
            Line2D([0], [0], color="#00ff00", linewidth=1.1, label="Detector mask edge"),
        ],
        loc="upper right",
        fontsize=8,
        framealpha=0.85,
    )

    fig.suptitle(
        f"Frame {frame_row['frame_number']} | {sample_axis_label.lower()} {frame_start_sample} | saved GT boxes {len(gt_overlays)}",
        fontsize=13,
    )
    return fig, axes, frame_row, gt_overlays


def print_run_summary(run_context: RunContext) -> list[FrameRow]:
    frames_with_saved_spectrogram = sum(
        1
        for frame_row in run_context["manifest_rows"]
        if frame_row.get("spectrogram_tensor_path") is not None or frame_row.get("spectrogram_preview_path") is not None
    )
    frames_with_saved_mask = sum(
        1
        for frame_row in run_context["manifest_rows"]
        if frame_row.get("mask_npy_path") is not None or frame_row.get("mask_preview_path") is not None
    )
    frames_with_gt_content = frames_with_saved_gt_content(run_context)

    print(f"Frames in manifest: {len(run_context['manifest_rows'])}")
    print(f"Frames with saved spectrogram artifacts: {frames_with_saved_spectrogram}")
    print(f"Frames with saved detector masks: {frames_with_saved_mask}")
    print(f"Frames with saved GT artifacts: {len(run_context['frames_with_saved_gt'])}")
    print(f"First frames with saved GT: {run_context['frames_with_saved_gt'][:10]}")
    print(f"Default GT-priority frames: {frames_with_gt_content[:10]}")

    if not run_context["frames_with_saved_gt"]:
        print("Current output root does not contain replay-saved GT artifacts yet.")
        print("Rebuild and rerun the offline eval to populate aligned GT outputs.")

    pprint(run_context["summary"])
    return run_context["manifest_rows"][:3]


def show_single_frame(run_context: RunContext,
                      frame_number: int | None = DEFAULT_FRAME_NUMBER,
                      channel: int = 0,
                      figsize: tuple[float, float] = DEFAULT_FIGSIZE) -> tuple[Any, Any, FrameRow, list[dict[str, Any]]]:
    selected_frame_number = choose_default_frame_number(run_context, frame_number=frame_number, channel=channel)
    fig, axes, plotted_frame, plotted_overlays = plot_frame(selected_frame_number, run_context, figsize=figsize)
    gt_support = mask_support(load_ground_truth_mask(plotted_frame))
    detector_support = mask_support(load_detector_mask(plotted_frame))

    print(f"Plotted frame {plotted_frame['frame_number']}")
    print(f"Spectrogram artifact: {plotted_frame['spectrogram_tensor_path'] or plotted_frame['spectrogram_preview_path']}")
    print(f"Detector mask artifact: {plotted_frame['mask_npy_path'] or plotted_frame['mask_preview_path']}")
    print(f"GT annotations artifact: {plotted_frame['gt_annotations_path']}")
    print(f"GT mask artifact: {plotted_frame['gt_mask_npy_path']}")
    print(f"Saved GT boxes in frame: {len(plotted_overlays)}")
    print(describe_mask_support("GT mask support", gt_support))
    print(describe_mask_support("Detector mask support", detector_support))
    if gt_support is not None and detector_support is not None and gt_support["active_pixels"] > 0:
        if detector_support["active_freq_bins"] > gt_support["active_freq_bins"] or detector_support["active_time_bins"] > gt_support["active_time_bins"]:
            print(
                "Detector mask is the final merged-box union. If it looks broader than the GT, that reflects detector merge behavior rather than a manifest or sample-axis offset."
            )
    plt.show()
    return fig, axes, plotted_frame, plotted_overlays


def show_first_frames(run_context: RunContext,
                      first_n_frames: int = DEFAULT_FIRST_N_FRAMES,
                      channel: int = 0,
                      figsize: tuple[float, float] = DEFAULT_FIGSIZE) -> list[int]:
    first_frame_numbers = choose_first_frame_numbers(run_context, first_n_frames=first_n_frames, channel=channel)
    for frame_number in first_frame_numbers:
        plot_frame(frame_number, run_context, figsize=figsize)
        plt.show()

    print(f"Rendered {len(first_frame_numbers)} frame(s): {first_frame_numbers}")
    return first_frame_numbers


__all__ = [
    "DEFAULT_FIGSIZE",
    "DEFAULT_FIRST_N_FRAMES",
    "DEFAULT_FRAME_NUMBER",
    "DEFAULT_OUTPUT_ROOT",
    "build_run_context",
    "choose_default_frame_number",
    "choose_first_frame_numbers",
    "frames_with_saved_gt_content",
    "print_run_summary",
    "resolve_notebook_dir",
    "show_first_frames",
    "show_single_frame",
]