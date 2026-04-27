#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_STAGE_KEYS = [
    "corrected_resized_npy",
    "dino_score_raw_npy",
    "dino_score_raw_deweighted_npy",
    "coherence_gate_npy",
    "combined_score_npy",
    "hybrid_filled_mask_npy",
    "hybrid_component_filtered_mask_npy",
    "grouped_mask_npy",
    "final_mask_npy",
    "final_mask_source_npy",
    "final_mask_projected_npy",
]


SUMMARY_STAGE_KEYS = [
    "projected_grouped_mask_npy",
    "projected_grouped_score_npy",
    "merged_box_mask_npy",
    "final_mask_npy",
]


TIMING_METRIC_MAP = [
    ("runtime_warmup", None, "runtime_warmup", 0),
    ("power_db_from_tensor", "power_db", "power_db_from_tensor", 0),
    ("frontend_correction", "frontend", "frontend_correction", 0),
    ("chunk_planning", "chunk_plan", "chunk_planning", 0),
    ("chunk_pack", "chunk_pack", None, 0),
    ("chunk_coherence", "coherence_batch", "chunk_coherence", 0),
    ("chunk_torch_runtime_batch", "runtime_batch", "chunk_torch_runtime_batch", 0),
    ("chunk_model_prep_batch", "runtime_model_prep", "chunk_model_prep_batch", 1),
    ("chunk_torch_forward_batch", "runtime_torch_forward", "chunk_torch_forward_batch", 1),
    ("chunk_dino_score_batch", "runtime_dino_score", "chunk_dino_score_batch", 1),
    ("chunk_score_projection", "raw_score_projection", "chunk_score_projection", 0),
    ("chunk_hybrid_support_batch", "hybrid_batch", "chunk_hybrid_support_batch", 0),
    ("hybrid_normalization", "hybrid_normalization", None, 1),
    ("hybrid_residual_stack", "hybrid_residual_stack", None, 1),
    ("hybrid_threshold_extract", "hybrid_threshold_extract", None, 1),
    ("hybrid_closing", "hybrid_closing", None, 1),
    ("hybrid_fill_holes", "hybrid_fill_holes", None, 1),
    ("hybrid_component_filter", "hybrid_component_filter", None, 1),
    ("hybrid_output_copy", "hybrid_output_copy", None, 1),
    ("debug_device_to_host", "debug_device_to_host", None, 0),
    ("global_merge", "global_merge", "global_merge", 0),
    ("artifact_serialization", "artifact_serialization", "artifact_serialization", 0),
    ("debug_chunk_rerun_total", None, "debug_chunk_rerun_total", 0),
    ("debug_artifact_serialization", None, "debug_artifact_serialization", 1),
]


def format_timing_label(label: str, indent_level: int) -> str:
    return f"{'  ' * max(indent_level, 0)}{label}"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_host_path(raw: str | Path) -> Path:
    path = Path(raw)
    text = str(path)
    if text.startswith("/workspace/spectrograms/"):
        return Path("/tmp/usrp_spectrograms") / text.removeprefix("/workspace/spectrograms/")
    if text == "/workspace/spectrograms":
        return Path("/tmp/usrp_spectrograms")
    if text.startswith("/workspace/dino_masks/"):
        return Path("/tmp/usrp_dino_masks") / text.removeprefix("/workspace/dino_masks/")
    if text == "/workspace/dino_masks":
        return Path("/tmp/usrp_dino_masks")
    if text.startswith("/workspace/holohub/"):
        return repo_root() / text.removeprefix("/workspace/holohub/")
    if text == "/workspace/holohub":
        return repo_root()
    if text.startswith("/workspace/holohub-dev/"):
        return repo_root() / text.removeprefix("/workspace/holohub-dev/")
    if text == "/workspace/holohub-dev":
        return repo_root()
    return path.expanduser().resolve()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def resolve_debug_summary(output_dir: Path) -> tuple[Path, dict]:
    manifest_path = output_dir / "cuda_artifact_manifest.json"
    if manifest_path.exists():
        manifest = load_json(manifest_path)
        summary_path = resolve_host_path(manifest["chunk_debug_summary_json"])
    else:
        summary_path = (output_dir / "chunk_debug" / "chunk_debug_summary.json").resolve()
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing chunk debug summary: {summary_path}")
    return summary_path, load_json(summary_path)


def resolve_validation_summary(output_dir: Path) -> tuple[Path, dict]:
    manifest_path = output_dir / "cuda_artifact_manifest.json"
    if manifest_path.exists():
        manifest = load_json(manifest_path)
        summary_key = "summary_json" if "summary_json" in manifest else "offline_validation_summary_json"
        summary_path = resolve_host_path(manifest.get(summary_key, output_dir / "offline_validation_summary.json"))
    else:
        summary_path = (output_dir / "offline_validation_summary.json").resolve()
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing validation summary: {summary_path}")
    return summary_path, load_json(summary_path)


def resolve_stage_profile(output_dir: Path) -> Path | None:
    stage_profile_path = (output_dir / "offline_stage_profile.json").resolve()
    if stage_profile_path.exists():
        return stage_profile_path
    return None


def load_stage_array(summary: dict, key: str) -> np.ndarray:
    path = resolve_host_path(str(summary.get(key, "") or ""))
    if not path.exists():
        raise FileNotFoundError(f"Missing artifact for {key}: {path}")
    return np.load(path, allow_pickle=False).astype(np.float32)


def load_stage_arrays(summary: dict, keys: Iterable[str]) -> list[tuple[str, np.ndarray]]:
    return [(key, load_stage_array(summary, key)) for key in common_stage_keys(summary, keys)]


def common_stage_keys(summary: dict, preferred: Iterable[str]) -> list[str]:
    keys = []
    for key in preferred:
        if key in summary and str(summary.get(key, "") or ""):
            keys.append(key)
    return keys


def require_stage_keys(summary: dict, preferred: Iterable[str], label: str) -> list[str]:
    preferred_list = list(preferred)
    keys = common_stage_keys(summary, preferred_list)
    if keys != preferred_list:
        missing = [key for key in preferred_list if key not in keys]
        raise SystemExit(f"{label} is missing requested stages: {missing}")
    return keys


def stage_title(key: str) -> str:
    title = key.removesuffix("_npy")
    return title.replace("_", " ")


def format_shape(shape: tuple[int, ...]) -> str:
    return "x".join(str(dim) for dim in shape)


def is_binaryish(array: np.ndarray) -> bool:
    if array.size == 0:
        return False
    unique = np.unique(array)
    return bool(np.all(np.isin(unique, [0.0, 1.0])))


def summarize_stage_pair(reference_array: np.ndarray, cuda_array: np.ndarray) -> dict[str, str]:
    summary = {
        "ref_shape": format_shape(reference_array.shape),
        "cuda_shape": format_shape(cuda_array.shape),
        "mean_abs": "n/a",
        "max_abs": "n/a",
        "overlap": "n/a",
    }
    if reference_array.shape != cuda_array.shape:
        return summary

    diff = np.abs(cuda_array.astype(np.float32) - reference_array.astype(np.float32))
    summary["mean_abs"] = f"{float(diff.mean()):.6f}"
    summary["max_abs"] = f"{float(diff.max(initial=0.0)):.6f}"

    if is_binaryish(reference_array) and is_binaryish(cuda_array):
        reference_mask = reference_array > 0.5
        cuda_mask = cuda_array > 0.5
        intersection = int(np.logical_and(reference_mask, cuda_mask).sum())
        union = int(np.logical_or(reference_mask, cuda_mask).sum())
        differing = int(np.logical_xor(reference_mask, cuda_mask).sum())
        iou = 1.0 if union == 0 else intersection / union
        summary["overlap"] = f"iou={iou:.4f} diff_px={differing}"
    return summary


def print_stage_summary(cuda_summary: dict,
                        reference_summary: dict,
                        cuda_arrays: list[tuple[str, np.ndarray]],
                        reference_arrays: list[tuple[str, np.ndarray]],
                        heading: str = "Stage parity summary:") -> None:
    print(heading)
    print(f"{'stage':<28} {'reference':>12} {'cuda':>12} {'mean_abs':>12} {'max_abs':>12} {'overlap':>24}")
    reference_map = dict(reference_arrays)
    cuda_map = dict(cuda_arrays)
    for stage_name, cuda_array in cuda_arrays:
        reference_array = reference_map[stage_name]
        summary = summarize_stage_pair(reference_array, cuda_array)
        print(
            f"{stage_name:<28} {summary['ref_shape']:>12} {summary['cuda_shape']:>12} "
            f"{summary['mean_abs']:>12} {summary['max_abs']:>12} {summary['overlap']:>24}"
        )

    reference_boxes = int(reference_summary.get("grouped_box_count", 0) or 0)
    cuda_boxes = int(cuda_summary.get("grouped_box_count", 0) or 0)
    if reference_boxes or cuda_boxes:
        print(f"Grouped boxes: reference={reference_boxes} cuda={cuda_boxes} delta={cuda_boxes - reference_boxes:+d}")


def print_summary_stage_comparison(cuda_validation_summary: dict, reference_validation_summary: dict) -> None:
    require_stage_keys(cuda_validation_summary, SUMMARY_STAGE_KEYS, "CUDA validation summary")
    require_stage_keys(reference_validation_summary, SUMMARY_STAGE_KEYS, "Reference validation summary")
    cuda_arrays = load_stage_arrays(cuda_validation_summary, SUMMARY_STAGE_KEYS)
    reference_arrays = load_stage_arrays(reference_validation_summary, SUMMARY_STAGE_KEYS)
    print_stage_summary(cuda_validation_summary,
                        reference_validation_summary,
                        cuda_arrays,
                        reference_arrays,
                        heading="Global merge parity summary:")
    projected_reference = int(reference_validation_summary.get("projected_grouped_box_count", 0) or 0)
    projected_cuda = int(cuda_validation_summary.get("projected_grouped_box_count", 0) or 0)
    merged_reference = int(reference_validation_summary.get("merged_grouped_box_count", 0) or 0)
    merged_cuda = int(cuda_validation_summary.get("merged_grouped_box_count", 0) or 0)
    print(f"Projected boxes: reference={projected_reference} cuda={projected_cuda} delta={projected_cuda - projected_reference:+d}")
    print(f"Merged boxes: reference={merged_reference} cuda={merged_cuda} delta={merged_cuda - merged_reference:+d}")


def load_stage_profile_aggregates(path: Path | None) -> dict[str, float]:
    if path is None:
        return {}
    payload = load_json(path)
    aggregates = {}
    for entry in payload.get("aggregates", []):
        stage = str(entry.get("stage", "") or "")
        if not stage:
            continue
        aggregates[stage] = float(entry.get("total_ms", 0.0) or 0.0)
    return aggregates


def load_operator_timing_summary(validation_summary: dict, debug_summary: dict) -> dict[str, float]:
    for payload in (validation_summary, debug_summary):
        timing = payload.get("operator_timing_ms")
        if isinstance(timing, dict):
            result: dict[str, float] = {}
            for key, value in timing.items():
                try:
                    parsed = float(value)
                except (TypeError, ValueError):
                    continue
                if parsed <= 0.0:
                    continue
                result[str(key)] = parsed
            if result:
                return result
    return {}


def build_timing_rows(cuda_elapsed_ms: float | None,
                      reference_elapsed_ms: float | None,
                      reference_stage_profile_path: Path | None,
                      cuda_validation_summary: dict,
                      cuda_debug_summary: dict) -> list[tuple[str, str, int, float | None, float | None]]:
    rows: list[tuple[str, str, int, float | None, float | None]] = [
        ("wall_clock_total", format_timing_label("wall_clock_total", 0), 0, cuda_elapsed_ms, reference_elapsed_ms),
    ]

    reference_aggregates = load_stage_profile_aggregates(reference_stage_profile_path)
    cuda_aggregates = load_operator_timing_summary(cuda_validation_summary, cuda_debug_summary)
    for label, cuda_key, reference_key, indent_level in TIMING_METRIC_MAP:
        cuda_value = cuda_aggregates.get(cuda_key) if cuda_key else None
        reference_value = reference_aggregates.get(reference_key) if reference_key else None
        if cuda_value is None and reference_value is None:
            continue
        rows.append((label, format_timing_label(label, indent_level), indent_level, cuda_value, reference_value))
    return rows


def print_timing_summary(cuda_elapsed_ms: float | None,
                         reference_elapsed_ms: float | None,
                         reference_stage_profile_path: Path | None,
                         cuda_validation_summary: dict,
                         cuda_debug_summary: dict) -> None:
    print("Timing summary (ms):")
    print(f"{'metric':<28} {'cuda':>12} {'reference':>12} {'delta':>12}")

    def format_ms(value: float | None) -> str:
        if value is None:
            return "n/a"
        return f"{value:.3f}"

    def format_delta(cuda_value: float | None, reference_value: float | None) -> str:
        if cuda_value is None or reference_value is None:
            return "n/a"
        return f"{cuda_value - reference_value:+.3f}"

    rows = build_timing_rows(cuda_elapsed_ms,
                             reference_elapsed_ms,
                             reference_stage_profile_path,
                             cuda_validation_summary,
                             cuda_debug_summary)

    for _, display_label, _, cuda_value, reference_value in rows:
        print(
            f"{display_label:<28} {format_ms(cuda_value):>12} {format_ms(reference_value):>12} "
            f"{format_delta(cuda_value, reference_value):>12}"
        )

    if cuda_elapsed_ms is not None and reference_elapsed_ms is not None and reference_elapsed_ms > 0.0:
        ratio = cuda_elapsed_ms / reference_elapsed_ms
        delta = cuda_elapsed_ms - reference_elapsed_ms
        print(f"Wall-clock delta: {delta:+.3f} ms ({ratio:.3f}x cuda/reference)")


def timing_plot_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_timing{output_path.suffix}")


def plot_timing_summary(rows: list[tuple[str, str, int, float | None, float | None]], output_path: Path) -> None:
    labels = [display_label for _, display_label, _, _, _ in rows]
    cuda_values = np.array([np.nan if cuda_value is None else float(cuda_value) for _, _, _, cuda_value, _ in rows], dtype=np.float32)
    reference_values = np.array([np.nan if reference_value is None else float(reference_value) for _, _, _, _, reference_value in rows], dtype=np.float32)

    y = np.arange(len(labels), dtype=np.float32)
    bar_height = 0.38
    fig_height = max(4.0, 0.45 * len(labels) + 1.5)
    fig, axis = plt.subplots(figsize=(14, fig_height))

    cuda_plot = np.nan_to_num(cuda_values, nan=0.0)
    reference_plot = np.nan_to_num(reference_values, nan=0.0)
    cuda_mask = ~np.isnan(cuda_values)
    reference_mask = ~np.isnan(reference_values)

    axis.barh(y[cuda_mask] - bar_height / 2.0, cuda_plot[cuda_mask], height=bar_height, label="CUDA", color="#1f77b4")
    axis.barh(y[reference_mask] + bar_height / 2.0, reference_plot[reference_mask], height=bar_height, label="Reference", color="#ff7f0e")

    axis.set_yticks(y)
    axis.set_yticklabels(labels)
    axis.invert_yaxis()
    axis.set_xlabel("Milliseconds")
    axis.set_title("CUDA vs Reference Timing Summary")
    axis.grid(axis="x", linestyle=":", alpha=0.4)
    axis.legend()

    for index, (_, _, _, cuda_value, reference_value) in enumerate(rows):
        delta_text = "n/a"
        if cuda_value is not None and reference_value is not None:
            delta_text = f"Δ {cuda_value - reference_value:+.3f}"
        elif cuda_value is not None:
            delta_text = f"CUDA {cuda_value:.3f}"
        elif reference_value is not None:
            delta_text = f"Ref {reference_value:.3f}"
        max_value = max(
            0.0,
            0.0 if cuda_value is None else float(cuda_value),
            0.0 if reference_value is None else float(reference_value),
        )
        axis.text(max_value + max(1.0, 0.01 * max_value), y[index], delta_text, va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_single_bundle(stage_arrays: list[tuple[str, np.ndarray]], output_path: Path) -> None:
    cols = 2
    rows = max(1, (len(stage_arrays) + cols - 1) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows), squeeze=False)
    for axis in axes.flat:
        axis.axis("off")
    for index, (key, array) in enumerate(stage_arrays):
        axis = axes.flat[index]
        image = axis.imshow(array, aspect="auto", origin="lower", cmap="viridis")
        axis.set_title(stage_title(key))
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        axis.axis("on")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_comparison(cuda_arrays: list[tuple[str, np.ndarray]],
                    reference_arrays: list[tuple[str, np.ndarray]],
                    output_path: Path) -> None:
    stage_names = [name for name, _ in cuda_arrays]
    fig, axes = plt.subplots(len(stage_names), 3, figsize=(16, 4 * max(1, len(stage_names))), squeeze=False)
    for row_index, stage_name in enumerate(stage_names):
        cuda_array = dict(cuda_arrays)[stage_name]
        reference_array = dict(reference_arrays)[stage_name]
        if cuda_array.shape != reference_array.shape:
            raise ValueError(f"Shape mismatch for {stage_name}: {cuda_array.shape} vs {reference_array.shape}")
        diff = np.abs(cuda_array - reference_array)
        panels = [
            (reference_array, f"Reference: {stage_title(stage_name)}"),
            (cuda_array, f"CUDA: {stage_title(stage_name)}"),
            (diff, f"Abs diff: {stage_title(stage_name)}"),
        ]
        for col_index, (panel, title) in enumerate(panels):
            axis = axes[row_index, col_index]
            image = axis.imshow(panel, aspect="auto", origin="lower", cmap="viridis")
            axis.set_title(title)
            fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot one CUDA validator artifact bundle and optionally compare it against a reference bundle.")
    parser.add_argument("--cuda-output-dir", required=True, help="CUDA validator output directory")
    parser.add_argument("--reference-output-dir", help="Reference validator output directory for side-by-side comparison")
    parser.add_argument("--stages", nargs="*", default=DEFAULT_STAGE_KEYS, help="Chunk-debug artifact summary keys to plot")
    parser.add_argument("--output", help="Output PNG path")
    parser.add_argument("--cuda-elapsed-ms", type=float, help="Wall-clock runtime for the CUDA operator replay")
    parser.add_argument("--reference-elapsed-ms", type=float, help="Wall-clock runtime for the reference validator")
    args = parser.parse_args()

    cuda_output_dir = Path(args.cuda_output_dir).expanduser().resolve()
    _, cuda_summary = resolve_debug_summary(cuda_output_dir)
    selected_stages = require_stage_keys(cuda_summary, args.stages, "CUDA chunk debug summary")

    cuda_arrays = [(key, load_stage_array(cuda_summary, key)) for key in selected_stages]
    output_path = Path(args.output).expanduser().resolve() if args.output else (cuda_output_dir / "chunk_debug" / "cuda_artifact_compare.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.reference_output_dir:
        reference_output_dir = Path(args.reference_output_dir).expanduser().resolve()
        _, reference_summary = resolve_debug_summary(reference_output_dir)
        _, cuda_validation_summary = resolve_validation_summary(cuda_output_dir)
        _, reference_validation_summary = resolve_validation_summary(reference_output_dir)
        require_stage_keys(reference_summary, selected_stages, "Reference chunk debug summary")
        reference_arrays = [(key, load_stage_array(reference_summary, key)) for key in selected_stages]
        plot_comparison(cuda_arrays, reference_arrays, output_path)
        print_stage_summary(cuda_summary, reference_summary, cuda_arrays, reference_arrays)
        print_summary_stage_comparison(cuda_validation_summary, reference_validation_summary)
        timing_rows = build_timing_rows(args.cuda_elapsed_ms,
                                        args.reference_elapsed_ms,
                                        resolve_stage_profile(reference_output_dir),
                                        cuda_validation_summary,
                                        cuda_summary)
        print_timing_summary(args.cuda_elapsed_ms,
                             args.reference_elapsed_ms,
                             resolve_stage_profile(reference_output_dir),
                             cuda_validation_summary,
                             cuda_summary)
        timing_output = timing_plot_path(output_path)
        plot_timing_summary(timing_rows, timing_output)
        print(f"Wrote CUDA vs reference artifact comparison plot: {output_path}")
        print(f"Wrote CUDA vs reference timing plot: {timing_output}")
    else:
        plot_single_bundle(cuda_arrays, output_path)
        print(f"Wrote CUDA artifact plot: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())