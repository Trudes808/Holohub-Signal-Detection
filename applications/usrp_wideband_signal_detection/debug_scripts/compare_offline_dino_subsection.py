#!/usr/bin/env python3
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

try:
    import yaml
except ImportError as exc:
    raise SystemExit("PyYAML is required to run compare_offline_dino_subsection.py") from exc


def load_retry_helper_module(app_dir: Path):
    workspace_root = app_dir.parents[2]
    helper_path = workspace_root / "Dinov3-RF-Signal-Detection" / "signal_detection_holoscan_retry_dino_helpers.py"
    if not helper_path.exists():
        raise FileNotFoundError(f"Retry helper not found: {helper_path}")
    spec = importlib.util.spec_from_file_location("retry_dino_helpers", helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load helper module from {helper_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("retry_dino_helpers", module)
    spec.loader.exec_module(module)
    return module


def default_output_dir(tensor_path: Path) -> Path:
    return Path("/tmp/usrp_spectrograms/dino_validator_artifacts") / tensor_path.stem


def load_config_values(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}
    fft_cfg = raw.get("fft", {}) or {}
    dino_cfg = raw.get("dinov3_signal_detector", {}) or {}
    coherent_cfg = raw.get("coherent_power_signal_detector", {}) or {}
    resolution_hz = float(fft_cfg.get("resolution", raw.get("resolution", 0.0)) or 0.0)
    span_hz = float(fft_cfg.get("span", raw.get("span", 0.0)) or 0.0)
    return {
        "resolution_hz": resolution_hz,
        "span_hz": span_hz,
        "chunk_bandwidth_hz": float(coherent_cfg.get("chunk_bandwidth_hz", 25.0e6)),
        "chunk_overlap_hz": float(coherent_cfg.get("chunk_overlap_hz", 6.25e6)),
        "uncalibrated_chunk_fraction": float(coherent_cfg.get("uncalibrated_chunk_fraction", 0.40)),
        "uncalibrated_overlap_fraction": float(coherent_cfg.get("uncalibrated_overlap_fraction", 0.20)),
        "ignore_sideband_hz": float(dino_cfg.get("ignore_sideband_hz", coherent_cfg.get("ignore_sideband_hz", 0.0)) or 0.0),
        "grouping": {
            "bridge_freq_px": 33,
            "bridge_time_px": 5,
            "min_component_size": 24,
            "min_freq_span_px": 18,
            "min_time_span_px": 2,
            "min_density": 0.06,
            "time_continuity_ratio": 0.85,
        },
    }


def load_bool_npy(path: Path) -> np.ndarray:
    array = np.load(path, allow_pickle=False)
    return np.asarray(array, dtype=np.float32) >= 0.5


def mask_metrics(mask_a: np.ndarray, mask_b: np.ndarray) -> dict:
    if mask_a.shape != mask_b.shape:
        raise ValueError(f"Mask shape mismatch: {mask_a.shape} vs {mask_b.shape}")
    agree = np.equal(mask_a, mask_b)
    intersection = np.logical_and(mask_a, mask_b)
    union = np.logical_or(mask_a, mask_b)
    return {
        "pixel_agreement": float(np.mean(agree)),
        "iou": float(np.count_nonzero(intersection) / max(1, np.count_nonzero(union))),
        "cpp_fraction": float(np.mean(mask_a)),
        "python_fraction": float(np.mean(mask_b)),
    }


def box_signature(box: dict) -> tuple:
    return (
        int(box.get("freq_start", 0)),
        int(box.get("freq_stop", 0)),
        int(box.get("time_start", 0)),
        int(box.get("time_stop", 0)),
        str(box.get("split_role", "unsplit")),
    )


def box_iou(box_a: dict, box_b: dict) -> float:
    freq_start = max(int(box_a.get("freq_start", 0)), int(box_b.get("freq_start", 0)))
    freq_stop = min(int(box_a.get("freq_stop", 0)), int(box_b.get("freq_stop", 0)))
    time_start = max(int(box_a.get("time_start", 0)), int(box_b.get("time_start", 0)))
    time_stop = min(int(box_a.get("time_stop", 0)), int(box_b.get("time_stop", 0)))
    inter = max(0, freq_stop - freq_start) * max(0, time_stop - time_start)
    area_a = max(0, int(box_a.get("freq_stop", 0)) - int(box_a.get("freq_start", 0))) * max(0, int(box_a.get("time_stop", 0)) - int(box_a.get("time_start", 0)))
    area_b = max(0, int(box_b.get("freq_stop", 0)) - int(box_b.get("freq_start", 0))) * max(0, int(box_b.get("time_stop", 0)) - int(box_b.get("time_start", 0)))
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def best_iou_summary(reference_boxes: list[dict], candidate_boxes: list[dict]) -> dict:
    if not reference_boxes:
        return {"mean_best_iou": 1.0, "min_best_iou": 1.0}
    best_ious = []
    for reference_box in reference_boxes:
      best_ious.append(max((box_iou(reference_box, candidate_box) for candidate_box in candidate_boxes), default=0.0))
    return {
        "mean_best_iou": float(np.mean(best_ious)),
        "min_best_iou": float(np.min(best_ious)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare one offline DINO validator chunk against the Python grouping reference.")
    parser.add_argument("--tensor-npy", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--debug-chunk-index", type=int)
    parser.add_argument("--report-json")
    args = parser.parse_args()

    app_dir = Path(__file__).resolve().parent
    helper = load_retry_helper_module(app_dir)
    tensor_path = Path(args.tensor_npy).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else default_output_dir(tensor_path)
    chunk_debug_dir = output_dir / "chunk_debug"
    debug_summary_path = chunk_debug_dir / "chunk_debug_summary.json"
    debug_boxes_path = chunk_debug_dir / "chunk_grouped_boxes.json"

    if not debug_summary_path.exists():
        raise FileNotFoundError(f"Missing validator debug summary: {debug_summary_path}")
    if not debug_boxes_path.exists():
        raise FileNotFoundError(f"Missing validator grouped boxes JSON: {debug_boxes_path}")

    config_values = load_config_values(config_path)
    with debug_summary_path.open("r", encoding="utf-8") as file:
        debug_summary = json.load(file)
    with debug_boxes_path.open("r", encoding="utf-8") as file:
        cpp_grouped_boxes_payload = json.load(file)
    cpp_grouped_boxes = list(cpp_grouped_boxes_payload.get("boxes", []))

    input_record = helper.load_input_record(str(tensor_path), input_kind="tensor_npy")
    rows = int(np.asarray(input_record["sxx_db"]).shape[0])
    resolution_hz = config_values["resolution_hz"]
    if (not np.isfinite(resolution_hz) or resolution_hz <= 0.0) and config_values["span_hz"] > 0.0 and rows > 0:
        resolution_hz = float(config_values["span_hz"]) / float(rows)
    if np.isfinite(resolution_hz) and resolution_hz > 0.0:
        input_record["freq_axis_hz"] = np.arange(rows, dtype=np.float32) * np.float32(resolution_hz)

    planning = helper.build_frequency_chunks_with_minimal_uniform_sideband_trim(
        np.asarray(input_record["freq_axis_hz"], dtype=np.float32),
        chunk_bandwidth_hz=config_values["chunk_bandwidth_hz"],
        chunk_overlap_hz=config_values["chunk_overlap_hz"],
        ignore_sideband_percent=0.0,
        min_keep_rows=16,
        ignore_sideband_hz=config_values["ignore_sideband_hz"] if config_values["ignore_sideband_hz"] > 0.0 else None,
        min_rows=16,
        uncalibrated_chunk_fraction=config_values["uncalibrated_chunk_fraction"],
        uncalibrated_overlap_fraction=config_values["uncalibrated_overlap_fraction"],
    )
    chunk_plan = planning["chunk_plan"]

    debug_chunk_index = int(debug_summary.get("chunk_index", args.debug_chunk_index if args.debug_chunk_index is not None else 13))
    if args.debug_chunk_index is not None and args.debug_chunk_index != debug_chunk_index:
        raise ValueError(f"Validator debug chunk index {debug_chunk_index} does not match requested {args.debug_chunk_index}")
    python_chunk = next((chunk for chunk in chunk_plan if int(chunk.get("chunk_index", -1)) == debug_chunk_index), None)
    if python_chunk is None:
        raise ValueError(f"Chunk index {debug_chunk_index} not found in Python chunk plan")

    combined_score = np.load(chunk_debug_dir / "chunk_combined_score.npy", allow_pickle=False).astype(np.float32)
    final_mask = load_bool_npy(chunk_debug_dir / "chunk_final_mask.npy")
    valid_mask_2d = load_bool_npy(chunk_debug_dir / "chunk_valid_mask.npy")
    cpp_grouped_mask = load_bool_npy(chunk_debug_dir / "chunk_grouped_mask.npy")
    cpp_bridged_mask = load_bool_npy(chunk_debug_dir / "chunk_bridged_mask.npy")
    valid_row_mask = np.any(valid_mask_2d, axis=1)

    grouping_cfg = config_values["grouping"]
    python_grouping = helper.group_signal_mask_regions(
        final_mask,
        score_map=combined_score,
        valid_row_mask=valid_row_mask,
        bridge_freq_px=grouping_cfg["bridge_freq_px"],
        bridge_time_px=grouping_cfg["bridge_time_px"],
        min_component_size=grouping_cfg["min_component_size"],
        min_freq_span_px=grouping_cfg["min_freq_span_px"],
        min_time_span_px=grouping_cfg["min_time_span_px"],
        min_density=grouping_cfg["min_density"],
        time_continuity_ratio=grouping_cfg["time_continuity_ratio"],
    )

    python_grouped_mask = np.asarray(python_grouping["grouped_mask"], dtype=bool)
    python_bridged_mask = np.asarray(python_grouping["bridged_mask"], dtype=bool)
    python_boxes = list(python_grouping.get("boxes", []))
    cpp_signatures = sorted(box_signature(box) for box in cpp_grouped_boxes)
    python_signatures = sorted(box_signature(box) for box in python_boxes)
    exact_box_matches = sum(1 for left, right in zip(cpp_signatures, python_signatures) if left == right) if len(cpp_signatures) == len(python_signatures) else 0

    report = {
        "tensor_path": str(tensor_path),
        "config_path": str(config_path),
        "output_dir": str(output_dir),
        "chunk_index": debug_chunk_index,
        "chunk_plan_match": {
            "row_start_matches": int(debug_summary.get("row_start", -1)) == int(python_chunk.get("row_start", -2)),
            "row_stop_matches": int(debug_summary.get("row_stop", -1)) == int(python_chunk.get("row_stop", -2)),
            "freq_start_hz_matches": abs(float(debug_summary.get("freq_start_hz", 0.0)) - float(python_chunk.get("freq_start_hz", 0.0))) <= max(1.0, abs(float(python_chunk.get("freq_start_hz", 0.0))) * 1.0e-6),
            "freq_stop_hz_matches": abs(float(debug_summary.get("freq_stop_hz", 0.0)) - float(python_chunk.get("freq_stop_hz", 0.0))) <= max(1.0, abs(float(python_chunk.get("freq_stop_hz", 0.0))) * 1.0e-6),
            "python_chunk": python_chunk,
            "cpp_chunk": {
                "row_start": int(debug_summary.get("row_start", -1)),
                "row_stop": int(debug_summary.get("row_stop", -1)),
                "freq_start_hz": float(debug_summary.get("freq_start_hz", 0.0)),
                "freq_stop_hz": float(debug_summary.get("freq_stop_hz", 0.0)),
            },
        },
        "bridged_mask_metrics": mask_metrics(cpp_bridged_mask, python_bridged_mask),
        "grouped_mask_metrics": mask_metrics(cpp_grouped_mask, python_grouped_mask),
        "box_comparison": {
            "cpp_box_count": len(cpp_grouped_boxes),
            "python_box_count": len(python_boxes),
            "exact_signature_match": cpp_signatures == python_signatures,
            "exact_box_matches_when_aligned": exact_box_matches,
            "cpp_to_python": best_iou_summary(cpp_grouped_boxes, python_boxes),
            "python_to_cpp": best_iou_summary(python_boxes, cpp_grouped_boxes),
        },
        "python_peak_score_floor": float(python_grouping.get("peak_score_floor", 0.0)),
    }

    report_path = Path(args.report_json).expanduser().resolve() if args.report_json else chunk_debug_dir / "chunk_python_comparison.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, sort_keys=True)
        file.write("\n")

    print(f"Chunk {debug_chunk_index} comparison report: {report_path}")
    print(f"  chunk plan row match: {report['chunk_plan_match']['row_start_matches']} / {report['chunk_plan_match']['row_stop_matches']}")
    print(f"  grouped mask agreement: {report['grouped_mask_metrics']['pixel_agreement']:.6f}")
    print(f"  grouped mask IoU: {report['grouped_mask_metrics']['iou']:.6f}")
    print(f"  box counts cpp/python: {len(cpp_grouped_boxes)} / {len(python_boxes)}")
    print(f"  exact box signature match: {report['box_comparison']['exact_signature_match']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())