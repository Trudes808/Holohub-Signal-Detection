from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image
from scipy import ndimage


WORKSPACE_ROOT = Path("/home/sat3737/holoscan_demo_workspace")
APP_DIR = WORKSPACE_ROOT / "holohub-dev" / "applications" / "usrp_wideband_signal_detection"
REFERENCE_REPO = WORKSPACE_ROOT / "Dinov3-RF-Signal-Detection"
DEFAULT_CONFIG_PATH = APP_DIR / "config_torchscript_validation_capture_single_channel.yaml"
DEFAULT_ARTIFACT_ROOT = Path("/tmp/usrp_spectrograms/dino_validator_artifacts")


def _load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_retry_helper_module(workspace_root: Path = WORKSPACE_ROOT):
    helper_path = workspace_root / "Dinov3-RF-Signal-Detection" / "signal_detection_holoscan_retry_dino_helpers.py"
    if not helper_path.exists():
        raise FileNotFoundError(f"Retry helper not found: {helper_path}")
    return _load_module("retry_dino_helpers_v2", helper_path)


def translate_workspace_path(raw_path: str | Path, workspace_root: Path = WORKSPACE_ROOT) -> Path:
    raw = str(raw_path)
    replacements = {
        "/workspace/models/dinov3": workspace_root / "dinov3",
        "/workspace/holohub": workspace_root / "holohub-dev",
        "/workspace/spectrograms": Path("/tmp/usrp_spectrograms"),
        "/workspace/dino_masks": Path("/tmp/usrp_dino_masks"),
        "/workspace/coherent_power_masks": Path("/tmp/coherent_power_masks"),
    }
    for prefix, replacement in replacements.items():
        if raw == prefix:
            return replacement.resolve()
        if raw.startswith(prefix + "/"):
            suffix = raw[len(prefix) + 1 :]
            return (replacement / suffix).resolve()
    return Path(raw).expanduser().resolve()


def default_output_dir(tensor_path: str | Path) -> Path:
    return DEFAULT_ARTIFACT_ROOT / Path(tensor_path).stem


def choose_dino_device(preferred: str | None = None) -> str:
    if preferred:
        return preferred
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def apply_tensor_frequency_axis_calibration(input_record: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    if str(input_record.get("input_kind", "")) != "tensor_npy":
        return input_record

    rows = int(np.asarray(input_record.get("sxx_db", np.empty((0, 0), dtype=np.float32))).shape[0])
    if rows <= 0:
        return input_record

    resolution_hz = float(config.get("resolution_hz", 0.0) or 0.0)
    span_hz = float(config.get("span_hz", 0.0) or 0.0)
    if (not np.isfinite(resolution_hz) or resolution_hz <= 0.0) and span_hz > 0.0:
        resolution_hz = span_hz / float(rows)
    if not np.isfinite(resolution_hz) or resolution_hz <= 0.0:
        return input_record

    calibrated = dict(input_record)
    calibrated["freq_axis_hz"] = np.arange(rows, dtype=np.float32) * np.float32(resolution_hz)
    calibrated["frequency_axis_calibrated"] = True
    return calibrated


def load_validation_config(config_path: str | Path = DEFAULT_CONFIG_PATH, workspace_root: Path = WORKSPACE_ROOT) -> dict[str, Any]:
    resolved_path = Path(config_path).expanduser().resolve()
    with resolved_path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}
    dino_cfg = raw.get("dinov3_signal_detector", {}) or {}
    coherent_cfg = raw.get("coherent_power_signal_detector", {}) or {}
    fft_cfg = raw.get("fft", {}) or {}
    return {
        "config_path": resolved_path,
        "raw": raw,
        "fft": fft_cfg,
        "dino": dino_cfg,
        "coherent": coherent_cfg,
        "model_name": str(dino_cfg.get("model_name", "dinov3_vitb16")),
        "model_repo_path": translate_workspace_path(dino_cfg.get("model_repo_path", workspace_root / "dinov3"), workspace_root),
        "weights_path": translate_workspace_path(dino_cfg.get("weights_path", workspace_root / "dinov3" / "weights"), workspace_root),
        "chunk_bandwidth_hz": float(coherent_cfg.get("chunk_bandwidth_hz", 25.0e6)),
        "chunk_overlap_hz": float(coherent_cfg.get("chunk_overlap_hz", 6.25e6)),
        "uncalibrated_chunk_fraction": float(coherent_cfg.get("uncalibrated_chunk_fraction", 0.40)),
        "uncalibrated_overlap_fraction": float(coherent_cfg.get("uncalibrated_overlap_fraction", 0.20)),
        "ignore_sideband_hz": float(dino_cfg.get("ignore_sideband_hz", coherent_cfg.get("ignore_sideband_hz", 7.0e6)) or 0.0),
        "frontend_row_q": float(dino_cfg.get("frontend_correction_row_q", coherent_cfg.get("frontend_row_q", 25.0))),
        "frontend_reference_q": float(dino_cfg.get("frontend_correction_reference_q", coherent_cfg.get("frontend_reference_q", 75.0))),
        "frontend_smooth_sigma": float(dino_cfg.get("frontend_correction_smooth_sigma", coherent_cfg.get("frontend_smooth_sigma", 12.0))),
        "frontend_max_boost_db": float(dino_cfg.get("frontend_correction_max_boost_db", coherent_cfg.get("frontend_max_boost_db", 12.0))),
        "min_component_size": int(coherent_cfg.get("min_component_size", 6)),
        "grouping_bridge_freq_px": int(coherent_cfg.get("grouping_bridge_freq_px", 33)),
        "grouping_bridge_time_px": int(coherent_cfg.get("grouping_bridge_time_px", 5)),
        "grouping_min_component_size": int(coherent_cfg.get("grouping_min_component_size", 24)),
        "grouping_min_freq_span_px": int(coherent_cfg.get("grouping_min_freq_span_px", 18)),
        "grouping_min_time_span_px": int(coherent_cfg.get("grouping_min_time_span_px", 2)),
        "grouping_min_density": float(coherent_cfg.get("grouping_min_density", 0.06)),
        "grouping_time_continuity_ratio": float(coherent_cfg.get("grouping_time_continuity_ratio", 0.85)),
        "resolution_hz": float(fft_cfg.get("resolution", 0.0) or 0.0),
        "span_hz": float(fft_cfg.get("span", 0.0) or 0.0),
    }


def load_json(path: str | Path) -> Any:
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as file:
        return json.load(file)


def load_bool_npy(path: str | Path) -> np.ndarray:
    return np.asarray(np.load(Path(path).expanduser().resolve(), allow_pickle=False), dtype=np.float32) >= 0.5


def normalize01_masked(array: np.ndarray, mask: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    active = np.asarray(mask, dtype=bool) & np.isfinite(array)
    out = np.zeros_like(array, dtype=np.float32)
    if not np.any(active):
        return out
    low = float(np.min(array[active]))
    high = float(np.max(array[active]))
    if high <= low + 1.0e-12:
        return out
    out[active] = np.clip((array[active] - low) / (high - low), 0.0, 1.0)
    return out.astype(np.float32)


def keep_large_components(mask: np.ndarray, min_size: int) -> np.ndarray:
    labels, count = ndimage.label(np.asarray(mask, dtype=bool))
    if count <= 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = ndimage.sum(np.ones_like(labels, dtype=np.uint8), labels, index=np.arange(1, count + 1))
    keep = np.where(np.asarray(sizes) >= float(min_size))[0] + 1
    if keep.size == 0:
        return np.zeros_like(mask, dtype=bool)
    return np.isin(labels, keep)


def component_speckle_score(mask: np.ndarray, min_size: int = 6) -> float:
    mask_u8 = np.asarray(mask, dtype=np.uint8)
    active = float(np.count_nonzero(mask_u8))
    if active <= 0.0:
        return 0.0
    component_labels, component_count = ndimage.label(mask_u8)
    if component_count <= 0:
        return 0.0
    component_sizes = ndimage.sum(mask_u8, component_labels, index=np.arange(1, component_count + 1))
    component_sizes = np.asarray(component_sizes, dtype=np.float32)
    kept = float(np.sum(component_sizes[component_sizes >= float(max(1, min_size))]))
    return float(np.clip(kept / max(active, 1.0), 0.0, 1.0))


def texture_passthrough_from_structure(normalize_map01_local, structure_map: np.ndarray, min_size: int = 6) -> dict[str, float]:
    structure_map = normalize_map01_local(np.asarray(structure_map, dtype=np.float32), 5.0, 95.0)
    structure_thr = float(np.quantile(structure_map, 0.85))
    structure_top_mask = structure_map >= structure_thr
    structure_peak = float(np.mean(structure_map[structure_top_mask])) if np.any(structure_top_mask) else 0.0
    structure_speckle = component_speckle_score(structure_top_mask.astype(np.uint8), min_size=min_size)
    structure_clean_score = float(np.clip(0.70 * structure_speckle + 0.30 * structure_peak, 0.0, 1.0))
    speckle_clean_thr = 0.96
    speckle_strong_thr = 0.50
    if structure_speckle >= speckle_clean_thr:
        clean_ramp = float(np.clip((structure_speckle - speckle_clean_thr) / max(1.0 - speckle_clean_thr, 1e-6), 0.0, 1.0))
        texture_passthrough = 0.25 + 0.75 * clean_ramp
    elif structure_speckle >= speckle_strong_thr:
        speckle_ramp = float(np.clip((structure_speckle - speckle_strong_thr) / max(speckle_clean_thr - speckle_strong_thr, 1e-6), 0.0, 1.0))
        texture_passthrough = 0.05 + 0.20 * speckle_ramp
    else:
        low_ramp = float(np.clip(structure_speckle / max(speckle_strong_thr, 1e-6), 0.0, 1.0))
        texture_passthrough = 0.02 + 0.03 * low_ramp
    return {
        "structure_speckle_score": float(structure_speckle),
        "structure_peak": float(structure_peak),
        "structure_clean_score": float(structure_clean_score),
        "texture_passthrough": float(np.clip(texture_passthrough, 0.0, 1.0)),
        "speckle_strong_threshold": float(speckle_strong_thr),
        "speckle_clean_threshold": float(speckle_clean_thr),
    }


def build_structure_tensor_coherence_products(
    retry_helper,
    corrected_chunk: np.ndarray,
    valid_score_mask: np.ndarray,
    *,
    min_component_size: int,
) -> dict[str, Any]:
    normalize_map01_local = retry_helper._normalize_map01_local
    structure_maps = retry_helper.multi_scale_structure_tensor_gate(np.asarray(corrected_chunk, dtype=np.float32))
    coherence_gate_px = normalize_map01_local(np.asarray(structure_maps["gate_px"], dtype=np.float32), 5.0, 99.0)
    coherence_region_threshold = float(np.quantile(coherence_gate_px[valid_score_mask], 0.80)) if np.any(valid_score_mask) else 1.0
    coherence_region_mask = np.logical_and(coherence_gate_px >= coherence_region_threshold, valid_score_mask)
    coherence_region_mask = ndimage.binary_closing(coherence_region_mask, structure=np.ones((3, 7), dtype=bool), iterations=1)
    coherence_region_mask = ndimage.binary_fill_holes(coherence_region_mask)
    coherence_region_mask = keep_large_components(coherence_region_mask, min_size=max(6, int(min_component_size)))
    coherence_region_mask = np.logical_and(coherence_region_mask, valid_score_mask)
    return {
        "structure_maps": structure_maps,
        "coherence_gate_px": coherence_gate_px.astype(np.float32),
        "coherence_region_threshold": coherence_region_threshold,
        "coherence_region_mask": coherence_region_mask,
    }


def build_python_retry_hybrid_products(
    retry_helper,
    corrected_chunk: np.ndarray,
    valid_score_mask: np.ndarray,
    *,
    dino_repo_dir: str | Path,
    dino_weights_path: str | Path,
    dino_model_name: str,
    dino_device: str,
    min_component_size: int,
    dino_db_min: float,
    dino_db_max: float,
) -> dict[str, Any]:
    dino_texture_experiment = retry_helper.run_subsection_dino_texture_experiment(
        corrected_chunk,
        dino_repo_dir=dino_repo_dir,
        dino_weights_path=dino_weights_path,
        dino_model_name=dino_model_name,
        dino_device=dino_device,
        min_component_size=min_component_size,
        dino_feature_knn=8,
        dino_spatial_weight=0.35,
        dino_score_q=0.60,
        texture_knn=6,
        texture_q=0.90,
        dino_db_min=dino_db_min,
        dino_db_max=dino_db_max,
    )

    coherence_products = build_structure_tensor_coherence_products(
        retry_helper,
        corrected_chunk,
        valid_score_mask,
        min_component_size=min_component_size,
    )
    normalize_map01_local = retry_helper._normalize_map01_local
    coherence_gate_px = np.asarray(coherence_products["coherence_gate_px"], dtype=np.float32)
    coherence_region_threshold = float(coherence_products["coherence_region_threshold"])
    coherence_region_mask = np.asarray(coherence_products["coherence_region_mask"], dtype=bool)

    dino_score_px = normalize_map01_local(np.asarray(dino_texture_experiment["dino_score_px"], dtype=np.float32), 5.0, 95.0)
    dino_mask_px = np.asarray(dino_texture_experiment["dino_mask_px"], dtype=bool)
    texture_score_px = normalize_map01_local(np.asarray(dino_texture_experiment["texture_score_px"], dtype=np.float32), 5.0, 95.0)
    texture_top_threshold = float(np.quantile(texture_score_px[valid_score_mask], 0.90)) if np.any(valid_score_mask) else 1.0
    texture_top_mask = np.logical_and(texture_score_px >= texture_top_threshold, valid_score_mask)

    texture_policy = texture_passthrough_from_structure(normalize_map01_local, coherence_gate_px, min_size=max(6, int(min_component_size)))
    texture_passthrough_active = float(texture_policy["structure_speckle_score"]) >= float(texture_policy["speckle_strong_threshold"])

    hybrid_dino_mask = np.logical_and(dino_mask_px, coherence_region_mask)
    texture_union_mask = np.zeros_like(texture_top_mask, dtype=bool)
    if texture_passthrough_active:
        texture_union_mask = keep_large_components(
            texture_top_mask,
            min_size=max(2, int(round((1.0 - float(texture_policy["texture_passthrough"])) * 8.0))),
        )
        texture_union_mask = np.logical_and(texture_union_mask, valid_score_mask)

    hybrid_mask = np.logical_or(hybrid_dino_mask, texture_union_mask)
    hybrid_mask = keep_large_components(hybrid_mask, min_size=max(3, int(min_component_size // 2)))
    hybrid_mask = np.logical_and(hybrid_mask, valid_score_mask)

    hybrid_dino_contrib = (dino_score_px * coherence_gate_px).astype(np.float32)
    hybrid_texture_contrib = (texture_score_px * float(texture_policy["texture_passthrough"])).astype(np.float32)

    return {
        "dino_texture_experiment": dino_texture_experiment,
        "coherence_gate_px": coherence_gate_px.astype(np.float32),
        "coherence_region_threshold": coherence_region_threshold,
        "coherence_region_mask": coherence_region_mask,
        "dino_score_px": dino_score_px.astype(np.float32),
        "dino_mask_px": dino_mask_px,
        "texture_score_px": texture_score_px.astype(np.float32),
        "texture_top_threshold": texture_top_threshold,
        "texture_top_mask": texture_top_mask,
        "texture_policy": texture_policy,
        "texture_passthrough_active": bool(texture_passthrough_active),
        "hybrid_dino_mask": hybrid_dino_mask,
        "texture_union_mask": texture_union_mask,
        "hybrid_mask": hybrid_mask,
        "hybrid_dino_contrib": hybrid_dino_contrib,
        "hybrid_texture_contrib": hybrid_texture_contrib,
    }


def build_retry_frequency_support_mask(hybrid_dino_contrib: np.ndarray, valid_mask: np.ndarray) -> dict[str, Any]:
    base_map = np.asarray(hybrid_dino_contrib, dtype=np.float32)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    base_norm = normalize01_masked(base_map, valid_mask)
    envelope_map = normalize01_masked(ndimage.gaussian_filter(base_norm, sigma=(6.0, 1.4)), valid_mask)
    residual_penalty = normalize01_masked(
        ndimage.gaussian_filter(np.abs(base_norm - ndimage.gaussian_filter(base_norm, sigma=(4.0, 1.0))), sigma=(2.0, 0.8)),
        valid_mask,
    )
    freq_curvature_penalty = normalize01_masked(
        np.abs(ndimage.gaussian_filter1d(base_norm, sigma=0.8, axis=0, order=2)),
        valid_mask,
    )
    keep_freq = normalize01_masked(envelope_map - 0.90 * freq_curvature_penalty, valid_mask)
    keep_res = normalize01_masked(envelope_map - 1.00 * residual_penalty, valid_mask)
    residual_veto_gate = np.clip((keep_res - 0.30) / 0.70, 0.0, 1.0).astype(np.float32)
    combined_score = normalize01_masked(keep_freq * (0.35 + 0.65 * residual_veto_gate), valid_mask)
    combined_score = np.where(valid_mask, combined_score, 0.0).astype(np.float32)

    active_freq = keep_freq[valid_mask]
    active_res = keep_res[valid_mask]
    active_combined = combined_score[valid_mask]
    seed_freq_thr = float(np.quantile(active_freq, 0.90)) if active_freq.size else 1.0
    seed_res_thr = float(np.quantile(active_res, 0.82)) if active_res.size else 1.0
    combined_thr = float(np.quantile(active_combined, 0.78)) if active_combined.size else 1.0

    seed_mask = np.logical_and.reduce((keep_freq >= seed_freq_thr, keep_res >= seed_res_thr, valid_mask))
    seed_mask = keep_large_components(seed_mask, min_size=8)

    final_mask = np.logical_and(seed_mask, combined_score >= combined_thr * 0.85)
    final_mask = ndimage.binary_closing(final_mask, structure=np.ones((7, 3), dtype=bool), iterations=1)
    final_mask = ndimage.binary_fill_holes(final_mask)
    final_mask = keep_large_components(final_mask, min_size=24)
    final_mask = np.logical_and(final_mask, valid_mask)

    return {
        "hybrid_dino_contrib": base_map.astype(np.float32),
        "base_norm": base_norm,
        "envelope_map": envelope_map,
        "residual_penalty": residual_penalty,
        "freq_curvature_penalty": freq_curvature_penalty,
        "keep_freq": keep_freq,
        "keep_res": keep_res,
        "residual_veto_gate": residual_veto_gate,
        "combined_score": combined_score,
        "seed_mask": seed_mask,
        "final_mask": final_mask,
        "seed_freq_threshold": seed_freq_thr,
        "seed_res_threshold": seed_res_thr,
        "combined_threshold": combined_thr,
    }


def resize_array_to_shape(array: np.ndarray, target_shape: tuple[int, int], order: int = 1) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    target_rows, target_cols = int(target_shape[0]), int(target_shape[1])
    if array.shape == (target_rows, target_cols):
        return array.astype(np.float32, copy=False)
    zoom_factors = (target_rows / max(array.shape[0], 1), target_cols / max(array.shape[1], 1))
    resized = ndimage.zoom(array, zoom=zoom_factors, order=order)
    if resized.shape != (target_rows, target_cols):
        resized = resized[:target_rows, :target_cols]
        if resized.shape != (target_rows, target_cols):
            padded = np.zeros((target_rows, target_cols), dtype=np.float32)
            padded[: resized.shape[0], : resized.shape[1]] = resized
            resized = padded
    return resized.astype(np.float32)


def resize_mask_to_shape(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    resized = resize_array_to_shape(np.asarray(mask, dtype=np.float32), target_shape, order=0)
    return resized >= 0.5


def mask_metrics(lhs: np.ndarray, rhs: np.ndarray) -> dict[str, float]:
    lhs = np.asarray(lhs, dtype=bool)
    rhs = np.asarray(rhs, dtype=bool)
    if lhs.shape != rhs.shape:
        raise ValueError(f"Mask shape mismatch: {lhs.shape} vs {rhs.shape}")
    agreement = float(np.mean(lhs == rhs))
    intersection = float(np.count_nonzero(lhs & rhs))
    union = float(np.count_nonzero(lhs | rhs))
    return {
        "agreement": agreement,
        "iou": 1.0 if union == 0.0 else intersection / union,
        "lhs_fraction": float(np.mean(lhs)),
        "rhs_fraction": float(np.mean(rhs)),
    }


def float_map_metrics(lhs: np.ndarray, rhs: np.ndarray, valid_mask: np.ndarray | None = None) -> dict[str, float]:
    lhs = np.asarray(lhs, dtype=np.float32)
    rhs = np.asarray(rhs, dtype=np.float32)
    if lhs.shape != rhs.shape:
        raise ValueError(f"Float map shape mismatch: {lhs.shape} vs {rhs.shape}")
    active = np.isfinite(lhs) & np.isfinite(rhs)
    if valid_mask is not None:
        active &= np.asarray(valid_mask, dtype=bool)
    if not np.any(active):
        return {
            "mae": 0.0,
            "rmse": 0.0,
            "max_abs": 0.0,
            "corr": 1.0,
        }
    lhs_values = lhs[active]
    rhs_values = rhs[active]
    diff = lhs_values - rhs_values
    corr = 1.0
    if lhs_values.size > 1 and float(np.std(lhs_values)) > 0.0 and float(np.std(rhs_values)) > 0.0:
        corr = float(np.corrcoef(lhs_values, rhs_values)[0, 1])
    return {
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "max_abs": float(np.max(np.abs(diff))),
        "corr": corr,
    }


def normalize_map01_percentile(array: np.ndarray, low_q: float, high_q: float) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    out = np.zeros_like(array, dtype=np.float32)
    finite = np.isfinite(array)
    if not np.any(finite):
        return out
    low = float(np.quantile(array[finite], np.clip(low_q / 100.0, 0.0, 1.0)))
    high = float(np.quantile(array[finite], np.clip(high_q / 100.0, 0.0, 1.0)))
    if high <= low + 1.0e-12:
        return out
    out[finite] = np.clip((array[finite] - low) / (high - low), 0.0, 1.0)
    return out.astype(np.float32)


def build_python_raw_feature_energy_score(python_retry: dict[str, Any], output_shape: tuple[int, int]) -> dict[str, Any]:
    dino_run = python_retry["dino_texture_experiment"]["dino"]
    features = np.asarray(dino_run["features"], dtype=np.float32)
    patch_h, patch_w = (int(value) for value in dino_run["shape"])
    if features.ndim != 2 or features.shape[0] != patch_h * patch_w:
        raise ValueError(f"Unexpected Python DINO feature shape {features.shape} for patch grid {(patch_h, patch_w)}")
    raw_patch_score = np.sqrt(np.mean(features * features, axis=1) + 1.0e-6).reshape(patch_h, patch_w).astype(np.float32)
    raw_patch_score = normalize_map01_percentile(raw_patch_score, 5.0, 95.0)
    raw_pixel_score = resize_array_to_shape(raw_patch_score, tuple(int(value) for value in output_shape), order=1)
    grouped_pixel_score = resize_array_to_shape(np.asarray(python_retry["dino_score_px"], dtype=np.float32), tuple(int(value) for value in output_shape), order=1)
    return {
        "raw_patch_score": raw_patch_score,
        "raw_pixel_score": raw_pixel_score,
        "grouped_pixel_score": grouped_pixel_score,
        "patch_shape": (patch_h, patch_w),
    }


def build_signal_agnostic_input_debug(
    retry_helper,
    corrected_chunk: np.ndarray,
    output_shape: tuple[int, int],
    *,
    db_min: float,
    db_max: float,
) -> dict[str, np.ndarray]:
    _, input_debug = retry_helper.build_signal_agnostic_dino_input(
        np.asarray(corrected_chunk, dtype=np.float32),
        db_min=float(db_min),
        db_max=float(db_max),
    )
    input_gray = np.asarray(input_debug["input_gray01"], dtype=np.float32)
    input_gray = np.clip(np.round(255.0 * input_gray), 0.0, 255.0).astype(np.float32) / 255.0
    fixed_gray = np.asarray(input_debug["fixed_gray01"], dtype=np.float32)
    resized_shape = tuple(int(value) for value in output_shape)
    return {
        "input_gray": resize_array_to_shape(input_gray, resized_shape, order=1),
        "fixed_gray": resize_array_to_shape(fixed_gray, resized_shape, order=1),
    }


def build_python_expected_dino_runtime_input(
    retry_helper,
    corrected_chunk: np.ndarray,
    chunk_freq_axis_hz: np.ndarray,
    *,
    ignore_sideband_hz: float,
    patch_size: int,
    target_shape: tuple[int, int],
    db_min: float,
    db_max: float,
) -> dict[str, Any] | None:
    local_ignore_info = retry_helper.compute_ignore_sideband_rows(
        np.asarray(chunk_freq_axis_hz, dtype=np.float32),
        ignore_sideband_percent=0.0,
        min_keep_rows=max(int(patch_size), 16),
        ignore_sideband_hz=float(ignore_sideband_hz),
    )
    valid_row_mask = np.asarray(local_ignore_info["valid_row_mask"], dtype=bool)
    valid_row_idx = np.flatnonzero(valid_row_mask)
    if valid_row_idx.size == 0:
        return None

    row_start = int(valid_row_idx[0])
    row_stop = int(valid_row_idx[-1]) + 1
    cropped_chunk = np.asarray(corrected_chunk[row_start:row_stop, :], dtype=np.float32)

    expected_rows = (cropped_chunk.shape[0] // int(patch_size)) * int(patch_size)
    expected_cols = (cropped_chunk.shape[1] // int(patch_size)) * int(patch_size)
    if expected_rows < int(patch_size) or expected_cols < int(patch_size):
        return None

    python_runtime_chunk = np.asarray(cropped_chunk[:expected_rows, :expected_cols], dtype=np.float32)
    dino_input_debug = build_signal_agnostic_input_debug(
        retry_helper,
        python_runtime_chunk,
        target_shape,
        db_min=db_min,
        db_max=db_max,
    )
    return {
        "row_start": row_start,
        "row_stop": row_stop,
        "expected_shape": (int(expected_rows), int(expected_cols)),
        "runtime_chunk": np.asarray(python_runtime_chunk, dtype=np.float32),
        "input_gray": np.asarray(dino_input_debug["input_gray"], dtype=np.float32),
        "fixed_gray": np.asarray(dino_input_debug["fixed_gray"], dtype=np.float32),
    }


def embed_runtime_crop_to_chunk_grid(
    runtime_map: np.ndarray,
    source_shape: tuple[int, int],
    *,
    row_start: int,
    expected_shape: tuple[int, int],
    target_shape: tuple[int, int],
    order: int,
) -> np.ndarray:
    runtime_map = np.asarray(runtime_map, dtype=np.float32)
    source_rows, source_cols = (int(value) for value in source_shape)
    expected_rows, expected_cols = (int(value) for value in expected_shape)
    if runtime_map.shape != (expected_rows, expected_cols):
        raise ValueError(
            f"Runtime map shape {runtime_map.shape} does not match expected runtime crop {(expected_rows, expected_cols)}"
        )

    canvas = np.zeros((source_rows, source_cols), dtype=np.float32)
    safe_row_start = max(0, min(source_rows, int(row_start)))
    copy_rows = min(expected_rows, max(0, source_rows - safe_row_start))
    copy_cols = min(expected_cols, source_cols)
    if copy_rows > 0 and copy_cols > 0:
        canvas[safe_row_start:safe_row_start + copy_rows, :copy_cols] = runtime_map[:copy_rows, :copy_cols]
    return resize_array_to_shape(canvas, target_shape, order=order)


def box_signature(box: dict[str, Any]) -> tuple[int, int, int, int, str]:
    return (
        int(box.get("freq_start", 0)),
        int(box.get("freq_stop", 0)),
        int(box.get("time_start", 0)),
        int(box.get("time_stop", 0)),
        str(box.get("split_role", "unsplit")),
    )


def box_iou(box_a: dict[str, Any], box_b: dict[str, Any]) -> float:
    freq_start = max(int(box_a.get("freq_start", 0)), int(box_b.get("freq_start", 0)))
    freq_stop = min(int(box_a.get("freq_stop", 0)), int(box_b.get("freq_stop", 0)))
    time_start = max(int(box_a.get("time_start", 0)), int(box_b.get("time_start", 0)))
    time_stop = min(int(box_a.get("time_stop", 0)), int(box_b.get("time_stop", 0)))
    inter = max(0, freq_stop - freq_start) * max(0, time_stop - time_start)
    area_a = max(0, int(box_a.get("freq_stop", 0)) - int(box_a.get("freq_start", 0))) * max(0, int(box_a.get("time_stop", 0)) - int(box_a.get("time_start", 0)))
    area_b = max(0, int(box_b.get("freq_stop", 0)) - int(box_b.get("freq_start", 0))) * max(0, int(box_b.get("time_stop", 0)) - int(box_b.get("time_start", 0)))
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def best_iou_summary(reference_boxes: list[dict[str, Any]], candidate_boxes: list[dict[str, Any]]) -> dict[str, float]:
    if not reference_boxes:
        return {"mean_best_iou": 1.0, "min_best_iou": 1.0}
    best_ious = [max((box_iou(reference_box, candidate_box) for candidate_box in candidate_boxes), default=0.0) for reference_box in reference_boxes]
    return {
        "mean_best_iou": float(np.mean(best_ious)),
        "min_best_iou": float(np.min(best_ious)),
    }


def load_cpp_artifact_bundle(output_dir: str | Path, debug_chunk_index: int | None = None, workspace_root: Path = WORKSPACE_ROOT) -> dict[str, Any]:
    output_path = Path(output_dir).expanduser().resolve()
    summary = load_json(output_path / "offline_validation_summary.json")
    chunk_plan = load_json(output_path / "offline_chunk_plan.json")
    chunk_results = load_json(output_path / "offline_chunk_results.json")

    debug_summary = load_json(output_path / "chunk_debug" / "chunk_debug_summary.json")
    if debug_chunk_index is not None and int(debug_summary.get("chunk_index", -1)) != int(debug_chunk_index):
        raise ValueError(
            f"Requested chunk {debug_chunk_index} but chunk_debug artifacts contain chunk {debug_summary.get('chunk_index')}"
        )
    chunk_index = int(debug_summary["chunk_index"])
    comparison_report_path = output_path / "chunk_debug" / "chunk_python_comparison.json"
    comparison_report = load_json(comparison_report_path) if comparison_report_path.exists() else None

    cpp_boxes_payload = load_json(output_path / "chunk_debug" / "chunk_grouped_boxes.json")
    chunk_results_by_index = {
        int(chunk["chunk_index"]): chunk for chunk in chunk_results.get("chunks", [])
    }
    chunk_plan_by_index = {
        int(chunk["chunk_index"]): chunk for chunk in chunk_plan.get("chunks", [])
    }

    output_rows = int(summary["output_rows"])
    output_cols = int(summary["output_cols"])
    canonical_rows = int(summary.get("canonical_rows", summary.get("input_rows", output_rows)))
    row_start = int(debug_summary["row_start"])
    row_stop = int(debug_summary["row_stop"])
    mapped_row_start = max(0, min(output_rows - 1, int(np.floor(row_start * output_rows / max(canonical_rows, 1)))))
    mapped_row_stop = max(mapped_row_start + 1, min(output_rows, int(np.ceil(row_stop * output_rows / max(canonical_rows, 1)))))
    chunk_dino_score = np.load(output_path / "chunk_debug" / "chunk_dino_score.npy", allow_pickle=False).astype(np.float32)
    chunk_dino_score_raw_path = output_path / "chunk_debug" / "chunk_dino_score_raw.npy"
    chunk_dino_score_raw = np.load(chunk_dino_score_raw_path, allow_pickle=False).astype(np.float32) if chunk_dino_score_raw_path.exists() else chunk_dino_score.copy()
    chunk_combined_score = np.load(output_path / "chunk_debug" / "chunk_combined_score.npy", allow_pickle=False).astype(np.float32)
    runtime_input_gray_path = output_path / "chunk_debug" / "chunk_runtime_input_gray.npy"
    chunk_grouped_mask = load_bool_npy(output_path / "chunk_debug" / "chunk_grouped_mask.npy")
    chunk_grouped_combined_score = np.where(chunk_grouped_mask, chunk_combined_score, 0.0).astype(np.float32)
    chunk_grouped_dino_score = np.where(chunk_grouped_mask, chunk_dino_score, 0.0).astype(np.float32)
    chunk_grouped_raw_dino_score = np.where(chunk_grouped_mask, chunk_dino_score_raw, 0.0).astype(np.float32)
    runtime_input_gray = np.load(runtime_input_gray_path, allow_pickle=False).astype(np.float32) if runtime_input_gray_path.exists() else None
    runtime_input_gray_rows = int(debug_summary.get("runtime_input_gray_rows", 0) or 0)
    runtime_input_gray_cols = int(debug_summary.get("runtime_input_gray_cols", 0) or 0)
    patch_features_path = output_path / "chunk_debug" / "chunk_patch_features.npy"
    patch_features = np.load(patch_features_path, allow_pickle=False).astype(np.float32) if patch_features_path.exists() else None
    patch_rows = int(debug_summary.get("patch_rows", 0) or 0)
    patch_cols = int(debug_summary.get("patch_cols", 0) or 0)
    feature_dim = int(debug_summary.get("feature_dim", 0) or 0)
    artifact_contract = str(debug_summary.get("artifact_contract", "") or "")
    has_patch_feature_grouping = bool(
        patch_features is not None
        and patch_rows > 0
        and patch_cols > 0
        and feature_dim > 0
        and patch_features.ndim == 2
        and patch_features.shape == (patch_rows * patch_cols, feature_dim)
    )
    artifact_warnings: list[str] = []
    if not chunk_dino_score_raw_path.exists():
        artifact_warnings.append(
            "chunk_debug/chunk_dino_score_raw.npy is missing, so the notebook would compare the C++ grouped/fallback DINO surface against the Python raw feature score."
        )
    if runtime_input_gray is None:
        artifact_warnings.append(
            "chunk_debug/chunk_runtime_input_gray.npy is missing, so the exact pre-model grayscale panel cannot use the true C++ runtime dump."
        )
    if runtime_input_gray is not None and runtime_input_gray_rows > 0 and runtime_input_gray_cols > 0:
        src_rows = int(debug_summary.get("src_rows", runtime_input_gray_rows) or runtime_input_gray_rows)
        src_cols = int(debug_summary.get("src_cols", runtime_input_gray_cols) or runtime_input_gray_cols)
        ignore_bins_per_side = int(debug_summary.get("ignore_bins_per_side", 0) or 0)
        if ignore_bins_per_side == 0 and runtime_input_gray_rows < src_rows:
            artifact_warnings.append(
                f"C++ runtime_input_gray shape is {(runtime_input_gray_rows, runtime_input_gray_cols)} for a {(src_rows, src_cols)} chunk even though ignore_bins_per_side=0; this is the legacy extra-runtime-crop signature."
            )
    if not patch_features_path.exists() or not has_patch_feature_grouping:
        artifact_warnings.append(
            "chunk_debug/chunk_patch_features.npy is missing or inconsistent, so the saved C++ grouped DINO surface is not a verified patch-feature grouping result."
        )
    if artifact_contract and artifact_contract != "chunk_no_extra_sideband_crop_v2":
        artifact_warnings.append(
            f"Chunk artifact contract is '{artifact_contract}' instead of 'chunk_no_extra_sideband_crop_v2'."
        )

    return {
        "output_dir": output_path,
        "summary": summary,
        "chunk_plan": chunk_plan,
        "chunk_results": chunk_results,
        "debug_summary": debug_summary,
        "comparison_report": comparison_report,
        "debug_chunk_index": chunk_index,
        "selected_chunk": chunk_results_by_index[chunk_index],
        "selected_plan_entry": chunk_plan_by_index[chunk_index],
        "full_frame": {
            "corrected_resized": np.load(output_path / "offline_corrected_resized.npy", allow_pickle=False).astype(np.float32),
            "final_mask": load_bool_npy(output_path / "offline_final_mask.npy"),
        },
        "chunk_debug": {
            "corrected_resized": np.load(output_path / "chunk_debug" / "chunk_corrected_resized.npy", allow_pickle=False).astype(np.float32),
            "runtime_input_gray": runtime_input_gray,
            "runtime_input_gray_rows": runtime_input_gray_rows,
            "runtime_input_gray_cols": runtime_input_gray_cols,
            "patch_features": patch_features,
            "patch_rows": patch_rows,
            "patch_cols": patch_cols,
            "feature_dim": feature_dim,
            "has_patch_feature_grouping": has_patch_feature_grouping,
            "grouped_dino_source": "patch_feature_grouping" if has_patch_feature_grouping else "raw_score_fallback",
            "dino_score": chunk_dino_score,
            "dino_score_raw": chunk_dino_score_raw,
            "dino_score_grouped": chunk_dino_score,
            "grouped_dino_score": chunk_grouped_dino_score,
            "grouped_combined_score": chunk_grouped_combined_score,
            "grouped_raw_dino_score": chunk_grouped_raw_dino_score,
            "coherence_gate": np.load(output_path / "chunk_debug" / "chunk_coherence_gate.npy", allow_pickle=False).astype(np.float32),
            "hybrid_contrib": np.load(output_path / "chunk_debug" / "chunk_hybrid_contrib.npy", allow_pickle=False).astype(np.float32),
            "combined_score": chunk_combined_score,
            "valid_mask": load_bool_npy(output_path / "chunk_debug" / "chunk_valid_mask.npy"),
            "bridged_mask": load_bool_npy(output_path / "chunk_debug" / "chunk_bridged_mask.npy"),
            "grouped_mask": chunk_grouped_mask,
            "final_mask": load_bool_npy(output_path / "chunk_debug" / "chunk_final_mask.npy"),
            "grouped_boxes": list(cpp_boxes_payload.get("boxes", [])),
            "mapped_row_start": mapped_row_start,
            "mapped_row_stop": mapped_row_stop,
            "output_shape": (output_rows, output_cols),
            "artifact_contract": artifact_contract,
        },
        "artifact_warnings": artifact_warnings,
    }


def build_python_chunk_reference(
    tensor_path: str | Path,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    debug_chunk_index: int = 13,
    dino_device: str | None = None,
    workspace_root: Path = WORKSPACE_ROOT,
) -> dict[str, Any]:
    retry_helper = load_retry_helper_module(workspace_root)
    config = load_validation_config(config_path, workspace_root)
    dino_device = choose_dino_device(dino_device)

    input_record = retry_helper.load_input_record(
        str(Path(tensor_path).expanduser().resolve()),
        input_kind="tensor_npy",
        fft_size=20480,
        noverlap=15360,
        sigmf_capture_index=0,
        sigmf_channel=0,
        sigmf_window_start_s=0.0,
        sigmf_window_duration_s=1.0,
        tensor_target_height=None,
        tensor_target_width=None,
    )
    input_record = apply_tensor_frequency_axis_calibration(input_record, config)

    active_cfg = retry_helper.adapt_chunk_config_for_input_record(
        input_record,
        retry_helper.CoherentPowerConfig(
            chunk_bandwidth_hz=config["chunk_bandwidth_hz"],
            chunk_overlap_hz=config["chunk_overlap_hz"],
            uncalibrated_chunk_fraction=config["uncalibrated_chunk_fraction"],
            uncalibrated_overlap_fraction=config["uncalibrated_overlap_fraction"],
            ignore_sideband_percent=0.0,
            ignore_sideband_hz=config["ignore_sideband_hz"] if config["ignore_sideband_hz"] > 0.0 else None,
            frontend_row_q=config["frontend_row_q"],
            frontend_reference_q=config["frontend_reference_q"],
            frontend_smooth_sigma=config["frontend_smooth_sigma"],
            frontend_max_boost_db=config["frontend_max_boost_db"],
            min_component_size=config["min_component_size"],
            grouping_bridge_freq_px=config["grouping_bridge_freq_px"],
            grouping_bridge_time_px=config["grouping_bridge_time_px"],
            grouping_min_component_size=config["grouping_min_component_size"],
            grouping_min_freq_span_px=config["grouping_min_freq_span_px"],
            grouping_min_time_span_px=config["grouping_min_time_span_px"],
            grouping_min_density=config["grouping_min_density"],
            grouping_time_continuity_ratio=config["grouping_time_continuity_ratio"],
        ),
        target_chunk_rows=1024,
        target_overlap_rows=256,
    )

    freq_axis_hz = np.asarray(input_record["freq_axis_hz"], dtype=np.float32)
    ignore_info = retry_helper.compute_ignore_sideband_rows(
        freq_axis_hz,
        ignore_sideband_percent=float(active_cfg.ignore_sideband_percent),
        min_keep_rows=16,
        ignore_sideband_hz=active_cfg.ignore_sideband_hz,
    )
    correction = retry_helper.apply_global_frontend_correction(
        np.asarray(input_record["sxx_db"], dtype=np.float32),
        row_q=float(active_cfg.frontend_row_q),
        reference_q=float(active_cfg.frontend_reference_q),
        smooth_sigma=float(active_cfg.frontend_smooth_sigma),
        max_boost_db=float(active_cfg.frontend_max_boost_db),
        valid_row_mask=np.asarray(ignore_info["valid_row_mask"], dtype=bool),
    )
    corrected_sxx_db = np.asarray(correction["corrected_sxx_db"], dtype=np.float32)

    chunk_plan = retry_helper.build_frequency_chunks(
        freq_axis_hz,
        chunk_bandwidth_hz=float(active_cfg.chunk_bandwidth_hz),
        chunk_overlap_hz=float(active_cfg.chunk_overlap_hz),
        min_rows=16,
        valid_row_mask=np.asarray(correction["valid_row_mask"], dtype=bool),
        uncalibrated_chunk_fraction=float(active_cfg.uncalibrated_chunk_fraction),
        uncalibrated_overlap_fraction=float(active_cfg.uncalibrated_overlap_fraction),
    )
    chunk = next((entry for entry in chunk_plan if int(entry["chunk_index"]) == int(debug_chunk_index)), None)
    if chunk is None:
        raise ValueError(f"Chunk index {debug_chunk_index} not found in Python chunk plan")

    row_start = int(chunk["row_start"])
    row_stop = int(chunk["row_stop"])
    corrected_chunk = np.asarray(corrected_sxx_db[row_start:row_stop, :], dtype=np.float32)
    local_valid_row_mask = np.asarray(correction["valid_row_mask"], dtype=bool)[row_start:row_stop]
    valid_score_mask = np.repeat(local_valid_row_mask[:, None], corrected_chunk.shape[1], axis=1)
    valid_values = corrected_chunk[valid_score_mask] if np.any(valid_score_mask) else corrected_chunk.reshape(-1)
    dino_db_min = float(np.percentile(valid_values, 2.0))
    dino_db_max = float(np.percentile(valid_values, 98.0))

    python_retry = build_python_retry_hybrid_products(
        retry_helper,
        corrected_chunk,
        valid_score_mask,
        dino_repo_dir=config["model_repo_path"],
        dino_weights_path=config["weights_path"],
        dino_model_name=config["model_name"],
        dino_device=dino_device,
        min_component_size=config["min_component_size"],
        dino_db_min=dino_db_min,
        dino_db_max=dino_db_max,
    )
    python_support = build_retry_frequency_support_mask(python_retry["hybrid_dino_contrib"], valid_score_mask)
    python_grouping = retry_helper.group_signal_mask_regions(
        python_support["final_mask"],
        score_map=python_support["combined_score"],
        valid_row_mask=local_valid_row_mask,
        bridge_freq_px=config["grouping_bridge_freq_px"],
        bridge_time_px=config["grouping_bridge_time_px"],
        min_component_size=config["grouping_min_component_size"],
        min_freq_span_px=config["grouping_min_freq_span_px"],
        min_time_span_px=config["grouping_min_time_span_px"],
        min_density=config["grouping_min_density"],
        time_continuity_ratio=config["grouping_time_continuity_ratio"],
    )
    return {
        "config": config,
        "dino_device": dino_device,
        "input_record": input_record,
        "active_cfg": active_cfg,
        "ignore_info": ignore_info,
        "correction": correction,
        "chunk_plan": chunk_plan,
        "selected_chunk": chunk,
        "corrected_chunk": corrected_chunk,
        "valid_row_mask": local_valid_row_mask,
        "valid_score_mask": valid_score_mask,
        "dino_db_min": dino_db_min,
        "dino_db_max": dino_db_max,
        "python_retry": python_retry,
        "python_support": python_support,
        "python_grouping": python_grouping,
    }


def build_chunk_comparison_bundle(
    tensor_path: str | Path,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    output_dir: str | Path | None = None,
    debug_chunk_index: int = 13,
    dino_device: str | None = None,
    workspace_root: Path = WORKSPACE_ROOT,
) -> dict[str, Any]:
    resolved_tensor_path = Path(tensor_path).expanduser().resolve()
    resolved_output_dir = Path(output_dir).expanduser().resolve() if output_dir else default_output_dir(resolved_tensor_path)
    cpp_bundle = load_cpp_artifact_bundle(resolved_output_dir, debug_chunk_index, workspace_root)
    python_bundle = build_python_chunk_reference(resolved_tensor_path, config_path, debug_chunk_index, dino_device, workspace_root)
    retry_helper = load_retry_helper_module(workspace_root)

    cpp_chunk = cpp_bundle["chunk_debug"]
    target_shape = tuple(int(value) for value in cpp_chunk["corrected_resized"].shape)
    python_retry = python_bundle["python_retry"]
    python_support = python_bundle["python_support"]
    python_grouping = python_bundle["python_grouping"]
    python_corrected_cpp_grid = resize_array_to_shape(python_bundle["corrected_chunk"], target_shape, order=1)
    python_valid_cpp_grid = resize_mask_to_shape(python_bundle["valid_score_mask"], target_shape)
    python_dino_score_variants = build_python_raw_feature_energy_score(python_retry, target_shape)
    python_dino_input_debug = build_signal_agnostic_input_debug(
        retry_helper,
        python_bundle["corrected_chunk"],
        target_shape,
        db_min=python_bundle["dino_db_min"],
        db_max=python_bundle["dino_db_max"],
    )
    cpp_dino_input_debug = build_signal_agnostic_input_debug(
        retry_helper,
        cpp_chunk["corrected_resized"],
        target_shape,
        db_min=python_bundle["dino_db_min"],
        db_max=python_bundle["dino_db_max"],
    )
    cpp_runtime_input_gray = cpp_chunk.get("runtime_input_gray")
    cpp_runtime_shape = None
    if cpp_runtime_input_gray is not None:
        cpp_runtime_shape = (
            int(cpp_chunk.get("runtime_input_gray_rows", 0) or 0),
            int(cpp_chunk.get("runtime_input_gray_cols", 0) or 0),
        )
    python_dino_patch_size = int(
        python_bundle["python_retry"]["dino_texture_experiment"]["dino"].get("patch_size", 16)
    )
    local_chunk_freq_axis_hz = np.asarray(
        python_bundle["input_record"]["freq_axis_hz"][int(python_bundle["selected_chunk"]["row_start"]):int(python_bundle["selected_chunk"]["row_stop"])],
        dtype=np.float32,
    )
    python_expected_runtime_input = build_python_expected_dino_runtime_input(
        retry_helper,
        python_bundle["corrected_chunk"],
        local_chunk_freq_axis_hz,
        ignore_sideband_hz=0.0,
        patch_size=python_dino_patch_size,
        target_shape=target_shape,
        db_min=python_bundle["dino_db_min"],
        db_max=python_bundle["dino_db_max"],
    )
    python_runtime_retry = None
    python_runtime_raw_variants = None
    python_runtime_mapped_raw = None
    python_runtime_mapped_grouped = None
    if python_expected_runtime_input is not None:
        runtime_chunk = np.asarray(python_expected_runtime_input["runtime_chunk"], dtype=np.float32)
        runtime_valid_mask = np.ones(runtime_chunk.shape, dtype=bool)
        python_runtime_retry = build_python_retry_hybrid_products(
            retry_helper,
            runtime_chunk,
            runtime_valid_mask,
            dino_repo_dir=python_bundle["config"]["model_repo_path"],
            dino_weights_path=python_bundle["config"]["weights_path"],
            dino_model_name=python_bundle["config"]["model_name"],
            dino_device=python_bundle["dino_device"],
            min_component_size=python_bundle["config"]["min_component_size"],
            dino_db_min=python_bundle["dino_db_min"],
            dino_db_max=python_bundle["dino_db_max"],
        )
        python_runtime_raw_variants = build_python_raw_feature_energy_score(
            python_runtime_retry,
            tuple(int(value) for value in python_expected_runtime_input["expected_shape"]),
        )
        python_runtime_mapped_raw = embed_runtime_crop_to_chunk_grid(
            np.asarray(python_runtime_raw_variants["raw_pixel_score"], dtype=np.float32),
            tuple(int(value) for value in python_bundle["corrected_chunk"].shape),
            row_start=int(python_expected_runtime_input["row_start"]),
            expected_shape=tuple(int(value) for value in python_expected_runtime_input["expected_shape"]),
            target_shape=target_shape,
            order=1,
        )
        python_runtime_mapped_grouped = embed_runtime_crop_to_chunk_grid(
            np.asarray(python_runtime_raw_variants["grouped_pixel_score"], dtype=np.float32),
            tuple(int(value) for value in python_bundle["corrected_chunk"].shape),
            row_start=int(python_expected_runtime_input["row_start"]),
            expected_shape=tuple(int(value) for value in python_expected_runtime_input["expected_shape"]),
            target_shape=target_shape,
            order=1,
        )
    artifact_warnings = list(cpp_bundle.get("artifact_warnings", []))
    if python_expected_runtime_input is not None:
        expected_runtime_shape = tuple(int(value) for value in python_expected_runtime_input["expected_shape"])
        if cpp_runtime_shape is None or cpp_runtime_shape[0] <= 0 or cpp_runtime_shape[1] <= 0:
            artifact_warnings.append(
                f"Expected a C++ runtime grayscale dump for runtime crop {expected_runtime_shape}, but the artifact bundle does not contain a valid runtime_input_gray shape."
            )
        elif cpp_runtime_shape != expected_runtime_shape:
            artifact_warnings.append(
                f"C++ runtime grayscale shape {cpp_runtime_shape} does not match the Python-expected runtime crop {expected_runtime_shape}."
            )
    if artifact_warnings:
        formatted_warnings = "\n".join(f"- {warning}" for warning in artifact_warnings)
        raise RuntimeError(
            "Offline validator artifact bundle is stale or was produced by a mismatched validator binary, so this notebook comparison would not be apples-to-apples:\n"
            f"{formatted_warnings}\n"
            "Rebuild the validator binary actually used by validate_offline_dino_subsection.sh, rerun the validator, then rerun Cell 3."
        )
    if cpp_runtime_input_gray is not None:
        cpp_runtime_input_gray = np.asarray(cpp_runtime_input_gray, dtype=np.float32)
        runtime_rows = int(cpp_chunk.get("runtime_input_gray_rows", 0) or 0)
        runtime_cols = int(cpp_chunk.get("runtime_input_gray_cols", 0) or 0)
        if runtime_rows > 0 and runtime_cols > 0 and cpp_runtime_input_gray.shape == (runtime_rows, runtime_cols):
            cpp_runtime_input_gray = resize_array_to_shape(cpp_runtime_input_gray, target_shape, order=1)
    python_coherence_cpp_grid = build_structure_tensor_coherence_products(
        retry_helper,
        python_corrected_cpp_grid,
        python_valid_cpp_grid,
        min_component_size=python_bundle["config"]["min_component_size"],
    )
    python_coherence_source_mapped = np.asarray(
        retry_helper.resize_float_image(
            np.asarray(python_retry["coherence_gate_px"], dtype=np.float32),
            width=int(target_shape[1]),
            height=int(target_shape[0]),
        ),
        dtype=np.float32,
    )

    python_mapped = {
        "corrected": python_corrected_cpp_grid,
        "coherence_gate": python_coherence_source_mapped,
        "coherence_gate_source_mapped": python_coherence_source_mapped,
        "coherence_gate_cpp_grid": np.asarray(python_coherence_cpp_grid["coherence_gate_px"], dtype=np.float32),
        "coherence_region_mask_cpp_grid": np.asarray(python_coherence_cpp_grid["coherence_region_mask"], dtype=bool),
        "dino_score": np.asarray(python_dino_score_variants["raw_pixel_score"], dtype=np.float32),
        "dino_score_raw_feature": np.asarray(python_dino_score_variants["raw_pixel_score"], dtype=np.float32),
        "dino_score_grouped": np.asarray(python_dino_score_variants["grouped_pixel_score"], dtype=np.float32),
        "dino_score_runtime_raw_feature": None if python_runtime_mapped_raw is None else np.asarray(python_runtime_mapped_raw, dtype=np.float32),
        "dino_score_runtime_grouped": None if python_runtime_mapped_grouped is None else np.asarray(python_runtime_mapped_grouped, dtype=np.float32),
        "dino_input_gray": np.asarray(python_dino_input_debug["input_gray"], dtype=np.float32),
        "dino_input_fixed_gray": np.asarray(python_dino_input_debug["fixed_gray"], dtype=np.float32),
        "dino_input_expected_runtime_gray": None if python_expected_runtime_input is None else np.asarray(python_expected_runtime_input["input_gray"], dtype=np.float32),
        "hybrid_contrib": resize_array_to_shape(python_retry["hybrid_dino_contrib"], target_shape, order=1),
        "combined_score": resize_array_to_shape(python_support["combined_score"], target_shape, order=1),
        "valid_mask": python_valid_cpp_grid,
        "hybrid_mask": resize_mask_to_shape(python_retry["hybrid_mask"], target_shape),
        "final_mask": resize_mask_to_shape(python_support["final_mask"], target_shape),
        "bridged_mask": resize_mask_to_shape(np.asarray(python_grouping["bridged_mask"], dtype=bool), target_shape),
        "grouped_mask": resize_mask_to_shape(np.asarray(python_grouping["grouped_mask"], dtype=bool), target_shape),
        "grouped_boxes": [
            scale_box_to_shape(
                box,
                tuple(int(value) for value in np.asarray(python_grouping["grouped_mask"]).shape),
                target_shape,
            )
            for box in list(python_grouping.get("boxes", []))
        ],
    }
    dino_input_valid_mask = np.ones(target_shape, dtype=bool) if (cpp_runtime_input_gray is not None and python_expected_runtime_input is not None) else python_mapped["valid_mask"]

    cpp_metrics = {
        "coherence_gate": mask_metrics(cpp_chunk["coherence_gate"] >= np.quantile(cpp_chunk["coherence_gate"], 0.80), python_mapped["coherence_gate"] >= np.quantile(python_mapped["coherence_gate"], 0.80)),
        "final_mask": mask_metrics(cpp_chunk["final_mask"], python_mapped["final_mask"]),
        "bridged_mask": mask_metrics(cpp_chunk["bridged_mask"], python_mapped["bridged_mask"]),
        "grouped_mask": mask_metrics(cpp_chunk["grouped_mask"], python_mapped["grouped_mask"]),
    }

    coherence_diagnostics = {
        "source_mapped": {
            "float_metrics": float_map_metrics(cpp_chunk["coherence_gate"], python_mapped["coherence_gate_source_mapped"], python_mapped["valid_mask"]),
            "mask_metrics": mask_metrics(
                cpp_chunk["coherence_gate"] >= np.quantile(cpp_chunk["coherence_gate"][python_mapped["valid_mask"]], 0.80),
                python_mapped["coherence_gate_source_mapped"] >= np.quantile(python_mapped["coherence_gate_source_mapped"][python_mapped["valid_mask"]], 0.80),
            ),
        },
        "cpp_grid": {
            "float_metrics": float_map_metrics(cpp_chunk["coherence_gate"], python_mapped["coherence_gate_cpp_grid"], python_mapped["valid_mask"]),
            "mask_metrics": mask_metrics(
                cpp_chunk["coherence_gate"] >= np.quantile(cpp_chunk["coherence_gate"][python_mapped["valid_mask"]], 0.80),
                python_mapped["coherence_gate_cpp_grid"] >= np.quantile(python_mapped["coherence_gate_cpp_grid"][python_mapped["valid_mask"]], 0.80),
            ),
        },
    }

    dino_diagnostics = {
        "input_gray": {
            "float_metrics": float_map_metrics(
                cpp_runtime_input_gray if cpp_runtime_input_gray is not None else cpp_dino_input_debug["input_gray"],
                python_mapped["dino_input_expected_runtime_gray"] if python_mapped["dino_input_expected_runtime_gray"] is not None else python_mapped["dino_input_gray"],
                dino_input_valid_mask,
            ),
        },
        "grouped": {
            "float_metrics": float_map_metrics(cpp_chunk["dino_score_grouped"], python_mapped["dino_score_grouped"], python_mapped["valid_mask"]),
            "mask_metrics": mask_metrics(
                cpp_chunk["dino_score_grouped"] >= np.quantile(cpp_chunk["dino_score_grouped"][python_mapped["valid_mask"]], 0.80),
                python_mapped["dino_score_grouped"] >= np.quantile(python_mapped["dino_score_grouped"][python_mapped["valid_mask"]], 0.80),
            ),
        },
        "raw_feature": {
            "float_metrics": float_map_metrics(
                cpp_chunk["dino_score_raw"],
                python_mapped["dino_score_runtime_raw_feature"] if python_mapped["dino_score_runtime_raw_feature"] is not None else python_mapped["dino_score_raw_feature"],
                python_mapped["valid_mask"],
            ),
            "mask_metrics": mask_metrics(
                cpp_chunk["dino_score_raw"] >= np.quantile(cpp_chunk["dino_score_raw"][python_mapped["valid_mask"]], 0.80),
                (python_mapped["dino_score_runtime_raw_feature"] if python_mapped["dino_score_runtime_raw_feature"] is not None else python_mapped["dino_score_raw_feature"])
                >= np.quantile((python_mapped["dino_score_runtime_raw_feature"] if python_mapped["dino_score_runtime_raw_feature"] is not None else python_mapped["dino_score_raw_feature"])[python_mapped["valid_mask"]], 0.80),
            ),
        },
        "python_patch_shape": list(int(value) for value in python_dino_score_variants["patch_shape"]),
    }

    cpp_boxes = list(cpp_chunk["grouped_boxes"])
    python_boxes = list(python_grouping.get("boxes", []))
    box_summary = {
        "cpp_box_count": len(cpp_boxes),
        "python_box_count": len(python_boxes),
        "exact_signature_match": sorted(box_signature(box) for box in cpp_boxes) == sorted(box_signature(box) for box in python_boxes),
        "cpp_to_python": best_iou_summary(cpp_boxes, python_boxes),
        "python_to_cpp": best_iou_summary(python_boxes, cpp_boxes),
    }

    return {
        "tensor_path": resolved_tensor_path,
        "config_path": Path(config_path).expanduser().resolve(),
        "output_dir": resolved_output_dir,
        "cpp": cpp_bundle,
        "cpp_proxy": {
            "dino_input_gray": np.asarray(cpp_dino_input_debug["input_gray"], dtype=np.float32),
            "dino_input_fixed_gray": np.asarray(cpp_dino_input_debug["fixed_gray"], dtype=np.float32),
            "runtime_input_gray": cpp_runtime_input_gray,
        },
        "python": python_bundle,
        "python_mapped": python_mapped,
        "metrics": cpp_metrics,
        "coherence_diagnostics": coherence_diagnostics,
        "dino_diagnostics": dino_diagnostics,
        "box_summary": box_summary,
    }


def disagreement_mask(lhs: np.ndarray, rhs: np.ndarray) -> dict[str, np.ndarray]:
    lhs = np.asarray(lhs, dtype=bool)
    rhs = np.asarray(rhs, dtype=bool)
    return {
        "lhs_only": np.logical_and(lhs, ~rhs),
        "rhs_only": np.logical_and(~lhs, rhs),
        "overlap": np.logical_and(lhs, rhs),
    }


def scale_box_to_shape(box: dict[str, Any], source_shape: tuple[int, int], target_shape: tuple[int, int]) -> dict[str, int]:
    src_rows, src_cols = int(source_shape[0]), int(source_shape[1])
    dst_rows, dst_cols = int(target_shape[0]), int(target_shape[1])
    return {
        "freq_start": int(np.floor(int(box.get("freq_start", 0)) * dst_rows / max(src_rows, 1))),
        "freq_stop": int(np.ceil(int(box.get("freq_stop", 0)) * dst_rows / max(src_rows, 1))),
        "time_start": int(np.floor(int(box.get("time_start", 0)) * dst_cols / max(src_cols, 1))),
        "time_stop": int(np.ceil(int(box.get("time_stop", 0)) * dst_cols / max(src_cols, 1))),
    }


def build_notebook_display_bundle(comparison: dict[str, Any]) -> dict[str, Any]:
    cpp_bundle = comparison["cpp"]
    python_bundle = comparison["python"]
    python_mapped = comparison["python_mapped"]
    cpp_chunk = cpp_bundle["chunk_debug"]
    cpp_full = cpp_bundle["full_frame"]
    script_report = cpp_bundle["comparison_report"]
    has_patch_feature_grouping = bool(cpp_chunk.get("has_patch_feature_grouping", False))
    grouped_surface_label = "Grouped DINO score" if has_patch_feature_grouping else "Fallback DINO score"
    grouped_surface_note = (
        "Current C++ artifact includes patch-feature exports, so this compares the grouped patch-feature score surface directly."
        if has_patch_feature_grouping
        else "Current C++ artifact is missing patch-feature exports, so the saved C++ DINO surface is a fallback derived from the raw runtime score_map rather than the grouped patch-feature path."
    )

    summary_rows = [
        {
            "metric": "chunk_count",
            "value": int(cpp_bundle["summary"]["chunk_count"]),
        },
        {
            "metric": "dino_input_mae",
            "value": float(comparison["dino_diagnostics"]["input_gray"]["float_metrics"]["mae"]),
        },
        {
            "metric": "dino_input_corr",
            "value": float(comparison["dino_diagnostics"]["input_gray"]["float_metrics"]["corr"]),
        },
        {
            "metric": "raw_dino_mae",
            "value": float(comparison["dino_diagnostics"]["raw_feature"]["float_metrics"]["mae"]),
        },
        {
            "metric": "raw_dino_corr",
            "value": float(comparison["dino_diagnostics"]["raw_feature"]["float_metrics"]["corr"]),
        },
        {
            "metric": "script_grouped_mask_agreement",
            "value": None if script_report is None else float(script_report["grouped_mask_metrics"]["pixel_agreement"]),
        },
        {
            "metric": "script_grouped_mask_iou",
            "value": None if script_report is None else float(script_report["grouped_mask_metrics"]["iou"]),
        },
        {
            "metric": "mapped_final_mask_iou",
            "value": float(comparison["metrics"]["final_mask"]["iou"]),
        },
        {
            "metric": "mapped_bridged_mask_iou",
            "value": float(comparison["metrics"]["bridged_mask"]["iou"]),
        },
        {
            "metric": "mapped_grouped_mask_iou",
            "value": float(comparison["metrics"]["grouped_mask"]["iou"]),
        },
        {
            "metric": "cpp_box_count",
            "value": int(comparison["box_summary"]["cpp_box_count"]),
        },
        {
            "metric": "python_box_count",
            "value": int(comparison["box_summary"]["python_box_count"]),
        },
        {
            "metric": "cpp_grouped_patch_features_present",
            "value": int(has_patch_feature_grouping),
        },
    ]

    script_report_rows = []
    if script_report is not None:
        script_report_rows.append(
            {
                "kind": "script_report",
                "chunk_plan_row_match": bool(script_report["chunk_plan_match"]["row_start_matches"]) and bool(script_report["chunk_plan_match"]["row_stop_matches"]),
                "chunk_plan_freq_match": bool(script_report["chunk_plan_match"]["freq_start_hz_matches"]) and bool(script_report["chunk_plan_match"]["freq_stop_hz_matches"]),
                "grouped_mask_agreement": float(script_report["grouped_mask_metrics"]["pixel_agreement"]),
                "grouped_mask_iou": float(script_report["grouped_mask_metrics"]["iou"]),
                "cpp_box_count": int(script_report["box_comparison"]["cpp_box_count"]),
                "python_box_count": int(script_report["box_comparison"]["python_box_count"]),
                "exact_box_signature_match": bool(script_report["box_comparison"]["exact_signature_match"]),
            }
        )

    overview_panels = [
        ("Offline corrected resized", cpp_full["corrected_resized"], "magma"),
        ("Offline final mask", cpp_full["final_mask"].astype(np.float32), "gray"),
    ]

    coherence_variants = [
        {
            "title": "Python source gate mapped to C++ grid",
            "python_image": python_mapped["coherence_gate_source_mapped"],
            "diagnostics": comparison["coherence_diagnostics"]["source_mapped"],
        },
        {
            "title": "Python gate recomputed on resized corrected chunk",
            "python_image": python_mapped["coherence_gate_cpp_grid"],
            "diagnostics": comparison["coherence_diagnostics"]["cpp_grid"],
        },
    ]

    runtime_input_gray = comparison["cpp_proxy"].get("runtime_input_gray")
    runtime_python_input_gray = python_mapped.get("dino_input_expected_runtime_gray")
    dino_input_panel = {
        "cpp_image": np.asarray(runtime_input_gray, dtype=np.float32) if runtime_input_gray is not None else comparison["cpp_proxy"]["dino_input_gray"],
        "cpp_title": "C++ exact pre-model grayscale input" if runtime_input_gray is not None else "C++-side signal-agnostic DINO input proxy",
        "python_image": runtime_python_input_gray if runtime_python_input_gray is not None else python_mapped["dino_input_gray"],
        "python_title": "Python expected pre-model grayscale input" if runtime_python_input_gray is not None else "Python signal-agnostic DINO input",
        "diagnostics": comparison["dino_diagnostics"]["input_gray"]["float_metrics"],
        "is_runtime_dump": bool(runtime_input_gray is not None),
    }

    raw_dino_panel = {
        "cpp_title": "C++ raw runtime score_map (feature-energy proxy)"
        if has_patch_feature_grouping
        else "C++ raw runtime score_map (also backing the current fallback DINO surface)",
        "cpp_image": cpp_chunk["dino_score_raw"],
        "python_title": "Python runtime-cropped raw feature-energy score"
        if python_mapped.get("dino_score_runtime_raw_feature") is not None
        else "Python raw feature-energy score",
        "python_image": python_mapped["dino_score_runtime_raw_feature"]
        if python_mapped.get("dino_score_runtime_raw_feature") is not None
        else python_mapped["dino_score_raw_feature"],
        "diagnostics": comparison["dino_diagnostics"]["raw_feature"],
        "valid_mask": python_mapped["valid_mask"],
        "display_vmin": 0.0,
        "display_vmax": 1.0,
    }

    python_grouped_combined = np.where(python_mapped["grouped_mask"], python_mapped["combined_score"], 0.0).astype(np.float32)
    grouped_note_rows = [
        {
            "comparison": "grouped_score_surface",
            "cpp_surface": "grouped DINO score from C++ patch-feature grouping"
            if has_patch_feature_grouping
            else "fallback C++ DINO score derived without exported patch features",
            "python_surface": "Python grouped DINO support score",
            "note": grouped_surface_note,
        },
        {
            "comparison": "raw_dino_surface",
            "cpp_surface": "runtime score_map exported by TorchScript runtime",
            "python_surface": "runtime-cropped raw feature-energy proxy built from Python features"
            if python_mapped.get("dino_score_runtime_raw_feature") is not None
            else "raw feature-energy proxy built from Python features",
            "note": "This row now compares the raw feature-energy proxy on the runtime-cropped DINO slice contract when that runtime crop can be reconstructed in Python. It is still not the full grouped/postprocessed Python DINO path.",
        }
    ]

    stage_pairs = [
        ("Corrected", cpp_chunk["corrected_resized"], python_mapped["corrected"], "magma"),
        (
            grouped_surface_label,
            cpp_chunk["dino_score_grouped"],
            python_mapped["dino_score_runtime_grouped"]
            if (not has_patch_feature_grouping and python_mapped.get("dino_score_runtime_grouped") is not None)
            else python_mapped["dino_score_grouped"],
            "plasma",
        ),
        ("Hybrid contrib", cpp_chunk["hybrid_contrib"], python_mapped["hybrid_contrib"], "cividis"),
        ("Combined score", cpp_chunk["combined_score"], python_mapped["combined_score"], "plasma"),
        ("Grouped combined score", cpp_chunk["grouped_combined_score"], python_grouped_combined, "plasma"),
        ("Final mask", cpp_chunk["final_mask"].astype(np.float32), python_mapped["final_mask"].astype(np.float32), "gray"),
    ]

    cpp_boxes = list(cpp_chunk["grouped_boxes"])
    python_boxes_scaled = list(python_mapped.get("grouped_boxes", []))
    match_rows = build_box_match_rows(cpp_boxes, python_boxes_scaled)

    return {
        "summary_rows": summary_rows,
        "script_report_rows": script_report_rows,
        "overview_panels": overview_panels,
        "coherence_variants": coherence_variants,
        "dino_input_panel": dino_input_panel,
        "raw_dino_panel": raw_dino_panel,
        "grouped_note_rows": grouped_note_rows,
        "stage_pairs": stage_pairs,
        "cpp_boxes": cpp_boxes,
        "python_boxes_scaled": python_boxes_scaled,
        "match_rows": match_rows,
    }


def build_box_match_rows(cpp_boxes: list[dict[str, Any]], python_boxes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, cpp_box in enumerate(cpp_boxes):
        best_python_index = -1
        best_overlap = 0.0
        for candidate_index, python_box in enumerate(python_boxes):
            overlap = box_iou(cpp_box, python_box)
            if overlap > best_overlap:
                best_overlap = overlap
                best_python_index = candidate_index
        rows.append(
            {
                "cpp_index": index,
                "best_python_index": best_python_index,
                "best_iou": float(best_overlap),
                "cpp_box": cpp_box,
                "python_box": python_boxes[best_python_index] if best_python_index >= 0 else None,
            }
        )
    return rows