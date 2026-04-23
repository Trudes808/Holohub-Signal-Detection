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
    "runtime_input_gray_npy",
    "dino_score_raw_npy",
    "dino_score_raw_deweighted_npy",
    "coherence_gate_npy",
    "hybrid_contrib_npy",
    "combined_score_npy",
    "final_mask_npy",
]


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


def load_stage_array(summary: dict, key: str) -> np.ndarray:
    path = resolve_host_path(str(summary.get(key, "") or ""))
    if not path.exists():
        raise FileNotFoundError(f"Missing artifact for {key}: {path}")
    return np.load(path, allow_pickle=False).astype(np.float32)


def common_stage_keys(summary: dict, preferred: Iterable[str]) -> list[str]:
    keys = []
    for key in preferred:
                if key in summary and str(summary.get(key, "") or ""):
                        keys.append(key)
    return keys


def stage_title(key: str) -> str:
    title = key.removesuffix("_npy")
    return title.replace("_", " ")


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
    args = parser.parse_args()

    cuda_output_dir = Path(args.cuda_output_dir).expanduser().resolve()
    _, cuda_summary = resolve_debug_summary(cuda_output_dir)
    selected_stages = common_stage_keys(cuda_summary, args.stages)
    if not selected_stages:
        raise SystemExit("No requested stages were found in the CUDA chunk debug summary")

    cuda_arrays = [(key, load_stage_array(cuda_summary, key)) for key in selected_stages]
    output_path = Path(args.output).expanduser().resolve() if args.output else (cuda_output_dir / "chunk_debug" / "cuda_artifact_compare.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.reference_output_dir:
        reference_output_dir = Path(args.reference_output_dir).expanduser().resolve()
        _, reference_summary = resolve_debug_summary(reference_output_dir)
        reference_stages = common_stage_keys(reference_summary, selected_stages)
        if reference_stages != selected_stages:
            missing = sorted(set(selected_stages) - set(reference_stages))
            raise SystemExit(f"Reference bundle is missing requested stages: {missing}")
        reference_arrays = [(key, load_stage_array(reference_summary, key)) for key in selected_stages]
        plot_comparison(cuda_arrays, reference_arrays, output_path)
        print(f"Wrote CUDA vs reference artifact comparison plot: {output_path}")
    else:
        plot_single_bundle(cuda_arrays, output_path)
        print(f"Wrote CUDA artifact plot: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())