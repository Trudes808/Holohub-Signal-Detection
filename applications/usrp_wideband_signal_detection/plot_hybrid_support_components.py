#!/usr/bin/env python3

import argparse
import json
import math
from pathlib import Path
import sys
import tempfile

import matplotlib.pyplot as plt
import numpy as np


HOST_REPO_ROOT = Path(__file__).resolve().parents[2]
HOST_SPECTROGRAM_ROOT = Path("/tmp/usrp_spectrograms")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the saved hybrid support components for a CUDA DINO debug chunk bundle."
    )
    parser.add_argument(
        "summary",
        type=Path,
        help="Path to chunk_debug_summary.json produced by the CUDA DINO operator artifact bundle.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output PNG path. Defaults to chunk_hybrid_support_components.png next to the summary.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot interactively after saving it.",
    )
    return parser.parse_args()


def load_summary(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_artifact_path(raw_path: str, summary_path: Path) -> Path:
    path = Path(str(raw_path or ""))
    text = str(path)
    if not text:
        return path
    if text.startswith("/workspace/spectrograms/"):
        return HOST_SPECTROGRAM_ROOT / text.removeprefix("/workspace/spectrograms/")
    if text == "/workspace/spectrograms":
        return HOST_SPECTROGRAM_ROOT
    if text.startswith("/workspace/holohub/"):
        return HOST_REPO_ROOT / text.removeprefix("/workspace/holohub/")
    if text == "/workspace/holohub":
        return HOST_REPO_ROOT
    if text.startswith("/workspace/holohub-dev/"):
        return HOST_REPO_ROOT / text.removeprefix("/workspace/holohub-dev/")
    if text == "/workspace/holohub-dev":
        return HOST_REPO_ROOT
    if path.is_absolute():
        return path
    return (summary_path.parent / path).resolve()


def existing_component_specs(summary: dict) -> list[tuple[str, str, str]]:
    specs = [
        ("DINO Raw Deweighted", "dino_score_raw_deweighted_npy", "magma"),
        ("Coherence Gate", "coherence_gate_npy", "viridis"),
        ("Veto Input Score", "initial_product_npy", "magma"),
        ("Keep Freq", "hybrid_keep_freq_npy", "magma"),
        ("Keep Res", "hybrid_keep_res_npy", "magma"),
        ("Combined Score", "combined_score_npy", "magma"),
        ("Coherence Band", "coherence_band_mask_npy", "gray"),
        ("DINO Structure", "dino_structure_mask_npy", "gray"),
        ("Seed Mask", "hybrid_seed_mask_npy", "gray"),
        ("Closed Mask", "hybrid_closed_mask_npy", "gray"),
        ("Filled Mask", "hybrid_filled_mask_npy", "gray"),
        ("Component Filtered", "hybrid_component_filtered_mask_npy", "gray"),
        ("Coherence Rescue", "coherence_rescue_mask_npy", "gray"),
        ("Final Mask", "final_mask_npy", "gray"),
    ]
    return [spec for spec in specs if summary.get(spec[1])]


def plot_components(summary: dict, summary_path: Path, output_path: Path, show_plot: bool) -> None:
    specs = existing_component_specs(summary)
    if not specs:
        raise RuntimeError("No hybrid component paths were found in the summary JSON.")

    thresholds = summary.get("hybrid_thresholds", {})
    cols = min(5, len(specs))
    rows = math.ceil(len(specs) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.4 * cols, 3.6 * rows), squeeze=False)
    axes_flat = axes.flatten()

    title_bits = []
    if "seed_freq" in thresholds:
        title_bits.append(f"seed_freq={thresholds['seed_freq']:.3f}")
    if "seed_res" in thresholds:
        title_bits.append(f"seed_res={thresholds['seed_res']:.3f}")
    if "combined" in thresholds:
        title_bits.append(f"combined={thresholds['combined']:.3f}")

    for axis, (title, key, cmap) in zip(axes_flat, specs):
        array = np.load(resolve_artifact_path(summary[key], summary_path))
        image = axis.imshow(array, origin="lower", aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    for axis in axes_flat[len(specs):]:
        axis.axis("off")

    chunk_index = summary.get("chunk_index", "?")
    chunk_count = summary.get("chunk_count", "?")
    figure_title = f"Hybrid Support Components: chunk {chunk_index}/{chunk_count}"
    if title_bits:
        figure_title += " | " + ", ".join(title_bits)
    fig.suptitle(figure_title)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    actual_output_path = output_path
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160, bbox_inches="tight")
    except PermissionError:
        fallback_dir = Path(tempfile.gettempdir()) / "usrp_hybrid_support_plots"
        fallback_dir.mkdir(parents=True, exist_ok=True)
        actual_output_path = fallback_dir / output_path.name
        print(
            f"Permission denied writing {output_path}; saved hybrid support plot to {actual_output_path} instead.",
            file=sys.stderr,
        )
        fig.savefig(actual_output_path, dpi=160, bbox_inches="tight")

    if show_plot:
        plt.show()
    plt.close(fig)
    return actual_output_path


def main() -> None:
    args = parse_args()
    summary = load_summary(args.summary)
    output_path = args.output or args.summary.with_name("chunk_hybrid_support_components.png")
    actual_output_path = plot_components(summary, args.summary, output_path, args.show)
    print(actual_output_path)


if __name__ == "__main__":
    main()