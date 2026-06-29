#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess
import sys



REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "applications/usrp_wideband_signal_detection/config_cuda_dino_performance_single_channel.yaml"
CONTAINER_NAME = "usrp_x410_signal_detection_demo"
HOST_SCRATCH_ROOT = Path("/tmp/usrp_spectrograms")
CONTAINER_SCRATCH_ROOT = Path("/workspace/spectrograms")
CONTAINER_REPO_ROOT = Path("/workspace/holohub")
GENERATED_CONFIG_ROOT = REPO_ROOT / "applications/usrp_wideband_signal_detection/generated_configs"
CONTAINER_BINARY_CANDIDATES = (
    CONTAINER_REPO_ROOT / "build/usrp_wideband_signal_detection/applications/usrp_wideband_signal_detection/run_offline_cuda_detector_eval",
    CONTAINER_REPO_ROOT / "build/applications/usrp_wideband_signal_detection/run_offline_cuda_detector_eval",
    CONTAINER_REPO_ROOT / "build/usrp_wideband_signal_detection/run_offline_cuda_detector_eval",
)
DEFAULT_OUTPUT_ROOT = HOST_SCRATCH_ROOT / "offline_cuda_dino"
TWO_CHUNK_CHUNK_BANDWIDTH_HZ = 253000000.0
TWO_CHUNK_CHUNK_OVERLAP_HZ = 20019531.25


def _format_yaml_scalar(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.12g}"
    text = str(value)
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def set_block_scalar_values(config_text: str, block_name: str, values: dict[str, object]) -> str:
    lines = config_text.splitlines()
    block_start = None
    block_end = len(lines)
    for index, line in enumerate(lines):
        if line.strip() == f"{block_name}:" and not line[:1].isspace():
            block_start = index
            break

    if block_start is None:
        raise ValueError(f"Missing top-level YAML block: {block_name}")

    for index in range(block_start + 1, len(lines)):
        stripped = lines[index].strip()
        if stripped and not lines[index][:1].isspace():
            block_end = index
            break

    indent = "  "
    for key, value in values.items():
        replacement = f"{indent}{key}: {_format_yaml_scalar(value)}"
        key_prefix = f"{key}:"
        replaced = False
        for index in range(block_start + 1, block_end):
            stripped = lines[index].strip()
            if stripped.startswith(key_prefix):
                lines[index] = replacement
                replaced = True
                break
        if not replaced:
            lines.insert(block_end, replacement)
            block_end += 1

    return "\n".join(lines) + "\n"


def detector_chunk_overrides(target_chunk_count: int | None, debug_chunk_index: int | None) -> dict[str, object]:
    overrides: dict[str, object] = {}
    if target_chunk_count is not None:
        if target_chunk_count != 2:
            raise ValueError(f"Unsupported target chunk count: {target_chunk_count}. Only 2 is currently supported.")
        overrides["chunk_bandwidth_hz"] = TWO_CHUNK_CHUNK_BANDWIDTH_HZ
        overrides["chunk_overlap_hz"] = TWO_CHUNK_CHUNK_OVERLAP_HZ
    if debug_chunk_index is not None:
        overrides["debug_chunk_index"] = int(debug_chunk_index)
    return overrides


def expected_binary_path(explicit_binary: str | None) -> Path:
    if explicit_binary:
        return Path(explicit_binary)
    return CONTAINER_BINARY_CANDIDATES[0]


def sigmf_meta_for_data_path(data_path: Path) -> Path:
    if data_path.name.endswith(".sigmf-data"):
        return data_path.with_name(data_path.name[: -len(".sigmf-data")] + ".sigmf-meta")
    return data_path.with_suffix(".sigmf-meta")


def default_output_root_for_input(input_path: Path) -> Path:
    stem = input_path.name
    if stem.endswith(".sigmf-data"):
        stem = stem[: -len(".sigmf-data")]
    else:
        stem = input_path.stem
    return DEFAULT_OUTPUT_ROOT / stem


def map_host_path_to_container(host_path: Path) -> Path:
    resolved = host_path.expanduser().resolve()
    if resolved == REPO_ROOT or REPO_ROOT in resolved.parents:
        relative = resolved.relative_to(REPO_ROOT)
        return CONTAINER_REPO_ROOT / relative
    if resolved == HOST_SCRATCH_ROOT or HOST_SCRATCH_ROOT in resolved.parents:
        relative = resolved.relative_to(HOST_SCRATCH_ROOT)
        return CONTAINER_SCRATCH_ROOT / relative
    raise ValueError(
        f"Path is not mounted into the demo container: {resolved}. "
        f"Use a path under {HOST_SCRATCH_ROOT} or under {REPO_ROOT}."
    )


def has_top_level_key(config_text: str, key: str) -> bool:
    prefix = f"{key}:"
    for line in config_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if line[:1].isspace():
            continue
        if stripped.startswith(prefix):
            return True
    return False


def ensure_offline_eval_config(
    config_path: Path,
    input_file_path: Path,
    output_root: Path,
    progress_every: int | None,
    target_chunk_count: int | None,
    debug_chunk_index: int | None,
) -> Path:
    config_text = config_path.read_text(encoding="utf-8")

    GENERATED_CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    generated_path = GENERATED_CONFIG_ROOT / f"{config_path.stem}_{input_file_path.stem}_offline_eval.yaml"
    config_text = set_block_scalar_values(
        config_text,
        "cuda_dino_detector",
        detector_chunk_overrides(target_chunk_count, debug_chunk_index),
    )
    progress_value = progress_every if progress_every is not None and progress_every > 0 else 0
    offline_eval_values = {
        "run_offline_on_file": True,
        "input_file_path": str(input_file_path),
        "output_root": str(output_root),
        "save_detector_debug_artifacts": True,
        "save_spectrogram_preview": True,
        "save_spectrogram_tensor": True,
        "save_mask_preview": True,
        "save_mask_npy": True,
        "progress_every_n_frames": progress_value,
        "drain_frame_count": 32,
        "channel_number": 0,
    }
    if has_top_level_key(config_text, "offline_eval"):
        config_text = set_block_scalar_values(config_text, "offline_eval", offline_eval_values)
    else:
        offline_eval_block = "\n".join(
            [
                "",
                "offline_eval:",
                *[f"  {key}: {_format_yaml_scalar(value)}" for key, value in offline_eval_values.items()],
                "",
            ]
        )
        config_text = config_text.rstrip() + "\n" + offline_eval_block

    generated_path.write_text(config_text, encoding="utf-8")
    return generated_path


def stage_input_into_scratch(input_file_path: Path, meta_path: Path) -> tuple[Path, Path]:
    staged_dir = HOST_SCRATCH_ROOT / "offline_inputs" / input_file_path.stem
    subprocess.run(["sudo", "mkdir", "-p", str(staged_dir)], check=True)
    staged_input_path = staged_dir / input_file_path.name
    staged_meta_path = staged_dir / meta_path.name
    subprocess.run(["sudo", "cp", "-f", str(input_file_path), str(staged_input_path)], check=True)
    subprocess.run(["sudo", "cp", "-f", str(meta_path), str(staged_meta_path)], check=True)
    return staged_input_path, staged_meta_path


def build_command(
    binary_path: Path,
    config_path: Path,
    input_file_path: Path,
    output_root: Path,
    progress_every: int | None,
) -> list[str]:
    container_binary_path = str(binary_path)
    container_config_path = str(map_host_path_to_container(config_path))
    container_input_path = str(map_host_path_to_container(input_file_path))
    container_output_root = str(map_host_path_to_container(output_root))
    inner_command = [
        container_binary_path,
        "--config",
        container_config_path,
        "--input-file",
        container_input_path,
        "--output-root",
        container_output_root,
    ]
    if progress_every is not None and progress_every > 0:
        inner_command.extend(["--progress-every", str(progress_every)])
    quoted_inner = " ".join(shlex.quote(part) for part in inner_command)
    return ["sudo", "docker", "exec", "-i", CONTAINER_NAME, "bash", "-lc", quoted_inner]

def build_prep_commands(input_file_path: Path, meta_path: Path, output_root: Path) -> list[list[str]]:
    staged_dir = HOST_SCRATCH_ROOT / "offline_inputs" / input_file_path.stem
    staged_input_path = staged_dir / input_file_path.name
    staged_meta_path = staged_dir / meta_path.name
    return [
        ["sudo", "mkdir", "-p", str(staged_dir)],
        ["sudo", "cp", "-f", str(input_file_path), str(staged_input_path)],
        ["sudo", "cp", "-f", str(meta_path), str(staged_meta_path)],
        ["sudo", "mkdir", "-p", str(output_root.parent)],
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the maintained offline CUDA DINO evaluation binary on a local SigMF capture "
            "using the same cuda_dino_detector config block as the live application."
        )
    )
    parser.add_argument(
        "input_file",
        help="Path to the offline capture file, typically *.sigmf-data",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Application config to load. Defaults to config_cuda_dino_performance_single_channel.yaml.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Output directory for spectrogram tensors, masks, and summary files.",
    )
    parser.add_argument(
        "--binary",
        default=None,
        help="Optional container path to the built run_offline_cuda_detector_eval binary.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=None,
        help="Optional progress log cadence in frames.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved command without executing it.",
    )
    parser.add_argument(
        "--target-chunk-count",
        type=int,
        default=None,
        help="Optional detector chunk-count preset. Currently only 2 is supported.",
    )
    parser.add_argument(
        "--debug-chunk-index",
        type=int,
        default=None,
        help="Optional detector debug chunk index override.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_file_path = Path(args.input_file).expanduser().resolve()
    if not input_file_path.is_file():
        raise FileNotFoundError(f"Offline input file does not exist: {input_file_path}")

    sigmf_meta_path = sigmf_meta_for_data_path(input_file_path)
    if not sigmf_meta_path.is_file():
        raise FileNotFoundError(f"Missing SigMF metadata sidecar: {sigmf_meta_path}")

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    output_root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root
        else default_output_root_for_input(input_file_path).resolve()
    )
    effective_config_path = ensure_offline_eval_config(
        config_path=config_path,
        input_file_path=input_file_path,
        output_root=output_root,
        progress_every=args.progress_every,
        target_chunk_count=args.target_chunk_count,
        debug_chunk_index=args.debug_chunk_index,
    )

    staged_dir = HOST_SCRATCH_ROOT / "offline_inputs" / input_file_path.stem
    staged_input_file_path = staged_dir / input_file_path.name
    staged_sigmf_meta_path = staged_dir / sigmf_meta_path.name

    binary_path = expected_binary_path(args.binary)
    command = build_command(
        binary_path=binary_path,
        config_path=effective_config_path,
        input_file_path=staged_input_file_path,
        output_root=output_root,
        progress_every=args.progress_every,
    )
    prep_commands = build_prep_commands(input_file_path, sigmf_meta_path, output_root)

    if args.dry_run:
        print("Preparation commands:")
        for prep_command in prep_commands:
            print(" ".join(shlex.quote(part) for part in prep_command))

    print("Offline CUDA DINO command:")
    print(" ".join(shlex.quote(part) for part in command))
    print(f"Config used: {effective_config_path}")
    print(f"SigMF sidecar: {staged_sigmf_meta_path}")
    print(f"Staged input: {staged_input_file_path}")
    print(f"Output root: {output_root}")

    if args.dry_run:
        return 0

    for prep_command in prep_commands:
        subprocess.run(prep_command, check=True)

    staged_input_file_path, staged_sigmf_meta_path = stage_input_into_scratch(input_file_path, sigmf_meta_path)

    completed = subprocess.run(command, cwd=str(REPO_ROOT), check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FileNotFoundError as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2)