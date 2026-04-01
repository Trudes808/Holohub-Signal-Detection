#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn


class DinoV3ExportAdapter(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.model(x)
        tensor = self._extract_tensor(output)
        if tensor.dim() == 0:
            return tensor.unsqueeze(0)
        return tensor

    def _extract_tensor(self, output):
        if torch.is_tensor(output):
            return output

        if isinstance(output, dict):
            for key in ("x_norm_patchtokens", "x_norm_clstoken", "x_prenorm", "x"):
                value = output.get(key)
                if torch.is_tensor(value):
                    return value
            for value in output.values():
                if torch.is_tensor(value):
                    return value

        if isinstance(output, (list, tuple)):
            for value in output:
                if torch.is_tensor(value):
                    return value

        raise TypeError(f"Unsupported model output type for export: {type(output)!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a local DINOv3 model to TorchScript.")
    parser.add_argument("--repo", required=True, help="Path to the local DINOv3 repository.")
    parser.add_argument("--weights", required=True, help="Path to the DINOv3 weights file.")
    parser.add_argument("--output", required=True, help="Path for the exported TorchScript file.")
    parser.add_argument("--model-name", default="dinov3_vitb16", help="Torch Hub model name.")
    parser.add_argument("--height", type=int, default=256, help="Example input height.")
    parser.add_argument("--width", type=int, default=512, help="Example input width.")
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Export device. GPU-only bring-up should use 'cuda'.",
    )
    return parser.parse_args()


def resolve_device(requested_device: str) -> torch.device:
    device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA export requested, but torch.cuda.is_available() is false. "
            "Verify the container GPU runtime and PyTorch CUDA wheel."
        )
    return device


def main() -> int:
    args = parse_args()

    repo = Path(args.repo)
    weights = Path(args.weights)
    output = Path(args.output)

    if not repo.exists():
        raise FileNotFoundError(f"DINOv3 repo not found: {repo}")
    if not weights.exists():
        raise FileNotFoundError(f"DINOv3 weights not found: {weights}")

    device = resolve_device(args.device)

    model = torch.hub.load(
        repo_or_dir=str(repo),
        model=args.model_name,
        source="local",
        weights=str(weights),
    )
    model = model.to(device).eval()

    adapter = DinoV3ExportAdapter(model).to(device).eval()
    example = torch.randn(1, 3, args.height, args.width, device=device, dtype=torch.float32)

    with torch.no_grad():
        scripted = torch.jit.trace(adapter, example, strict=False)

    output.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(output))
    print(f"Saved TorchScript model to {output} using device={device.type}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())