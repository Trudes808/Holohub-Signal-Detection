#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    script_path = Path(__file__).resolve()
    workspace_root = script_path.parents[3]
    default_repo = workspace_root / "dinov3"
    default_weights = Path("/workspace/models/dinov3/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth")
    default_onnx = Path("/workspace/models/dinov3/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.single_channel.onnx")
    default_engine = Path("/workspace/models/dinov3/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.single_channel.fp16.engine")

    parser = argparse.ArgumentParser(description="Export a fused single-channel DINOv3 ONNX model and optional TensorRT engine.")
    parser.add_argument("--model-repo", type=Path, default=default_repo, help="Path to the local dinov3 repository.")
    parser.add_argument("--model-name", default="dinov3_vitb16", help="DINOv3 hub model factory name.")
    parser.add_argument("--weights-path", type=Path, default=default_weights, help="Path to the original DINO weights (.pth).")
    parser.add_argument("--output-onnx", type=Path, default=default_onnx, help="Path for the exported ONNX model.")
    parser.add_argument("--output-engine", type=Path, default=default_engine, help="Path for the optional TensorRT engine.")
    parser.add_argument("--input-height", type=int, default=256, help="Static exported input height.")
    parser.add_argument("--input-width", type=int, default=512, help="Static exported input width.")
    parser.add_argument("--patch-size", type=int, default=16, help="Patch size used by the DINO backbone.")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version.")
    parser.add_argument("--max-batch-size", type=int, default=16, help="Maximum TensorRT batch size for engine build.")
    parser.add_argument("--opt-batch-size", type=int, default=4, help="Optimization batch size for TensorRT engine build.")
    parser.add_argument("--build-engine", action="store_true", help="Invoke trtexec after ONNX export.")
    parser.add_argument("--trtexec", default="trtexec", help="Path to the trtexec binary.")
    parser.add_argument("--no-fp16", action="store_true", help="Disable FP16 engine build.")
    parser.add_argument("--imagenet-mean", type=float, nargs=3, default=(0.485, 0.456, 0.406), help="ImageNet means used by the current Torch runtime path.")
    parser.add_argument("--imagenet-std", type=float, nargs=3, default=(0.229, 0.224, 0.225), help="ImageNet standard deviations used by the current Torch runtime path.")
    return parser.parse_args()


def workspace_root() -> Path:
    return Path(__file__).resolve().parents[2]


def translate_workspace_path(path: Path) -> Path:
    if path.exists():
        return path.expanduser().resolve()

    text = str(path)
    root = workspace_root()
    container_models_root = Path("/workspace/models/dinov3")
    if text.startswith("/workspace/models/dinov3/") and container_models_root.exists():
        return (container_models_root / text.removeprefix("/workspace/models/dinov3/")).resolve()
    if text == "/workspace/models/dinov3" and container_models_root.exists():
        return container_models_root.resolve()
    container_repo_root = Path("/workspace/holohub")
    if text.startswith("/workspace/holohub/") and container_repo_root.exists():
        return (container_repo_root / text.removeprefix("/workspace/holohub/")).resolve()
    if text == "/workspace/holohub" and container_repo_root.exists():
        return container_repo_root.resolve()
    container_repo_dev_root = Path("/workspace/holohub-dev")
    if text.startswith("/workspace/holohub-dev/") and container_repo_dev_root.exists():
        return (container_repo_dev_root / text.removeprefix("/workspace/holohub-dev/")).resolve()
    if text == "/workspace/holohub-dev" and container_repo_dev_root.exists():
        return container_repo_dev_root.resolve()
    if text.startswith("/workspace/models/dinov3/"):
        return (root / "dinov3" / text.removeprefix("/workspace/models/dinov3/")).resolve()
    if text == "/workspace/models/dinov3":
        return (root / "dinov3").resolve()
    if text.startswith("/workspace/holohub/"):
        return (root / text.removeprefix("/workspace/holohub/")).resolve()
    if text == "/workspace/holohub":
        return root.resolve()
    if text.startswith("/workspace/holohub-dev/"):
        return (root / text.removeprefix("/workspace/holohub-dev/")).resolve()
    if text == "/workspace/holohub-dev":
        return root.resolve()
    return path.expanduser().resolve()


def extract_model_state_dict(raw_checkpoint: object) -> dict[str, torch.Tensor]:
    if isinstance(raw_checkpoint, dict):
        for key in ("state_dict", "model", "teacher", "student"):
            value = raw_checkpoint.get(key)
            if isinstance(value, dict) and value:
                return {str(name): tensor for name, tensor in value.items() if isinstance(tensor, torch.Tensor)}
        if raw_checkpoint and all(isinstance(name, str) for name in raw_checkpoint.keys()):
            tensor_items = {str(name): tensor for name, tensor in raw_checkpoint.items() if isinstance(tensor, torch.Tensor)}
            if tensor_items:
                return tensor_items
    raise RuntimeError("Could not locate a model state_dict in the supplied checkpoint")


def load_model_factory(model_repo: Path, model_name: str):
    resolved_repo = model_repo.expanduser().resolve()
    if not resolved_repo.exists():
                raise RuntimeError(f"DINO repository not found: {resolved_repo}")
    sys.path.insert(0, str(resolved_repo))
    import hubconf  # type: ignore

    factory = getattr(hubconf, model_name, None)
    if factory is None:
        raise RuntimeError(f"Model factory '{model_name}' was not found in {resolved_repo / 'hubconf.py'}")
    return factory


def fold_patch_embed_to_single_channel(state_dict: dict[str, torch.Tensor], mean: tuple[float, float, float], std: tuple[float, float, float]) -> dict[str, torch.Tensor]:
    weight_key = "patch_embed.proj.weight"
    bias_key = "patch_embed.proj.bias"
    if weight_key not in state_dict:
        raise RuntimeError("Checkpoint does not contain patch_embed.proj.weight")

    weight = state_dict[weight_key].detach().float().cpu()
    if weight.ndim != 4 or weight.shape[1] != 3:
        raise RuntimeError(f"Expected a 3-channel patch embedding weight tensor, got shape {tuple(weight.shape)}")

    bias = state_dict.get(bias_key)
    if bias is None:
        bias = torch.zeros(weight.shape[0], dtype=weight.dtype)
    else:
        bias = bias.detach().float().cpu()

    mean_tensor = torch.tensor(mean, dtype=weight.dtype).view(1, 3, 1, 1)
    std_tensor = torch.tensor(std, dtype=weight.dtype).view(1, 3, 1, 1)
    folded_weight = (weight / std_tensor).sum(dim=1, keepdim=True)
    folded_bias = bias - (weight * (mean_tensor / std_tensor)).sum(dim=(1, 2, 3))

    fused_state = dict(state_dict)
    fused_state[weight_key] = folded_weight.contiguous()
    fused_state[bias_key] = folded_bias.contiguous()
    return fused_state


def convert_backbone_to_single_channel(backbone: torch.nn.Module) -> None:
    patch_embed = getattr(backbone, "patch_embed", None)
    if patch_embed is None or not hasattr(patch_embed, "proj"):
        raise RuntimeError("Backbone does not expose patch_embed.proj for single-channel conversion")

    proj = patch_embed.proj
    if not isinstance(proj, torch.nn.Conv2d):
        raise RuntimeError("patch_embed.proj is not a Conv2d module")

    if proj.in_channels == 1:
        patch_embed.in_chans = 1
        return

    replacement = torch.nn.Conv2d(
        in_channels=1,
        out_channels=proj.out_channels,
        kernel_size=proj.kernel_size,
        stride=proj.stride,
        padding=proj.padding,
        dilation=proj.dilation,
        groups=proj.groups,
        bias=proj.bias is not None,
        padding_mode=proj.padding_mode,
    )
    replacement = replacement.to(dtype=proj.weight.dtype, device=proj.weight.device)
    patch_embed.proj = replacement
    patch_embed.in_chans = 1


class PatchFeatureExportWrapper(torch.nn.Module):
    def __init__(self, backbone: torch.nn.Module, patch_size: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.patch_size = int(patch_size)

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        features = self.backbone.forward_features(spectrogram)["x_norm_patchtokens"]
        batch, patch_count, feature_dim = features.shape
        patch_rows = spectrogram.shape[2] // self.patch_size
        patch_cols = spectrogram.shape[3] // self.patch_size
        if patch_rows * patch_cols != patch_count:
            raise RuntimeError(
                f"Patch feature count mismatch: got {patch_count}, expected {patch_rows}x{patch_cols}"
            )
        return features.view(batch, patch_rows, patch_cols, feature_dim)


def export_onnx(args: argparse.Namespace) -> None:
    model_repo = translate_workspace_path(args.model_repo)
    weights_path = translate_workspace_path(args.weights_path)
    output_onnx = translate_workspace_path(args.output_onnx)

    factory = load_model_factory(model_repo, args.model_name)
    checkpoint = torch.load(weights_path, map_location="cpu")
    raw_state_dict = extract_model_state_dict(checkpoint)
    fused_state_dict = fold_patch_embed_to_single_channel(raw_state_dict, tuple(args.imagenet_mean), tuple(args.imagenet_std))

    backbone = factory(pretrained=False)
    convert_backbone_to_single_channel(backbone)
    missing, unexpected = backbone.load_state_dict(fused_state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Unexpected fused-state load result. Missing={missing} Unexpected={unexpected}")
    backbone.eval()

    export_model = PatchFeatureExportWrapper(backbone, args.patch_size).eval()
    dummy_input = torch.zeros((1, 1, args.input_height, args.input_width), dtype=torch.float32)

    output_onnx.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            export_model,
            dummy_input,
            output_onnx,
            input_names=["spectrogram"],
            output_names=["patch_features"],
            dynamic_axes={"spectrogram": {0: "batch"}, "patch_features": {0: "batch"}},
            opset_version=args.opset,
            dynamo=False,
        )
    print(f"Exported ONNX model to {output_onnx}")


def build_engine(args: argparse.Namespace) -> None:
    trtexec_candidates = [
        args.trtexec,
        "/usr/src/tensorrt/bin/trtexec",
        "/usr/local/tensorrt/bin/trtexec",
        "/usr/local/bin/trtexec",
        "/usr/bin/trtexec",
    ]
    trtexec_path = None
    for candidate in trtexec_candidates:
        resolved = shutil.which(candidate) if "/" not in candidate else (candidate if Path(candidate).is_file() else None)
        if resolved:
            trtexec_path = resolved
            break
    if trtexec_path is None:
        raise RuntimeError(
            "Could not find trtexec binary. Checked: " + ", ".join(trtexec_candidates)
        )

    output_onnx = translate_workspace_path(args.output_onnx)
    output_engine = translate_workspace_path(args.output_engine)
    output_engine.parent.mkdir(parents=True, exist_ok=True)

    opt_batch = max(1, min(args.max_batch_size, args.opt_batch_size))
    min_shape = f"spectrogram:1x1x{args.input_height}x{args.input_width}"
    opt_shape = f"spectrogram:{opt_batch}x1x{args.input_height}x{args.input_width}"
    max_shape = f"spectrogram:{max(1, args.max_batch_size)}x1x{args.input_height}x{args.input_width}"

    command = [
        trtexec_path,
        f"--onnx={output_onnx}",
        f"--saveEngine={output_engine}",
        f"--minShapes={min_shape}",
        f"--optShapes={opt_shape}",
        f"--maxShapes={max_shape}",
        "--skipInference",
    ]
    if not args.no_fp16:
        command.append("--fp16")

    print("Running:", " ".join(str(part) for part in command))
    subprocess.run(command, check=True)
    print(f"Built TensorRT engine at {output_engine}")


def main() -> int:
    args = parse_args()
    export_onnx(args)
    if args.build_engine:
        build_engine(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())