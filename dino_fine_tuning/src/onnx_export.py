"""Export a DinoSegmenter checkpoint to ONNX (fp32, dynamic batch) for TensorRT engine building."""
import argparse, sys, torch, yaml
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--train-yaml", default="configs/train.yaml")
ap.add_argument("--out", required=True)
ap.add_argument("--tile-rows", type=int, default=256)
ap.add_argument("--nfft", type=int, default=1024)
ap.add_argument("--batch", type=int, default=2)
ap.add_argument("--dinov3-repo", default=None, help="dinov3 checkout (default: $DINOV3_ROOT or the repo's sibling)")
args = ap.parse_args()

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import dinov3_root, resolve  # noqa: E402
sys.path.insert(0, args.dinov3_repo or str(dinov3_root()))
from model import DinoSegmenter  # noqa: E402

cfg = yaml.safe_load(open(resolve(args.train_yaml)))
cfg["weights_path"] = str(resolve(cfg["weights_path"]))
ck = torch.load(args.ckpt, map_location="cpu")
mode = ck.get("mode", "ft_lastN")
m = DinoSegmenter(cfg["weights_path"], feat_layers=tuple(cfg["feat_layers"]),
                  mode=mode, unfreeze_last_n=cfg["unfreeze_last_n"])
m.load_state_dict(ck["model"]); m.eval().cuda()
x = torch.randn(args.batch, 1, args.tile_rows, args.nfft, device="cuda")
with torch.no_grad():
    torch.onnx.export(m, x, args.out, input_names=["input"], output_names=["logits"],
                      dynamic_axes={"input": {0: "b"}, "logits": {0: "b"}},
                      opset_version=17, do_constant_folding=True, dynamo=False)
print(f"wrote {args.out}  (input [b,1,{args.tile_rows},{args.nfft}], fp32, dynamic batch)")
