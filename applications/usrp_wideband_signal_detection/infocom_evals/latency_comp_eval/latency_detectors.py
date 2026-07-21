#!/usr/bin/env python3
"""Device-switchable (CPU/GPU) reference implementations of all six detectors, for
the per-frame latency + compute-load eval.

The deployed ``coherent_power`` / ``cuda_dino`` detectors are C++/CUDA operators with
no CPU path, so we cannot time them on the CPU as shipped. To compare all six on the
SAME footing on BOTH devices, every detector here is a **torch reference
implementation** that runs identically on ``cpu`` and ``cuda`` (torch has FFT, conv,
pooling and the ViT models on both). These are faithful to the deployed algorithms'
*operations* (FFT front-end + power/CFAR/morphology, or the DINOv3 ViT forward), not a
byte-for-byte port of the CUDA kernels -- they exist to make the CPU-vs-GPU latency and
the per-frame FLOP/peak-memory numbers meaningful and comparable, per the eval design.

Detector -> what is timed (one frame):
  * ``3dB_power``      full-frame spectrogram (512 x nfft) FFT + single-scalar threshold
  * ``blob_detection`` FFT + gaussian/sobel conv + percentile edges + morphology
  * ``coherent_power`` FFT + per-freq equalization + box-mean CFAR support + per-freq
                       floor(+2dB)/strong-rescue(+8dB) power views OR-combined + majority
                       filter + morphological open/close  (the deployed hybrid pipeline)
  * ``cuda_dino``      full-frame spectrogram tiled into 256x512 chunks -> frozen DINOv3
                       ViT-B/16 forward (zero-shot) -> patch-score threshold
  * ``yolo``           fine-tuned YOLO26-m over native nfft=1024 / 256-row tiles (reused
                       yolo_infer.YoloDetector; tile count scales with sample rate)
  * ``dino_finetuned`` fine-tuned DINOv3 segmenter, native nfft=1024 / 256-row tiles
                       (reused finetuned_infer.FinetunedDetector)

The power detectors consume the full-frame spectrogram at the rate's auto FFT size, so
their cost scales with nfft. The ML detectors tile the frame at their native geometry,
so their cost scales with the number of tiles -> with sample rate. Both scalings are
exactly what the deployed system sees at 20/100/250/500 MHz.

Detector-only timing
--------------------
Each detector's ``prepare()`` computes the **shared FFT front-end** (IQ -> dB spectrogram)
ONCE, outside the timed region, and returns a ``run()`` that starts from that spectrogram.
So the measured latency is the *detector operator's* own compute (spectrogram -> mask),
NOT the shared FFT that the deployed pipeline runs once upstream for every detector. The
compute-load FLOPs, by contrast, still add the analytic FFT term (total per-frame work).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F

from fft_sizing import FrameGeometry, fft_flops

# --------------------------------------------------------------------------- #
# sys.path wiring for the reused ML detector classes + the dinov3 backbone.
# Mirrors run_ml_detectors_offline.py so we score/time the SAME model code.
# --------------------------------------------------------------------------- #
_THIS_DIR = Path(__file__).resolve().parent
_INFOCOM = _THIS_DIR.parent
_REPO_ROOT = _THIS_DIR.parents[3]                                  # holohub-dev/
_DINO_SRC = _REPO_ROOT / "dino_fine_tuning" / "src"
_YOLO_SRC = _REPO_ROOT / "yolo_training" / "src"


def wire_syspath(dinov3_repo: Optional[str]) -> None:
    for p in (str(_DINO_SRC), str(_YOLO_SRC)):
        if p not in sys.path:
            sys.path.insert(0, p)
    if dinov3_repo and dinov3_repo not in sys.path:
        sys.path.insert(0, dinov3_repo)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------- #
# Shared front-end + torch morphology helpers
# --------------------------------------------------------------------------- #
def frames_to_db(iq: torch.Tensor, nfft: int, rows: int) -> torch.Tensor:
    """Flat complex IQ (len >= rows*nfft) -> (rows, nfft) dB power spectrogram.

    Matches rfdata.frames_to_db / eval_viz.spectrogram_db_from_iq: per-row
    fftshift(fft), 10*log10(|X|^2). Runs on whatever device ``iq`` lives on.
    """
    usable = rows * nfft
    block = iq[:usable].reshape(rows, nfft)
    spec = torch.fft.fftshift(torch.fft.fft(block, dim=-1), dim=-1)
    power = spec.real ** 2 + spec.imag ** 2 + 1e-12
    return 10.0 * torch.log10(power)


def _dilate(x: torch.Tensor, k: int) -> torch.Tensor:
    return F.max_pool2d(x, k, stride=1, padding=k // 2)


def _erode(x: torch.Tensor, k: int) -> torch.Tensor:
    return -F.max_pool2d(-x, k, stride=1, padding=k // 2)


def _box_mean(x: torch.Tensor, kh: int, kw: int) -> torch.Tensor:
    return F.avg_pool2d(x, (kh, kw), stride=1, padding=(kh // 2, kw // 2),
                        count_include_pad=False)


def _gaussian_kernel(sigma: float, radius: int, device, dtype) -> torch.Tensor:
    ax = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    g1 = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
    g1 = g1 / g1.sum()
    return torch.outer(g1, g1)[None, None]


def _safe_quantile(x: torch.Tensor, q: float) -> torch.Tensor:
    """torch.quantile with a subsample fallback above its 2**24-element cap."""
    xf = x.reshape(-1).float()
    cap = 2 ** 24 - 1
    if xf.numel() > cap:
        idx = torch.linspace(0, xf.numel() - 1, cap, device=xf.device).long()
        xf = xf[idx]
    return torch.quantile(xf, q)


# --------------------------------------------------------------------------- #
# Detector base
# --------------------------------------------------------------------------- #
class Detector:
    """Base: load()/unload() manage any model; prepare() returns a zero-arg run fn."""
    name: str = "base"
    kind: str = "power"          # "power" (full-frame) or "ml" (native tiling)

    def load(self, device: str) -> None:
        self._device = device

    def unload(self) -> None:
        if getattr(self, "_device", "cpu") == "cuda":
            torch.cuda.empty_cache()

    def prepare(self, iq_np: np.ndarray, geom: FrameGeometry, device: str) -> Callable[[], object]:
        raise NotImplementedError

    def fft_component(self, geom: FrameGeometry) -> tuple[int, int]:
        """(nfft, n_rows) of the front-end FFT, for the analytic FFT-flop term."""
        return (geom.actual_fft_size, geom.num_ffts_per_batch)

    def analytic_fft_flops(self, geom: FrameGeometry) -> float:
        nfft, rows = self.fft_component(geom)
        return fft_flops(nfft, rows)


# --------------------------------------------------------------------------- #
# 1. 3dB power (single scalar per-frame floor)
# --------------------------------------------------------------------------- #
class ThreeDBPower(Detector):
    name = "3dB_power"
    kind = "power"

    def __init__(self, threshold_db: float = 3.0, noise_percentile: float = 50.0):
        self.threshold_db = float(threshold_db)
        self.q = float(noise_percentile) / 100.0

    def prepare(self, iq_np, geom, device):
        iq = torch.from_numpy(np.ascontiguousarray(iq_np.astype(np.complex64))).to(device)
        db = frames_to_db(iq, geom.actual_fft_size, geom.num_ffts_per_batch)  # front-end (untimed)

        def run():                                                # detector-only
            floor = _safe_quantile(db, self.q)
            return (db > floor + self.threshold_db).to(torch.uint8)
        return run


# --------------------------------------------------------------------------- #
# 2. blob detection (edge -> morphology -> area)
# --------------------------------------------------------------------------- #
class BlobDetection(Detector):
    name = "blob_detection"
    kind = "power"

    def __init__(self, smooth_sigma: float = 1.0, edge_percentile: float = 90.0,
                 close_iters: int = 2, min_blob_area: int = 64):
        self.sigma = float(smooth_sigma)
        self.edge_q = float(edge_percentile) / 100.0
        self.close_iters = int(close_iters)
        self.open_k = max(3, int(round(math.sqrt(min_blob_area))) | 1)   # odd kernel ~ area filter

    def prepare(self, iq_np, geom, device):
        iq = torch.from_numpy(np.ascontiguousarray(iq_np.astype(np.complex64))).to(device)
        dtype = torch.float32
        db4 = frames_to_db(iq, geom.actual_fft_size,                    # front-end (untimed)
                           geom.num_ffts_per_batch)[None, None].float()
        gauss = _gaussian_kernel(self.sigma, 2, device, dtype)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=device, dtype=dtype)[None, None]
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=device, dtype=dtype)[None, None]

        def run():                                                      # detector-only
            db = db4
            sm = F.conv2d(db, gauss, padding=2)
            gx = F.conv2d(sm, sobel_x, padding=1)
            gy = F.conv2d(sm, sobel_y, padding=1)
            grad = torch.sqrt(gx * gx + gy * gy)
            thr = _safe_quantile(grad, self.edge_q)
            edges = (grad > thr).float()
            m = edges
            for _ in range(self.close_iters):
                m = _dilate(m, 3)
            for _ in range(self.close_iters):
                m = _erode(m, 3)
            m = _erode(m, self.open_k)          # opening ~ discard small blobs (area filter)
            m = _dilate(m, self.open_k)
            return (m[0, 0] > 0.5).to(torch.uint8)
        return run


# --------------------------------------------------------------------------- #
# 3. coherent_power (deployed hybrid image + power pipeline, reference port)
# --------------------------------------------------------------------------- #
class CoherentPower(Detector):
    name = "coherent_power"
    kind = "power"

    def __init__(self, box_bins: int = 65, support_threshold: float = 0.5,
                 power_offset_db: float = 2.0, rescue_offset_db: float = 8.0):
        self.box_bins = int(box_bins) | 1
        self.support_threshold = float(support_threshold)
        self.power_offset_db = float(power_offset_db)
        self.rescue_offset_db = float(rescue_offset_db)

    def prepare(self, iq_np, geom, device):
        iq = torch.from_numpy(np.ascontiguousarray(iq_np.astype(np.complex64))).to(device)
        nfft, rows = geom.actual_fft_size, geom.num_ffts_per_batch
        box_w = min(self.box_bins, nfft if nfft % 2 == 1 else nfft - 1)
        db = frames_to_db(iq, nfft, rows)                          # front-end (untimed): power dB

        def run():                                                 # detector-only
            x = db[None, None]
            # per-freq (per-bin) mean-over-time equalization
            col_mean = db.mean(dim=0, keepdim=True)
            corrected = db - col_mean
            # local box-mean background across frequency -> CFAR-style support view
            bg = _box_mean(x, 1, box_w)[0, 0]
            floor = _safe_quantile(corrected, 0.5)
            span = corrected.std() + 1e-6
            support = (db - bg - floor) / span
            support_mask = support >= self.support_threshold
            # per-freq calibrated floor -> power views (+2 dB fill, +8 dB strong rescue)
            perfreq_floor = torch.quantile(db, 0.5, dim=0)         # (nfft,)
            power_mask = db > (perfreq_floor[None, :] + self.power_offset_db)
            rescue_mask = db > (perfreq_floor[None, :] + self.rescue_offset_db)
            combined = (support_mask | power_mask | rescue_mask).float()[None, None]
            # majority filter + morphological close then open
            maj = (_box_mean(combined, 3, 3) >= 0.5).float()
            m = _erode(_dilate(maj, 3), 3)                         # close
            m = _dilate(_erode(m, 3), 3)                           # open
            return (m[0, 0] > 0.5).to(torch.uint8)
        return run


# --------------------------------------------------------------------------- #
# 4. cuda_dino  (zero-shot frozen DINOv3 ViT-B/16 over 256x512 chunks)
# --------------------------------------------------------------------------- #
class ZeroShotDino(Detector):
    name = "cuda_dino"
    kind = "ml"

    def __init__(self, weights_path: str, chunk_rows: int = 256, chunk_cols: int = 512,
                 vmin: float = -20.0, vmax: float = 15.0, feat_layer: int = 11,
                 max_batch: int = 16, score_percentile: float = 90.0):
        self.weights_path = weights_path
        self.chunk_rows = int(chunk_rows)
        self.chunk_cols = int(chunk_cols)
        self.vmin, self.vmax = float(vmin), float(vmax)
        self.feat_layer = int(feat_layer)
        self.max_batch = int(max_batch)
        self.q = float(score_percentile) / 100.0
        self.model = None

    def load(self, device):
        self._device = device
        import dinov3.hub.backbones as B                          # noqa: E402
        self.model = B.dinov3_vitb16(pretrained=True, weights=self.weights_path).to(device).eval()
        self._mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        self._std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    def unload(self):
        self.model = None
        super().unload()

    def fft_component(self, geom):
        return (geom.actual_fft_size, geom.num_ffts_per_batch)

    @torch.no_grad()
    def prepare(self, iq_np, geom, device):
        iq = torch.from_numpy(np.ascontiguousarray(iq_np.astype(np.complex64))).to(device)
        nfft, rows = geom.actual_fft_size, geom.num_ffts_per_batch
        cr, cc = self.chunk_rows, self.chunk_cols
        amp = (device == "cuda")
        db = frames_to_db(iq, nfft, rows)                          # front-end (untimed)

        def run():                                                 # detector-only
            img = torch.clamp((db - self.vmin) / max(self.vmax - self.vmin, 1e-6), 0, 1)
            # tile the full-frame spectrogram into chunk_rows x chunk_cols patches
            tiles = []
            for r0 in range(0, rows, cr):
                for c0 in range(0, nfft, cc):
                    t = img[r0:r0 + cr, c0:c0 + cc]
                    if t.shape[0] < cr or t.shape[1] < cc:
                        t = F.pad(t, (0, cc - t.shape[1], 0, cr - t.shape[0]))
                    tiles.append(t)
            x = torch.stack(tiles)[:, None].repeat(1, 3, 1, 1)         # B,3,cr,cc
            xn = (x - self._mean) / self._std
            masks = []
            for i in range(0, xn.shape[0], self.max_batch):
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                    feats = self.model.get_intermediate_layers(
                        xn[i:i + self.max_batch], n=[self.feat_layer], reshape=True, norm=True)[0]
                score = feats.float().norm(dim=1)                      # B,h,w patch-score
                thr = torch.quantile(score.flatten(1), self.q, dim=1).view(-1, 1, 1)
                masks.append((score >= thr).to(torch.uint8))
            return torch.cat(masks)
        return run


# --------------------------------------------------------------------------- #
# 5 + 6. Reused fine-tuned ML detectors (YOLO26-m, DINOv3 segmenter)
# --------------------------------------------------------------------------- #
class MLReused(Detector):
    """Wraps yolo_infer.YoloDetector / finetuned_infer.FinetunedDetector so the exact
    deployed-sweep model code is what we time. They take numpy IQ and tile at nfft=1024."""
    kind = "ml"

    def __init__(self, name: str, spec: dict, dinov3_repo: Optional[str]):
        self.name = name
        self.spec = spec
        self.dinov3_repo = dinov3_repo
        self.det = None
        self.optimize = bool(spec.get("optimize", False))          # torch.compile + channels_last
        self.compile_mode = spec.get("compile_mode", "max-autotune")

    def load(self, device):
        self._device = device
        spec = self.spec
        ds_meta = json.loads(Path(spec["dataset_meta"]).read_text())
        if spec["kind"] == "yolo":
            import rfdata  # noqa: F401
            from yolo_infer import YoloDetector
            self.det = YoloDetector(spec["ckpt"], ds_meta, device=device,
                                    conf=float(spec.get("conf", 0.25)),
                                    imgsz=int(spec.get("imgsz", 1024)), name=self.name)
        elif spec["kind"] == "dino_finetuned":
            import yaml
            import rfdata  # noqa: F401
            import model   # noqa: F401
            import finetuned_infer as fi
            train_cfg = yaml.safe_load(Path(spec["train_cfg"]).read_text())
            thr = fi.load_threshold(spec["eval_meta"]) if spec.get("eval_meta") else spec.get("threshold")
            self.det = fi.FinetunedDetector(spec["ckpt"], train_cfg, ds_meta,
                                            device=device, threshold=thr)
            if self.optimize and device == "cuda":
                # output-preserving latency opts: channels_last + torch.compile (backbone is
                # ~86% of the time; compile fuses the ViT -> ~1.5x throughput / ~1.9x @ 500 MHz,
                # masks unchanged/IoU~0.996). Raise the dynamo cache limit so per-rate shapes
                # don't overflow it and silently fall back to (slow) eager.
                import torch._dynamo as _dyn
                _dyn.config.cache_size_limit = 64
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True
                self.det.model = self.det.model.to(memory_format=torch.channels_last)
                self.det.model = torch.compile(self.det.model, mode=self.compile_mode,
                                               fullgraph=False)
        else:
            raise ValueError(f"unknown ML kind {spec['kind']!r}")

    def unload(self):
        self.det = None
        super().unload()

    def fft_component(self, geom):
        nfft = int(self.det.nfft) if self.det is not None else 1024
        rows = geom.samples_per_frame // nfft
        return (nfft, rows)

    def prepare(self, iq_np, geom, device):
        """Precompute the nfft=1024 dB spectrogram (shared FFT front-end, untimed) and time
        only the post-FFT tiling + model path, replicating yolo_infer/finetuned_infer's
        mask_for_iq minus the FFT so the timed region is detector-only. The MODEL objects
        (self.det.model) are the exact reused classes' models -- fidelity preserved."""
        import rfdata as rf
        det = self.det
        nfft = int(det.nfft)
        iq = np.ascontiguousarray(iq_np.astype(np.complex64))
        n = (len(iq) // nfft) * nfft
        rows = n // nfft
        iqt = torch.from_numpy(iq[:n]).to(device)

        if self.spec["kind"] == "yolo":
            db = rf.frames_to_db(iqt[None], nfft, rows)[0].cpu().numpy()   # front-end (untimed)

            def run():                                                     # detector-only
                img = rf.db_to_uint8(db, det.vmin, det.vmax)
                spans = [(r0, min(rows, r0 + det.tile)) for r0 in range(0, rows, det.tile)]
                tiles = []
                for r0, r1 in spans:
                    t = img[r0:r1]
                    if t.shape[0] < det.tile:
                        t = np.pad(t, ((0, det.tile - t.shape[0]), (0, 0)))
                    tiles.append(np.repeat(t[:, :, None], 3, axis=2))
                res = det.model.predict(tiles, imgsz=det.imgsz, conf=det.conf,
                                        device=det.device, verbose=False)
                mask = np.zeros((rows, nfft), np.uint8)
                for k, (r0, r1) in enumerate(spans):
                    tm = np.zeros((det.tile, nfft), np.uint8)
                    for x0, y0, x1, y1 in res[k].boxes.xyxy.cpu().numpy():
                        xi0, xi1 = max(0, int(np.floor(x0))), min(nfft, int(np.ceil(x1)))
                        yi0, yi1 = max(0, int(np.floor(y0))), min(det.tile, int(np.ceil(y1)))
                        if xi1 > xi0 and yi1 > yi0:
                            tm[yi0:yi1, xi0:xi1] = 1
                    mask[r0:r1] = tm[:r1 - r0]
                return mask
            return run

        # dino_finetuned
        db = rf.frames_to_db(iqt[None], nfft, rows)[0]                     # front-end (untimed), device

        channels_last = self.optimize and device == "cuda"

        @torch.no_grad()
        def run():                                                         # detector-only
            img = torch.clamp((db - det.vmin) / max(det.vmax - det.vmin, 1e-6), 0, 1)
            spans = [(r0, min(rows, r0 + det.tile)) for r0 in range(0, rows, det.tile)]
            batch = []
            for r0, r1 in spans:
                t = img[r0:r1]
                if t.shape[0] < det.tile:
                    t = F.pad(t, (0, 0, 0, det.tile - t.shape[0]))
                batch.append(t)
            x = torch.stack(batch)[:, None]
            if channels_last:
                x = x.contiguous(memory_format=torch.channels_last)
            # optimized path runs all of a frame's tiles as ONE batch (one compiled shape per
            # rate); baseline keeps chunks of 16 to bound memory.
            step = x.shape[0] if self.optimize else 16
            out = []
            for i in range(0, x.shape[0], step):
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=det.amp):
                    logits = det.model(x[i:i + step])
                out.append((torch.sigmoid(logits.float()) >= det.threshold)[:, 0].to(torch.uint8))
            pm = torch.cat(out)
            mask = torch.zeros((rows, nfft), dtype=torch.uint8, device=device)
            for k, (r0, r1) in enumerate(spans):
                mask[r0:r1] = pm[k, :r1 - r0]
            return mask

        if self.optimize and device == "cuda":     # trigger torch.compile OUTSIDE timing
            for _ in range(3):
                run(); torch.cuda.synchronize()
        return run


# --------------------------------------------------------------------------- #
# Registry / factory
# --------------------------------------------------------------------------- #
CANONICAL_ORDER = ["coherent_power", "cuda_dino", "3dB_power", "blob_detection",
                   "yolo", "dino_finetuned", "dino_finetuned_opt"]


def build_detectors(cfg: dict) -> dict[str, Detector]:
    """Instantiate the requested detectors from the eval config's ``detectors`` block."""
    dcfg = cfg.get("detectors", {})
    dinov3_repo = cfg.get("dinov3_repo")
    out: dict[str, Detector] = {}
    if "3dB_power" in dcfg:
        out["3dB_power"] = ThreeDBPower(**dcfg["3dB_power"].get("params", {}))
    if "blob_detection" in dcfg:
        out["blob_detection"] = BlobDetection(**dcfg["blob_detection"].get("params", {}))
    if "coherent_power" in dcfg:
        out["coherent_power"] = CoherentPower(**dcfg["coherent_power"].get("params", {}))
    if "cuda_dino" in dcfg:
        spec = dcfg["cuda_dino"]
        out["cuda_dino"] = ZeroShotDino(weights_path=spec["weights_path"],
                                        **spec.get("params", {}))
    for name, spec in dcfg.items():
        if isinstance(spec, dict) and spec.get("kind") in ("yolo", "dino_finetuned"):
            out[name] = MLReused(name, spec, dinov3_repo)
    # canonical order (unknown names appended)
    ordered = [k for k in CANONICAL_ORDER if k in out]
    ordered += [k for k in out if k not in CANONICAL_ORDER]
    return {k: out[k] for k in ordered}
