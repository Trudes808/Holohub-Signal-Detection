"""Torch replica of the deployed finetuned_dino_detector downsample front-end.

Produces the EXACT model input the operator feeds the segmenter at deployment, so the fine-tune trains
on deployment-identical spectrograms. Ported 1:1 from operators/finetuned_dino_detector/
finetuned_dino_detector.cu (ft_power_to_db_shift / ft_col_mean / ft_smooth_cols /
ft_frontend_reference / ft_frontend_correction / robust histogram-percentile normalization) and
finetuned_dino_torch_helpers.cpp (bilinear freq resize wide->nfft, then tile into tile_rows).

Pipeline per frame (rows_wide x fft_size complex IQ):
  hann(rms-norm) window -> FFT -> fftshift -> power->dB - gain_offset
  -> per-freq floor flatten (2-pass, signal-capped) -> robust p-low floor normalization to [0,1]
  -> bilinear resize freq (fft_size -> nfft) -> split rows into tile_rows tiles.

auto_fft_size(rate) matches the operator: at 491.52 MSps -> 20480 (gain_offset 13.01 dB);
at 245.76 -> 10240 (10.0 dB).
"""
from __future__ import annotations
from dataclasses import dataclass, field
import math
import numpy as np
import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
@dataclass
class FrontEndCfg:
    nfft: int = 1024
    tile_rows: int = 256
    fft_window: str = "hann"
    db_vmin: float = -47.6526985168457
    db_vmax: float = 20.45633316040039
    # flatten (per-freq floor)
    flatten: bool = True
    flatten_reference_q: float = 75.0
    flatten_smooth_frac: float = 0.005
    flatten_max_boost_db: float = 12.0
    flatten_signal_cap_db: float = 6.0
    # power-level match (both default off/zero at 491.52 -> level_offset 0)
    match_training_power_level: bool = False
    reference_sample_rate_hz: float = 245.76e6
    power_level_trim_db: float = 0.0
    # adaptive robust normalization
    adaptive: bool = True
    adaptive_robust_floor: bool = True
    adaptive_span_db: float = 34.0
    adaptive_floor_frac: float = 0.12
    adaptive_low_pct: float = 20.0
    adaptive_high_pct: float = 95.0
    adaptive_min_range_db: float = 8.0
    adaptive_floor_below_calib_db: float = 25.0


def auto_fft_size(rate_hz: float, ref_span_hz=500.0e6, ref_fft=20480, packet=1024) -> int:
    span_ratio = rate_hz / ref_span_hz
    snapped = 2.0 ** round(math.log2(span_ratio)) if span_ratio > 0 else 1.0
    packets = max(1, int(round(ref_fft * snapped / packet)))
    return max(packet, packets * packet)


def _hann_rms(fft_size: int, window: str, device, dtype=torch.float32) -> torch.Tensor | None:
    if window == "none":
        return None
    i = torch.arange(fft_size, device=device, dtype=torch.float64)
    x = 2.0 * math.pi * i / fft_size
    if window == "hamming":
        w = 0.54 - 0.46 * torch.cos(x)
    elif window == "blackman":
        w = 0.42 - 0.5 * torch.cos(x) + 0.08 * torch.cos(2.0 * x)
    else:  # hann
        w = 0.5 - 0.5 * torch.cos(x)
    rms = torch.sqrt((w * w).mean()).clamp_min(1e-12)
    return (w / rms).to(dtype)


def _gaussian_smooth_cols(col: torch.Tensor, sigma: float, radius: int) -> torch.Tensor:
    """col: [B, nfft] -> gaussian-smoothed along freq with clamp(replicate)-edge, matching the kernel."""
    off = torch.arange(-radius, radius + 1, device=col.device, dtype=torch.float32)
    w = torch.exp(-(off * off) / (2.0 * sigma * sigma))
    w = (w / w.sum()).view(1, 1, -1)
    x = col.unsqueeze(1)  # [B,1,nfft]
    x = F.pad(x, (radius, radius), mode="replicate")
    return F.conv1d(x, w).squeeze(1)


@dataclass
class FrontEndOut:
    tiles: torch.Tensor            # [B*n_tiles, 1, tile_rows, nfft] in [0,1]
    wide_norm: torch.Tensor        # [B, rows_wide, fft_size] in [0,1] (pre-resize; for operator A/B)
    resized: torch.Tensor          # [B, rows_wide, nfft] in [0,1]
    fft_size: int = 0
    gain_offset_db: float = 0.0
    used_robust: list = field(default_factory=list)   # per-frame bool
    floor_db: list = field(default_factory=list)
    n_tiles: int = 0


def compute_front_end(iq: torch.Tensor, rate_hz: float, cfg: FrontEndCfg,
                      fft_size: int | None = None) -> FrontEndOut:
    """iq: [B, rows_wide*fft_size] complex (or [B, rows_wide, fft_size]). Returns model-input tiles."""
    dev = iq.device
    nfft = cfg.nfft
    if fft_size is None:
        fft_size = max(nfft, auto_fft_size(rate_hz))
    if iq.dim() == 2:
        rows_wide = iq.shape[1] // fft_size
        block = iq[:, : rows_wide * fft_size].reshape(iq.shape[0], rows_wide, fft_size)
    else:
        rows_wide, block = iq.shape[1], iq
    B = block.shape[0]

    win = _hann_rms(fft_size, cfg.fft_window, dev)
    if win is not None:
        block = block * win  # broadcast over rows
    spec = torch.fft.fftshift(torch.fft.fft(block, dim=-1), dim=-1)  # [B, rows, fft_size]
    power = spec.real ** 2 + spec.imag ** 2 + 1e-12
    gain_offset_db = 10.0 * math.log10(fft_size / nfft)
    db = 10.0 * torch.log10(power) - gain_offset_db

    level_offset_db = cfg.power_level_trim_db
    if cfg.match_training_power_level:
        level_offset_db += 10.0 * math.log10(rate_hz / max(1.0, cfg.reference_sample_rate_hz))

    if cfg.flatten:
        sigma = max(2.0, cfg.flatten_smooth_frac * fft_size)
        radius = max(1, int(math.ceil(sigma * 1.5)))
        # pass 1: uncapped per-freq mean over rows -> smooth -> reference
        col = db.mean(dim=1)                                   # [B, fft_size]
        col_s = _gaussian_smooth_cols(col, sigma, radius)
        ref = _reference(col_s, cfg.flatten_reference_q)       # [B]
        if cfg.flatten_signal_cap_db > 0.0:
            cap = (ref + cfg.flatten_signal_cap_db).view(B, 1, 1)
            col = torch.minimum(db, cap).mean(dim=1)
            col_s = _gaussian_smooth_cols(col, sigma, radius)
            ref = _reference(col_s, cfg.flatten_reference_q)
        boost = torch.clamp(ref.view(B, 1) - col_s, min=0.0, max=cfg.flatten_max_boost_db)  # [B,fft_size]
        db = db + boost.unsqueeze(1)

    fixed_vmin = cfg.db_vmin + level_offset_db
    fixed_inv_span = 1.0 / max(cfg.db_vmax - cfg.db_vmin, 1e-6)
    span = max(1.0, cfg.adaptive_span_db)

    used_robust, floor_list = [], []
    clip_vmin = torch.full((B,), float(fixed_vmin), device=dev)
    clip_inv_span = torch.full((B,), float(fixed_inv_span), device=dev)
    if cfg.adaptive and cfg.adaptive_robust_floor:
        flat = db.reshape(B, -1)
        qs = torch.tensor([cfg.adaptive_low_pct / 100.0, cfg.adaptive_high_pct / 100.0], device=dev)
        pcts = torch.quantile(flat, qs, dim=1)                 # [2, B]
        floor_db, high_db = pcts[0], pcts[1]
        drange = high_db - floor_db
        too_flat = drange < cfg.adaptive_min_range_db
        implausible = floor_db < (fixed_vmin - cfg.adaptive_floor_below_calib_db)
        use_robust = ~(too_flat | implausible)
        rvmin = floor_db - cfg.adaptive_floor_frac * span
        clip_vmin = torch.where(use_robust, rvmin, clip_vmin)
        clip_inv_span = torch.where(use_robust, torch.full_like(clip_inv_span, 1.0 / span), clip_inv_span)
        used_robust = use_robust.tolist()
        floor_list = floor_db.tolist()
    elif cfg.adaptive:  # legacy q-blend anchor (kept for completeness)
        ref = _reference(_gaussian_smooth_cols(db.mean(dim=1),
                         max(2.0, cfg.flatten_smooth_frac * fft_size),
                         max(1, int(math.ceil(cfg.flatten_smooth_frac * fft_size * 1.5)))),
                         cfg.flatten_reference_q)
        clip_vmin = ref - cfg.adaptive_floor_frac * span
        clip_inv_span = torch.full_like(clip_inv_span, 1.0 / span)

    norm = torch.clamp((db - clip_vmin.view(B, 1, 1)) * clip_inv_span.view(B, 1, 1), 0.0, 1.0)

    # bilinear freq resize wide -> nfft (matches F::interpolate in forward_downsampled)
    resized = F.interpolate(norm.unsqueeze(1), size=(rows_wide, nfft),
                            mode="bilinear", align_corners=False).squeeze(1)  # [B, rows_wide, nfft]

    n_tiles = rows_wide // cfg.tile_rows
    usable = n_tiles * cfg.tile_rows
    tiles = resized[:, :usable, :].reshape(B, n_tiles, cfg.tile_rows, nfft)
    tiles = tiles.reshape(B * n_tiles, 1, cfg.tile_rows, nfft)
    return FrontEndOut(tiles=tiles, wide_norm=norm, resized=resized, fft_size=fft_size,
                       gain_offset_db=gain_offset_db, used_robust=used_robust,
                       floor_db=floor_list, n_tiles=n_tiles)


def rasterize_gt(anns, frame_start: int, fft_size: int, rows_wide: int, nfft: int,
                 fs: float = 491_520_000.0) -> torch.Tensor:
    """GT mask on the model grid for downsample mode: rows span fft_size samples each (wide-FFT time
    rows), freq mapped to nfft bins over the full band (post-resize). anns = list with .sample_start/
    .sample_count/.freq_lower_hz/.freq_upper_hz (or the SigMF dict keys). Returns [rows_wide, nfft] uint8.
    Note samples_per_row = fft_size (NOT nfft) -- that is the downsample-mode time geometry.
    """
    mask = torch.zeros((rows_wide, nfft), dtype=torch.uint8)
    fend = frame_start + rows_wide * fft_size
    for a in anns:
        s = getattr(a, "sample_start", None)
        if s is None:  # SigMF dict
            s = int(a["core:sample_start"]); c = int(a["core:sample_count"])
            lo = float(a["core:freq_lower_edge"]); hi = float(a["core:freq_upper_edge"])
        else:
            s = int(a.sample_start); c = int(a.sample_count)
            lo = float(a.freq_lower_hz); hi = float(a.freq_upper_hz)
        st = s + c
        if s >= fend or st <= frame_start:
            continue
        r0 = max(0, (s - frame_start) // fft_size)
        r1 = min(rows_wide, -(-(st - frame_start) // fft_size))  # ceil div
        c0 = max(0, int(np.floor((lo + fs / 2.0) / fs * nfft)))
        c1 = min(nfft, int(np.ceil((hi + fs / 2.0) / fs * nfft)))
        r1 = max(r1, r0 + 1); c1 = max(c1, c0 + 1)
        if r1 <= r0 or c1 <= c0:
            continue
        mask[r0:r1, c0:c1] = 1
    return mask


def _reference(col_smooth: torch.Tensor, quantile_pct: float) -> torch.Tensor:
    """blend(mean->max) of the smoothed per-freq floor, matching ft_frontend_reference_kernel."""
    q = quantile_pct / 100.0
    mean = col_smooth.mean(dim=1)
    mx = col_smooth.max(dim=1).values
    blend = min(max((q - 0.5) / 0.5, 0.0), 1.0)
    return mean + blend * (mx - mean)
