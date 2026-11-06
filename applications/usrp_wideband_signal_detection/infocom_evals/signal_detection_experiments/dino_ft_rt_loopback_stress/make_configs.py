#!/usr/bin/env python3
"""Generate the loopback-stress experiment configs FROM the production configs.

Every leg config is the shipped live config of that detector with a small, explicit list of
edits (below), so the legs stay traceable to production. Re-run after a production config
change: `python3 make_configs.py` (writes ./configs/*.yaml).

Legs
  coherent            config_live_v3_single_channel.yaml  (coherent_power, dynamic floor)
  dinoft_native_m2dr  config_live_v3_dino_ft_m5s.yaml -> detector block set to the validated
                      M2_dr NATIVE contract (dedicated 1024-pt FFT, threshold 0.95, no flatten,
                      fixed clip) = "DINO-FT" at full resolution
  dinoft_rt_m5s       config_live_v3_dino_ft_m5s.yaml  (deployed M5-S, real_time_downsample)
                      = "DINO-FT real time"
  dinoft_native_m5s   M5-S block with real_time_downsample=false   (path ablation, same model)
  dinoft_rt_m2dr      M2_dr block with real_time_downsample=true   (path ablation, same model)
  dinoft_rt_m5s_nosnip        deployed M5-S RT without the snipper/compression/sink branch   (load ablation)
  dinoft_rt_m5s_nosnip_noviz  ... and without the dashboard: headless lean DINO-FT (needs the main.cpp
                              FFT-branch fix of 2026-10-21; used for stall / ring-guard validation)

Common edits (why):
  * finetuned_dino_detector.invalid_frame_guard=false and coherent max_emit_occupancy=0.0:
    the Argus composite legitimately lights a large fraction of band-time, which the OTA garbage
    guards (tuned for sparse live spectrum) would blank; we measure ingest health from the CHDR /
    DPDK counters instead.
  * signal_snipper.max_mask_occupancy=0.90 (dense composite; same as the loopback demo configs).
  * coherent timing_summary every 64 frames (finer latency sampling for the record).
  * emit_stride=1 everywhere: the question is whether each detector keeps up PER FRAME.
  * DPDK flow match udp 49160 -> 1240 (see LB_UDP_SRC/LB_UDP_DST): the replay pcap is port-mapped
    so the live X410 stream sharing the physical port never reaches the app.
"""
from __future__ import annotations

import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
APP = HERE.parents[2]
OUT = HERE / "configs"
OUT.mkdir(exist_ok=True)

M2DR_TS = "/workspace/holohub/dino_fine_tuning/weights/finetuned_dino_m2_dr_bf16.ts"
M5S_TS = "/workspace/holohub/dino_fine_tuning/weights/finetuned_dino_m5s_491_bf16.ts"


def set_in_block(text: str, block: str, values: dict[str, str]) -> str:
    """Set/insert `key: value` lines inside a top-level YAML block (comments preserved)."""
    lines = text.splitlines()
    start = next(i for i, l in enumerate(lines) if l.rstrip() == f"{block}:")
    end = len(lines)
    for i in range(start + 1, len(lines)):
        if lines[i].strip() and not lines[i][0].isspace():
            end = i
            break
    for key, value in values.items():
        pat = re.compile(rf"^(\s+){re.escape(key)}:\s*([^#\n]*?)\s*(#.*)?$")
        for i in range(start + 1, end):
            m = pat.match(lines[i])
            if m:
                comment = f"   {m.group(3)}" if m.group(3) else ""
                lines[i] = f"{m.group(1)}{key}: {value}{comment}"
                break
        else:
            lines.insert(start + 1, f"  {key}: {value}")
            end += 1
    return "\n".join(lines) + "\n"


# The replay is switched INSIDE the ConnectX-7 (sent from PF 0002:01:00.0, received by the DPDK PF
# 0000:01:00.0 -- same physical port p0, no cable) while the X410 keeps streaming its own CHDR
# (udp 49153 -> 1234/1235) into that port. The replay pcap is port-mapped to 49160 -> 1240 and the
# flow rule matches ONLY that, so flow isolation drops the radio's packets in hardware.
LB_UDP_SRC, LB_UDP_DST = 49160, 1240


def retarget_flow(text: str) -> str:
    assert text.count("udp_src: 49153") == 1 and text.count("udp_dst: 1234") == 1, "unexpected flow block"
    return text.replace("udp_src: 49153", f"udp_src: {LB_UDP_SRC}").replace("udp_dst: 1234", f"udp_dst: {LB_UDP_DST}")


# Off-screen dashboard (2026-10-21, after the first runs): the HoloViz window must not open on the
# shared desktop. Holoviz renders headless with `visualization.holoviz.headless: true`, so the full
# GPU load of the dashboard is still present. Results recorded before this switch were windowed.
HEADLESS = True


def set_holoviz_headless(text: str) -> str:
    lines = text.splitlines()
    vis = next(i for i, l in enumerate(lines) if l.rstrip() == "visualization:")
    hv = next(i for i in range(vis + 1, len(lines)) if lines[i].rstrip() == "  holoviz:")
    end = next((i for i in range(hv + 1, len(lines)) if lines[i].strip() and not lines[i].startswith("    ")), len(lines))
    for i in range(hv + 1, end):
        if lines[i].strip().startswith("headless:"):
            lines[i] = "    headless: true"
            break
    else:
        lines.insert(hv + 1, "    headless: true")
    return "\n".join(lines) + "\n"


def header(leg: str, base: str, notes: str) -> str:
    return (
        f"# GENERATED by infocom_evals/signal_detection_experiments/dino_ft_rt_loopback_stress/make_configs.py\n"
        f"# leg: {leg}   base: {base}\n"
        f"# {notes}\n"
        f"# Loopback eval only (dense Argus composite) -- NOT a demo config; do not use on live OTA.\n"
    )


def coherent() -> None:
    base = "config_live_v3_single_channel.yaml"
    t = (APP / base).read_text()
    t = set_in_block(t, "coherent_power_signal_detector", {
        "max_emit_occupancy": "0.0", "emit_stride": "1",
        "timing_summary_every_n": "64", "timing_summary_window": "64",
    })
    t = set_in_block(t, "signal_snipper", {"max_mask_occupancy": "0.90"})
    t = set_in_block(t, "visualization", {"detector_label": '"Coherent Power (loopback eval)"'})
    t = set_in_block(t, "chdr_converter", {
        "channel_sample_rates_hz": '"245.76e6"', "channel_center_frequencies_hz": '"2400e6"'})
    t = retarget_flow(t)
    if HEADLESS:
        t = set_holoviz_headless(t)
    (OUT / "config_lb_coherent.yaml").write_text(
        header("coherent", base, "coherent_power dynamic floor; occupancy fault gate off for the dense composite") + t)


def dino(leg: str, model: str, rt: bool, label: str, snipper: bool = True, viz: bool = True,
         extra: dict | None = None) -> None:
    base = "old_configs/config_live_v3_dino_ft_m5s.yaml"   # moved 2026-11-06 (M7-S replaced M5-S)
    t = (APP / base).read_text()
    # Pipeline-load ablations (same detector block -> same offline reference as the full-pipeline leg):
    # *_nosnip drops the snipper+compression+sink branch, *_nosnip_noviz also drops the dashboard.
    if not snipper:
        t = set_in_block(t, "pipeline", {"enable_signal_snipper": "false", "enable_snippet_compression": "false"})
    if not viz:
        t = set_in_block(t, "visualization", {"enable": "false"})
    common = {"emit_stride": "1", "invalid_frame_guard": "false",
              "real_time_downsample": "true" if rt else "false"}
    if model == "m2dr":
        # Validated M2_dr deployment contract (config_dino_finetuned_viz_demo.yaml + its meta.json):
        # fixed clip, threshold 0.95, trained UNflattened on raw captures -> flatten/adaptive off.
        common.update({
            "model_script_path": f'"{M2DR_TS}"', "threshold": "0.95",
            "db_vmin": "-47.6526985168457", "db_vmax": "20.45633316040039",
            "flatten_noise_floor": "false", "adaptive_normalization": "false",
            "circular_edge_inference": "false",
        })
    elif model == "m5s":
        common.update({"model_script_path": f'"{M5S_TS}"'})  # keep the deployed block otherwise
    if extra:
        common.update(extra)
    t = set_in_block(t, "finetuned_dino_detector", common)
    t = set_in_block(t, "signal_snipper", {"max_mask_occupancy": "0.90"})
    t = set_in_block(t, "visualization", {"detector_label": f'"{label}"'})
    notes = (f"model={model} real_time_downsample={'true' if rt else 'false'}; "
             f"invalid-frame guard off (dense composite); emit_stride 1")
    t = retarget_flow(t)
    if HEADLESS:
        t = set_holoviz_headless(t)
    (OUT / f"config_lb_{leg}.yaml").write_text(header(leg, base, notes) + t)


if __name__ == "__main__":
    coherent()
    dino("dinoft_native_m2dr", "m2dr", rt=False, label="DINO-FT native (M2_dr)")
    dino("dinoft_rt_m5s", "m5s", rt=True, label="DINO-FT RT (M5-S)")
    dino("dinoft_native_m5s", "m5s", rt=False, label="DINO-FT native (M5-S)")
    dino("dinoft_rt_m2dr", "m2dr", rt=True, label="DINO-FT RT (M2_dr)")
    dino("dinoft_rt_m5s_nosnip", "m5s", rt=True, label="DINO-FT RT (M5-S) no snipper", snipper=False)
    # "detector only" (dashboard off too). Before the 2026-10-21 main.cpp fix this deadlock-stopped the
    # graph after ~4 batches (the IQ-tapping detector left the FFT output without a consumer); main.cpp
    # now skips the FFT branch in that case, so this is the headless / lean DINO-FT configuration.
    dino("dinoft_rt_m5s_nosnip_noviz", "m5s", rt=True, label="DINO-FT RT (M5-S) detector only", snipper=False, viz=False)
    dino("dinoft_native_m2dr_nosnip_noviz", "m2dr", rt=False, label="DINO-FT native (M2_dr) detector only", snipper=False, viz=False)
    dino("dinoft_rt_m5s_noviz", "m5s", rt=True, label="DINO-FT RT (M5-S) snipper, no dashboard", snipper=True, viz=False)
    # Short-FFT front-end (downsample_time_pool): FFT at 2048 / 4096 points, rows area-averaged back to
    # the 512-row reference grid; the model sees a 2:1 / 4:1 bilinear column resize instead of 10:1 / 20:1.
    for F in (2048, 4096):
        for dom, tag in (("db", ""), ("power", "p")):
            dino(f"dinoft_rt_m5s_sf{F}{tag}", "m5s", rt=True, label=f"DINO-FT RT (M5-S) short FFT {F} {dom}",
                 extra={"downsample_fft_size": str(F), "downsample_time_pool": "true",
                        "downsample_time_pool_domain": f'"{dom}"'})
    for p in sorted(OUT.glob("config_lb_*.yaml")):
        print("wrote", p.relative_to(HERE))
