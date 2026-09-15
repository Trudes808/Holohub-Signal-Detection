#!/usr/bin/env python3
"""Assemble the self-contained DINO-FT retrain report (base64-embeds the figures)."""
from __future__ import annotations
import base64, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
OUT = Path("/home/genesys-dgx1/.claude/jobs/323ea032/tmp/dino_ft_report.html")


def b64(p: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()


def main():
    ab = {r["model"] + "_" + r["composite"]: r for r in json.loads((RES / "ab_summary.json").read_text())}
    fig_snr = b64(RES / "m3_heldout" / "heldout_snr.png")
    fig_ota = b64(RES / "live_ab_ota.png")
    fig_norm = b64(RES / "robust_norm_compare.png")
    fig_gen = b64(RES / "gen491_check.png")

    html = f"""<title>DINO-FT Weak-Signal Fix</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
:root {{
  --bg:#f4f3f7; --panel:#ffffff; --ink:#191622; --muted:#5b5670; --line:#e2dfeb;
  --accent:#e8791f; --accent2:#0891a8; --good:#1f9d6b; --bad:#c8442e;
  --mono:'IBM Plex Mono',ui-monospace,monospace; --sans:'IBM Plex Sans',system-ui,sans-serif;
}}
:root:not([data-theme=light]) {{ }}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme=light]) {{
    --bg:#100e17; --panel:#1a1724; --ink:#ece9f4; --muted:#a49fb8; --line:#2c2838;
    --accent:#f79a45; --accent2:#3bc2d6; --good:#4cd396; --bad:#f0715a;
  }}
}}
:root[data-theme=dark] {{
  --bg:#100e17; --panel:#1a1724; --ink:#ece9f4; --muted:#a49fb8; --line:#2c2838;
  --accent:#f79a45; --accent2:#3bc2d6; --good:#4cd396; --bad:#f0715a;
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink); font-family:var(--sans);
  line-height:1.6; -webkit-font-smoothing:antialiased; }}
.wrap {{ max-width:60rem; margin:0 auto; padding:3rem 1.5rem 5rem; }}
header {{ border-bottom:2px solid var(--accent); padding-bottom:1.5rem; margin-bottom:2.5rem; }}
.eyebrow {{ font-family:var(--mono); font-size:.72rem; letter-spacing:.18em; text-transform:uppercase;
  color:var(--accent); margin:0 0 .6rem; }}
h1 {{ font-size:clamp(1.8rem,4vw,2.6rem); line-height:1.1; margin:0 0 .6rem; font-weight:600;
  letter-spacing:-.01em; text-wrap:balance; }}
.sub {{ color:var(--muted); font-size:1.05rem; max-width:44rem; margin:0; }}
h2 {{ font-size:1.3rem; margin:2.8rem 0 .8rem; font-weight:600; letter-spacing:-.01em;
  display:flex; align-items:baseline; gap:.6rem; }}
h2 .n {{ font-family:var(--mono); font-size:.85rem; color:var(--accent); }}
p {{ max-width:44rem; }}
figure {{ margin:1.4rem 0; background:var(--panel); border:1px solid var(--line); border-radius:10px;
  padding:.8rem; overflow-x:auto; }}
figure img {{ width:100%; max-width:100%; display:block; border-radius:6px; }}
figcaption {{ font-size:.85rem; color:var(--muted); margin-top:.6rem; font-family:var(--mono); }}
table {{ border-collapse:collapse; width:100%; margin:1.2rem 0; font-size:.92rem;
  font-variant-numeric:tabular-nums; }}
th,td {{ text-align:right; padding:.5rem .7rem; border-bottom:1px solid var(--line); }}
th:first-child,td:first-child {{ text-align:left; }}
thead th {{ font-family:var(--mono); font-size:.72rem; letter-spacing:.06em; text-transform:uppercase;
  color:var(--muted); border-bottom:2px solid var(--line); }}
.win {{ color:var(--good); font-weight:600; }}
.old {{ color:var(--muted); }}
.card {{ background:var(--panel); border:1px solid var(--line); border-left:3px solid var(--accent);
  border-radius:8px; padding:1rem 1.2rem; margin:1.4rem 0; }}
.kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:1rem; margin:1.6rem 0; }}
.kpi {{ background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:1rem 1.1rem; }}
.kpi .v {{ font-size:1.7rem; font-weight:600; font-family:var(--mono); letter-spacing:-.02em; }}
.kpi .v .arrow {{ color:var(--accent2); }}
.kpi .l {{ font-size:.78rem; color:var(--muted); margin-top:.2rem; }}
code {{ font-family:var(--mono); font-size:.85em; background:var(--line); padding:.1em .4em; border-radius:4px; }}
.foot {{ margin-top:3rem; padding-top:1.2rem; border-top:1px solid var(--line); color:var(--muted);
  font-size:.85rem; }}
.pending {{ border-left-color:var(--accent2); }}
</style>
<div class="wrap">
<header>
  <p class="eyebrow">USRP X410 · DINOv3 signal detector · 491.52 MSps · GB10</p>
  <h1>Fixing the DINO-FT detector's weak-signal blindness</h1>
  <p class="sub">The fine-tuned RF segmentation model under-detected live at 2.4&nbsp;GHz — masking only the
  dominant signal and missing clear, small ones. The root cause was a train/deploy mismatch, fixed with a
  density-robust normalization plus a domain-matched retrain. Offline-validated end-to-end.</p>
</header>

<div class="kpis">
  <div class="kpi"><div class="v">43<span class="arrow">→</span>86<span style="font-size:1rem">%</span></div>
    <div class="l">sparse-scene region detection (operator A/B vs old model)</div></div>
  <div class="kpi"><div class="v">83–100<span style="font-size:1rem">%</span></div>
    <div class="l">held-out detection across all SNR (down to −3&nbsp;dB)</div></div>
  <div class="kpi"><div class="v">0.96<span style="font-size:1rem"> IoU</span></div>
    <div class="l">val segmentation IoU (from 0.75 at epoch&nbsp;0)</div></div>
  <div class="kpi"><div class="v">2.6<span style="font-size:1rem">×</span></div>
    <div class="l">more real signal masked on the actual OTA capture</div></div>
</div>

<h2><span class="n">01</span> The problem</h2>
<p>Deployed at 491.52&nbsp;MSps, the checkpoint (<code>M2_dr</code>) masked the loudest signal and left clear,
smaller ones unmarked. Two compounding causes:</p>
<p><strong>Contrast compression.</strong> The model used a fixed 68&nbsp;dB clip calibrated to a wide bench
scene. A narrow live scene (~25&nbsp;dB span) got crushed toward the noise floor, so moderate signals read
as background.</p>
<p><strong>Geometry/level mismatch.</strong> <code>M2_dr</code> was trained at 245.76&nbsp;MSps native
(240&nbsp;kHz/bin) but deployed at 491.52 via wide-FFT downsample (480&nbsp;kHz/bin, full band) — every
signal seen at the wrong scale and brightness distribution.</p>

<h2><span class="n">02</span> Fix, part A — density/noise-robust normalization</h2>
<p>The per-frame adaptive normalization anchored to a signal-biased floor estimate and collapsed at
density/SNR extremes. The robust path takes a <strong>low percentile (p20) of the whole dB image</strong>
as the floor, anchors a fixed span to it, and falls back to the fixed clip only when a frame has no usable
dynamic range. It never collapses:</p>
<figure><img src="{fig_norm}" alt="normalization comparison">
  <figcaption>Model input [0,1] across scenes × modes. The old adaptive over-saturates the dense/noiseless
  frame (45% of pixels white); robust holds it at 7% while keeping contrast on every scene.</figcaption>
</figure>
<div class="card"><strong>Key insight.</strong> Robust normalization preserved the signals but the
<em>fixed-clip-trained</em> model didn't recognize its new brightness — so normalization alone wasn't
enough. The model had to be trained <em>on</em> the robust input. That is part B.</div>

<h2><span class="n">03</span> Fix, part B — domain-matched retrain (M3_491)</h2>
<p>A torch replica of the exact deployment front-end (wide&nbsp;FFT → flatten → robust&nbsp;norm → resize →
tile) was built and verified against the CUDA operator at <strong>correlation 1.0000</strong> on the real
capture. Real modulation waveforms (9 classes) were placed on a 491.52 canvas — dense <em>and</em> sparse,
with per-signal SNR spanning near/far — and streamed through that front-end into the training set. The
segmenter was fine-tuned from the DINOv3 backbone on this deployment-identical input.</p>
<figure><img src="{fig_gen}" alt="synthetic composites">
  <figcaption>Synthetic 491.52 composites (dense, top; sparse, bottom) with ground-truth boxes (cyan) — real
  modulation footprints across bandwidth, duration and SNR.</figcaption>
</figure>

<h2><span class="n">04</span> Results</h2>
<p>Held-out detection stays high across the whole SNR range, including the weakest signals — dense and
sparse alike:</p>
<figure><img src="{fig_snr}" alt="detection vs SNR">
  <figcaption>Region detection rate vs per-signal SNR on held-out composites (deploy threshold 0.6).</figcaption>
</figure>
<p>Through the <strong>real operator</strong> (deployment path) on held-out composites, new vs old:</p>
<table>
  <thead><tr><th>Model</th><th>Scene</th><th>Pixel recall</th><th>Region detection</th><th>IoU</th></tr></thead>
  <tbody>
  <tr class="old"><td>M2_dr (old)</td><td>dense</td><td>{ab['m2_dense_s9000']['pix_recall']:.0f}%</td><td>{ab['m2_dense_s9000']['region_detrate']:.0f}%</td><td>{ab['m2_dense_s9000']['pix_iou']:.0f}%</td></tr>
  <tr class="old"><td>M2_dr (old)</td><td>sparse</td><td>{ab['m2_sparse_s10000']['pix_recall']:.0f}%</td><td>{ab['m2_sparse_s10000']['region_detrate']:.0f}%</td><td>{ab['m2_sparse_s10000']['pix_iou']:.0f}%</td></tr>
  <tr><td><strong>M3_491 (new)</strong></td><td>dense</td><td class="win">{ab['m3_dense_s9000']['pix_recall']:.0f}%</td><td class="win">{ab['m3_dense_s9000']['region_detrate']:.0f}%</td><td class="win">{ab['m3_dense_s9000']['pix_iou']:.0f}%</td></tr>
  <tr><td><strong>M3_491 (new)</strong></td><td>sparse</td><td class="win">{ab['m3_sparse_s10000']['pix_recall']:.0f}%</td><td class="win">{ab['m3_sparse_s10000']['region_detrate']:.0f}%</td><td class="win">{ab['m3_sparse_s10000']['pix_iou']:.0f}%</td></tr>
  </tbody>
</table>
<p>And on the <strong>real 2.4&nbsp;GHz OTA capture</strong> — the scene that originally looked "atrocious" —
the new model marks the signals the old one missed, landing on real energy rather than the noise field:</p>
<figure><img src="{fig_ota}" alt="real OTA masks old vs new">
  <figcaption>Old M2_dr (left) vs new M3_491 (right) masks on the real capture. Frame 16 goes from a single
  faint blob to several detections; overall masked signal area rises 2.6×.</figcaption>
</figure>

<h2><span class="n">05</span> Status &amp; what's next</h2>
<div class="card pending"><strong>Shipped &amp; offline-validated</strong> on branch <code>live_demo</code>
(committed, not yet pushed). The five DINO-FT demo configs point at <code>M3_491</code> with the robust
normalization on and threshold&nbsp;0.6.<br><br>
<strong>Remaining:</strong> a live radio run for the final on-air confirmation, then push.</div>

<p class="foot">Robust normalization + DINOv3 ViT-B/16 segmenter fine-tune · trained and validated on a
DGX&nbsp;Spark (GB10). Figures regenerate from the committed pipeline in
<code>infocom_evals/signal_detection_experiments/dino_ft_robust_norm/</code>.</p>
</div>
"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html)
    print("wrote", OUT, f"({len(html)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
