#!/usr/bin/env python3
"""Assemble the DINO-FT offline-vs-loopback A/B report as a self-contained HTML artifact."""
import base64
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
OUT = HERE / "dino_ft_ab_report.html"

s = json.loads((RES / "summary.json").read_text())
off, lb = s["offline"], s["loopback"]

# Fixed run (detector invalid-frame guard).
RESG = HERE / "results_guard"
sg = json.loads((RESG / "summary.json").read_text())
lbg = sg["loopback"]
rg = sg.get("occ_spectrum_pearson_r", float("nan"))


def img(name: str) -> str:
    b = (RES / name).read_bytes()
    return "data:image/png;base64," + base64.b64encode(b).decode()


def pct(x):
    return f"{x:.2f}%"


r = s.get("occ_spectrum_pearson_r", float("nan"))
spike_ratio = lb["frame_occ_pct_max"] / off["frame_occ_pct_max"]

HTML = r"""<title>DINO-FT Ingest A/B</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
<style>
  :root{
    --bg:#eef1f5; --panel:#ffffff; --panel2:#f5f7fa; --ink:#161b22; --muted:#5b6672;
    --line:#d6dde6; --offline:#0d8f86; --loopback:#c2620d; --good:#1a7f52; --warn:#b8860b;
    --shadow:0 1px 2px rgba(16,22,34,.06),0 8px 24px rgba(16,22,34,.06);
    --mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace;
    --sans:"IBM Plex Sans",system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
  }
  @media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
    --bg:#0b0e14; --panel:#131924; --panel2:#0f141d; --ink:#e6edf3; --muted:#94a1b2;
    --line:#232c39; --offline:#35d0c8; --loopback:#f0a35e; --good:#4ad08a; --warn:#e9c46a;
    --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 30px rgba(0,0,0,.35);
  }}
  :root[data-theme="dark"]{
    --bg:#0b0e14; --panel:#131924; --panel2:#0f141d; --ink:#e6edf3; --muted:#94a1b2;
    --line:#232c39; --offline:#35d0c8; --loopback:#f0a35e; --good:#4ad08a; --warn:#e9c46a;
    --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 30px rgba(0,0,0,.35);
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);
       line-height:1.6;-webkit-font-smoothing:antialiased}
  .wrap{max-width:940px;margin:0 auto;padding:clamp(20px,4vw,56px)}
  .eyebrow{font-family:var(--mono);font-size:12px;letter-spacing:.16em;text-transform:uppercase;
           color:var(--muted);margin:0 0 10px}
  h1{font-size:clamp(28px,5vw,44px);line-height:1.08;margin:0 0 12px;font-weight:700;text-wrap:balance;letter-spacing:-.01em}
  h2{font-size:22px;margin:44px 0 14px;font-weight:600;letter-spacing:-.01em}
  h2 .n{font-family:var(--mono);color:var(--muted);font-size:15px;margin-right:10px}
  p{margin:0 0 14px;max-width:70ch}
  .lede{font-size:18px;color:var(--muted);max-width:72ch}
  .verdict{margin:26px 0;padding:20px 22px;border-radius:14px;background:var(--panel);
           border:1px solid var(--line);border-left:4px solid var(--good);box-shadow:var(--shadow)}
  .verdict b{color:var(--good)}
  .grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:22px 0}
  @media(max-width:640px){.grid{grid-template-columns:1fr}}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:18px 20px;box-shadow:var(--shadow)}
  .card h3{margin:0 0 6px;font-size:13px;font-family:var(--mono);letter-spacing:.08em;text-transform:uppercase;color:var(--muted);font-weight:600}
  .stat{font-family:var(--mono);font-size:30px;font-weight:600;letter-spacing:-.02em;font-variant-numeric:tabular-nums}
  .stat.off{color:var(--offline)} .stat.lb{color:var(--loopback)}
  .sub{font-size:13px;color:var(--muted)}
  table{width:100%;border-collapse:collapse;margin:8px 0 4px;font-size:15px}
  th,td{padding:11px 12px;text-align:left;border-bottom:1px solid var(--line)}
  th{font-family:var(--mono);font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);font-weight:600}
  td.num{font-family:var(--mono);text-align:right;font-variant-numeric:tabular-nums}
  .off{color:var(--offline)} .lb{color:var(--loopback)}
  .tablewrap{overflow-x:auto;border:1px solid var(--line);border-radius:14px;background:var(--panel);box-shadow:var(--shadow)}
  .tablewrap table{margin:0} .tablewrap td,.tablewrap th{padding-left:18px;padding-right:18px}
  .tablewrap tr:last-child td{border-bottom:none}
  figure{margin:20px 0;background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:14px;box-shadow:var(--shadow)}
  figure img{display:block;width:100%;height:auto;border-radius:8px;background:#000}
  figcaption{font-size:13.5px;color:var(--muted);margin-top:10px;padding:0 4px}
  figcaption b{color:var(--ink);font-weight:600}
  .legend{display:flex;gap:18px;flex-wrap:wrap;font-family:var(--mono);font-size:12.5px;margin:2px 0 0}
  .swatch{display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:7px;vertical-align:middle}
  ul{max-width:72ch;padding-left:22px} li{margin:6px 0}
  .foot{margin-top:48px;padding-top:18px;border-top:1px solid var(--line);color:var(--muted);
        font-family:var(--mono);font-size:12px;line-height:1.8}
  code{font-family:var(--mono);font-size:.92em;background:var(--panel2);border:1px solid var(--line);
       padding:.08em .4em;border-radius:5px}
</style>

<div class="wrap">
  <p class="eyebrow">USRP wideband signal detection · validation</p>
  <h1>Is the real-time DINO-FT&nbsp;bad, or is the ingest?</h1>
  <p class="lede">The fine-tuned DINOv3 segmenter (M2_dr) looks atrocious live. We ran the
  <em>identical compiled operator</em> two ways on one 1&nbsp;s / 491.52&nbsp;MSps / 2.4&nbsp;GHz capture —
  reading the file directly vs. replaying it over the real DPDK path — so the only variable is the ingest.</p>

  <div class="verdict">
    <b>Verdict — the model is not the problem.</b> Offline and loopback agree almost perfectly
    (occupancy-spectrum correlation <b style="color:var(--ink)">r&nbsp;=&nbsp;__R__</b>, same bands, same global
    occupancy, zero full-width bars in either). The live "atrocious" look comes from <b>intermittent
    ingest-frame corruption</b>: ~1&ndash;2 frames in 250 blow up into a large spurious off-band blob
    (peak frame occupancy <span class="lb">__LBMAX__</span> vs offline's <span class="off">__OFFMAX__</span>,
    a __SPIKEX__&times; spike). Fix the ingest, not the model.
  </div>

  <div class="verdict" style="border-left-color:var(--offline)">
    <b style="color:var(--offline)">Fixed &amp; verified.</b> The confirmed cause is <b style="color:var(--ink)">dropped-packet
    fill corruption under GPU saturation</b> (a perf ceiling, not the ring). A detector <b style="color:var(--ink)">invalid-frame
    guard</b> now emits an empty mask when a frame's occupancy is a gross outlier &mdash; a drop-corrupted
    frame is invalid, so it produces no detections. Result: peak frame occupancy
    <span class="lb">__LBMAX__</span> &rarr; <span class="off">__GMAX__</span>, the dramatic 6&ndash;10% spikes go to
    <b style="color:var(--ink)">zero</b>, and guarded loopback matches clean offline (spectrum r&nbsp;=&nbsp;<b style="color:var(--ink)">__GR__</b>).
  </div>

  <div class="grid">
    <div class="card"><h3>Occupancy-spectrum agreement</h3><div class="stat">r = __R__</div>
      <div class="sub">Pearson correlation, offline vs loopback per-frequency detection rate</div></div>
    <div class="card"><h3>Global occupancy</h3><div class="stat">__RATIO__&times;</div>
      <div class="sub">loopback __LBG__ vs offline __OFFG__ &mdash; essentially equal</div></div>
    <div class="card"><h3>Peak frame — offline</h3><div class="stat off">__OFFMAX__</div>
      <div class="sub">clean file input never exceeds this</div></div>
    <div class="card"><h3>Peak frame — loopback</h3><div class="stat lb">__LBMAX__</div>
      <div class="sub">rare corrupted frames, __SPIKEN__ / 250 above offline max</div></div>
  </div>

  <h2><span class="n">01</span>Method</h2>
  <p>One capture, one compiled <code>finetuned_dino_detector</code> (real-time downsample: wide FFT →
  bilinear-resize freq → tile → segment → circular-edge stitch → flatten), two delivery paths:</p>
  <ul>
    <li><b class="off">offline</b> — <code>run_cuda_dino_offline_file.py</code> reads the SigMF file
      directly. Deterministic, no packets.</li>
    <li><b class="lb">loopback</b> — the same capture packetized to a CHDR pcap and replayed over DPDK
      at the live 491.52&nbsp;MSps into the real <code>chdr_converter → FFT → detector</code> pipeline.
      The operator dumps every emitted mask (new default-off <code>debug_mask_dump_dir</code>).</li>
  </ul>
  <p>Detector config blocks are byte-identical; masks are 512×20480 uint8 on the same wide-FFT grid.
  Because the legs aren't frame-aligned (loopback loops the second, drops under load), comparison is on
  ingest-robust aggregate statistics.</p>

  <h2><span class="n">02</span>The bands match — the outliers don't</h2>
  <figure>
    <img alt="occupancy raster, offline vs loopback" src="__RASTER__">
    <div class="legend"><span><span class="swatch" style="background:var(--offline)"></span>offline</span>
      <span><span class="swatch" style="background:var(--loopback)"></span>loopback</span></div>
    <figcaption>Per-frame occupancy (frame × frequency). <b>Both</b> concentrate on the real 2.4&nbsp;GHz
    activity (center bins). The loopback panel adds <b>bright horizontal streaks at isolated frames</b>
    (e.g. frames ~160, ~175, ~235) spanning frequencies with no real signal — the corrupted frames.</figcaption>
  </figure>
  <figure>
    <img alt="occupancy spectrum overlay" src="__SPECTRUM__">
    <figcaption>Time-averaged occupancy vs frequency. The two curves overlay almost exactly
    (<b>r = __R__</b>): the detector selects the same frequencies regardless of ingest path.</figcaption>
  </figure>

  <h2><span class="n">03</span>Offline is clean and accurate</h2>
  <figure>
    <img alt="offline sample frames" src="__OFFRAMES__">
    <figcaption>Highest-occupancy <b class="off">offline</b> frames: raw spectrogram with the DINO-FT
    mask (red). Detections sit on the real bursts; band edges are clean (circular-edge inference);
    no bars, no off-band firing. This is the model behaving correctly.</figcaption>
  </figure>

  <h2><span class="n">04</span>Loopback's worst frames are corrupted ingest</h2>
  <figure>
    <img alt="loopback worst frames" src="__LBFRAMES__">
    <figcaption>Highest-occupancy <b class="lb">loopback</b> masks. Frame 560 (<b>6.98%</b>) is a solid
    spurious blob across the low half of the band where there is no signal — a corrupted frame. Frame
    944 fires off-center. The remaining "worst" frames (~1.3%) are actually <b>legitimate</b> busy-signal
    frames at band center. Only ~1&ndash;2 of 250 are truly corrupt — rare, but dramatic on screen.</figcaption>
  </figure>

  <h2><span class="n">05</span>What this means</h2>
  <ul>
    <li><b>Do not touch the model or its downsample geometry.</b> Offline RT is clean; loopback matches
      it to r&nbsp;=&nbsp;__R__.</li>
    <li>The live artifact is <b>intermittent ingest-frame corruption</b> → DINO-FT fires a large spurious
      off-band blob on the bad frame. At the live mask cadence that's a blob every few seconds.</li>
    <li>This occurred with <b>clean uniform 1024-sample packets</b>. The real radio streams mixed
      1024/1008 native-DDC framing (~44&ndash;50% short packets), which drives more corruption — so
      live-on-radio is worse than this loopback.</li>
    <li>The live "yellow bar" is a <b>raw-spectrogram</b> artifact, not a detection — neither leg shows
      full-width bars in the mask, matching that it never appears in the PSD.</li>
  </ul>
  <p><b>Next:</b> chase the corruption in <code>chdr_converter</code> (short-packet gather / slot-ring
  recycling / drop-on-incomplete-frame), and add a detector-side guard that suppresses a mask when its
  source frame is flagged incomplete. Optionally re-run this A/B with a mixed-framing pcap to reproduce
  the radio's exact packet mix.</p>

  <h2><span class="n">06</span>The fix &mdash; invalid-frame guard</h2>
  <p>Two converter "root fixes" targeting the ring were tried and reverted (they left the spikes and one
  halved throughput). The confirmed cause is <b>dropped-packet fill corruption under saturation</b> &mdash; a
  perf ceiling. Since a drop-corrupted frame is <em>invalid</em>, the detector now suppresses it: emit an
  empty mask when a frame's occupancy is a gross outlier vs an adaptive baseline
  (<code>invalid_frame_min_occupancy 0.03</code>, <code>k&times;baseline 6</code>).</p>
  <div class="grid">
    <div class="card"><h3>Peak frame occ &mdash; before</h3><div class="stat lb">__LBMAX__</div>
      <div class="sub">loopback, drop-corrupted blobs</div></div>
    <div class="card"><h3>Peak frame occ &mdash; guarded</h3><div class="stat off">__GMAX__</div>
      <div class="sub">= legitimate dense signal; __GSUP__ frames suppressed to empty</div></div>
    <div class="card"><h3>Spectrum agreement</h3><div class="stat">r = __GR__</div>
      <div class="sub">guarded loopback vs offline (was __R__)</div></div>
    <div class="card"><h3>Global occ vs offline</h3><div class="stat">__GRATIO__&times;</div>
      <div class="sub">guarded __GG__ vs offline __OFFG__</div></div>
  </div>
  <figure>
    <img alt="guarded occupancy raster" src="__GRASTER__">
    <figcaption>Guarded loopback occupancy raster vs offline. The dramatic off-band blobs are gone and the
    two panels track the same real activity. <b>Residual:</b> a few faint thin/wide streaks (~1.4%, just
    under the floor) survive &mdash; drop artifacts an occupancy threshold can't separate from dense real
    frames; a frequency-span discriminator or a lower rate would clear them.</figcaption>
  </figure>

  <h2><span class="n">07</span>Real-time masks on the actual spectrogram</h2>
  <p>The clinching view: the <b>real-time (loopback) DINO-FT mask</b> overlaid on the <b>exact input
  spectrogram the model saw</b> that frame (the operator dumps its normalized input alongside the mask;
  captured guard-OFF so a corrupted frame is shown intact). Red = detection.</p>
  <figure>
    <img alt="RT DINO-FT masks overlaid on the spectrogram" src="__RTOVERLAY__">
    <figcaption>Top two frames (9.0%, 7.0%): a large red blob sits over a <b>pure-noise</b> region &mdash; the
    real signals are near bin ~1150 &mdash; a textbook drop-corrupted frame; the <b>invalid-frame guard
    suppresses these</b>. Middle-lower frames: masks land tightly on the <b>real</b> signals (clean, kept).
    Frame 276 (2.63%) is an honest borderline &mdash; a partial spurious blob just under the 3% floor that
    still survives, i.e. the documented residual a frequency-span discriminator would catch.</figcaption>
  </figure>

  <h2><span class="n">08</span>Weak signals: lowering the threshold is not the fix</h2>
  <p>A natural idea for the missed faint signals is to lower the 0.95 decision threshold. Tested offline
  (same capture, no drops) at 0.70 vs 0.95 &mdash; <b class="lb">red = detected at 0.95</b>,
  <b style="color:#22d3ee">cyan = what 0.70 adds</b>. Occupancy rises 0.293% &rarr; 0.382% (&times;1.30), but
  the recovered pixels are mostly the wrong things.</p>
  <figure>
    <img alt="threshold 0.95 vs 0.70 comparison" src="__THRCOMPARE__">
    <figcaption>Top frame: the cyan gain is a <b>spurious full-width row line</b> plus noise, not a signal.
    Other frames: cyan mostly <b>grows skirts around already-detected strong signals</b>. Crucially, the
    genuinely faint signals (e.g. the streaks on the far right edge) stay <b>unmasked even at 0.70</b>. So a
    global threshold drop adds false positives faster than it recovers real weak signals &mdash; the small
    signals are lost in the 491.52 downsample front-end (bilinear resize diluting narrow signals; flatten
    swallowing weak edge signals), not at the threshold. Real fixes: max-pool the freq resize, freq-tile to
    native 240 kHz/bin, level/flatten tuning, or a domain-match fine-tune.</figcaption>
  </figure>

  <div class="foot">
    capture x410_ota_2g4_gain10_20260908 · 491.52 MSps · 2.4 GHz · 1 s OTA (ci16)<br>
    offline __OFFN__ frames · loopback __LBN__ frames · masks 512×20480 · emit_stride 4 · threshold 0.95<br>
    infocom_evals/signal_detection_experiments/dino_ft_offline_vs_loopback · 2026-09-14
  </div>
</div>
"""

repl = {
    "__R__": f"{r:.3f}",
    "__OFFMAX__": pct(off["frame_occ_pct_max"]),
    "__LBMAX__": pct(lb["frame_occ_pct_max"]),
    "__SPIKEX__": f"{spike_ratio:.1f}",
    "__SPIKEN__": "4",
    "__RATIO__": f"{s['global_occ_ratio_loopback_over_offline']:.2f}",
    "__LBG__": pct(lb["global_occ_pct"]),
    "__OFFG__": pct(off["global_occ_pct"]),
    "__OFFN__": str(off["n_frames"]),
    "__LBN__": str(lb["n_frames"]),
    "__RASTER__": img("occupancy_raster.png"),
    "__SPECTRUM__": img("occupancy_spectrum.png"),
    "__OFFRAMES__": img("offline_sample_frames.png"),
    "__LBFRAMES__": img("loopback_worst_frames.png"),
    "__GMAX__": pct(lbg["frame_occ_pct_max"]),
    "__GR__": f"{rg:.3f}",
    "__GSUP__": "6",
    "__GRATIO__": f"{sg['global_occ_ratio_loopback_over_offline']:.2f}",
    "__GG__": pct(lbg["global_occ_pct"]),
    "__GRASTER__": "data:image/png;base64," +
                   base64.b64encode((RESG / "occupancy_raster.png").read_bytes()).decode(),
    "__RTOVERLAY__": "data:image/png;base64," +
                     base64.b64encode((RESG / "rt_mask_overlays.png").read_bytes()).decode(),
    "__THRCOMPARE__": "data:image/png;base64," +
                      base64.b64encode((RESG / "threshold_compare.png").read_bytes()).decode(),
}
for k, v in repl.items():
    HTML = HTML.replace(k, v)

OUT.write_text(HTML)
print("wrote", OUT, f"({OUT.stat().st_size/1e6:.2f} MB)")
