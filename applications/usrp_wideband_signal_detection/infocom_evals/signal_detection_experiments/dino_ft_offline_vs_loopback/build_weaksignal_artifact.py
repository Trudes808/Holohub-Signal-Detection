#!/usr/bin/env python3
"""Assemble the DINO-FT weak-signal-recovery deep-dive as a self-contained HTML artifact."""
import base64
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESG = HERE / "results_guard"
OUT = HERE / "dino_ft_weak_signal_recovery.html"


def img(name):
    return "data:image/png;base64," + base64.b64encode((RESG / name).read_bytes()).decode()


HTML = r"""<title>DINO-FT Weak-Signal Recovery</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
<style>
  :root{--bg:#eef1f5;--panel:#fff;--panel2:#f5f7fa;--ink:#161b22;--muted:#5b6672;--line:#d6dde6;
    --sig:#0d8f86;--warn:#c2620d;--good:#1a7f52;--bad:#c0392b;--cyan:#0aa;
    --shadow:0 1px 2px rgba(16,22,34,.06),0 8px 24px rgba(16,22,34,.06);
    --mono:"IBM Plex Mono",ui-monospace,Menlo,monospace;--sans:"IBM Plex Sans",system-ui,-apple-system,Segoe UI,Roboto,sans-serif;}
  @media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0b0e14;--panel:#131924;--panel2:#0f141d;--ink:#e6edf3;
    --muted:#94a1b2;--line:#232c39;--sig:#35d0c8;--warn:#f0a35e;--good:#4ad08a;--bad:#f0736a;--cyan:#22d3ee;
    --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 30px rgba(0,0,0,.35);}}
  :root[data-theme="dark"]{--bg:#0b0e14;--panel:#131924;--panel2:#0f141d;--ink:#e6edf3;--muted:#94a1b2;--line:#232c39;
    --sig:#35d0c8;--warn:#f0a35e;--good:#4ad08a;--bad:#f0736a;--cyan:#22d3ee;
    --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 30px rgba(0,0,0,.35);}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);line-height:1.6;-webkit-font-smoothing:antialiased}
  .wrap{max-width:940px;margin:0 auto;padding:clamp(20px,4vw,56px)}
  .eyebrow{font-family:var(--mono);font-size:12px;letter-spacing:.16em;text-transform:uppercase;color:var(--muted);margin:0 0 10px}
  h1{font-size:clamp(28px,5vw,44px);line-height:1.08;margin:0 0 12px;font-weight:700;text-wrap:balance;letter-spacing:-.01em}
  h2{font-size:22px;margin:44px 0 14px;font-weight:600;letter-spacing:-.01em}
  h2 .n{font-family:var(--mono);color:var(--muted);font-size:15px;margin-right:10px}
  p{margin:0 0 14px;max-width:70ch} .lede{font-size:18px;color:var(--muted);max-width:72ch}
  .verdict{margin:26px 0;padding:20px 22px;border-radius:14px;background:var(--panel);border:1px solid var(--line);
    border-left:4px solid var(--good);box-shadow:var(--shadow)} .verdict b{color:var(--good)}
  figure{margin:20px 0;background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:14px;box-shadow:var(--shadow)}
  figure img{display:block;width:100%;height:auto;border-radius:8px;background:#000}
  figcaption{font-size:13.5px;color:var(--muted);margin-top:10px;padding:0 4px} figcaption b{color:var(--ink);font-weight:600}
  .tablewrap{overflow-x:auto;border:1px solid var(--line);border-radius:14px;background:var(--panel);box-shadow:var(--shadow);margin:18px 0}
  table{width:100%;border-collapse:collapse;font-size:15px} th,td{padding:11px 16px;text-align:left;border-bottom:1px solid var(--line)}
  th{font-family:var(--mono);font-size:12px;letter-spacing:.05em;text-transform:uppercase;color:var(--muted);font-weight:600}
  td.num{font-family:var(--mono);text-align:right;font-variant-numeric:tabular-nums} tr:last-child td{border-bottom:none}
  tr.rec td{background:color-mix(in srgb, var(--good) 12%, transparent)} tr.bad td{color:var(--muted)}
  code{font-family:var(--mono);font-size:.92em;background:var(--panel2);border:1px solid var(--line);padding:.08em .4em;border-radius:5px}
  ul{max-width:72ch;padding-left:22px} li{margin:6px 0}
  .foot{margin-top:48px;padding-top:18px;border-top:1px solid var(--line);color:var(--muted);font-family:var(--mono);font-size:12px;line-height:1.8}
  .key{display:flex;gap:18px;flex-wrap:wrap;font-family:var(--mono);font-size:12.5px;margin:2px 0 0}
  .sw{display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:7px;vertical-align:middle}
</style>

<div class="wrap">
  <p class="eyebrow">USRP wideband signal detection · DINO-FT diagnostics</p>
  <h1>Why DINO-FT drops moderate signals — and how to get them back</h1>
  <p class="lede">The real-time DINO-FT detector masks the strongest signal in a frame but skips clearly
  visible yellow/green narrowband signals right next to it. This traces to how the model's input is
  normalized, not to the threshold or to the signals being weak — and a per-frame adaptive normalization
  recovers them without adding false positives.</p>

  <div class="verdict">
    <b>Root cause: contrast compression in the model input.</b> The fixed <code>db_vmin −47.65 … db_vmax
    +20.46</code> clip is a 68 dB window calibrated on the training sweep (0–60 dB attenuation); a live
    scene spans only ~25 dB, so it lands in the middle third of <code>[0,1]</code> — noise ≈0.32, moderate
    signals ≈0.40–0.45, strongest peak only 0.69. Moderate signals sit barely above noise, so only the
    strongest clears the bar. <b>Fix:</b> per-frame adaptive normalization (anchor <code>[0,1]</code> to the
    data-derived floor) recovers the moderate signals <b>and cuts noise false positives ~19%</b>.
  </div>

  <h2><span class="n">01</span>The model doesn't see what you see</h2>
  <p>The dashboard renders a −20…+15 dB window; the model normalizes with the fixed 68 dB clip, then
  flattens and maps to <code>[0,1]</code>. So a signal that's obviously yellow/green on screen can be a
  flat mid-grey to the model.</p>
  <figure>
    <img alt="model input as-is vs contrast-stretched" src="__MODELINPUT__">
    <figcaption><b>Top — the model input as-is</b> (<code>[0,1]</code>): a flat blue wash; only the one strong
    signal (red mask) stands out. <b>Bottom — the same tensor contrast-stretched</b>: it's full of real
    narrowband signals the model input <b>contains but did not mask</b>. In the signal band the model input
    tops out at 0.69 (p99 0.48, noise 0.32) — the scene's ~25 dB is crushed into the middle of the 68 dB clip.</figcaption>
  </figure>

  <h2><span class="n">02</span>Lowering the threshold is not the fix</h2>
  <figure>
    <img alt="threshold 0.95 vs 0.70" src="__THRCOMPARE__">
    <figcaption><b class="warn">red</b> = detected at 0.95, <b style="color:var(--cyan)">cyan</b> = what 0.70
    adds. Occupancy rises ×1.30, but the extra pixels are a spurious full-width row line, noise blobs, and
    skirt-growth on already-detected signals — and the genuinely faint edge signals stay unmasked even at
    0.70. A global threshold drop trades far more FP than real recovery.</figcaption>
  </figure>

  <h2><span class="n">03</span>A naïve re-level recovers them — but with false positives</h2>
  <figure>
    <img alt="fixed re-leveled window" src="__DBWIN__">
    <figcaption>Re-leveling to a narrow fixed window (<code>db_vmin −28, db_vmax 0</code>) <b class="good">does</b>
    recover the moderate signals (cyan on the real signals near bin ~880), proving the diagnosis — but it also
    over-brightens the noise/DC region and fires large <b class="bad">false-positive blobs on the left</b>. A
    fixed narrow window amplifies noise fluctuations. Same recall/FP tension as the threshold.</figcaption>
  </figure>

  <h2><span class="n">04</span>The fix: per-frame adaptive normalization</h2>
  <p>Instead of a fixed clip, anchor the <code>[0,1]</code> mapping to the <b>per-frame data-derived noise
  floor</b> (reusing the flatten reference) with a fixed span: <code>vmin = floor − floor_frac·span</code>,
  <code>1/span</code>. The floor always lands at ~0.12 (dark, so noise doesn't fire) and signals brighten to
  fill the range like training — and it tracks gain/rate automatically. Default <code>adaptive_span_db 34</code>.</p>
  <figure>
    <img alt="current vs adaptive normalization" src="__ADAPT__">
    <figcaption><b class="warn">red</b> = current, <b style="color:var(--cyan)">cyan</b> = added by adaptive
    (span 34). It recovers the real narrowband yellow/green signals (frame 9: bins ~960–1050; frame 15: ~790)
    <b>with a clean left half — no noise false positives</b>, unlike the fixed re-level above.</figcaption>
  </figure>

  <h2><span class="n">05</span>Span sweep — recall by band vs false positives</h2>
  <p>Columns classified by time-averaged energy: <b>strong</b> (&gt;6 dB over floor), <b>moderate</b> (2–6 dB),
  <b>noise</b> (&lt;1.5 dB). Recall = mask occupancy in that band; noise-FP = mask occupancy in noise columns.</p>
  <div class="tablewrap"><table>
    <tr><th>normalization</th><th class="num">strong %</th><th class="num">moderate %</th><th class="num">noise-FP %</th></tr>
    <tr><td>current — fixed 68 dB clip</td><td class="num">3.75</td><td class="num">0.805</td><td class="num">0.154</td></tr>
    <tr><td>adaptive span 28</td><td class="num">3.53</td><td class="num">0.692</td><td class="num">0.120</td></tr>
    <tr class="rec"><td>adaptive span 34 — recommended</td><td class="num">3.76</td><td class="num">0.854</td><td class="num">0.124</td></tr>
    <tr><td>adaptive span 40</td><td class="num">3.53</td><td class="num">0.902</td><td class="num">0.127</td></tr>
    <tr class="bad"><td>fixed −28/0 (naïve re-level)</td><td class="num">3.98</td><td class="num">1.214</td><td class="num">0.237</td></tr>
  </table></div>
  <p><b>Read:</b> adaptive span 34 keeps full strong recall (3.76 ≈ 3.75), lifts moderate +6% (0.805→0.854),
  and cuts noise FP ~19% (0.154→0.124). Span 40 recovers a bit more moderate (0.902) at a small strong-recall
  cost. The naïve fixed −28/0 recovers the most moderate (1.21%) but at 1.5× the noise FP — the bad operating
  point. All numbers on the same 1 s / 491.52 MSps / 2.4 GHz offline capture (46 frames).</p>

  <h2><span class="n">06</span>Recommendation &amp; limits</h2>
  <ul>
    <li><b>Ship per-frame adaptive normalization</b> (<code>adaptive_normalization: true</code>,
      <code>adaptive_span_db 34</code>): recovers moderate signals, reduces FP, robust to gain — no retrain.
      Default OFF; a config flag flips it on.</li>
    <li><b>Not the threshold</b> (adds FP) and <b>not a fixed re-level</b> (gain-fragile + FP).</li>
    <li><b>The ceiling is the model.</b> Beyond ~+6–12% moderate recovery, pushing harder re-introduces FP
      because the checkpoint is off-distribution at the 491.52 deployment geometry/contrast. A
      <b>domain-match fine-tune</b> (train on the deployment-normalized inputs) is the durable way to raise
      moderate-signal recall further without the FP trade.</li>
  </ul>

  <div class="foot">
    capture x410_ota_2g4_gain10_20260908 · 491.52 MSps · 2.4 GHz · 1 s OTA · 46 frames · offline (deterministic)<br>
    operators/finetuned_dino_detector: adaptive_normalization / adaptive_span_db / adaptive_floor_frac<br>
    infocom_evals/signal_detection_experiments/dino_ft_offline_vs_loopback · 2026-09-14
  </div>
</div>
"""

for k, v in {
    "__MODELINPUT__": img("missed_signals_zoom540.png"),
    "__THRCOMPARE__": img("threshold_compare.png"),
    "__DBWIN__": img("dbwindow_compare.png"),
    "__ADAPT__": img("adaptive_compare.png"),
}.items():
    HTML = HTML.replace(k, v)

OUT.write_text(HTML)
print("wrote", OUT, f"({OUT.stat().st_size/1e6:.2f} MB)")
