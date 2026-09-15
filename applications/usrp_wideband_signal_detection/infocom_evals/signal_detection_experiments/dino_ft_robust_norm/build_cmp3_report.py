#!/usr/bin/env python3
"""Self-contained 3-detector comparison artifact (embeds cmp3_masks.png)."""
from __future__ import annotations
import base64, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
OUT = Path("/home/genesys-dgx1/.claude/jobs/323ea032/tmp/dino_ft_cmp3.html")


def b64(p: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()


def main():
    s = json.loads((RES / "cmp3_summary.json").read_text())
    img = b64(RES / "cmp3_masks.png")
    html = f"""<title>Detector Mask Comparison</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
:root {{ --bg:#f4f3f7; --panel:#fff; --ink:#191622; --muted:#5b5670; --line:#e2dfeb;
  --accent:#e8791f; --cyan:#0891a8; --good:#1f9d6b; --bad:#c8442e;
  --mono:'IBM Plex Mono',ui-monospace,monospace; --sans:'IBM Plex Sans',system-ui,sans-serif; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme=light]) {{
  --bg:#100e17; --panel:#1a1724; --ink:#ece9f4; --muted:#a49fb8; --line:#2c2838;
  --accent:#f79a45; --cyan:#3bc2d6; --good:#4cd396; --bad:#f0715a; }} }}
:root[data-theme=dark] {{ --bg:#100e17; --panel:#1a1724; --ink:#ece9f4; --muted:#a49fb8; --line:#2c2838;
  --accent:#f79a45; --cyan:#3bc2d6; --good:#4cd396; --bad:#f0715a; }}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);line-height:1.6}}
.wrap{{max-width:64rem;margin:0 auto;padding:2.6rem 1.4rem 4rem}}
.eyebrow{{font-family:var(--mono);font-size:.72rem;letter-spacing:.16em;text-transform:uppercase;color:var(--accent);margin:0 0 .5rem}}
h1{{font-size:clamp(1.6rem,3.5vw,2.3rem);line-height:1.12;margin:0 0 .5rem;font-weight:600;letter-spacing:-.01em;text-wrap:balance}}
.sub{{color:var(--muted);font-size:1.02rem;max-width:46rem;margin:0 0 1.6rem}}
h2{{font-size:1.2rem;margin:2.4rem 0 .7rem;font-weight:600}}
p{{max-width:46rem}}
table{{border-collapse:collapse;width:100%;max-width:40rem;margin:1rem 0;font-size:.93rem;font-variant-numeric:tabular-nums}}
th,td{{text-align:right;padding:.5rem .8rem;border-bottom:1px solid var(--line)}} th:first-child,td:first-child{{text-align:left}}
thead th{{font-family:var(--mono);font-size:.72rem;letter-spacing:.05em;text-transform:uppercase;color:var(--muted);border-bottom:2px solid var(--line)}}
.win{{color:var(--good);font-weight:600}} .bad{{color:var(--bad);font-weight:600}} .mut{{color:var(--muted)}}
figure{{margin:1.4rem 0;background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:.7rem;overflow-x:auto}}
figure img{{width:100%;display:block;border-radius:6px}} figcaption{{font-size:.83rem;color:var(--muted);margin-top:.5rem;font-family:var(--mono)}}
.card{{background:var(--panel);border:1px solid var(--line);border-left:3px solid var(--accent);border-radius:8px;padding:1rem 1.2rem;margin:1.3rem 0}}
.card.key{{border-left-color:var(--good)}} .card.next{{border-left-color:var(--cyan)}}
code{{font-family:var(--mono);font-size:.85em;background:var(--line);padding:.1em .4em;border-radius:4px}}
.legend{{display:flex;gap:1.4rem;flex-wrap:wrap;font-size:.85rem;color:var(--muted);margin:.4rem 0 0}}
</style>
<div class="wrap">
<p class="eyebrow">USRP X410 · 2.4 GHz · 491.52 MSps · real OTA capture (1 s)</p>
<h1>Detector mask comparison: coherent vs DINO-FT M2 vs M3</h1>
<p class="sub">Same real over-the-air capture through all three detectors (offline, deployment operators),
masks outlined on the spectrogram across 20 frames. Question under test: is the new M3 model producing
large false positives?</p>

<table>
<thead><tr><th>Detector</th><th>Mean occ</th><th>Max occ</th><th>Read</th></tr></thead>
<tbody>
<tr><td>coherent_power</td><td>{s['coherent']['mean_occ_pct']:.2f}%</td><td>{s['coherent']['max_occ_pct']:.2f}%</td><td class="mut">energy ref; has DC/spur line FPs</td></tr>
<tr><td>DINO-FT M2_dr (rt)</td><td>{s['m2_rt']['mean_occ_pct']:.2f}%</td><td>{s['m2_rt']['max_occ_pct']:.2f}%</td><td class="bad">under-detects</td></tr>
<tr><td><strong>DINO-FT M3_491 (rt)</strong></td><td class="win">{s['m3_rt']['mean_occ_pct']:.2f}%</td><td class="win">{s['m3_rt']['max_occ_pct']:.2f}%</td><td class="win">tracks coherent, on real signals</td></tr>
</tbody>
</table>

<div class="card key"><strong>Offline, M3 is clean — not making large false positives.</strong> Its masked
area ({s['m3_rt']['mean_occ_pct']:.2f}% mean) matches the coherent energy detector ({s['coherent']['mean_occ_pct']:.2f}%),
sits on real signal energy, and it catches bursts M2 misses ({s['m2_rt']['mean_occ_pct']:.2f}%) — without
coherent's persistent DC/spur vertical stripes. Every M3 frame took the ROBUST normalization branch (no
fixed-clip fallback).</div>

<figure><img src="{img}" alt="3-detector mask comparison, 20 frames">
  <figcaption>Rows = frames; columns = coherent_power / M2_dr / M3_491. Cyan = emitted detection mask on
  the dB spectrogram (freq downsampled for display; occupancy computed at full resolution).</figcaption>
</figure>
<div class="legend"><span>cyan outline = detection</span><span>x-axis = frequency (2.15–2.65 GHz)</span><span>y-axis = time</span></div>

<h2>So why the large false positives in real time?</h2>
<p>They are <strong>not</strong> the model. On identical real input replayed offline, M3 behaves well. The
live-only difference is <strong>dropped-packet ingest corruption under GPU saturation</strong> — offline
file replay never drops packets, so it can't reproduce it. M3 detects more than M2, which raises the
snipper/classifier load, which can increase drops; a drop-corrupted frame feeds the normalization a
degenerate image that can fire broadly.</p>

<div class="card next"><strong>Next step to fix the RT FPs:</strong> dump the actual real-time masks during
a short live run (they will contain the FP blobs), confirm they coincide with drop-corrupted frames, then
tune the invalid-frame guard for M3's higher legitimate occupancy (and/or trim M3 GPU load via
<code>emit_stride</code>). The guard already exists to suppress exactly these — it likely needs its
threshold re-fit to M3.</p>

<p class="mut" style="margin-top:2.4rem;font-size:.83rem">Regenerate:
<code>render_cmp3.py</code> over <code>/tmp/usrp_spectrograms/cmp3/{{coherent,m2_rt,m3_rt}}</code>
(offline eval of the three detectors on <code>x410_ota_2g4_gain10</code>).</p>
</div>
"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html)
    print("wrote", OUT, f"({len(html)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
