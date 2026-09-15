#!/usr/bin/env python3
"""Self-contained report: RT DINO-FT 'misses the loud signal' was a viz overlay-color bug, not the masks."""
from __future__ import annotations
import base64
from pathlib import Path
HERE = Path(__file__).resolve().parent
RES = HERE / "results"
OUT = Path("/home/genesys-dgx1/.claude/jobs/323ea032/tmp/dino_ft_overlay_report.html")


def b64(p): return "data:image/png;base64," + base64.b64encode(Path(p).read_bytes()).decode()


def main():
    colors = b64(RES / "overlay_color_options.png")
    faithful = b64(RES / "faithful_overlay.png")
    html = f"""<title>Overlay Not Detector</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
:root {{ --bg:#f4f3f7; --panel:#fff; --ink:#191622; --muted:#5b5670; --line:#e2dfeb; --accent:#2fb46b;
  --bad:#c8442e; --mono:'IBM Plex Mono',ui-monospace,monospace; --sans:'IBM Plex Sans',system-ui,sans-serif; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme=light]) {{
  --bg:#0f1512; --panel:#18201b; --ink:#e9f4ec; --muted:#9db4a6; --line:#28322c; --accent:#5fe39a; --bad:#f0715a; }} }}
:root[data-theme=dark] {{ --bg:#0f1512; --panel:#18201b; --ink:#e9f4ec; --muted:#9db4a6; --line:#28322c; --accent:#5fe39a; --bad:#f0715a; }}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);line-height:1.6}}
.wrap{{max-width:62rem;margin:0 auto;padding:2.6rem 1.4rem 4rem}}
.eyebrow{{font-family:var(--mono);font-size:.72rem;letter-spacing:.16em;text-transform:uppercase;color:var(--accent);margin:0 0 .5rem}}
h1{{font-size:clamp(1.6rem,3.5vw,2.4rem);line-height:1.12;margin:0 0 .5rem;font-weight:600;letter-spacing:-.01em;text-wrap:balance}}
.sub{{color:var(--muted);font-size:1.05rem;max-width:46rem;margin:0 0 1.4rem}}
h2{{font-size:1.25rem;margin:2.4rem 0 .7rem;font-weight:600}}
p{{max-width:47rem}} code{{font-family:var(--mono);font-size:.85em;background:var(--line);padding:.1em .4em;border-radius:4px}}
table{{border-collapse:collapse;width:100%;max-width:44rem;margin:1rem 0;font-size:.92rem;font-variant-numeric:tabular-nums}}
th,td{{text-align:right;padding:.5rem .8rem;border-bottom:1px solid var(--line)}} th:first-child,td:first-child{{text-align:left}}
thead th{{font-family:var(--mono);font-size:.72rem;letter-spacing:.05em;text-transform:uppercase;color:var(--muted);border-bottom:2px solid var(--line)}}
figure{{margin:1.3rem 0;background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:.7rem;overflow-x:auto}}
figure img{{width:100%;display:block;border-radius:6px}} figcaption{{font-size:.83rem;color:var(--muted);margin-top:.5rem;font-family:var(--mono)}}
.card{{background:var(--panel);border:1px solid var(--line);border-left:3px solid var(--accent);border-radius:8px;padding:1rem 1.2rem;margin:1.3rem 0}}
.good{{color:var(--accent);font-weight:600}} .bad{{color:var(--bad);font-weight:600}}
</style>
<div class="wrap">
<p class="eyebrow">USRP X410 · 2.4 GHz · 491.52 MSps · DINO-FT M3 real-time</p>
<h1>The RT masks were right — the overlay color was wrong</h1>
<p class="sub">"DINO-FT misses the loud signal and puts detections in odd places" turned out to be a
<strong>visualization</strong> bug: the mask overlay color washed out on bright, high-power signals. The
detector, masks, timing, and frame-sync are all correct.</p>

<h2>1 · The masks are correct (offline, loopback, and real-time all agree)</h2>
<table>
<thead><tr><th>Check</th><th>Result</th></tr></thead>
<tbody>
<tr><td>Loud burst detected (per frame)</td><td class="good">96–100%</td></tr>
<tr><td>Loud-burst coverage vs coherent (pixels)</td><td class="good">20% vs 1%</td></tr>
<tr><td>RT vs offline occupancy</td><td>0.70% vs 0.67% (match)</td></tr>
<tr><td>Keeping up (inference vs budget)</td><td class="good">52 ms &lt; 85 ms</td></tr>
<tr><td>Dropped / skipped frames</td><td class="good">0 (partial_drops=0, no gaps)</td></tr>
<tr><td>Overlay frame-sync (MASKSYNC probe)</td><td class="good">lag = 0, matched every frame</td></tr>
</tbody>
</table>
<p>A temporary in-viz probe confirmed the overlay is matched to the exact displayed frame
(<code>curmatch=true, lag=0</code>) even under 2× oversaturation — so it is not drift, dropped frames,
or a detector fault.</p>

<h2>2 · Root cause: the overlay color matched the bright signal</h2>
<p>The mask overlay color blended toward <strong>pale-yellow</strong> at high mask values — nearly the
same color as a bright/yellow high-power signal on the spectrogram. Alpha-blended over those bright
pixels, the mask <em>disappeared exactly on the loudest signals</em>, while staying visible on the dark
noise floor. That reads as "missing the loud signal, detections in odd places." Reproduced by replaying
the exact overlay math offline:</p>
<figure><img src="{faithful}" alt="faithful overlay washout">
  <figcaption>Same loud-burst frame, without (top) and with (bottom) the original overlay. The mask IS on
  the bright burst but is invisible (pale-on-bright); small detections on dark areas stand out.</figcaption>
</figure>

<h2>3 · Fix: constant high-contrast color</h2>
<p>Make the mask a constant lime-green (visible on both the dark-blue noise floor and bright signals;
confidence stays encoded via opacity), and raise the low-value alpha floor so faint detections aren't
nearly transparent.</p>
<figure><img src="{colors}" alt="overlay color options">
  <figcaption>Overlay color options on the loud-burst frame. Top = current (pale-yellow, invisible on the
  burst). Lime / spring-green / cyan all make the loud-burst mask unmistakable. Shipped: lime-green.</figcaption>
</figure>
<div class="card"><strong>Shipped</strong> (commit <code>01f08d3b</code>, rebuilt): <code>mask_overlay_color</code>
→ constant lime-green; alpha floor 0.18→0.35 (both overlay + ROI-ring paths). Applies to coherent and
DINO overlays alike. <strong>Verify on the dashboard:</strong> the loud burst should now be clearly
green-masked.</div>

<h2>4 · One honest caveat</h2>
<p>The radio-only misregistration you first described could not be reproduced in any controlled run
(offline, real-time loopback, 2× loopback were all clean and frame-synced). This fix addresses the
confirmed cause — the mask washing out on bright signals. If, after this, the live radio still shows
placement drift under heavy load, that would be a separate ingest-timing issue; re-add the MASKSYNC probe
and read <code>curmatch/lag</code> on the radio to confirm.</p>
</div>
"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html)
    print("wrote", OUT, f"({len(html)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
