"""Build the OTA stress-test page (HTML + JPEG panels) from run_ota_detections / compare_overlays outputs.

  PYTHONPATH=src .venv-ml/bin/python src/make_ota_page.py --root ~/Datasets/ARGUS/OOD_captures/OTA --stage <dir> \
      [--synth-eval eval_out/ota_synth_uav_M5S eval_out/ota_synth_uav_M6S eval_out/ota_synth_uav_M7S]
"""
from __future__ import annotations
import argparse, csv, html, json
from collections import defaultdict
from pathlib import Path
from PIL import Image


def jpg(src: Path, dst: Path, width=1400, q=86):
    im = Image.open(src).convert("RGB"); w, h = im.size
    im = im.resize((width, int(h * width / w)), Image.LANCZOS); dst.parent.mkdir(parents=True, exist_ok=True); im.save(dst, "JPEG", quality=q, optimize=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--stage", required=True)
    ap.add_argument("--synth-eval", nargs="*", default=[])
    a = ap.parse_args()
    root = Path(a.root).expanduser(); stage = Path(a.stage); (stage / "img").mkdir(parents=True, exist_ok=True)
    caps = []
    for d in sorted(root.glob("n320_*")):
        dj = d / "detections.json"
        if not dj.exists():
            continue
        det = json.loads(dj.read_text()); meta = json.loads(sorted(d.glob("*.sigmf-meta"))[0].read_text())["global"]
        pan = sorted((d / "panels").glob("f*.png"))
        if not pan:
            continue
        name = f"{d.name}_{pan[0].stem}.jpg"; jpg(pan[0], stage / "img" / name)
        caps.append((d.name, det, meta, name))
    names = list(caps[0][1]["models"].keys()) if caps else []
    # synthetic UAV: per-file region detection from the eval runs
    synth = defaultdict(dict)   # scene -> model -> (det01, det05, n)
    for ev in a.synth_eval:
        p = Path(ev) / "regions.csv"
        if not p.exists():
            continue
        model = Path(ev).name.split("_")[-1]
        agg = defaultdict(lambda: [0, 0, 0])
        for r in csv.DictReader(open(p)):
            g = agg[r["scene"]]; g[0] += int(r["det01"]); g[1] += int(r["det05"]); g[2] += 1
        for sc, v in agg.items():
            synth[sc][model] = v
    synth_panels = []
    for d in sorted(root.glob("n320_*/synth_uav")):
        for m in sorted(d.glob("*.sigmf-meta")):
            pd = d / f"panels_{m.stem}"
            pngs = sorted(pd.glob("*.png"))
            if not pngs:
                continue
            name = f"{m.stem}.jpg"; jpg(pngs[0], stage / "img" / name)
            gmeta = json.loads(m.read_text())["global"]
            synth_panels.append((m.stem, gmeta, name))
    E = html.escape
    def band(label):
        return label.split("_")[0] if label else ""
    sections = defaultdict(list)
    for name, det, meta, img in caps:
        sections[band(meta.get("argus:band_label", ""))].append((name, det, meta, img))
    order = ["fm", "hdtv", "cell700", "cell850", "ism900", "adsb", "quiet", "ism2g4", "wifi5g", "cell1900", "cell2100"]
    titles = {"fm": "FM broadcast (VERT900)", "hdtv": "UHF HDTV (ATSC)", "cell700": "700 MHz cellular", "cell850": "850 MHz cellular", "ism900": "900 MHz ISM",
              "adsb": "ADS-B / DME 1090 MHz", "quiet": "Quiet reference band", "ism2g4": "2.4 GHz ISM (VERT2450)", "wifi5g": "5 GHz WiFi", "cell1900": "PCS 1900 downlink", "cell2100": "AWS 2100 downlink"}
    body = []
    for b in order + [k for k in sections if k not in order]:
        if b not in sections:
            continue
        body.append(f"<section><h2>{E(titles.get(b, b))}</h2>")
        for name, det, meta, img in sections[b]:
            occ = "".join(f"<tr><th scope='row'>{E(n)}</th><td>{100*det['models'][n]['occupancy_mean']:.2f}%</td><td>{det['models'][n]['frames_with_detection']}/{len(det['frame_index'])}</td></tr>" for n in names)
            agr = " · ".join(f"{E(k)} {v:.2f}" for k, v in det["agreement_iou"].items())
            body.append(f"""
<figure class="panel">
  <figcaption>
    <h3>{E(meta.get('argus:band_label',''))}</h3>
    <p class="meta"><code>{E(name)}</code> · {det['rate_hz']/1e6:.2f} MSps · gain {meta.get('argus:gain_db','')} dB · {len(det['frame_index'])} of {det['frames_total']} frames scored · antenna {E(str(meta.get('argus:antenna','')))}</p>
    <table class="stats"><thead><tr><th></th><th>occupancy</th><th>frames with detections</th></tr></thead><tbody>{occ}</tbody></table>
    <p class="note">{E(meta.get('argus:expected_waveforms',''))}. Mask agreement (IoU): {agr}.</p>
  </figcaption>
  <img src="img/{E(img)}" alt="panel for {E(name)}" loading="lazy" width="1400" height="1600">
</figure>""")
        body.append("</section>")
    if synth_panels:
        body.append("<section><h2>Mixed real + synthetic UAV (2.4 / 5 GHz captures with an added OcuSync-class link)</h2><p class='lead'>The real recording is unchanged; a synthetic video link (OFDM 10 to 40 MHz or analog FM) at the stated level above the measured floor plus a hopping 1 to 2 MHz control link are added, and only those synthetic bursts carry labels. Detection = share of synthetic bursts whose box is at least 10 % (50 %) covered.</p>")
        for stem, gmeta, img in synth_panels:
            ov = gmeta.get("argus:synthetic_overlay", {})
            rows = "".join(f"<tr><th scope='row'>{E(m)}</th><td>{100*v[0]/max(v[2],1):.0f}%</td><td>{100*v[1]/max(v[2],1):.0f}%</td><td>{v[2]}</td></tr>" for m, v in sorted(synth.get(stem, {}).items()))
            body.append(f"""
<figure class="panel">
  <figcaption>
    <h3>{E(stem.split('__')[-1])}</h3>
    <p class="meta"><code>{E(stem)}</code> · video {ov.get('mode','')} {float(ov.get('video_bw_hz',0))/1e6:g} MHz at {ov.get('snr_psd_db','')} dB, centre offset {float(ov.get('fc_offset_hz',0))/1e6:+.1f} MHz · control {ov.get('ctrl_snr_psd_db','')} dB</p>
    <table class="stats"><thead><tr><th></th><th>det@0.1</th><th>det@0.5</th><th>bursts</th></tr></thead><tbody>{rows}</tbody></table>
  </figcaption>
  <img src="img/{E(img)}" alt="panel for {E(stem)}" loading="lazy" width="1400" height="1900">
</figure>""")
        body.append("</section>")
    page = f"""<title>Argus N320 OTA Stress Test</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* layout: one reading column; per-band sections of figure cards (caption + stat table above a full-width panel) */
:root {{ --bg:#f6f7f5; --fg:#1b1f1c; --muted:#5a6360; --line:#d6dbd7; --card:#ffffff; --code:#eceeea; --accent:#1f8a4c;
  --sans:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif; --mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#141816; --fg:#e6e9e6; --muted:#9aa49f; --line:#2b332e; --card:#1b201d; --code:#232927; --accent:#4cc47a; color-scheme:dark }} }}
:root[data-theme="dark"] {{ --bg:#141816; --fg:#e6e9e6; --muted:#9aa49f; --line:#2b332e; --card:#1b201d; --code:#232927; --accent:#4cc47a; color-scheme:dark }}
body {{ background:var(--bg); color:var(--fg); font-family:var(--sans); line-height:1.5; padding-left:max(16px, env(safe-area-inset-left, 0px)); padding-right:max(16px, env(safe-area-inset-right, 0px)); padding-block:24px 56px; }}
main {{ max-width:1180px; margin:0 auto; }}
h1 {{ font-size:clamp(1.6rem, 2.6vw, 2.2rem); font-weight:600; letter-spacing:-0.01em; margin:0 0 .25rem; text-wrap:balance; }}
h2 {{ font-size:1.25rem; font-weight:600; margin:2.5rem 0 .25rem; padding-top:1.5rem; border-top:1px solid var(--line); text-wrap:balance; }}
h3 {{ font-size:1.05rem; font-weight:600; margin:0 0 .15rem; text-wrap:balance; }}
p {{ max-width:70ch; }} .sub, .lead, .meta {{ color:var(--muted); }} .sub {{ margin:0 0 1rem; }} .meta {{ font-size:.88rem; margin:0; }}
code {{ font-family:var(--mono); font-size:.86em; background:var(--code); padding:.08em .35em; border-radius:3px; overflow-wrap:anywhere; }}
.howto {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(220px, 1fr)); gap:14px 24px; margin:1rem 0 0; padding:14px 16px; border:1px solid var(--line); border-radius:6px; background:var(--card); }}
.howto div {{ min-width:0; font-size:.93rem; }} .howto b {{ display:block; font-weight:600; margin-bottom:.15rem; }}
.panel {{ margin:1.5rem 0 0; padding:0; }}
.panel figcaption {{ display:grid; grid-template-columns:minmax(0, 1.4fr) minmax(0, 1fr); gap:10px 28px; align-items:start; margin-bottom:10px; }}
.panel figcaption > h3, .panel figcaption > .meta, .panel figcaption > .note {{ grid-column:1; }} .panel figcaption > .note {{ margin:0; font-size:.95rem; }}
.panel figcaption > .stats {{ grid-column:2; grid-row:1 / span 3; }}
table.stats {{ border-collapse:collapse; font-variant-numeric:tabular-nums; width:100%; font-size:.92rem; }}
table.stats th, table.stats td {{ padding:.3rem .5rem; border-bottom:1px solid var(--line); text-align:right; }}
table.stats th[scope=row] {{ text-align:left; font-weight:400; color:var(--muted); }} table.stats thead th {{ font-weight:500; color:var(--muted); }}
.panel img {{ display:block; width:100%; max-width:100%; height:auto; border:1px solid var(--line); border-radius:4px; background:var(--card); }}
@media (max-width: 720px) {{ .panel figcaption {{ grid-template-columns:1fr; }} .panel figcaption > .stats {{ grid-column:1; grid-row:auto; }} }}
.foot {{ margin-top:2.5rem; padding-top:1rem; border-top:1px solid var(--line); color:var(--muted); font-size:.92rem; }}
a {{ color:var(--accent); }} a:focus-visible {{ outline:2px solid var(--accent); outline-offset:2px; }}
</style>
<main>
<h1>Argus N320 OTA Stress Test</h1>
<p class="sub">Short over-the-air snippets recorded 3 November 2026 on a USRP N320 (VERT900 on RF1, VERT2450 on RF0) at several sample rates and centre offsets per band, run through the deployed M5-S and the two retrains M6-S and M7-S at threshold 0.6. Raw IQ, masks and panels live together under <code>~/Datasets/ARGUS/OOD_captures/OTA/&lt;capture&gt;/</code>.</p>
<div class="howto">
  <div><b>Rows in each panel</b>model input after the live front-end · M5-S mask · M6-S mask · M7-S mask · difference of the last two (white both, red only M6-S, green only M7-S).</div>
  <div><b>Numbers</b>occupancy is the share of the 512 × 1024 grid marked as signal, averaged over the scored frames; these recordings carry no labels except the synthetic UAV bursts in the last section.</div>
  <div><b>What to look for</b>whole carriers vs fragments, tight station lines vs smears between them, the receiver DC line at 0 MHz (the N320's DC offset; by convention every emission in the capture is flagged and spurs are filtered downstream, so marking it is correct), and behaviour at the band edges.</div>
</div>
{''.join(body)}
<div class="foot"><p>Capture tooling: <code>bash_scripts/capture_n320_sigmf.sh</code>, <code>capture_n320_matrix.sh</code>; detections: <code>dino_fine_tuning/src/run_ota_detections.py</code>; synthetic UAV overlays: <code>src/overlay_synthetic_uav.py</code>; experiment record: <code>infocom_evals/signal_detection_experiments/dino_ft_multirate_scenes/README.md</code>.</p></div>
</main>
"""
    (stage / "index.html").write_text(page)
    print("page written:", stage / "index.html", "| captures", len(caps), "| synth panels", len(synth_panels), "| images", len(list((stage / 'img').glob('*.jpg'))))


if __name__ == "__main__":
    main()
