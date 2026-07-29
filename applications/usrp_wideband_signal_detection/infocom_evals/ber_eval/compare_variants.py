#!/usr/bin/env python3
"""Compare coherent_power between two sweep results folders (e.g. the baseline snip
vs the 75 kHz + 1 ms gate) and plot the trade-off it buys.

The gate's purpose is data reduction, so the question is what the reduced coverage
costs in BER. This separates the two effects the same way the main figures do:

  detect rate / snippets kept   -> what the gate threw away
  snip-fidelity BER (matched)   -> whether what it KEPT decodes any better
  overall BER                   -> the net, with unsaved signals charged 100%

Usage:  python compare_variants.py [baseline_dir] [variant_dir] [--label "..."]
"""
from pathlib import Path
import csv, sys, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

HERE = Path(__file__).resolve().parent
pos = [a for a in sys.argv[1:] if not a.startswith("--")]
BASE = Path(pos[0]).resolve() if len(pos) > 0 else HERE / "results"
VAR = Path(pos[1]).resolve() if len(pos) > 1 else HERE / "results_coh_75k_1ms"
LABEL = sys.argv[sys.argv.index("--label") + 1] if "--label" in sys.argv else "75 kHz + 1 ms gate"
DET = "coherent_power"
LEVELS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
SNR0 = 54.0
C_BASE, C_VAR, C_GT = "#1f77b4", "#4a3aa7", "#4d4b47"
INK, INK2, GRID = "#1a1a1a", "#4a4a4a", "#d8d8d4"
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 150, "savefig.bbox": "tight", "font.size": 11,
    "axes.labelsize": 12, "axes.titlesize": 12.5, "axes.edgecolor": "#9a9a95",
    "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.linewidth": 0.9, "legend.frameon": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def _num(x, k):
    v = x.get(k, "")
    return None if v in ("", "NaN") else float(v)


def rows(d, det, L):
    f = Path(d) / f"ber_{det}_attenuation_dB_{L}.csv"
    if not f.exists():
        return None
    with open(f) as fh:
        return {(x["variation"], x["sample_start"]): x for x in csv.DictReader(fh)}


def stats(d, L):
    """overall BER %, matched-subset (det, genie) BER %, detect rate % for coherent in dir d"""
    r = rows(d, DET, L)
    g = rows(d, "ground_truth", L)
    if not r or not g:
        return None
    e = b = de = db = ge = gb = 0.0
    nsig = nmiss = 0
    for k, v in r.items():
        nb, be = _num(v, "numBits"), _num(v, "bitErrors")
        if nb is None or be is None:
            continue
        e += be; b += nb; nsig += 1
        st = v["status"].split(":")[0]
        if st == "miss":
            nmiss += 1
        if st != "decoded":
            continue
        gg = g.get(k)
        if gg is None or gg["status"] != "decoded":
            continue
        gbe, gnb = _num(gg, "bitErrors"), _num(gg, "numBits")
        if gbe is None or gnb is None:
            continue
        de += be; db += nb; ge += gbe; gb += gnb
    if not b:
        return None
    return dict(overall=100 * e / b,
                matched=100 * de / db if db else float("nan"),
                genie=100 * ge / gb if gb else float("nan"),
                cov=100 * (1 - nmiss / max(1, nsig)))


def snippet_counts(d):
    """snippets generated per level, scraped from the COHERENT sweep logs only.
    (Globbing sweep_*.log would also match the dino/GT logs and silently overwrite
    coherent's counts with theirs -- they share the same message format.)"""
    out = {}
    for lg in sorted(Path(d).glob("sweep_coherent*.log")):
        for line in lg.read_text(errors="ignore").splitlines():
            m = re.search(r"attenuation_dB_(\d+): (\d+) snippets generated", line)
            if m:
                out[int(m.group(1))] = int(m.group(2))
    return out


SB, SV = snippet_counts(BASE), snippet_counts(VAR)
data = []
for L in LEVELS:
    a, c = stats(BASE, L), stats(VAR, L)
    if a and c:
        data.append((L, a, c))
if not data:
    sys.exit(f"no overlapping coherent levels between {BASE} and {VAR}")

print(f"coherent_power: baseline ({BASE.name}) vs {LABEL} ({VAR.name})")
print(f"  {'SNR':>5} | {'snips base':>10} {'snips var':>9} {'kept':>6} | "
      f"{'cov base':>8} {'cov var':>7} | {'BER base':>8} {'BER var':>7} | "
      f"{'snipΔ base':>10} {'snipΔ var':>10}")
for L, a, c in data:
    sb, sv = SB.get(L), SV.get(L)
    # NB: 0 is a real value here (a gate can legitimately save nothing) -- test for
    # None explicitly rather than truthiness, or zero renders as "missing".
    keep = f"{100*sv/sb:5.1f}%" if (sb not in (None, 0) and sv is not None) else "    -"
    print(f"  {SNR0-L:5.0f} | {sb if sb is not None else '-':>10} "
          f"{sv if sv is not None else '-':>9} {keep:>6} | "
          f"{a['cov']:7.1f}% {c['cov']:6.1f}% | {a['overall']:7.2f}% {c['overall']:6.2f}% | "
          f"{a['matched']-a['genie']:+9.2f} {c['matched']-c['genie']:+9.2f}")

snr = [SNR0 - L for L, _, _ in data]


def tidy(ax, ylabel):
    ax.grid(True, color=GRID, lw=0.7, alpha=0.9)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.invert_xaxis()
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel(ylabel)


pct = FuncFormatter(lambda v, _: f"{v:.0f}%")
fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.0))

# 1. coverage -- what the gate threw away
ax = axes[0]
ax.axhline(100, color=C_GT, ls=(0, (6, 3)), lw=2.0, label="perfect detection")
ax.plot(snr, [a["cov"] for _, a, _ in data], "-o", color=C_BASE, lw=2.0, ms=8, label="baseline snip")
ax.plot(snr, [c["cov"] for _, _, c in data], "-s", color=C_VAR, lw=2.0, ms=7.5, label=LABEL)
tidy(ax, "signals saved (% of decodable)")
ax.yaxis.set_major_formatter(pct); ax.set_ylim(-3, 108)
ax.set_title("What the gate discards", pad=8); ax.legend(loc="lower left", fontsize=10)

# 2. snip fidelity -- does what it KEPT decode better?
ax = axes[1]
ax.axhline(0, color=C_GT, ls=(0, (6, 3)), lw=2.0, label="channel only (reference)")
ax.plot(snr, [a["matched"] - a["genie"] for _, a, _ in data], "-o", color=C_BASE, lw=2.0, ms=8,
        label="baseline snip")
ax.plot(snr, [c["matched"] - c["genie"] for _, _, c in data], "-s", color=C_VAR, lw=2.0, ms=7.5,
        label=LABEL)
tidy(ax, "excess BER from snipping (points)")
ax.set_title("Fidelity of what it kept", pad=8); ax.legend(loc="best", fontsize=10)

# 3. net overall BER
ax = axes[2]
ax.plot(snr, [100 * 0 + a["genie"] for _, a, _ in data], ls=(0, (6, 3)), color=C_GT, lw=2.0,
        label="channel only")
ax.plot(snr, [a["overall"] for _, a, _ in data], "-o", color=C_BASE, lw=2.0, ms=8, label="baseline snip")
ax.plot(snr, [c["overall"] for _, _, c in data], "-s", color=C_VAR, lw=2.0, ms=7.5, label=LABEL)
tidy(ax, "overall BER %")
ax.yaxis.set_major_formatter(pct); ax.set_ylim(0, 105)
ax.set_title("Net effect (unsaved = 100%)", pad=8); ax.legend(loc="lower right", fontsize=10)

fig.suptitle(f"Coherent Power snip gate trade-off — baseline vs {LABEL}", y=1.04,
             fontsize=13.5, color=INK)
out = Path(VAR) / "fig_gate_tradeoff.png"
fig.savefig(out); plt.close(fig)
print("\nwrote", out)
