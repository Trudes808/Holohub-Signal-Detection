#!/usr/bin/env python3
"""Compare coherent_power between two sweep results folders (e.g. the baseline snip
vs the 75 kHz + 1 ms gate) and plot the trade-off it buys.

The gate's purpose is data reduction, so the question is what the reduced coverage
costs in BER. This separates the two effects the same way the main figures do:

  detect rate / snippets kept   -> what the gate threw away
  snip-fidelity BER (matched)   -> whether what it KEPT decodes any better
  overall BER                   -> the net, with unsaved signals charged 100%

Usage:  python compare_variants.py BASE_DIR VARIANT_DIR [VARIANT_DIR ...] \
            [--labels "label1,label2,..."]
        First dir is the baseline; the rest are variants plotted against it.
"""
from pathlib import Path
import csv, sys, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

HERE = Path(__file__).resolve().parent
pos = []
i = 1
while i < len(sys.argv):
    a = sys.argv[i]
    if a.startswith("--"):
        i += 2                                   # skip the flag and its value
        continue
    pos.append(a); i += 1
if len(pos) < 2:
    sys.exit(__doc__)
BASE = Path(pos[0]).resolve()
VARS = [Path(p).resolve() for p in pos[1:]]
if "--labels" in sys.argv:
    LABELS = [s.strip() for s in sys.argv[sys.argv.index("--labels") + 1].split(",")]
elif "--label" in sys.argv:                      # back-compat with the single-variant form
    LABELS = [sys.argv[sys.argv.index("--label") + 1]]
else:
    LABELS = [v.name for v in VARS]
if len(LABELS) < len(VARS):
    LABELS += [v.name for v in VARS[len(LABELS):]]
elif len(LABELS) > len(VARS):
    # more labels than variants: almost always a comma INSIDE a label (--labels is
    # comma-separated), which would otherwise misalign every series. Fail loudly.
    sys.exit(f"--labels gave {len(LABELS)} labels for {len(VARS)} variant dirs "
             f"({LABELS}). Commas separate labels, so avoid commas within a label.")
DET = "coherent_power"
LEVELS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
SNR0 = 54.0
C_BASE, C_GT = "#1f77b4", "#4d4b47"
# distinct hues for variants (kept clear of the baseline blue and the grey reference)
C_VARS = ["#4a3aa7", "#1baf7a", "#eb6834", "#9467bd"]
MK_VARS = ["s", "D", "v", "^"]
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


SB = snippet_counts(BASE)
SVS = [snippet_counts(v) for v in VARS]
data = []                                        # (L, base_stats, [variant_stats...])
for L in LEVELS:
    a = stats(BASE, L)
    if not a:
        continue
    cs = [stats(v, L) for v in VARS]
    data.append((L, a, cs))
if not data:
    sys.exit(f"no coherent levels found under {BASE}")

names = ["baseline"] + LABELS
print(f"coherent_power snip-gate comparison")
for nm, d in zip(names, [BASE] + VARS):
    print(f"    {nm:<26} {d.name}")
HDR = "  " + f"{'SNR':>5} |" + "".join(f" {nm[:12]:>12}" for nm in names)


def block(title, cell):
    """print one table: a row per level, a column per variant. `cell(stats)` -> str."""
    print(f"\n  {title}")
    print(HDR)
    for L, a, cs in data:
        row = f"  {SNR0 - L:5.0f} |"
        for st in [a] + cs:
            row += f" {(cell(st) if st else '-'):>12}"
        print(row)


def snips_block():
    # COUNT only -- deliberately not called a volume proxy. The 75 kHz per-row mask
    # pre-filter de-fuses components (it strips the spur rows that 4-connected separate
    # signals into one giant box), so a gated variant can emit MORE but much SMALLER
    # boxes than the baseline. Comparing counts across variants says nothing about
    # stored bytes; use snip_eval's GB/hr metrics for that.
    print("\n  snippet COUNT per level (not a volume proxy -- box sizes differ per variant)")
    print(HDR)
    for L, _, _ in data:
        row = f"  {SNR0 - L:5.0f} |"
        for s in [SB] + SVS:
            v = s.get(L)                          # 0 is a real value -- test None, not truthiness
            row += f" {(str(v) if v is not None else '-'):>12}"
        print(row)


snips_block()
block("coverage (% of decodable signals saved)", lambda s: f"{s['cov']:.1f}%")
block("overall BER % (unsaved signals count as 100%)", lambda s: f"{s['overall']:.2f}%")
block("snip excess, points above genie on the SAME signals (fidelity of what was kept)",
      lambda s: f"{s['matched'] - s['genie']:+.2f}")

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

def series(key, idx):
    """values of `key` for variant idx (-1 = baseline), None-safe -> (xs, ys)"""
    xs, ys = [], []
    for L, a, cs in data:
        st = a if idx < 0 else cs[idx]
        if st is None:
            continue
        xs.append(SNR0 - L)
        ys.append(st[key] if key != "excess" else st["matched"] - st["genie"])
    return xs, ys


def draw(ax, key, ylabel, title):
    x, y = series(key, -1)
    ax.plot(x, y, "-o", color=C_BASE, lw=2.0, ms=8, label="baseline snip", zorder=4)
    for j, lab in enumerate(LABELS):
        x, y = series(key, j)
        ax.plot(x, y, ls="-", marker=MK_VARS[j % len(MK_VARS)], color=C_VARS[j % len(C_VARS)],
                lw=2.0, ms=7.5, label=lab, zorder=4)
    tidy(ax, ylabel)
    ax.set_title(title, pad=8)


# 1. coverage -- what the gate threw away
ax = axes[0]
ax.axhline(100, color=C_GT, ls=(0, (6, 3)), lw=2.0, label="perfect detection", zorder=3)
draw(ax, "cov", "signals saved (% of decodable)", "What the gate discards")
ax.yaxis.set_major_formatter(pct); ax.set_ylim(-3, 108)
ax.legend(loc="lower left", fontsize=9.5)

# 2. snip fidelity -- does what it KEPT decode any better?
ax = axes[1]
ax.axhline(0, color=C_GT, ls=(0, (6, 3)), lw=2.0, label="channel only (reference)", zorder=3)
draw(ax, "excess", "excess BER from snipping (points)", "Fidelity of what it kept")
ax.legend(loc="best", fontsize=9.5)

# 3. net overall BER
ax = axes[2]
gx, gy = series("genie", -1)
ax.plot(gx, gy, ls=(0, (6, 3)), color=C_GT, lw=2.0, label="channel only", zorder=3)
draw(ax, "overall", "overall BER %", "Net effect (unsaved = 100%)")
ax.yaxis.set_major_formatter(pct); ax.set_ylim(0, 105)
ax.legend(loc="lower right", fontsize=9.5)

fig.suptitle("Coherent Power snip-gate trade-off", y=1.04, fontsize=13.5, color=INK)
out = VARS[-1] / "fig_gate_tradeoff.png"
fig.savefig(out); plt.close(fig)
print("\nwrote", out)
