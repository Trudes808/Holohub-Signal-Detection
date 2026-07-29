#!/usr/bin/env python3
"""BER figures for the attenuation sweep.

Two groups of output (all into results/):

A. THE CLAIM figures -- "snipping a signal is minimally destructive compared to
   what the channel already did":
     fig_snip_vs_channel.png        per detector: channel-only vs snipped, same signals
     fig_snip_excess.png            excess BER added by snipping (percentage points)
     fig_snip_vs_channel_byclass.png   the same claim, per modulation class

   These use a MATCHED SUBSET: a detector is compared against the genie evaluated
   on *exactly the signals that detector saved*. That isolates "what the snip did"
   from "what the detector missed" -- without it, a detector that misses the weak
   signals looks artificially good (survivorship bias), and one that saves a lot
   looks bad for reasons that have nothing to do with snip fidelity.

B. The full-sweep reference figures (restyled, BER in %):
     ber_sweep_overall.png          overall BER vs SNR (includes misses at 100%)
     ber_sweep_byclass.png          per-class overall BER vs SNR

Detector colors/markers follow the repo-wide convention in
signal_detection_experiments/plot_eval_results.py (DETECTOR_STYLE) so these
figures match every other eval; ground truth is the neutral grey baseline.

Usage:  python plot_ber_figs.py [results_dir] [--title-suffix "..."]
        (default results_dir = ./results; a variant sweep such as the 75 kHz + 1 ms
         coherent snipper gate lives in its own folder and plots into it)
"""
from pathlib import Path
import csv, collections, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

HERE = Path(__file__).resolve().parent
_args = [a for a in sys.argv[1:] if not a.startswith("--")]
RES = Path(_args[0]).expanduser().resolve() if _args else HERE / "results"
SUFFIX = ""
if "--title-suffix" in sys.argv:
    SUFFIX = sys.argv[sys.argv.index("--title-suffix") + 1]
if not RES.is_dir():
    sys.exit(f"no such results dir: {RES}")
LEVELS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
SNR0 = 54.0                                   # SNR ~= 54 - attenuation (snr_calibration.json)
CLASSES = ["BPSK", "QPSK", "16QAM", "OFDM", "5G_Downlink", "802_11ax", "Bluetooth"]
CLASS_LABEL = {"5G_Downlink": "5G downlink", "802_11ax": "802.11ax"}

# ---- repo-wide detector style (DETECTOR_STYLE in plot_eval_results.py) ----------------
GT = "ground_truth"
STYLE = {
    GT:                  dict(color="#4d4b47", marker=None, ls=(0, (6, 3)), label="channel only (ground truth)"),
    "coherent_power":    dict(color="#1f77b4", marker="o",  ls="-", label="Coherent Power"),
    "finetuned_dino_m2": dict(color="#8c564b", marker="P",  ls="-", label="DINO FT"),
}
DETS = ["coherent_power", "finetuned_dino_m2"]
INK, INK2, GRID = "#1a1a1a", "#4a4a4a", "#d8d8d4"

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 12.5,
    "axes.edgecolor": "#9a9a95", "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.9,
    "legend.frameon": False, "figure.facecolor": "white", "axes.facecolor": "white",
})


# ---------------------------------------------------------------- data ---------------- #
def load(det, L):
    """per-signal rows keyed by signal identity -> dict"""
    f = RES / f"ber_{det}_attenuation_dB_{L}.csv"
    if not f.exists():
        return None
    out = {}
    with open(f) as fh:
        for x in csv.DictReader(fh):
            out[(x["variation"], x["sample_start"])] = x
    return out


def _num(x, k):
    v = x.get(k, "")
    return None if v in ("", "NaN") else float(v)


def overall_ber(rows):
    """bit-weighted BER over every scored signal (misses count as 1.0) -> as in the harness"""
    e = b = 0.0
    for x in rows.values():
        nb, be = _num(x, "numBits"), _num(x, "bitErrors")
        if nb is None or be is None:
            continue
        e += be; b += nb
    return e / b if b else float("nan")


def matched(det_rows, gt_rows, cls=None):
    """BER for (detector, genie) over the SAME signals: those the detector decoded
    and the genie also decoded. Returns (det_ber, gt_ber, n_signals)."""
    de = db = ge = gb = 0.0
    n = 0
    for k, v in det_rows.items():
        if v["status"] != "decoded":
            continue
        if cls is not None and v["class"] != cls:
            continue
        g = gt_rows.get(k)
        if g is None or g["status"] != "decoded":
            continue
        dbe, dnb = _num(v, "bitErrors"), _num(v, "numBits")
        gbe, gnb = _num(g, "bitErrors"), _num(g, "numBits")
        if None in (dbe, dnb, gbe, gnb):
            continue
        de += dbe; db += dnb; ge += gbe; gb += gnb; n += 1
    if not db or not gb:
        return None, None, 0
    return 100.0 * de / db, 100.0 * ge / gb, n     # percent


print("loading per-signal results ...")
DATA = {}                                   # (det, L) -> rows
for L in LEVELS:
    for det in [GT] + DETS:
        r = load(det, L)
        if r:
            DATA[(det, L)] = r
have = sorted({L for (_, L) in DATA})
print(f"  levels: {have}")

# matched-subset series (the claim) + overall series (reference)
CLAIM = {d: dict(snr=[], det=[], gt=[], n=[]) for d in DETS}
OVER = {d: dict(snr=[], ber=[]) for d in [GT] + DETS}
for L in have:
    g = DATA.get((GT, L))
    for det in [GT] + DETS:
        r = DATA.get((det, L))
        if r:
            OVER[det]["snr"].append(SNR0 - L)
            OVER[det]["ber"].append(100.0 * overall_ber(r))
    if not g:
        continue
    for det in DETS:
        r = DATA.get((det, L))
        if not r:
            continue
        db, gb, n = matched(r, g)
        if db is None:
            continue
        CLAIM[det]["snr"].append(SNR0 - L)
        CLAIM[det]["det"].append(db); CLAIM[det]["gt"].append(gb); CLAIM[det]["n"].append(n)


def save(fig, out, stamp=True):
    """Save, stamping the variant label (if any) so a variant figure is never mistaken
    for the baseline one. stamp=False for figures specified to carry no sub-label
    (the v2 presentation figure), where the caption belongs in the surrounding doc."""
    if SUFFIX and stamp:
        fig.text(0.995, 0.002, SUFFIX, ha="right", va="bottom", fontsize=9, color=INK2)
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


def pctfmt(decimals=0):
    return FuncFormatter(lambda v, _: f"{v:.{decimals}f}%")


LOG_TICKS = [1, 2, 5, 10, 20, 50, 100]        # readable decade+midpoint ticks on a log % axis


def log_pct_axis(ax, lo=None, ticks=None):
    """Label a log BER axis in percent (matplotlib otherwise labels only decades).
    Small multiples want sparse decade ticks; a single large axis can afford midpoints."""
    ticks = LOG_TICKS if ticks is None else ticks
    ax.set_yscale("log")
    ax.set_yticks(ticks)
    ax.set_yticklabels([(f"{t:g}%" if t >= 1 else f"{t:g}%") for t in ticks])
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    ax.set_ylim(top=140, bottom=lo)


def tidy(ax, logy=False, xlabel="SNR (dB)", ylabel="BER %"):
    ax.grid(True, which="major", color=GRID, lw=0.7, alpha=0.9)
    if logy:
        ax.grid(True, which="minor", color=GRID, lw=0.4, alpha=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.invert_xaxis()                        # high SNR (low attenuation) on the left
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)


# ============================== A1. the claim, per detector ========================== #
fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0), sharey=True)
for ax, det in zip(axes, DETS):
    c = CLAIM[det]
    if not c["snr"]:
        continue
    s = STYLE[det]; sg = STYLE[GT]
    ax.plot(c["snr"], c["gt"], color=sg["color"], ls=sg["ls"], lw=2.0, zorder=3,
            label="channel only (no snip)")
    ax.plot(c["snr"], c["det"], color=s["color"], ls=s["ls"], lw=2.0, marker=s["marker"],
            ms=8, mew=1.6, zorder=4, label="after snipping")
    ax.fill_between(c["snr"], c["gt"], c["det"], color=s["color"], alpha=0.16,
                    lw=0, zorder=2, label="cost of snipping")
    # headline callout: the channel's swing vs the worst snip penalty
    dz = [d - g for d, g in zip(c["det"], c["gt"])]
    ax.annotate(f"channel: {min(c['gt']):.1f}% → {max(c['gt']):.0f}%   (+{max(c['gt'])-min(c['gt']):.0f} pts)\n"
                f"snipping: at most +{max(dz):.1f} pts, median +{sorted(dz)[len(dz)//2]:.1f}",
                xy=(0.97, 0.06), xycoords="axes fraction", ha="right", va="bottom",
                fontsize=10.5, color=INK2,
                bbox=dict(boxstyle="round,pad=0.5", fc="#f7f7f5", ec="#d8d8d4", lw=0.8))
    ax.set_title(s["label"], color=INK, pad=8)
    tidy(ax, ylabel="BER %" if det == DETS[0] else None)
    ax.yaxis.set_major_formatter(pctfmt())
    ax.legend(loc="upper left", fontsize=10)
axes[0].set_ylim(bottom=0)
fig.suptitle("Snipping is minimally destructive compared to the channel\n"
             "same signals, decoded from the detector's saved snippet vs from the full capture",
             y=1.06, fontsize=13.5, color=INK)
out = RES / "fig_snip_vs_channel.png"
save(fig, out)

# ============================== A2. excess BER from snipping ========================= #
fig, ax = plt.subplots(figsize=(8.6, 5.0))
ax.axhline(0, color=STYLE[GT]["color"], ls=STYLE[GT]["ls"], lw=2.0, zorder=3,
           label="channel only (reference)")
for det in DETS:
    c = CLAIM[det]
    if not c["snr"]:
        continue
    s = STYLE[det]
    dz = [d - g for d, g in zip(c["det"], c["gt"])]
    ax.plot(c["snr"], dz, color=s["color"], ls=s["ls"], lw=2.0, marker=s["marker"],
            ms=8, mew=1.6, zorder=4, label=s["label"])
tidy(ax, ylabel="excess BER from snipping (percentage points)")
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+.1f}"))   # one consistent format
ax.set_title("Cost of snipping, isolated\n(BER after snipping − BER of the same signals from the full capture)",
             pad=10)
ax.legend(loc="best", fontsize=10)
out = RES / "fig_snip_excess.png"
save(fig, out)

# ============================== A3. the claim, per class ============================= #
BYCLS = {d: {c: dict(snr=[], det=[], gt=[]) for c in CLASSES} for d in DETS}
for L in have:
    g = DATA.get((GT, L))
    if not g:
        continue
    for det in DETS:
        r = DATA.get((det, L))
        if not r:
            continue
        for c in CLASSES:
            db, gb, n = matched(r, g, cls=c)
            if db is None:
                continue
            BYCLS[det][c]["snr"].append(SNR0 - L)
            BYCLS[det][c]["det"].append(db); BYCLS[det][c]["gt"].append(gb)

fig, axes = plt.subplots(2, 4, figsize=(17.0, 8.2), sharex=True)
for i, c in enumerate(CLASSES):
    ax = axes.flat[i]
    # one grey reference per panel: genie on the union subset (use coherent's, they coincide)
    for det in DETS:
        d = BYCLS[det][c]
        if not d["snr"]:
            continue
        s = STYLE[det]
        ax.plot(d["snr"], d["gt"], color=STYLE[GT]["color"], ls=STYLE[GT]["ls"], lw=1.6,
                zorder=3, alpha=0.85 if det == DETS[0] else 0.45)
        ax.plot(d["snr"], d["det"], color=s["color"], ls=s["ls"], lw=1.9, marker=s["marker"],
                ms=6.5, mew=1.4, zorder=4)
    ax.set_title(CLASS_LABEL.get(c, c), pad=6)
    tidy(ax, xlabel=None, ylabel="BER %" if i % 4 == 0 else None)
    ax.yaxis.set_major_formatter(pctfmt())
    ax.set_ylim(bottom=0)
for ax in axes.flat[len(CLASSES):]:
    ax.axis("off")
    h = [plt.Line2D([], [], color=STYLE[GT]["color"], ls=STYLE[GT]["ls"], lw=2.0,
                    label="channel only (no snip)")]
    h += [plt.Line2D([], [], color=STYLE[d]["color"], ls="-", lw=2.0, marker=STYLE[d]["marker"],
                     ms=8, mew=1.6, label=f"after snipping · {STYLE[d]['label']}") for d in DETS]
    lg = ax.legend(handles=h, loc="center", fontsize=11)
    # each detector is compared against the genie on ITS OWN matched subset, so a panel
    # carries one grey reference per detector; they separate only where the two
    # detectors saved noticeably different sets of signals.
    ax.text(0.5, 0.24, "one grey reference per detector\n(genie on that detector's own signals)",
            transform=ax.transAxes, ha="center", va="top", fontsize=9.5, color=INK2)
for ax in axes[1]:
    ax.set_xlabel("SNR (dB)")
fig.suptitle("Snipping is minimally destructive, per modulation class\n"
             "same signals, decoded from the saved snippet vs from the full capture",
             y=1.02, fontsize=13.5, color=INK)
out = RES / "fig_snip_vs_channel_byclass.png"
save(fig, out)

# ============================== B1. overall sweep (restyled) ======================== #
fig, ax = plt.subplots(figsize=(9.0, 5.4))
for det in [GT] + DETS:
    o = OVER[det]
    if not o["snr"]:
        continue
    s = STYLE[det]
    ax.plot(o["snr"], o["ber"], color=s["color"], ls=s["ls"], lw=2.0,
            marker=s["marker"], ms=8, mew=1.6, zorder=4 if det != GT else 3,
            label=s["label"])
tidy(ax, logy=True)
log_pct_axis(ax, lo=1.4)
ax.set_title("Overall BER vs SNR\n(all signals; a signal the detector never saved counts as 100%)",
             pad=10)
# legend outside right (same convention as the other snip_eval figures) so it never
# sits on top of the curves
ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=10.5)
out = RES / "ber_sweep_overall.png"
save(fig, out)

# ============================== B1b. overall sweep, v2 (paper style) =============== #
# Presentation variant: SNR ascending left->right over a fixed -20..40 dB window,
# linear BER% axis capped at 100 with 0 at the bottom, full box outline, legend inside
# the (empty) lower-left corner, bare title. Deliberately NOT the log/inverted view.
fig, ax = plt.subplots(figsize=(9.0, 5.6))
for det in [GT] + DETS:
    o = OVER[det]
    if not o["snr"]:
        continue
    s = STYLE[det]
    pts = sorted(zip(o["snr"], o["ber"]))            # ascending SNR for a left->right line
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    ax.plot(xs, ys, color=s["color"], ls=s["ls"], lw=2.2, marker=s["marker"],
            ms=8, mew=1.6, zorder=4 if det != GT else 3, label=s["label"],
            clip_on=True)
ax.set_xlim(-20, 40)                                  # ascending: low SNR left, high SNR right
ax.set_ylim(0, 100)
ax.set_yticks(range(0, 101, 20))
ax.yaxis.set_major_formatter(pctfmt())
ax.grid(True, color=GRID, lw=0.7, alpha=0.9)
ax.set_axisbelow(True)
for sp in ax.spines.values():                         # full outline: top + right included
    sp.set_visible(True)
    sp.set_linewidth(0.9)
    sp.set_color("#9a9a95")
ax.set_xlabel("SNR (dB)")
ax.set_ylabel("BER %")
ax.set_title("Bit Error Rate (BER) vs. SNR", pad=10)  # no subtitle
ax.legend(loc="lower left", fontsize=10.5, framealpha=0.0)
out = RES / "ber_sweep_overall_v2.png"
save(fig, out, stamp=False)          # spec: bare title, no sub-label of any kind

# ============================== B2. per-class sweep (restyled) ===================== #
CLS_OVER = {d: {c: dict(snr=[], ber=[]) for c in CLASSES} for d in [GT] + DETS}
for L in have:
    for det in [GT] + DETS:
        f = RES / f"ber_{det}_attenuation_dB_{L}_byclass.csv"
        if not f.exists():
            continue
        with open(f) as fh:
            for x in csv.DictReader(fh):
                c = x["class"].strip('"')
                if c in CLS_OVER[det]:
                    CLS_OVER[det][c]["snr"].append(SNR0 - L)
                    CLS_OVER[det][c]["ber"].append(100.0 * float(x["BER"]))

fig, axes = plt.subplots(2, 4, figsize=(17.0, 8.2), sharex=True, sharey=True)
for i, c in enumerate(CLASSES):
    ax = axes.flat[i]
    for det in [GT] + DETS:
        d = CLS_OVER[det][c]
        if not d["snr"]:
            continue
        s = STYLE[det]
        ax.plot(d["snr"], d["ber"], color=s["color"], ls=s["ls"], lw=1.9,
                marker=s["marker"], ms=6.5, mew=1.4, zorder=4 if det != GT else 3)
    ax.set_title(CLASS_LABEL.get(c, c), pad=6)
    tidy(ax, logy=True, xlabel=None, ylabel="BER %" if i % 4 == 0 else None)
    log_pct_axis(ax, lo=0.07, ticks=[0.1, 1, 10, 100])   # sparse decades read better in a 2x4 grid
for ax in axes.flat[len(CLASSES):]:
    ax.axis("off")
    h = [plt.Line2D([], [], color=STYLE[d]["color"], ls=STYLE[d]["ls"], lw=2.0,
                    marker=STYLE[d]["marker"], ms=8, mew=1.6, label=STYLE[d]["label"])
         for d in [GT] + DETS]
    ax.legend(handles=h, loc="center", fontsize=11)
for ax in axes[1]:
    ax.set_xlabel("SNR (dB)")
fig.suptitle("Per-class overall BER vs SNR", y=1.02, fontsize=13.5, color=INK)
out = RES / "ber_sweep_byclass.png"
save(fig, out)

# ============================== C. the cost of MISSED detections ==================== #
# Additive error budget on a common denominator (every scored bit at that level):
#
#   detector overall BER  =  channel floor            (what perfect detection would give)
#                         +  snip excess              (saved signals decoded worse than genie)
#                         +  decode-fail excess       (saved, but the decoder threw)
#                         +  miss excess              (never saved: 1.0 minus what genie got)
#
# A missed signal is charged only the EXCESS over what the channel would have done to
# it anyway -- at deep noise the genie itself is at ~0.49, so a miss costs ~0.51, not 1.0.
# That keeps the three effects comparable instead of letting the 1.0 convention dominate.
def budget(det_rows, gt_rows):
    tot = ch = snip = miss = dfail = 0.0
    nsig = nmiss = 0
    for k, v in det_rows.items():
        nb, be = _num(v, "numBits"), _num(v, "bitErrors")
        if nb is None or be is None:
            continue                                  # 'insufficient' -> excluded for everyone
        g = gt_rows.get(k)
        gref = _num(g, "bitErrors") if g else None     # what the channel alone cost this signal
        if gref is None:
            gref = 0.0
        tot += nb; ch += gref; nsig += 1
        st = v["status"].split(":")[0]
        if st == "decoded":
            snip += be - gref
        elif st == "miss":
            miss += be - gref; nmiss += 1
        else:
            dfail += be - gref
    if not tot:
        return None
    f = 100.0 / tot
    return dict(ch=ch * f, snip=snip * f, dfail=dfail * f, miss=miss * f,
                overall=(ch + snip + dfail + miss) * f, nsig=nsig, nmiss=nmiss)


BUD = {d: dict(snr=[], ch=[], snip=[], dfail=[], miss=[], overall=[], cov=[]) for d in DETS}
for L in have:
    g = DATA.get((GT, L))
    if not g:
        continue
    for det in DETS:
        r = DATA.get((det, L))
        if not r:
            continue
        b = budget(r, g)
        if not b:
            continue
        for key in ("ch", "snip", "dfail", "miss", "overall"):
            BUD[det][key].append(b[key])
        BUD[det]["snr"].append(SNR0 - L)
        BUD[det]["cov"].append(100.0 * (1 - b["nmiss"] / max(1, b["nsig"])))

# ---- C1. detection coverage vs SNR --------------------------------------------------- #
fig, ax = plt.subplots(figsize=(8.8, 5.0))
for det in DETS:
    b = BUD[det]
    if not b["snr"]:
        continue
    s = STYLE[det]
    ax.plot(b["snr"], b["cov"], color=s["color"], ls=s["ls"], lw=2.0, marker=s["marker"],
            ms=8, mew=1.6, zorder=4, label=s["label"])
ax.axhline(100, color=STYLE[GT]["color"], ls=STYLE[GT]["ls"], lw=2.0, zorder=3,
           label="perfect detection (ground truth)")
tidy(ax, ylabel="signals saved (% of decodable signals)")
ax.yaxis.set_major_formatter(pctfmt())
ax.set_ylim(-3, 108)
ax.set_title("Detection coverage: what fraction of decodable signals each detector saved", pad=10)
ax.legend(loc="lower left", fontsize=10.5)
out = RES / "fig_miss_coverage.png"
save(fig, out)

# ---- C2. error budget: channel vs snipping vs missed detections ---------------------- #
fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.4), sharey=True)
for ax, det in zip(axes, DETS):
    b = BUD[det]
    if not b["snr"]:
        continue
    s = STYLE[det]
    x = b["snr"]
    y0 = b["ch"]
    y1 = [a + c for a, c in zip(y0, b["snip"])]
    y2 = [a + c for a, c in zip(y1, b["dfail"])]
    y3 = [a + c for a, c in zip(y2, b["miss"])]
    ax.fill_between(x, 0, y0, color=STYLE[GT]["color"], alpha=0.30, lw=0.8,
                    ec="white", zorder=2, label="channel (what perfect detection gives)")
    ax.fill_between(x, y0, y1, color=s["color"], alpha=0.85, lw=0.8, ec="white",
                    zorder=3, label="snipping")
    ax.fill_between(x, y1, y2, color="#b0aca4", alpha=0.9, lw=0.8, ec="white",
                    zorder=3, hatch="///", label="decoder failed on saved snippet")
    ax.fill_between(x, y2, y3, color=s["color"], alpha=0.30, lw=0.8, ec="white",
                    zorder=3, hatch="...", label="MISSED detections")
    ax.plot(x, y3, color=s["color"], lw=2.0, marker=s["marker"], ms=7, mew=1.5,
            zorder=5, label="overall BER")
    ax.set_title(s["label"], pad=8)
    tidy(ax, ylabel="BER %" if det == DETS[0] else None)
    ax.yaxis.set_major_formatter(pctfmt())
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left", fontsize=9.5)
fig.suptitle("Where the bit errors come from: the channel, snipping, or never saving the signal\n"
             "a missed signal is charged only the excess over what the channel would have cost it anyway",
             y=1.06, fontsize=13.5, color=INK)
out = RES / "fig_error_budget.png"
save(fig, out)

# ---- C3. per-class detection coverage ----------------------------------------------- #
COV = {d: {c: dict(snr=[], cov=[]) for c in CLASSES} for d in DETS}
for L in have:
    for det in DETS:
        f = RES / f"ber_{det}_attenuation_dB_{L}_byclass.csv"
        if not f.exists():
            continue
        with open(f) as fh:
            for x in csv.DictReader(fh):
                c = x["class"].strip('"')
                if c in COV[det]:
                    COV[det][c]["snr"].append(SNR0 - L)
                    COV[det][c]["cov"].append(100.0 * float(x["detectRate"]))

fig, axes = plt.subplots(2, 4, figsize=(17.0, 8.2), sharex=True, sharey=True)
for i, c in enumerate(CLASSES):
    ax = axes.flat[i]
    ax.axhline(100, color=STYLE[GT]["color"], ls=STYLE[GT]["ls"], lw=1.6, zorder=3)
    for det in DETS:
        d = COV[det][c]
        if not d["snr"]:
            continue
        s = STYLE[det]
        ax.plot(d["snr"], d["cov"], color=s["color"], ls=s["ls"], lw=1.9,
                marker=s["marker"], ms=6.5, mew=1.4, zorder=4)
    ax.set_title(CLASS_LABEL.get(c, c), pad=6)
    tidy(ax, xlabel=None, ylabel="signals saved %" if i % 4 == 0 else None)
    ax.yaxis.set_major_formatter(pctfmt())
    ax.set_ylim(-4, 108)
for ax in axes.flat[len(CLASSES):]:
    ax.axis("off")
    h = [plt.Line2D([], [], color=STYLE[GT]["color"], ls=STYLE[GT]["ls"], lw=2.0,
                    label="perfect detection")]
    h += [plt.Line2D([], [], color=STYLE[d]["color"], ls="-", lw=2.0, marker=STYLE[d]["marker"],
                     ms=8, mew=1.6, label=STYLE[d]["label"]) for d in DETS]
    ax.legend(handles=h, loc="center", fontsize=11)
for ax in axes[1]:
    ax.set_xlabel("SNR (dB)")
fig.suptitle("Detection coverage per modulation class", y=1.02, fontsize=13.5, color=INK)
out = RES / "fig_miss_coverage_byclass.png"
save(fig, out)

print("\nerror budget (BER %, share of every scored bit):")
for det in DETS:
    b = BUD[det]
    print(f"  -- {STYLE[det]['label']}")
    print(f"    {'SNR':>5} {'saved':>7} | {'channel':>8} {'snip':>7} {'decfail':>8} {'MISSES':>8} | {'overall':>8}")
    for i, snr in enumerate(b["snr"]):
        print(f"    {snr:5.0f} {b['cov'][i]:6.1f}% | {b['ch'][i]:7.2f}% {b['snip'][i]:+6.2f}% "
              f"{b['dfail'][i]:+7.2f}% {b['miss'][i]:+7.2f}% | {b['overall'][i]:7.2f}%")

# ---------------------------------------------------------------- summary ----------- #
print("\nheadline numbers for the claim (matched signals, BER %):")
print(f"  {'SNR':>5} | {'channel':>8} {'coh snip':>9} {'Δ':>6} | {'channel':>8} {'dino snip':>9} {'Δ':>6}")
for i, snr in enumerate(CLAIM[DETS[0]]["snr"]):
    row = f"  {snr:5.0f} |"
    for det in DETS:
        c = CLAIM[det]
        j = c["snr"].index(snr) if snr in c["snr"] else None
        if j is None:
            row += f" {'-':>8} {'-':>9} {'-':>6} |"
        else:
            row += f" {c['gt'][j]:7.3f}% {c['det'][j]:8.3f}% {c['det'][j]-c['gt'][j]:+6.3f} |"
    print(row)
