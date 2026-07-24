#!/usr/bin/env python3
"""Real spectrogram (from raw IQ) + detector mask overlay (high-contrast) + snipper bounding boxes.
Bottom row zooms on coherent power's persistence streak so it's visible against the spectrogram.

Usage: python3 render_spectrogram_overlay.py [atten] [frame]   (default 70 100)
"""
from __future__ import annotations
import numpy as np, glob, sys
from scipy import ndimage
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap
from pathlib import Path

SE = Path(__file__).resolve().parent; OUT = SE / "figs_minsize"
CAPS = Path("/home/bqn82/captures")
FS=245.76e6; ROWS=512; COLS=10240; PER_ROW=10240; FRAME=ROWS*PER_ROW   # 5,242,880 samples/frame
HZ_COL=FS/COLS; S_ROW=PER_ROW/FS
MIN_PX=256; GAP_R=16; GAP_C=80; MIN_BW=100e3; MIN_DUR=5e-3
fmhz=lambda c:(c/COLS-0.5)*FS/1e6
tms =lambda r:r*S_ROW*1e3

def load_mask(det, atten, fr):
    c=[x for x in glob.glob(str(SE/f"snip_run/{det}/attenuation_dB_{atten}/mask_arrays/*.packed.npz")) if f"_f{fr}_" in x]
    if not c: return None
    z=np.load(c[0]); return np.unpackbits(z["packed"])[:int(z["rows"])*int(z["cols"])].reshape(int(z["rows"]),int(z["cols"]))

def spectrogram(atten, fr):
    """FFT each 10240-sample row of frame `fr` (1-indexed) -> 512x10240 dB spectrogram (fftshifted)."""
    f = CAPS / f"attenuation_dB_{atten}.sigmf-data"
    start = (fr-1)*FRAME
    iq = np.fromfile(f, dtype=np.complex64, count=FRAME, offset=start*8)
    if iq.size < FRAME: iq = np.pad(iq, (0, FRAME-iq.size))
    rows = iq.reshape(ROWS, PER_ROW)
    spec = np.fft.fftshift(np.fft.fft(rows, axis=1), axes=1)
    return 20*np.log10(np.abs(spec)+1e-6)

def boxes(m):
    lbl,n=ndimage.label(m,structure=[[0,1,0],[1,1,1],[0,1,0]]); sizes=np.bincount(lbl.ravel()); objs=ndimage.find_objects(lbl)
    bs=[[s[0].start,s[0].stop-1,s[1].start,s[1].stop-1,int(sizes[i])] for i,s in enumerate(objs,1) if sizes[i]>=MIN_PX]
    ch=True
    while ch:
        ch=False; out=[]
        for b in bs:
            hit=False
            for r in out:
                if b[0]<=r[1]+GAP_R and r[0]<=b[1]+GAP_R and b[2]<=r[3]+GAP_C and r[2]<=b[3]+GAP_C:
                    r[0]=min(r[0],b[0]);r[1]=max(r[1],b[1]);r[2]=min(r[2],b[2]);r[3]=max(r[3],b[3]);r[4]+=b[4];hit=True;ch=True;break
            if not hit: out.append(b[:])
        bs=out
    return bs

MASKCM = ListedColormap(["#00e5ff"])   # high-contrast cyan for detected pixels
def show(ax, spec, m, extent, draw_boxes=True, vlo=None, vhi=None):
    vlo = np.percentile(spec, 40) if vlo is None else vlo
    vhi = np.percentile(spec, 99.7) if vhi is None else vhi
    ax.imshow(spec, aspect="auto", cmap="magma", extent=extent, vmin=vlo, vmax=vhi, interpolation="nearest")
    ax.imshow(np.ma.masked_where(m==0, m), aspect="auto", cmap=MASKCM, extent=extent, alpha=0.55, interpolation="nearest")
    if draw_boxes:
        for r0,r1,c0,c1,cnt in boxes(m):
            bw=(c1-c0+1)*HZ_COL; dur=(r1-r0+1)*S_ROW; ok = bw>=MIN_BW and dur>=MIN_DUR
            ax.add_patch(Rectangle((fmhz(c0), tms(r0)), fmhz(c1+1)-fmhz(c0), tms(r1+1)-tms(r0),
                         fill=False, edgecolor=("#18ff6d" if ok else "#ff3b3b"),
                         lw=2.4 if ok else 1.1, ls="-" if ok else (0,(3,2)), zorder=6))

atten = int(sys.argv[1]) if len(sys.argv)>1 else 70
fr    = int(sys.argv[2]) if len(sys.argv)>2 else 100
spec = spectrogram(atten, fr)
mc = load_mask("coherent_power", atten, fr); md = load_mask("finetuned_dino_m2", atten, fr)
FEXT=[fmhz(0), fmhz(COLS), tms(ROWS), 0]

# zoom on the streak INSIDE the box that actually passes the gate (its tallest column)
_lbl, _n = ndimage.label(mc, structure=[[0,1,0],[1,1,1],[0,1,0]]); _sz = np.bincount(_lbl.ravel()); _objs = ndimage.find_objects(_lbl)
zc = None
for _i, _s in enumerate(_objs, 1):
    if _sz[_i] < MIN_PX: continue
    _r0,_r1,_c0,_c1 = _s[0].start,_s[0].stop-1,_s[1].start,_s[1].stop-1
    if (_c1-_c0+1)*HZ_COL >= MIN_BW and (_r1-_r0+1)*S_ROW >= MIN_DUR:
        _colsum = (_lbl[:, _c0:_c1+1] == _i).sum(axis=0)
        zc = _c0 + int(np.argmax(_colsum)); break
if zc is None: zc = int(np.argmax(mc.sum(axis=0)))
zlo, zhi = max(0, zc-160), min(COLS, zc+160)     # ~+/-3.8 MHz around the streak

fig = plt.figure(figsize=(16, 9));
axc = fig.add_subplot(2,2,1); show(axc, spec, mc, FEXT); axc.set_title(f"Coherent Power @ atten {atten}, frame {fr}\nspectrogram + mask (cyan) + boxes (green=snipped, red=dropped)", fontsize=9)
axd = fig.add_subplot(2,2,2); show(axd, spec, md, FEXT); axd.set_title(f"DINO FT @ atten {atten}, frame {fr}\n(same spectrogram; bursts wide but too short → all dropped)", fontsize=9)
axz = fig.add_subplot(2,1,2)
ZEXT=[fmhz(zlo), fmhz(zhi), tms(ROWS), 0]
show(axz, spec[:, zlo:zhi], mc[:, zlo:zhi], ZEXT, draw_boxes=False)
axz.axvline(fmhz(zc), color="#00e5ff", ls=":", lw=0.8, alpha=0.6)
axz.set_title(f"ZOOM on coherent's persistence streak near {fmhz(zc):.1f} MHz — a narrowband line lit across the FULL 21 ms "
              f"(this is what gives the box its ≥5 ms height); wideband bursts cross it at ~2.5 & 20 ms", fontsize=9)
for ax in (axc, axd, axz): ax.set_xlabel("freq (MHz)"); ax.set_ylabel("time (ms)")
fig.tight_layout(); fig.savefig(OUT/"debug_spectrogram_overlay.png", dpi=150); print("wrote", OUT/"debug_spectrogram_overlay.png")
print(f"streak column {zc} ({fmhz(zc):.2f} MHz), lit in {int(mc[:, zc].sum())}/{ROWS} rows")
