#!/usr/bin/env python3
"""Overlay the snipper's bounding boxes on the mask pixels underneath, for coherent_power vs DINO FT
at low SNR. Replicates the snipper: 4-connected components >= min_box_pixels, fixed-point merge, then
the 100 kHz / 5 ms gate. Green solid box = passes gate (SNIPPED, stored as a solid rectangle);
red dashed box = dropped. Gray pixels underneath = the actual mask (what was really detected).
"""
from __future__ import annotations
import numpy as np, glob, sys
from scipy import ndimage
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from pathlib import Path

SE = Path(__file__).resolve().parent; OUT = SE / "figs_minsize"
FS=245.76e6; ROWS=512; COLS=10240; FRAME=5242880
HZ_COL=FS/COLS; S_ROW=FRAME/ROWS/FS
MIN_PX=256; GAP_R=16; GAP_C=80; MIN_BW=100e3; MIN_DUR=5e-3
fmhz=lambda c:(c/COLS-0.5)*FS/1e6
tms =lambda r:r*S_ROW*1e3

def load(f):
    z=np.load(f); return np.unpackbits(z["packed"])[:int(z["rows"])*int(z["cols"])].reshape(int(z["rows"]),int(z["cols"]))
def comps(m):
    lbl,n=ndimage.label(m,structure=[[0,1,0],[1,1,1],[0,1,0]]); sizes=np.bincount(lbl.ravel()); objs=ndimage.find_objects(lbl)
    return [[s[0].start,s[0].stop-1,s[1].start,s[1].stop-1,int(sizes[i])] for i,s in enumerate(objs,1) if sizes[i]>=MIN_PX]
def merge(bs):
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
def find(det, atten, fr):
    c=[x for x in glob.glob(str(SE/f"snip_run/{det}/attenuation_dB_{atten}/mask_arrays/*.packed.npz")) if f"_f{fr}_" in x]
    return c[0] if c else None

def draw(ax, f, title):
    m=load(f)
    ax.imshow(m, aspect="auto", cmap="gray_r", extent=[fmhz(0), fmhz(COLS), tms(ROWS), 0],
              vmin=0, vmax=1, interpolation="nearest")
    boxes=merge(comps(m)); npass=0
    for r0,r1,c0,c1,cnt in boxes:
        bw=(c1-c0+1)*HZ_COL; dur=(r1-r0+1)*S_ROW; fill=cnt/((c1-c0+1)*(r1-r0+1))*100
        ok = bw>=MIN_BW and dur>=MIN_DUR
        col = "#18a558" if ok else "#e34948"
        ax.add_patch(Rectangle((fmhz(c0), tms(r0)), fmhz(c1+1)-fmhz(c0), tms(r1+1)-tms(r0),
                     fill=False, edgecolor=col, lw=2.2 if ok else 1.1, ls="-" if ok else (0,(3,2)), zorder=5))
        if ok:
            npass+=1
            ax.annotate(f"SNIPPED: {bw/1e6:.0f} MHz × {dur*1e3:.0f} ms  (only {fill:.1f}% of this box is lit)",
                        (fmhz((c0+c1)/2), tms(r1)+1.2), color="#0c6b39", fontsize=8, ha="center", fontweight="bold")
    ax.set_title(f"{title} — {len(boxes)} boxes, {npass} pass the 100 kHz & 5 ms gate", fontsize=9)
    ax.set_xlabel("freq (MHz)"); ax.set_ylabel("time (ms)"); ax.set_xlim(fmhz(0), fmhz(COLS))
    return npass

det_a, det_b = "coherent_power", "finetuned_dino_m2"
atten, fr = int(sys.argv[1]) if len(sys.argv)>1 else 70, int(sys.argv[2]) if len(sys.argv)>2 else 100
fig, axs = plt.subplots(1, 2, figsize=(16, 6))
draw(axs[0], find(det_a, atten, fr), f"Coherent Power @ atten {atten}")
draw(axs[1], find(det_b, atten, fr), f"DINO FT @ atten {atten}")
handles=[Line2D([],[],color="#18a558",lw=2.2,ls="-",label="passes gate → snipped (stored as solid rectangle)"),
         Line2D([],[],color="#e34948",lw=1.1,ls=(0,(3,2)),label="dropped (fails 100 kHz and/or 5 ms)")]
fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.02))
fig.suptitle("Snipper bounding boxes over the real mask (gray = detected pixels). "
             "Coherent's box is wide (bursts) AND tall (persistence streak) but ~2.6% filled; DINO's boxes are wide but too short.",
             fontsize=10)
fig.tight_layout(); fig.savefig(OUT/"debug_bbox_overlay.png", dpi=150, bbox_inches="tight")
print("wrote", OUT/"debug_bbox_overlay.png")
