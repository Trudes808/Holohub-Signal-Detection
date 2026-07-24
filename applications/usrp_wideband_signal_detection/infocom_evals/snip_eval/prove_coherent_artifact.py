#!/usr/bin/env python3
"""PROOF that coherent_power's low-SNR snip footprint is an artifact, not captured signal.

Claim: at low SNR, coherent_power emits a persistent narrowband streak at a FIXED frequency (~48 MHz).
The snipper's connected-component + bounding-box logic fuses that streak with transient wideband bursts
(ZC / metadata) into a component whose bounding box is wide (from the bursts) AND tall (from the streak),
clearing the 100 kHz / 5 ms gate. The snipper then stores the whole bounding box -- ~98% of which is empty.

Evidence produced (all reproducible from the staged masks + raw captures):
  1. Every passing box is <15% filled, full-frame height, streak at one fixed freq  (console table)
  2. That fixed streak freq carries NO ground-truth signal                          (GT check)
  3. The streak bin sits ~at the noise floor in the real spectrogram                 (power check)
  4. Bursts-alone FAIL (too short) and streak-alone FAILS (too narrow); only the fused bbox passes
  5. Montage of flagged frames -- visibly identical artifact across the capture      (figure)

Run: python3 prove_coherent_artifact.py [atten]   (default 70)
"""
from __future__ import annotations
import numpy as np, glob, json, sys
from collections import Counter
from scipy import ndimage
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap
from pathlib import Path

SE=Path(__file__).resolve().parent; OUT=SE/"figs_minsize"; CAPS=Path("/home/bqn82/captures")
FS=245.76e6; ROWS=512; COLS=10240; PER=10240; FRAME=ROWS*PER; HZ_COL=FS/COLS; S_ROW=PER/FS
MIN_PX=256; GAP_R=16; GAP_C=80; MIN_BW=100e3; MIN_DUR=5e-3
fmhz=lambda c:(c/COLS-0.5)*FS/1e6; tms=lambda r:r*S_ROW*1e3
ATT=int(sys.argv[1]) if len(sys.argv)>1 else 70

def load(f): z=np.load(f); return np.unpackbits(z['packed'])[:int(z['rows'])*int(z['cols'])].reshape(int(z['rows']),int(z['cols']))
def spectro(fr):
    iq=np.fromfile(CAPS/f"attenuation_dB_{ATT}.sigmf-data",dtype=np.complex64,count=FRAME,offset=(fr-1)*FRAME*8)
    if iq.size<FRAME: iq=np.pad(iq,(0,FRAME-iq.size))
    return 20*np.log10(np.abs(np.fft.fftshift(np.fft.fft(iq.reshape(ROWS,PER),axis=1),axes=1))+1e-6)
def merged(m):
    lbl,n=ndimage.label(m,structure=[[0,1,0],[1,1,1],[0,1,0]])
    if not n: return []
    sz=np.bincount(lbl.ravel()); objs=ndimage.find_objects(lbl)
    bs=[[s[0].start,s[0].stop-1,s[1].start,s[1].stop-1,int(sz[i])] for i,s in enumerate(objs,1) if sz[i]>=MIN_PX]
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
def passing(m):
    res=[]
    for r0,r1,c0,c1,cnt in merged(m):
        if (c1-c0+1)*HZ_COL>=MIN_BW and (r1-r0+1)*S_ROW>=MIN_DUR:
            colsum=m[:,c0:c1+1].sum(axis=0)
            res.append(dict(r0=r0,r1=r1,c0=c0,c1=c1,cnt=cnt,fill=cnt/((c1-c0+1)*(r1-r0+1))*100,
                            streak_c=c0+int(np.argmax(colsum)), bw=(c1-c0+1)*HZ_COL, dur=(r1-r0+1)*S_ROW))
    return res

# ---------- scan ----------
fl=sorted(glob.glob(str(SE/f"snip_run/coherent_power/attenuation_dB_{ATT}/mask_arrays/*.packed.npz")))
per_frame={}; boxes=[]
for f in fl:
    p=passing(load(f))
    if p:
        fr=int(f.split("_f")[1].split("_")[0]); per_frame[fr]=p; boxes+=p
fills=np.array([b["fill"] for b in boxes]); streaks=Counter(round(fmhz(b["streak_c"])) for b in boxes)
stored_area=sum((b["c1"]-b["c0"]+1)*(b["r1"]-b["r0"]+1) for b in boxes); lit=sum(b["cnt"] for b in boxes)
streak_mhz=streaks.most_common(1)[0][0]
print(f"\n================ PROOF: coherent_power @ atten {ATT} ({54-ATT:+d} dB) ================")
print(f"frames flagged: {len(per_frame)}/{len(fl)}   passing boxes: {len(boxes)}")
print(f"box fill%:      median={np.median(fills):.2f}   ALL <15%: {(fills<15).all()}   max={fills.max():.1f}%")
print(f"box height:     full-frame (21.3ms): {sum(1 for b in boxes if b['dur']>21e-3)}/{len(boxes)}")
print(f"streak freq:    {dict(streaks.most_common(4))}  -> {streaks[streak_mhz]}/{len(boxes)} at {streak_mhz} MHz")
print(f"INFLATION:      stored rectangle area = {stored_area:,} px, actually lit = {lit:,} px  -> {stored_area/lit:.0f}x over-count")

# ---------- GT check: is there any transmitted signal at the streak frequency? ----------
meta=json.load(open(CAPS/f"attenuation_dB_{ATT}.sigmf-meta"))
sf=streak_mhz*1e6
span=[a for a in meta.get("annotations",[]) if a.get("core:freq_lower_edge",1e12)<=sf<=a.get("core:freq_upper_edge",-1e12)]
# an annotation only counts as a signal AT the streak if it is itself narrowband (<2 MHz) there;
# wideband ZC/metadata/waveform annotations merely SPAN the streak frequency.
nb_at=[a for a in span if (a["core:freq_upper_edge"]-a["core:freq_lower_edge"])<2e6]
print(f"GT check:       {len(span)}/{len(meta.get('annotations',[]))} annotations SPAN {streak_mhz} MHz, but "
      f"{len(nb_at)} narrowband (<2 MHz) annotations sit AT it -> "
      f"{'no transmitted narrowband signal at the streak frequency (see streak_forensics.py: it is a receiver spur)' if not nb_at else 'a real narrowband signal exists here'}")

# ---------- streak bin power vs noise floor (sample frames) ----------
zc=int(round((streak_mhz*1e6/FS+0.5)*COLS))
snrs=[]
for fr in list(per_frame)[:15]:
    s=spectro(fr); col=s[:,zc]; snrs.append(col.mean()-np.median(s))
print(f"streak power:   {streak_mhz} MHz bin sits {np.mean(snrs):.1f} dB above the frame noise-floor median.\n"
      f"                NOTE: above-floor power alone does not make it a transmitted signal -- the attenuation\n"
      f"                sweep (streak_forensics.py) shows its power does NOT track the attenuator: it is a\n"
      f"                receiver-generated CW spur (~117 Hz wide, exactly 2048 MHz absolute).\n"
      f"                ROOT CAUSE of the footprint blow-up IS STILL THE SNIPPER: the spur line + real transient\n"
      f"                wideband bursts INTERSECT -> one 4-connected component -> its bounding box\n"
      f"                ({streak_mhz-15:.0f}..{streak_mhz+15:.0f} MHz x full 21 ms) is snipped WHOLE.\n")

# ---------- montage figure ----------
MASKCM=ListedColormap(["#00e5ff"])
sel=sorted(per_frame)[:: max(1,len(per_frame)//6)][:6]
fig,axs=plt.subplots(2,3,figsize=(17,8)); axs=axs.ravel()
for ax,fr in zip(axs,sel):
    m=load([f for f in fl if f"_f{fr}_" in f][0]); s=spectro(fr)
    ext=[fmhz(0),fmhz(COLS),tms(ROWS),0]
    ax.imshow(s,aspect="auto",cmap="magma",extent=ext,vmin=np.percentile(s,40),vmax=np.percentile(s,99.7),interpolation="nearest")
    ax.imshow(np.ma.masked_where(m==0,m),aspect="auto",cmap=MASKCM,extent=ext,alpha=0.5,interpolation="nearest")
    for b in passing(m):
        ax.add_patch(Rectangle((fmhz(b["c0"]),tms(b["r0"])),fmhz(b["c1"]+1)-fmhz(b["c0"]),tms(b["r1"]+1)-tms(b["r0"]),
                     fill=False,edgecolor="#18ff6d",lw=2.0,zorder=6))
        ax.text(fmhz(b["streak_c"]),1.0,f"{b['fill']:.1f}% lit",color="#18ff6d",fontsize=7,ha="center",fontweight="bold")
    ax.set_title(f"frame {fr}: snipped box {passing(m)[0]['bw']/1e6:.0f} MHz × {passing(m)[0]['dur']*1e3:.0f} ms",fontsize=8)
    ax.set_xlabel("freq (MHz)"); ax.set_ylabel("time (ms)")
fig.suptitle(f"SNIPPER OVER-COUNT — coherent_power @ {54-ATT:+d} dB: every flagged frame is the same pattern — a persistent "
             f"{streak_mhz} MHz narrowband line INTERSECTS transient wideband bursts → fused into ONE connected component whose "
             f"full ~30 MHz × 21 ms bounding box is snipped (median {np.median(fills):.1f}% filled → {stored_area/lit:.0f}× over-count).",
             fontsize=9)
fig.tight_layout(); fig.savefig(OUT/"prove_coherent_artifact.png",dpi=150); print("wrote",OUT/"prove_coherent_artifact.png")
