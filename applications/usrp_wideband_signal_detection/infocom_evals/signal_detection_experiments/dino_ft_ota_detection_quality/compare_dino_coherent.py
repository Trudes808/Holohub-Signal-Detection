#!/usr/bin/env python3
"""DINO-FT vs coherent masks on the SAME real 2.4 GHz OTA frame (teacher check)."""
import glob, os, re
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

DD="/tmp/usrp_spectrograms/dino_ota_char"; DC="/tmp/usrp_spectrograms/coh_ota"
OUT="/tmp/claude-1000/-home-genesys-dgx1-Documents-Holohub-Signal-Detection/323ea032-eab1-4437-b121-04a515160984/scratchpad"
def fnum(p):
    m=re.search(r"_f(\d+)_",os.path.basename(p)); return int(m.group(1)) if m else -1
def loadspec(p):
    s=np.load(p); return (10*np.log10(np.abs(s)**2+1e-12)).astype(np.float32) if np.iscomplexobj(s) else s.astype(np.float32)

dmask={fnum(p):p for p in glob.glob(f"{DD}/mask_arrays/mask_ch0_f*.npy")}
cmask={fnum(p):p for p in glob.glob(f"{DC}/mask_arrays/mask_ch0_f*.npy")}
spec={fnum(p):p for p in glob.glob(f"{DD}/spectrogram_tensors/spectrogram_tensor_ch0_f*.npy")}
frames=sorted(set(dmask)&set(cmask)&set(spec))
# aggregate occupancy
docc=np.mean([np.load(dmask[f]).mean() for f in frames]); cocc=np.mean([np.load(cmask[f]).mean() for f in frames])
print(f"frames={len(frames)}  DINO occ={docc*100:.2f}%  coherent occ={cocc*100:.2f}%")

fs=491.52e6; fc=2400e6
f=frames[len(frames)//2]
s=loadspec(spec[f]); md=np.load(dmask[f]).astype(float); mc=np.load(cmask[f]).astype(float)
nf=s.shape[1]; fghz=(fc+(np.arange(nf)/nf-0.5)*fs)/1e9
fig,ax=plt.subplots(3,1,figsize=(14,9))
for a in ax: a.set_xlabel("GHz")
ax[0].imshow(s,aspect='auto',origin='lower',extent=[fghz[0],fghz[-1],0,s.shape[0]],cmap='gray'); ax[0].set_title(f"frame {f}: input spectrogram (real 2.4 GHz OTA)")
ax[1].imshow(s,aspect='auto',origin='lower',extent=[fghz[0],fghz[-1],0,s.shape[0]],cmap='gray')
ax[1].imshow(np.ma.masked_where(md<0.5,md),aspect='auto',origin='lower',extent=[fghz[0],fghz[-1],0,s.shape[0]],cmap='autumn',alpha=0.6)
ax[1].set_title(f"DINO-FT mask (occ {np.load(dmask[f]).mean()*100:.2f}%) -- coarse blobs, merges channels+gaps")
ax[2].imshow(s,aspect='auto',origin='lower',extent=[fghz[0],fghz[-1],0,s.shape[0]],cmap='gray')
ax[2].imshow(np.ma.masked_where(mc<0.5,mc),aspect='auto',origin='lower',extent=[fghz[0],fghz[-1],0,s.shape[0]],cmap='winter',alpha=0.7)
ax[2].set_title(f"coherent mask (occ {np.load(cmask[f]).mean()*100:.2f}%) -- tighter per-signal (candidate teacher)")
plt.tight_layout(); plt.savefig(f"{OUT}/dino_vs_coherent_ota.png",dpi=90); print(f"saved {OUT}/dino_vs_coherent_ota.png")
