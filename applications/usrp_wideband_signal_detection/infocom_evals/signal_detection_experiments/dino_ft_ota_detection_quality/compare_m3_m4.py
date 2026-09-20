#!/usr/bin/env python3
"""M3 (synthetic-only) vs M4 (synthetic+OTA fine-tune) on the SAME real 2.4 GHz OTA capture."""
import glob, os, re
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

M3="/tmp/usrp_spectrograms/dino_ota_char"; M4="/tmp/usrp_spectrograms/dino_ota_m4"
OUT="/tmp/claude-1000/-home-genesys-dgx1-Documents-Holohub-Signal-Detection/323ea032-eab1-4437-b121-04a515160984/scratchpad"
def fnum(p): m=re.search(r"_f(\d+)_",os.path.basename(p)); return int(m.group(1)) if m else -1
def loadspec(p):
    s=np.load(p); return (10*np.log10(np.abs(s)**2+1e-12)).astype(np.float32) if np.iscomplexobj(s) else s.astype(np.float32)

m3={fnum(p):p for p in glob.glob(f"{M3}/mask_arrays/mask_ch0_f*.npy")}
m4={fnum(p):p for p in glob.glob(f"{M4}/mask_arrays/mask_ch0_f*.npy")}
sp={fnum(p):p for p in glob.glob(f"{M3}/spectrogram_tensors/spectrogram_tensor_ch0_f*.npy")}
frames=sorted(set(m3)&set(m4)&set(sp))

# quiet threshold from the input distribution
s0=loadspec(sp[frames[0]]); qlo=np.percentile(s0,20); qhi=np.percentile(s0,99); quiet=qlo+0.15*(qhi-qlo)

def stats(masks):
    occ=[]; fp=0; on=0
    for f in frames:
        m=np.load(masks[f]).astype(bool); s=loadspec(sp[f])
        occ.append(m.mean()); fp+=int(np.count_nonzero(m&(s<quiet))); on+=int(m.sum())
    return np.mean(occ)*100, 100*fp/max(1,on)
o3,fp3=stats(m3); o4,fp4=stats(m4)
print(f"M3: occ={o3:.2f}%  quiet-pixel-FP={fp3:.1f}% of detections")
print(f"M4: occ={o4:.2f}%  quiet-pixel-FP={fp4:.1f}% of detections")

fs=491.52e6; fc=2400e6
f=24 if 24 in frames else frames[len(frames)//2]
s=loadspec(sp[f]); a=np.load(m3[f]).astype(float); b=np.load(m4[f]).astype(float)
nf=s.shape[1]; fghz=(fc+(np.arange(nf)/nf-0.5)*fs)/1e9
fig,ax=plt.subplots(3,1,figsize=(14,9))
for x in ax: x.set_xlabel("GHz")
ax[0].imshow(s,aspect='auto',origin='lower',extent=[fghz[0],fghz[-1],0,s.shape[0]],cmap='gray'); ax[0].set_title(f"frame {f}: real 2.4 GHz OTA input")
ax[1].imshow(s,aspect='auto',origin='lower',extent=[fghz[0],fghz[-1],0,s.shape[0]],cmap='gray')
ax[1].imshow(np.ma.masked_where(a<0.5,a),aspect='auto',origin='lower',extent=[fghz[0],fghz[-1],0,s.shape[0]],cmap='autumn',alpha=0.6)
ax[1].set_title(f"M3 (synthetic-only): occ {np.load(m3[f]).mean()*100:.2f}% -- coarse blobs")
ax[2].imshow(s,aspect='auto',origin='lower',extent=[fghz[0],fghz[-1],0,s.shape[0]],cmap='gray')
ax[2].imshow(np.ma.masked_where(b<0.5,b),aspect='auto',origin='lower',extent=[fghz[0],fghz[-1],0,s.shape[0]],cmap='winter',alpha=0.7)
ax[2].set_title(f"M4 (synthetic+OTA): occ {np.load(m4[f]).mean()*100:.2f}% -- should be tighter/per-signal")
plt.tight_layout(); plt.savefig(f"{OUT}/m3_vs_m4_ota.png",dpi=90); print(f"saved {OUT}/m3_vs_m4_ota.png")
