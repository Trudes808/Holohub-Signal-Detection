#!/usr/bin/env python3
"""Characterize DINO-FT false-firing on real 2.4 GHz OTA: mask vs input-energy."""
import glob, os, re, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = "/tmp/usrp_spectrograms/dino_ota_char"
OUT = "/tmp/claude-1000/-home-genesys-dgx1-Documents-Holohub-Signal-Detection/323ea032-eab1-4437-b121-04a515160984/scratchpad"

def fnum(p):
    m = re.search(r"_f(\d+)_", os.path.basename(p)); return int(m.group(1)) if m else -1

def loadspec(p):
    s = np.load(p)
    if np.iscomplexobj(s):
        s = 10.0*np.log10(np.abs(s)**2 + 1e-12)   # power dB
    return s.astype(np.float32)

masks = sorted(glob.glob(f"{D}/mask_arrays/mask_ch0_f*.npy"), key=fnum)
spec_by = {fnum(p): p for p in glob.glob(f"{D}/spectrogram_tensors/spectrogram_tensor_ch0_f*.npy")}
specs = [spec_by[fnum(m)] for m in masks if fnum(m) in spec_by]
masks = [m for m in masks if fnum(m) in spec_by]
print(f"matched frames={len(masks)}")

# inspect one
m0 = np.load(masks[0]); s0 = loadspec(specs[0])
print(f"mask shape={m0.shape} dtype={m0.dtype} nz%={100*np.count_nonzero(m0)/m0.size:.3f}")
print(f"spec(dB) shape={s0.shape} min={s0.min():.1f} max={s0.max():.1f} mean={s0.mean():.1f}")

# Aggregate over all frames: per-frequency-column detection count and energy.
nfreq = m0.shape[1]
det_profile = np.zeros(nfreq); eng_profile = np.zeros(nfreq); nframes = 0
occ_list = []; fp_quiet = 0; tot_on = 0
# "quiet" energy threshold from the spectrogram distribution (robust floor + margin)
allvals = s0.ravel()
qlo = np.percentile(allvals, 20); qhi = np.percentile(allvals, 99)
quiet_thr = qlo + 0.15*(qhi-qlo)
print(f"spec p20={qlo:.3f} p99={qhi:.3f} quiet_thr={quiet_thr:.3f}")

for mp, sp in zip(masks, specs):
    m = np.load(mp).astype(bool); s = loadspec(sp)
    if m.shape != s.shape:
        print("shape mismatch", m.shape, s.shape); continue
    det_profile += m.sum(axis=0)
    eng_profile += s.mean(axis=0)
    nframes += 1
    occ_list.append(m.mean())
    on = m.sum(); tot_on += on
    fp_quiet += np.count_nonzero(m & (s < quiet_thr))
det_profile /= nframes; eng_profile /= nframes
occ = np.array(occ_list)
print(f"\nframes={nframes}  mask occ mean={occ.mean()*100:.2f}% min={occ.min()*100:.2f}% max={occ.max()*100:.2f}%")
print(f"FALSE-POSITIVE (mask-on where input is quiet/no-signal): {100*fp_quiet/max(1,tot_on):.1f}% of all detected pixels")

# Frequency axis in GHz: center 2.4 GHz, span 491.52 MHz (col 0..nfreq maps -fs/2..+fs/2)
fs = 491.52e6; fc = 2400e6
fghz = (fc + (np.arange(nfreq)/nfreq - 0.5)*fs)/1e9

fig, ax = plt.subplots(3,1, figsize=(14,10))
# (1) detection vs energy profile across frequency
ax[0].plot(fghz, det_profile/det_profile.max() if det_profile.max()>0 else det_profile, 'lime', label="DINO detection density (norm)")
ax[0].plot(fghz, (eng_profile-eng_profile.min())/(np.ptp(eng_profile)+1e-9), 'orange', label="input energy (norm)")
ax[0].axhline(0, color='gray', lw=0.5); ax[0].set_title("DINO-FT detection density vs actual input energy, across frequency (avg over 46 OTA frames)")
ax[0].set_xlabel("GHz"); ax[0].legend(); ax[0].grid(alpha=0.2)
# (2) a representative frame: input
fi = len(masks)//2
mi = np.load(masks[fi]).astype(float); si = loadspec(specs[fi]).astype(float)
ax[1].imshow(si, aspect='auto', origin='lower', extent=[fghz[0],fghz[-1],0,si.shape[0]], cmap='viridis')
ax[1].set_title(f"input spectrogram frame {fnum(masks[fi])}"); ax[1].set_xlabel("GHz")
# (3) same frame: mask overlay (red = detected)
ax[2].imshow(si, aspect='auto', origin='lower', extent=[fghz[0],fghz[-1],0,si.shape[0]], cmap='gray')
mo = np.ma.masked_where(mi<0.5, mi)
ax[2].imshow(mo, aspect='auto', origin='lower', extent=[fghz[0],fghz[-1],0,si.shape[0]], cmap='autumn', alpha=0.6)
ax[2].set_title(f"frame {fnum(masks[fi])}: DINO mask (red) over input — red where no energy = false positive"); ax[2].set_xlabel("GHz")
plt.tight_layout(); plt.savefig(f"{OUT}/dino_ota_analysis.png", dpi=90)
print(f"\nsaved {OUT}/dino_ota_analysis.png")

# Where does detection extend beyond energy? column-wise
det_n = det_profile/ (det_profile.max()+1e-9); eng_n = (eng_profile-eng_profile.min())/(np.ptp(eng_profile)+1e-9)
false_cols = (det_n>0.2) & (eng_n<0.15)
print(f"freq columns with detection but ~no energy: {false_cols.sum()} / {nfreq} "
      f"({100*false_cols.sum()/nfreq:.1f}% of band)")
# report the GHz ranges of the strongest false-positive columns
if false_cols.any():
    idx = np.where(false_cols)[0]
    print(f"  false-positive band spans ~{fghz[idx.min()]:.3f} - {fghz[idx.max()]:.3f} GHz")
