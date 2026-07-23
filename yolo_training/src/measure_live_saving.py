"""Data-saving on the LIVE OTA captures (2.45 GHz, 200 MHz BW, dense bursty traffic).
Runs the offline-runnable detectors (FT-DINO M1/M2, YOLO26m) on the live captures, computes
time-slice occupancy + TF-coverage per model -> live_data_saving.csv for the notebook's Figure 3.
Coherent-power + zero-shot DINO are container-only (no live masks) -> rows marked 'pending(container)'.
Run with the dinov3 env python (has DINO + ultralytics). No ground truth (unlabeled) -> no retention.
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
FT=Path.home()/"Holohub-Signal-Detection/dino_fine_tuning"
for p in (Path.home()/"dinov3", FT/"src", Path.home()/"Holohub-Signal-Detection/yolo_training/src"):
    sys.path.insert(0,str(p))
import finetuned_infer as fi
from yolo_infer import YoloDetector
import yaml
LIVE=Path.home()/"captures/live_data/sigmf_out"
RATE=245.76e6; BPS=8; FRAME_ROWS, NFFT = 512, 10240
FRAME_SAMPLES=FRAME_ROWS*NFFT; SAVE_ALL_TB_HR=RATE*BPS*3600/1e12
NPF=int(sys.argv[1]) if len(sys.argv)>1 else 50   # frames/file

cfg=yaml.safe_load(open(FT/"configs/train.yaml")); ds=json.loads((FT/"data/dataset/dataset_meta.json").read_text())
M1=fi.FinetunedDetector(str(FT/"checkpoints/M1_ft/best.pt"),cfg,ds,threshold=fi.load_threshold(FT/"eval_out/M1_ft/eval_meta.json"))
M2=fi.FinetunedDetector(str(FT/"checkpoints/M2_ft/best.pt"),cfg,ds,threshold=fi.load_threshold(FT/"eval_out/M2_ft/eval_meta.json"))
YR=Path.home()/"Holohub-Signal-Detection/yolo_training/runs/detect"
YM=YoloDetector(str(YR/"yolo26m_signal/weights/best.pt"),ds,conf=0.25,name="yolo26m")
MODELS={"finetuned_dino":M1,"finetuned_dino_m2":M2,"yolo26m":YM}

acc={k:{"ts":[],"cov":[]} for k in MODELS}
for df in sorted(LIVE.glob("*.sigmf-data")):
    mm=np.memmap(df,dtype=np.complex64,mode="r"); nfr=len(mm)//FRAME_SAMPLES
    for fidx in np.linspace(0,nfr-1,min(NPF,nfr)).astype(int):
        s=int(fidx)*FRAME_SAMPLES; iq=np.asarray(mm[s:s+FRAME_SAMPLES],dtype=np.complex64)
        for name,det in MODELS.items():
            m=fi.to_display_grid(det.mask_for_iq(iq),FRAME_ROWS,NFFT)
            acc[name]["ts"].append(m.any(axis=1).mean()); acc[name]["cov"].append((m!=0).mean())
    print(f"  {df.stem}: done", flush=True)

rows=[dict(model="naive_save_all", timeslice_frac=1.0, tf_coverage=1.0,
           stored_TB_hr_timeslice=SAVE_ALL_TB_HR, stored_TB_hr_resample_proj=SAVE_ALL_TB_HR, status="baseline")]
# Container detectors (coherent_power, zero-shot cuda_dino): auto-read their masks if a container
# batch has produced them for the live captures; else mark pending. DROP-IN: run the container batch
# with --output-root LIVE_CONTAINER (see applications/usrp_wideband_signal_detection/infocom_evals/snip_eval/README.md) and re-run this script.
LIVE_CONTAINER = Path("/tmp/usrp_spectrograms/batch_eval/live_ota")
def _load_mask(fp):
    z=np.load(fp); return np.unpackbits(z["packed"])[:int(z["rows"])*int(z["cols"])].reshape(int(z["rows"]),int(z["cols"]))
def container_stats(det_dir):
    mfs=sorted(det_dir.glob("*/mask_arrays/*.packed.npz"))[:NPF*3]
    if not mfs: return None
    ts=[]; cov=[]
    for mf in mfs:
        m=_load_mask(mf); ts.append(m.any(axis=1).mean()); cov.append((m!=0).mean())
    return float(np.mean(ts)), float(np.mean(cov))
for fig_name, cdir in (("coherent_power","coherent_power"), ("cuda_dino_zeroshot","cuda_dino")):
    st=container_stats(LIVE_CONTAINER/cdir)
    if st is None:
        rows.append(dict(model=fig_name, timeslice_frac=np.nan, tf_coverage=np.nan,
                         stored_TB_hr_timeslice=np.nan, stored_TB_hr_resample_proj=np.nan, status="pending(container)"))
    else:
        ts,cov=st
        rows.append(dict(model=fig_name, timeslice_frac=ts, tf_coverage=cov,
                         stored_TB_hr_timeslice=SAVE_ALL_TB_HR*ts, stored_TB_hr_resample_proj=SAVE_ALL_TB_HR*cov,
                         status="container"))
for name,a in acc.items():
    ts=float(np.mean(a["ts"])); cov=float(np.mean(a["cov"]))
    rows.append(dict(model=name, timeslice_frac=ts, tf_coverage=cov,
                     stored_TB_hr_timeslice=SAVE_ALL_TB_HR*ts, stored_TB_hr_resample_proj=SAVE_ALL_TB_HR*cov, status="offline"))
out=Path("applications/usrp_wideband_signal_detection/infocom_evals/snip_eval/live_data_saving.csv"); pd.DataFrame(rows).to_csv(out,index=False)
print(f"\nSAVE-ALL(OTA)={SAVE_ALL_TB_HR:.2f} TB/hr  (frames/file={NPF})")
print(pd.DataFrame(rows)[["model","timeslice_frac","tf_coverage","stored_TB_hr_timeslice","status"]].to_string(index=False))
print(f"wrote {out}")
