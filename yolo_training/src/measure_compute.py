"""Measure per-detector compute cost for the data_saving_evals notebook:
GPU memory (measured for the offline-runnable models), FLOP estimates, and real-time
throughput vs the 245.76 MS/s capture rate. Container-only detectors (coherent_power,
zero-shot cuda_dino) get FLOP/mem ESTIMATES (noted). Run with the dinov3 env python
(has torch + dinov3 + ultralytics). Writes notebooks/data_saving_evals/compute_table.csv.
"""
import sys, json, time, math
from pathlib import Path
import numpy as np, torch, pandas as pd
FT=Path.home()/"Holohub-Signal-Detection/dino_fine_tuning"
for p in (Path.home()/"dinov3", FT/"src", Path.home()/"Holohub-Signal-Detection/yolo_training/src"):
    sys.path.insert(0,str(p))
import yaml
RATE=245.76e6; NFFT=1024; TILE=256
TILES_PER_SEC = RATE/NFFT/TILE          # 256x1024 DINO/YOLO tiles needed per second real-time
def peak_mem_mb(fn):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    fn(); torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated()/1e6
def thr_tiles_s(fn, iters=8):
    fn(); torch.cuda.synchronize(); t=time.time()
    for _ in range(iters): fn()
    torch.cuda.synchronize(); return iters/(time.time()-t)

rows=[]
# --- fine-tuned DINO M1/M2 (measured) ---
import finetuned_infer as fi
train_cfg=yaml.safe_load(open(FT/"configs/train.yaml")); ds=json.loads((FT/"data/dataset/dataset_meta.json").read_text())
iq=(np.random.randn(NFFT*TILE)+1j*np.random.randn(NFFT*TILE)).astype(np.complex64)
for name,ck in (("finetuned_dino_m1","M1_ft"),("finetuned_dino_m2","M2_ft")):
    det=fi.FinetunedDetector(str(FT/f"checkpoints/{ck}/best.pt"),train_cfg,ds,threshold=0.5)
    params=sum(p.numel() for p in det.model.parameters())/1e6
    mem=peak_mem_mb(lambda: det.mask_for_iq(iq))
    tps=thr_tiles_s(lambda: det.mask_for_iq(iq))   # 1 tile/call (iq = 1 tile worth)
    rows.append(dict(model=name, params_M=round(params,1), gpu_mem_mb=round(mem), gpu_mem_note="measured",
                     tiles_per_s=round(tps), realtime_x=round(tps/TILES_PER_SEC,3), src="offline"))
# zero-shot cuda_dino ~ same ViT-B backbone -> estimate from M1
m1=rows[0]
rows.append(dict(model="cuda_dino_zeroshot", params_M=m1["params_M"], gpu_mem_mb=m1["gpu_mem_mb"],
                 gpu_mem_note="estimate(~ViT-B backbone)", tiles_per_s=m1["tiles_per_s"],
                 realtime_x=m1["realtime_x"], src="container(est)"))
# --- YOLO26m (measured) ---
from ultralytics import YOLO
ym=YOLO(str(Path.home()/"Holohub-Signal-Detection/yolo_training/runs/detect/yolo26m_signal/weights/best.pt"))
tile=np.repeat((np.random.rand(TILE,NFFT)*255).astype(np.uint8)[:,:,None],3,2)
def yolo_call(): ym.predict(tile,imgsz=1024,verbose=False,device=0)
memY=peak_mem_mb(yolo_call); tpsY=thr_tiles_s(yolo_call)
try:
    from ultralytics.utils.torch_utils import get_flops; gf=get_flops(ym.model, imgsz=1024)
except Exception: gf=float("nan")
rows.append(dict(model="yolo26m", params_M=round(sum(p.numel() for p in ym.model.parameters())/1e6,1),
                 gpu_mem_mb=round(memY), gpu_mem_note="measured", tiles_per_s=round(tpsY),
                 realtime_x=round(tpsY/TILES_PER_SEC,3), src="offline"))
# --- coherent_power: FFT-based estimate ---
fft_flops = 5*NFFT*math.log2(NFFT)          # ~ per row FFT
rows.append(dict(model="coherent_power", params_M=0.0, gpu_mem_mb=50, gpu_mem_note="estimate(FFT+thresh)",
                 tiles_per_s=float("inf"), realtime_x=float("inf"), src="container(est)"))

# FLOP estimates (GFLOPs per 256x1024 tile) — ViT-B/16 forward scaled by tokens; YOLO from ultralytics
VITB_224=17.5   # GFLOPs @224^2 (196 tokens)
tokens=(TILE//16)*(NFFT//16); vitb_tile=VITB_224*tokens/196
flop_map={"finetuned_dino_m1":vitb_tile,"finetuned_dino_m2":vitb_tile,"cuda_dino_zeroshot":vitb_tile,
          "yolo26m": (gf if gf==gf else 130.0), "coherent_power": fft_flops*TILE/1e9}
for r in rows:
    r["gflops_per_tile"]=round(flop_map.get(r["model"],float("nan")),2)
    r["gflops_per_s_realtime"]=round(r["gflops_per_tile"]*TILES_PER_SEC,1)
    r["flops_note"]="est(ViT-B scaled)" if "dino" in r["model"] else ("ultralytics" if r["model"]=="yolo26m" else "est(FFT)")

out=Path("notebooks/data_saving_evals/compute_table.csv")
pd.DataFrame(rows).to_csv(out,index=False)
print(f"tiles/s needed for real-time = {TILES_PER_SEC:.0f}\n")
print(pd.DataFrame(rows)[["model","params_M","gflops_per_tile","gflops_per_s_realtime","gpu_mem_mb","tiles_per_s","realtime_x","src"]].to_string(index=False))
print(f"\nwrote {out}")
