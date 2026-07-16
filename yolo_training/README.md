# YOLO training / eval (Ultralytics YOLO26)

Bring **Ultralytics YOLO26** in as another signal-detection model alongside the DINO work,
emitting masks in the **same format as every other detector** so it scores through the
identical `eval_detector_masks.py` pipeline (consistent metrics/buckets/thresholds).

## Goal (parallels `dino_fine_tuning/`)
- **YOLO26 fine-tuned** for signal-vs-noise on RF spectrograms (the primary model, à la DINO M1/M2).
- ~~YOLO26 zero-shot~~ — **dropped**: stock YOLO is closed-vocabulary (COCO), no signal class; not a
  meaningful zero-shot detector for spectrograms. Fine-tuned only.
- YOLO masks materialized into the batch-eval format (like `gen_finetuned_run.py`), then scored
  by the canonical `eval_detector_masks.py` → `compare_tables_*`.
- **Eval notebooks in `notebooks/yolo_evals/`** (repo root, alongside `notebooks/dino_fine_tuning_evals/`):
  a YOLO-only eval, a YOLO-vs-DINO/coherent comparison, an OOD-LTE comparison, and a live-data gallery —
  reusing `eval_viz` / `plot_eval_results` like the DINO notebooks.

## Environment
Conda env **`yolo`** (torch cu124 to match the RTX 4000 Ada / dinov3 env). Create with:
```bash
mamba env create -f environment.yml         # or: bash scripts/setup_env.sh
conda activate yolo
```
`scripts/setup_env.sh` also registers a Jupyter kernel "Python (yolo)" and freezes exact
versions into `requirements.txt`.

## Layout
- `environment.yml` / `requirements.txt` — env spec (loose / pinned).
- `scripts/setup_env.sh` — create env + kernel + freeze; `scripts/train_both.sh` — train yolo26s + yolo26m.
- `src/` — `build_yolo_dataset.py`, `train_yolo.py`, `predict_yolo.py`, `yolo_infer.py`, `gen_yolo_run.py`,
  `assemble_yolo_eval.py`, `assemble_yolo_lte.py`.
- `configs/dataset.yaml.example` — YOLO dataset config template (binary signal class).
- `data/`, `runs/`, `weights/`, `reports/` — datasets / training outputs / checkpoints / figures (git-ignored data).
- **Notebooks + their eval tables/sweeps live in `notebooks/yolo_evals/`** (repo root, next to
  `notebooks/dino_fine_tuning_evals/`) — moved out of this folder to sit with the other eval notebooks.

## Design decisions (locked)
1. **Fine-tuned only** — no zero-shot YOLO (closed-vocab COCO head can't detect RF signals; the DINO
   backbone can transfer, a YOLO head can't). One or more fine-tuned YOLO26 detectors, à la DINO M1/M2.
2. **Masks via detection boxes** — train YOLO26 *detection*; fill each predicted box into the binary
   mask grid. Matches the GT filled-box convention + box-IoU/pixel metrics, apples-to-apples with DINO.
3. **Reuse DINO framing** — build the YOLO dataset from the same 256 time-row x 1024 freq-bin frames
   (nfft=1024) as `dino_fine_tuning`, so labels, geometry, and the eval display grid align.


## Pipeline status
- [x] **Dataset builder** (`src/build_yolo_dataset.py`) — DINO frames+boxes+splits → YOLO format (14588/3038/3150).
- [x] **Train** fine-tuned YOLO26 detection (`scripts/train_both.sh`): yolo26s + yolo26m done (mAP50 ≈ 0.90 each).
- [x] **Mask emitter** (`src/yolo_infer.py` + `src/gen_yolo_run.py`) — YOLO boxes → binary masks on the batch grid
      (`to_display_grid`, verbatim from the DINO path). Geometry validated.
- [x] **Assembler + eval** — `src/assemble_yolo_eval.py` (main sweep) + `src/assemble_yolo_lte.py` (OOD-LTE):
      materialize YOLO masks + symlink the DINO detectors, run `eval_detector_masks.py` →
      `notebooks/yolo_evals/compare_tables*`.
- [x] **Notebooks** (in `notebooks/yolo_evals/`; kernel "Python (yolo)", live uses "Python (dinov3)"):
      `yolo_eval` (YOLO-only), `yolo_vs_dino_comparison` (vs coherent / zero-shot DINO / fine-tuned DINO),
      `lte_ood_yolo_eval` (OOD generalization), `live_data_yolo_vs_dino` (unlabeled live captures).

## Run the eval (after training finishes)
```bash
conda activate yolo
python src/assemble_yolo_eval.py          # main sweep -> notebooks/yolo_evals/compare_tables
python src/assemble_yolo_lte.py           # OOD-LTE    -> notebooks/yolo_evals/compare_tables_lte
```
Then open the notebooks in `notebooks/yolo_evals/` (kernel **Python (yolo)**; the live notebook uses
**Python (dinov3)**) and Run All. Weights are auto-located by glob (`runs/**/yolo26{s,m}_signal/weights/best.pt`),
so the nested Ultralytics run path doesn't matter. Relocate eval tables/sweeps by setting `YOLO_EVAL_ROOT`
(default `notebooks/yolo_evals`).
