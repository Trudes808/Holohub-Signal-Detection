# debug_scripts/

Ad-hoc debugging, analysis, and plotting helpers that are **not** part of the build or the
documented build/calibrate/run workflows. They are kept here so the app root stays focused on
the pipeline itself. Nothing in the main app depends on these.

- `compare_offline_dino_subsection.py` — compare two offline DINO runs over a sub-section.
- `plot_hybrid_support_components.py` — plot the hybrid-fusion support/coherence components.
- `plot_offline_dino_cuda_artifact_compare.py` — visual diff of offline DINO CUDA artifacts.
- `realtime_budget.py` — back-of-envelope real-time latency/throughput budget.

Run them from the app root, e.g. `python3 debug_scripts/realtime_budget.py`.
