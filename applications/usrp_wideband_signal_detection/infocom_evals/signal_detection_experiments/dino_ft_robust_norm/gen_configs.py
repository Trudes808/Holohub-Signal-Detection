#!/usr/bin/env python3
"""Generate the 3 normalization-mode configs for the robust-normalization validation.

All three share config_noadapt.yaml's detector block byte-for-byte EXCEPT the normalization knobs and
the debug dump fields, so any difference in the dumped model input / mask is attributable to the
normalization mode alone:
  - fixed : adaptive_normalization off              (fixed calibrated db_vmin/db_vmax clip)
  - adapt : adaptive on, robust floor off           (legacy q-blend flatten-reference anchor)
  - robust: adaptive on, robust floor on            (low-percentile floor + fixed-clip fallback)
The dump dir is per-mode; the driver moves the host dump to a per-(mode,scene) dir after each run.
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "dino_ft_snr_benchmark" / "config_noadapt.yaml"

# The detector block's normalization stanza in the base config (exact text).
ANCHOR = """  adaptive_normalization: false
  adaptive_span_db: 34.0
  adaptive_floor_frac: 0.12"""

DEBUG = """  # --- robust-normalization validation: dump the [0,1] model input + mask for a few frames ---
  debug_mask_dump_dir: "/workspace/spectrograms/robust_val/{mode}"
  debug_mask_dump_max_frames: 16
  debug_spectrogram_dump: true"""

MODES = {
    # adaptive_normalization, adaptive_robust_floor
    "fixed":  (False, False),
    "adapt":  (True,  False),
    "robust": (True,  True),
}


def main():
    base = BASE.read_text(encoding="utf-8")
    assert ANCHOR in base, "anchor stanza not found in base config"
    for mode, (adaptive, robust) in MODES.items():
        stanza = (
            f"  adaptive_normalization: {'true' if adaptive else 'false'}\n"
            f"  adaptive_span_db: 34.0\n"
            f"  adaptive_floor_frac: 0.12\n"
            f"  adaptive_robust_floor: {'true' if robust else 'false'}\n"
            f"  adaptive_low_pct: 20.0\n"
            f"  adaptive_high_pct: 95.0\n"
            f"  adaptive_min_range_db: 8.0\n"
            f"  adaptive_floor_below_calib_db: 25.0\n"
            + DEBUG.format(mode=mode)
        )
        out = base.replace(ANCHOR, stanza)
        p = HERE / f"config_{mode}.yaml"
        p.write_text(out, encoding="utf-8")
        print("wrote", p)


if __name__ == "__main__":
    main()
