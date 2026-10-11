"""Portable path resolution for the fine-tuning scripts.

Every script used to hard-code the first author's home directory. Paths in the YAML configs and
CLI defaults are now resolved here:

* ``~`` and ``$ENV_VARS`` are expanded;
* a relative path is taken relative to ``dino_fine_tuning/`` (``FT_ROOT``), not the CWD, so the
  scripts work from any directory;
* the DINOv3 checkout defaults to a sibling of the repository (``<repo>/../dinov3``), the same
  location the demo container build expects; override with ``DINOV3_ROOT``.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

FT_ROOT = Path(__file__).resolve().parents[1]            # .../dino_fine_tuning
REPO_ROOT = FT_ROOT.parent                                # .../Holohub-Signal-Detection
APP_DIR = REPO_ROOT / "applications" / "usrp_wideband_signal_detection"
EVAL_DIR = APP_DIR / "infocom_evals" / "signal_detection_experiments"


def resolve(path: str | os.PathLike, base: Path = FT_ROOT) -> Path:
    """Expand ``~``/``$VARS`` and anchor relative paths at ``base`` (default: dino_fine_tuning/)."""
    p = Path(os.path.expandvars(os.path.expanduser(str(path))))
    return p if p.is_absolute() else (base / p)


def dinov3_root() -> Path:
    """The facebookresearch/dinov3 checkout (package dir ``dinov3/dinov3`` + ``weights/``)."""
    env = os.environ.get("DINOV3_ROOT")
    if env:
        return resolve(env)
    for cand in (REPO_ROOT.parent / "dinov3", Path.home() / "Documents" / "dinov3", Path.home() / "dinov3"):
        if (cand / "dinov3").is_dir():
            return cand
    return REPO_ROOT.parent / "dinov3"


def backbone_weights(ckpt_weights_path: str | None, cfg_weights_path: str | None) -> str:
    """Pick the DINOv3 backbone weights for rebuilding a checkpoint's architecture.

    Order: the path recorded in the checkpoint (if it exists here) -> a file with the same name under
    ``<dinov3_root>/weights`` -> the training config's ``weights_path``. Checkpoints trained on another
    machine carry that machine's absolute path, so the recorded path alone is not reliable.
    """
    cands: list[Path] = []
    if ckpt_weights_path:
        cands.append(resolve(ckpt_weights_path))
        cands.append(dinov3_root() / "weights" / Path(ckpt_weights_path).name)
    if cfg_weights_path:
        cands.append(resolve(cfg_weights_path))
    for c in cands:
        if c.is_file():
            return str(c)
    return str(cands[0]) if cands else ""


def ensure_import_paths() -> None:
    """Make ``dinov3``, this ``src/`` dir and the eval-metrics dir importable (idempotent)."""
    for p in (dinov3_root(), FT_ROOT / "src", EVAL_DIR):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)
