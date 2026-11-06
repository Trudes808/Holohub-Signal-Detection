"""Real receiver-noise backgrounds for the scene generator (v5, 2026-11-05).

Index the terminated-input N320 recordings (bash_scripts/capture_n320_term_backgrounds.sh ->
data/real_backgrounds/term/*.sigmf-*), measure per file the noise power, the DC line and any persistent narrow
spur lines (which the "flag everything" convention labels as signals), and serve unit-power background segments
at a requested scene rate (exact rate, or the nearest higher rate decimated; never upsampled, which would leave
the outer band empty).

  PYTHONPATH=src .venv-ml/bin/python src/real_backgrounds.py --build     # writes data/real_backgrounds/index.json
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE / "data" / "real_backgrounds"
INDEX = ROOT / "index.json"
_INDEX_CACHE = None
_SEG_CACHE: dict = {}


def _load_iq(path: Path, dtype: str, offset: int, n: int) -> np.ndarray:
    if dtype.startswith("ci16"):
        raw = np.fromfile(str(path), dtype="<i2", count=2 * n, offset=4 * offset); m = (raw.size // 2) * 2
        return ((raw[0:m:2].astype(np.float32) + 1j * raw[1:m:2].astype(np.float32)) / 32767.0).astype(np.complex64)
    return np.fromfile(str(path), dtype=np.complex64, count=n, offset=8 * offset)


def analyse(path: Path, rate: float, dtype: str) -> dict:
    """noise power, DC line and persistent spur lines of one recording (first 512 rows of the app's wide FFT)."""
    import rates as R
    F = R.app_fft_size(rate); n = min(512 * F, (path.stat().st_size // (4 if dtype.startswith("ci16") else 8)) // F * F)
    iq = _load_iq(path, dtype, 0, n); rows = iq.size // F
    x = iq[: rows * F].reshape(rows, F)
    w = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(F) / F)).astype(np.float32); w /= np.sqrt(np.mean(w ** 2))
    spec = np.fft.fftshift(np.fft.fft(x * w, axis=1), axes=1)
    db = 10 * np.log10(np.abs(spec) ** 2 + 1e-20)
    col = np.median(db, axis=0)
    sigma = max(2.0, 0.005 * F); r = int(math.ceil(3 * sigma)); k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2); k /= k.sum()
    smooth = np.convolve(np.pad(col, r, mode="edge"), k, mode="valid")
    excess = col - smooth
    c = F // 2
    dc_db = float(excess[c - 1: c + 2].max())
    # persistent lines: contiguous columns > 4 dB above the smoothed floor (excluding the DC bins)
    lines = []
    above = excess > 4.0; above[c - 2: c + 3] = False
    edges = np.flatnonzero(np.diff(np.r_[0, above.astype(np.int8), 0]))
    for s0, s1 in zip(edges[0::2], edges[1::2]):
        lines.append({"f_hz": float(((s0 + s1) / 2.0 - F / 2.0) / F * rate), "bw_hz": float(max(s1 - s0, 1) * rate / F), "excess_db": float(excess[s0:s1].max())})
    p = np.abs(iq) ** 2
    return {"noise_power": float(np.mean(p)), "peak": float(np.max(np.abs(iq.real))), "dc_excess_db": dc_db, "lines": lines[:32],
            "floor_db_median": float(np.median(col)), "edge_rolloff_db": [float(smooth[0] - np.median(smooth)), float(smooth[-1] - np.median(smooth))]}


def build_index(root: Path = ROOT) -> list[dict]:
    entries = []
    for mp in sorted(root.rglob("*.sigmf-meta")):
        meta = json.loads(mp.read_text()); g = meta["global"]
        dp = mp.with_suffix(".sigmf-data")
        if not dp.exists():
            continue
        rate = float(g["core:sample_rate"]); dtype = g.get("core:datatype", "ci16_le")
        n = dp.stat().st_size // (4 if dtype.startswith("ci16") else 8)
        try:
            a = analyse(dp, rate, dtype)
        except Exception as e:
            print("skip", mp.name, e); continue
        entries.append({"path": str(dp), "dtype": dtype, "rate_hz": rate, "n_samples": int(n), "gain_db": g.get("argus:gain_db"),
                        "channel": g.get("argus:rx_channel"), "antenna": g.get("argus:antenna"), **a})
        print(f"  {mp.stem}: noise {10*math.log10(a['noise_power']+1e-30):.1f} dB, DC +{a['dc_excess_db']:.1f} dB, {len(a['lines'])} spur lines, edges {a['edge_rolloff_db'][0]:.1f}/{a['edge_rolloff_db'][1]:.1f} dB")
    # clean flag: a terminated input should be quiet and nearly line-free; recordings whose noise power is >20x the
    # median of their gain group or that show > 6 persistent lines (e.g. the 100 MHz LO looks that leak the FM band
    # through the termination) are kept in the index but not served as backgrounds.
    import statistics
    by_gain = {}
    for e in entries:
        by_gain.setdefault(e["gain_db"], []).append(e["noise_power"])
    for e in entries:
        med = statistics.median(by_gain[e["gain_db"]])
        e["clean"] = bool(e["noise_power"] <= 20 * med and len(e["lines"]) <= 6)
    print(f"clean backgrounds: {sum(e['clean'] for e in entries)} / {len(entries)}")
    INDEX.write_text(json.dumps(entries, indent=1)); print(f"index: {len(entries)} backgrounds -> {INDEX}")
    return entries


def index() -> list[dict]:
    global _INDEX_CACHE
    if _INDEX_CACHE is None:
        _INDEX_CACHE = json.loads(INDEX.read_text()) if INDEX.exists() else []
    return _INDEX_CACHE


def pick(rng: np.random.Generator, rate_hz: float, n_samples: int, max_ratio: float = 2.05):
    """A unit-noise-power background of n_samples at rate_hz, or None. Candidates: same rate (tolerance 0.5 %) or a
    higher rate up to max_ratio (decimated with resample_poly). Returns (iq complex64, entry, labelled lines)."""
    import synth_mods as sm
    cands = [e for e in index() if e.get("clean", True) and 0.995 * rate_hz <= e["rate_hz"] <= max_ratio * rate_hz]
    if not cands:
        return None
    exact = [e for e in cands if abs(e["rate_hz"] - rate_hz) <= 0.005 * rate_hz]
    pool = exact if (exact and rng.random() < 0.8) else cands
    e = pool[rng.integers(len(pool))]
    ratio = e["rate_hz"] / rate_hz
    need_src = int(math.ceil(n_samples * ratio)) + 4096
    # random segment (tile with a random offset when the recording is shorter than needed)
    if e["n_samples"] >= need_src:
        off = int(rng.integers(0, e["n_samples"] - need_src + 1))
        seg = _load_iq(Path(e["path"]), e["dtype"], off, need_src)
    else:
        whole = _load_iq(Path(e["path"]), e["dtype"], 0, e["n_samples"])
        off = int(rng.integers(0, e["n_samples"])); whole = np.roll(whole, -off)
        seg = np.tile(whole, int(math.ceil(need_src / e["n_samples"])))[:need_src]
    if abs(ratio - 1.0) > 0.005:
        seg = sm.resample_to(seg, e["rate_hz"], rate_hz)
    seg = seg[:n_samples]
    if seg.size < n_samples:
        seg = np.tile(seg, int(math.ceil(n_samples / max(seg.size, 1))))[:n_samples]
    seg = (seg / np.float32(math.sqrt(float(np.mean(np.abs(seg) ** 2)) + 1e-30))).astype(np.complex64)
    lines = [l for l in e["lines"] if abs(l["f_hz"]) < 0.5 * rate_hz]   # spur lines that fall inside the decimated band
    return seg, e, lines


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--build", action="store_true"); a = ap.parse_args()
    if a.build:
        build_index()
    else:
        print(len(index()), "backgrounds in index")
