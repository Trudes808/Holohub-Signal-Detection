#!/usr/bin/env python3
"""Round-trip test of to_sigmf.py on synthetic files in every supported container (no GPU, no datasets needed).

    python3 test_to_sigmf.py            # exits 0 when every case passes

Each case writes the same known complex samples in one format, converts it, and checks that the SigMF output reads
back to the same samples (exactly for float formats, within quantisation for integer ones) with the right rate,
centre frequency and segment layout.
"""
from __future__ import annotations

import json
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import to_sigmf as tsg  # noqa: E402

RATE, FC = 20e6, 2.442e9
rng = np.random.default_rng(7)
X = (0.3 * (rng.standard_normal(8192) + 1j * rng.standard_normal(8192))).astype(np.complex64)
CLIPS = (0.3 * (rng.standard_normal((3, 2048)) + 1j * rng.standard_normal((3, 2048)))).astype(np.complex64)


def read_back(meta_path: Path) -> tuple[dict, np.ndarray]:
    meta = json.loads(meta_path.read_text())
    data = meta_path.with_name(meta_path.name[: -len(".sigmf-meta")] + ".sigmf-data")
    return meta, tsg.read_raw(data, meta["global"]["core:datatype"])


def check(name: str, conv: tsg.Converted, want: np.ndarray, tol: float, rate=RATE, fc=FC, segments=1) -> bool:
    meta, got = read_back(conv.meta_path)
    err = float(np.max(np.abs(got - want))) if got.size == want.size else float("nan")
    ok = (got.size == want.size and err <= tol and abs(conv.rate - rate) < 1e-3 and abs(conv.fc - fc) < 1e-3
          and len(meta["captures"]) == segments and meta["annotations"] == [])
    print(f"{'PASS' if ok else 'FAIL'}  {name:<36} {conv.dtype:<8} n={got.size:<6} segs={len(meta['captures'])} "
          f"max|err|={err:.2e} rate={conv.rate:.4g} fc={conv.fc:.6g}")
    return ok


def main() -> int:
    ok = True
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        src, out = d / "src", d / "out"
        src.mkdir()
        ov = tsg.Overrides(rate=RATE, fc=FC)

        # raw files: extension, a guess from the bytes, a sidecar JSON, RTL-SDR offset binary
        X.tofile(src / "a.cf32")
        ok &= check("raw .cf32 (extension)", tsg.convert(src / "a.cf32", out, ov), X, 0)
        X.astype(np.complex128).tofile(src / "b.bin")
        ok &= check("raw .bin complex128 (guessed)", tsg.convert(src / "b.bin", out, ov), X, 1e-7)
        q = np.round(np.stack([X.real, X.imag], -1) * 32767 / 2).astype("<i2")
        q.tofile(src / "c.dat")
        (src / "c.json").write_text(json.dumps({"sample_rate": RATE, "center_frequency": FC, "datatype": "ci16"}))
        conv = tsg.convert(src / "c.dat", out, tsg.Overrides())
        ok &= check("raw ci16 + sidecar JSON", conv, (q[:, 0] + 1j * q[:, 1]) / 32767.0, 1e-6)
        ok &= json.loads(conv.meta_path.read_text())["global"]["argus:source_sidecar"].endswith("c.json")
        lnk = d / "elsewhere"
        lnk.mkdir()
        (lnk / "c_link.dat").symlink_to(src / "c.dat")
        (src / "c_link.json").write_text((src / "c.json").read_text())      # sidecar next to the link target's name
        conv = tsg.convert(lnk / "c_link.dat", out, tsg.Overrides())
        ok &= check("symlinked raw, sidecar at the target", conv, (q[:, 0] + 1j * q[:, 1]) / 32767.0, 1e-6)
        u8 = np.clip(np.round(np.stack([X.real, X.imag], -1) * 127.5 / 2 + 127.5), 0, 255).astype(np.uint8)
        u8.tofile(src / "d.cu8")
        ok &= check("raw .cu8 (RTL-SDR, converted)", tsg.convert(src / "d.cu8", out, ov),
                    ((u8[:, 0] - 127.5) + 1j * (u8[:, 1] - 127.5)) / 127.5, 1e-6)
        ok &= check("raw ci16 forced to cf32 (--cf32)", tsg.convert(src / "c.dat", out / "cf32",
                                                                   tsg.Overrides(force_cf32=True)),
                    (q[:, 0] + 1j * q[:, 1]) / 32767.0, 1e-6)

        # SigMF: pre-1.0 wrapper, frequency in MHz, string numbers; output stem for the pipeline
        X.tofile(src / "e.sigmf-data")
        (src / "e.sigmf-meta").write_text(json.dumps({"_metadata": {
            "global": {"core:datatype": "cf32", "core:sample_rate": str(RATE), "core:version": "0.02"},
            "captures": [{"core:sample_start": 0, "frequency": FC / 1e6}], "annotations": [{"core:sample_start": 0}]}}))
        conv = tsg.convert(src / "e.sigmf-meta", out, tsg.Overrides(), out_stem="e_softlabel")
        ok &= check("SigMF v0.02, MHz centre, out_stem", conv, X, 0)
        ok &= conv.meta_path.name == "e_softlabel.sigmf-meta"
        ok &= json.loads(conv.meta_path.read_text())["global"]["argus:source_annotations"] == 1

        # arrays
        np.save(src / "f.npy", CLIPS)
        ok &= check("npy complex (3 clips)", tsg.convert(src / "f.npy", out, ov), CLIPS.reshape(-1), 0, segments=3)
        il = np.empty((3, 4096), dtype=np.int16)
        qi = np.round(CLIPS * 10000)
        il[:, 0::2], il[:, 1::2] = qi.real, qi.imag
        np.save(src / "g.npy", il)
        ok &= check("npy int16 interleaved", tsg.convert(src / "g.npy", out, ov), (qi / 32767.0).reshape(-1), 1e-6,
                    segments=3)
        np.savez(src / "h.npz", iq=X, fs=np.array(RATE), fc=np.array(FC))
        ok &= check("npz with fs/fc scalars", tsg.convert(src / "h.npz", out, tsg.Overrides()), X, 0)
        from scipy.io import savemat
        savemat(src / "i.mat", {"f_sig": X.reshape(-1, 1), "Fs": RATE, "txBits": np.zeros((100, 1), np.uint8)})
        ok &= check("mat v5 column vector + Fs", tsg.convert(src / "i.mat", out, tsg.Overrides(fc=FC)), X, 0)
        pairs = np.stack([CLIPS.real, CLIPS.imag], -1).astype(np.float64)
        with open(src / "j.pkl", "wb") as f:
            pickle.dump({"data": [pairs[:2], pairs[2:]], "node_list": ["a", "b"]}, f)
        ok &= check("pickle list of (n,L,2) pairs", tsg.convert(src / "j.pkl", out, ov), CLIPS.reshape(-1), 1e-7,
                    segments=3)
        from scipy.io import wavfile
        wavfile.write(src / "k_2442000000Hz_IQ.wav", int(RATE), np.stack([X.real, X.imag], -1).astype(np.float32))
        ok &= check("IQ WAV, centre from name", tsg.convert(src / "k_2442000000Hz_IQ.wav", out, tsg.Overrides()), X, 0)

        # a pickle that tries to run code must be refused
        class Evil:
            def __reduce__(self):
                return (print, ("this must never run",))
        with open(src / "z.pkl", "wb") as f:
            pickle.dump({"data": Evil()}, f)
        try:
            tsg.convert(src / "z.pkl", out, ov)
            print("FAIL  unsafe pickle was loaded")
            ok = False
        except pickle.UnpicklingError as e:
            print(f"PASS  unsafe pickle refused ({e})")

        # raw samples are symlinked, not copied; the pipeline's own outputs are not picked up as inputs
        ok &= (out / "a.sigmf-data").is_symlink()
        found = {f.name for f, _ in tsg.find_inputs([d], recursive=True)}
        ok &= not any(n.endswith("_softlabel.sigmf-meta") for n in found) and "a.cf32" in found
    print("ALL PASS" if ok else "SOME CASES FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
