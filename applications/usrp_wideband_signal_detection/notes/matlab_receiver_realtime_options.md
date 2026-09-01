# Making the MATLAB receivers "real-time" for the demo — options analysis

Context: the original composite (`comprehensive_ordered`, 245.76 MSps, 3,594 annotations)
contains classes whose receivers only exist in MATLAB (`decode_waveforms_24576.m`):
**5G NR downlink** (5G Toolbox, MCS/code-rate variations), **802.11ax** (WLAN Toolbox,
CBW20–160, MCS 0/4/7/9), **Bluetooth BR/EDR + LE** (Bluetooth Toolbox), plus generic
single-carrier PSK/QAM and framed OFDM (portable — being ported to the Python daemon as
Tier 1, so this document is about the standards classes).

Hard platform fact: **MATLAB does not run on the DGX Spark.** The Spark is Linux aarch64;
MATLAB (and the free MATLAB Compiler Runtime) ship x86-64 Linux only. Any option that
"runs MATLAB" implies a second x86 machine.

Ground truth is NOT a blocker for any option: every record's `txBits` loads in Python via
`scipy.io.loadmat`, each record has a JSON sidecar, and composite annotations carry
`wfgt:source_mat` — so BER scoring, bit-loss denominators, and truth joins never need MATLAB.

---

## Option 1 — Remote MATLAB decode bridge (RECOMMENDED)

A persistent MATLAB session on an x86 box (the bench) runs a small service loop wrapping
`decode_waveforms_24576`'s received-IQ entry point
(`decode_waveforms_24576(rx, "Fs",..., "Metadata",md, "TxBits",bits)`). The Spark daemon
ships only the 5G/Wi-Fi/BT snips (sc16-compressed SigMF — a ~20 ms 100 MHz-wide snip is a
few MB; narrowband BT snips are KB) over SSH/NFS/TCP; per-snippet BER streams back and is
folded into `rt_metrics.json` asynchronously. The dashboard shows these classes' BER
~1–3 s behind true-live.

- Fidelity: **exact original receive chains**, bit-for-bit the published numbers.
- Effort: ~1 day (MATLAB watch-folder/TCP loop + daemon transport + metrics merge).
- Latency/throughput: MATLAB decode of one PPDU/slot is ~0.1–1 s; keep the session warm
  (30 s cold start) and use `parfeval` if the snip rate outruns one worker.
- Risks: demo depends on a second machine + network; license seat tied up during demos.
- Future-proof: any richer toolbox-generated corpus decodes on the MATLAB side with zero
  Spark changes. This is also the only option that scales to "arbitrary new standards".

## Option 2 — MATLAB Coder → C/C++ → cross-compile to aarch64 (viable endgame, weeks)

Generate C from the receive chains, compile for the Spark's Grace cores, wrap for the
daemon (ctypes/pybind11) or as a Holoscan operator. Generated C is license-free to deploy.

- **5G Toolbox**: much of the function surface (nrPDSCHDecode, nrLDPCDecode, nrChannelEstimate…)
  officially supports C codegen, BUT the example-style receiver script must be reworked into
  codegen-compatible form (no dynamic structs/strings, fixed sizes per configuration). Realistic:
  1–3 weeks for the specific MCS/CBW variations used, done by someone with MATLAB + Coder license
  on x86. CPU-only output; snippet-rate decode on Grace cores should be fine (not line-rate).
- **WLAN Toolbox (802.11ax)**: HE recovery functions have *partial* codegen support — audit
  `wlanHEDataRecover`/`wlanHEDemodulate` per release before committing. Same rework caveats.
- **Bluetooth Toolbox**: decode functions have limited codegen coverage; BUT BR/EDR/LE PHYs are
  simple enough that the Tier-1 Python port is likely cheaper than Coder here.
- Verdict: choose this only if the second-machine dependency is unacceptable for the final demo.
  It freezes the receiver at codegen time (new variations ⇒ regenerate), and debugging generated
  C against MATLAB behavior is its own project.

## Option 3 — MATLAB Compiler / MCR or MATLAB Production Server (x86-only; a worse Option 1)

Compiling the receivers to a standalone (MCR) binary removes the license need at *runtime*,
but MCR is x86-only — the binary still runs on the bench, so architecturally this is Option 1
with less transparency and no interactive debugging. `matlab.engine` (Python↔MATLAB) likewise
requires MATLAB local to the Python process. Production Server is a paid product solving a
problem (many concurrent clients) we don't have. **Not recommended** except as a licensing
tweak to Option 1 (compile once, run the service under free MCR).

## Option 4 — Re-implement on open-source / Python stacks (mostly UNADVISABLE)

- **802.11ax: effectively impossible.** No open-source HE (11ax) receiver exists
  (gr-ieee802-11 stops at 11a/g/p). Writing one is a research project. Don't.
- **5G NR: possible but weeks, different numerics.** NVIDIA Sionna (GPU-native, has LDPC,
  OFDM, channel estimation blocks) or srsRAN internals could be assembled into a
  PDSCH-only file decoder. Interesting *if* a GPU-native 5G receiver is itself a research
  goal for the paper; otherwise strictly worse than Option 1 for matching MATLAB's numbers.
- **Bluetooth: moderate and reasonable.** BR/EDR and LE are GFSK with published packet
  formats (access code / preamble, whitening, CRC) — a numpy receiver is a few days and
  belongs in Tier 1 rather than here.

## Option 5 — Precomputed BER replay (demo-pragmatic backup; label it honestly)

The loopback demo replays a FIXED composite through a deterministic chain. Decode every
record ONCE per SNR variant offline in MATLAB; at demo time, when the pipeline detects and
identifies a record (time/freq join, or Tier-1 sync), display its precomputed BER keyed by
the active SNR rung. Looks live, zero live compute, no second machine at demo time.

- Honest for the loopback demo *if labeled* ("reference BER, precomputed through the same
  chain") because the loopback channel is fixed; **dishonest for OTA**, where the channel
  varies — never present it as live decode there.
- Also useful as the fallback lane when the x86 box is unreachable.

## Option 6 — Don't decode these classes live (baseline fallback)

Detection, classification accuracy (11ax/5G → OFDM family, BT → FSK family — a real
generalization test for the classifiers), snipping, and compression all work without
receivers, and the whole-BER *denominator* (bits lost to missed detections) is computable
from `txBits` lengths alone. Decode column reads "offline (MATLAB)". Zero effort.

---

## Recommendation

| Horizon | Choice |
| --- | --- |
| Now (Tier 0/1) | Option 6 accounting + Tier-1 Python ports (generic SC/OFDM, then BT) |
| Demo with x86 on LAN | **Option 1 bridge** for 5G/11ax (+BT until ported) |
| Demo with no second machine | Option 5 precomputed replay, clearly labeled |
| Long-term fully-local | Option 2 Coder for 5G first (best codegen support), re-audit WLAN |
| Avoid | Option 3 as an architecture; Option 4 for 11ax |
