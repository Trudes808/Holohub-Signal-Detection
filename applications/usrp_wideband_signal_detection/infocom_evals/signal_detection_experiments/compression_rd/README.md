# Compression rate–distortion–task eval

Offline sweep for the `snippet_compression` operator (compression branch): for each
codec × SNR capture, the compiled offline eval binary runs the real pipeline
(coherent_power detect → snip → **compress on GPU** → SigMF container), then the AMC
decode daemon consumes the compressed packs (numpy dequantization mirrors the CUDA
kernels) and reports the task metrics.

```
python3 infocom_evals/signal_detection_experiments/compression_rd/run_rd_eval.py \
    [--codecs none sc16 bfp12 bfp8] [--snrs 30 20 12 6] [--keep-scratch]
```

Prereqs: app rebuilt with the compression operator (`rebuild_demo_container_app.sh`),
staircase captures in `~/Documents/holoscan_waveform_generation/composition/composites/`
(`snr_single_<db>db.sigmf-data`, per-signal SNR labels, burst-structured), and the
`.venv-ml` python for the daemon.

Columns in `results.csv`:

- `ratio` — measured compressed ratio vs cf32 (from the daemon's per-annotation
  logical/stored byte accounting; matches the operator's own log).
- `stored_mb` — actual bytes on disk for the run's snippet packs.
- `ber_whole` / `ber_attempted` — PN9 BER through the classifier-gated decode
  (whole charges lost bits at 1.0).
- `chber` — oracle-routed channel BER (known branch + known symbol rate): the
  decode floor with perfect classification. Codec damage shows here first.
- `acc_*` — per-model classification accuracy on the dequantized snips.
  NOTE: each capture is only ~0.1 s (4 frames), so absolute accuracies are
  small-N; read them codec-relative (against the `none` row at the same SNR).

Codec expectations (theory): sc16 = 2.0× (~96 dB quant SNR, decode-transparent),
bfp12 = 2.65× (~66 dB), bfp8 = 3.97× (~42 dB — first codec where high-SNR rungs
should show measurable chBER movement).

Scratch: `/tmp/usrp_spectrograms/rd_eval/<codec>_<snr>db{,_daemon}` (container
writes are root-owned; the harness sudo-cleans them).
