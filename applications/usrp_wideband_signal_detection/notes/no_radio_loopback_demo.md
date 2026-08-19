# No-radio live demo: cable-loopback replay of pycodec composites

Goal: run the FULL live pipeline (DPDK ingest → detect → snip → classify →
decode → dashboard) with **no USRP**, by replaying a pycodec composite as
bit-exact X410 CHDR/UDP packets over a QSFP cable between the Spark's own
two ConnectX-7 NICs. Everything below except the final `tcpreplay` was
prepared and verified offline on 2026-08-17; the wire step needs the cable.

## Lab state observed 2026-08-17 (collaborator working on radios)

- `enp1s0f0np0` (DPDK data port, 0000:01:00.0): **NO-CARRIER** — QSFP to the
  X410 removed.
- `enp1s0f1np1` (was UHD control, 192.168.21.1): **re-addressed to
  192.168.6.10/24 and link-up to collaborator gear — do not touch / do not
  run `after_reboot.sh` IP enforcement against it without coordinating.**
- `enP2p1s0f0np0` / `enP2p1s0f1np1` (second CX7, 0002:01:00.x): down, no
  cables — these are the free ports for the loopback.

## One-time prep (already done)

- `tcpreplay` installed on the host.
- Replay pcaps generated from the SigMF composites with
  `applications/usrp_freq_detection/replay_rx_to_buff.py`
  (Spark addressing: dst-mac `4c:bb:47:2c:45:13` = enp1s0f0np0,
  udp 49153→1234 = the app's flow rule; sender iface `enP2p1s0f0np0`):
  - `~/Documents/holoscan_waveform_generation/composition/composites/comprehensive_4class_py.pcap`
    (4-class AMC composite: BPSK/16QAM/4FSK/OFDM + GRCON text entry)
  - `~/Documents/holoscan_waveform_generation/composition/composites/snr_staircase_4class.pcap`
    (30→6 dB staircase: whole-vs-attempted BER divergence on the dashboard)

## In-person bring-up (VALIDATED 2026-08-18 — this is the working recipe)

Cabling used: QSFP between the first CX7's two ports —
`enp1s0f1np1` (sender) <-> `enp1s0f0np0` (DPDK receiver). The second CX7
(`enP2p1s0*`) works too; pass its name as the sender everywhere below.

```bash
cd ~/Documents/Holohub-Signal-Detection/applications/usrp_wideband_signal_detection

# 0. X access for the container's window (resets on login/lock! rerun if the
#    window ever fails to appear — it fails SILENTLY otherwise)
DISPLAY=:1 xhost +local:

# 1. sender port up at jumbo MTU
sudo ip link set enp1s0f1np1 up mtu 9000

# 2. the app: snipper + visualizer variant, replay rate pinned via env
sudo docker exec usrp_x410_sig_det_sat3737 bash -lc \
  "pkill -f '(^|/)usrp_wideband_signal_detection( |\$)' || true"
sudo docker exec -d -e DISPLAY=:1 -e USRP_SAMPLE_RATE_HZ=245760000 \
  -e USRP_CENTER_FREQ_HZ=2400000000 usrp_x410_sig_det_sat3737 bash -lc \
  "mkdir -p /tmp/xdg-runtime-root && chmod 700 /tmp/xdg-runtime-root && \
   export XDG_RUNTIME_DIR=/tmp/xdg-runtime-root && \
   cd /workspace/holohub/build/usrp_wideband_signal_detection/applications/usrp_wideband_signal_detection && \
   exec ./usrp_wideband_signal_detection config_snipper_viz_demo.yaml \
   > /workspace/spectrograms/demo_app.log 2>&1"

# 3. the conductor (watches DEMO CONTROLS; starts/loops tcpreplay per the SNR
#    dropdown; restarts the app on detector switches)
cd infocom_evals/pycodec_e2e
sudo bash -c '(python3 demo_conductor.py --iface enp1s0f1np1 \
  --rate-hz 245760000 --display :1 > /tmp/usrp_spectrograms/conductor.log 2>&1 &)'

# 4. the decode daemon (LIVE DECODE panel; --no-amc = blind cascade until the
#    wire-path classifier gap is closed, then drop the flag for gate routing)
(~/Documents/holoscan_waveform_generation/.venv-ml/bin/python rt_decode_daemon.py \
  --snips /tmp/usrp_spectrograms/snippets --no-amc \
  --metrics-out /tmp/usrp_spectrograms/rt_metrics.json \
  > /tmp/usrp_spectrograms/daemon_live.log 2>&1 &)

# 5. snippet janitor (looped replay writes ~500 MB/s of snippets at real time)
sudo bash -c '(while true; do \
  find /tmp/usrp_spectrograms/snippets -name "snip_pack*" -mmin +5 -delete 2>/dev/null; \
  sleep 30; done > /dev/null 2>&1 &)'
```

Teardown: `sudo pkill -f '^tcpreplay'`, `sudo pkill -f '^python3 demo_conductor'`,
`pkill -f 'rt_decode_daemon.py --snips'` (run each as its OWN command — a
compound line whose text contains the pattern kills your own shell), and the
app pkill from step 2.

### Hard-won gotchas (2026-08-18 bring-up)

- **pcap CHDR header line must be 64 B** for this app's 42/64/4096 NIC split
  (matches the real X410 CG_400 packets). The tool's old 32 B default silently
  ZEROED THE LAST 8 SAMPLES OF EVERY PACKET: ~1e-3 BER with packet-grid
  structure, an impulsive floor that widened/merged detection boxes, and
  classifier garbage — while spectrograms and masks looked fine.
  `replay_rx_to_buff.py` now defaults to 64 B (`CHDR_HEADER_LINE_BYTES` env
  to override); all committed pcap recipes regenerate correctly.
- **`xhost +local:` resets** on login/lock — the app then runs fine but
  windowless (no error in the log; `visualization.enable: false` in the plain
  snipper config does the same, which is why `config_snipper_viz_demo.yaml`
  exists).
- The AMC classifier currently misreads wire-path snips even when decode is
  clean (domain gap under investigation) — hence `--no-amc` above.

## Dashboard panels (final demo layout)

- Sidebar LIVE DECODE: frames/CRC, BER att (+ whole when truth-scored),
  windowed **frames/s** + **snips decoded %** (collapse immediately on SNR
  changes — the cumulative numbers and survivor-averaged BER do not), inst-BER
  sparkline, per-mod bars, CLASSIFIER block (3 models, live accuracy vs
  decoded truth, '>' = gate).
- Footer left: PN9 BER strip. Footer right: **CLASSIFIER x SNR** table —
  one row per SNR selection visited: gate acc per class, decoded BER,
  frames, snippets written, GB stored; header shows the data-reduction
  headline ("stored X GB vs Y GB full-rate = Zx less", baseline = daemon
  uptime x stream rate x 8 B cf32). Note: the clean composite is
  artificially signal-dense, so the reduction factor is modest there and
  grows at low SNR / on sparse real spectrum.

## Dashboard demo controls (gate / SNR / detector from the UI)

The windowed dashboard now has a **DEMO CONTROLS** panel (above Display
Controls; appears when the config sets `visualization.renderer.demo_control_json`,
wired in the 3 demo configs to `/workspace/spectrograms/demo_control.json`
= host `/tmp/usrp_spectrograms/demo_control.json`). Selections are written
atomically to that JSON and honored by two watchers:

- **Gate** (VT-CNN2 / ResNet1D / T-PRIME): `rt_decode_daemon.py` applies it
  live within one poll (~0.5 s); the CLASSIFIER panel's `>` marker follows.
  Validated offline 2026-08-17 (gate flip mid-sweep).
- **SNR** (Clean / 30 / 20 / 15 / 12 / 9 / 6 / Stair): run
  `sudo python3 infocom_evals/pycodec_e2e/demo_conductor.py` on the host —
  it switches which pcap tcpreplay loops (~1 s gap). Per-SNR pcaps are
  pregenerated (`snr_single_<db>db.pcap`, noise floor referenced to the
  30 dB step so the detector floor holds across switches; "Clean" = the
  noise-free 4-class composite, so expect the dynamic floor to dip and
  re-learn for ~1 s on that transition). Conductor dry-run validated
  (`--dry`); the wire step needs the loopback cable.
- **Detector** (CoherentPower / CUDA-DINO / DINO-FT (M2_dr)): the conductor
  restarts the app with the other config (~15 s blink; selections persist —
  the panel re-reads the control file on startup). NOTE: cuda_dino needs its
  TorchScript weights present in the container; verify before demoing that
  button, and check the conductor's XAUTHORITY matches the desktop session
  for the windowed relaunch.
- **DINO-FT (M2_dr)** is Sage's fine-tuned DINOv3 segmenter as a real-time
  operator (`operators/finetuned_dino_detector/`, config
  `config_dino_finetuned_viz_demo.yaml`). Needs
  `dino_fine_tuning/weights/finetuned_dino_m2_dr_bf16.ts` + `.meta.json` in
  the checkout (bind-mounted into the container; sha256s in the handoff doc
  `~/Downloads/finetuned_dino_realtime_handoff.md`). It taps raw IQ and runs
  its own 1024-pt FFT (240 kHz/bin — the training physics; replay's
  245.76 MS/s is exactly the top of its rate-invariant training range).
  GB10 budget: ~10.4 ms/tile ⇒ `emit_stride: 16` in the demo config keeps
  full-rate 240 kpps replay with a fresh mask every ~340 ms. Unlike the
  zero-shot cuda_dino it needs no coherence calibration and masks individual
  bursts (finer than coherent_power's persistence boxes).

Demo-day order: app (windowed, on the desktop session) → conductor →
daemon → click things.

## Notes / gotchas

- `replay_rx_to_buff.py --live` (raw socket) cannot sustain 245.76 Msps —
  use tcpreplay with the pregenerated pcaps.
- Regenerate a pcap after rebuilding a composite (same command as prep;
  it also rewrites the sidecar).
- Restoring the REAL radio later: coordinate the control-port IP first
  (collaborator is on 192.168.6.x), then `sudo ./bash_scripts/after_reboot.sh`
  re-enforces the data-port IP and `sudo ./bash_scripts/check_radio_topology.sh`
  validates the two-cable topology before `run_live_demo.sh`.
- Real-time for 245.76 Msps sc16 is ≈ 8.1 Gbps; the CX7 link is 100/200 G so
  rate is not a wire concern. If the sender host thread can't hold 240 kpps,
  the pipeline tolerates slower-than-real-time input (frames just arrive
  less often) — start plain, then add `--pps 240000`.
