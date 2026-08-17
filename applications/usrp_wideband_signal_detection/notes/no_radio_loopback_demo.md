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

## In-person bring-up (one cable, ~5 commands)

1. **Cable**: QSFP between `enP2p1s0f0np0` (sender, kernel) and
   `enp1s0f0np0` (receiver, DPDK). Verify both show LOWER_UP:
   `ip -br link show | grep -E "enp1s0f0np0|enP2p1s0f0np0"`.
2. **Sender MTU** (frames are ~4170 B):
   `sudo ip link set enP2p1s0f0np0 up mtu 9000`
3. **Stream-params sidecar** so the app auto-adopts the replayed rate/center
   (the run wrapper consumes + deletes it at each launch):
   ```bash
   echo '{"sample_rate_hz": 245760000.0, "center_freq_hz": 2400000000.0, "source": "replay"}' \
     | sudo tee /tmp/usrp_stream_params.json
   ```
4. **App** (from the app dir; snipper config for the decode demo, or the
   dynamic single-channel config for detection-only):
   `sudo ./bash_scripts/run_torchscript_performance_test.sh config_signal_snipper_single_channel.yaml`
   (loopback needs no radio topology: `SKIP_TOPOLOGY_CHECK=1` if a wrapper
   insists on it.)
5. **Replay** (generator-printed recipe; `--pps 240000` = 1024-sample packets
   at exactly 245.76 Msps, `--loop 0` sustains it, Ctrl-C stops):
   `sudo tcpreplay --preload-pcap --loop 0 --pps 240000 -i enP2p1s0f0np0 ~/Documents/holoscan_waveform_generation/composition/composites/comprehensive_4class_py.pcap`
   (pcap timestamps are written at true spacing, so omitting `--pps` also
   replays at rate; swap in `snr_staircase_4class.pcap` for the BER-degradation
   staircase.)
6. **Decode daemon** (classifier-routed, live metrics for the dashboard):
   ```bash
   cd infocom_evals/pycodec_e2e
   ~/Documents/holoscan_waveform_generation/.venv-ml/bin/python rt_decode_daemon.py \
     --snips /tmp/usrp_spectrograms/<run>/snippets \
     --truth-meta ~/Documents/holoscan_waveform_generation/composition/composites/comprehensive_4class_py.sigmf-meta \
     --metrics-out /tmp/usrp_spectrograms/rt_metrics.json
   ```
   (drop `--truth-meta` for a fully blind demo — accuracy shows `--`,
   whole BER is omitted, everything else works.)

Expected: the waterfall shows the composite's actual slot structure (much
prettier than ambient spectrum), detection boxes on every burst, decode
markers + CLASSIFIER panel + dual BER live, GRCON text on the ticker.

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
