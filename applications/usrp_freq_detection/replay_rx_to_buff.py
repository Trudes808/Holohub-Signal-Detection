#!/usr/bin/env python3
#
# SPDX-License-Identifier: Apache-2.0
#
"""Replay a SigMF recording onto the live CHDR/UDP ingest path.

This is the offline-replay counterpart to ``rx_to_remote_udp.py``. Instead of a
USRP generating IQ packets, it reads a SigMF recording (``*.sigmf-data`` +
``*.sigmf-meta``), frames the samples into the **exact same CHDR-over-UDP
packets** an X410 emits with ``--keep-hdr``, and writes them to a ``.pcap`` (or
sends them live on a raw socket). Replaying that pcap onto the DPDK NIC drives
the unchanged ``usrp_wideband_signal_detection`` app through its real
``chdr_converter -> FFT -> detector`` pipeline.

Wire format (decoded from a live X410 UC_200 capture; see
``infocom_evals/signal_detection_experiments/chdr_reference_header_bytes.txt``):

    [14B Ethernet][20B IPv4][8B UDP][32B CHDR header line][4096B sc16 payload]

  * CHDR header u64 (little-endian): dst_epid=3, length=4128,
    seq_num (+1/packet), num_mdata=0, pkt_type=7 (DATA_WITH_TS).
  * 8-byte timestamp, then 16 bytes of zero padding -> 32-byte header line
    (CHDR_W=256). Payload begins at offset 32.
  * Payload: interleaved little-endian int16 I/Q, 1024 complex samples/packet.

The advanced_network manager splits each packet into 42 / 64 / remaining
segments by byte offset, so only the byte layout has to match; the converter
ignores CHDR header *contents*.

Egress (default = pcap):
    replay_rx_to_buff.py --sigmf-data capture.sigmf-data --out-pcap replay.pcap
    sudo ip link set ens4f0np0 mtu 9000           # frames are ~4170 B (jumbo)
    sudo tcpreplay -i ens4f0np0 --mbps=2000 replay.pcap

Live raw-socket egress (no pcap; slower, needs CAP_NET_RAW):
    sudo replay_rx_to_buff.py --sigmf-data capture.sigmf-data --live --src-iface ens4f0np0

Defaults target the committed loopback-replay topology: DPDK on 0000:a2:00.1,
so packets are addressed to that port's MAC (e0:9d:73:e0:5b:6b), UDP 49153->1234.
"""

import argparse
import json
import os
import socket
import struct
import sys
import time
from typing import Optional

import numpy as np

# --- Constants matching the live CHDR/config contract ----------------------
# Sidecar consumed by the wideband run wrapper so the receiving pipeline auto-adopts the replayed
# stream's rate/center (keep in sync with rx_to_remote_udp.py + run_torchscript_performance_test.sh).
STREAM_PARAMS_SIDECAR = "/tmp/usrp_stream_params.json"
SAMPLES_PER_PACKET = 1024          # chdr_converter.num_complex_samples_per_packet
BYTES_PER_SAMPLE = 4               # sc16: int16 I + int16 Q
PAYLOAD_BYTES = SAMPLES_PER_PACKET * BYTES_PER_SAMPLE      # 4096
CHDR_HEADER_LINE_BYTES = 32        # CHDR_W=256: 8B hdr + 8B timestamp + 16B pad
CHDR_LENGTH = CHDR_HEADER_LINE_BYTES + PAYLOAD_BYTES        # 4128
CHDR_PKT_TYPE_DATA_WITH_TS = 0x7
CHDR_DST_EPID = 3
ETHERTYPE_IPV4 = 0x0800
IPPROTO_UDP = 17

# SigMF core:datatype -> numpy dtype (little-endian variants seen in this repo)
SIGMF_DTYPES = {
    "cf32_le": ("complex", np.dtype("<c8")),
    "cf64_le": ("complex", np.dtype("<c16")),
    "ci16_le": ("int16", np.dtype("<i2")),   # interleaved int16 I,Q
}


class Config(argparse.Namespace):
    sigmf_data: str
    sigmf_meta: Optional[str]
    out_pcap: Optional[str]
    live: bool
    src_iface: Optional[str]
    dst_mac: str
    src_mac: str
    dst_ip: str
    src_ip: str
    udp_src: int
    udp_dst: int
    scale: float
    start_sample: int
    max_samples: Optional[int]
    info: bool


def parse_args() -> Config:
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__
    )
    p.add_argument("--sigmf-data", required=True,
                   help="Path to the .sigmf-data file (meta inferred if --sigmf-meta omitted).")
    p.add_argument("--sigmf-meta", default=None,
                   help="Path to the .sigmf-meta file [default: derived from --sigmf-data].")
    p.add_argument("--out-pcap", default=None,
                   help="Write frames to this pcap (default mode). Replay with tcpreplay.")
    p.add_argument("--live", action="store_true",
                   help="Send frames live on a raw AF_PACKET socket instead of writing a pcap.")
    p.add_argument("--src-iface", default="ens4f0np0",
                   help="Egress interface for --live [default: %(default)s].")
    p.add_argument("--dst-mac", default="e0:9d:73:e0:5b:6b",
                   help="Destination MAC = DPDK NIC port MAC [default: %(default)s = 0000:a2:00.1].")
    p.add_argument("--src-mac", default="e0:9d:73:e0:5b:6a",
                   help="Source MAC = sender port MAC [default: %(default)s = 0000:a2:00.0].")
    p.add_argument("--dst-ip", default="192.168.10.51",
                   help="Destination IP (not flow-matched, cosmetic) [default: %(default)s].")
    p.add_argument("--src-ip", default="192.168.10.2",
                   help="Source IP (cosmetic) [default: %(default)s].")
    p.add_argument("--udp-src", type=int, default=49153,
                   help="UDP source port (flow match) [default: %(default)s].")
    p.add_argument("--udp-dst", type=int, default=1234,
                   help="UDP destination port (flow match) [default: %(default)s].")
    p.add_argument("--scale", type=float, default=32767.0,
                   help="Float->int16 scale for cf32/cf64 inputs [default: %(default)s = full-scale].")
    p.add_argument("--gain-db", type=float, default=0.0,
                   help="Replay gain (dB) applied before int16 quantization, to use int16 headroom "
                        "on quiet/high-attenuation recordings. Emulates raising RX gain. "
                        "Clips if too high [default: %(default)s].")
    p.add_argument("--start-sample", type=int, default=0,
                   help="First complex sample to replay [default: %(default)s].")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Limit number of complex samples replayed [default: all].")
    p.add_argument("--info", action="store_true",
                   help="Parse and print the SigMF/derived parameters, then exit.")
    return p.parse_args(namespace=Config())


def infer_meta_path(data_path: str) -> str:
    if data_path.endswith(".sigmf-data"):
        return data_path[: -len(".sigmf-data")] + ".sigmf-meta"
    return data_path + ".sigmf-meta"


def load_sigmf(data_path: str, meta_path: Optional[str]):
    """Return (iq_int16_interleaved: np.ndarray[int16], meta: dict)."""
    meta_path = meta_path or infer_meta_path(data_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)
    glob = meta.get("global", {})
    datatype = glob.get("core:datatype")
    if datatype not in SIGMF_DTYPES:
        raise ValueError(
            f"Unsupported core:datatype {datatype!r}; supported: {sorted(SIGMF_DTYPES)}"
        )
    num_channels = int(glob.get("core:num_channels", 1))
    if num_channels != 1:
        raise ValueError(f"Only single-channel SigMF is supported (got num_channels={num_channels}).")
    return datatype, meta


def read_samples_as_sc16(data_path: str, datatype: str, scale: float,
                         start_sample: int, max_samples: Optional[int]) -> np.ndarray:
    """Read the .sigmf-data and return a flat int16 array of interleaved I,Q (sc16)."""
    kind, dt = SIGMF_DTYPES[datatype]
    if kind == "complex":
        count = -1 if max_samples is None else max_samples
        offset = start_sample * dt.itemsize
        data = np.fromfile(data_path, dtype=dt, count=count, offset=offset)
        # complex float -> interleaved int16, scaled and clipped
        iq = np.empty(data.size * 2, dtype=np.float32)
        iq[0::2] = np.real(data)
        iq[1::2] = np.imag(data)
        iq = np.clip(np.round(iq * scale), -32768, 32767).astype("<i2")
        return iq
    else:  # ci16_le: already interleaved int16 I,Q
        count = -1 if max_samples is None else max_samples * 2
        offset = start_sample * 2 * dt.itemsize
        raw = np.fromfile(data_path, dtype=dt, count=count, offset=offset)
        factor = scale / 32767.0  # scale=32767 -> identity; >32767 -> replay gain
        if abs(factor - 1.0) < 1e-9:
            return raw.astype("<i2")
        return np.clip(np.round(raw.astype(np.float32) * factor), -32768, 32767).astype("<i2")


def build_chdr_header_line(seq_num: int, timestamp: int) -> bytes:
    header = (
        (CHDR_DST_EPID & 0xFFFF)
        | ((CHDR_LENGTH & 0xFFFF) << 16)
        | ((seq_num & 0xFFFF) << 32)
        | (0 << 48)                              # num_mdata
        | ((CHDR_PKT_TYPE_DATA_WITH_TS & 0x7) << 53)
        # eov(56)=0, eob(57)=0, vc(63:58)=0
    )
    return struct.pack("<Q", header) + struct.pack("<Q", timestamp & 0xFFFFFFFFFFFFFFFF) + b"\x00" * 16


def mac_to_bytes(mac: str) -> bytes:
    return bytes(int(b, 16) for b in mac.split(":"))


def ip_checksum(header: bytes) -> int:
    s = 0
    for i in range(0, len(header), 2):
        s += (header[i] << 8) + header[i + 1]
    s = (s >> 16) + (s & 0xFFFF)
    s += s >> 16
    return (~s) & 0xFFFF


def build_frame(chdr_packet: bytes, cfg: Config, ip_id: int) -> bytes:
    eth = mac_to_bytes(cfg.dst_mac) + mac_to_bytes(cfg.src_mac) + struct.pack("!H", ETHERTYPE_IPV4)
    udp_len = 8 + len(chdr_packet)
    udp = struct.pack("!HHHH", cfg.udp_src, cfg.udp_dst, udp_len, 0)  # checksum 0 (optional for IPv4)
    total_len = 20 + udp_len
    ip_no_csum = struct.pack(
        "!BBHHHBBH4s4s",
        0x45, 0x00, total_len, ip_id & 0xFFFF, 0x4000, 64, IPPROTO_UDP, 0,
        socket.inet_aton(cfg.src_ip), socket.inet_aton(cfg.dst_ip),
    )
    csum = ip_checksum(ip_no_csum)
    ip = ip_no_csum[:10] + struct.pack("!H", csum) + ip_no_csum[12:]
    return eth + ip + udp + chdr_packet


def iter_frames(iq: np.ndarray, cfg: Config):
    """Yield Ethernet frames, one per CHDR packet of SAMPLES_PER_PACKET samples."""
    int16_per_packet = SAMPLES_PER_PACKET * 2
    total_int16 = iq.size
    seq = 0
    ts = 0
    ip_id = 0
    pos = 0
    while pos < total_int16:
        chunk = iq[pos: pos + int16_per_packet]
        if chunk.size < int16_per_packet:  # zero-pad final partial packet
            chunk = np.concatenate([chunk, np.zeros(int16_per_packet - chunk.size, dtype="<i2")])
        payload = chunk.tobytes()
        chdr = build_chdr_header_line(seq, ts) + payload
        yield build_frame(chdr, cfg, ip_id)
        seq = (seq + 1) & 0xFFFF
        ts = (ts + SAMPLES_PER_PACKET) & 0xFFFFFFFFFFFFFFFF
        ip_id = (ip_id + 1) & 0xFFFF
        pos += int16_per_packet


# --- pcap writer (libpcap format, link type 1 = Ethernet) ------------------
def write_pcap(path: str, frames, packet_interval_s: float) -> int:
    """Write frames with timestamps spaced at packet_interval_s, so native-timing
    tcpreplay reproduces the original sample rate. Index-based timestamps avoid
    float accumulation error over long captures."""
    n = 0
    with open(path, "wb") as f:
        f.write(struct.pack("<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 1))
        for i, frame in enumerate(frames):
            t = i * packet_interval_s
            ts_sec = int(t)
            ts_usec = int((t - ts_sec) * 1e6)
            f.write(struct.pack("<IIII", ts_sec, ts_usec, len(frame), len(frame)))
            f.write(frame)
            n += 1
    return n


def send_live(iface: str, frames) -> int:
    s = socket.socket(socket.AF_PACKET, socket.SOCK_RAW)
    s.bind((iface, 0))
    n = 0
    for frame in frames:
        s.send(frame)
        n += 1
    s.close()
    return n


def main() -> int:
    cfg = parse_args()
    if not os.path.exists(cfg.sigmf_data):
        print(f"SigMF data not found: {cfg.sigmf_data}", file=sys.stderr)
        return 1

    datatype, meta = load_sigmf(cfg.sigmf_data, cfg.sigmf_meta)
    glob = meta.get("global", {})
    sample_rate = float(glob.get("core:sample_rate", 0.0))
    captures = meta.get("captures", [{}])
    center_freq = float(captures[0].get("core:frequency", 0.0)) if captures else 0.0

    # Sidecar so the receiving Holoscan pipeline auto-adopts this recording's rate/center (matches
    # rx_to_remote_udp.py; read by run_torchscript_performance_test.sh into USRP_*_HZ env vars).
    try:
        with open(STREAM_PARAMS_SIDECAR, "w") as sidecar:
            json.dump(
                {"sample_rate_hz": sample_rate, "center_freq_hz": center_freq, "source": "replay_rx_to_buff"},
                sidecar,
            )
        print(f"  stream params sidecar : {STREAM_PARAMS_SIDECAR}")
    except OSError as exc:
        print(f"  Warning: could not write stream params sidecar {STREAM_PARAMS_SIDECAR}: {exc}")

    gain_lin = 10.0 ** (cfg.gain_db / 20.0)
    effective_scale = cfg.scale * gain_lin
    iq = read_samples_as_sc16(cfg.sigmf_data, datatype, effective_scale,
                              cfg.start_sample, cfg.max_samples)
    num_samples = iq.size // 2
    peak_i16 = int(np.max(np.abs(iq.astype(np.int32)))) if iq.size else 0
    clip_pct = float(np.mean(np.abs(iq) >= 32767) * 100.0) if iq.size else 0.0
    eff_bits = (np.log2(peak_i16) if peak_i16 > 0 else 0.0)
    num_packets = (num_samples + SAMPLES_PER_PACKET - 1) // SAMPLES_PER_PACKET
    resolution = round(sample_rate / 20480) if sample_rate else 0
    # Real-time pacing: one packet carries SAMPLES_PER_PACKET samples, so
    # pps = sample_rate / SAMPLES_PER_PACKET reproduces the true sample rate.
    pps = (sample_rate / SAMPLES_PER_PACKET) if sample_rate else 0.0
    packet_interval_s = (SAMPLES_PER_PACKET / sample_rate) if sample_rate else 1e-6
    frame_bytes = 14 + 20 + 8 + CHDR_LENGTH
    wire_mbps = pps * frame_bytes * 8 / 1e6

    print("=== SigMF replay parameters ===")
    print(f"  data            : {cfg.sigmf_data}")
    print(f"  datatype        : {datatype}")
    print(f"  sample_rate     : {sample_rate/1e6:.6g} MSps")
    print(f"  center_freq     : {center_freq/1e6:.6g} MHz")
    print(f"  samples         : {num_samples} (start={cfg.start_sample}"
          f"{', max=' + str(cfg.max_samples) if cfg.max_samples else ''})")
    print(f"  CHDR packets    : {num_packets}  (@ {SAMPLES_PER_PACKET} samples, {CHDR_LENGTH} B each)")
    print(f"  frame size      : {frame_bytes} B (requires jumbo MTU >= ~4200 on egress)")
    print(f"  real-time rate  : {pps:.0f} pps  (~{wire_mbps:.0f} Mbps wire)  -> matches {sample_rate/1e6:.6g} MSps")
    print(f"  int16 headroom  : peak={peak_i16}/32767  (~{eff_bits:.1f} effective bits)  "
          f"clipped={clip_pct:.3f}%  gain_db={cfg.gain_db}")
    if clip_pct > 0.01:
        print(f"  WARNING: {clip_pct:.2f}% of samples clipped at int16 full-scale; lower --gain-db.")
    if 0 < peak_i16 < 64 and cfg.gain_db == 0.0:
        print(f"  NOTE: very low int16 peak ({peak_i16}, ~{eff_bits:.1f} bits) -> coarse quantization. "
              f"If signal-limited, --gain-db helps; on noise-dominated captures it won't improve SNR.")
    print("  --- replay config FFT settings (derive from sample_rate) ---")
    print(f"  fft.span             : {int(sample_rate)}")
    print(f"  fft.reference_span_hz: {int(sample_rate)}")
    print(f"  fft.resolution       : {resolution}   (= round(sample_rate / 20480))")
    print(f"  visualization.renderer.center_frequency_hz: {center_freq}")
    print(f"  dst_mac/udp     : {cfg.dst_mac}  udp {cfg.udp_src}->{cfg.udp_dst}")

    if cfg.info:
        return 0

    frames = iter_frames(iq, cfg)
    if cfg.live:
        print(f"Sending live on {cfg.src_iface} (raw socket)...")
        n = send_live(cfg.src_iface, frames)
        print(f"Sent {n} frames.")
    else:
        out = cfg.out_pcap or (os.path.splitext(cfg.sigmf_data)[0] + ".replay.pcap")
        print(f"Writing pcap: {out}")
        n = write_pcap(out, frames, packet_interval_s)
        print(f"Wrote {n} frames to {out}")
        print("Replay at the real sample rate with:")
        print(f"  sudo ip link set {cfg.src_iface} mtu 9000")
        print(f"  sudo tcpreplay --preload-pcap --loop 0 --pps {pps:.0f} -i {cfg.src_iface} {out}")
        print("  (--pps ties packet rate to the sample rate; --loop 0 sustains it; Ctrl-C to stop.")
        print("   pcap timestamps are also written at the true spacing, so plain")
        print(f"   'tcpreplay --preload-pcap --loop 0 -i {cfg.src_iface} {out}' replays at rate too.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
