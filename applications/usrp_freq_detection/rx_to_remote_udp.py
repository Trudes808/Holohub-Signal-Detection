#!/usr/bin/env python3
#
# Copyright 2025 Ettus Research, a National Instruments Brand
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
"""Stream IQ data from a USRP to a remote destination.

The script is designed to stream IQ (In-phase and Quadrature) data from a
USRP device (connected to a host computer) to a remote destination over a
network. The remote destination can be different from the host computer.
The example allows users to configure various parameters such as frequency,
sampling rate, gain, and destination details for the streaming process.
Data is streamed until the configured duration expires or the user interrupts
the process (Ctrl-C).

This script is useful for applications where IQ data needs to be streamed
from a USRP device to a remote server or application for further processing,
such as signal analysis, demodulation, or storage. It is particularly suited
for networked SDR setups.

The hardware setup includes
- a USRP device
- a host computer connected to the USRP device
- a remote destination (IP address and port) where the IQ data will be sent
- a network connection between the host computer, USRP device and the remote
destination

Example Usage:
rx_to_remote_udp.py --args addr=192.168.10.2 --rate 1e6 --freq 2.4e9 --gain 10
                    --duration 10 --channels 0 1 --dest-addr 192.168.10.100
                    --dest-port 12345 12346
"""

import argparse
import sys
import time
from typing import Optional

import uhd

INIT_DELAY = 0.05  # 50mS initial delay before receive


class Config(argparse.Namespace):
    """Configuration for remote RX streaming."""
    args: str
    rate: float
    freq: list[float]
    gain: float
    duration: Optional[float]
    channels: list[int]
    dest_addr: str
    dest_port: list[int]
    adapter: Optional[str]
    dest_mac_addr: Optional[str]
    keep_hdr: bool
    spp: Optional[int]
    mtu: Optional[int]


def parse_args() -> Config:
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    parser.add_argument(
        "-a", "--args", default="",
        help="""specifies the USRP device arguments, which holds
        multiple key value pairs separated by commas
        (e.g., addr=192.168.40.2,type=x300) [default = "%(default)s"].""",
    )
    parser.add_argument(
        "-r", "--rate", type=float, default=1e6,
        help="specifies the sample rate in samples/sec [default: %(default)s].",
    )
    parser.add_argument(
        "-f", "--freq", type=float, nargs="+", required=True,
        help=(
            "specifies the center frequency in Hz. Provide one value to apply to all "
            "selected channels, or one value per selected channel [input is required]."
        ),
    )
    parser.add_argument(
        "-g", "--gain", type=float, default=0.0,
        help="specifies the RX gain in dB [default: %(default)s].",
    )
    parser.add_argument(
        "-d", "--duration", default=None, type=float,
        help="""specifies the stream duration in seconds, leave out to stream until stopped
        [default: %(default)s]."""
    )
    parser.add_argument(
        "-c", "--channels", default=[0], nargs="+", type=int,
        help="""specifies the RX channel(s) to use (specify "0", "1", "0 1", etc)
        [default: %(default)s].""",
    )
    parser.add_argument(
        "-i", "--dest-addr", type=str, required=True,
        help="specifies the remote destination IP address [input is required].",
    )
    parser.add_argument(
        "-p", "--dest-port", nargs="+", type=int, required=True,
        help="specifies the remote destination UDP port(s) [input is required].",
    )
    parser.add_argument(
        "--adapter", type=str,
        help="""specifies the adapter to use for remote streaming (e.g. 'sfp0')""",
    )
    parser.add_argument(
        "--dest-mac-addr",
        help="""specifies the destination MAC address in the format 01:a2:4f:6d:7e:5f.
        If this argument is not used, the USRP device will use ARP to identify
        the MAC address.""",
    )
    parser.add_argument(
        "--keep-hdr",
        action="store_true",
        help="""specify this argument to keep CHDR headers on outgoing packets. If not
            specified, the headers will be stripped from the packets""",
    )
    parser.add_argument(
        "--spp", type=int,
        help="""Specifies the number of samples per packet for all channels.
        If not specified, the USRP will use the available MTU size.""",
    )
    parser.add_argument(
        "--mtu", type=int,
        help="""Specifies the MTU size for all channels.
        If not specified, the USRP will use the default MTU size.""",
    )
    return parser.parse_args()


def get_stream_cmd(usrp, rate, duration, start_time=None):
    """Generate a stream command based on rate and duration."""
    if duration:
        stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.num_done)
        stream_cmd.num_samps = int(rate * duration)
    else:
        stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
    if start_time is None:
        stream_cmd.stream_now = True
    else:
        stream_cmd.stream_now = False
        stream_cmd.time_spec = start_time
    return stream_cmd


def check_channels(usrp, args):
    """Check that the device has sufficient RX channels available."""
    # Check that each channel specified is less than the number of total number of rx channels
    # the device can support
    channels = args.channels
    dev_rx_channels = usrp.get_rx_num_channels()
    if not all(map((lambda chan: chan < dev_rx_channels), channels)):
        print("Invalid channel(s) specified.")
        return []
    return channels


def expand_per_channel(values: list[float], channels: list[int], label: str) -> list[float]:
    """Broadcast one value to all channels or validate one value per channel."""
    if len(values) == 1:
        return values * len(channels)
    if len(values) != len(channels):
        raise ValueError(
            f"Number of {label} values must be 1 or match the number of channels. "
            f"Got {len(values)} {label} value(s) for {len(channels)} channel(s)."
        )
    return values


def main():
    """Run remote Rx."""
    args = parse_args()
    usrp = uhd.usrp.MultiUSRP(args.args)
    channels = check_channels(usrp, args)
    if not channels:
        return False
    if len(channels) != len(args.dest_port):
        print("Number of channels must match number of ports.")
        return False
    try:
        freqs = expand_per_channel(args.freq, channels, "frequency")
    except ValueError as exc:
        print(exc)
        return False
    print("Selected RX channels: {}.".format(", ".join(str(ch) for ch in channels)))
    if args.rate:
        print(f"Requesting sampling rate {args.rate/1e6} Msps...")
        for chan in channels:
            usrp.set_rx_rate(args.rate, chan)
    actual_rates = [usrp.get_rx_rate(chan) for chan in channels]
    for chan, actual_rate in zip(channels, actual_rates):
        print(f"Using sampling rate for channel {chan}: {actual_rate/1e6} Msps.")
    for chan, freq in zip(channels, freqs):
        print(f"Requesting center frequency for channel {chan}: {freq/1e6} MHz...")
        usrp.set_rx_freq(freq, chan)
    actual_freqs = [usrp.get_rx_freq(chan) for chan in channels]
    for chan, actual_freq in zip(channels, actual_freqs):
        print(f"Actual center frequency for channel {chan}: {actual_freq/1e6} MHz.")
    print(f"Requesting gain {args.gain} dB...")
    for chan in channels:
        usrp.set_rx_gain(args.gain, chan)
    print(f"Actual gain: {usrp.get_rx_gain(channels[0])} dB.")
    print("Generating RX streamer object...")
    rx_streamers = []
    for chx, chan in enumerate(channels):
        stream_args = uhd.usrp.StreamArgs("sc16", "sc16")
        stream_args.channels = [chan]
        stream_args.args = (
            f"dest_addr={args.dest_addr},dest_port={args.dest_port[chx]},"
            f"stream_mode={'full_packet' if args.keep_hdr else 'raw_payload'}"
            + (f",adapter={args.adapter}" if args.adapter else "")
            + (f",dest_mac_addr={args.dest_mac_addr}" if args.dest_mac_addr else "")
            + (f",mtu={args.mtu}" if args.mtu else "")
        )
        print(f"Stream args for channel {chan}:\n{stream_args.args}")
        rx_streamers.append(usrp.get_rx_stream(stream_args))
        if args.spp:
            print(f"Configuring samples per packet for channel {chan} to {args.spp}...")
            usrp.get_radio_control(chan).set_properties(f"spp={args.spp}", chan % 2)
    print("Starting stream(s)...")
    start_time = (
        uhd.types.TimeSpec(usrp.get_time_now().get_real_secs() + INIT_DELAY)
        if len(channels) > 1
        else None
    )
    for rx_streamer in rx_streamers:
        stream_cmd = get_stream_cmd(usrp, actual_rates[0], args.duration, start_time)
        rx_streamer.issue_stream_cmd(stream_cmd)
    print("Stream started. Press Ctrl-C to stop.")
    timeout = time.monotonic() + args.duration if args.duration else None
    if timeout and len(channels) > 1:
        timeout += INIT_DELAY
    try:
        while timeout is None or time.monotonic() < timeout:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    print("Stopping stream...")
    for rx_streamer in rx_streamers:
        rx_streamer.issue_stream_cmd(uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont))
    print("Streaming complete. Exiting.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
