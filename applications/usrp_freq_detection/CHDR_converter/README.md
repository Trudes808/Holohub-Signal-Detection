
# CHDR Converter Operator

## Overview

An operator to load CHDR packets and format them for FFT processing

## Description

The CHDR Converter takes in Advanced Network Operator bursts and does
a few things:

1. Accumulates a configurable number of packets
2. Parses CHDR packets and extracts the contained RF data
3. Converts from network byte-order to little-endian
4. Casts incoming data from 16-bit complex integer to 32-bit complex float
   (scaling to -1.0 thru +1.0)

## Requirements

- [ANO](https://github.com/nvidia-holoscan/holohub/tree/main/operators/advanced_network)
  (and associated hardware)
- [MatX](https://github.com/NVIDIA/MatX) (dependency - assumed to be installed on system)

## Configuration

```yaml
chdr_converter:
  interface_name: sdr_data
  num_complex_samples_per_packet: 1024
  num_packets_per_fft: 20
  num_ffts_per_batch: 125
  num_simul_batches: 2
  num_channels: 2
  log_packets: false
  log_data: false
```

- `interface_name`: Name of the RX port from the advanced_network config
- `num_complex_samples_per_packet`: Number of complex samples contained in every CHDR data packet
- `num_packets_per_fft`: Number of packets you'd like to process in each FFT
- `num_ffts_per_batch`: Number of FFTs you'd like to perform in one downstream run
- `num_simul_batches`: Number of simultaneous batches to process (ping-pong style)
- `num_channels`: Number of channels to support
- `log_packets`: Log the first packet of a burst to console
- `log_data`: Log the complex floating point data for the first packet of a burst to console

These parameters impact the shape of the data tensor that is assembled for downstream
processing. In the example above, the CHDR converter would emit a 125x20480 sample
`tensor_t`.
