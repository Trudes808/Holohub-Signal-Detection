# dev/ — one-off diagnostics kept as breadcrumbs

These are **not** part of the eval pipeline. Each was written to root-cause one specific
decode bug during development and is kept because it documents *how* the bug was found and
gives a minimal reproducer if it ever regresses. They hardcode paths and expect the
author's machine; treat them as reference, not as tools.

Most load a single waveform and decode it under varying conditions, printing BER, so a
change in the decode path can be bisected quickly.

| script | what it was for |
|---|---|
| `dev_validate_decodepath.m` | end-to-end sanity: does a known waveform decode at BER 0 with no channel? |
| `dev_test_identity.m` | identity check — decoder fed its own TX waveform |
| `dev_debug_qpsk.m`, `dev_debug_standards.m` | first-pass failures on single-carrier vs standards decoders |
| `dev_diag_5g.m`, `dev_diag_5g_bt.m` | 5G downlink decode (DM-RS timing, MMSE equalisation) |
| `dev_test_cfo_da.m`, `dev_measure_cfo.m` | building/validating the data-aided CFO estimator |
| `dev_test_ble.m`, `dev_test_bt_lpf.m`, `dev_test_acquire.m` | Bluetooth LE/BR decode + low-pass and acquisition behaviour |
| `dev_debug_bt_snippet.m` | **the Bluetooth snippet bug** — snippet vs genie extraction, which found that the ±2% time guard misaligned the sync-free BT receivers |
| `dev_test_fulllen.m` | full-length vs slot-truncated input |
| `dev_envelope.m`, `dev_presence.m` | power-envelope checks that established waveforms *tile* their slot |
| `dev_setup.m` | scratch path/setup helper |

The findings these produced are written up in the parent `README.md` under
"Decode fixes (the load-bearing details)". Start there; come here only to reproduce.
