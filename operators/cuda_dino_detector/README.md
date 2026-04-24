<!--
SPDX-FileCopyrightText: 2026 National Instruments Corporation

SPDX-License-Identifier: Apache-2.0
-->
# CUDA DINO Detector Operator

## Overview

This operator is the new dedicated implementation surface for the custom CUDA DINO detector path.

It exists so the CUDA detector migration can proceed in parallel with the existing `dinov3_signal_detector` operator instead of continuing to overload the validated reference path.

The current scaffold only establishes the operator boundary, configuration surface, and input contract. The validated detector stages will be ported here incrementally from the offline performance validator.

This operator is intended to be performance-first rather than architecture-faithful to the existing C++ implementation. The current chunked C++ path is the parity reference, but the CUDA implementation is allowed to use a different execution organization if it preserves detector outputs and improves throughput.

## Performance Principles

1. Non-debug runs should keep intermediate products on device and avoid host copies.
2. Debug artifact export should be opt-in and limited to a selected chunk or tile or other focused surface.
3. The CUDA path does not have to preserve the current C++ chunking organization if a more efficient strategy preserves parity.
4. The DINO runtime should be driven by a token-budget-aware strategy rather than by historical chunk boundaries alone.

## Candidate Execution Strategies

The operator should support experimentation with multiple execution strategies behind one stable output contract.

1. `reference_chunks`: match the current validator chunk plan exactly for parity work.
2. `adaptive_tiles`: tile the full corrected spectrogram on GPU using a token-budget-aware tile planner.
3. `coarse_to_fine`: run a coarse full-frame or reduced-grid pass to find candidate regions, then refine only selected windows at higher resolution.

The default implementation should begin with `reference_chunks` for parity and then keep the door open to `adaptive_tiles` or `coarse_to_fine` when measurements justify it.

## Initial Scope

The intended implementation order is:

1. shared CUDA workspace and buffer reuse
2. full-frame correction and chunk staging
3. structure-tensor gate generation
4. raw DINO energy and positional deweighting
5. residual-veto hybrid support
6. chunk projection and merge
7. operator debug artifact hooks

## I/O Contract

- Input: `tuple<tensor_t<complex, 2>, cudaStream_t>`
- Output: none currently

## Current Status

- Build scaffold only
- No production detector output yet
- Intended backend modes: `reference`, `cuda_partial`, `cuda_full_detector`
- Intended execution strategies: `reference_chunks`, `adaptive_tiles`, `coarse_to_fine`
- Intended debug behavior: `debug_mode=false` should imply no unnecessary host copies