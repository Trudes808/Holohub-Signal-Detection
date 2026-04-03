# Signal Detection Demo Visualization Plan

## Objective

Build the visualization path as part of the real C++ Holoscan pipeline, not as a disconnected prototype. The final system must:

- branch directly from the existing spectrogram output in the live app
- render the spectrogram in HoloViz with the same image contract in live and offline modes
- support offline replay from saved `.pgm` spectrogram frames so development does not depend on a connected radio
- leave a clean insertion point for detector overlays once the detector output contract is finalized

## Short Answers

### Can we enable offline replay while building the real viewer?

Yes. That should be a first-class part of the design.

The correct split is:

- live app: branch from `spectrogramOp` into a visualization renderer
- offline harness: replay saved spectrogram artifacts and feed the same HoloViz image contract

This keeps the production render path in C++ while still letting us debug without RF hardware.

### Will it work over a remote connection?

Yes, with the normal HoloViz caveat: it opens a native window on the machine running the process.

Practical remote options:

- local desktop on the host machine
- X11 forwarding if OpenGL and latency are acceptable
- remote desktop sessions such as VNC or NICE DCV
- containerized runs with the host display forwarded into the container

Not a primary target for the first milestone:

- plain headless SSH with no display server
- browser delivery without adding a streaming layer such as WebRTC

## Architecture Decision

Use HoloViz as the renderer and keep the final path in C++.

### Live path

1. `chdrConverterOp -> fftOp -> spectrogramOp`
2. branch `spectrogramOp` to the detector path and to a spectrogram visualization renderer
3. feed the renderer output into `HolovizOp`

### Offline path

1. replay saved `.pgm` spectrogram frames from disk
2. convert them into the same HoloViz image tensor contract
3. feed them into `HolovizOp`

### Why this is the correct pipeline

- the live viewer stays attached to the real data path instead of a disposable side executable
- the offline harness exists only to replace the source, not the renderer contract
- the render contract becomes reusable for future remote streaming or recording paths
- detector overlays can be added without rewriting the image renderer

## Current Contract

The spectrogram operator already forwards its tensor output, so it is the right branch point.

The image contract for the viewer should be:

- one color image tensor emitted to HoloViz on `receivers`
- a stable tensor name for the spectrogram image
- overlay specs emitted later on `input_specs` when detector overlays are introduced

This matches the HoloViz patterns used elsewhere in Holohub.

## Overlay Plan

The detector side should ultimately emit visualization-friendly data rather than forcing HoloViz-specific logic into the detector itself.

Preferred overlay contract:

- rectangles for quick validation and operator debugging
- optional text labels for confidence or class summaries
- optional mask output later if the segmentation view is more useful than boxes

Recommended sequence:

1. finish spectrogram-only rendering in the live and offline paths
2. add a detector postprocessor that emits HoloViz `InputSpec` data plus overlay tensors
3. map RF coordinates into normalized screen coordinates in that postprocessor
4. keep the renderer itself agnostic to detector internals

## Remote Strategy

Primary path:

- HoloViz native window for local or remote-desktop sessions

Fallback path after the first milestone:

- export annotated frames or clips when no display is available
- optionally add a streaming frontend later without changing the core spectrogram renderer contract

## Implementation Phases

### Phase 1: Shared spectrogram rendering contract

Deliverable:

- a reusable C++ spectrogram-to-HoloViz image path

Scope:

- create a renderer operator for live spectrogram tensors
- define a stable image tensor name for HoloViz
- add visualization config to the main app

Success criteria:

- the live app can open a spectrogram window when visualization is enabled

### Phase 2: Offline replay harness

Deliverable:

- an offline replay executable that uses the same HoloViz image contract

Scope:

- enumerate `.pgm` frames from disk
- replay with configurable frame rate and loop behavior
- support development without Advanced Network or a radio

Success criteria:

- replay works from `/workspace/spectrograms` or mounted host artifacts

### Phase 3: Detector overlay contract

Deliverable:

- a dedicated overlay postprocessor between detector output and HoloViz

Scope:

- emit overlay tensors and `InputSpec` metadata
- support rectangles first, masks later if needed
- preserve a clean separation between inference and rendering

Success criteria:

- live detections can be drawn without changing the spectrogram renderer or HoloViz setup

### Phase 4: Remote-friendly fallback

Deliverable:

- a no-display export mode or later streaming path

Scope:

- save annotated frames or video when no interactive display is available
- keep the underlying render contract unchanged

Success criteria:

- the viewer workflow still produces debuggable output on headless systems

## Immediate Work Items

1. Add the shared C++ spectrogram renderer used by the live pipeline.
2. Wire an optional visualization branch off `spectrogramOp` in the main app.
3. Add an offline replay executable that reuses the same HoloViz tensor contract.
4. Keep detector overlay work as the next focused step instead of baking fake overlays into the renderer.

## Recommended Direction

Proceed with:

- C++ live visualization branch in the main app
- offline replay as a debug harness, not the core architecture
- HoloViz as the renderer
- detector overlays added via a separate postprocessor contract

This is the shortest path that is still structurally correct for a fast production viewer.