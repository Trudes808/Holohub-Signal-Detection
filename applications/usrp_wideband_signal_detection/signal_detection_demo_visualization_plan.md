# Signal Detection Demo Visualization Plan

## Objective

Create a visualization path for the USRP wideband signal detection work that can:

- display a live spectrogram in a Holoscan window
- overlay signal detection results on top of the spectrogram
- run without a connected radio by replaying saved spectrogram frames
- remain usable when the app is launched on a remote machine

## Short Answers

### Can we add a flag to use offline data?

Yes. This is practical and should be part of the first implementation slice.

The cleanest approach is to support an input mode flag such as:

```text
--input-source live
--input-source offline
--input-source synthetic
```

For offline mode, the visualizer can read previously saved spectrogram frames from a directory such as `/tmp/usrp_spectrograms` on the host or `/workspace/spectrograms` inside the demo container.

### Will it work over a remote connection?

Yes, with an important caveat: HoloViz renders a native window on the machine running the application. That means remote viewing is possible, but only if the remote session supports graphics forwarding or desktop remoting.

Practical remote options:

- local desktop session on the machine running the app
- SSH with X11 forwarding if OpenGL support is sufficient
- VNC, NICE DCV, or another remote desktop session
- running inside a container while forwarding the host display into the container

What is not guaranteed to work well:

- plain headless SSH with no display server
- browser-only access without an additional streaming layer

Because of that, the visualization plan should also include a fallback mode that writes frames or clips to disk when no display is available.

## Recommended Architecture

Use HoloViz as the default renderer.

Why:

- it is already the standard Holoscan visualization operator
- it supports image rendering and overlay layers
- it supports an ImGui callback for lightweight controls
- it fits the current C++ app better than introducing a separate GUI stack

Recommended app split:

1. Keep the current radio ingest and detection executable focused on processing.
2. Add a dedicated visualization executable in the same application directory.
3. Let the visualization executable support both offline replay and, later, live detector output.

This avoids forcing offline development to initialize Advanced Network or require a connected USRP.

## Proposed Viewer Modes

### Mode 1: Offline replay

Read saved spectrogram frames from disk and display them in HoloViz.

Primary use:

- UI development without hardware
- overlay prototyping
- remote debugging
- demo recording

Inputs:

- directory of `.pgm` spectrogram files
- optional directory of matching detector mask files

### Mode 2: Synthetic replay

Generate a fake spectrogram and fake detections in-process.

Primary use:

- rapid UI work
- sanity checks in environments without saved artifacts
- CI-friendly smoke testing later

### Mode 3: Live stream

Render the real spectrogram output and real detection overlays from the Holoscan pipeline.

Primary use:

- final integrated demo
- live RF debugging

## Overlay Strategy

The current spectrogram operator already forwards its tensor downstream, which makes it a good branch point for visualization.

The detection side should eventually expose one of these visualization-friendly outputs:

- mask tensor for alpha overlay
- bounding boxes
- contours or polygons
- detection metadata with confidence and frequency-time extents

Recommended sequence:

1. Start with spectrogram-only viewing.
2. Add a fake overlay generator for rectangles or masks.
3. Add a small postprocessor that converts detector outputs into a HoloViz-friendly overlay representation.
4. Replace fake overlays with real detections once the detector contract is finalized.

## Remote Viewing Strategy

Primary path:

- HoloViz native window for local or remote-desktop sessions

Fallback path:

- headless mode that saves annotated frames to disk
- optional future path for streamed viewing, such as WebRTC or video publishing

Recommendation:

Do not block the first visualizer on browser delivery. Start with HoloViz, but keep the offline replay format reusable so a later streaming frontend can read the same data.

## Phased Implementation Plan

### Phase 1: Offline spectrogram viewer

Deliverable:

- a new executable that replays `.pgm` spectrograms in HoloViz

Scope:

- command-line flags for `--input-source offline` and `--offline-dir`
- file enumeration and replay timing controls
- grayscale or false-color rendering
- window title and basic playback controls if convenient

Success criteria:

- can run without USRP hardware
- can display saved spectrograms from `/tmp/usrp_spectrograms`

### Phase 2: Fake overlay support

Deliverable:

- simple boxes or mask overlays drawn on top of offline spectrogram frames

Scope:

- deterministic fake detections
- optional ImGui toggles for overlay visibility, threshold, and channel selection

Success criteria:

- verify overlay rendering path before real detector integration

### Phase 3: Real detector overlay contract

Deliverable:

- detection output path from the detector or a detector postprocessor

Scope:

- define emitted structure for detections
- decide whether mask, boxes, or both should be supported
- map RF coordinates to screen coordinates

Success criteria:

- live detector output can be rendered without changing the visualizer core

### Phase 4: Live integrated viewer

Deliverable:

- visualization branch connected to the real-time Holoscan pipeline

Scope:

- branch from spectrogram output to HoloViz
- branch from detector output to overlay renderer
- optional runtime flag to enable or disable the viewer

Success criteria:

- same viewer code works for offline replay and live mode

### Phase 5: Remote-friendly fallback

Deliverable:

- no-display mode for remote or automated runs

Scope:

- save annotated frames
- optional MP4 or image sequence export
- detect missing display and fail clearly or switch modes

Success criteria:

- viewer workflow remains usable on systems without an interactive desktop

## First Implementation Slice

Start with a new executable dedicated to offline replay.

Reasoning:

- it removes dependency on the radio and Advanced Network during UI work
- it gives a fast development loop
- it exercises HoloViz immediately
- it can later share rendering logic with the live pipeline

First slice tasks:

1. Add a new C++ source file for an offline spectrogram viewer application.
2. Add a small source operator that loads `.pgm` files from a directory and emits image tensors.
3. Connect that source to `HolovizOp`.
4. Add command-line flags for offline directory and replay rate.
5. Add an optional fake overlay layer callback.
6. Update CMake to build the visualizer executable.
7. Update the application README with offline and remote usage notes.

## Open Design Decisions

- whether the viewer should be a separate executable or a mode inside the main app
- whether the first overlay should be boxes or masks
- whether false-color mapping should happen in a preprocessor operator or directly in the viewer path
- whether headless export should be part of the first milestone or immediately after

## Recommended Decision

Proceed with:

- separate visualization executable
- offline replay as the first supported mode
- HoloViz as the renderer
- fake overlays before real detector overlays
- remote desktop and display-forwarded sessions as the first remote target

This gives the shortest path to a usable result while the radio remains disconnected.