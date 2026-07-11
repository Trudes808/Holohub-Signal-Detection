#!/usr/bin/env bash
# Shared container identity for the USRP wideband signal-detection demo scripts.
#
# Why this file exists: the demo container/image names are used by the build,
# rebuild, and run wrappers. `sudo` strips your interactive shell environment,
# so exporting CONTAINER_NAME in your profile does NOT reach these scripts.
# Centralizing the names here (sourced by every wrapper) makes the whole
# toolchain target one container without repeating `CONTAINER_NAME=...` on each
# invocation.
#
# To point the toolchain at a different container, change CONTAINER_NAME below.
# You can still override for a single run by exporting CONTAINER_NAME / IMAGE_NAME
# (e.g. `sudo env CONTAINER_NAME=other ./rebuild_demo_container_app.sh`).
#
# The image is shared and reused as-is (pass SKIP_IMAGE_BUILD=1 to
# build_demo_container.sh to skip rebuilding it); only the container instance
# needs to be yours.
: "${CONTAINER_NAME:=usrp_x410_sig_det_sat3737}"
: "${IMAGE_NAME:=usrp_x410_signal_detection_demo:latest}"
export CONTAINER_NAME IMAGE_NAME
