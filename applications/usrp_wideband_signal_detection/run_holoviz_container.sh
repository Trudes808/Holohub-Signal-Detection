#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME=${CONTAINER_NAME:-holoviz_recent}
if ! sudo docker inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
  echo "Container not found: ${CONTAINER_NAME}" >&2
  echo "Run ./build_demo_container.sh first to create and provision it." >&2
  exit 1
fi

if [[ "$(sudo docker inspect -f '{{.State.Running}}' "${CONTAINER_NAME}")" == "true" ]]; then
  echo "Container ${CONTAINER_NAME} is already running."
else
  sudo docker start "${CONTAINER_NAME}" >/dev/null
  echo "Started container ${CONTAINER_NAME}."
fi

echo "Open a shell with: ./enter_holoviz_container.sh"