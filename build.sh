#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
: "${OFFICIAL_BASE:?Set the organizer image name@sha256:digest}"
: "${IMAGE_TAG:=hls-agent:submission}"
if [[ ! "$OFFICIAL_BASE" =~ ^[^[:space:]@]+@sha256:[0-9a-f]{64}$ ]]; then
  echo 'OFFICIAL_BASE must be an exact image digest' >&2; exit 2
fi
# Require a preloaded base; the build performs no package/model downloads.
docker image inspect "$OFFICIAL_BASE" >/dev/null
docker build --pull=false --network=none --build-arg "OFFICIAL_BASE=$OFFICIAL_BASE" -t "$IMAGE_TAG" .
