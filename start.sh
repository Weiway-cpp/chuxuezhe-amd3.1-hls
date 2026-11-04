#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
exec "${PYTHON:-python3}" -m serve.offline --deployment "${DEPLOYMENT_FILE:?Set DEPLOYMENT_FILE to the prepared asset manifest}" "$@"
