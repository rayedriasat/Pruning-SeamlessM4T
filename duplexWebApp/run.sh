#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"

exec python -m uvicorn backend.app:app --host "$HOST" --port "$PORT" --ws-max-size 16777216
