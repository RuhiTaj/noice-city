#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

command -v uv >/dev/null 2>&1 || {
  echo "uv is required. Install it once, then rerun setup." >&2
  exit 1
}

uv sync --locked
echo "Noice City is installed. Start it with ./start.sh and complete the one-time model download in the browser."
