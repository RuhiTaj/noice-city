#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

[[ -x .venv/bin/python ]] || {
  echo "Run ./setup.sh once before starting Noice City." >&2
  exit 1
}

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN OPENAI_API_KEY \
  HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost

# Keep one private loopback-session key across restarts. This lets an open
# Noice City tab reconnect after a local server restart without exposing the
# API to the LAN.
SESSION_DIR="${VOICETRAINER_APP_ROOT:-$HOME/Library/Application Support/VoiceTrainer}"
TOKEN_FILE="$SESSION_DIR/session-token"
mkdir -p "$SESSION_DIR"
chmod 700 "$SESSION_DIR"
if [[ ! -f "$TOKEN_FILE" ]] || ! grep -Eq '^[a-f0-9]{64}$' "$TOKEN_FILE"; then
  umask 077
  openssl rand -hex 32 > "$TOKEN_FILE.tmp"
  mv "$TOKEN_FILE.tmp" "$TOKEN_FILE"
fi
chmod 600 "$TOKEN_FILE"
export VOICETRAINER_SESSION_TOKEN="$(tr -d '\r\n' < "$TOKEN_FILE")"

if ! .venv/bin/python -m voice_trainer.bootstrap --verify-only --quick >/dev/null 2>&1; then
  echo "Noice City will finish its one-time Qwen setup in the browser."
fi
PORT="${VOICETRAINER_PORT:-8787}"
URL="http://127.0.0.1:${PORT}/#${VOICETRAINER_SESSION_TOKEN}"
(sleep 1; open "$URL" >/dev/null 2>&1 || true) &
exec .venv/bin/python -m uvicorn voice_trainer.app:app --host 127.0.0.1 --port "$PORT"
