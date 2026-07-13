from __future__ import annotations

import os
from pathlib import Path


# These must be set before any Hugging Face, Transformers, or MLX imports.
# The API and every training/generation worker stay offline. Only the explicit
# first-run model installer subprocess opts out through this internal flag.
ALLOW_MODEL_DOWNLOAD = os.environ.get("VOICETRAINER_ALLOW_MODEL_DOWNLOAD") == "1"
if ALLOW_MODEL_DOWNLOAD:
    for offline_name in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ.pop(offline_name, None)
else:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("DO_NOT_TRACK", "1")
for secret_name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "OPENAI_API_KEY"):
    os.environ.pop(secret_name, None)


def _configured_path(name: str, fallback: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else fallback


APP_ROOT = _configured_path(
    "VOICETRAINER_APP_ROOT",
    Path.home() / "Library" / "Application Support" / "VoiceTrainer",
)
CACHE_ROOT = _configured_path(
    "VOICETRAINER_CACHE_ROOT",
    Path.home() / "Library" / "Caches" / "VoiceTrainer",
)
LOG_ROOT = _configured_path(
    "VOICETRAINER_LOG_ROOT",
    Path.home() / "Library" / "Logs" / "VoiceTrainer",
)

MODEL_REPO = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"
MODEL_REVISION = "a6eb4f68e4b056f1215157bb696209bc82a6db48"
MODEL_ROOT = APP_ROOT / "models" / "qwen3-tts-1.7b-base" / MODEL_REVISION
MODEL_MANIFEST = MODEL_ROOT / "manifest.json"

ASR_REPO = "mlx-community/whisper-large-v3-turbo-asr-fp16"
ASR_REVISION = "624c19c9af5603fa73b83bce14d4aeea96156d18"
ASR_ROOT = APP_ROOT / "models" / "whisper-large-v3-turbo" / ASR_REVISION
ASR_MANIFEST = ASR_ROOT / "manifest.json"

PROFILES_ROOT = APP_ROOT / "profiles"
DATASETS_ROOT = APP_ROOT / "datasets"
STATE_DB = APP_ROOT / "state.sqlite"

HOST = "127.0.0.1"
PORT = int(os.environ.get("VOICETRAINER_PORT", "8787"))
if not 1024 <= PORT <= 65535:
    raise ValueError("VOICETRAINER_PORT must be between 1024 and 65535")
ORIGIN = f"http://{HOST}:{PORT}"


def ensure_private_directories() -> None:
    os.umask(0o077)
    for path in (
        APP_ROOT,
        CACHE_ROOT,
        LOG_ROOT,
        APP_ROOT / "models",
        PROFILES_ROOT,
        DATASETS_ROOT,
        APP_ROOT / ".trash",
    ):
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
        path.chmod(0o700)
