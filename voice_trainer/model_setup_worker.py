"""Explicit online boundary for one-time, pinned model installation."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


PHASE_PROGRESS = {
    ("Qwen", "ready"): 0.68,
    ("Qwen", "adopting"): 0.08,
    ("Qwen", "downloading"): 0.08,
    ("Qwen", "verifying"): 0.62,
    ("Qwen", "repairing"): 0.04,
    ("Qwen", "installed"): 0.68,
    ("Whisper", "ready"): 0.96,
    ("Whisper", "downloading"): 0.72,
    ("Whisper", "verifying"): 0.93,
    ("Whisper", "repairing"): 0.70,
    ("Whisper", "installed"): 0.96,
}


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    temporary.chmod(0o600)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Install pinned Noice City models")
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--failure", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("VOICETRAINER_ALLOW_MODEL_DOWNLOAD") != "1":
        raise RuntimeError("Model setup worker requires explicit download authorization")

    # Import only after checking the dedicated setup-process boundary. Config
    # leaves Hub access enabled only for this worker.
    from .bootstrap import install_required_models

    args.progress.parent.mkdir(parents=True, exist_ok=True, mode=0o700)

    def emit(event: dict) -> None:
        payload = {
            **event,
            "progress": float(event.get("progress", PHASE_PROGRESS.get(
                (str(event.get("component")), str(event.get("phase"))), 0.02
            ))),
            "timestamp": time.time(),
        }
        with args.progress.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, sort_keys=True) + "\n")

    try:
        emit(
            {
                "component": "setup",
                "phase": "starting",
                "message": "Preparing the one-time local model installation",
            }
        )
        installed = install_required_models(progress=emit)
        result = {"status": "complete", "models": installed, "progress": 1.0}
        _write_json(args.result, result)
        emit(
            {
                "component": "setup",
                "phase": "complete",
                "message": "Noice City is ready to use offline",
                "progress": 1.0,
            }
        )
    except Exception as exc:
        _write_json(args.failure, {"status": "failed", "error": str(exc)})
        raise


if __name__ == "__main__":
    main()
