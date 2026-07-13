"""Explicit online boundary for one-time, pinned model installation."""

from __future__ import annotations

import argparse
import json
import os
import signal
import threading
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


class SetupCancelled(Exception):
    """Raised in the worker's main thread when setup is canceled."""


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    temporary.chmod(0o600)
    os.replace(temporary, path)


def _watch_parent(parent_pid: int, stop: threading.Event) -> None:
    """Cancel this worker if its API parent disappears unexpectedly."""
    while not stop.wait(0.5):
        if os.getppid() != parent_pid:
            # This worker is launched in its own session.  Signal its whole
            # process group so downloader helpers cannot outlive the API app.
            os.killpg(os.getpgrp(), signal.SIGTERM)
            return


def main() -> int:
    parser = argparse.ArgumentParser(description="Install pinned Noice City models")
    parser.add_argument("--component", choices=("qwen", "whisper"), required=True)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--failure", type=Path, required=True)
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--parent-pid", type=int, required=True)
    args = parser.parse_args()
    if os.environ.get("VOICETRAINER_ALLOW_MODEL_DOWNLOAD") != "1":
        raise RuntimeError("Model setup worker requires explicit download authorization")

    # Import only after checking the dedicated setup-process boundary. Config
    # leaves Hub access enabled only for this worker.
    from .bootstrap import install_asr_model, install_qwen_model

    args.progress.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    parent_watch_stop = threading.Event()
    parent_watch = threading.Thread(
        target=_watch_parent,
        args=(args.parent_pid, parent_watch_stop),
        name="voicetrainer-model-setup-parent-watch",
        daemon=True,
    )

    def cancel_handler(signum, frame) -> None:
        del signum, frame
        raise SetupCancelled("Model setup canceled")

    old_sigterm = signal.signal(signal.SIGTERM, cancel_handler)
    old_sigint = signal.signal(signal.SIGINT, cancel_handler)
    parent_watch.start()

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
                "message": (
                    "Preparing Qwen3-TTS"
                    if args.component == "qwen"
                    else "Preparing Whisper transcription and checkpoint evaluation"
                ),
            }
        )
        if args.component == "qwen":
            installed = {
                "qwen": str(
                    install_qwen_model(progress=emit, staging_path=args.staging)
                )
            }
        else:
            installed = {
                "whisper": str(
                    install_asr_model(progress=emit, staging_path=args.staging)
                )
            }
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
        return 0
    except SetupCancelled as exc:
        _write_json(args.failure, {"status": "cancelled", "error": str(exc)})
        return 130
    except Exception as exc:
        _write_json(args.failure, {"status": "failed", "error": str(exc)})
        raise
    finally:
        parent_watch_stop.set()
        parent_watch.join(timeout=1.0)
        signal.signal(signal.SIGTERM, old_sigterm)
        signal.signal(signal.SIGINT, old_sigint)


if __name__ == "__main__":
    raise SystemExit(main())
