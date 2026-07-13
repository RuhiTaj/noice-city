"""Serialized first-run model installation managed by the offline API process."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from . import config
from .bootstrap import verify_asr_model, verify_model


MODEL_DOWNLOAD_BYTES = 4_540_000_000
ASR_DOWNLOAD_BYTES = 1_500_000_000


def _ready(check) -> tuple[bool, str | None]:
    try:
        check(full_hash=False)
        return True, None
    except Exception as exc:
        return False, str(exc)


class ModelSetupJobs:
    """Own one explicit network-enabled installer subprocess at a time."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self._lock = threading.Lock()
        self._process: subprocess.Popen | None = None
        self._state: dict[str, Any] = {
            "status": "idle",
            "progress": 0.0,
            "phase": "idle",
            "message": "Download the model pack once to begin",
            "error": None,
            "started_at": None,
            "finished_at": None,
        }

    @property
    def active(self) -> bool:
        with self._lock:
            return self._process is not None and self._process.poll() is None

    def status(self) -> dict[str, Any]:
        qwen_ready, qwen_error = _ready(verify_model)
        asr_ready, asr_error = _ready(verify_asr_model)
        with self._lock:
            state = dict(self._state)
            active = self._process is not None and self._process.poll() is None
        if qwen_ready and asr_ready and not active:
            state.update(
                {
                    "status": "complete",
                    "progress": 1.0,
                    "phase": "complete",
                    "message": "Models installed; Noice City is ready offline",
                    "error": None,
                }
            )
        return {
            **state,
            "active": active,
            "ready": qwen_ready and asr_ready,
            "download_bytes_approx": MODEL_DOWNLOAD_BYTES + ASR_DOWNLOAD_BYTES,
            "models": {
                "qwen": {
                    "ready": qwen_ready,
                    "repo": config.MODEL_REPO,
                    "revision": config.MODEL_REVISION,
                    "download_bytes_approx": MODEL_DOWNLOAD_BYTES,
                    "error": None if qwen_ready else qwen_error,
                },
                "whisper": {
                    "ready": asr_ready,
                    "repo": config.ASR_REPO,
                    "revision": config.ASR_REVISION,
                    "download_bytes_approx": ASR_DOWNLOAD_BYTES,
                    "error": None if asr_ready else asr_error,
                },
            },
        }

    def start(self) -> dict[str, Any]:
        current = self.status()
        if current["ready"]:
            return current
        already_active = False
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                already_active = True
            else:
                job_id = f"model-setup-{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
                control = config.CACHE_ROOT / "model-setup" / job_id
                control.mkdir(parents=True, exist_ok=False, mode=0o700)
                progress_path = control / "progress.jsonl"
                result_path = control / "result.json"
                failure_path = control / "failure.json"
                log_path = control / "worker.log"
                environment = dict(os.environ)
                environment["VOICETRAINER_ALLOW_MODEL_DOWNLOAD"] = "1"
                environment["HF_HUB_DISABLE_TELEMETRY"] = "1"
                environment["DO_NOT_TRACK"] = "1"
                environment["NO_PROXY"] = "127.0.0.1,localhost"
                for name in (
                    "HF_HUB_OFFLINE",
                    "HF_DATASETS_OFFLINE",
                    "TRANSFORMERS_OFFLINE",
                    "HF_TOKEN",
                    "HUGGING_FACE_HUB_TOKEN",
                    "OPENAI_API_KEY",
                ):
                    environment.pop(name, None)
                command = [
                    sys.executable,
                    "-m",
                    "voice_trainer.model_setup_worker",
                    "--progress",
                    str(progress_path),
                    "--result",
                    str(result_path),
                    "--failure",
                    str(failure_path),
                ]
                with log_path.open("ab", buffering=0) as log:
                    process = subprocess.Popen(
                        command,
                        cwd=self.project_root,
                        env=environment,
                        stdin=subprocess.DEVNULL,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        close_fds=True,
                    )
                self._process = process
                self._state = {
                    "status": "running",
                    "progress": 0.01,
                    "phase": "starting",
                    "message": "Starting one-time model installation",
                    "error": None,
                    "started_at": time.time(),
                    "finished_at": None,
                }
        if already_active:
            return self.status()
        threading.Thread(
            target=self._monitor,
            args=(process, progress_path, result_path, failure_path),
            name="voicetrainer-model-setup",
            daemon=True,
        ).start()
        return self.status()

    def _monitor(
        self,
        process: subprocess.Popen,
        progress_path: Path,
        result_path: Path,
        failure_path: Path,
    ) -> None:
        position = 0
        try:
            while process.poll() is None:
                position = self._consume_progress(progress_path, position)
                time.sleep(0.35)
            self._consume_progress(progress_path, position)
            if process.returncode == 0 and result_path.is_file():
                with self._lock:
                    self._state.update(
                        {
                            "status": "complete",
                            "progress": 1.0,
                            "phase": "complete",
                            "message": "Models installed; Noice City is ready offline",
                            "error": None,
                            "finished_at": time.time(),
                        }
                    )
            else:
                error = f"Model installer exited with status {process.returncode}"
                if failure_path.is_file():
                    error = str(json.loads(failure_path.read_text()).get("error") or error)
                with self._lock:
                    self._state.update(
                        {
                            "status": "failed",
                            "phase": "failed",
                            "message": "Model installation failed",
                            "error": error,
                            "finished_at": time.time(),
                        }
                    )
        finally:
            with self._lock:
                if self._process is process:
                    self._process = None

    def _consume_progress(self, path: Path, position: int) -> int:
        if not path.is_file():
            return position
        with path.open("r", encoding="utf-8") as stream:
            stream.seek(position)
            for line in stream:
                event = json.loads(line)
                with self._lock:
                    self._state.update(
                        {
                            "status": "running",
                            "progress": max(
                                float(self._state.get("progress", 0.0)),
                                float(event.get("progress", 0.0)),
                            ),
                            "phase": event.get("phase", self._state.get("phase")),
                            "message": event.get("message", self._state.get("message")),
                        }
                    )
            return stream.tell()
