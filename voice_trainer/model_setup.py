"""Serialized first-run model installation managed by the offline API process."""

from __future__ import annotations

import json
import os
import shutil
import signal
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
ASR_DOWNLOAD_BYTES = 1_623_000_000
CANCEL_WAIT_SECONDS = 5.0


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
        self._monitor_thread: threading.Thread | None = None
        self._monitor_wakeup = threading.Event()
        self._control_path: Path | None = None
        self._staging_path: Path | None = None
        self._cancel_requested_for: subprocess.Popen | None = None
        self._closed = False
        self._state: dict[str, Any] = {
            "status": "idle",
            "component": None,
            "progress": 0.0,
            "phase": "idle",
            "message": "Install Qwen3-TTS once to begin",
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
        if qwen_ready and not active and state.get("component") != "whisper":
            state.update(
                {
                    "status": "complete",
                    "progress": 1.0,
                    "phase": "complete",
                    "message": "Qwen3-TTS is installed; Noice City is ready offline",
                    "error": None,
                }
            )
        return {
            **state,
            "active": active,
            "cancelable": active,
            "ready": qwen_ready,
            "speech_checks_ready": asr_ready,
            "download_bytes_approx": 0 if qwen_ready else MODEL_DOWNLOAD_BYTES,
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

    def start(self, component: str = "qwen") -> dict[str, Any]:
        if component not in {"qwen", "whisper"}:
            raise ValueError("Unknown local model component")
        with self._lock:
            if self._closed:
                raise RuntimeError("Model setup is shutting down")
        current = self.status()
        target_ready = (
            current["models"]["qwen"]["ready"]
            if component == "qwen"
            else current["models"]["whisper"]["ready"]
        )
        if target_ready:
            return current
        already_active = False
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                already_active = True
            else:
                job_id = f"model-setup-{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
                control = config.CACHE_ROOT / "model-setup" / job_id
                control.mkdir(parents=True, exist_ok=False, mode=0o700)
                staging_path = config.APP_ROOT / "models" / f".{job_id}"
                progress_path = control / "progress.jsonl"
                result_path = control / "result.json"
                failure_path = control / "failure.json"
                log_path = control / "worker.log"
                hub_home = config.CACHE_ROOT / "huggingface"
                hub_home.mkdir(parents=True, exist_ok=True, mode=0o700)
                environment = dict(os.environ)
                environment["VOICETRAINER_ALLOW_MODEL_DOWNLOAD"] = "1"
                environment["HF_HUB_DISABLE_TELEMETRY"] = "1"
                environment["DO_NOT_TRACK"] = "1"
                # Keep Hub metadata, Xet chunks, and setup logs inside the
                # private VoiceTrainer cache tree described by PRIVACY.md.
                # This also prevents a one-time install from leaving data in
                # the user's unrelated ~/.cache/huggingface directory.
                environment["HF_HOME"] = str(hub_home)
                environment["HF_HUB_CACHE"] = str(hub_home / "hub")
                environment["HUGGINGFACE_HUB_CACHE"] = str(hub_home / "hub")
                environment["HF_XET_CACHE"] = str(hub_home / "xet")
                environment["XDG_CACHE_HOME"] = str(config.CACHE_ROOT / "xdg")
                environment["NO_PROXY"] = "127.0.0.1,localhost"
                environment["no_proxy"] = "127.0.0.1,localhost"
                for name in (
                    "HF_HUB_OFFLINE",
                    "HF_DATASETS_OFFLINE",
                    "TRANSFORMERS_OFFLINE",
                    "HF_TOKEN",
                    "HUGGING_FACE_HUB_TOKEN",
                    "OPENAI_API_KEY",
                    "HTTP_PROXY",
                    "HTTPS_PROXY",
                    "ALL_PROXY",
                    "http_proxy",
                    "https_proxy",
                    "all_proxy",
                ):
                    environment.pop(name, None)
                command = [
                    sys.executable,
                    "-m",
                    "voice_trainer.model_setup_worker",
                    "--component",
                    component,
                    "--progress",
                    str(progress_path),
                    "--result",
                    str(result_path),
                    "--failure",
                    str(failure_path),
                    "--staging",
                    str(staging_path),
                    "--parent-pid",
                    str(os.getpid()),
                ]
                try:
                    with log_path.open("ab", buffering=0) as log:
                        process = subprocess.Popen(
                            command,
                            cwd=self.project_root,
                            env=environment,
                            stdin=subprocess.DEVNULL,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            close_fds=True,
                            start_new_session=True,
                        )
                except Exception:
                    self._cleanup_job_artifacts(control, staging_path)
                    raise
                self._process = process
                self._control_path = control
                self._staging_path = staging_path
                self._cancel_requested_for = None
                self._monitor_wakeup.clear()
                self._state = {
                    "status": "running",
                    "component": component,
                    "progress": 0.01,
                    "phase": "starting",
                    "message": (
                        "Starting Qwen3-TTS installation"
                        if component == "qwen"
                        else (
                            "Starting Whisper transcription and "
                            "checkpoint-evaluation installation"
                        )
                    ),
                    "error": None,
                    "started_at": time.time(),
                    "finished_at": None,
                }
        if already_active:
            return self.status()
        monitor = threading.Thread(
            target=self._monitor,
            args=(
                process,
                progress_path,
                result_path,
                failure_path,
                control,
                staging_path,
            ),
            name="voicetrainer-model-setup",
            daemon=True,
        )
        with self._lock:
            self._monitor_thread = monitor
        monitor.start()
        return self.status()

    def cancel(self, component: str | None = None) -> dict[str, Any]:
        """Stop and reap the active installer, then remove only its artifacts."""
        if component is not None and component not in {"qwen", "whisper"}:
            raise ValueError("Unknown local model component")
        with self._lock:
            process = self._process
            active_component = self._state.get("component")
            process_active = process is not None and process.poll() is None
            if (
                process_active
                and component is not None
                and active_component not in {None, component}
            ):
                raise ValueError(f"The active model setup is for {active_component}")
            if not process_active:
                inactive = True
                monitor = None
                control = None
                staging_path = None
            else:
                inactive = False
                self._cancel_requested_for = process
                self._state.update(
                    {
                        "status": "cancelling",
                        "phase": "cancelling",
                        "message": "Stopping the local model download safely",
                        "error": None,
                    }
                )
                monitor = self._monitor_thread
                control = self._control_path
                staging_path = self._staging_path

        if inactive:
            return self.status()

        self._signal_process_group(process, signal.SIGTERM)
        try:
            process.wait(timeout=CANCEL_WAIT_SECONDS)
        except subprocess.TimeoutExpired:
            self._signal_process_group(process, signal.SIGKILL)
            process.wait(timeout=CANCEL_WAIT_SECONDS)

        self._monitor_wakeup.set()

        if monitor is not None and monitor is not threading.current_thread():
            monitor.join(timeout=max(CANCEL_WAIT_SECONDS, 1.0))

        # A monitor normally performs this transition.  Retain the fallback so
        # cancellation is also safe in startup/shutdown races.
        with self._lock:
            if self._process is process:
                self._mark_cancelled_locked(process)
        if control is not None and staging_path is not None:
            self._cleanup_job_artifacts(control, staging_path)
        return self.status()

    def shutdown(self) -> None:
        """Prevent new setup work and synchronously reap any active worker."""
        with self._lock:
            self._closed = True
        self.cancel()

    def _monitor(
        self,
        process: subprocess.Popen,
        progress_path: Path,
        result_path: Path,
        failure_path: Path,
        control: Path,
        staging_path: Path,
    ) -> None:
        position = 0
        try:
            while process.poll() is None:
                position = self._consume_progress(progress_path, position)
                self._monitor_wakeup.wait(0.35)
                self._monitor_wakeup.clear()
            self._consume_progress(progress_path, position)
            with self._lock:
                cancelled = self._cancel_requested_for is process
            if cancelled:
                with self._lock:
                    self._mark_cancelled_locked(process)
            elif process.returncode == 0 and result_path.is_file():
                with self._lock:
                    self._state.update(
                        {
                            "status": "complete",
                            "progress": 1.0,
                            "phase": "complete",
                            "message": "Local model installation complete",
                            "error": None,
                            "finished_at": time.time(),
                        }
                    )
            else:
                error = f"Model installer exited with status {process.returncode}"
                if failure_path.is_file():
                    try:
                        failure = json.loads(failure_path.read_text())
                        error = str(failure.get("error") or error)
                    except (OSError, json.JSONDecodeError):
                        pass
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
                    self._monitor_thread = None
                    self._control_path = None
                    self._staging_path = None
                cancelled = self._cancel_requested_for is process
                if cancelled:
                    self._cancel_requested_for = None
            if cancelled:
                try:
                    self._cleanup_job_artifacts(control, staging_path)
                except RuntimeError:
                    # Refusal is the safe failure mode if paths are ever
                    # inconsistent; never broaden cleanup to compensate.
                    pass

    def _mark_cancelled_locked(self, process: subprocess.Popen) -> None:
        self._state.update(
            {
                "status": "cancelled",
                "phase": "cancelled",
                "message": "Model installation canceled; you can restart it anytime",
                "error": None,
                "finished_at": time.time(),
            }
        )
        if self._process is process:
            self._process = None
            self._monitor_thread = None
            self._control_path = None
            self._staging_path = None

    @staticmethod
    def _signal_process_group(process: subprocess.Popen, signum: int) -> None:
        """Signal the dedicated installer session, including helper children."""
        try:
            os.killpg(process.pid, signum)
        except ProcessLookupError:
            pass

    @staticmethod
    def _cleanup_job_artifacts(control: Path, staging_path: Path) -> None:
        """Delete only the two directories derived from one generated job ID."""
        control_parent = (config.CACHE_ROOT / "model-setup").resolve()
        staging_parent = (config.APP_ROOT / "models").resolve()
        job_id = control.name
        if (
            control.parent.resolve() != control_parent
            or not job_id.startswith("model-setup-")
            or staging_path.parent.resolve() != staging_parent
            or staging_path.name != f".{job_id}"
        ):
            raise RuntimeError("Refusing to clean an unrecognized model setup path")
        for path in (staging_path, control):
            if path.is_symlink():
                path.unlink(missing_ok=True)
            elif path.exists():
                shutil.rmtree(path)

    def _consume_progress(self, path: Path, position: int) -> int:
        if not path.is_file():
            return position
        with path.open("r", encoding="utf-8") as stream:
            stream.seek(position)
            for line in stream:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                with self._lock:
                    if self._cancel_requested_for is not None:
                        continue
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
