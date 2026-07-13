"""Loopback-only Noice City API and local application server."""

from __future__ import annotations

# Importing config first sets every supported offline/telemetry switch before
# Hugging Face, MLX, or model code can be imported by a request worker.
from . import config as config

import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .bootstrap import verify_asr_model, verify_model
from .model_setup import ModelSetupJobs
from .model_runtime import NamedSpeakerRuntime
from .schemas import GenerationRequest
from .security import LocalSessionMiddleware
from .store import DEFAULT_VARIANT_ID, VoiceStore
from .training import TrainingConfig
from .transcription import transcribe_local


PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_ROOT = PROJECT_ROOT / "static"
MAX_RECORDING_BYTES = 8 * 1024 * 1024


class VariantMismatchError(ValueError):
    """Raised when a resource is addressed through the wrong voice lineage."""

config.ensure_private_directories()
store = VoiceStore(config.APP_ROOT).initialize()

app = FastAPI(
    title="Noice City",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=[config.HOST, "localhost"])
app.add_middleware(LocalSessionMiddleware)


class StrictBody(BaseModel):
    model_config = ConfigDict(extra="forbid")


class VoiceVariantCreate(StrictBody):
    name: str = Field(min_length=1, max_length=80)
    description: str | None = Field(default=None, max_length=500)


class VoiceVariantUpdate(StrictBody):
    name: str | None = Field(default=None, min_length=1, max_length=80)
    description: str | None = Field(default=None, max_length=500)
    state: Literal["active", "archived"] | None = None


class RecordingSessionCreate(StrictBody):
    variant_id: str = Field(min_length=1, max_length=100)
    mode: Literal["guided", "interview", "hybrid"] = "hybrid"


class CustomScriptCreate(StrictBody):
    variant_id: str = Field(min_length=1, max_length=100)
    text: str = Field(min_length=1, max_length=20_000)
    title: str | None = Field(default=None, max_length=100)


class CustomScriptPromptAction(StrictBody):
    variant_id: str = Field(min_length=1, max_length=100)


class RecordingReview(StrictBody):
    variant_id: str = Field(min_length=1, max_length=100)
    status: Literal["approved", "quarantined", "rejected", "pending"]
    transcript: str | None = Field(default=None, max_length=20_000)
    transcript_confirmed: bool = False
    playback_confirmed: bool = False
    note: str | None = Field(default=None, max_length=500)


class BulkRecordingApproval(StrictBody):
    variant_id: str = Field(min_length=1, max_length=100)
    confirm_playback: Literal[True]
    confirm_transcripts: Literal[True]


class BulkRecordingReview(StrictBody):
    variant_id: str = Field(min_length=1, max_length=100)
    recording_ids: list[str] = Field(min_length=1, max_length=500)
    status: Literal["approved", "quarantined", "rejected"]
    confirm_playback: bool = False
    confirm_transcripts: bool = False
    override_failed_qa: bool = False
    transcripts: dict[str, str] = Field(default_factory=dict)


class DatasetFreeze(StrictBody):
    variant_id: str = Field(min_length=1, max_length=100)
    name: str | None = Field(default=None, max_length=80)
    purpose: Literal["pilot", "full"] | None = None


class TrainingRunRequest(StrictBody):
    variant_id: str = Field(min_length=1, max_length=100)
    dataset_version: str = Field(min_length=1, max_length=160)
    stage: Literal["pilot", "full"]
    label: str | None = Field(default=None, max_length=48)
    epochs: int = Field(ge=1, le=3)


class BlindVote(StrictBody):
    prompt_id: str = Field(min_length=1, max_length=80)
    choice: Literal["A", "B", "tie"]


DELIVERY_INSTRUCTIONS = {
    "natural": None,
    "warm": "Use a warm, natural narration style.",
    "clear": "Speak clearly and precisely while remaining conversational.",
    "measured": "Use a measured, unhurried narration pace.",
}


class TrainingJobs:
    """One serialized MLX subprocess with safe-boundary cancellation."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active_run: str | None = None
        self._processes: dict[str, subprocess.Popen] = {}
        self._cancel_files: dict[str, Path] = {}

    @property
    def active_run(self) -> str | None:
        with self._lock:
            return self._active_run

    def start(self, run: dict[str, Any], dataset: dict[str, Any], cfg: TrainingConfig) -> None:
        with self._lock:
            if self._active_run is not None:
                raise ValueError(f"Training run {self._active_run} is already active")
            run_id = str(run["id"])
            control = config.CACHE_ROOT / "jobs" / run_id
            control.mkdir(parents=True, exist_ok=False, mode=0o700)
            progress_path = control / "progress.jsonl"
            result_path = control / "result.json"
            failure_path = control / "failure.json"
            cancel_path = control / "cancel.requested"
            log_path = control / "worker.log"
            environment = dict(os.environ)
            environment.update(
                {
                    "HF_HUB_OFFLINE": "1",
                    "HF_DATASETS_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                    "HF_HUB_DISABLE_TELEMETRY": "1",
                    "DO_NOT_TRACK": "1",
                    "NO_PROXY": "127.0.0.1,localhost",
                }
            )
            for key in (
                "HF_TOKEN",
                "HUGGING_FACE_HUB_TOKEN",
                "OPENAI_API_KEY",
                "HTTP_PROXY",
                "HTTPS_PROXY",
                "ALL_PROXY",
            ):
                environment.pop(key, None)
            command = [
                sys.executable,
                "-m",
                "voice_trainer.training_worker",
                "--model",
                str(config.MODEL_ROOT),
                "--dataset",
                str(dataset["manifest_path"]),
                "--artifacts",
                str(store.paths.adapters / str(dataset["variant_id"]) / run_id),
                "--config",
                json.dumps(cfg.to_dict(), separators=(",", ":")),
                "--progress",
                str(progress_path),
                "--result",
                str(result_path),
                "--failure",
                str(failure_path),
                "--cancel",
                str(cancel_path),
            ]
            log = log_path.open("ab", buffering=0)
            try:
                process = subprocess.Popen(
                    command,
                    cwd=PROJECT_ROOT,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    close_fds=True,
                )
            except Exception:
                log.close()
                raise
            log.close()
            self._processes[run_id] = process
            self._cancel_files[run_id] = cancel_path
            self._active_run = run_id
        monitor = threading.Thread(
            target=self._monitor,
            args=(run_id, process, cfg, control, dataset),
            name=f"voicetrainer-monitor-{run_id}",
            daemon=True,
        )
        monitor.start()

    def cancel(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            cancel_path = self._cancel_files.get(run_id)
        run = store.get_training_run(run_id)
        if run["status"] not in {"pending", "running", "cancelling"}:
            return run
        if cancel_path is None:
            raise ValueError("This run is not active in the current local process")
        cancel_path.touch(exist_ok=True, mode=0o600)
        return store.update_training_run(
            run_id,
            status="cancelling",
            metadata={**run["metadata_json"], "message": "Cancellation requested"},
        )

    @staticmethod
    def _progress(event: dict[str, Any]) -> tuple[float, dict[str, Any]]:
        name = str(event.get("event", "working"))
        if name == "loading_model":
            progress = 0.01
        elif name in {"extracting_speaker", "speaker_embedding", "speaker_embedding_ready"}:
            progress = 0.04
        elif name == "training_step":
            done = float(event.get("microsteps_completed", 0))
            total = max(1.0, float(event.get("microsteps_total", 1)))
            progress = 0.05 + 0.90 * done / total
        elif name == "checkpoint_saved":
            progress = 0.96
        elif name == "training_complete":
            progress = 1.0
        else:
            progress = 0.03
        message = name.replace("_", " ").capitalize()
        metadata = {**event, "message": message, "current_step": message}
        return min(1.0, max(0.0, progress)), metadata

    def _monitor(
        self,
        run_id: str,
        process: subprocess.Popen,
        cfg: TrainingConfig,
        control: Path,
        dataset: dict[str, Any],
    ) -> None:
        started = time.monotonic()
        last_metadata: dict[str, Any] = {}
        progress_path = control / "progress.jsonl"
        result_path = control / "result.json"
        failure_path = control / "failure.json"
        position = 0
        try:
            store.update_training_run(
                run_id,
                status="running",
                progress=0.0,
                metadata={"message": "Starting native MLX training"},
            )
            while process.poll() is None:
                if progress_path.exists():
                    with progress_path.open("r", encoding="utf-8") as stream:
                        stream.seek(position)
                        for line in stream:
                            event = json.loads(line)
                            value, metadata = self._progress(event)
                            metadata["elapsed_seconds"] = round(time.monotonic() - started, 1)
                            last_metadata = metadata
                            store.update_training_run(
                                run_id, status="running", progress=value, metadata=metadata
                            )
                        position = stream.tell()
                time.sleep(0.4)
            # Consume any event written immediately before worker exit.
            if progress_path.exists():
                with progress_path.open("r", encoding="utf-8") as stream:
                    stream.seek(position)
                    for line in stream:
                        event = json.loads(line)
                        _, last_metadata = self._progress(event)
            if process.returncode != 0:
                failure = (
                    json.loads(failure_path.read_text(encoding="utf-8"))
                    if failure_path.exists()
                    else {"status": "failed", "error": f"worker exited {process.returncode}"}
                )
                final_status = "cancelled" if failure.get("status") == "cancelled" else "failed"
                store.update_training_run(
                    run_id,
                    status=final_status,
                    error=str(failure.get("error", "training worker failed")),
                    metadata={
                        **last_metadata,
                        "message": "Training cancelled" if final_status == "cancelled" else "Training failed",
                        "elapsed_seconds": round(time.monotonic() - started, 1),
                    },
                )
                return
            if not result_path.is_file():
                raise RuntimeError("training worker exited without a result artifact")
            result = json.loads(result_path.read_text(encoding="utf-8"))
            run_config = store.get_training_run(run_id)["config_json"]
            checkpoint_rows: list[dict[str, Any]] = []
            for artifact_value in result["checkpoints"]:
                artifact_path = Path(artifact_value)
                artifact = json.loads((artifact_path / "artifact.json").read_text(encoding="utf-8"))
                metrics = json.loads((artifact_path / "metrics.json").read_text(encoding="utf-8"))
                checkpoint_rows.append(
                    store.add_checkpoint(
                        run_id,
                        int(artifact["epoch"]),
                        artifact_path,
                        {
                            **metrics,
                            "stage": cfg.stage,
                            "label": run_config.get("label"),
                            "variant_id": dataset["variant_id"],
                            "evaluation_status": "pending",
                        },
                    )
                )
            final_metadata = {
                **last_metadata,
                "message": "Training complete; checkpoints preserved for evaluation",
                "result": result,
                "checkpoint_ids": [row["id"] for row in checkpoint_rows],
                "epoch": cfg.epochs,
                "total_epochs": cfg.epochs,
                "elapsed_seconds": round(time.monotonic() - started, 1),
            }
            store.update_training_run(
                run_id, status="complete", progress=1.0, metadata=final_metadata
            )
            for checkpoint in checkpoint_rows:
                evaluation_jobs.enqueue(checkpoint, dataset)
        except Exception as exc:
            store.update_training_run(
                run_id,
                status="failed",
                error=str(exc),
                metadata={**last_metadata, "message": "Training failed"},
            )
        finally:
            with self._lock:
                self._processes.pop(run_id, None)
                self._cancel_files.pop(run_id, None)
                if self._active_run == run_id:
                    self._active_run = None


class EvaluationJobs:
    """Serialize Base/candidate evaluations in fresh, offline processes."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._queue: list[tuple[dict[str, Any], dict[str, Any]]] = []
        self._running = False
        self._active_checkpoint: str | None = None

    @property
    def active_checkpoint(self) -> str | None:
        with self._lock:
            return self._active_checkpoint

    def enqueue(self, checkpoint: dict[str, Any], dataset: dict[str, Any]) -> None:
        with self._lock:
            checkpoint_id = str(checkpoint["id"])
            if self._active_checkpoint == checkpoint_id or any(
                queued[0]["id"] == checkpoint_id for queued in self._queue
            ):
                return
            self._queue.append((checkpoint, dataset))
            if self._running:
                return
            self._running = True
        threading.Thread(
            target=self._drain,
            name="voicetrainer-evaluation-queue",
            daemon=True,
        ).start()

    @staticmethod
    def _held_out_audio(dataset: dict[str, Any]) -> list[Path]:
        manifest_path = Path(dataset["manifest_path"]).resolve()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        held_out = [
            (manifest_path.parent / row["audio"]).resolve()
            for row in manifest.get("recordings", [])
            if row.get("split") == "test"
        ]
        if len(held_out) < 2:
            raise ValueError("Evaluation requires at least two locked test recordings")
        return held_out

    @staticmethod
    def _environment() -> dict[str, str]:
        environment = dict(os.environ)
        environment.update(
            {
                "HF_HUB_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_HUB_DISABLE_TELEMETRY": "1",
                "DO_NOT_TRACK": "1",
                "NO_PROXY": "127.0.0.1,localhost",
            }
        )
        for key in (
            "HF_TOKEN",
            "HUGGING_FACE_HUB_TOKEN",
            "OPENAI_API_KEY",
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
        ):
            environment.pop(key, None)
        return environment

    def _worker(
        self,
        *,
        mode: str,
        variant_id: str,
        output: Path,
        held_out: list[Path],
        control: Path,
        artifact: Path | None = None,
    ) -> Path:
        progress = control / f"{mode}-progress.jsonl"
        result = control / f"{mode}-result.json"
        failure = control / f"{mode}-failure.json"
        command = [
            sys.executable,
            "-m",
            "voice_trainer.evaluation_worker",
            "--mode",
            mode,
            "--variant-id",
            variant_id,
            "--model",
            str(config.MODEL_ROOT),
            "--output",
            str(output),
            "--progress",
            str(progress),
            "--result",
            str(result),
            "--failure",
            str(failure),
        ]
        if artifact is not None:
            command.extend(("--artifact", str(artifact)))
        for path in held_out:
            command.extend(("--held-out", str(path)))
        log_path = control / f"{mode}-worker.log"
        with log_path.open("ab", buffering=0) as log:
            completed = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                env=self._environment(),
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                close_fds=True,
                check=False,
            )
        if completed.returncode != 0:
            detail = (
                json.loads(failure.read_text(encoding="utf-8")).get("error")
                if failure.exists()
                else f"evaluation worker exited {completed.returncode}"
            )
            raise RuntimeError(str(detail))
        report_path = output / "evaluation.json"
        if not report_path.is_file():
            raise RuntimeError("evaluation worker did not preserve its report")
        return report_path

    def _evaluate(self, checkpoint: dict[str, Any], dataset: dict[str, Any]) -> None:
        from .evaluation import create_blind_comparison

        _unload_generation_runtime()
        checkpoint_id = str(checkpoint["id"])
        store.update_checkpoint(
            checkpoint_id,
            metrics={"evaluation_status": "running", "evaluation_started_at": time.time()},
        )
        held_out = self._held_out_audio(dataset)
        control = config.CACHE_ROOT / "evaluations" / checkpoint_id
        control.mkdir(parents=True, exist_ok=True, mode=0o700)
        attempt = time.strftime("%Y%m%dT%H%M%S") + f"-{uuid.uuid4().hex[:8]}"
        variant_id = str(dataset["variant_id"])
        baseline_root = (
            store.paths.evaluations
            / "variants"
            / variant_id
            / "baselines"
            / str(dataset["manifest_sha256"])
        )
        existing_baselines = sorted(baseline_root.glob("*/evaluation.json"))
        if existing_baselines:
            baseline_report_path = existing_baselines[-1]
        else:
            baseline_dir = baseline_root / attempt
            baseline_report_path = self._worker(
                mode="base",
                variant_id=variant_id,
                output=baseline_dir,
                held_out=held_out,
                control=control,
            )
        candidate_dir = (
            store.paths.evaluations
            / "variants"
            / variant_id
            / "checkpoints"
            / checkpoint_id
            / attempt
        )
        candidate_report_path = self._worker(
            mode="checkpoint",
            variant_id=variant_id,
            output=candidate_dir,
            held_out=held_out,
            control=control,
            artifact=Path(checkpoint["artifact_path"]),
        )
        baseline_report = json.loads(baseline_report_path.read_text(encoding="utf-8"))
        candidate_report = json.loads(candidate_report_path.read_text(encoding="utf-8"))
        _require_report_variant(baseline_report, variant_id, "Baseline evaluation")
        _require_report_variant(candidate_report, variant_id, "Candidate evaluation")
        packaged_evaluation = Path(checkpoint["artifact_path"]) / "evaluation.json"
        if not packaged_evaluation.exists():
            shutil.copyfile(candidate_report_path, packaged_evaluation)
            packaged_evaluation.chmod(0o400)
        previous_report_path = baseline_report_path
        for prior in store.list_checkpoints(variant_id=variant_id):
            if prior["status"] == "promoted" and prior["id"] != checkpoint_id:
                value = prior["metrics_json"].get("evaluation_report")
                if value and Path(value).is_file():
                    previous_report_path = Path(value)
                    break
        previous_report = json.loads(previous_report_path.read_text(encoding="utf-8"))
        _require_report_variant(previous_report, variant_id, "Previous evaluation")
        comparison_dir = candidate_dir / "blind-comparison"
        create_blind_comparison(
            candidate_report=candidate_report,
            previous_report=previous_report,
            output_dir=comparison_dir,
        )
        store.update_checkpoint(
            checkpoint_id,
            metrics={
                "evaluation_status": "awaiting_blind_review",
                "variant_id": variant_id,
                "evaluation_report": str(candidate_report_path),
                "baseline_report": str(baseline_report_path),
                "previous_report": str(previous_report_path),
                "comparison_manifest": str(comparison_dir / "comparison.json"),
                "comparison_key": str(comparison_dir / "comparison-key.private.json"),
                "evaluation": candidate_report["aggregate"],
                "blind_votes": {},
            },
        )

    def _drain(self) -> None:
        while True:
            with self._lock:
                if not self._queue:
                    self._running = False
                    self._active_checkpoint = None
                    return
                checkpoint, dataset = self._queue.pop(0)
                self._active_checkpoint = str(checkpoint["id"])
            try:
                self._evaluate(checkpoint, dataset)
            except Exception as exc:
                store.update_checkpoint(
                    str(checkpoint["id"]),
                    metrics={"evaluation_status": "failed", "evaluation_error": str(exc)},
                )


evaluation_jobs = EvaluationJobs()
jobs = TrainingJobs()
model_setup = ModelSetupJobs(PROJECT_ROOT)
_runtime_lock = threading.Lock()
_runtime_checkpoint: str | None = None
_runtime: NamedSpeakerRuntime | None = None


def _unload_generation_runtime() -> None:
    global _runtime, _runtime_checkpoint
    with _runtime_lock:
        if _runtime is not None:
            _runtime.unload()
        _runtime = None
        _runtime_checkpoint = None


def _checkpoint(checkpoint_id: str) -> dict[str, Any]:
    return store.get_checkpoint(checkpoint_id)


def _require_variant(row: dict[str, Any], variant_id: str, resource: str) -> dict[str, Any]:
    if str(row.get("variant_id")) != variant_id:
        raise VariantMismatchError(f"{resource} belongs to a different voice variant")
    return row


def _require_report_variant(
    report: dict[str, Any], variant_id: str, resource: str
) -> dict[str, Any]:
    # Schema-v1 reports predate voice variants and belong only to the migrated
    # Legacy default lineage. New reports always carry the immutable variant ID.
    report_variant = report.get("voice_variant_id") or DEFAULT_VARIANT_ID
    if str(report_variant) != variant_id:
        raise VariantMismatchError(f"{resource} belongs to a different voice variant")
    return report


def _require_custom_script_gate(variant_id: str) -> dict[str, Any]:
    summary = store.status(variant_id=variant_id)
    gate_seconds = float(
        summary.get("custom_script_gate_seconds", summary["accepted_seconds"])
    )
    if not bool(summary.get("custom_script_unlocked", gate_seconds >= 120.0)):
        raise ValueError(
            "My Script unlocks after this voice variant has two approved minutes"
        )
    return summary


def _require_models_ready() -> None:
    try:
        verify_model(full_hash=False)
        verify_asr_model(full_hash=False)
    except Exception as exc:
        raise ValueError(
            "Complete the one-time local model download before using Noice City"
        ) from exc


def _public_recording(row: dict[str, Any]) -> dict[str, Any]:
    value = dict(row)
    value["audio_url"] = (
        f"/api/recordings/{row['id']}/audio"
        f"?variant_id={row['variant_id']}"
    )
    return value


def _public_run(run: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(run.get("metadata_json") or {})
    return {
        **run,
        **{key: value for key, value in metadata.items() if key != "result"},
        "progress_percent": round(float(run["progress"]) * 100, 2),
    }


def _safe_http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, VariantMismatchError):
        return HTTPException(status_code=409, detail=str(exc))
    if isinstance(exc, KeyError):
        return HTTPException(status_code=404, detail=str(exc).strip("'"))
    if isinstance(exc, (ValueError, FileNotFoundError)):
        return HTTPException(status_code=400, detail=str(exc))
    return HTTPException(status_code=500, detail=str(exc))


@app.get("/api/variants")
def api_variants():
    return store.list_voice_variants()


@app.post("/api/variants")
def api_create_variant(body: VoiceVariantCreate):
    try:
        return store.create_voice_variant(body.name, description=body.description)
    except Exception as exc:
        raise _safe_http_error(exc) from exc


@app.patch("/api/variants/{variant_id}")
def api_update_variant(variant_id: str, body: VoiceVariantUpdate):
    try:
        return store.update_voice_variant(
            variant_id,
            name=body.name,
            description=body.description,
            state=body.state,
        )
    except Exception as exc:
        raise _safe_http_error(exc) from exc


@app.post("/api/recording-sessions")
def api_create_recording_session(body: RecordingSessionCreate):
    try:
        variant = store.get_voice_variant(body.variant_id)
        if variant["state"] != "active":
            raise ValueError("Archived voice variants cannot accept new recordings")
        return store.create_session(body.mode, variant_id=body.variant_id)
    except Exception as exc:
        raise _safe_http_error(exc) from exc


@app.post("/api/custom-scripts")
def api_create_custom_script(body: CustomScriptCreate):
    try:
        variant = store.get_voice_variant(body.variant_id)
        if variant["state"] != "active":
            raise ValueError("Archived voice variants cannot accept custom scripts")
        _require_custom_script_gate(body.variant_id)
        return store.create_custom_script(
            body.variant_id,
            body.text,
            title=body.title,
        )
    except Exception as exc:
        raise _safe_http_error(exc) from exc


@app.get("/api/custom-scripts/next")
def api_next_custom_script_prompt(
    variant_id: str = Query(..., min_length=1, max_length=100),
):
    try:
        _require_custom_script_gate(variant_id)
        prompt = store.next_custom_script_prompt(variant_id)
        if prompt is None:
            raise HTTPException(status_code=404, detail="No custom script takes are waiting")
        return prompt
    except HTTPException:
        raise
    except Exception as exc:
        raise _safe_http_error(exc) from exc


@app.post("/api/custom-scripts/prompts/{prompt_id}/skip")
def api_skip_custom_script_prompt(prompt_id: str, body: CustomScriptPromptAction):
    try:
        return store.skip_custom_script_prompt(body.variant_id, prompt_id)
    except Exception as exc:
        raise _safe_http_error(exc) from exc


@app.get("/api/model-setup")
def api_model_setup_status() -> dict[str, Any]:
    return model_setup.status()


@app.post("/api/model-setup")
def api_start_model_setup() -> dict[str, Any]:
    try:
        if jobs.active_run is not None or evaluation_jobs.active_checkpoint is not None:
            raise ValueError("Wait for local training and evaluation to finish")
        return model_setup.start()
    except Exception as exc:
        raise _safe_http_error(exc) from exc


@app.get("/api/status")
def api_status(variant_id: str = Query(DEFAULT_VARIANT_ID)) -> dict[str, Any]:
    summary = store.status(variant_id=variant_id)
    recordings = store.list_recordings(variant_id=variant_id)
    approved = [row for row in recordings if row["review_status"] == "approved"]
    kinds = {
        "guided": sum(row["kind"] == "guided" for row in approved),
        "interview": sum(row["kind"] == "interview" for row in approved),
    }
    try:
        verify_model(full_hash=False)
        model_ready = True
        model_error = None
    except Exception as exc:
        model_ready = False
        model_error = str(exc)
    try:
        verify_asr_model(full_hash=False)
        asr_ready = True
        asr_error = None
    except Exception as exc:
        asr_ready = False
        asr_error = str(exc)
    active_training_run = jobs.active_run
    if active_training_run is not None:
        try:
            store.get_training_run(active_training_run, variant_id=variant_id)
        except KeyError:
            active_training_run = None
    active_evaluation_checkpoint = evaluation_jobs.active_checkpoint
    if active_evaluation_checkpoint is not None:
        try:
            store.get_checkpoint(active_evaluation_checkpoint, variant_id=variant_id)
        except KeyError:
            active_evaluation_checkpoint = None
    return {
        **summary,
        "model_ready": model_ready,
        "model": {
            "ready": model_ready,
            "error": model_error,
            "revision": config.MODEL_REVISION,
            "path": str(config.MODEL_ROOT),
        },
        "asr": {"ready": asr_ready, "error": asr_error, "revision": config.ASR_REVISION},
        "dataset": {"accepted_seconds": summary["accepted_seconds"], "kinds": kinds},
        "variant": store.get_voice_variant(variant_id),
        "variants": store.list_voice_variants(),
        "datasets": store.list_datasets(variant_id=variant_id),
        "checkpoints": store.list_checkpoints(variant_id=variant_id),
        # Job IDs are lineage-scoped. A different voice may still occupy the
        # serialized MLX worker, but its resource identity is never disclosed
        # through this variant's status response.
        "active_training_run": active_training_run,
        "active_evaluation_checkpoint": active_evaluation_checkpoint,
        "local_worker_busy": bool(jobs.active_run or evaluation_jobs.active_checkpoint),
        "model_setup": model_setup.status(),
        "setup_network_active": model_setup.active,
        "offline": not model_setup.active,
        "runtime_offline": True,
        "bind": f"{config.HOST}:{config.PORT}",
    }


@app.get("/api/prompts/next")
def api_next_prompt(
    session_id: str = Query(min_length=1, max_length=100),
    kind: Literal["guided", "interview", "hybrid"] = Query("hybrid"),
    exclude: list[str] = Query(default=[]),
):
    try:
        session = store.get_session(session_id)
        if session["status"] != "active":
            raise ValueError("Recording session is no longer active")
        prompt = store.next_prompt(
            kind,
            session_id=session_id,
            exclude_prompt_ids=tuple(exclude[:100]),
        )
        if prompt is None:
            raise HTTPException(status_code=404, detail="No prompt is available")
        return prompt
    except HTTPException:
        raise
    except Exception as exc:
        raise _safe_http_error(exc) from exc


@app.post("/api/recordings")
async def api_add_recording(
    audio: UploadFile = File(...),
    prompt_id: str = Form(...),
    session_id: str = Form(...),
    prompt_kind: str | None = Form(default=None),
    expected_text: str | None = Form(default=None),
    expected_transcript: str | None = Form(default=None),
    sample_rate: int | None = Form(default=None),
    bit_depth: int | None = Form(default=None),
    duration_seconds: float | None = Form(default=None),
    split: str | None = Form(default=None),
):
    try:
        _require_models_ready()
    except Exception as exc:
        raise _safe_http_error(exc) from exc
    del prompt_kind, expected_text, expected_transcript, duration_seconds, split
    if sample_rate not in {None, 48_000} or bit_depth not in {None, 24}:
        raise HTTPException(status_code=400, detail="Recordings must be 48 kHz mono PCM24 WAV")
    payload = await audio.read(MAX_RECORDING_BYTES + 1)
    if len(payload) > MAX_RECORDING_BYTES:
        raise HTTPException(status_code=413, detail="Recording exceeds the local intake limit")
    if not payload:
        raise HTTPException(status_code=400, detail="Recording is empty")

    intake = config.CACHE_ROOT / "intake"
    intake.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary_name = tempfile.mkstemp(prefix="take-", suffix=".wav", dir=intake)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        session = store.get_session(session_id)
        prompt = store.get_prompt(prompt_id, variant_id=str(session["variant_id"]))
        expected = str(prompt["text"]) if prompt["kind"] == "guided" else ""
        transcription = await asyncio.to_thread(transcribe_local, temporary, expected)
        row = await asyncio.to_thread(
            store.add_recording,
            prompt_id,
            session_id=session_id,
            master_bytes=payload,
            transcript=transcription["text"],
        )
        return _public_recording(row)
    except Exception as exc:
        raise _safe_http_error(exc) from exc
    finally:
        temporary.unlink(missing_ok=True)


@app.get("/api/recordings")
def api_recordings(
    variant_id: str = Query(..., min_length=1, max_length=100),
    status: str | None = None,
):
    try:
        _require_models_ready()
        return [
            _public_recording(row)
            for row in store.list_recordings(status, variant_id=variant_id)
        ]
    except Exception as exc:
        raise _safe_http_error(exc) from exc


@app.post("/api/recordings/approve-all")
def api_approve_all_recordings(body: BulkRecordingApproval):
    try:
        return store.approve_all_pending(variant_id=body.variant_id)
    except Exception as exc:
        raise _safe_http_error(exc) from exc


@app.post("/api/recordings/bulk")
def api_bulk_review_recordings(body: BulkRecordingReview):
    try:
        rows = [store.get_recording(recording_id) for recording_id in body.recording_ids]
        for row in rows:
            _require_variant(row, body.variant_id, "Recording")
        if body.status == "approved" and not (
            body.confirm_playback and body.confirm_transcripts
        ):
            raise ValueError(
                "Bulk approval requires confirmation of playback and transcripts"
            )
        transcripts = {
            recording_id: transcript.strip()
            for recording_id, transcript in body.transcripts.items()
            if recording_id in body.recording_ids and transcript.strip()
        }
        if any(len(transcript) > 20_000 for transcript in transcripts.values()):
            raise ValueError("A confirmed transcript exceeds 20,000 characters")
        return store.bulk_update_recordings(
            body.recording_ids,
            body.status,
            transcripts=transcripts,
            confirm_review=body.status == "approved",
            override_failed_qa=body.status == "approved" and body.override_failed_qa,
            variant_id=body.variant_id,
        )
    except Exception as exc:
        raise _safe_http_error(exc) from exc


@app.get("/api/recordings/{recording_id}/audio")
def api_recording_audio(
    recording_id: str,
    variant_id: str = Query(..., min_length=1, max_length=100),
):
    try:
        row = store.get_recording(recording_id, variant_id=variant_id)
        return FileResponse(
            row["master_path"], media_type="audio/wav", filename=f"{recording_id}-master.wav"
        )
    except Exception as exc:
        raise _safe_http_error(exc) from exc


@app.patch("/api/recordings/{recording_id}")
def api_review_recording(recording_id: str, body: RecordingReview):
    try:
        _require_variant(store.get_recording(recording_id), body.variant_id, "Recording")
        transcript = body.transcript if body.transcript_confirmed else None
        return _public_recording(
            store.update_recording(
                recording_id,
                body.status,
                transcript,
                note=body.note,
                override_qa=False,
                playback_confirmed=body.playback_confirmed,
            )
        )
    except Exception as exc:
        raise _safe_http_error(exc) from exc


@app.get("/api/datasets")
def api_datasets(variant_id: str = Query(..., min_length=1, max_length=100)):
    return store.list_datasets(variant_id=variant_id)


@app.post("/api/datasets/freeze")
def api_freeze_dataset(body: DatasetFreeze):
    try:
        variant = store.get_voice_variant(body.variant_id)
        if variant["state"] != "active":
            raise ValueError("Archived voice variants cannot freeze new datasets")
        accepted = float(store.status(variant_id=body.variant_id)["accepted_seconds"])
        purpose = body.purpose or ("full" if accepted >= 1800 else "pilot")
        name = body.name or time.strftime(f"voice-{purpose}-%Y%m%d-%H%M%S")
        return store.freeze_dataset(name, purpose=purpose, variant_id=body.variant_id)
    except Exception as exc:
        raise _safe_http_error(exc) from exc


@app.get("/api/checkpoints")
def api_checkpoints(variant_id: str = Query(..., min_length=1, max_length=100)):
    return store.list_checkpoints(variant_id=variant_id)


@app.get("/api/checkpoints/{checkpoint_id}/comparison")
def api_checkpoint_comparison(
    checkpoint_id: str,
    variant_id: str = Query(..., min_length=1, max_length=100),
):
    try:
        checkpoint = store.get_checkpoint(checkpoint_id, variant_id=variant_id)
        manifest_value = checkpoint["metrics_json"].get("comparison_manifest")
        if not manifest_value:
            raise ValueError("This checkpoint has not completed objective evaluation")
        manifest = json.loads(Path(manifest_value).read_text(encoding="utf-8"))
        public = dict(manifest)
        public["pairs"] = [
            {
                **pair,
                "option_a_audio": f"/api/checkpoints/{checkpoint_id}/comparison/{pair['pair_id']}/A?variant_id={variant_id}",
                "option_b_audio": f"/api/checkpoints/{checkpoint_id}/comparison/{pair['pair_id']}/B?variant_id={variant_id}",
            }
            for pair in manifest["pairs"]
        ]
        public["votes"] = checkpoint["metrics_json"].get("blind_votes", {})
        return public
    except Exception as exc:
        raise _safe_http_error(exc) from exc


@app.post("/api/checkpoints/{checkpoint_id}/evaluations")
def api_retry_checkpoint_evaluation(
    checkpoint_id: str,
    variant_id: str = Query(..., min_length=1, max_length=100),
):
    try:
        if jobs.active_run is not None:
            raise ValueError("Wait for training to finish before evaluating")
        checkpoint = store.get_checkpoint(checkpoint_id, variant_id=variant_id)
        run = store.get_training_run(
            checkpoint["training_run_id"], variant_id=variant_id
        )
        dataset = store.dataset(run["dataset_id"])
        _require_variant(dataset, variant_id, "Dataset")
        evaluation_jobs.enqueue(checkpoint, dataset)
        return {"checkpoint_id": checkpoint_id, "status": "queued"}
    except Exception as exc:
        raise _safe_http_error(exc) from exc


@app.get("/api/checkpoints/{checkpoint_id}/comparison/{pair_id}/{option}")
def api_checkpoint_comparison_audio(
    checkpoint_id: str,
    pair_id: str,
    option: Literal["A", "B"],
    variant_id: str = Query(..., min_length=1, max_length=100),
):
    try:
        checkpoint = store.get_checkpoint(checkpoint_id, variant_id=variant_id)
        manifest_path = Path(checkpoint["metrics_json"]["comparison_manifest"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        pair = next(row for row in manifest["pairs"] if row["pair_id"] == pair_id)
        path = Path(pair[f"option_{option.lower()}_audio"]).resolve()
        if not path.is_file() or manifest_path.parent.resolve() not in path.parents:
            raise ValueError("Comparison audio is unavailable")
        return FileResponse(path, media_type="audio/wav")
    except StopIteration as exc:
        raise HTTPException(status_code=404, detail="Unknown blinded comparison pair") from exc
    except Exception as exc:
        raise _safe_http_error(exc) from exc


@app.post("/api/checkpoints/{checkpoint_id}/blind-votes")
def api_checkpoint_vote(
    checkpoint_id: str,
    body: BlindVote,
    variant_id: str = Query(..., min_length=1, max_length=100),
):
    try:
        from .evaluation import promotion_decision, resolve_blind_votes

        checkpoint = store.get_checkpoint(checkpoint_id, variant_id=variant_id)
        metrics = checkpoint["metrics_json"]
        required = ("evaluation_report", "baseline_report", "previous_report", "comparison_key")
        if any(not metrics.get(key) for key in required):
            raise ValueError("This checkpoint is not ready for blinded review")
        option_votes = dict(metrics.get("blind_votes") or {})
        option_votes[body.prompt_id] = body.choice
        key = json.loads(Path(metrics["comparison_key"]).read_text(encoding="utf-8"))
        resolved = resolve_blind_votes(private_key=key, option_votes=option_votes)
        variant_id = str(checkpoint["variant_id"])
        candidate_report = json.loads(
            Path(metrics["evaluation_report"]).read_text(encoding="utf-8")
        )
        baseline_report = json.loads(
            Path(metrics["baseline_report"]).read_text(encoding="utf-8")
        )
        previous_report = json.loads(
            Path(metrics["previous_report"]).read_text(encoding="utf-8")
        )
        _require_report_variant(candidate_report, variant_id, "Candidate evaluation")
        _require_report_variant(baseline_report, variant_id, "Baseline evaluation")
        _require_report_variant(previous_report, variant_id, "Previous evaluation")
        decision = promotion_decision(
            candidate_report=candidate_report,
            baseline_report=baseline_report,
            previous_report=previous_report,
            blind_votes=resolved,
        )
        next_status = (
            "promoted" if decision["status"] == "promoted" else
            "rejected" if decision["status"] == "rejected" else None
        )
        updated = store.update_checkpoint(
            checkpoint_id,
            status=next_status,
            metrics={
                "blind_votes": option_votes,
                "promotion_decision": decision,
                "evaluation_status": decision["status"],
            },
        )
        if next_status is not None:
            promotion_path = Path(checkpoint["artifact_path"]) / "promotion.json"
            if not promotion_path.exists():
                temporary = promotion_path.with_name(".promotion.json.partial")
                temporary.write_text(
                    json.dumps(decision, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                    encoding="utf-8",
                )
                temporary.chmod(0o400)
                os.replace(temporary, promotion_path)
        return {"checkpoint": updated, "decision": decision}
    except Exception as exc:
        raise _safe_http_error(exc) from exc


@app.post("/api/training-runs")
def api_create_training_run(body: TrainingRunRequest):
    try:
        _require_models_ready()
        variant = store.get_voice_variant(body.variant_id)
        if variant["state"] != "active":
            raise ValueError("Archived voice variants cannot start training")
        if evaluation_jobs.active_checkpoint is not None:
            raise ValueError("Wait for checkpoint evaluation to finish before training")
        unresolved_checkpoint = next(
            (
                row
                for row in store.list_checkpoints(variant_id=body.variant_id)
                if row["status"] == "candidate"
                and row["metrics_json"].get("evaluation_status")
                not in {"promoted", "rejected", "failed"}
            ),
            None,
        )
        if unresolved_checkpoint is not None:
            raise ValueError(
                "Finish the current checkpoint evaluation and blinded review before training again"
            )
        if body.stage == "full" and body.epochs != 3:
            raise ValueError("Full training requires exactly three versioned epochs")
        if body.stage == "pilot" and body.epochs != 1:
            raise ValueError("Pilot training uses one small checkpoint")
        if body.stage == "full":
            promoted_pilot = any(
                row["metrics_json"].get("stage") == "pilot"
                and (
                    row["status"] == "promoted"
                    or row["metrics_json"].get("promotion_decision", {}).get("status") == "promoted"
                )
                for row in store.list_checkpoints(variant_id=body.variant_id)
            )
            if not promoted_pilot:
                raise ValueError(
                    "Promote a recognizable, reference-free pilot before full training"
                )
        dataset = store.dataset(body.dataset_version)
        _require_variant(dataset, body.variant_id, "Dataset")
        if dataset["purpose"] != body.stage:
            raise ValueError(f"Select a frozen {body.stage} dataset for {body.stage} training")
        cfg = TrainingConfig(stage=body.stage, epochs=body.epochs)
        _unload_generation_runtime()
        run = store.create_training_run(
            body.dataset_version,
            {**cfg.to_dict(), "label": body.label},
        )
        try:
            jobs.start(run, dataset, cfg)
        except Exception as exc:
            store.update_training_run(
                run["id"], status="failed", error=str(exc), metadata={"message": "Worker did not start"}
            )
            raise
        return _public_run(run)
    except Exception as exc:
        raise _safe_http_error(exc) from exc


@app.get("/api/training-runs/{run_id}")
def api_training_run(
    run_id: str,
    variant_id: str = Query(..., min_length=1, max_length=100),
):
    try:
        return _public_run(store.get_training_run(run_id, variant_id=variant_id))
    except Exception as exc:
        raise _safe_http_error(exc) from exc


@app.delete("/api/training-runs/{run_id}")
def api_cancel_training(
    run_id: str,
    variant_id: str = Query(..., min_length=1, max_length=100),
):
    try:
        store.get_training_run(run_id, variant_id=variant_id)
        return _public_run(jobs.cancel(run_id))
    except Exception as exc:
        raise _safe_http_error(exc) from exc


def _generate(body: GenerationRequest, output: Path) -> dict[str, Any]:
    global _runtime, _runtime_checkpoint
    checkpoint = _checkpoint(body.checkpoint_version)
    _require_variant(checkpoint, body.variant_id, "Checkpoint")
    if checkpoint["status"] == "rejected":
        raise ValueError("Rejected checkpoints are retained for rollback records, not generation")
    artifact_path = Path(checkpoint["artifact_path"])
    with _runtime_lock:
        if _runtime is None or _runtime_checkpoint != checkpoint["id"]:
            if _runtime is not None:
                _runtime.unload()
            _runtime = NamedSpeakerRuntime(
                config.MODEL_ROOT,
                artifact_path,
                verify_model=True,
                expected_variant_id=body.variant_id,
            )
            _runtime_checkpoint = checkpoint["id"]
        return _runtime.generate(
            body.text,
            output,
            {"instruction": DELIVERY_INSTRUCTIONS[body.delivery]},
            seed=body.seed,
        )


@app.post("/api/generations")
async def api_generate(body: GenerationRequest):
    try:
        _require_models_ready()
    except Exception as exc:
        raise _safe_http_error(exc) from exc
    if jobs.active_run is not None:
        raise HTTPException(status_code=409, detail="Wait for local training to finish before generating")
    if evaluation_jobs.active_checkpoint is not None:
        raise HTTPException(status_code=409, detail="Wait for checkpoint evaluation to finish before generating")
    generation_root = store.paths.generations / body.variant_id
    generation_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    output = generation_root / f"generation-{uuid.uuid4().hex}.wav"
    try:
        metrics = await asyncio.to_thread(_generate, body, output)
        generation = store.add_generation(
            body.checkpoint_version,
            body.text,
            output,
            {"preset": body.delivery, "seed": body.seed},
            status="complete",
            metrics=metrics,
        )
        response_headers = {"X-VoiceTrainer-Generation-Id": generation["id"]}
        if body.seed is not None:
            response_headers["X-VoiceTrainer-Seed"] = str(body.seed)
        return FileResponse(
            output,
            media_type="audio/wav",
            filename="voice-generated.wav",
            headers=response_headers,
        )
    except Exception as exc:
        output.unlink(missing_ok=True)
        raise _safe_http_error(exc) from exc


@app.get("/")
def index():
    return FileResponse(STATIC_ROOT / "index.html", media_type="text/html")


@app.get("/recorder-worklet.js")
def recorder_worklet():
    return FileResponse(STATIC_ROOT / "recorder-worklet.js", media_type="text/javascript")


app.mount("/static", StaticFiles(directory=STATIC_ROOT), name="static")
