from __future__ import annotations

import asyncio
import importlib
import io
import math
import sys
import wave
from pathlib import Path

import pytest
from fastapi import HTTPException, UploadFile

from voice_trainer.store import DEFAULT_VARIANT_ID, VoiceStore


def _speech_wav() -> bytes:
    sample_rate = 48_000
    frame_count = sample_rate * 4
    silence_frames = sample_rate // 4
    values: list[float] = []
    for index in range(frame_count):
        if index < silence_frames or index >= frame_count - silence_frames:
            values.append(0.0)
            continue
        time_value = index / sample_rate
        envelope = 0.25 + 0.75 * math.sin(2 * math.pi * 2.3 * time_value) ** 2
        values.append(
            envelope
            * (
                math.sin(2 * math.pi * 125 * time_value)
                + 0.46 * math.sin(2 * math.pi * 250 * time_value + 0.2)
                + 0.30 * math.sin(2 * math.pi * 500 * time_value + 0.4)
                + 0.24 * math.sin(2 * math.pi * 750 * time_value + 0.7)
                + 0.18 * math.sin(2 * math.pi * 1_350 * time_value + 0.1)
                + 0.13 * math.sin(2 * math.pi * 2_300 * time_value + 0.6)
            )
        )
    scale = 2_560_000 / max(abs(value) for value in values)
    samples = [round(value * scale) for value in values]
    pcm = b"".join(
        ((sample if sample >= 0 else sample + (1 << 24)) & 0xFFFFFF).to_bytes(3, "little")
        for sample in samples
    )
    result = io.BytesIO()
    with wave.open(result, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(3)
        output.setframerate(sample_rate)
        output.writeframes(pcm)
    return result.getvalue()


def _isolated_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Importing the application initializes its store, so redirect every
    # configured path before that import. This test never opens the user's real
    # Application Support tree.
    import voice_trainer.config as config

    app_root = tmp_path / "app"
    cache_root = tmp_path / "cache"
    log_root = tmp_path / "logs"
    model_root = app_root / "models" / "qwen" / "test"
    asr_root = app_root / "models" / "whisper" / "test"
    replacements = {
        "APP_ROOT": app_root,
        "CACHE_ROOT": cache_root,
        "LOG_ROOT": log_root,
        "MODEL_ROOT": model_root,
        "MODEL_MANIFEST": model_root / "manifest.json",
        "ASR_ROOT": asr_root,
        "ASR_MANIFEST": asr_root / "manifest.json",
        "PROFILES_ROOT": app_root / "profiles",
        "DATASETS_ROOT": app_root / "datasets",
        "STATE_DB": app_root / "state.sqlite",
    }
    for name, value in replacements.items():
        monkeypatch.setattr(config, name, value)
    monkeypatch.setenv("VOICETRAINER_SESSION_TOKEN", "a" * 64)
    sys.modules.pop("voice_trainer.app", None)
    module = importlib.import_module("voice_trainer.app")
    module.store = VoiceStore(app_root).initialize()
    monkeypatch.setattr(module, "_require_voice_model_ready", lambda: None)
    return module


def _post_recording(module, *, session_id: str, prompt_id: str) -> dict:
    return asyncio.run(
        module.api_add_recording(
            audio=UploadFile(io.BytesIO(_speech_wav()), filename="take.wav"),
            prompt_id=prompt_id,
            session_id=session_id,
            prompt_kind=None,
            expected_text=None,
            expected_transcript=None,
            sample_rate=48_000,
            bit_depth=24,
            duration_seconds=None,
            split=None,
        )
    )


def test_recording_api_preserves_take_when_whisper_is_missing_or_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _isolated_app(tmp_path, monkeypatch)
    prompt = next(prompt for prompt in module.store.catalog.prompts if prompt.kind == "guided")

    missing_session = module.store.create_session("guided")
    monkeypatch.setattr(module, "_speech_checks_ready", lambda: False)
    monkeypatch.setattr(
        module,
        "transcribe_local",
        lambda *_args, **_kwargs: pytest.fail("ASR must not run when Whisper is missing"),
    )
    missing = _post_recording(
        module, session_id=missing_session["id"], prompt_id=prompt.id
    )
    assert missing["transcript_status"] == "pending"
    assert missing["qa_passed"] is True
    assert missing["qa_needs_review"] is True
    assert missing["intake_warnings"] == [
        {
            "code": "whisper_not_installed",
            "message": (
                "Whisper is not installed. The audio was saved; confirm its transcript manually."
            ),
            "action": "confirm_transcript_manually",
        }
    ]
    assert Path(missing["master_path"]).is_file()
    assert Path(missing["training_path"]).is_file()

    failed_session = module.store.create_session("guided")
    monkeypatch.setattr(module, "_speech_checks_ready", lambda: True)

    def fail_transcription(*_args, **_kwargs):
        raise RuntimeError("synthetic local ASR failure")

    monkeypatch.setattr(module, "transcribe_local", fail_transcription)
    failed = _post_recording(
        module, session_id=failed_session["id"], prompt_id=prompt.id
    )
    assert failed["transcript_status"] == "pending"
    assert failed["qa_passed"] is True
    assert failed["qa_needs_review"] is True
    assert failed["qa_json"]["transcription"] == {
        "status": "pending",
        "reason": "whisper_transcription_failed",
        "manual_confirmation_required": True,
    }
    assert failed["intake_warnings"][0]["code"] == "whisper_transcription_failed"
    assert Path(failed["master_path"]).is_file()
    assert Path(failed["training_path"]).is_file()
    assert len(module.store.list_recordings(variant_id=DEFAULT_VARIANT_ID)) == 2

    # Training remains a stricter boundary: mandatory checkpoint WER/evaluation
    # cannot proceed until the optional-at-record-time Whisper pack is present.
    monkeypatch.setattr(module, "_speech_checks_ready", lambda: False)
    with pytest.raises(HTTPException) as error:
        module.api_create_training_run(
            module.TrainingRunRequest(
                variant_id=DEFAULT_VARIANT_ID,
                dataset_version="not-reached",
                stage="pilot",
                epochs=1,
            )
        )
    assert error.value.status_code == 400
    assert "Install Whisper before training" in str(error.value.detail)
