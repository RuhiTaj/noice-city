from __future__ import annotations

import hashlib
import json
import wave
from pathlib import Path

import pytest

from voice_trainer.model_runtime import DeliverySettings
from voice_trainer.training import (
    DEFAULT_LORA_TARGETS,
    DatasetItem,
    TrainingConfig,
    TrainingError,
    fingerprint_model,
    load_frozen_dataset,
    select_speaker_reference,
    validate_local_model_path,
)


def _write_silence(path: Path, seconds: float, sample_rate: int = 24_000) -> None:
    frames = int(seconds * sample_rate)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x00\x00" * frames)


def test_training_defaults_lock_named_speaker_objective() -> None:
    config = TrainingConfig()

    assert config.speaker_name == "ruhi"
    assert config.speaker_slot == 3000
    assert config.epochs == 3
    assert config.batch_size == 1
    assert config.gradient_accumulation_steps == 4
    assert config.gradient_checkpointing is True
    assert config.learning_rate == 1e-5
    assert config.seed == 3407
    assert config.lora_rank == 16
    assert config.lora_alpha == 16
    assert config.lora_scale == 1.0
    assert config.lora_targets == DEFAULT_LORA_TARGETS
    assert config.sub_talker_loss_weight == 0.3
    assert config.minimum_dataset_seconds == 1800.0


def test_training_config_round_trip_and_pilot_minimum() -> None:
    original = TrainingConfig(stage="pilot", epochs=1)
    restored = TrainingConfig.from_dict(original.to_dict())

    assert restored == original
    assert restored.minimum_dataset_seconds == 120.0


@pytest.mark.parametrize(
    "values",
    [
        {"speaker_slot": 2999},
        {"speaker_name": "Ruhi"},
        {"lora_rank": 8},
        {"lora_alpha": 32},
        {"sub_talker_loss_weight": 1.0},
        {"batch_size": 2},
        {"language": "auto"},
        {"lora_targets": DEFAULT_LORA_TARGETS[:-1]},
    ],
)
def test_training_config_rejects_incompatible_artifacts(values: dict) -> None:
    with pytest.raises(ValueError):
        TrainingConfig(**values)


def test_delivery_settings_have_no_reference_audio_surface() -> None:
    fields = set(DeliverySettings.__dataclass_fields__)
    assert "ref_audio" not in fields
    assert "ref_text" not in fields
    assert DeliverySettings.from_value({"temperature": 0.7}).temperature == 0.7

    with pytest.raises(ValueError, match="does not accept reference"):
        DeliverySettings.from_value({"reference_audio": "/tmp/voice.wav"})
    with pytest.raises(ValueError, match="unsupported delivery"):
        DeliverySettings.from_value({"speed": 1.2})


def test_frozen_dataset_reads_plain_manifest_and_selects_clean_reference(
    tmp_path: Path,
) -> None:
    audio_a = tmp_path / "a.wav"
    audio_b = tmp_path / "b.wav"
    _write_silence(audio_a, 7.0)
    _write_silence(audio_b, 8.0)
    manifest = tmp_path / "dataset.json"
    manifest.write_text(
        json.dumps(
            {
                "frozen": True,
                "version": "dataset-001",
                "recordings": [
                    {
                        "audio": audio_a.name,
                        "training_text": "The first approved sentence.",
                        "audio_sha256": hashlib.sha256(audio_a.read_bytes()).hexdigest(),
                        "split": "train",
                        "status": "approved",
                        "qa": {"snr_db": 27.0},
                    },
                    {
                        "audio": audio_b.name,
                        "transcript": "The cleaner approved sentence.",
                        "split": "train",
                        "status": "approved",
                        "qa": {"snr_db": 35.0},
                    },
                    {
                        "audio": audio_b.name,
                        "transcript": "Do not use this rejected take.",
                        "split": "train",
                        "status": "rejected",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    items, metadata = load_frozen_dataset(manifest)

    assert len(items) == 2
    assert metadata["approved_items"] == 2
    assert metadata["sha256"]
    assert select_speaker_reference(items).audio_path == audio_b


def test_reference_selection_requires_six_to_ten_seconds(tmp_path: Path) -> None:
    item = DatasetItem(tmp_path / "short.wav", "short", "train", 4.0)
    with pytest.raises(TrainingError, match="6–10 second"):
        select_speaker_reference([item])


def test_local_model_validation_and_fingerprint_are_offline(tmp_path: Path) -> None:
    model = tmp_path / "model"
    tokenizer = model / "speech_tokenizer"
    tokenizer.mkdir(parents=True)
    (model / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_tts",
                "tts_model_type": "base",
                "tts_model_size": "1b7",
                "talker_config": {"vocab_size": 3072},
            }
        ),
        encoding="utf-8",
    )
    (model / "model.safetensors").write_bytes(b"local model")
    (tokenizer / "model.safetensors").write_bytes(b"local tokenizer")
    (tokenizer / "config.json").write_text("{}", encoding="utf-8")

    assert validate_local_model_path(model) == model.resolve()
    fingerprint = fingerprint_model(model)
    assert fingerprint["path"] == str(model.resolve())
    assert set(fingerprint["files"]) == {
        "config.json",
        "model.safetensors",
        "speech_tokenizer/config.json",
        "speech_tokenizer/model.safetensors",
    }


def test_local_model_path_must_be_absolute() -> None:
    with pytest.raises(TrainingError, match="absolute local"):
        validate_local_model_path("relative/model")
