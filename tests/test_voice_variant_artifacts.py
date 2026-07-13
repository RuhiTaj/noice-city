from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from voice_trainer.model_runtime import (
    ArtifactError,
    NamedSpeakerArtifact,
    NamedSpeakerRuntime,
)
from voice_trainer.training import (
    ARTIFACT_KIND,
    ARTIFACT_SCHEMA_VERSION,
    DEFAULT_VARIANT_ID,
    LEGACY_ARTIFACT_SCHEMA_VERSION,
    TrainingConfig,
    TrainingError,
    resolve_dataset_voice_variant,
    resolve_training_variant,
    stable_speaker_key,
)


def _write_local_model(path: Path) -> Path:
    tokenizer = path / "speech_tokenizer"
    tokenizer.mkdir(parents=True)
    (path / "config.json").write_text(
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
    (path / "model.safetensors").write_bytes(b"model")
    (tokenizer / "model.safetensors").write_bytes(b"tokenizer")
    return path.resolve()


def _canonical_variant(variant_id: str = "variant_warm") -> dict[str, str]:
    return {
        "variant_id": variant_id,
        "profile": "ruhi",
        "display_name": "Warm Narration",
        "language": "english",
        "speaker_key": stable_speaker_key("ruhi", variant_id),
        "accent_descriptor": "Warm, measured English narration",
    }


def _write_artifact(
    path: Path,
    model_path: Path,
    *,
    schema_version: int,
    voice_variant: dict[str, str] | None = None,
    mutate: Any = None,
) -> Path:
    path.mkdir()
    if schema_version == LEGACY_ARTIFACT_SCHEMA_VERSION:
        speaker_name = "ruhi"
        config = TrainingConfig(stage="pilot", epochs=1)
    else:
        voice_variant = dict(voice_variant or _canonical_variant())
        speaker_name = voice_variant["speaker_key"]
        config = TrainingConfig(
            stage="pilot",
            epochs=1,
            variant_id=voice_variant["variant_id"],
            speaker_key=speaker_name,
        )
    payload: dict[str, Any] = {
        "kind": ARTIFACT_KIND,
        "schema_version": schema_version,
        "mode": "custom_voice",
        "reference_audio_required": False,
        "speaker": {
            "name": speaker_name,
            "slot": 3000,
            "embedding_file": "speaker_embedding.safetensors",
        },
        "base_model": {"path": str(model_path), "files": {"config.json": "unused"}},
        "dataset": {},
        "adapter": {
            "file": "adapters.safetensors",
            "rank": 16,
            "alpha": 16,
            "target_modules": [],
        },
        "training_config_file": "training_config.json",
    }
    if schema_version == ARTIFACT_SCHEMA_VERSION:
        assert voice_variant is not None
        payload["voice_variant"] = copy.deepcopy(voice_variant)
        payload["dataset"] = {"voice_variant": copy.deepcopy(voice_variant)}
    if mutate is not None:
        mutate(payload, config)
    (path / "artifact.json").write_text(json.dumps(payload), encoding="utf-8")
    (path / "training_config.json").write_text(
        json.dumps(config.to_dict()), encoding="utf-8"
    )
    (path / "adapters.safetensors").write_bytes(b"adapter")
    (path / "speaker_embedding.safetensors").write_bytes(b"embedding")
    return path


def test_dataset_variant_is_normalized_and_training_config_is_bound() -> None:
    manifest_metadata = {
        "profile": "ruhi",
        "variant_id": "variant_warm",
        "voice_variant": {
            "id": "variant_warm",
            "name": "Warm Narration",
            "language": "en-US",
            "accent_descriptor": "Warm, measured English narration",
        },
    }

    variant = resolve_dataset_voice_variant(manifest_metadata)
    config, resolved = resolve_training_variant(
        TrainingConfig(stage="pilot", epochs=1), manifest_metadata
    )

    assert variant == resolved == _canonical_variant()
    assert config.variant_id == "variant_warm"
    assert config.speaker_key == stable_speaker_key("ruhi", "variant_warm")
    assert TrainingConfig.from_dict(config.to_dict()) == config


def test_legacy_dataset_maps_to_default_variant() -> None:
    variant = resolve_dataset_voice_variant({"schema_version": 1, "profile": "ruhi"})

    assert variant["variant_id"] == DEFAULT_VARIANT_ID
    assert variant["profile"] == "ruhi"
    assert variant["display_name"] == "Ruhi Default"
    assert variant["speaker_key"] == "ruhi_variant_ruhi_default"


def test_stable_speaker_key_is_bounded_and_deterministic() -> None:
    variant_id = "variant_" + "a" * 55

    first = stable_speaker_key("ruhi", variant_id)
    second = stable_speaker_key("ruhi", variant_id)

    assert first == second
    assert len(first) == 32
    assert first.startswith("ruhi_variant_")


@pytest.mark.parametrize(
    "metadata, message",
    [
        (
            {
                "variant_id": "variant_one",
                "voice_variant": {"variant_id": "variant_two"},
            },
            "disagrees",
        ),
        (
            {
                "voice_variant": {
                    "variant_id": "variant_one",
                    "profile": "ruhi",
                    "speaker_key": "ruhi_wrong",
                }
            },
            "stable key",
        ),
        (
            {"voice_variant": {"variant_id": "variant_one", "language": "spanish"}},
            "English-only",
        ),
        (
            {"voice_variant": {"variant_id": "", "name": "Blank identity"}},
            "lowercase local identifier",
        ),
        (
            {"voice_variant": {"variant_id": "variant_one", "name": ""}},
            "display_name",
        ),
    ],
)
def test_dataset_variant_rejects_ambiguous_or_unstable_identity(
    metadata: dict[str, Any], message: str
) -> None:
    with pytest.raises(TrainingError, match=message):
        resolve_dataset_voice_variant(metadata)


def test_training_config_cannot_override_frozen_dataset_variant() -> None:
    with pytest.raises(TrainingError, match="disagrees"):
        resolve_training_variant(
            TrainingConfig(
                stage="pilot",
                epochs=1,
                variant_id="variant_other",
                speaker_key=stable_speaker_key("ruhi", "variant_other"),
            ),
            {"voice_variant": _canonical_variant()},
        )


def test_schema_v1_artifact_loads_as_default_variant(tmp_path: Path) -> None:
    model = _write_local_model(tmp_path / "model")
    checkpoint = _write_artifact(
        tmp_path / "legacy",
        model,
        schema_version=LEGACY_ARTIFACT_SCHEMA_VERSION,
    )

    artifact = NamedSpeakerArtifact.load(checkpoint)
    runtime = NamedSpeakerRuntime(
        model,
        checkpoint,
        verify_model=False,
        expected_variant_id=DEFAULT_VARIANT_ID,
    )

    assert artifact.schema_version == 1
    assert artifact.variant_id == DEFAULT_VARIANT_ID
    assert artifact.profile_name == "ruhi"
    assert artifact.speaker_key == "ruhi"
    assert artifact.voice_variant["legacy_schema"] is True
    assert runtime.variant_id == DEFAULT_VARIANT_ID


def test_schema_v2_artifact_exposes_variant_and_accepts_expected_id(
    tmp_path: Path,
) -> None:
    model = _write_local_model(tmp_path / "model")
    variant = _canonical_variant()
    checkpoint = _write_artifact(
        tmp_path / "checkpoint",
        model,
        schema_version=ARTIFACT_SCHEMA_VERSION,
        voice_variant=variant,
    )

    artifact = NamedSpeakerArtifact.load(checkpoint)
    runtime = NamedSpeakerRuntime(
        model,
        checkpoint,
        verify_model=False,
        expected_variant_id=variant["variant_id"],
    )

    assert artifact.schema_version == 2
    assert artifact.voice_variant == variant
    assert artifact.variant_id == "variant_warm"
    assert artifact.speaker_key == stable_speaker_key("ruhi", "variant_warm")
    assert runtime.variant_id == "variant_warm"
    assert runtime.loaded is False


def test_runtime_rejects_checkpoint_from_another_variant(tmp_path: Path) -> None:
    model = _write_local_model(tmp_path / "model")
    checkpoint = _write_artifact(
        tmp_path / "checkpoint",
        model,
        schema_version=ARTIFACT_SCHEMA_VERSION,
    )

    with pytest.raises(ArtifactError, match="expected 'variant_other'.*'variant_warm'"):
        NamedSpeakerRuntime(
            model,
            checkpoint,
            verify_model=False,
            expected_variant_id="variant_other",
        )


@pytest.mark.parametrize(
    "mutate, message",
    [
        (
            lambda payload, _config: payload["dataset"]["voice_variant"].update(
                {
                    "variant_id": "variant_other",
                    "speaker_key": stable_speaker_key("ruhi", "variant_other"),
                }
            ),
            "differs from its dataset metadata",
        ),
        (
            lambda payload, _config: payload["speaker"].update({"name": "ruhi_wrong"}),
            "speaker name differs",
        ),
        (
            lambda _payload, config: object.__setattr__(
                config, "variant_id", "variant_other"
            ),
            "training configuration",
        ),
    ],
)
def test_schema_v2_artifact_rejects_identity_mismatch(
    tmp_path: Path, mutate: Any, message: str
) -> None:
    model = _write_local_model(tmp_path / "model")
    checkpoint = _write_artifact(
        tmp_path / "checkpoint",
        model,
        schema_version=ARTIFACT_SCHEMA_VERSION,
        mutate=mutate,
    )

    with pytest.raises(ArtifactError, match=message):
        NamedSpeakerArtifact.load(checkpoint)
