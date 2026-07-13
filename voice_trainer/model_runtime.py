"""Offline runtime for a persistent Qwen3-TTS named-speaker artifact."""

from __future__ import annotations

import json
import os
import re
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from .training import (
    ARTIFACT_KIND,
    ARTIFACT_SCHEMA_VERSION,
    DEFAULT_SPEAKER_NAME,
    DEFAULT_VARIANT_DISPLAY_NAME,
    DEFAULT_VARIANT_ID,
    LEGACY_ARTIFACT_SCHEMA_VERSION,
    RESERVED_SPEAKER_SLOT,
    TrainingConfig,
    TrainingError,
    _apply_lora,
    _offline_environment,
    _sha256,
    resolve_dataset_voice_variant,
    validate_local_model_path,
)


MAX_GENERATION_SEED = 4_294_967_295


class ArtifactError(RuntimeError):
    """Raised when a checkpoint cannot be safely reconstructed."""


def _validated_seed(seed: int | None) -> int | None:
    if seed is None:
        return None
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("generation seed must be an integer")
    if not 0 <= seed <= MAX_GENERATION_SEED:
        raise ValueError(
            f"generation seed must be between 0 and {MAX_GENERATION_SEED}"
        )
    return seed


@dataclass(frozen=True)
class DeliverySettings:
    """Reference-free generation controls supported by the local runtime."""

    language: str = "english"
    instruction: str | None = None
    temperature: float = 0.9
    max_tokens: int = 4096
    top_k: int = 50
    top_p: float = 1.0
    repetition_penalty: float = 1.05

    def __post_init__(self) -> None:
        if self.language.lower() != "english":
            raise ValueError("v1 generation is English-only")
        if self.instruction is not None:
            instruction = self.instruction.strip()
            if not instruction or len(instruction) > 500:
                raise ValueError("instruction must contain 1–500 characters")
        if not 0.1 <= self.temperature <= 2.0:
            raise ValueError("temperature must be between 0.1 and 2.0")
        if not 64 <= self.max_tokens <= 4096:
            raise ValueError("max_tokens must be between 64 and 4096")
        if not 1 <= self.top_k <= 1000:
            raise ValueError("top_k must be between 1 and 1000")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if not 1.0 <= self.repetition_penalty <= 2.0:
            raise ValueError("repetition_penalty must be between 1.0 and 2.0")

    @classmethod
    def from_value(cls, value: "DeliverySettings | Mapping[str, Any] | None") -> "DeliverySettings":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        payload = dict(value)
        forbidden = {
            "ref_audio",
            "ref_text",
            "reference_audio",
            "reference_text",
            "voice_sample",
        }
        present = sorted(key for key in forbidden if key in payload)
        if present:
            raise ValueError(
                "persistent named-speaker generation does not accept reference input: "
                + ", ".join(present)
            )
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValueError("unsupported delivery settings: " + ", ".join(unknown))
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NamedSpeakerArtifact:
    path: Path
    payload: Mapping[str, Any]

    @property
    def schema_version(self) -> int:
        return int(self.payload["schema_version"])

    @property
    def speaker_name(self) -> str:
        return str(self.payload["speaker"]["name"])

    @property
    def speaker_key(self) -> str:
        """Stable internal Qwen speaker lookup key (not a user-facing label)."""

        return self.speaker_name

    @property
    def speaker_slot(self) -> int:
        return int(self.payload["speaker"]["slot"])

    @property
    def voice_variant(self) -> Mapping[str, Any]:
        if self.schema_version == LEGACY_ARTIFACT_SCHEMA_VERSION:
            return {
                "variant_id": DEFAULT_VARIANT_ID,
                "profile": DEFAULT_SPEAKER_NAME,
                "display_name": DEFAULT_VARIANT_DISPLAY_NAME,
                "language": "english",
                "speaker_key": self.speaker_name,
                "legacy_schema": True,
            }
        return self.payload["voice_variant"]

    @property
    def variant_id(self) -> str:
        return str(self.voice_variant["variant_id"])

    @property
    def profile_name(self) -> str:
        return str(self.voice_variant["profile"])

    @property
    def base_model_path(self) -> Path:
        return Path(str(self.payload["base_model"]["path"])).expanduser().resolve()

    @property
    def adapter_path(self) -> Path:
        return self.path / str(self.payload["adapter"]["file"])

    @property
    def embedding_path(self) -> Path:
        return self.path / str(self.payload["speaker"]["embedding_file"])

    @property
    def training_config_path(self) -> Path:
        return self.path / str(self.payload["training_config_file"])

    def load_training_config(self) -> TrainingConfig:
        """Read and validate the immutable training configuration."""

        try:
            payload = json.loads(self.training_config_path.read_text(encoding="utf-8"))
            if not isinstance(payload, Mapping):
                raise ValueError("training configuration must be an object")
            return TrainingConfig.from_dict(payload)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ArtifactError(f"invalid artifact training configuration: {exc}") from exc

    @classmethod
    def load(cls, artifact_path: str | Path) -> "NamedSpeakerArtifact":
        path = Path(artifact_path).expanduser().resolve()
        manifest_path = path / "artifact.json" if path.is_dir() else path
        if not manifest_path.is_file():
            raise ArtifactError(f"checkpoint artifact manifest not found: {manifest_path}")
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ArtifactError(f"invalid checkpoint artifact: {exc}") from exc
        path = manifest_path.parent
        if payload.get("kind") != ARTIFACT_KIND:
            raise ArtifactError("checkpoint is not a Noice City named-speaker artifact")
        schema_version = payload.get("schema_version")
        if schema_version not in {
            LEGACY_ARTIFACT_SCHEMA_VERSION,
            ARTIFACT_SCHEMA_VERSION,
        }:
            raise ArtifactError("unsupported named-speaker artifact schema")
        if payload.get("mode") != "custom_voice":
            raise ArtifactError("artifact is not configured for custom_voice inference")
        if payload.get("reference_audio_required") is not False:
            raise ArtifactError("artifact does not guarantee reference-free inference")
        speaker = payload.get("speaker") or {}
        name = str(speaker.get("name", ""))
        if not re.fullmatch(r"[a-z][a-z0-9_-]{0,31}", name):
            raise ArtifactError("artifact has an invalid named speaker")
        if int(speaker.get("slot", -1)) != RESERVED_SPEAKER_SLOT:
            raise ArtifactError("artifact does not use reserved speaker slot 3000")
        if schema_version == ARTIFACT_SCHEMA_VERSION:
            variant = payload.get("voice_variant")
            if not isinstance(variant, Mapping):
                raise ArtifactError("schema-v2 artifact has no voice_variant metadata")
            try:
                normalized_variant = resolve_dataset_voice_variant(
                    {"voice_variant": variant}
                )
            except (TrainingError, ValueError) as exc:
                raise ArtifactError(f"invalid voice_variant metadata: {exc}") from exc
            expected_speaker_key = normalized_variant["speaker_key"]
            canonical_fields = (
                "variant_id",
                "profile",
                "display_name",
                "language",
                "speaker_key",
            )
            if any(
                variant.get(field) != normalized_variant[field]
                for field in canonical_fields
            ):
                raise ArtifactError("schema-v2 voice_variant metadata is not canonical")
            if variant.get("speaker_key") != expected_speaker_key:
                raise ArtifactError("voice_variant speaker_key is not stable")
            if name != expected_speaker_key:
                raise ArtifactError("artifact speaker name differs from voice_variant")
            dataset = payload.get("dataset")
            if not isinstance(dataset, Mapping):
                raise ArtifactError("schema-v2 artifact has no dataset metadata")
            dataset_variant = dataset.get("voice_variant")
            if not isinstance(dataset_variant, Mapping):
                raise ArtifactError("schema-v2 dataset metadata has no voice_variant")
            try:
                normalized_dataset_variant = resolve_dataset_voice_variant(
                    {"voice_variant": dataset_variant}
                )
            except (TrainingError, ValueError) as exc:
                raise ArtifactError(
                    f"invalid dataset voice_variant metadata: {exc}"
                ) from exc
            if normalized_dataset_variant != normalized_variant:
                raise ArtifactError(
                    "artifact voice variant differs from its dataset metadata"
                )
        adapter = payload.get("adapter") or {}
        if int(adapter.get("rank", 0)) != 16 or int(adapter.get("alpha", 0)) != 16:
            raise ArtifactError("artifact does not use the required rank-16/alpha-16 adapter")
        instance = cls(path, payload)
        for required in (
            instance.adapter_path,
            instance.embedding_path,
            instance.training_config_path,
        ):
            if not required.is_file():
                raise ArtifactError(f"artifact resource is missing: {required}")
            # Resources must live directly inside the immutable checkpoint.
            if required.parent != path:
                raise ArtifactError("artifact resource path escapes the checkpoint directory")
        training_config = instance.load_training_config()
        if schema_version == LEGACY_ARTIFACT_SCHEMA_VERSION:
            if training_config.speaker_name != instance.speaker_name:
                raise ArtifactError(
                    "artifact speaker name differs from its training configuration"
                )
        else:
            if training_config.variant_id != instance.variant_id:
                raise ArtifactError(
                    "artifact variant differs from its training configuration"
                )
            if training_config.speaker_name != instance.profile_name:
                raise ArtifactError(
                    "artifact profile differs from its training configuration"
                )
            if training_config.speaker_key != instance.speaker_key:
                raise ArtifactError(
                    "artifact speaker key differs from its training configuration"
                )
        return instance


def _verify_base_model(artifact: NamedSpeakerArtifact, model_path: Path) -> None:
    expected = artifact.payload.get("base_model", {}).get("files", {})
    if not isinstance(expected, Mapping) or not expected:
        raise ArtifactError("artifact contains no base-model integrity manifest")
    for relative, expected_hash in expected.items():
        relative_path = Path(str(relative))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ArtifactError("base-model integrity path escapes the model directory")
        # Hugging Face snapshot files are normally symlinks to the blob cache,
        # so validate the lexical path but intentionally hash through symlinks.
        candidate = model_path / relative_path
        if not candidate.is_file():
            raise ArtifactError(f"base-model resource is missing: {relative}")
        if _sha256(candidate) != expected_hash:
            raise ArtifactError(f"base-model integrity check failed: {relative}")


class NamedSpeakerRuntime:
    """A lazily loaded, local-only Qwen3-TTS custom-voice runtime."""

    def __init__(
        self,
        model_path: str | Path,
        artifact_path: str | Path,
        *,
        verify_model: bool = True,
        expected_variant_id: str | None = None,
    ) -> None:
        _offline_environment()
        self.artifact = NamedSpeakerArtifact.load(artifact_path)
        if expected_variant_id is not None:
            if expected_variant_id != self.artifact.variant_id:
                raise ArtifactError(
                    "checkpoint voice variant differs from expected_variant_id: "
                    f"expected {expected_variant_id!r}, found {self.artifact.variant_id!r}"
                )
        self.model_path = validate_local_model_path(model_path)
        if self.model_path.resolve() != self.artifact.base_model_path.resolve():
            raise ArtifactError(
                "checkpoint base-model path differs from the requested local model path"
            )
        if verify_model:
            _verify_base_model(self.artifact, self.model_path)
        self._model: Any | None = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def variant_id(self) -> str:
        return self.artifact.variant_id

    def load(self) -> "NamedSpeakerRuntime":
        if self._model is not None:
            return self
        try:
            import mlx.core as mx
            from mlx_audio.tts.utils import load_model
        except ImportError as exc:
            raise ArtifactError(
                "local MLX inference dependencies are unavailable in the locked environment"
            ) from exc

        training_config = self.artifact.load_training_config()

        model = load_model(self.model_path, lazy=False, strict=True)
        model.freeze()
        applied = _apply_lora(model, training_config)
        recorded = list(self.artifact.payload["adapter"].get("target_modules", []))
        if recorded and applied != recorded:
            raise ArtifactError("local MLX model layout differs from the adapter checkpoint")
        model.load_weights(str(self.artifact.adapter_path), strict=False)

        embedding_payload = mx.load(str(self.artifact.embedding_path))
        if "speaker_embedding" not in embedding_payload:
            raise ArtifactError("speaker embedding file lacks speaker_embedding")
        speaker_embedding = embedding_payload["speaker_embedding"].reshape(-1)
        codec_embedding = model.talker.get_input_embeddings()
        if speaker_embedding.shape != codec_embedding.weight.shape[1:]:
            raise ArtifactError(
                "saved speaker embedding shape does not match the local Qwen talker"
            )
        codec_embedding.weight[self.artifact.speaker_slot] = speaker_embedding.astype(
            codec_embedding.weight.dtype
        )

        # Convert the loaded Base instance into the exact persistent named-speaker
        # inference mode expected by MLX-Audio.  The speaker vector now lives at
        # row 3000; generation never passes reference audio or reference text.
        model.config.tts_model_type = "custom_voice"
        model.config.talker_config.spk_id = {
            self.artifact.speaker_name: self.artifact.speaker_slot
        }
        model.config.talker_config.spk_is_dialect = {
            self.artifact.speaker_name: False
        }
        model.supported_speakers = [self.artifact.speaker_name]
        model.eval()
        mx.eval(model.parameters())
        self._model = model
        return self

    def generate(
        self,
        text: str,
        output_path: str | Path,
        delivery: DeliverySettings | Mapping[str, Any] | None = None,
        *,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Generate WAV speech using only text, settings, and the named checkpoint."""

        normalized_text = text.strip()
        if not normalized_text:
            raise ValueError("generation text cannot be empty")
        if len(normalized_text) > 20_000:
            raise ValueError("generation text exceeds 20,000 characters")
        settings = DeliverySettings.from_value(delivery)
        generation_seed = _validated_seed(seed)
        self.load()
        assert self._model is not None

        import mlx.core as mx
        import numpy as np

        # Adapter construction consumes MLX randomness during model loading.
        # Seed only after the complete artifact is loaded, immediately before
        # acoustic sampling, so the same request is reproducible after restart.
        # A missing seed intentionally preserves stochastic generation.
        if generation_seed is not None:
            mx.random.seed(generation_seed)
        results = list(
            self._model.generate(
                text=normalized_text,
                voice=self.artifact.speaker_name,
                instruct=settings.instruction,
                temperature=settings.temperature,
                lang_code=settings.language,
                max_tokens=settings.max_tokens,
                top_k=settings.top_k,
                top_p=settings.top_p,
                repetition_penalty=settings.repetition_penalty,
                verbose=False,
                stream=False,
            )
        )
        if not results:
            raise ArtifactError("Qwen3-TTS returned no generated audio")
        sample_rates = {int(result.sample_rate) for result in results}
        if len(sample_rates) != 1:
            raise ArtifactError("Qwen3-TTS returned inconsistent sample rates")
        sample_rate = sample_rates.pop()
        audio_parts = [result.audio.reshape(-1) for result in results]
        audio = audio_parts[0] if len(audio_parts) == 1 else mx.concatenate(audio_parts)
        mx.eval(audio)
        samples = np.asarray(audio, dtype=np.float32)
        if samples.size == 0 or not np.isfinite(samples).all():
            raise ArtifactError("Qwen3-TTS returned empty or non-finite audio")

        destination = Path(output_path).expanduser().resolve()
        if destination.suffix.lower() != ".wav":
            raise ValueError("generation output_path must end in .wav")
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        temporary = destination.with_name(f".{destination.name}.partial")
        pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
        try:
            with wave.open(str(temporary), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(sample_rate)
                wav.writeframes(pcm.tobytes())
            temporary.chmod(0o600)
            os.replace(temporary, destination)
            destination.chmod(0o600)
        finally:
            temporary.unlink(missing_ok=True)

        metadata = {
            "output_path": str(destination),
            "sample_rate": sample_rate,
            "samples": int(samples.size),
            "duration_seconds": float(samples.size / sample_rate),
            "checkpoint": str(self.artifact.path),
            "speaker": self.artifact.speaker_name,
            "profile": self.artifact.profile_name,
            "variant_id": self.artifact.variant_id,
            "delivery": settings.to_dict(),
            "reference_audio_required": False,
        }
        if generation_seed is not None:
            metadata["seed"] = generation_seed
        return metadata

    def unload(self) -> None:
        if self._model is None:
            return
        self._model = None
        try:
            import mlx.core as mx

            mx.clear_cache()
        except ImportError:
            pass


def load_named_speaker_runtime(
    model_path: str | Path,
    artifact_path: str | Path,
    *,
    verify_model: bool = True,
    expected_variant_id: str | None = None,
) -> NamedSpeakerRuntime:
    """Create and eagerly load a local named-speaker runtime."""

    return NamedSpeakerRuntime(
        model_path,
        artifact_path,
        verify_model=verify_model,
        expected_variant_id=expected_variant_id,
    ).load()


def generate(
    text: str,
    artifact_path: str | Path,
    output_path: str | Path,
    delivery: DeliverySettings | Mapping[str, Any] | None = None,
    *,
    seed: int | None = None,
    model_path: str | Path | None = None,
    verify_model: bool = True,
    expected_variant_id: str | None = None,
) -> dict[str, Any]:
    """One-shot app API for reference-free local generation.

    ``model_path`` may be omitted because the immutable artifact records the
    exact absolute local base path.  No reference-audio argument exists.
    """

    artifact = NamedSpeakerArtifact.load(artifact_path)
    selected_model = artifact.base_model_path if model_path is None else model_path
    runtime = NamedSpeakerRuntime(
        selected_model,
        artifact.path,
        verify_model=verify_model,
        expected_variant_id=expected_variant_id,
    )
    try:
        return runtime.generate(text, output_path, delivery, seed=seed)
    finally:
        runtime.unload()


__all__ = [
    "ArtifactError",
    "DeliverySettings",
    "MAX_GENERATION_SEED",
    "NamedSpeakerArtifact",
    "NamedSpeakerRuntime",
    "generate",
    "load_named_speaker_runtime",
]
