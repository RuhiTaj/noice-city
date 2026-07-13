"""Native MLX training for a persistent Qwen3-TTS named speaker.

This module deliberately imports MLX only inside execution paths.  The web
application can therefore inspect configurations and artifacts without loading
several gigabytes of model weights (or attempting a network request).

The loss construction mirrors Qwen's 12 Hz single-speaker SFT recipe:

* the talker predicts codebook 0 autoregressively;
* the code predictor predicts codebooks 1..15 from the talker hidden state;
* the combined loss is ``codebook_0 + 0.3 * codebooks_1_to_15``.

Unlike the upstream full-weight recipe, only rank-16 LoRA matrices on attention
and MLP projections are trainable.  A single, fixed speaker embedding is used
for every example and is persisted separately in each checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import shutil
import time
import wave
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


ARTIFACT_KIND = "voicetrainer.qwen3_tts.named_speaker_adapter"
ARTIFACT_SCHEMA_VERSION = 2
LEGACY_ARTIFACT_SCHEMA_VERSION = 1
DEFAULT_SPEAKER_NAME = "ruhi"
DEFAULT_VARIANT_ID = "variant_ruhi_default"
# Kept for schema-v1 artifact compatibility; current variants carry their own
# user-defined display names.
DEFAULT_VARIANT_DISPLAY_NAME = "Ruhi Default"
RESERVED_SPEAKER_SLOT = 3000
EXPECTED_CODEBOOKS = 16
VARIANT_ID_PATTERN = re.compile(r"[a-z][a-z0-9_-]{0,63}")
SPEAKER_KEY_PATTERN = re.compile(r"[a-z][a-z0-9_-]{0,31}")

# Every projection in the 28-layer talker and five-layer code predictor is
# adapted.  Keeping the suffixes in the public configuration makes checkpoint
# reconstruction deterministic.
DEFAULT_LORA_TARGETS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)

ProgressCallback = Callable[[dict[str, Any]], None]
CancelCallback = Callable[[], bool]


class TrainingError(RuntimeError):
    """Raised when the local dataset/model cannot satisfy the training contract."""


class TrainingCancelled(TrainingError):
    """Raised when a caller requests cancellation between safe training steps."""


def stable_speaker_key(profile: str, variant_id: str) -> str:
    """Derive one deterministic Qwen speaker key for a persistent variant."""

    if not SPEAKER_KEY_PATTERN.fullmatch(profile):
        raise ValueError("voice profile must be a lowercase local identifier")
    if not VARIANT_ID_PATTERN.fullmatch(variant_id):
        raise ValueError("variant_id must be a lowercase local identifier")
    candidate = f"{profile}_{variant_id}"
    if len(candidate) <= 32:
        return candidate
    digest = hashlib.sha256(candidate.encode("utf-8")).hexdigest()[:10]
    prefix_length = 32 - len(digest) - 1
    return f"{candidate[:prefix_length]}_{digest}"


def resolve_dataset_voice_variant(metadata: Mapping[str, Any]) -> dict[str, str]:
    """Normalize authoritative variant identity from a frozen manifest.

    Schema-v1 datasets had only the legacy internal profile key. They map to
    the legacy default variant while every new explicit variant receives a
    stable internal speaker key.
    """

    raw_value = metadata.get("voice_variant")
    top_level_variant = metadata.get("variant_id")
    if raw_value is None:
        raw: Mapping[str, Any] = {}
    elif isinstance(raw_value, Mapping):
        raw = raw_value
    else:
        raise TrainingError("dataset voice_variant must be an object")

    raw_variant = raw["variant_id"] if "variant_id" in raw else raw.get("id")
    if raw_value is not None and raw_variant is None:
        raise TrainingError("dataset voice_variant requires variant_id")
    selected_variant = (
        raw_variant
        if raw_variant is not None
        else top_level_variant
        if top_level_variant is not None
        else DEFAULT_VARIANT_ID
    )
    variant_id = str(selected_variant)
    if top_level_variant is not None and str(top_level_variant) != variant_id:
        raise TrainingError("dataset variant_id disagrees with voice_variant")
    if not VARIANT_ID_PATTERN.fullmatch(variant_id):
        raise TrainingError("dataset variant_id must be a lowercase local identifier")

    top_level_profile = metadata.get("profile")
    selected_profile = (
        raw["profile"]
        if "profile" in raw
        else top_level_profile
        if top_level_profile is not None
        else DEFAULT_SPEAKER_NAME
    )
    profile = str(selected_profile)
    if top_level_profile is not None and str(top_level_profile) != profile:
        raise TrainingError("dataset profile disagrees with voice_variant")
    if not SPEAKER_KEY_PATTERN.fullmatch(profile):
        raise TrainingError("dataset voice profile must be a lowercase local identifier")

    language_value = str(
        raw["language"] if "language" in raw else "english"
    ).lower().replace("_", "-")
    if language_value not in {"english", "en", "en-us"}:
        raise TrainingError("voice variants are English-only")
    fallback_display_name = (
        DEFAULT_VARIANT_DISPLAY_NAME
        if variant_id == DEFAULT_VARIANT_ID
        else variant_id
    )
    selected_display_name = (
        raw["display_name"]
        if "display_name" in raw
        else raw["name"]
        if "name" in raw
        else fallback_display_name
    )
    display_name = str(selected_display_name).strip()
    if not display_name or len(display_name) > 100:
        raise TrainingError("voice variant display_name must contain 1–100 characters")
    accent = str(raw.get("accent_descriptor", raw.get("accent", ""))).strip()
    if len(accent) > 200:
        raise TrainingError("voice variant accent descriptor exceeds 200 characters")

    speaker_key = stable_speaker_key(profile, variant_id)
    supplied_key = raw.get("speaker_key")
    if supplied_key is not None and str(supplied_key) != speaker_key:
        raise TrainingError("dataset speaker_key is not the stable key for its variant")
    normalized = {
        "variant_id": variant_id,
        "profile": profile,
        "display_name": display_name,
        "language": "english",
        "speaker_key": speaker_key,
    }
    if accent:
        normalized["accent_descriptor"] = accent
    return normalized


@dataclass(frozen=True)
class TrainingConfig:
    """Decision-complete configuration for named-speaker adaptation."""

    speaker_name: str = DEFAULT_SPEAKER_NAME
    speaker_slot: int = RESERVED_SPEAKER_SLOT
    variant_id: str | None = None
    speaker_key: str | None = None
    stage: str = "full"
    epochs: int = 3
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    gradient_checkpointing: bool = True
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0
    seed: int = 3407
    lora_rank: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_targets: tuple[str, ...] = field(default_factory=lambda: DEFAULT_LORA_TARGETS)
    sub_talker_loss_weight: float = 0.3
    language: str = "english"
    minimum_pilot_seconds: float = 120.0
    minimum_full_seconds: float = 1800.0

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-z][a-z0-9_-]{0,31}", self.speaker_name):
            raise ValueError("speaker_name must be a lowercase local identifier")
        if self.speaker_slot != RESERVED_SPEAKER_SLOT:
            raise ValueError(f"speaker_slot is reserved and must be {RESERVED_SPEAKER_SLOT}")
        if self.variant_id is not None and not VARIANT_ID_PATTERN.fullmatch(self.variant_id):
            raise ValueError("variant_id must be a lowercase local identifier")
        if self.speaker_key is not None:
            if not SPEAKER_KEY_PATTERN.fullmatch(self.speaker_key):
                raise ValueError("speaker_key must be a lowercase local identifier")
            if self.variant_id is None:
                raise ValueError("speaker_key requires variant_id")
            if self.speaker_key != stable_speaker_key(self.speaker_name, self.variant_id):
                raise ValueError("speaker_key is not stable for speaker_name and variant_id")
        if self.stage not in {"pilot", "full"}:
            raise ValueError("stage must be 'pilot' or 'full'")
        if self.epochs < 1:
            raise ValueError("epochs must be at least 1")
        if self.batch_size != 1:
            raise ValueError("Qwen3-TTS MLX v1 training requires batch_size=1")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be at least 1")
        if not 0.0 < self.learning_rate <= 1e-3:
            raise ValueError("learning_rate must be in (0, 1e-3]")
        if self.lora_rank != 16 or self.lora_alpha != 16:
            raise ValueError("the persistent speaker adapter requires rank=16, alpha=16")
        if tuple(self.lora_targets) != DEFAULT_LORA_TARGETS:
            raise ValueError("all talker and code-predictor attention/MLP targets are required")
        if not math.isclose(self.sub_talker_loss_weight, 0.3):
            raise ValueError("sub_talker_loss_weight must be 0.3")
        if self.language.lower() != "english":
            raise ValueError("v1 training is English-only")
        if self.minimum_pilot_seconds < 120.0:
            raise ValueError("pilot training requires at least two accepted minutes")
        if self.minimum_full_seconds < 1800.0:
            raise ValueError("full training requires at least 30 accepted minutes")

    @property
    def lora_scale(self) -> float:
        """MLX-LoRA scale (alpha / rank)."""

        return self.lora_alpha / self.lora_rank

    @property
    def minimum_dataset_seconds(self) -> float:
        return (
            self.minimum_pilot_seconds
            if self.stage == "pilot"
            else self.minimum_full_seconds
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["lora_targets"] = list(self.lora_targets)
        payload["lora_scale"] = self.lora_scale
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TrainingConfig":
        values = dict(payload)
        values.pop("lora_scale", None)
        if "lora_targets" in values:
            values["lora_targets"] = tuple(values["lora_targets"])
        return cls(**values)


@dataclass(frozen=True)
class DatasetItem:
    audio_path: Path
    text: str
    split: str
    duration_seconds: float
    qa: Mapping[str, Any] = field(default_factory=dict)


def resolve_training_variant(
    config: TrainingConfig, dataset_metadata: Mapping[str, Any]
) -> tuple[TrainingConfig, dict[str, str]]:
    """Bind an unresolved training config to its frozen dataset variant."""

    voice_variant = resolve_dataset_voice_variant(dataset_metadata)
    variant_id = voice_variant["variant_id"]
    speaker_key = voice_variant["speaker_key"]
    if config.speaker_name != voice_variant["profile"]:
        raise TrainingError("training speaker profile disagrees with the dataset variant")
    if config.variant_id is not None and config.variant_id != variant_id:
        raise TrainingError("training variant_id disagrees with the frozen dataset")
    if config.speaker_key is not None and config.speaker_key != speaker_key:
        raise TrainingError("training speaker_key disagrees with the frozen dataset")
    return replace(config, variant_id=variant_id, speaker_key=speaker_key), voice_variant


def _emit(callback: ProgressCallback | None, event: str, **values: Any) -> None:
    if callback is not None:
        callback({"event": event, "timestamp": time.time(), **values})


def _check_cancelled(callback: CancelCallback | None) -> None:
    if callback is not None and callback():
        raise TrainingCancelled("training cancelled at a safe step boundary")


def _offline_environment() -> None:
    """Disable supported remote/telemetry paths before ML libraries import."""

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("DO_NOT_TRACK", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def validate_local_model_path(model_path: str | Path) -> Path:
    """Validate an absolute local Qwen3-TTS 1.7B Base BF16 directory."""

    path = Path(model_path).expanduser()
    if not path.is_absolute():
        raise TrainingError("model_path must be an absolute local filesystem path")
    if not path.is_dir():
        raise TrainingError(f"local model directory does not exist: {path}")
    config_path = path / "config.json"
    if not config_path.is_file():
        raise TrainingError(f"missing local model config: {config_path}")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainingError(f"invalid model config: {exc}") from exc

    if config.get("model_type") != "qwen3_tts":
        raise TrainingError("the selected local model is not Qwen3-TTS")
    if config.get("tts_model_type") != "base":
        raise TrainingError("training must start from a Qwen3-TTS Base model")
    if config.get("tts_model_size") != "1b7":
        raise TrainingError("v1 requires the Qwen3-TTS 1.7B Base model")
    talker_config = config.get("talker_config") or {}
    if int(talker_config.get("vocab_size", 0)) <= RESERVED_SPEAKER_SLOT:
        raise TrainingError("base model codec vocabulary has no reserved speaker slot 3000")
    if not list(path.glob("*.safetensors")):
        raise TrainingError("local model has no safetensors weights")
    speech_tokenizer = path / "speech_tokenizer"
    if not speech_tokenizer.is_dir() or not list(speech_tokenizer.glob("*.safetensors")):
        raise TrainingError("local model is missing its speech tokenizer weights")
    return path.resolve()


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_model(model_path: str | Path) -> dict[str, Any]:
    """Hash every model/config resource needed for offline reconstruction."""

    root = validate_local_model_path(model_path)
    files: dict[str, str] = {}
    allowed_suffixes = {".json", ".safetensors", ".txt", ".jinja", ".model"}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        if path.suffix.lower() in allowed_suffixes:
            files[path.relative_to(root).as_posix()] = _sha256(path)
    revision = root.parent.name if root.parent.name == "snapshots" else root.name
    # HF snapshots have .../snapshots/<revision>, so the resolved directory name
    # is the authoritative revision in the common case.
    if len(root.name) >= 12 and re.fullmatch(r"[0-9a-f]+", root.name):
        revision = root.name
    return {"path": str(root), "revision": revision, "files": files}


def _read_json_or_jsonl(path: Path) -> Any:
    if path.suffix.lower() == ".jsonl":
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    return json.loads(path.read_text(encoding="utf-8"))


def _audio_duration(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as wav:
            rate = wav.getframerate()
            return wav.getnframes() / rate if rate else 0.0
    except (wave.Error, OSError):
        return 0.0


def load_frozen_dataset(
    dataset_manifest: str | Path | Mapping[str, Any],
) -> tuple[list[DatasetItem], dict[str, Any]]:
    """Load a frozen manifest while accepting plain dict/path store contracts."""

    manifest_path: Path | None = None
    if isinstance(dataset_manifest, Mapping):
        payload: Any = dict(dataset_manifest)
        base_dir = Path.cwd()
    else:
        manifest_path = Path(dataset_manifest).expanduser().resolve()
        if not manifest_path.is_file():
            raise TrainingError(f"frozen dataset manifest not found: {manifest_path}")
        payload = _read_json_or_jsonl(manifest_path)
        base_dir = manifest_path.parent

    if isinstance(payload, list):
        raw_items = payload
        metadata: dict[str, Any] = {}
    elif isinstance(payload, dict):
        raw_items = (
            payload.get("recordings")
            or payload.get("items")
            or payload.get("samples")
            or []
        )
        metadata = {k: v for k, v in payload.items() if k not in {"recordings", "items", "samples"}}
        if payload.get("frozen") is False:
            raise TrainingError("training refuses a mutable dataset; freeze it first")
    else:
        raise TrainingError("dataset manifest must contain a JSON object or list")

    items: list[DatasetItem] = []
    for index, raw in enumerate(raw_items):
        if not isinstance(raw, Mapping):
            raise TrainingError(f"dataset item {index} is not an object")
        status = str(raw.get("status", raw.get("review_status", "approved"))).lower()
        if status not in {"approved", "accepted"}:
            continue
        split = str(raw.get("split", "train")).lower()
        if split not in {"train", "validation", "test"}:
            raise TrainingError(f"dataset item {index} has invalid split {split!r}")
        audio_value = next(
            (
                raw.get(key)
                for key in ("training_audio", "training_path", "derived_path", "audio_path", "audio", "path")
                if raw.get(key)
            ),
            None,
        )
        text = str(
            raw.get("training_text", raw.get("transcript", raw.get("text", "")))
        ).strip()
        if audio_value is None or not text:
            raise TrainingError(f"dataset item {index} lacks training audio or transcript")
        audio_path = Path(str(audio_value)).expanduser()
        if not audio_path.is_absolute():
            audio_path = (base_dir / audio_path).resolve()
        if not audio_path.is_file():
            raise TrainingError(f"dataset audio is missing: {audio_path}")
        expected_audio_hash = raw.get("audio_sha256")
        if expected_audio_hash and _sha256(audio_path) != str(expected_audio_hash):
            raise TrainingError(
                f"frozen dataset audio checksum mismatch: {audio_path}"
            )
        duration = float(raw.get("duration_seconds") or _audio_duration(audio_path))
        if duration <= 0:
            raise TrainingError(f"cannot determine positive audio duration: {audio_path}")
        qa = raw.get("qa") if isinstance(raw.get("qa"), Mapping) else {}
        items.append(DatasetItem(audio_path, text, split, duration, dict(qa)))

    if not items:
        raise TrainingError("frozen dataset contains no approved recordings")
    if not any(item.split == "train" for item in items):
        raise TrainingError("frozen dataset contains no approved training split")

    source_hash = (
        _sha256(manifest_path)
        if manifest_path is not None
        else hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
    )
    metadata = {
        **metadata,
        "manifest_path": str(manifest_path) if manifest_path else None,
        "sha256": source_hash,
        "approved_items": len(items),
        "accepted_seconds": sum(item.duration_seconds for item in items),
    }
    return items, metadata


def select_speaker_reference(items: Sequence[DatasetItem]) -> DatasetItem:
    """Choose the cleanest approved 6–10 second training clip deterministically."""

    candidates = [
        item
        for item in items
        if item.split == "train" and 6.0 <= item.duration_seconds <= 10.0
    ]
    if not candidates:
        raise TrainingError(
            "speaker extraction requires an approved 6–10 second training clip"
        )

    def score(item: DatasetItem) -> tuple[float, float, float, str]:
        qa = item.qa
        snr = float(qa.get("snr_db", 0.0) or 0.0)
        clipping = float(qa.get("clipping_ratio", 0.0) or 0.0)
        wer = float(qa.get("wer", 0.0) or 0.0)
        # Prefer high SNR, then less clipping/transcript mismatch, then duration
        # near eight seconds.  The path makes ties stable across runs.
        return (snr, -clipping, -wer - abs(item.duration_seconds - 8.0), str(item.audio_path))

    return max(candidates, key=score)


def _target_module_path(path: str, targets: Sequence[str]) -> bool:
    in_transformer = (
        "talker.model.layers." in path
        or "talker.code_predictor.model.layers." in path
    )
    return in_transformer and any(path.endswith(target) for target in targets)


def _apply_lora(model: Any, config: TrainingConfig) -> list[str]:
    """Replace every configured talker/code-predictor projection with LoRA."""

    import mlx.nn as nn
    from mlx.utils import tree_unflatten
    from mlx_lm.tuner.lora import LoRALinear

    replacements: list[tuple[str, Any]] = []
    for path, module in model.named_modules():
        if not _target_module_path(path, config.lora_targets):
            continue
        if isinstance(module, LoRALinear):
            continue
        if not isinstance(module, (nn.Linear, nn.QuantizedLinear)):
            raise TrainingError(f"LoRA target is not linear: {path} ({type(module).__name__})")
        replacements.append(
            (
                path,
                LoRALinear.from_base(
                    module,
                    r=config.lora_rank,
                    scale=config.lora_scale,
                    dropout=config.lora_dropout,
                ),
            )
        )
    if not replacements:
        raise TrainingError("no Qwen talker/code-predictor LoRA targets were found")
    model.update_modules(tree_unflatten(replacements))
    return [path for path, _ in replacements]


def _enable_gradient_checkpointing(model: Any) -> Callable[[], None]:
    """Checkpoint both transformer stacks and return an exact restoration hook.

    MLX-LM implements checkpointing by replacing a decoder block class's
    ``__call__``.  Qwen training and generation share those Python classes, so
    leaving the patch installed would also checkpoint later inference models
    in the long-lived local server.  The returned hook confines the global
    class patch to one training run.
    """

    import mlx.core as mx

    patched: list[tuple[type, Any]] = []

    def patch(layer: Any) -> None:
        cls = type(layer)
        if getattr(cls, "_voicetrainer_gradient_checkpointed", False):
            return
        original_call = cls.__call__

        def checkpointed_call(module: Any, *args: Any, **kwargs: Any) -> Any:
            def inner(parameters: Any, *inner_args: Any, **inner_kwargs: Any) -> Any:
                module.update(parameters)
                return original_call(module, *inner_args, **inner_kwargs)

            return mx.checkpoint(inner)(module.trainable_parameters(), *args, **kwargs)

        cls.__call__ = checkpointed_call
        cls._voicetrainer_gradient_checkpointed = True
        patched.append((cls, original_call))

    patch(model.talker.model.layers[0])
    patch(model.talker.code_predictor.model.layers[0])

    def restore() -> None:
        for cls, original_call in reversed(patched):
            cls.__call__ = original_call
            try:
                delattr(cls, "_voicetrainer_gradient_checkpointed")
            except AttributeError:
                pass

    return restore


def _build_training_example(
    model: Any,
    item: DatasetItem,
    speaker_embedding: Any,
    *,
    language: str = "english",
    speaker_slot: int = RESERVED_SPEAKER_SLOT,
) -> dict[str, Any]:
    """Construct Qwen's two-channel teacher-forcing sequence for one clip."""

    import mlx.core as mx
    from mlx_audio.utils import load_audio

    audio = load_audio(str(item.audio_path), sample_rate=24000)
    if model.speech_tokenizer is None or not model.speech_tokenizer.has_encoder:
        raise TrainingError("the local Base model speech tokenizer has no encoder")
    encoder_audio = audio[None, None, :] if audio.ndim == 1 else audio
    codes = model.speech_tokenizer.encode(encoder_audio)
    # MLX-Audio returns [batch, codebook, time]; the SFT recipe uses [time, 16].
    if codes.ndim != 3:
        raise TrainingError(f"unexpected tokenizer code shape: {codes.shape}")
    codes = mx.transpose(codes[0], (1, 0)).astype(mx.int32)
    if codes.shape[1] != EXPECTED_CODEBOOKS:
        raise TrainingError(f"expected {EXPECTED_CODEBOOKS} audio codebooks, got {codes.shape[1]}")

    chat = f"<|im_start|>assistant\n{item.text}<|im_end|>\n<|im_start|>assistant\n"
    text_ids = mx.array(model.tokenizer.encode(chat), dtype=mx.int32)[None, :-5]
    text_length = text_ids.shape[1]
    codec_length = codes.shape[0]
    cfg = model.config
    talker_cfg = cfg.talker_config

    language_key = language.lower()
    language_ids = talker_cfg.codec_language_id or {}
    if language_key not in language_ids:
        raise TrainingError(f"Qwen talker has no codec language id for {language!r}")
    # Match the exact CustomVoice prefix used by MLX-Audio when generation is
    # called with ``lang_code='english'``: think, think-BOS, language,
    # think-EOS, speaker, pad, then codec-BOS.  The final BOS is teacher-forced
    # immediately before the first acoustic target rather than placed here.
    codec_prefix = mx.array(
        [
            talker_cfg.codec_think_id,
            talker_cfg.codec_think_bos_id,
            language_ids[language_key],
            talker_cfg.codec_think_eos_id,
            speaker_slot,
            talker_cfg.codec_pad_id,
        ],
        dtype=mx.int32,
    )
    prefix_start = 3
    text_start = prefix_start + int(codec_prefix.shape[0])
    text_token_count = text_length - 3
    if text_token_count < 1:
        raise TrainingError("training transcript produced no text tokens")
    # The CustomVoice prefill ends with the first transcript token overlaid on
    # codec BOS.  Acoustic teacher forcing then begins at the following
    # position, exactly as autoregressive inference does after its prefill.
    audio_start = text_start + 1
    trailing_text_count = max(text_token_count - 1, 0) + 1  # remaining text + TTS EOS
    if trailing_text_count > codec_length + 1:
        raise TrainingError(
            "transcript is too long for its encoded audio; review the take and transcript"
        )
    total_length = audio_start + codec_length + 1

    text_channel = mx.zeros((1, total_length), dtype=mx.int32)
    codec_channel = mx.zeros((1, total_length), dtype=mx.int32)
    all_codes = mx.zeros((1, total_length, EXPECTED_CODEBOOKS), dtype=mx.int32)
    text_mask = mx.zeros((1, total_length, 1), dtype=mx.bool_)
    codec_embedding_mask = mx.zeros((1, total_length, 1), dtype=mx.bool_)
    codec_mask = mx.zeros((1, total_length), dtype=mx.bool_)
    attention_mask = mx.zeros((1, total_length), dtype=mx.float32)
    labels = mx.full((1, total_length), -100, dtype=mx.int32)

    # Text channel.  The prefix through the first real text token is identical
    # to ``Model._prepare_generation_inputs(..., language='english',
    # speaker='ruhi')``.  The remainder overlays each trailing text embedding
    # with the preceding 16-codebook acoustic frame, matching autoregressive
    # inference one-for-one.
    # MLX 0.32 exposes indexed replacement through normal assignment.  Its
    # ``ArrayAt`` helper only implements reduction updates (add/multiply/etc.),
    # so JAX-style ``.at[...].set(...)`` is not available in our locked stack.
    text_channel[:, :3] = text_ids[:, :3]
    text_channel[:, prefix_start : text_start - 1] = cfg.tts_pad_token_id
    text_channel[:, text_start - 1] = cfg.tts_bos_token_id
    text_channel[:, text_start] = text_ids[:, 3]
    remaining_text = text_ids[:, 4:]
    remaining_end = audio_start + remaining_text.shape[1]
    text_channel[:, audio_start:remaining_end] = remaining_text
    text_channel[:, remaining_end] = cfg.tts_eos_token_id
    text_channel[:, remaining_end + 1 :] = cfg.tts_pad_token_id
    text_mask[:, :, :] = 1.0

    codec_channel[:, prefix_start:text_start] = codec_prefix
    codec_channel[:, text_start] = talker_cfg.codec_bos_id
    codec_channel[:, audio_start : audio_start + codec_length] = codes[:, 0]
    codec_channel[:, audio_start + codec_length] = talker_cfg.codec_eos_token_id
    labels[:, audio_start : audio_start + codec_length] = codes[:, 0]
    labels[:, audio_start + codec_length] = talker_cfg.codec_eos_token_id
    all_codes[:, audio_start : audio_start + codec_length, :] = codes
    codec_embedding_mask[:, prefix_start:, :] = 1.0
    codec_mask[:, audio_start : audio_start + codec_length] = True
    attention_mask[:, :] = 1.0

    text_embedding = model.talker.text_projection(
        model.talker.get_text_embeddings()(text_channel)
    ) * text_mask
    codec_embedding_layer = model.talker.get_input_embeddings()
    if speaker_embedding.reshape(-1).shape != codec_embedding_layer.weight.shape[1:]:
        raise TrainingError("speaker embedding shape does not match reserved codec slot")
    # The extracted vector is installed in the actual frozen slot used at
    # inference.  It is persisted separately, not folded into the LoRA file.
    codec_embedding_layer.weight[speaker_slot] = speaker_embedding.reshape(-1).astype(
        codec_embedding_layer.weight.dtype
    )
    codec_embedding = codec_embedding_layer(codec_channel) * codec_embedding_mask
    for codebook in range(1, EXPECTED_CODEBOOKS):
        embedded = model.talker.code_predictor.model.codec_embedding[codebook - 1](
            all_codes[:, :, codebook]
        )
        codec_embedding = codec_embedding + embedded * codec_mask[..., None]
    input_embeddings = text_embedding + codec_embedding

    return {
        "input_embeddings": input_embeddings,
        "attention_mask": attention_mask,
        "labels": labels,
        "codec_mask": codec_mask,
        "codes": all_codes,
        "audio_tokens": int(codec_length),
        "audio_start": int(audio_start),
    }


def _masked_cross_entropy(logits: Any, labels: Any, mask: Any) -> Any:
    import mlx.core as mx
    import mlx.nn as nn

    safe_labels = mx.where(mask, labels, mx.zeros_like(labels))
    losses = nn.losses.cross_entropy(logits, safe_labels, reduction="none")
    denominator = mx.maximum(mask.sum(), mx.array(1))
    return (losses * mask).astype(mx.float32).sum() / denominator


def _speaker_sft_loss(model: Any, example: Mapping[str, Any], sub_loss_weight: float) -> tuple[Any, Any, Any]:
    """Compute codebook-0 and weighted 15-codebook predictor losses."""

    import mlx.core as mx

    # Accept either the complete MLX-Audio model or its talker.  Training the
    # talker directly is important with MLX-Audio 0.4.5: the loaded root model
    # owns a compiled speech-tokenizer decoder (``mlx.gc_func``), and
    # ``nn.value_and_grad(root, ...)`` tries to recurse into that opaque object
    # while installing parameter tracers.  The only trainable parameters are
    # in the talker, so differentiating this exact submodule is both correct
    # and avoids touching the frozen tokenizer/speaker encoder.
    talker = getattr(model, "talker", model)

    embeddings = example["input_embeddings"]
    attention = example["attention_mask"]
    labels = example["labels"]
    codec_mask = example["codec_mask"]
    codes = example["codes"]

    logits, hidden = talker(
        embeddings[:, :-1, :], attention_mask=attention[:, :-1]
    )
    shifted_labels = labels[:, 1:]
    main_mask = shifted_labels >= 0
    main_loss = _masked_cross_entropy(logits, shifted_labels, main_mask)

    # MLX does not support boolean advanced indexing.  Batch size is fixed at
    # one and the audio span is contiguous, so use the exact collator offsets.
    audio_start = int(example["audio_start"])
    audio_length = int(example["audio_tokens"])
    audio_hidden = hidden[
        :, audio_start - 1 : audio_start - 1 + audio_length, :
    ].reshape(audio_length, -1)
    audio_codes = codes[
        :, audio_start : audio_start + audio_length, :
    ].reshape(audio_length, EXPECTED_CODEBOOKS)
    if audio_hidden.shape[0] != audio_codes.shape[0]:
        raise TrainingError(
            "talker hidden/audio-code alignment failed; refusing a corrupt checkpoint"
        )

    predictor_inputs = [audio_hidden[:, None, :]]
    predictor_inputs.append(talker.get_input_embeddings()(audio_codes[:, :1]))
    for index in range(1, EXPECTED_CODEBOOKS - 1):
        predictor_inputs.append(
            talker.code_predictor.model.codec_embedding[index - 1](
                audio_codes[:, index : index + 1]
            )
        )
    predictor_inputs = mx.concatenate(predictor_inputs, axis=1)
    predictor = talker.code_predictor
    if predictor.small_to_mtp_projection is not None:
        predictor_inputs = predictor.small_to_mtp_projection(predictor_inputs)
    predictor_hidden = predictor.model(predictor_inputs)
    sub_logits = mx.stack(
        [predictor.lm_head[index](predictor_hidden[:, index + 1, :]) for index in range(15)],
        axis=1,
    )
    sub_labels = audio_codes[:, 1:]
    sub_loss = _masked_cross_entropy(
        sub_logits,
        sub_labels,
        mx.ones(sub_labels.shape, dtype=mx.bool_),
    )
    total = main_loss + sub_loss_weight * sub_loss
    return total, main_loss, sub_loss


def _copy_dataset_manifest(
    dataset_manifest: str | Path | Mapping[str, Any], destination: Path
) -> None:
    if isinstance(dataset_manifest, Mapping):
        destination.write_text(
            json.dumps(dict(dataset_manifest), indent=2, sort_keys=True), encoding="utf-8"
        )
    else:
        source = Path(dataset_manifest).expanduser().resolve()
        if source.suffix.lower() == ".jsonl":
            destination = destination.with_suffix(".jsonl")
        shutil.copy2(source, destination)


def _save_checkpoint(
    *,
    model: Any,
    speaker_embedding: Any,
    checkpoint_dir: Path,
    config: TrainingConfig,
    base_model: Mapping[str, Any],
    dataset: Mapping[str, Any],
    voice_variant: Mapping[str, str],
    target_modules: Sequence[str],
    epoch: int,
    metrics: Mapping[str, Any],
    dataset_manifest: str | Path | Mapping[str, Any],
) -> Path:
    import mlx.core as mx
    from mlx.utils import tree_flatten

    if not config.variant_id or not config.speaker_key:
        raise TrainingError("checkpoint configuration has no resolved voice variant")
    if config.variant_id != voice_variant.get("variant_id"):
        raise TrainingError("checkpoint variant differs from its dataset variant")
    if config.speaker_key != voice_variant.get("speaker_key"):
        raise TrainingError("checkpoint speaker key differs from its dataset variant")

    if checkpoint_dir.exists():
        raise TrainingError(f"refusing to overwrite checkpoint: {checkpoint_dir}")
    partial = checkpoint_dir.with_name(f".{checkpoint_dir.name}.partial")
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True, mode=0o700)
    try:
        adapter_weights = dict(tree_flatten(model.trainable_parameters()))
        if not adapter_weights:
            raise TrainingError("LoRA adapter unexpectedly has no trainable weights")
        mx.save_safetensors(str(partial / "adapters.safetensors"), adapter_weights)
        mx.save_safetensors(
            str(partial / "speaker_embedding.safetensors"),
            {"speaker_embedding": speaker_embedding.reshape(-1)},
        )
        (partial / "training_config.json").write_text(
            json.dumps(config.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
        )
        (partial / "metrics.json").write_text(
            json.dumps(dict(metrics), indent=2, sort_keys=True), encoding="utf-8"
        )
        _copy_dataset_manifest(dataset_manifest, partial / "dataset_manifest.json")
        artifact = {
            "kind": ARTIFACT_KIND,
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "created_at": time.time(),
            "epoch": epoch,
            "speaker": {
                "name": config.speaker_key,
                "slot": config.speaker_slot,
                "embedding_file": "speaker_embedding.safetensors",
            },
            "voice_variant": dict(voice_variant),
            "mode": "custom_voice",
            "reference_audio_required": False,
            "base_model": dict(base_model),
            "dataset": dict(dataset),
            "adapter": {
                "file": "adapters.safetensors",
                "rank": config.lora_rank,
                "alpha": config.lora_alpha,
                "scale": config.lora_scale,
                "dropout": config.lora_dropout,
                "targets": list(config.lora_targets),
                "target_modules": list(target_modules),
            },
            "objective": {
                "main": "autoregressive_codebook_0_cross_entropy",
                "auxiliary": "codebooks_1_to_15_cross_entropy",
                "auxiliary_weight": config.sub_talker_loss_weight,
            },
            "training_config_file": "training_config.json",
            "metrics_file": "metrics.json",
            "evaluation": {"status": "pending", "promoted": False},
        }
        (partial / "artifact.json").write_text(
            json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8"
        )
        for path in partial.iterdir():
            path.chmod(0o400)
        partial.rename(checkpoint_dir)
        checkpoint_dir.chmod(0o700)
    except Exception:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    return checkpoint_dir


def train_named_speaker(
    model_path: str | Path,
    dataset_manifest: str | Path | Mapping[str, Any],
    artifact_root: str | Path,
    config: TrainingConfig | Mapping[str, Any] | None = None,
    progress_cb: ProgressCallback | None = None,
    cancel_cb: CancelCallback | None = None,
) -> dict[str, Any]:
    """Train and save versioned reference-free named-speaker checkpoints.

    No package resolution or model download is performed.  ``model_path`` must
    be an existing absolute local directory.  The return value contains plain
    paths/dicts suitable for the FastAPI/store layers.
    """

    _offline_environment()
    if config is None:
        training_config = TrainingConfig()
    elif isinstance(config, TrainingConfig):
        training_config = config
    else:
        training_config = TrainingConfig.from_dict(config)

    local_model_path = validate_local_model_path(model_path)
    items, dataset_metadata = load_frozen_dataset(dataset_manifest)
    training_config, voice_variant = resolve_training_variant(
        training_config, dataset_metadata
    )
    train_items = [item for item in items if item.split == "train"]
    accepted_seconds = sum(item.duration_seconds for item in items)
    if accepted_seconds < training_config.minimum_dataset_seconds:
        required = training_config.minimum_dataset_seconds
        raise TrainingError(
            f"{training_config.stage} training requires {required:.0f}s accepted audio; "
            f"dataset has {accepted_seconds:.1f}s"
        )
    reference_item = select_speaker_reference(items)
    root = Path(artifact_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    run_id = time.strftime("%Y%m%dT%H%M%S") + f"-{training_config.stage}-{training_config.seed}"
    run_dir = root / run_id
    if run_dir.exists():
        raise TrainingError(f"training run already exists: {run_dir}")
    run_dir.mkdir(mode=0o700)

    _emit(progress_cb, "loading_model", model_path=str(local_model_path))
    _check_cancelled(cancel_cb)
    try:
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim
        from mlx.utils import tree_flatten, tree_map
        from mlx_audio.tts.utils import load_model
        from mlx_audio.utils import load_audio
    except ImportError as exc:
        raise TrainingError(
            "native training dependencies are unavailable; install the locked local environment"
        ) from exc

    mx.random.seed(training_config.seed)
    randomizer = random.Random(training_config.seed)
    model = load_model(local_model_path, lazy=False, strict=True)
    if model.tokenizer is None or model.speech_tokenizer is None:
        raise TrainingError("failed to load the local text or speech tokenizer")
    model.freeze()
    target_modules = _apply_lora(model, training_config)
    model.train()

    _emit(progress_cb, "extracting_speaker", audio_path=str(reference_item.audio_path))
    reference_audio = load_audio(str(reference_item.audio_path), sample_rate=24000)
    speaker_embedding = model.extract_speaker_embedding(reference_audio).reshape(-1)
    if speaker_embedding.shape[0] != model.config.talker_config.hidden_size:
        raise TrainingError(
            "speaker encoder embedding size does not match the talker codec embedding"
        )
    mx.eval(speaker_embedding)
    _check_cancelled(cancel_cb)

    _emit(progress_cb, "fingerprinting_model")
    base_fingerprint = fingerprint_model(local_model_path)
    dataset_metadata = {
        **dataset_metadata,
        "voice_variant": voice_variant,
        "reference_recording": str(reference_item.audio_path),
        "training_items": len(train_items),
        "stage": training_config.stage,
    }

    optimizer = optim.AdamW(
        learning_rate=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    def loss_function(current_talker: Any, example: Mapping[str, Any]) -> tuple[Any, tuple[Any, Any]]:
        total, main, auxiliary = _speaker_sft_loss(
            current_talker, example, training_config.sub_talker_loss_weight
        )
        return total, (main, auxiliary)

    value_and_grad = nn.value_and_grad(model.talker, loss_function)
    checkpoints: list[str] = []
    total_microsteps = len(train_items) * training_config.epochs
    completed_microsteps = 0
    restore_gradient_checkpointing: Callable[[], None] = lambda: None
    if training_config.gradient_checkpointing:
        restore_gradient_checkpointing = _enable_gradient_checkpointing(model)

    try:
        for epoch in range(1, training_config.epochs + 1):
            epoch_items = list(train_items)
            randomizer.shuffle(epoch_items)
            accumulated_grad = None
            accumulated_count = 0
            epoch_total = 0.0
            epoch_main = 0.0
            epoch_auxiliary = 0.0
            epoch_tokens = 0

            for item_index, item in enumerate(epoch_items, start=1):
                _check_cancelled(cancel_cb)
                example = _build_training_example(
                    model,
                    item,
                    speaker_embedding,
                    language=training_config.language,
                    speaker_slot=training_config.speaker_slot,
                )
                (loss_value, (main_loss, auxiliary_loss)), gradients = value_and_grad(
                    model.talker, example
                )
                accumulated_grad = (
                    gradients
                    if accumulated_grad is None
                    else tree_map(lambda left, right: left + right, accumulated_grad, gradients)
                )
                accumulated_count += 1
                is_epoch_end = item_index == len(epoch_items)
                if (
                    accumulated_count >= training_config.gradient_accumulation_steps
                    or is_epoch_end
                ):
                    divisor = float(accumulated_count)
                    accumulated_grad = tree_map(lambda value: value / divisor, accumulated_grad)
                    if training_config.gradient_clip_norm > 0:
                        flat_gradients = [
                            value for _, value in tree_flatten(accumulated_grad)
                        ]
                        squared_norm = sum(
                            (value.astype(mx.float32) ** 2).sum()
                            for value in flat_gradients
                        )
                        gradient_norm = mx.sqrt(squared_norm)
                        gradient_scale = mx.minimum(
                            mx.array(1.0),
                            training_config.gradient_clip_norm
                            / (gradient_norm + 1e-6),
                        )
                        accumulated_grad = tree_map(
                            lambda value: value * gradient_scale, accumulated_grad
                        )
                    optimizer.update(model.talker, accumulated_grad)
                    mx.eval(model.parameters(), optimizer.state)
                    accumulated_grad = None
                    accumulated_count = 0

                mx.eval(loss_value, main_loss, auxiliary_loss)
                epoch_total += float(loss_value.item())
                epoch_main += float(main_loss.item())
                epoch_auxiliary += float(auxiliary_loss.item())
                epoch_tokens += int(example["audio_tokens"])
                completed_microsteps += 1
                _emit(
                    progress_cb,
                    "training_step",
                    epoch=epoch,
                    epochs=training_config.epochs,
                    step=item_index,
                    steps=len(epoch_items),
                    microsteps_completed=completed_microsteps,
                    microsteps_total=total_microsteps,
                    loss=float(loss_value.item()),
                    main_loss=float(main_loss.item()),
                    auxiliary_loss=float(auxiliary_loss.item()),
                    peak_memory_gb=mx.get_peak_memory() / 1e9,
                )
                del example, gradients
                mx.clear_cache()

            count = len(epoch_items)
            metrics = {
                "epoch": epoch,
                "training_loss": epoch_total / count,
                "codebook_0_loss": epoch_main / count,
                "codebooks_1_to_15_loss": epoch_auxiliary / count,
                "audio_tokens": epoch_tokens,
                "evaluation_status": "pending",
            }
            checkpoint = _save_checkpoint(
                model=model,
                speaker_embedding=speaker_embedding,
                checkpoint_dir=run_dir / f"checkpoint-epoch-{epoch}",
                config=training_config,
                base_model=base_fingerprint,
                dataset=dataset_metadata,
                voice_variant=voice_variant,
                target_modules=target_modules,
                epoch=epoch,
                metrics=metrics,
                dataset_manifest=dataset_manifest,
            )
            checkpoints.append(str(checkpoint))
            _emit(progress_cb, "checkpoint_saved", epoch=epoch, path=str(checkpoint))
    except TrainingCancelled:
        (run_dir / "cancelled.json").write_text(
            json.dumps({"cancelled_at": time.time(), "checkpoints": checkpoints}, indent=2),
            encoding="utf-8",
        )
        raise
    except Exception as exc:
        (run_dir / "failed.json").write_text(
            json.dumps(
                {"failed_at": time.time(), "error": str(exc), "checkpoints": checkpoints},
                indent=2,
            ),
            encoding="utf-8",
        )
        raise
    finally:
        restore_gradient_checkpointing()

    run_manifest = {
        "run_id": run_id,
        "stage": training_config.stage,
        "speaker": training_config.speaker_key,
        "voice_variant": voice_variant,
        "reference_audio_required": False,
        "checkpoints": checkpoints,
        "latest_checkpoint": checkpoints[-1],
        "training_config": training_config.to_dict(),
        "dataset": dataset_metadata,
        "completed_at": time.time(),
    }
    (run_dir / "run.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    _emit(progress_cb, "training_complete", **run_manifest)
    return run_manifest


__all__ = [
    "ARTIFACT_KIND",
    "ARTIFACT_SCHEMA_VERSION",
    "DEFAULT_VARIANT_DISPLAY_NAME",
    "DEFAULT_VARIANT_ID",
    "DEFAULT_LORA_TARGETS",
    "DEFAULT_SPEAKER_NAME",
    "RESERVED_SPEAKER_SLOT",
    "DatasetItem",
    "TrainingCancelled",
    "TrainingConfig",
    "TrainingError",
    "fingerprint_model",
    "load_frozen_dataset",
    "resolve_dataset_voice_variant",
    "resolve_training_variant",
    "select_speaker_reference",
    "stable_speaker_key",
    "train_named_speaker",
    "validate_local_model_path",
]
