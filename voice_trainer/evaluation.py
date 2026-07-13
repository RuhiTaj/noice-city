"""Offline checkpoint evaluation and promotion gates for a trained voice.

Evaluation is deliberately separate from training and serving.  Every named
speaker checkpoint speaks the same immutable set of twenty English narration
sentences without reference audio, is transcribed by the local Whisper model,
and is compared with held-out real recordings using Qwen3-TTS's own local
ECAPA-TDNN speaker encoder.

The held-out recordings are evaluation inputs only.  They are never forwarded
to :class:`~voice_trainer.model_runtime.NamedSpeakerRuntime.generate`.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
import shutil
import wave
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .model_runtime import DeliverySettings, NamedSpeakerRuntime
from .training import _offline_environment, validate_local_model_path
from .transcription import normalize_transcript, transcribe_local, word_error_rate


EVALUATION_SCHEMA_VERSION = 1
EVALUATION_SEED = 3407
INTELLIGIBLE_WER_LIMIT = 0.50


class EvaluationError(RuntimeError):
    """Raised when a checkpoint cannot be evaluated under the locked contract."""


@dataclass(frozen=True)
class LockedTestPrompt:
    id: str
    text: str


# This is intentionally a code-level constant rather than a dynamically chosen
# subset of the recording catalog.  Changing any id, order, or sentence changes
# LOCKED_PROMPT_SET_SHA256 and therefore creates a different evaluation protocol.
LOCKED_TEST_PROMPTS = (
    LockedTestPrompt(
        "locked-narration-001",
        "At sunrise, a narrow ribbon of fog still rested above the quiet river.",
    ),
    LockedTestPrompt(
        "locked-narration-002",
        "The archive preserves every revision, so an earlier decision can be examined later.",
    ),
    LockedTestPrompt(
        "locked-narration-003",
        "By half past four, the final passenger had stepped safely onto the platform.",
    ),
    LockedTestPrompt(
        "locked-narration-004",
        "A practical explanation begins with the goal, then introduces each necessary detail.",
    ),
    LockedTestPrompt(
        "locked-narration-005",
        "Beyond the open window, dry leaves moved softly across the stone courtyard.",
    ),
    LockedTestPrompt(
        "locked-narration-006",
        "The research team checked the figures twice before publishing its short report.",
    ),
    LockedTestPrompt(
        "locked-narration-007",
        "Three small choices shaped the outcome: preparation, patience, and careful timing.",
    ),
    LockedTestPrompt(
        "locked-narration-008",
        "Although the road curved sharply, the distant lighthouse remained easy to see.",
    ),
    LockedTestPrompt(
        "locked-narration-009",
        "Maya lowered her voice and said, 'We should listen before we decide.'",
    ),
    LockedTestPrompt(
        "locked-narration-010",
        "The package contained six glass bottles, two notebooks, and a handwritten receipt.",
    ),
    LockedTestPrompt(
        "locked-narration-011",
        "Why did the simplest question produce the most useful answer?",
    ),
    LockedTestPrompt(
        "locked-narration-012",
        "For a moment, the crowded hall became completely still.",
    ),
    LockedTestPrompt(
        "locked-narration-013",
        "Good narration sounds composed and direct, even when the subject is complicated.",
    ),
    LockedTestPrompt(
        "locked-narration-014",
        "After reviewing the map, Elena chose the longer path through the northern valley.",
    ),
    LockedTestPrompt(
        "locked-narration-015",
        "The first test measured clarity; the second measured consistency under pressure.",
    ),
    LockedTestPrompt(
        "locked-narration-016",
        "A warm lamp illuminated the desk, but the rest of the library remained dark.",
    ),
    LockedTestPrompt(
        "locked-narration-017",
        "Each recording was labeled, reviewed, and stored before the next session began.",
    ),
    LockedTestPrompt(
        "locked-narration-018",
        "Nearly forty minutes passed before the clouds finally revealed a patch of blue.",
    ),
    LockedTestPrompt(
        "locked-narration-019",
        "Keep the conclusion brief, emphasize the evidence, and leave room for one question.",
    ),
    LockedTestPrompt(
        "locked-narration-020",
        "The finished story should feel natural, steady, and unmistakably human.",
    ),
)


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


LOCKED_PROMPT_SET_SHA256 = hashlib.sha256(
    _canonical_json([asdict(prompt) for prompt in LOCKED_TEST_PROMPTS]).encode("utf-8")
).hexdigest()


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(6)}.partial")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.chmod(0o400)
        os.replace(temporary, path)
        path.chmod(0o400)
    finally:
        temporary.unlink(missing_ok=True)


def _pcm_to_float(raw: bytes, sample_width: int) -> np.ndarray:
    if sample_width == 1:
        return (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    if sample_width == 2:
        return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if sample_width == 3:
        packed = np.frombuffer(raw, dtype=np.uint8)
        if packed.size % 3:
            raise EvaluationError("generated WAV contains incomplete 24-bit samples")
        packed = packed.reshape(-1, 3).astype(np.int32)
        values = packed[:, 0] | (packed[:, 1] << 8) | (packed[:, 2] << 16)
        values = np.where(values & 0x800000, values - 0x1000000, values)
        return values.astype(np.float32) / 8388608.0
    if sample_width == 4:
        return np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    raise EvaluationError(f"unsupported generated WAV sample width: {sample_width}")


def _read_mono_wav(path: Path) -> tuple[np.ndarray, int]:
    try:
        with wave.open(str(path), "rb") as wav:
            if wav.getnchannels() != 1:
                raise EvaluationError("generated evaluation audio must be mono")
            sample_rate = wav.getframerate()
            sample_width = wav.getsampwidth()
            samples = _pcm_to_float(wav.readframes(wav.getnframes()), sample_width)
    except (EOFError, wave.Error) as exc:
        raise EvaluationError(f"invalid generated WAV {path}: {exc}") from exc
    if sample_rate <= 0 or samples.size == 0 or not np.isfinite(samples).all():
        raise EvaluationError(f"generated WAV is empty or invalid: {path}")
    return samples, sample_rate


def _max_consecutive_word_run(words: Sequence[str]) -> int:
    maximum = 0
    current = 0
    previous: str | None = None
    for word in words:
        current = current + 1 if word == previous else 1
        maximum = max(maximum, current)
        previous = word
    return maximum


def _has_repeated_phrase_loop(words: Sequence[str]) -> bool:
    """Detect a two-to-four word phrase repeated three times without a gap."""

    for width in range(2, 5):
        span = width * 3
        for start in range(0, len(words) - span + 1):
            phrase = list(words[start : start + width])
            if all(
                list(words[start + repeat * width : start + (repeat + 1) * width])
                == phrase
                for repeat in (1, 2)
            ):
                return True
    return False


def analyze_generated_wav(
    audio_path: str | Path,
    *,
    expected_text: str,
    asr_text: str,
) -> dict[str, Any]:
    """Measure speech presence, silence, duration, repetition, and termination.

    The thresholds are intentionally broad enough for expressive narration.  A
    sentence is marked intelligible when local ASR finds speech and its WER is
    at most 50%; the stricter aggregate checkpoint gate is relative to the base
    model's WER.
    """

    path = Path(audio_path).expanduser().resolve()
    samples, sample_rate = _read_mono_wav(path)
    duration = float(samples.size / sample_rate)
    frame_size = max(1, round(sample_rate * 0.020))
    frame_count = math.ceil(samples.size / frame_size)
    padded = np.pad(samples, (0, frame_count * frame_size - samples.size))
    rms = np.sqrt(np.mean(padded.reshape(frame_count, frame_size) ** 2, axis=1))
    speech_threshold = max(0.003, float(np.max(rms)) * 0.08)
    voiced = rms >= speech_threshold
    voiced_indices = np.flatnonzero(voiced)
    if voiced_indices.size:
        leading_silence = float(voiced_indices[0] * frame_size / sample_rate)
        trailing_silence = float(
            max(0, samples.size - (voiced_indices[-1] + 1) * frame_size) / sample_rate
        )
    else:
        leading_silence = duration
        trailing_silence = duration
    silence_fraction = float(1.0 - np.mean(voiced))

    expected_words = normalize_transcript(expected_text).split()
    observed_words = normalize_transcript(asr_text).split()
    wer = word_error_rate(expected_text, asr_text)
    minimum_duration = max(0.35, len(expected_words) / 5.0)
    maximum_duration = max(2.0, len(expected_words) + 3.0)
    duration_ok = minimum_duration <= duration <= maximum_duration
    silence_ok = bool(
        voiced_indices.size
        and leading_silence <= 1.0
        and trailing_silence <= 1.5
        and silence_fraction <= 0.85
    )
    maximum_word_run = _max_consecutive_word_run(observed_words)
    phrase_loop = _has_repeated_phrase_loop(observed_words)
    repetition_ok = maximum_word_run <= 3 and not phrase_loop
    final_word_reached = bool(
        expected_words
        and observed_words
        and expected_words[-1] in observed_words[-3:]
    )
    termination_ok = final_word_reached and duration <= maximum_duration
    intelligible = bool(voiced_indices.size and observed_words and wer <= INTELLIGIBLE_WER_LIMIT)

    return {
        "sample_rate": sample_rate,
        "samples": int(samples.size),
        "duration_seconds": duration,
        "expected_duration_seconds": {
            "minimum": minimum_duration,
            "maximum": maximum_duration,
        },
        "peak_absolute": float(np.max(np.abs(samples))),
        "silence_fraction": silence_fraction,
        "leading_silence_seconds": leading_silence,
        "trailing_silence_seconds": trailing_silence,
        "silence_ok": silence_ok,
        "duration_ok": duration_ok,
        "maximum_consecutive_word_run": maximum_word_run,
        "repeated_phrase_loop": phrase_loop,
        "repetition_ok": repetition_ok,
        "final_expected_word_reached": final_word_reached,
        "termination_ok": termination_ok,
        "intelligible": intelligible,
        "wer": wer,
    }


def _unit_vector(value: Any) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    if not vector.size or not np.isfinite(vector).all():
        raise EvaluationError("speaker encoder returned an empty or non-finite embedding")
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise EvaluationError("speaker encoder returned a zero embedding")
    return vector / norm


class QwenSpeakerEmbedder:
    """Use the already-loaded Qwen Base model's local ECAPA speaker encoder."""

    def __init__(self, runtime: Any) -> None:
        runtime.load()
        model = getattr(runtime, "_model", None)
        if model is None or getattr(model, "speaker_encoder", None) is None:
            raise EvaluationError(
                "the installed Qwen3-TTS Base model has no local speaker encoder"
            )
        if not callable(getattr(model, "extract_speaker_embedding", None)):
            raise EvaluationError(
                "the installed Qwen3-TTS build cannot extract speaker embeddings"
            )
        self._model = model

    def __call__(self, audio_path: Path) -> np.ndarray:
        try:
            import mlx.core as mx
            from mlx_audio.utils import load_audio
        except ImportError as exc:
            raise EvaluationError("the locked MLX speaker-scoring runtime is unavailable") from exc
        audio = load_audio(str(audio_path), sample_rate=24000)
        embedding = self._model.extract_speaker_embedding(audio, sr=24000)
        mx.eval(embedding)
        return _unit_vector(embedding)


class UnadaptedBaseRuntime:
    """Reference-free runtime for establishing the unadapted Base benchmark.

    Qwen3-TTS Base supports unconditional generation when no voice, reference
    audio, or reference text is supplied.  This adapter deliberately exposes
    the same three-argument ``generate`` shape used by the checkpoint evaluator
    while never exposing a reference input.
    """

    def __init__(
        self,
        model_path: str | Path,
        _artifact_path: str | Path | None = None,
        *,
        verify_model: bool = True,
    ) -> None:
        _offline_environment()
        self.model_path = validate_local_model_path(model_path)
        if verify_model:
            _base_model_subject(self.model_path, full_hash=True)
        self._model: Any | None = None

    def load(self) -> "UnadaptedBaseRuntime":
        if self._model is not None:
            return self
        try:
            import mlx.core as mx
            from mlx_audio.tts.utils import load_model
        except ImportError as exc:
            raise EvaluationError("the locked MLX Base evaluation runtime is unavailable") from exc
        model = load_model(self.model_path, lazy=False, strict=True)
        if getattr(model.config, "tts_model_type", None) != "base":
            raise EvaluationError("the baseline evaluator requires an unadapted Base model")
        model.freeze()
        model.eval()
        mx.eval(model.parameters())
        self._model = model
        return self

    def generate(
        self,
        text: str,
        output_path: str | Path,
        delivery: DeliverySettings | Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        normalized_text = text.strip()
        if not normalized_text:
            raise ValueError("generation text cannot be empty")
        settings = DeliverySettings.from_value(delivery)
        if settings.instruction is not None:
            raise ValueError("the unadapted Qwen Base baseline does not support instructions")
        self.load()
        assert self._model is not None
        import mlx.core as mx

        results = list(
            self._model.generate(
                text=normalized_text,
                voice=None,
                instruct=None,
                temperature=settings.temperature,
                lang_code=settings.language,
                ref_audio=None,
                ref_text=None,
                max_tokens=settings.max_tokens,
                top_k=settings.top_k,
                top_p=settings.top_p,
                repetition_penalty=settings.repetition_penalty,
                verbose=False,
                stream=False,
            )
        )
        if not results:
            raise EvaluationError("Qwen3-TTS Base returned no generated audio")
        sample_rates = {int(result.sample_rate) for result in results}
        if len(sample_rates) != 1:
            raise EvaluationError("Qwen3-TTS Base returned inconsistent sample rates")
        sample_rate = sample_rates.pop()
        parts = [result.audio.reshape(-1) for result in results]
        audio = parts[0] if len(parts) == 1 else mx.concatenate(parts)
        mx.eval(audio)
        samples = np.asarray(audio, dtype=np.float32)
        if not samples.size or not np.isfinite(samples).all():
            raise EvaluationError("Qwen3-TTS Base returned empty or non-finite audio")
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
        return {
            "output_path": str(destination),
            "sample_rate": sample_rate,
            "samples": int(samples.size),
            "duration_seconds": float(samples.size / sample_rate),
            "base_model": str(self.model_path),
            "speaker": None,
            "delivery": settings.to_dict(),
            "reference_audio_required": False,
        }

    def unload(self) -> None:
        self._model = None
        try:
            import mlx.core as mx

            mx.clear_cache()
        except ImportError:
            pass


def _base_model_subject(model_path: str | Path, *, full_hash: bool) -> dict[str, Any]:
    """Validate the adopted local model manifest and describe the baseline."""

    root = validate_local_model_path(model_path)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise EvaluationError(
            "the unadapted baseline requires the adopted local model manifest"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"invalid local Base model manifest: {exc}") from exc
    records = manifest.get("files")
    if not isinstance(records, list) or not records:
        raise EvaluationError("local Base model manifest contains no files")
    for record in records:
        if not isinstance(record, Mapping):
            raise EvaluationError("local Base model manifest contains an invalid file record")
        relative = Path(str(record.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise EvaluationError("local Base model manifest path escapes its root")
        candidate = (root / relative).resolve(strict=True)
        if root not in candidate.parents:
            raise EvaluationError("local Base model manifest path escapes its root")
        if candidate.stat().st_size != int(record.get("size", -1)):
            raise EvaluationError(f"local Base model size mismatch: {relative}")
        if full_hash and _sha256(candidate) != record.get("sha256"):
            raise EvaluationError(f"local Base model hash mismatch: {relative}")
    return {
        "type": "unadapted_base_model",
        "model_path": str(root),
        "model_manifest_sha256": _sha256(manifest_path),
        "repo": manifest.get("repo"),
        "revision": manifest.get("revision"),
        "format": manifest.get("format"),
        "integrity": "full_hash" if full_hash else "size_and_manifest",
    }


RuntimeFactory = Callable[..., Any]
Transcriber = Callable[[Path, str], Mapping[str, Any]]
SpeakerEmbedder = Callable[[Path], Any]
ProgressCallback = Callable[[Mapping[str, Any]], None]


def _aggregate(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(items) != len(LOCKED_TEST_PROMPTS):
        raise EvaluationError("an evaluation must contain all twenty locked prompts")
    expected_words = [
        max(1, len(normalize_transcript(str(item["text"])).split())) for item in items
    ]
    weighted_errors = sum(
        float(item["asr"]["wer"]) * word_count
        for item, word_count in zip(items, expected_words, strict=True)
    )
    return {
        "locked_prompt_count": len(items),
        "wer": float(weighted_errors / sum(expected_words)),
        "mean_sentence_wer": float(
            np.mean([float(item["asr"]["wer"]) for item in items])
        ),
        "speaker_similarity": float(
            np.mean([float(item["speaker_similarity"]) for item in items])
        ),
        "intelligible_count": sum(bool(item["quality"]["intelligible"]) for item in items),
        "all_intelligible": all(bool(item["quality"]["intelligible"]) for item in items),
        "reference_free": all(
            item["generation"].get("reference_audio_required") is False for item in items
        ),
        "repetition_error_count": sum(
            not bool(item["quality"]["repetition_ok"]) for item in items
        ),
        "silence_error_count": sum(not bool(item["quality"]["silence_ok"]) for item in items),
        "termination_error_count": sum(
            not bool(item["quality"]["termination_ok"]) for item in items
        ),
        "duration_error_count": sum(
            not bool(item["quality"]["duration_ok"]) for item in items
        ),
        "total_duration_seconds": float(
            sum(float(item["quality"]["duration_seconds"]) for item in items)
        ),
    }


def evaluate_checkpoint(
    *,
    model_path: str | Path,
    artifact_path: str | Path,
    output_dir: str | Path,
    held_out_real_audio: Sequence[str | Path],
    delivery: DeliverySettings | Mapping[str, Any] | None = None,
    runtime_factory: RuntimeFactory = NamedSpeakerRuntime,
    transcriber: Transcriber = transcribe_local,
    speaker_embedder: SpeakerEmbedder | None = None,
    progress: ProgressCallback | None = None,
    verify_model: bool = True,
    voice_variant_id: str | None = None,
    _subject: Mapping[str, Any] | None = None,
    _report_kind: str = "voicetrainer.checkpoint_evaluation",
) -> dict[str, Any]:
    """Run and persist the full reference-free twenty-sentence evaluation.

    ``held_out_real_audio`` must contain at least two approved test-split takes.
    Their embeddings form a scoring centroid; their waveform is never passed to
    generation.  Dependency injection arguments exist for deterministic unit
    tests and do not weaken the production defaults.
    """

    destination = Path(output_dir).expanduser().resolve()
    report_path = destination / "evaluation.json"
    if report_path.exists():
        raise EvaluationError(f"immutable evaluation already exists: {report_path}")
    destination.mkdir(parents=True, exist_ok=True, mode=0o700)
    destination.chmod(0o700)
    audio_dir = destination / "audio"
    audio_dir.mkdir(exist_ok=False, mode=0o700)

    references = tuple(Path(path).expanduser().resolve() for path in held_out_real_audio)
    if len(references) < 2:
        raise EvaluationError("speaker evaluation requires at least two held-out real takes")
    for reference in references:
        if not reference.is_file():
            raise EvaluationError(f"held-out recording is missing: {reference}")

    settings = DeliverySettings.from_value(delivery)
    runtime = runtime_factory(model_path, artifact_path, verify_model=verify_model)
    try:
        runtime.load()
        embed = speaker_embedder or QwenSpeakerEmbedder(runtime)
        reference_vectors = [_unit_vector(embed(path)) for path in references]
        dimensions = {vector.shape for vector in reference_vectors}
        if len(dimensions) != 1:
            raise EvaluationError("held-out speaker embeddings have inconsistent dimensions")
        reference_centroid = _unit_vector(np.mean(reference_vectors, axis=0))

        results: list[dict[str, Any]] = []
        for index, prompt in enumerate(LOCKED_TEST_PROMPTS, start=1):
            if progress is not None:
                progress(
                    {
                        "event": "evaluation_prompt_started",
                        "completed": index - 1,
                        "total": len(LOCKED_TEST_PROMPTS),
                        "prompt_id": prompt.id,
                    }
                )
            # Candidate and Base runs use identical per-prompt MLX RNG seeds so
            # sampling noise does not give either side a systematic advantage.
            import mlx.core as mx

            mx.random.seed(EVALUATION_SEED + index - 1)
            output_path = audio_dir / f"{prompt.id}.wav"
            generation = dict(runtime.generate(prompt.text, output_path, settings))
            if generation.get("reference_audio_required") is not False:
                raise EvaluationError(
                    f"runtime did not attest reference-free generation for {prompt.id}"
                )
            transcription = dict(transcriber(output_path, prompt.text))
            observed = str(transcription.get("text", "")).strip()
            transcription["text"] = observed
            transcription["wer"] = word_error_rate(prompt.text, observed)
            quality = analyze_generated_wav(
                output_path,
                expected_text=prompt.text,
                asr_text=observed,
            )
            candidate_vector = _unit_vector(embed(output_path))
            if candidate_vector.shape != reference_centroid.shape:
                raise EvaluationError("generated and real speaker embeddings differ in shape")
            similarity = float(np.dot(candidate_vector, reference_centroid))
            output_path.chmod(0o400)
            results.append(
                {
                    "prompt_id": prompt.id,
                    "text": prompt.text,
                    "audio_path": str(output_path),
                    "audio_sha256": _sha256(output_path),
                    "generation": generation,
                    "asr": transcription,
                    "speaker_similarity": similarity,
                    "quality": quality,
                }
            )
            if progress is not None:
                progress(
                    {
                        "event": "evaluation_prompt_complete",
                        "completed": index,
                        "total": len(LOCKED_TEST_PROMPTS),
                        "prompt_id": prompt.id,
                    }
                )

        if _subject is None:
            artifact_manifest = Path(artifact_path).expanduser().resolve()
            if artifact_manifest.is_dir():
                artifact_manifest = artifact_manifest / "artifact.json"
            subject: dict[str, Any] = {
                "type": "named_speaker_checkpoint",
                "artifact_path": str(artifact_manifest.parent),
                "artifact_manifest_sha256": _sha256(artifact_manifest),
            }
        else:
            subject = dict(_subject)
        report: dict[str, Any] = {
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "kind": _report_kind,
            "created_at": _now(),
            "voice_variant_id": voice_variant_id,
            "subject": subject,
            "protocol": {
                "language": "english",
                "prompt_count": len(LOCKED_TEST_PROMPTS),
                "locked_prompt_set_sha256": LOCKED_PROMPT_SET_SHA256,
                "generation_seed": EVALUATION_SEED,
                "intelligible_sentence_wer_limit": INTELLIGIBLE_WER_LIMIT,
                "reference_audio_accepted_by_generation": False,
                "held_out_audio_used_for_generation": False,
                "speaker_similarity_backend": "qwen3_tts_ecapa_tdnn",
            },
            "delivery": settings.to_dict(),
            "held_out_real_audio": [
                {"path": str(path), "sha256": _sha256(path)} for path in references
            ],
            "items": results,
            "aggregate": _aggregate(results),
        }
        _atomic_json(report_path, report)
        if progress is not None:
            progress(
                {
                    "event": "evaluation_complete",
                    "completed": len(LOCKED_TEST_PROMPTS),
                    "total": len(LOCKED_TEST_PROMPTS),
                    "report_path": str(report_path),
                }
            )
        return report
    except Exception as exc:
        failure = {
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "kind": "voicetrainer.checkpoint_evaluation_failure",
            "created_at": _now(),
            "locked_prompt_set_sha256": LOCKED_PROMPT_SET_SHA256,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        _atomic_json(destination / "evaluation.failed.json", failure)
        raise
    finally:
        runtime.unload()


def evaluate_unadapted_baseline(
    *,
    model_path: str | Path,
    output_dir: str | Path,
    held_out_real_audio: Sequence[str | Path],
    delivery: DeliverySettings | Mapping[str, Any] | None = None,
    runtime_factory: RuntimeFactory = UnadaptedBaseRuntime,
    transcriber: Transcriber = transcribe_local,
    speaker_embedder: SpeakerEmbedder | None = None,
    progress: ProgressCallback | None = None,
    verify_model: bool = True,
    voice_variant_id: str | None = None,
) -> dict[str, Any]:
    """Create the one-time unadapted Base WER and similarity benchmark.

    This uses the same locked prompts, local ASR, held-out speaker centroid, and
    quality checks as checkpoint evaluation.  With ``verify_model=True`` the
    adopted model manifest and every recorded model hash are verified before
    generation.  The resulting ``evaluation.json`` is accepted directly as the
    ``baseline_report`` input to :func:`promotion_decision`.
    """

    subject = _base_model_subject(model_path, full_hash=verify_model)

    def factory(
        selected_model_path: str | Path,
        _ignored_artifact_path: str | Path,
        *,
        verify_model: bool,
    ) -> Any:
        # Integrity was checked above while constructing the immutable subject;
        # do not hash the multi-gigabyte model a second time.
        return runtime_factory(
            selected_model_path,
            None,
            verify_model=False,
        )

    return evaluate_checkpoint(
        model_path=model_path,
        artifact_path=model_path,
        output_dir=output_dir,
        held_out_real_audio=held_out_real_audio,
        delivery=delivery,
        runtime_factory=factory,
        transcriber=transcriber,
        speaker_embedder=speaker_embedder,
        progress=progress,
        verify_model=False,
        voice_variant_id=voice_variant_id,
        _subject=subject,
        _report_kind="voicetrainer.base_model_evaluation",
    )


def _report_aggregate(report: Mapping[str, Any], label: str) -> Mapping[str, Any]:
    protocol = report.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError(f"{label} lacks an evaluation protocol")
    if protocol.get("locked_prompt_set_sha256") != LOCKED_PROMPT_SET_SHA256:
        raise ValueError(f"{label} used a different locked prompt set")
    aggregate = report.get("aggregate")
    if not isinstance(aggregate, Mapping):
        raise ValueError(f"{label} lacks aggregate evaluation metrics")
    for key in ("wer", "speaker_similarity"):
        value = aggregate.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"{label} has no finite {key} metric")
    return aggregate


def promotion_decision(
    *,
    candidate_report: Mapping[str, Any],
    baseline_report: Mapping[str, Any],
    previous_report: Mapping[str, Any],
    blind_votes: Mapping[str, str] | None,
) -> dict[str, Any]:
    """Apply promotion gates, remaining pending until all twenty blind votes.

    Votes are keyed by locked prompt id and use ``candidate``, ``previous``, or
    ``tie``.  The previous report can be the unadapted baseline for the first
    candidate; subsequent candidates should compare with the promoted version.
    """

    candidate = _report_aggregate(candidate_report, "candidate report")
    baseline = _report_aggregate(baseline_report, "baseline report")
    _report_aggregate(previous_report, "previous report")
    votes = dict(blind_votes or {})
    valid_ids = {prompt.id for prompt in LOCKED_TEST_PROMPTS}
    unknown_ids = sorted(set(votes) - valid_ids)
    if unknown_ids:
        raise ValueError("blind votes contain unknown prompt ids: " + ", ".join(unknown_ids))
    invalid_votes = sorted(
        f"{prompt_id}={vote}"
        for prompt_id, vote in votes.items()
        if vote not in {"candidate", "previous", "tie"}
    )
    if invalid_votes:
        raise ValueError("invalid blind vote values: " + ", ".join(invalid_votes))

    reference_free = candidate.get("reference_free") is True
    all_intelligible = candidate.get("all_intelligible") is True
    candidate_wer = float(candidate["wer"])
    base_wer = float(baseline["wer"])
    candidate_similarity = float(candidate["speaker_similarity"])
    base_similarity = float(baseline["speaker_similarity"])
    wins = sum(vote == "candidate" for vote in votes.values())
    vote_complete = set(votes) == valid_ids
    gates = {
        "all_outputs_intelligible": all_intelligible,
        "reference_free_generation": reference_free,
        "wer_within_one_percentage_point_of_base": candidate_wer <= base_wer + 0.01,
        "speaker_similarity_improves_over_base": candidate_similarity > base_similarity,
        "blind_preference_at_least_12_of_20": vote_complete and wins >= 12,
    }
    objective_pass = all(
        value
        for key, value in gates.items()
        if key != "blind_preference_at_least_12_of_20"
    )
    if not vote_complete:
        status = "pending"
        action = "collect_blind_votes"
    elif objective_pass and gates["blind_preference_at_least_12_of_20"]:
        status = "promoted"
        action = "promote_candidate"
    else:
        status = "rejected"
        action = "retain_previous"
    return {
        "status": status,
        "action": action,
        "gates": gates,
        "metrics": {
            "candidate_wer": candidate_wer,
            "base_wer": base_wer,
            "maximum_allowed_wer": base_wer + 0.01,
            "candidate_speaker_similarity": candidate_similarity,
            "base_speaker_similarity": base_similarity,
            "candidate_blind_wins": wins,
            "votes_received": len(votes),
            "votes_required": len(LOCKED_TEST_PROMPTS),
        },
        "missing_vote_prompt_ids": sorted(valid_ids - set(votes)),
    }


def create_blind_comparison(
    *,
    candidate_report: Mapping[str, Any],
    previous_report: Mapping[str, Any],
    output_dir: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create locally copied, opaque A/B files plus a private answer key.

    The returned public manifest is safe for a listening UI.  The private key
    must not be shown until voting is finished; it converts A/B votes into the
    labels accepted by :func:`promotion_decision`.
    """

    _report_aggregate(candidate_report, "candidate report")
    _report_aggregate(previous_report, "previous report")
    candidate_items = {
        str(item["prompt_id"]): item for item in candidate_report.get("items", [])
    }
    previous_items = {
        str(item["prompt_id"]): item for item in previous_report.get("items", [])
    }
    expected_ids = {prompt.id for prompt in LOCKED_TEST_PROMPTS}
    if set(candidate_items) != expected_ids or set(previous_items) != expected_ids:
        raise ValueError("both reports must contain exactly the twenty locked prompt items")

    destination = Path(output_dir).expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise EvaluationError(f"blind comparison directory is not empty: {destination}")
    pairs_dir = destination / "pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    public_pairs: list[dict[str, Any]] = []
    private_answers: dict[str, str] = {}
    for index, prompt in enumerate(LOCKED_TEST_PROMPTS, start=1):
        pair_id = f"pair-{index:02d}"
        candidate_is_a = bool(secrets.randbits(1))
        sources = (
            (candidate_items[prompt.id], previous_items[prompt.id])
            if candidate_is_a
            else (previous_items[prompt.id], candidate_items[prompt.id])
        )
        option_paths: list[str] = []
        for label, source in zip(("A", "B"), sources, strict=True):
            source_path = Path(str(source["audio_path"])).expanduser().resolve()
            if not source_path.is_file():
                raise EvaluationError(f"comparison audio is missing: {source_path}")
            target = pairs_dir / f"{pair_id}-{label}.wav"
            shutil.copyfile(source_path, target)
            target.chmod(0o400)
            option_paths.append(str(target))
        public_pairs.append(
            {
                "pair_id": pair_id,
                "prompt_id": prompt.id,
                "text": prompt.text,
                "option_a_audio": option_paths[0],
                "option_b_audio": option_paths[1],
            }
        )
        private_answers[prompt.id] = "A" if candidate_is_a else "B"

    public = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "kind": "voicetrainer.blind_comparison",
        "created_at": _now(),
        "locked_prompt_set_sha256": LOCKED_PROMPT_SET_SHA256,
        "pairs": public_pairs,
    }
    private = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "kind": "voicetrainer.blind_comparison_key",
        "created_at": _now(),
        "locked_prompt_set_sha256": LOCKED_PROMPT_SET_SHA256,
        "candidate_option_by_prompt_id": private_answers,
    }
    _atomic_json(destination / "comparison.json", public)
    _atomic_json(destination / "comparison-key.private.json", private)
    return public, private


def resolve_blind_votes(
    *,
    private_key: Mapping[str, Any],
    option_votes: Mapping[str, str],
) -> dict[str, str]:
    """Convert blinded A/B/tie choices into candidate/previous/tie labels."""

    if private_key.get("locked_prompt_set_sha256") != LOCKED_PROMPT_SET_SHA256:
        raise ValueError("blind comparison key used a different locked prompt set")
    answers = private_key.get("candidate_option_by_prompt_id")
    if not isinstance(answers, Mapping):
        raise ValueError("blind comparison key has no candidate mapping")
    resolved: dict[str, str] = {}
    for prompt_id, vote in option_votes.items():
        if prompt_id not in answers:
            raise ValueError(f"blind vote has no answer-key entry: {prompt_id}")
        normalized = vote.upper()
        if normalized not in {"A", "B", "TIE"}:
            raise ValueError(f"blind vote must be A, B, or tie: {prompt_id}")
        if normalized == "TIE":
            resolved[prompt_id] = "tie"
        else:
            resolved[prompt_id] = (
                "candidate" if normalized == str(answers[prompt_id]).upper() else "previous"
            )
    return resolved


__all__ = [
    "EVALUATION_SCHEMA_VERSION",
    "EVALUATION_SEED",
    "EvaluationError",
    "INTELLIGIBLE_WER_LIMIT",
    "LOCKED_PROMPT_SET_SHA256",
    "LOCKED_TEST_PROMPTS",
    "LockedTestPrompt",
    "QwenSpeakerEmbedder",
    "UnadaptedBaseRuntime",
    "analyze_generated_wav",
    "create_blind_comparison",
    "evaluate_checkpoint",
    "evaluate_unadapted_baseline",
    "promotion_decision",
    "resolve_blind_votes",
]
