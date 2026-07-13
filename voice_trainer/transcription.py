from __future__ import annotations

import re
import threading
from pathlib import Path

import numpy as np

from .bootstrap import verify_asr_model
from .config import ASR_ROOT


_MODEL = None
_LOCK = threading.Lock()


def normalize_transcript(text: str) -> str:
    text = text.casefold().replace("’", "'")
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    # Keep apostrophes only when they are inside a word (for example, "don't").
    # Quotation marks and possessive trailing apostrophes are punctuation, not
    # ASR tokens; counting them as words can inflate WER and falsely report that
    # otherwise complete speech was cut off.
    text = re.sub(r"(?<![a-z0-9])'+", "", text)
    text = re.sub(r"'+(?![a-z0-9])", "", text)
    return " ".join(text.split())


def word_error_rate(expected: str, observed: str) -> float:
    reference = normalize_transcript(expected).split()
    hypothesis = normalize_transcript(observed).split()
    if not reference:
        return 0.0 if not hypothesis else 1.0
    previous = list(range(len(hypothesis) + 1))
    for row, ref_word in enumerate(reference, start=1):
        current = [row]
        for column, hyp_word in enumerate(hypothesis, start=1):
            substitution = previous[column - 1] + (ref_word != hyp_word)
            insertion = current[column - 1] + 1
            deletion = previous[column] + 1
            current.append(min(substitution, insertion, deletion))
        previous = current
    return previous[-1] / len(reference)


def _load_model():
    global _MODEL
    if _MODEL is None:
        verify_asr_model(full_hash=False)
        from mlx_audio.stt.utils import load

        _MODEL = load(str(ASR_ROOT))
    return _MODEL


def _decode_whisper_audio(audio_path: Path) -> np.ndarray:
    """Decode and resample locally to Whisper's 16 kHz mono input."""

    try:
        import miniaudio

        decoded = miniaudio.decode_file(
            str(audio_path),
            output_format=miniaudio.SampleFormat.SIGNED16,
            nchannels=1,
            sample_rate=16_000,
        )
    except Exception as exc:
        raise ValueError(f"could not decode local audio {audio_path}: {exc}") from exc
    samples = np.asarray(decoded.samples, dtype=np.int16).astype(np.float32) / 32768.0
    if samples.size == 0 or not np.isfinite(samples).all():
        raise ValueError(f"decoded local audio is empty or invalid: {audio_path}")
    return samples


def transcribe_local(audio_path: Path, expected: str = "") -> dict:
    """Transcribe one local WAV without any repository or network lookup."""
    audio = _decode_whisper_audio(audio_path)
    with _LOCK:
        result = _load_model().generate(
            audio,
            language="en",
            temperature=0.0,
            condition_on_previous_text=False,
            return_timestamps=True,
            word_timestamps=False,
            verbose=None,
        )
    # MLX-Audio 0.4.5 returns an STTOutput dataclass.  Keeping the small mapping
    # fallback makes this tolerant of the equivalent serialized form in tests.
    if isinstance(result, dict):
        text_value = result.get("text", "")
        language = result.get("language", "en")
    else:
        text_value = getattr(result, "text", "")
        language = getattr(result, "language", "en")
    text = str(text_value).strip()
    return {
        "text": text,
        "normalized": normalize_transcript(text),
        "expected_normalized": normalize_transcript(expected),
        "wer": word_error_rate(expected, text) if expected else None,
        "language": language or "en",
    }
