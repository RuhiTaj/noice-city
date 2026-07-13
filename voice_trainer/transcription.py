from __future__ import annotations

import re
import threading
from pathlib import Path

from .bootstrap import verify_asr_model
from .config import ASR_ROOT


_MODEL = None
_LOCK = threading.Lock()


def normalize_transcript(text: str) -> str:
    text = text.casefold().replace("’", "'")
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
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


def transcribe_local(audio_path: Path, expected: str = "") -> dict:
    """Transcribe one local WAV without any repository or network lookup."""
    with _LOCK:
        result = _load_model().generate(
            str(audio_path),
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
