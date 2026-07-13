import wave

import numpy as np

from voice_trainer.transcription import (
    _decode_whisper_audio,
    normalize_transcript,
    word_error_rate,
)


def test_normalize_transcript() -> None:
    assert normalize_transcript("Hello,  WORLD!") == "hello world"


def test_word_error_rate() -> None:
    assert word_error_rate("one two three", "one two three") == 0.0
    assert word_error_rate("one two three", "one four three") == 1 / 3


def test_whisper_audio_is_locally_decoded_to_16khz_mono(tmp_path) -> None:
    path = tmp_path / "master.wav"
    frames = (np.sin(np.arange(48_000) * 0.01) * 10_000).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(48_000)
        wav.writeframes(frames.tobytes())

    decoded = _decode_whisper_audio(path)

    assert decoded.dtype == np.float32
    assert 15_990 <= decoded.size <= 16_010
    assert np.isfinite(decoded).all()
