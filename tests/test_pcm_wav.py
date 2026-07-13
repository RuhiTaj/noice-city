from __future__ import annotations

import wave

import numpy as np
import pytest

from voice_trainer.pcm_wav import PcmWavError, read_mono_pcm_wav


def _write_pcm16(path, *, sample_rate: int = 24_000, channels: int = 1) -> None:
    samples = (np.sin(np.arange(sample_rate) * 0.01) * 10_000).astype("<i2")
    if channels == 2:
        samples = np.column_stack([samples, samples]).reshape(-1)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(samples.tobytes())


def test_reads_fixed_rate_mono_pcm_without_resampling(tmp_path):
    path = tmp_path / "take.wav"
    _write_pcm16(path)

    samples, sample_rate = read_mono_pcm_wav(path, expected_sample_rate=24_000)

    assert sample_rate == 24_000
    assert samples.shape == (24_000,)
    assert samples.dtype == np.float32
    assert np.max(np.abs(samples)) < 1.0


def test_rejects_rate_mismatch_instead_of_implicitly_resampling(tmp_path):
    path = tmp_path / "take.wav"
    _write_pcm16(path, sample_rate=48_000)

    with pytest.raises(PcmWavError, match="must be 24000 Hz"):
        read_mono_pcm_wav(path, expected_sample_rate=24_000)


def test_rejects_non_mono_audio(tmp_path):
    path = tmp_path / "take.wav"
    _write_pcm16(path, channels=2)

    with pytest.raises(PcmWavError, match="must be mono"):
        read_mono_pcm_wav(path)
