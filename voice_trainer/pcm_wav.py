"""Small, auditable PCM WAV reader for the fixed-rate Qwen audio paths."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np


class PcmWavError(ValueError):
    """Raised when a WAV cannot satisfy the local mono-PCM contract."""


def _pcm_to_float32(raw: bytes, sample_width: int) -> np.ndarray:
    if sample_width == 1:
        return (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    if sample_width == 2:
        return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if sample_width == 3:
        packed = np.frombuffer(raw, dtype=np.uint8)
        if packed.size % 3:
            raise PcmWavError("WAV contains an incomplete 24-bit sample")
        packed = packed.reshape(-1, 3).astype(np.int32)
        values = packed[:, 0] | (packed[:, 1] << 8) | (packed[:, 2] << 16)
        values = np.where(values & 0x800000, values - 0x1000000, values)
        return values.astype(np.float32) / 8388608.0
    if sample_width == 4:
        return np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    raise PcmWavError(f"unsupported PCM WAV sample width: {sample_width}")


def read_mono_pcm_wav(
    path: str | Path,
    *,
    expected_sample_rate: int | None = None,
) -> tuple[np.ndarray, int]:
    """Read an uncompressed mono PCM WAV without a resampling dependency."""

    source = Path(path)
    try:
        with wave.open(str(source), "rb") as wav:
            if wav.getcomptype() != "NONE":
                raise PcmWavError("WAV must contain uncompressed PCM audio")
            if wav.getnchannels() != 1:
                raise PcmWavError("WAV must be mono")
            sample_rate = wav.getframerate()
            samples = _pcm_to_float32(
                wav.readframes(wav.getnframes()),
                wav.getsampwidth(),
            )
    except (EOFError, wave.Error) as exc:
        raise PcmWavError(f"invalid WAV {source}: {exc}") from exc

    if expected_sample_rate is not None and sample_rate != expected_sample_rate:
        raise PcmWavError(
            f"WAV must be {expected_sample_rate} Hz; got {sample_rate} Hz"
        )
    if sample_rate <= 0 or samples.size == 0 or not np.isfinite(samples).all():
        raise PcmWavError(f"WAV is empty or invalid: {source}")
    return samples, sample_rate
