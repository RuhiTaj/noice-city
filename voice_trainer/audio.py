"""Local-only WAV validation, conversion, and recording quality checks.

The recorder deliberately accepts one wire format: 48 kHz, mono, signed 24-bit
PCM WAV.  Keeping that contract narrow makes the immutable master auditable and
lets us create the 24 kHz training copy without invoking ffmpeg or a networked
service.
"""

from __future__ import annotations

import math
import os
import struct
import tempfile
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO, Sequence

import numpy as np


MASTER_SAMPLE_RATE = 48_000
TRAINING_SAMPLE_RATE = 24_000
MASTER_SAMPLE_WIDTH_BYTES = 3
TRAINING_SAMPLE_WIDTH_BYTES = 2
MASTER_FULL_SCALE = (1 << 23) - 1


class AudioValidationError(ValueError):
    """Raised when a recording is not the lossless format the app promises."""


@dataclass(frozen=True)
class QualityThresholds:
    minimum_seconds: float = 2.0
    preferred_minimum_seconds: float = 3.0
    preferred_maximum_seconds: float = 10.0
    maximum_seconds: float = 15.0
    clipping_dbfs: float = -1.0
    minimum_snr_db: float = 25.0
    silence_dbfs: float = -50.0
    maximum_silence_ratio: float = 0.40
    preferred_maximum_silence_ratio: float = 0.20
    minimum_level_dbfs: float = -40.0
    preferred_minimum_level_dbfs: float = -30.0
    preferred_maximum_level_dbfs: float = -12.0
    maximum_level_dbfs: float = -6.0
    minimum_speech_ratio: float = 0.12
    minimum_speech_likeness_ratio: float = 0.18


@dataclass(frozen=True)
class SpeechActivityMetrics:
    """Deterministic acoustic evidence used by the model-free speech gate.

    ``active_frame_count`` is the ordinary energy VAD result. A frame is only
    counted as speech when it also has speech-like bandwidth, zero-crossing,
    and periodicity. This deliberately rejects steady test tones and broadband
    room noise without claiming to identify a speaker or recognize words.
    """

    algorithm: str
    frame_count: int
    active_frame_count: int
    speech_frame_count: int
    activity_ratio: float
    speech_ratio: float
    speech_likeness_ratio: float


@dataclass(frozen=True)
class AudioMetrics:
    sample_rate: int
    channels: int
    sample_width_bits: int
    frame_count: int
    duration_seconds: float
    peak_dbfs: float
    rms_dbfs: float
    clipping_ratio: float
    silence_ratio: float
    snr_db: float | None
    speech_activity: SpeechActivityMetrics

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class AudioQualityReport:
    metrics: AudioMetrics
    passed: bool
    needs_review: bool
    failures: tuple[str, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "metrics": self.metrics.to_dict(),
            "passed": self.passed,
            "needs_review": self.needs_review,
            "failures": list(self.failures),
            "warnings": list(self.warnings),
        }


def _dbfs(amplitude: float, full_scale: float = MASTER_FULL_SCALE) -> float:
    if amplitude <= 0:
        return -120.0
    return 20.0 * math.log10(min(amplitude, full_scale) / full_scale)


def _read_wav(source: str | os.PathLike[str] | BinaryIO) -> tuple[wave._wave_params, bytes]:
    wave_source: str | BinaryIO = os.fspath(source) if isinstance(source, os.PathLike) else source
    try:
        with wave.open(wave_source, "rb") as wav_file:
            params = wav_file.getparams()
            frames = wav_file.readframes(params.nframes)
    except (EOFError, wave.Error) as exc:
        raise AudioValidationError(f"Invalid WAV file: {exc}") from exc
    if params.comptype != "NONE":
        raise AudioValidationError("The master must be uncompressed PCM WAV")
    return params, frames


def _validate_master_params(params: wave._wave_params) -> None:
    if params.framerate != MASTER_SAMPLE_RATE:
        raise AudioValidationError(
            f"The immutable master must be {MASTER_SAMPLE_RATE} Hz; got {params.framerate} Hz"
        )
    if params.nchannels != 1:
        raise AudioValidationError("The immutable master must be mono")
    if params.sampwidth != MASTER_SAMPLE_WIDTH_BYTES:
        raise AudioValidationError("The immutable master must use signed 24-bit PCM")


def _pcm24_samples(frames: bytes) -> tuple[int, ...]:
    """Decode packed little-endian signed PCM24 into Python integers."""

    if len(frames) % MASTER_SAMPLE_WIDTH_BYTES:
        raise AudioValidationError("The PCM payload is truncated")
    if not frames:
        return ()
    samples: list[int] = []
    for offset in range(0, len(frames), MASTER_SAMPLE_WIDTH_BYTES):
        value = frames[offset] | frames[offset + 1] << 8 | frames[offset + 2] << 16
        if value & 0x800000:
            value -= 1 << 24
        samples.append(value)
    return tuple(samples)


def _rms(samples: Sequence[int]) -> float:
    if not samples:
        return 0.0
    return math.sqrt(sum(sample * sample for sample in samples) / len(samples))


def _frame_rms(samples: Sequence[int], frame_size: int) -> list[float]:
    return [_rms(samples[offset : offset + frame_size]) for offset in range(0, len(samples), frame_size)]


def _estimate_snr_db(frame_levels: Sequence[float], silence_amplitude: float) -> float | None:
    """Estimate speech-to-room-floor SNR from quiet and active 20 ms frames.

    This is intentionally conservative. A clip without a measurable quiet frame
    cannot produce a defensible room-noise estimate and is sent to review instead
    of being assigned a made-up score.
    """

    quiet = [level for level in frame_levels if level <= silence_amplitude]
    active = [level for level in frame_levels if level > silence_amplitude]
    if not quiet or not active:
        return None
    signal_rms = math.sqrt(sum(level * level for level in active) / len(active))
    nonzero_quiet = [level for level in quiet if level > 0]
    noise_rms = (
        math.sqrt(sum(level * level for level in nonzero_quiet) / len(nonzero_quiet))
        if nonzero_quiet
        else 1.0
    )
    return round(20.0 * math.log10(max(signal_rms, 1.0) / max(noise_rms, 1.0)), 3)


def detect_speech_activity(
    samples: Sequence[int],
    sample_rate: int,
    silence_amplitude: float,
) -> SpeechActivityMetrics:
    """Find speech-like activity using only deterministic acoustic features.

    This is intentionally a compact, model-free VAD. Thirty-millisecond frames
    first pass an absolute energy threshold. Active frames must then contain
    energy in multiple voice bands, avoid single-bin tonal concentration, have
    a plausible zero-crossing rate, and show pitch-range periodicity. Those
    checks reject silence, stable sine waves, and unstructured broadband noise
    while accepting ordinary voiced speech. No transcript, model, or network is
    involved.
    """

    frame_size = max(1, round(sample_rate * 0.030))
    hop_size = max(1, round(sample_rate * 0.015))
    values = np.asarray(samples, dtype=np.float64)
    if values.size < frame_size:
        return SpeechActivityMetrics("acoustic-vad-v1", 0, 0, 0, 0.0, 0.0, 0.0)

    window = np.hanning(frame_size)
    frequencies = np.fft.rfftfreq(frame_size, d=1.0 / sample_rate)
    voice_mask = (frequencies >= 70.0) & (frequencies <= 8_000.0)
    voice_frequencies = frequencies[voice_mask]
    bands = ((80.0, 300.0), (300.0, 800.0), (800.0, 2_000.0),
             (2_000.0, 4_000.0), (4_000.0, 8_000.0))
    pitch_lag_min = max(1, round(sample_rate / 400.0))
    pitch_lag_max = min(frame_size - 1, round(sample_rate / 60.0))

    frame_count = 0
    active_count = 0
    speech_count = 0
    for offset in range(0, values.size - frame_size + 1, hop_size):
        frame_count += 1
        frame = values[offset : offset + frame_size]
        frame_rms = float(np.sqrt(np.mean(frame * frame)))
        if frame_rms <= silence_amplitude:
            continue
        active_count += 1

        centered = frame - float(np.mean(frame))
        spectrum = np.fft.rfft(centered * window)
        power = np.square(np.abs(spectrum))[voice_mask]
        total_power = float(np.sum(power))
        if not math.isfinite(total_power) or total_power <= 0.0:
            continue
        top_count = min(5, power.size)
        top_concentration = float(np.sum(np.partition(power, -top_count)[-top_count:])) / total_power
        spectral_flatness = float(
            np.exp(np.mean(np.log(power + 1e-30))) / np.mean(power + 1e-30)
        )
        occupied_bands = sum(
            float(np.sum(power[(voice_frequencies >= lower) & (voice_frequencies < upper)]))
            / total_power
            >= 0.015
            for lower, upper in bands
        )
        zero_crossing_rate = float(
            np.mean(np.signbit(centered[1:]) != np.signbit(centered[:-1]))
        )

        # FFT autocorrelation keeps this bounded and deterministic. Normalized
        # pitch-range correlation separates voiced frames from random noise;
        # the spectral concentration limit separately excludes pure tones.
        autocorrelation_spectrum = np.fft.rfft(centered, n=frame_size * 2)
        autocorrelation = np.fft.irfft(
            np.square(np.abs(autocorrelation_spectrum)), n=frame_size * 2
        )[:frame_size]
        zero_lag = float(autocorrelation[0])
        periodicity = (
            float(np.max(autocorrelation[pitch_lag_min : pitch_lag_max + 1])) / zero_lag
            if zero_lag > 0.0 and pitch_lag_max >= pitch_lag_min
            else 0.0
        )

        if (
            occupied_bands >= 2
            and top_concentration < 0.93
            and spectral_flatness < 0.65
            and 0.004 <= zero_crossing_rate <= 0.40
            and periodicity >= 0.25
        ):
            speech_count += 1

    activity_ratio = active_count / frame_count if frame_count else 0.0
    speech_ratio = speech_count / frame_count if frame_count else 0.0
    speech_likeness_ratio = speech_count / active_count if active_count else 0.0
    return SpeechActivityMetrics(
        algorithm="acoustic-vad-v1",
        frame_count=frame_count,
        active_frame_count=active_count,
        speech_frame_count=speech_count,
        activity_ratio=round(activity_ratio, 6),
        speech_ratio=round(speech_ratio, 6),
        speech_likeness_ratio=round(speech_likeness_ratio, 6),
    )


def analyze_master_wav(
    source: str | os.PathLike[str] | BinaryIO,
    thresholds: QualityThresholds = QualityThresholds(),
) -> AudioQualityReport:
    """Validate and score a 48 kHz mono PCM master recording."""

    params, frames = _read_wav(source)
    _validate_master_params(params)
    samples = _pcm24_samples(frames)
    duration = params.nframes / params.framerate if params.framerate else 0.0
    peak = max((abs(sample) for sample in samples), default=0)
    rms = _rms(samples)
    clipping_amplitude = MASTER_FULL_SCALE * (10.0 ** (thresholds.clipping_dbfs / 20.0))
    clipping_count = sum(1 for sample in samples if abs(sample) >= clipping_amplitude)
    frame_levels = _frame_rms(samples, max(1, round(params.framerate * 0.020)))
    silence_amplitude = MASTER_FULL_SCALE * (10.0 ** (thresholds.silence_dbfs / 20.0))
    silent_frames = sum(1 for level in frame_levels if level <= silence_amplitude)
    silence_ratio = silent_frames / len(frame_levels) if frame_levels else 1.0
    snr = _estimate_snr_db(frame_levels, silence_amplitude)
    speech_activity = detect_speech_activity(samples, params.framerate, silence_amplitude)

    metrics = AudioMetrics(
        sample_rate=params.framerate,
        channels=params.nchannels,
        sample_width_bits=params.sampwidth * 8,
        frame_count=params.nframes,
        duration_seconds=round(duration, 6),
        peak_dbfs=round(_dbfs(peak), 3),
        rms_dbfs=round(_dbfs(rms), 3),
        clipping_ratio=round(clipping_count / len(samples), 8) if samples else 0.0,
        silence_ratio=round(silence_ratio, 6),
        snr_db=snr,
        speech_activity=speech_activity,
    )

    failures: list[str] = []
    warnings: list[str] = []
    if duration < thresholds.minimum_seconds:
        failures.append("duration_too_short")
    elif duration < thresholds.preferred_minimum_seconds:
        warnings.append("duration_short")
    if duration > thresholds.maximum_seconds:
        failures.append("duration_too_long")
    elif duration > thresholds.preferred_maximum_seconds:
        warnings.append("duration_long")
    if metrics.peak_dbfs > thresholds.clipping_dbfs:
        failures.append("peak_too_hot")
    if metrics.rms_dbfs < thresholds.minimum_level_dbfs:
        failures.append("level_too_quiet")
    elif metrics.rms_dbfs < thresholds.preferred_minimum_level_dbfs:
        warnings.append("level_quiet")
    if metrics.rms_dbfs > thresholds.maximum_level_dbfs:
        failures.append("level_too_loud")
    elif metrics.rms_dbfs > thresholds.preferred_maximum_level_dbfs:
        warnings.append("level_loud")
    if metrics.silence_ratio > thresholds.maximum_silence_ratio:
        failures.append("too_much_silence")
    elif metrics.silence_ratio > thresholds.preferred_maximum_silence_ratio:
        warnings.append("extra_silence")
    if metrics.snr_db is None:
        warnings.append("snr_requires_review")
    elif metrics.snr_db < thresholds.minimum_snr_db:
        failures.append("snr_too_low")
    if (
        speech_activity.speech_ratio < thresholds.minimum_speech_ratio
        or speech_activity.speech_likeness_ratio < thresholds.minimum_speech_likeness_ratio
    ):
        failures.append("speech_not_detected")

    return AudioQualityReport(
        metrics=metrics,
        passed=not failures,
        needs_review=bool(failures or warnings),
        failures=tuple(failures),
        warnings=tuple(warnings),
    )


def _atomic_write(path: Path, payload: bytes, mode: int = 0o400) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.exists():
        raise FileExistsError(f"Refusing to replace immutable audio: {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_master_wav(payload: bytes, destination: str | os.PathLike[str]) -> AudioQualityReport:
    """Validate then atomically preserve a never-overwritten 48 kHz master."""

    from io import BytesIO

    report = analyze_master_wav(BytesIO(payload))
    _atomic_write(Path(destination), payload)
    return report


def derive_training_wav(
    master: str | os.PathLike[str], destination: str | os.PathLike[str]
) -> None:
    """Create an immutable 24 kHz mono PCM derivative from a validated master.

    The master rate is exactly twice the target rate. A compact 7-tap symmetric
    low-pass FIR removes most energy above the new Nyquist limit before decimation.
    It is deterministic across machines and avoids native or network-fetched tools.
    """

    params, frames = _read_wav(master)
    _validate_master_params(params)
    samples = _pcm24_samples(frames)
    coefficients = (-1, 0, 9, 16, 9, 0, -1)
    coefficient_sum = sum(coefficients)
    filtered: list[int] = []
    for center in range(0, len(samples), 2):
        value = 0
        for tap, coefficient in enumerate(coefficients):
            index = min(max(center + tap - 3, 0), max(len(samples) - 1, 0))
            if samples:
                value += coefficient * samples[index]
        sample_24 = round(value / coefficient_sum) if samples else 0
        # PCM24 to PCM16 with rounding instead of truncation toward zero.
        sample_16 = round(sample_24 / 256.0)
        filtered.append(max(-32768, min(32767, sample_16)))
    pcm = struct.pack(f"<{len(filtered)}h", *filtered) if filtered else b""

    from io import BytesIO

    buffer = BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(TRAINING_SAMPLE_WIDTH_BYTES)
        output.setframerate(TRAINING_SAMPLE_RATE)
        output.writeframes(pcm)
    _atomic_write(Path(destination), buffer.getvalue())


def inspect_wav(source: str | os.PathLike[str] | BinaryIO) -> dict[str, int | float]:
    """Return basic WAV properties without applying the master-only QA contract."""

    params, _ = _read_wav(source)
    return {
        "sample_rate": params.framerate,
        "channels": params.nchannels,
        "sample_width_bits": params.sampwidth * 8,
        "frame_count": params.nframes,
        "duration_seconds": params.nframes / params.framerate,
    }
