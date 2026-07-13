from __future__ import annotations

from types import MethodType, SimpleNamespace

import mlx.core as mx
import pytest
from pydantic import ValidationError

from voice_trainer.model_runtime import (
    MAX_GENERATION_SEED,
    NamedSpeakerRuntime,
)
from voice_trainer.schemas import GenerationRequest


def _request(**overrides):
    values = {
        "text": "A deterministic local sentence.",
        "delivery": "natural",
        "checkpoint_version": "checkpoint-test",
        "variant_id": "variant_ruhi_default",
    }
    values.update(overrides)
    return GenerationRequest.model_validate(values)


def test_generation_request_accepts_only_bounded_integer_seeds() -> None:
    assert _request().seed is None
    assert _request(seed=0).seed == 0
    assert _request(seed=MAX_GENERATION_SEED).seed == MAX_GENERATION_SEED

    for invalid in (-1, MAX_GENERATION_SEED + 1, True, 1.5, "3407"):
        with pytest.raises(ValidationError):
            _request(seed=invalid)


@pytest.mark.parametrize("seed", [None, 0, 3407, MAX_GENERATION_SEED])
def test_runtime_seeds_after_load_immediately_before_sampling(
    tmp_path, monkeypatch, seed
) -> None:
    events: list[tuple[str, object]] = []

    class FakeModel:
        def generate(self, **kwargs):
            events.append(("generate", kwargs["text"]))
            return iter(
                [
                    SimpleNamespace(
                        sample_rate=24_000,
                        audio=mx.zeros((2_400,), dtype=mx.float32),
                    )
                ]
            )

    runtime = object.__new__(NamedSpeakerRuntime)
    runtime._model = None
    runtime.artifact = SimpleNamespace(
        speaker_name="ruhi",
        profile_name="ruhi",
        variant_id="variant_ruhi_default",
        path=tmp_path / "checkpoint",
    )

    def load(self):
        events.append(("load", None))
        self._model = FakeModel()
        return self

    runtime.load = MethodType(load, runtime)
    monkeypatch.setattr(
        mx.random,
        "seed",
        lambda value: events.append(("seed", value)),
    )

    result = runtime.generate(
        "A deterministic local sentence.",
        tmp_path / f"seed-{seed}.wav",
        seed=seed,
    )

    expected = [("load", None)]
    if seed is not None:
        expected.append(("seed", seed))
    expected.append(("generate", "A deterministic local sentence."))
    assert events == expected
    assert result.get("seed") == seed
    assert ("seed" in result) is (seed is not None)
    assert result["reference_audio_required"] is False


@pytest.mark.parametrize("seed", [-1, MAX_GENERATION_SEED + 1, True, 3.5, "3"])
def test_runtime_rejects_invalid_seeds_before_loading(tmp_path, seed) -> None:
    runtime = object.__new__(NamedSpeakerRuntime)
    runtime._model = None
    runtime.artifact = SimpleNamespace(speaker_name="ruhi", path=tmp_path)

    def should_not_load(self):
        raise AssertionError("invalid seeds must fail before model loading")

    runtime.load = MethodType(should_not_load, runtime)
    with pytest.raises(ValueError, match="generation seed"):
        runtime.generate("Text.", tmp_path / "invalid.wav", seed=seed)
