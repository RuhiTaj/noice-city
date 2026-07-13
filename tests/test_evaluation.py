from __future__ import annotations

import json
import hashlib
import math
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from voice_trainer.evaluation import (
    LOCKED_PROMPT_SET_SHA256,
    LOCKED_TEST_PROMPTS,
    UnadaptedBaseRuntime,
    analyze_generated_wav,
    create_blind_comparison,
    evaluate_checkpoint,
    evaluate_unadapted_baseline,
    promotion_decision,
    resolve_blind_votes,
)


def _write_tone(path: Path, duration: float = 3.0, sample_rate: int = 24000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    times = np.arange(round(duration * sample_rate), dtype=np.float32) / sample_rate
    samples = (0.18 * np.sin(2.0 * math.pi * 180.0 * times) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(samples.tobytes())


class _FakeRuntime:
    instances: list["_FakeRuntime"] = []

    def __init__(self, model_path, artifact_path, *, verify_model=True):
        self.model_path = model_path
        self.artifact_path = artifact_path
        self.verify_model = verify_model
        self.loaded = False
        self.unloaded = False
        self.generated: list[str] = []
        self.instances.append(self)

    def load(self):
        self.loaded = True
        return self

    def generate(self, text, output_path, delivery):
        assert self.loaded
        _write_tone(Path(output_path), duration=3.0)
        self.generated.append(text)
        return {
            "output_path": str(output_path),
            "speaker": "ruhi",
            "reference_audio_required": False,
            "delivery": delivery.to_dict(),
        }

    def unload(self):
        self.unloaded = True


def _report(*, wer=0.08, similarity=0.5, intelligible=True, reference_free=True, items=None):
    return {
        "protocol": {"locked_prompt_set_sha256": LOCKED_PROMPT_SET_SHA256},
        "aggregate": {
            "wer": wer,
            "speaker_similarity": similarity,
            "all_intelligible": intelligible,
            "reference_free": reference_free,
        },
        "items": items or [],
    }


def _make_adopted_model_stub(path: Path) -> None:
    (path / "speech_tokenizer").mkdir(parents=True)
    files = {
        "config.json": json.dumps(
            {
                "model_type": "qwen3_tts",
                "tts_model_type": "base",
                "tts_model_size": "1b7",
                "talker_config": {"vocab_size": 3072},
            }
        ).encode(),
        "model.safetensors": b"model-stub",
        "speech_tokenizer/model.safetensors": b"tokenizer-stub",
    }
    records = []
    for relative, content in files.items():
        target = path / relative
        target.write_bytes(content)
        records.append(
            {
                "path": relative,
                "size": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    (path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "repo": "local-test/qwen",
                "revision": "test-revision",
                "format": "mlx-bf16-safetensors",
                "files": records,
            }
        ),
        encoding="utf-8",
    )


def test_locked_protocol_has_twenty_unique_english_narration_prompts():
    assert len(LOCKED_TEST_PROMPTS) == 20
    assert len({prompt.id for prompt in LOCKED_TEST_PROMPTS}) == 20
    assert len({prompt.text for prompt in LOCKED_TEST_PROMPTS}) == 20
    assert len(LOCKED_PROMPT_SET_SHA256) == 64


def test_analyze_generated_wav_detects_repetition_and_silence(tmp_path):
    tone = tmp_path / "tone.wav"
    _write_tone(tone, duration=3.0)
    repeated = analyze_generated_wav(
        tone,
        expected_text="The final result is clear.",
        asr_text="the final final final final result is clear",
    )
    assert repeated["repetition_ok"] is False
    assert repeated["maximum_consecutive_word_run"] == 4
    assert repeated["silence_ok"] is True

    silence = tmp_path / "silence.wav"
    with wave.open(str(silence), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(np.zeros(24000, dtype="<i2").tobytes())
    silent_metrics = analyze_generated_wav(
        silence,
        expected_text="This should be audible.",
        asr_text="",
    )
    assert silent_metrics["silence_ok"] is False
    assert silent_metrics["intelligible"] is False


def test_analyze_generated_wav_ignores_quotation_marks_for_termination(tmp_path):
    tone = tmp_path / "quoted-speech.wav"
    _write_tone(tone, duration=3.0)

    metrics = analyze_generated_wav(
        tone,
        expected_text="Maya said, 'We should listen before we decide.'",
        asr_text="Maya said, We should listen before we decide.",
    )

    assert metrics["wer"] == pytest.approx(0.0)
    assert metrics["final_expected_word_reached"] is True
    assert metrics["termination_ok"] is True


def test_evaluate_checkpoint_generates_all_prompts_without_reference_audio(tmp_path):
    _FakeRuntime.instances.clear()
    artifact = tmp_path / "checkpoint"
    artifact.mkdir()
    (artifact / "artifact.json").write_text('{"test": true}\n', encoding="utf-8")
    references = [tmp_path / "real-1.wav", tmp_path / "real-2.wav"]
    for reference in references:
        _write_tone(reference)

    def transcriber(_audio_path: Path, expected: str):
        return {"text": expected, "language": "en"}

    def embedder(_audio_path: Path):
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)

    events = []
    output = tmp_path / "evaluation"
    report = evaluate_checkpoint(
        model_path=tmp_path / "model",
        artifact_path=artifact,
        output_dir=output,
        held_out_real_audio=references,
        runtime_factory=_FakeRuntime,
        transcriber=transcriber,
        speaker_embedder=embedder,
        progress=events.append,
        verify_model=False,
    )

    assert len(report["items"]) == 20
    assert report["aggregate"]["wer"] == pytest.approx(0.0)
    assert report["aggregate"]["speaker_similarity"] == pytest.approx(1.0)
    assert report["aggregate"]["all_intelligible"] is True
    assert report["aggregate"]["reference_free"] is True
    assert report["protocol"]["held_out_audio_used_for_generation"] is False
    assert len(list((output / "audio").glob("*.wav"))) == 20
    persisted = json.loads((output / "evaluation.json").read_text(encoding="utf-8"))
    assert persisted["aggregate"] == report["aggregate"]
    assert events[-1]["event"] == "evaluation_complete"
    runtime = _FakeRuntime.instances[-1]
    assert runtime.unloaded is True
    assert runtime.generated == [prompt.text for prompt in LOCKED_TEST_PROMPTS]


def test_unadapted_baseline_uses_same_protocol_and_is_directly_comparable(tmp_path):
    model = tmp_path / "model"
    _make_adopted_model_stub(model)
    references = [tmp_path / "real-a.wav", tmp_path / "real-b.wav"]
    for reference in references:
        _write_tone(reference)

    report = evaluate_unadapted_baseline(
        model_path=model,
        output_dir=tmp_path / "baseline",
        held_out_real_audio=references,
        runtime_factory=_FakeRuntime,
        transcriber=lambda _path, expected: {"text": expected},
        speaker_embedder=lambda _path: np.array([1.0, 0.0], dtype=np.float32),
        verify_model=True,
    )
    assert report["kind"] == "voicetrainer.base_model_evaluation"
    assert report["subject"]["type"] == "unadapted_base_model"
    assert report["subject"]["integrity"] == "full_hash"
    assert report["protocol"]["locked_prompt_set_sha256"] == LOCKED_PROMPT_SET_SHA256
    assert report["aggregate"]["reference_free"] is True


def test_unadapted_base_runtime_explicitly_passes_no_voice_or_reference(tmp_path):
    import mlx.core as mx

    model_path = tmp_path / "model"
    _make_adopted_model_stub(model_path)
    runtime = UnadaptedBaseRuntime(model_path, verify_model=True)

    class FakeBaseModel:
        def __init__(self):
            self.kwargs = None

        def generate(self, **kwargs):
            self.kwargs = kwargs
            return iter(
                [
                    SimpleNamespace(
                        sample_rate=24000,
                        audio=mx.array(np.full(24000, 0.1, dtype=np.float32)),
                    )
                ]
            )

    fake_model = FakeBaseModel()
    runtime._model = fake_model
    metadata = runtime.generate("A short baseline sentence.", tmp_path / "base.wav")
    assert fake_model.kwargs["voice"] is None
    assert fake_model.kwargs["ref_audio"] is None
    assert fake_model.kwargs["ref_text"] is None
    assert metadata["reference_audio_required"] is False


def test_promotion_waits_for_twenty_votes_and_enforces_all_gates():
    candidate = _report(wer=0.09, similarity=0.7)
    baseline = _report(wer=0.08, similarity=0.5)
    previous = _report(wer=0.085, similarity=0.65)
    prompt_ids = [prompt.id for prompt in LOCKED_TEST_PROMPTS]

    pending = promotion_decision(
        candidate_report=candidate,
        baseline_report=baseline,
        previous_report=previous,
        blind_votes={prompt_id: "candidate" for prompt_id in prompt_ids[:19]},
    )
    assert pending["status"] == "pending"
    assert pending["action"] == "collect_blind_votes"

    twelve_wins = {
        prompt_id: "candidate" if index < 12 else "previous"
        for index, prompt_id in enumerate(prompt_ids)
    }
    promoted = promotion_decision(
        candidate_report=candidate,
        baseline_report=baseline,
        previous_report=previous,
        blind_votes=twelve_wins,
    )
    assert promoted["status"] == "promoted"
    assert promoted["metrics"]["candidate_blind_wins"] == 12

    rejected = promotion_decision(
        candidate_report=candidate,
        baseline_report=baseline,
        previous_report=previous,
        blind_votes={
            prompt_id: "candidate" if index < 11 else "previous"
            for index, prompt_id in enumerate(prompt_ids)
        },
    )
    assert rejected["status"] == "rejected"
    assert rejected["action"] == "retain_previous"

    not_reference_free = promotion_decision(
        candidate_report=_report(wer=0.05, similarity=0.8, reference_free=False),
        baseline_report=baseline,
        previous_report=previous,
        blind_votes=twelve_wins,
    )
    assert not_reference_free["status"] == "rejected"
    assert not_reference_free["gates"]["reference_free_generation"] is False


def test_blind_comparison_hides_labels_and_resolves_votes(tmp_path):
    candidate_items = []
    previous_items = []
    for prompt in LOCKED_TEST_PROMPTS:
        candidate_audio = tmp_path / "candidate" / f"{prompt.id}.wav"
        previous_audio = tmp_path / "previous" / f"{prompt.id}.wav"
        _write_tone(candidate_audio)
        _write_tone(previous_audio)
        candidate_items.append({"prompt_id": prompt.id, "audio_path": str(candidate_audio)})
        previous_items.append({"prompt_id": prompt.id, "audio_path": str(previous_audio)})

    public, private = create_blind_comparison(
        candidate_report=_report(items=candidate_items),
        previous_report=_report(items=previous_items),
        output_dir=tmp_path / "comparison",
    )
    assert len(public["pairs"]) == 20
    assert "candidate_option_by_prompt_id" not in public
    assert len(private["candidate_option_by_prompt_id"]) == 20

    option_votes = dict(private["candidate_option_by_prompt_id"])
    resolved = resolve_blind_votes(private_key=private, option_votes=option_votes)
    assert set(resolved.values()) == {"candidate"}
