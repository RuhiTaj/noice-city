from __future__ import annotations

import hashlib
import io
import json
import math
import os
import sqlite3
import struct
import wave
from pathlib import Path

import pytest

from voice_trainer.audio import (
    AudioValidationError,
    analyze_master_wav,
    inspect_wav,
)
from voice_trainer.prompts import PromptCatalog
from voice_trainer.store import (
    DEFAULT_VARIANT_ID,
    VoiceStore,
    default_store_paths,
    word_error_rate,
)


def wav_bytes(
    *,
    seconds: float = 4.0,
    amplitude: int = 2_560_000,
    sample_rate: int = 48_000,
    channels: int = 1,
    silence_seconds: float = 0.25,
    sample_width: int = 3,
    signal: str = "speech",
) -> bytes:
    frames = int(seconds * sample_rate)
    silence_frames = int(silence_seconds * sample_rate)
    active_values: list[float] = []
    for index in range(frames):
        if index < silence_frames or index >= frames - silence_frames:
            active_values.append(0.0)
            continue
        time_value = index / sample_rate
        if signal == "tone":
            value = math.sin(2 * math.pi * 220 * time_value)
        elif signal == "noise":
            # Deterministic white-noise-like sequence: plenty of energy, but no
            # pitch-range periodicity for the model-free speech gate.
            state = (1_103_515_245 * (index + 1) + 12_345) & 0x7FFFFFFF
            value = (state / 0x7FFFFFFF) * 2.0 - 1.0
        elif signal == "speech":
            envelope = 0.25 + 0.75 * math.sin(2 * math.pi * 2.3 * time_value) ** 2
            value = envelope * (
                math.sin(2 * math.pi * 125 * time_value)
                + 0.46 * math.sin(2 * math.pi * 250 * time_value + 0.2)
                + 0.30 * math.sin(2 * math.pi * 500 * time_value + 0.4)
                + 0.24 * math.sin(2 * math.pi * 750 * time_value + 0.7)
                + 0.18 * math.sin(2 * math.pi * 1_350 * time_value + 0.1)
                + 0.13 * math.sin(2 * math.pi * 2_300 * time_value + 0.6)
            )
        else:
            raise AssertionError(f"unknown test signal: {signal}")
        active_values.append(value)
    scale = amplitude / max((abs(value) for value in active_values), default=1.0)
    samples: list[int] = []
    for raw_value in active_values:
        value = round(raw_value * scale)
        samples.extend([value] * channels)
    if sample_width == 3:
        payload = b"".join(
            ((sample if sample >= 0 else sample + (1 << 24)) & 0xFFFFFF).to_bytes(3, "little")
            for sample in samples
        )
    elif sample_width == 2:
        payload = struct.pack(f"<{len(samples)}h", *samples)
    else:
        raise AssertionError("test helper supports PCM16 and PCM24")
    result = io.BytesIO()
    with wave.open(result, "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(sample_width)
        output.setframerate(sample_rate)
        output.writeframes(payload)
    return result.getvalue()


def test_prompt_catalog_is_versioned_and_deterministically_stratified() -> None:
    first = PromptCatalog.load()
    second = PromptCatalog.load()
    assert first.summary() == {
        "version": 2,
        "total": 300,
        "kinds": {"guided": 240, "interview": 60},
        "styles": {"neutral": 270, "expressive": 30},
        "splits": {"train": 240, "test": 30, "validation": 30},
        "packs": {"en-IN": 150, "en-US": 150},
        "default_pack": "en-US",
    }
    assert [(item.id, item.split) for item in first.prompts] == [
        (item.id, item.split) for item in second.prompts
    ]
    for kind, expected in (("guided", {"train": 192, "validation": 24, "test": 24}),
                           ("interview", {"train": 48, "validation": 6, "test": 6})):
        counts = {split: 0 for split in ("train", "validation", "test")}
        for prompt in first.prompts:
            if prompt.kind == kind:
                counts[prompt.split] += 1
        assert counts == expected
    assert all(prompt.locked == (prompt.split == "test") for prompt in first.prompts)


def test_audio_qa_and_24k_derivative(tmp_path: Path) -> None:
    payload = wav_bytes()
    report = analyze_master_wav(io.BytesIO(payload))
    assert report.passed
    assert report.metrics.sample_rate == 48_000
    assert report.metrics.channels == 1
    assert report.metrics.sample_width_bits == 24
    assert report.metrics.duration_seconds == pytest.approx(4.0)
    assert report.metrics.snr_db is not None and report.metrics.snr_db >= 25

    store = VoiceStore(tmp_path).initialize()
    prompt = next(prompt for prompt in store.catalog.prompts if prompt.kind == "guided")
    session = store.create_session()
    recording = store.add_recording(
        prompt.id,
        session_id=session["id"],
        master_bytes=payload,
        transcript=prompt.text,
    )
    assert inspect_wav(recording["master_path"])["sample_rate"] == 48_000
    derivative = inspect_wav(recording["training_path"])
    assert derivative["sample_rate"] == 24_000
    assert derivative["channels"] == 1
    assert derivative["sample_width_bits"] == 16
    assert derivative["duration_seconds"] == pytest.approx(4.0, abs=1 / 24_000)
    assert recording["transcript_status"] == "verified"
    assert recording["origin"] == "voice_trainer_fresh"


def test_audio_rejects_wrong_format_and_detects_hot_peak() -> None:
    with pytest.raises(AudioValidationError, match="48000 Hz"):
        analyze_master_wav(io.BytesIO(wav_bytes(sample_rate=44_100)))
    with pytest.raises(AudioValidationError, match="mono"):
        analyze_master_wav(io.BytesIO(wav_bytes(channels=2)))
    with pytest.raises(AudioValidationError, match="24-bit"):
        analyze_master_wav(io.BytesIO(wav_bytes(amplitude=10_000, sample_width=2)))
    report = analyze_master_wav(io.BytesIO(wav_bytes(amplitude=8_200_000)))
    assert not report.passed
    assert "peak_too_hot" in report.failures


def test_model_free_speech_activity_rejects_a_tone_and_accepts_speech() -> None:
    speech = analyze_master_wav(io.BytesIO(wav_bytes(signal="speech")))
    tone = analyze_master_wav(io.BytesIO(wav_bytes(signal="tone")))
    noise = analyze_master_wav(io.BytesIO(wav_bytes(signal="noise")))

    assert speech.passed
    assert speech.metrics.speech_activity.algorithm == "acoustic-vad-v1"
    assert speech.metrics.speech_activity.speech_ratio >= 0.12
    assert "speech_not_detected" not in speech.failures

    assert not tone.passed
    assert tone.metrics.speech_activity.active_frame_count > 0
    assert tone.metrics.speech_activity.speech_ratio < 0.12
    assert "speech_not_detected" in tone.failures

    assert not noise.passed
    assert noise.metrics.speech_activity.active_frame_count > 0
    assert "speech_not_detected" in noise.failures


def test_review_rules_and_immutable_checksummed_freeze(tmp_path: Path) -> None:
    store = VoiceStore(tmp_path).initialize()
    session = store.create_session("hybrid")
    selected = [
        next(prompt for prompt in store.catalog.prompts if prompt.kind == "guided" and prompt.split == "train"),
        next(prompt for prompt in store.catalog.prompts if prompt.kind == "guided" and prompt.split == "validation"),
        *[prompt for prompt in store.catalog.prompts if prompt.kind == "guided" and prompt.split == "test"][:2],
    ]
    recordings = []
    for guided in selected:
        recording = store.add_recording(
            guided.id,
            session_id=session["id"],
            master_bytes=wav_bytes(),
            transcript=guided.text,
        )
        recordings.append(recording)
        approved = store.update_recording(recording["id"], "approved")
        assert approved["review_status"] == "approved"

    frozen = store.freeze_dataset(
        "pilot one",
        purpose="pilot",
        minimum_duration_seconds=0,
        enforce_distribution=False,
    )
    manifest_path = Path(frozen["manifest_path"])
    assert frozen["state"] == "frozen"
    assert frozen["checksum_valid"]
    assert frozen["manifest_sha256"] == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["source_policy"] == "fresh_voice_trainer_recordings_only"
    assert manifest["recording_count"] == 4
    frozen_audio = manifest_path.parent / manifest["recordings"][0]["audio"]
    before = hashlib.sha256(frozen_audio.read_bytes()).hexdigest()
    source_audio = Path(recordings[0]["training_path"])
    assert os.stat(recordings[0]["master_path"]).st_mode & 0o777 == 0o400
    assert os.stat(source_audio).st_mode & 0o777 == 0o400
    # Even if the owner deliberately re-enables writes on the live derivative,
    # the frozen copy remains independent and checksummed.
    source_audio.chmod(0o600)
    source_audio.write_bytes(source_audio.read_bytes() + b"changed")
    assert hashlib.sha256(frozen_audio.read_bytes()).hexdigest() == before
    assert os.stat(frozen_audio).st_mode & 0o777 == 0o400


def test_interview_transcript_requires_explicit_confirmation(tmp_path: Path) -> None:
    store = VoiceStore(tmp_path).initialize()
    session = store.create_session("interview")
    interview = next(prompt for prompt in store.catalog.prompts if prompt.kind == "interview")
    recording = store.add_recording(
        interview.id,
        session_id=session["id"],
        master_bytes=wav_bytes(),
        transcript="I begin with a glass of water and a quiet walk.",
    )
    assert recording["transcript_status"] == "review_required"
    with pytest.raises(ValueError, match="played back"):
        store.update_recording(recording["id"], "approved")
    approved = store.update_recording(
        recording["id"],
        "approved",
        transcript="I begin with a glass of water and a quiet walk.",
        playback_confirmed=True,
    )
    assert approved["transcript_status"] == "confirmed"
    assert approved["asr_transcript"] != ""
    assert approved["confirmed_transcript"] == approved["asr_transcript"]


def test_store_schema_and_training_records(tmp_path: Path) -> None:
    store = VoiceStore(tmp_path).initialize()
    assert default_store_paths().root == (
        Path.home() / "Library" / "Application Support" / "VoiceTrainer"
    ).resolve()
    with sqlite3.connect(store.paths.database) as connection:
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    assert {
        "sessions",
        "prompts",
        "recordings",
        "custom_scripts",
        "custom_script_items",
        "dataset_versions",
        "dataset_recordings",
        "training_runs",
        "checkpoints",
        "generations",
    } <= tables
    assert word_error_rate("one two three", "one blue three") == pytest.approx(1 / 3)


def test_explicit_batch_approval_keeps_failed_takes_pending(tmp_path: Path) -> None:
    store = VoiceStore(tmp_path).initialize()
    session = store.create_session("guided")
    prompts = [prompt for prompt in store.catalog.prompts if prompt.kind == "guided"][:2]
    clean = store.add_recording(
        prompts[0].id,
        session_id=session["id"],
        master_bytes=wav_bytes(),
        transcript=prompts[0].text,
    )
    failed = store.add_recording(
        prompts[1].id,
        session_id=session["id"],
        master_bytes=wav_bytes(amplitude=8_200_000),
        transcript=prompts[1].text,
    )

    result = store.bulk_update_recordings(
        [clean["id"], failed["id"]],
        "approved",
        confirm_review=True,
    )

    assert result["updated_ids"] == [clean["id"]]
    assert result["skipped_count"] == 1
    assert store.get_recording(clean["id"])["review_status"] == "approved"
    assert store.get_recording(clean["id"])["playback_confirmed"] is False
    assert store.get_recording(failed["id"])["review_status"] == "pending"

    overridden = store.bulk_update_recordings(
        [failed["id"]],
        "approved",
        confirm_review=True,
        playback_confirmed_ids={failed["id"]},
        override_failed_qa=True,
    )
    assert overridden["updated_ids"] == [failed["id"]]
    assert overridden["overridden_ids"] == [failed["id"]]
    assert store.get_recording(failed["id"])["review_status"] == "approved"

    quarantined = store.bulk_update_recordings([failed["id"]], "quarantined")
    assert quarantined["updated_ids"] == [failed["id"]]
    assert store.get_recording(failed["id"])["review_status"] == "quarantined"

    rejected = store.bulk_update_recordings([clean["id"]], "rejected")
    assert rejected["updated_ids"] == [clean["id"]]
    assert store.get_recording(clean["id"])["review_status"] == "rejected"

    with pytest.raises(ValueError, match="explicit review confirmation"):
        store.bulk_update_recordings([clean["id"]], "approved")


def test_batch_playback_confirmation_is_recording_specific(tmp_path: Path) -> None:
    store = VoiceStore(tmp_path).initialize()
    session = store.create_session("guided")
    prompts = [prompt for prompt in store.catalog.prompts if prompt.kind == "guided"][:2]
    clean = store.add_recording(
        prompts[0].id,
        session_id=session["id"],
        master_bytes=wav_bytes(),
        transcript=prompts[0].text,
    )
    manual = store.add_recording(
        prompts[1].id,
        session_id=session["id"],
        master_bytes=wav_bytes(),
        transcript=None,
    )

    result = store.bulk_update_recordings(
        [clean["id"], manual["id"]],
        "approved",
        transcripts={manual["id"]: prompts[1].text},
        confirm_review=True,
    )

    assert result["updated_ids"] == [clean["id"]]
    assert result["skipped"] == [
        {
            "id": manual["id"],
            "reason": "Uncertain recordings must be played back before approval",
        }
    ]
    assert store.get_recording(clean["id"])["playback_confirmed"] is False
    assert store.get_recording(manual["id"])["playback_confirmed"] is False

    reviewed = store.bulk_update_recordings(
        [manual["id"]],
        "approved",
        transcripts={manual["id"]: prompts[1].text},
        confirm_review=True,
        playback_confirmed_ids={manual["id"]},
    )
    assert reviewed["updated_ids"] == [manual["id"]]
    assert store.get_recording(manual["id"])["playback_confirmed"] is True


def _approved_variant_recordings(store: VoiceStore, variant_id: str) -> list[dict]:
    session = store.create_session("guided", variant_id=variant_id)
    prompts = [
        next(item for item in store.catalog.prompts if item.kind == "guided" and item.split == "train"),
        next(item for item in store.catalog.prompts if item.kind == "guided" and item.split == "validation"),
        *[item for item in store.catalog.prompts if item.kind == "guided" and item.split == "test"][:2],
    ]
    rows = []
    for prompt in prompts:
        row = store.add_recording(
            prompt.id,
            session_id=session["id"],
            master_bytes=wav_bytes(),
            transcript=prompt.text,
        )
        rows.append(store.update_recording(row["id"], "approved"))
    return rows


def test_voice_variant_lineage_filtering_and_promotions(tmp_path: Path) -> None:
    store = VoiceStore(tmp_path).initialize()
    default = store.get_voice_variant()
    assert default["id"] == DEFAULT_VARIANT_ID
    assert default["name"] == "My Voice"
    assert default["is_default"] is True

    variant = store.create_voice_variant(
        "Indian Names",
        slug="indian-names",
        description="Place-name pronunciation practice.",
        metadata={"language": "english"},
    )
    assert variant["id"].startswith("variant_")
    assert variant["id"] == variant["id"].lower()
    assert store.get_voice_variant(variant["id"])["metadata_json"] == {
        "language": "english",
        "prompt_pack": "en-US",
    }

    default_rows = _approved_variant_recordings(store, DEFAULT_VARIANT_ID)
    variant_rows = _approved_variant_recordings(store, variant["id"])
    assert {row["id"] for row in store.list_recordings(variant_id=DEFAULT_VARIANT_ID)} == {
        row["id"] for row in default_rows
    }
    assert {row["id"] for row in store.list_recordings(variant_id=variant["id"])} == {
        row["id"] for row in variant_rows
    }
    with pytest.raises(KeyError, match="Unknown recording"):
        store.get_recording(default_rows[0]["id"], variant_id=variant["id"])

    default_dataset = store.freeze_dataset(
        "default pilot", purpose="pilot", minimum_duration_seconds=0,
        enforce_distribution=False, variant_id=DEFAULT_VARIANT_ID,
    )
    variant_dataset = store.freeze_dataset(
        "names pilot", purpose="pilot", minimum_duration_seconds=0,
        enforce_distribution=False, variant_id=variant["id"],
    )
    assert default_dataset["recording_count"] == len(default_rows)
    assert variant_dataset["recording_count"] == len(variant_rows)
    manifest = json.loads(Path(variant_dataset["manifest_path"]).read_text())
    assert manifest["schema_version"] == 2
    assert manifest["variant_id"] == variant["id"]
    assert manifest["voice_variant"] == {
        "variant_id": variant["id"],
        "slug": "indian-names",
        "display_name": "Indian Names",
        "profile": "ruhi",
        "language": "english",
        "prompt_pack": "en-US",
    }

    default_run = store.create_training_run(default_dataset["id"], {"stage": "pilot"})
    variant_run = store.create_training_run(variant_dataset["id"], {"stage": "pilot"})
    default_checkpoint = store.add_checkpoint(
        default_run["id"], 1, tmp_path / "default-artifact", {}, status="promoted"
    )
    variant_checkpoint = store.add_checkpoint(
        variant_run["id"], 1, tmp_path / "variant-artifact", {}, status="promoted"
    )
    assert default_checkpoint["variant_id"] == DEFAULT_VARIANT_ID
    assert variant_checkpoint["variant_id"] == variant["id"]
    assert store.get_checkpoint(default_checkpoint["id"])["status"] == "promoted"
    assert store.get_checkpoint(variant_checkpoint["id"])["status"] == "promoted"
    default_second = store.add_checkpoint(
        default_run["id"], 2, tmp_path / "default-artifact-2", {}, status="candidate"
    )
    store.update_checkpoint(default_second["id"], status="promoted")
    assert store.get_checkpoint(default_checkpoint["id"])["status"] == "candidate"
    assert store.get_checkpoint(variant_checkpoint["id"])["status"] == "promoted"
    assert store.status(DEFAULT_VARIANT_ID)["promoted_checkpoint_id"] == default_second["id"]
    assert store.status(variant["id"])["promoted_checkpoint_id"] == variant_checkpoint["id"]
    assert [row["id"] for row in store.list_training_runs(variant_id=variant["id"])] == [
        variant_run["id"]
    ]

    generation = store.add_generation(
        variant_checkpoint["id"], "Hello Bengaluru.", tmp_path / "variant.wav",
        {"preset": "natural"}, status="complete",
    )
    assert generation["variant_id"] == variant["id"]
    assert [row["id"] for row in store.list_generations(variant_id=variant["id"])] == [
        generation["id"]
    ]

    with sqlite3.connect(store.paths.database) as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        index_sql = connection.execute(
            """SELECT sql FROM sqlite_master
               WHERE type='index' AND name='idx_one_promoted_checkpoint_per_variant'"""
        ).fetchone()[0]
        assert "ON checkpoints(variant_id) WHERE status = 'promoted'" in index_sql
        assert connection.execute(
            "SELECT COUNT(*) FROM checkpoints WHERE status='promoted'"
        ).fetchone()[0] == 2
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """INSERT INTO training_runs(
                    id,variant_id,dataset_id,status,progress,config_json,metadata_json,created_at
                ) VALUES ('cross-variant',?,?, 'pending',0,'{}','{}','now')""",
                (variant["id"], default_dataset["id"]),
            )


def test_variant_crud_and_session_prompt_validation(tmp_path: Path) -> None:
    store = VoiceStore(tmp_path).initialize()
    variant = store.create_voice_variant("Conversational", make_default=True)
    assert store.get_voice_variant()["id"] == variant["id"]
    assert len(store.list_voice_variants(include_archived=False)) == 2
    store.set_default_voice_variant(DEFAULT_VARIANT_ID)
    archived = store.update_voice_variant(
        variant["id"], state="archived", description="Archived test variant."
    )
    assert archived["state"] == "archived"
    with pytest.raises(ValueError, match="archived"):
        store.create_session(variant_id=variant["id"])

    session = store.create_session()
    assert store.get_session(session["id"])["variant_id"] == DEFAULT_VARIANT_ID
    store.end_session(session["id"])
    with pytest.raises(ValueError, match="not active"):
        store.next_prompt(session_id=session["id"])
    with pytest.raises(KeyError, match="Unknown session"):
        store.next_prompt(session_id="session_missing")


def test_prompt_pack_defaults_and_persists_per_voice_variant(tmp_path: Path) -> None:
    store = VoiceStore(tmp_path).initialize()
    default = store.get_voice_variant()
    assert default["prompt_pack"] == "en-US"
    default_session = store.create_session("guided", variant_id=default["id"])
    assert store.next_prompt(session_id=default_session["id"], mode="guided")["id"].startswith("us-")

    variant = store.create_voice_variant("Home voice")
    assert variant["prompt_pack"] == "en-US"
    updated = store.update_voice_variant(variant["id"], prompt_pack="en-IN")
    assert updated["prompt_pack"] == "en-IN"
    assert updated["metadata_json"]["prompt_pack"] == "en-IN"
    indian_session = store.create_session("guided", variant_id=variant["id"])
    assert not store.next_prompt(session_id=indian_session["id"], mode="guided")["id"].startswith("us-")
    assert store.status(variant["id"])["prompt_pack"] == "en-IN"


def test_pre_pack_variant_retains_indian_prompt_behavior(tmp_path: Path) -> None:
    store = VoiceStore(tmp_path).initialize()
    with sqlite3.connect(store.paths.database) as connection:
        connection.execute(
            "UPDATE voice_variants SET metadata_json='{}' WHERE id=?",
            (DEFAULT_VARIANT_ID,),
        )

    reopened = VoiceStore(tmp_path).initialize()
    assert reopened.get_voice_variant()["prompt_pack"] == "en-IN"
    session = reopened.create_session("guided")
    assert not reopened.next_prompt(session_id=session["id"], mode="guided")["id"].startswith("us-")


def _create_v1_database(path: Path, immutable_manifest: Path) -> None:
    schema = """
    CREATE TABLE app_meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
    CREATE TABLE sessions(id TEXT PRIMARY KEY,mode TEXT,status TEXT,created_at TEXT,ended_at TEXT,notes TEXT);
    CREATE TABLE prompts(id TEXT PRIMARY KEY,catalog_version INTEGER,kind TEXT,style TEXT,split TEXT,locked INTEGER,ordinal INTEGER,text TEXT);
    CREATE TABLE recordings(id TEXT PRIMARY KEY,session_id TEXT,prompt_id TEXT,origin TEXT,created_at TEXT,master_path TEXT UNIQUE,training_path TEXT UNIQUE,master_sha256 TEXT,training_sha256 TEXT,duration_seconds REAL,expected_transcript TEXT,asr_transcript TEXT,confirmed_transcript TEXT,transcript_wer REAL,transcript_status TEXT,qa_json TEXT,qa_passed INTEGER,qa_needs_review INTEGER,playback_confirmed INTEGER,review_status TEXT,review_note TEXT,reviewed_at TEXT);
    CREATE TABLE dataset_versions(id TEXT PRIMARY KEY,name TEXT,purpose TEXT,state TEXT,created_at TEXT,manifest_path TEXT UNIQUE,manifest_sha256 TEXT,recording_count INTEGER,duration_seconds REAL,catalog_version INTEGER,metadata_json TEXT);
    CREATE TABLE dataset_recordings(dataset_id TEXT,recording_id TEXT,prompt_id TEXT,split TEXT,kind TEXT,style TEXT,training_text TEXT,frozen_audio_path TEXT,frozen_audio_sha256 TEXT,source_master_sha256 TEXT,qa_json TEXT,PRIMARY KEY(dataset_id,recording_id));
    CREATE TABLE training_runs(id TEXT PRIMARY KEY,dataset_id TEXT,status TEXT,progress REAL,config_json TEXT,metadata_json TEXT,error TEXT,created_at TEXT,started_at TEXT,finished_at TEXT);
    CREATE TABLE checkpoints(id TEXT PRIMARY KEY,training_run_id TEXT,epoch INTEGER,artifact_path TEXT UNIQUE,status TEXT,metrics_json TEXT,created_at TEXT,UNIQUE(training_run_id,epoch));
    CREATE TABLE generations(id TEXT PRIMARY KEY,checkpoint_id TEXT,text TEXT,delivery_json TEXT,output_path TEXT UNIQUE,status TEXT,metrics_json TEXT,error TEXT,created_at TEXT,finished_at TEXT);
    """
    manifest_hash = hashlib.sha256(immutable_manifest.read_bytes()).hexdigest()
    with sqlite3.connect(path) as connection:
        connection.executescript(schema)
        connection.execute("INSERT INTO app_meta VALUES ('schema_version','1')")
        connection.execute("INSERT INTO sessions VALUES ('s1','hybrid','complete','now','later',NULL)")
        connection.execute("INSERT INTO prompts VALUES ('p1',1,'guided','neutral','train',0,1,'Legacy text.')")
        connection.execute("""INSERT INTO recordings VALUES(
            'r1','s1','p1','voice_trainer_fresh','now','/legacy/master.wav','/legacy/train.wav',
            'master-hash','training-hash',4.0,'Legacy text.','Legacy text.','Legacy text.',0.0,
            'confirmed','{}',1,0,1,'approved',NULL,'later')""")
        connection.execute(
            "INSERT INTO dataset_versions VALUES ('d1','legacy','pilot','frozen','now',?,?,1,4.0,1,'{}')",
            (str(immutable_manifest), manifest_hash),
        )
        connection.execute("INSERT INTO dataset_recordings VALUES ('d1','r1','p1','train','guided','neutral','Legacy text.','/legacy/frozen.wav','frozen-hash','master-hash','{}')")
        connection.execute("INSERT INTO training_runs VALUES ('t1','d1','complete',1.0,'{}','{}',NULL,'now','now','later')")
        connection.execute("INSERT INTO checkpoints VALUES ('c1','t1',1,'/legacy/artifact','promoted','{}','now')")
        connection.execute("INSERT INTO generations VALUES ('g1','c1','Legacy generation.','{}','/legacy/output.wav','complete','{}',NULL,'now','later')")


def test_schema_v1_copy_on_write_migration_preserves_lineage_and_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "legacy-manifest.json"
    manifest.write_text('{"schema_version":1,"profile":"ruhi"}\n')
    manifest.chmod(0o400)
    before = manifest.read_bytes()
    database = tmp_path / "state.sqlite"
    _create_v1_database(database, manifest)

    store = VoiceStore(tmp_path).initialize()

    assert manifest.read_bytes() == before
    assert store.get_voice_variant()["prompt_pack"] == "en-IN"
    legacy_backup = tmp_path / "state.sqlite.schema-v1.backup"
    assert legacy_backup.is_file()
    with sqlite3.connect(legacy_backup) as connection:
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert connection.execute(
            "SELECT value FROM app_meta WHERE key='schema_version'"
        ).fetchone()[0] == "1"
    with sqlite3.connect(store.paths.database) as connection:
        connection.row_factory = sqlite3.Row
        assert connection.execute(
            "SELECT value FROM app_meta WHERE key='schema_version'"
        ).fetchone()[0] == "3"
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 3
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        for table in (
            "sessions", "recordings", "dataset_versions", "dataset_recordings",
            "training_runs", "checkpoints", "generations",
        ):
            assert [row[0] for row in connection.execute(
                f"SELECT DISTINCT variant_id FROM {table}"
            ).fetchall()] == [DEFAULT_VARIANT_ID]
        assert tuple(connection.execute(
            "SELECT master_path,master_sha256,training_path,training_sha256 FROM recordings"
        ).fetchone()) == ("/legacy/master.wav", "master-hash", "/legacy/train.wav", "training-hash")
        assert tuple(connection.execute(
            "SELECT manifest_path,manifest_sha256 FROM dataset_versions WHERE id='d1'"
        ).fetchone()) == (str(manifest), hashlib.sha256(before).hexdigest())
        assert connection.execute(
            "SELECT COUNT(*) FROM checkpoints WHERE status='promoted'"
        ).fetchone()[0] == 1

    # Initialization is idempotent and never rewrites the immutable schema-v1 file.
    VoiceStore(tmp_path).initialize()
    assert manifest.read_bytes() == before


def _unlock_custom_scripts(store: VoiceStore, variant_id: str) -> dict:
    """Create one real approved catalog take, then age its duration for a fast gate test."""

    prompt = next(
        item for item in store.catalog.prompts
        if item.kind == "guided" and item.split == "train"
    )
    session = store.create_session("guided", variant_id=variant_id)
    recording = store.add_recording(
        prompt.id,
        session_id=session["id"],
        master_bytes=wav_bytes(),
        transcript=prompt.text,
    )
    recording = store.update_recording(recording["id"], "approved")
    with sqlite3.connect(store.paths.database) as connection:
        connection.execute(
            "UPDATE recordings SET duration_seconds=120 WHERE id=?",
            (recording["id"],),
        )
    return recording


def test_custom_script_gate_segmentation_idempotence_and_locked_prompts(tmp_path: Path) -> None:
    store = VoiceStore(tmp_path).initialize()
    assert store.status()["custom_script_gate_seconds"] == 0
    assert store.status()["custom_script_unlocked"] is False
    with pytest.raises(ValueError, match="120 seconds"):
        store.create_custom_script(DEFAULT_VARIANT_ID, "A short custom narration practice line.")

    _unlock_custom_scripts(store, DEFAULT_VARIANT_ID)
    status = store.status()
    assert status["custom_script_gate_seconds"] == pytest.approx(120)
    assert status["custom_script_unlocked"] is True

    words = [f"word{index}{'.' if index in {19, 39, 69} else ''}" for index in range(70)]
    source = "  \n".join(words)
    script = store.create_custom_script(DEFAULT_VARIANT_ID, source, title="Chapter one")
    assert script["created"] is True
    assert script["pending_count"] == script["item_count"]
    duplicate = store.create_custom_script(
        DEFAULT_VARIANT_ID, " ".join(words), title="Ignored retry title"
    )
    assert duplicate["id"] == script["id"]
    assert duplicate["created"] is False

    with sqlite3.connect(store.paths.database) as connection:
        connection.row_factory = sqlite3.Row
        items = connection.execute(
            """SELECT p.*,i.position,i.status FROM custom_script_items i
               JOIN prompts p ON p.id=i.prompt_id
               WHERE i.script_id=? ORDER BY i.position""",
            (script["id"],),
        ).fetchall()
    assert " ".join(row["text"] for row in items) == " ".join(words)
    assert all(8 <= len(row["text"].split()) <= 28 for row in items)
    assert all(len(row["text"]) <= 240 for row in items)
    assert all(
        (row["kind"], row["style"], row["split"], row["locked"], row["status"])
        == ("guided", "neutral", "train", 0, "pending")
        for row in items
    )
    assert store.status()["pending_custom_prompts"] == len(items)
    assert store.next_custom_script_prompt(DEFAULT_VARIANT_ID)["id"] == items[0]["id"]

    for row in items:
        skipped = store.skip_custom_script_prompt(DEFAULT_VARIANT_ID, row["id"])
        assert skipped["custom_status"] == "skipped"
    assert store.next_custom_script_prompt(DEFAULT_VARIANT_ID) is None
    assert store.status()["pending_custom_prompts"] == 0
    fresh_pass = store.create_custom_script(DEFAULT_VARIANT_ID, " ".join(words))
    assert fresh_pass["id"] != script["id"]
    assert fresh_pass["created"] is True

    locked = next(item for item in store.catalog.prompts if item.locked)
    with pytest.raises(ValueError, match="locked catalog prompt"):
        store.create_custom_script(DEFAULT_VARIANT_ID, locked.text)
    with pytest.raises(ValueError, match="240 characters"):
        store.create_custom_script(DEFAULT_VARIANT_ID, "x" * 241)


def test_custom_prompt_variant_ownership_recording_reopen_and_manifest(tmp_path: Path) -> None:
    store = VoiceStore(tmp_path).initialize()
    other = store.create_voice_variant("Other Voice")
    _unlock_custom_scripts(store, DEFAULT_VARIANT_ID)
    _unlock_custom_scripts(store, other["id"])
    text = "Please read this calm sentence with steady pacing and clear thoughtful emphasis."
    script = store.create_custom_script(DEFAULT_VARIANT_ID, text)
    prompt = store.next_custom_script_prompt(DEFAULT_VARIANT_ID)
    assert prompt is not None and prompt["custom_script_id"] == script["id"]
    assert prompt["is_custom"] is True
    with pytest.raises(ValueError, match="another voice variant"):
        store.get_prompt(prompt["id"], other["id"])
    catalog_prompt = next(item for item in store.catalog.prompts if item.split == "train")
    assert store.get_prompt(catalog_prompt.id, other["id"])["is_custom"] is False

    other_session = store.create_session("guided", variant_id=other["id"])
    with pytest.raises(ValueError, match="another voice variant"):
        store.add_recording(
            prompt["id"],
            session_id=other_session["id"],
            master_bytes=wav_bytes(),
            transcript=text,
        )

    session = store.create_session("guided", variant_id=DEFAULT_VARIANT_ID)
    recorded = store.add_recording(
        prompt["id"],
        session_id=session["id"],
        master_bytes=wav_bytes(),
        transcript=text,
    )
    assert store.get_prompt(prompt["id"], DEFAULT_VARIANT_ID)["custom_status"] == "recorded"
    with pytest.raises(ValueError, match="explicitly confirmed transcript"):
        store.update_recording(recorded["id"], "approved")
    approved = store.update_recording(recorded["id"], "approved", transcript=text)
    assert approved["confirmed_transcript"] == text

    store.update_recording(recorded["id"], "quarantined")
    reopened = store.next_custom_script_prompt(DEFAULT_VARIANT_ID)
    assert reopened is not None and reopened["id"] == prompt["id"]
    assert reopened["custom_status"] == "pending"
    replacement = store.add_recording(
        prompt["id"],
        session_id=session["id"],
        master_bytes=wav_bytes(),
        transcript=text,
    )
    store.update_recording(replacement["id"], "approved", transcript=text)

    # The frozen pilot includes validation/test catalog coverage, and marks the
    # custom train-only row as supplemental instead of skewing catalog ratios.
    coverage_prompts = [
        next(item for item in store.catalog.prompts if item.split == "validation"),
        *[item for item in store.catalog.prompts if item.split == "test"][:2],
    ]
    for catalog in coverage_prompts:
        row = store.add_recording(
            catalog.id,
            session_id=session["id"],
            master_bytes=wav_bytes(),
            transcript=catalog.text,
        )
        store.update_recording(row["id"], "approved")
    dataset = store.freeze_dataset(
        "custom pilot",
        purpose="pilot",
        minimum_duration_seconds=0,
        enforce_distribution=False,
        variant_id=DEFAULT_VARIANT_ID,
    )
    manifest = json.loads(Path(dataset["manifest_path"]).read_text())
    assert manifest["supplemental_custom_count"] == 1
    assert manifest["distribution"]["supplemental_custom_count"] == 1
    custom_entry = next(row for row in manifest["recordings"] if row["supplemental_custom"])
    assert custom_entry["custom_script_id"] == script["id"]
    assert custom_entry["training_text"] == text


def test_schema_v2_additive_migration_is_backed_up_and_idempotent(tmp_path: Path) -> None:
    store = VoiceStore(tmp_path).initialize()
    session = store.create_session("guided")
    with sqlite3.connect(store.paths.database) as connection:
        connection.execute("DROP TABLE custom_script_items")
        connection.execute("DROP TABLE custom_scripts")
        connection.execute("UPDATE app_meta SET value='2' WHERE key='schema_version'")
        connection.execute("PRAGMA user_version=2")

    migrated = VoiceStore(tmp_path).initialize()
    backup = tmp_path / "state.sqlite.schema-v2.backup"
    assert backup.is_file()
    with sqlite3.connect(backup) as connection:
        assert connection.execute(
            "SELECT value FROM app_meta WHERE key='schema_version'"
        ).fetchone()[0] == "2"
    with sqlite3.connect(migrated.paths.database) as connection:
        tables = {
            row[0] for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        assert {"custom_scripts", "custom_script_items"} <= tables
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 3
        assert connection.execute(
            "SELECT COUNT(*) FROM sessions WHERE id=?", (session["id"],)
        ).fetchone()[0] == 1
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
    VoiceStore(tmp_path).initialize()
