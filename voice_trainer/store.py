"""Private local state, fresh recording intake, and immutable datasets."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import tempfile
import unicodedata
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator, Mapping

from .audio import AudioQualityReport, analyze_master_wav, derive_training_wav, save_master_wav
from .prompts import Prompt, PromptCatalog


PROFILE_ID = "ruhi"
SCHEMA_VERSION = 3
DEFAULT_VARIANT_ID = "variant_ruhi_default"
DEFAULT_VARIANT_SLUG = "ruhi"
DEFAULT_VARIANT_NAME = "My Voice"
RECORDING_STATUSES = frozenset({"pending", "approved", "quarantined", "rejected"})
RUN_STATUSES = frozenset({"pending", "running", "cancelling", "cancelled", "failed", "complete"})
CHECKPOINT_STATUSES = frozenset({"candidate", "promoted", "rejected"})


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def _id(prefix: str) -> str:
    return f"{prefix}_{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:12]}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _json_object(value: str | None) -> dict[str, Any]:
    return json.loads(value) if value else {}


def _normalize_transcript(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text.casefold())


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Calculate word-level Levenshtein error without modifying either transcript."""

    expected = _normalize_transcript(reference)
    actual = _normalize_transcript(hypothesis)
    if not expected:
        return 0.0 if not actual else 1.0
    previous = list(range(len(actual) + 1))
    for ref_index, ref_word in enumerate(expected, start=1):
        current = [ref_index]
        for hyp_index, hyp_word in enumerate(actual, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[hyp_index] + 1,
                    previous[hyp_index - 1] + (ref_word != hyp_word),
                )
            )
        previous = current
    return previous[-1] / len(expected)


def _normalize_custom_script(text: str) -> str:
    if not isinstance(text, str):
        raise ValueError("Custom script text must be a string")
    if len(text) > 20_000:
        raise ValueError("Custom script text must contain at most 20,000 characters")
    normalized = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()
    if not normalized:
        raise ValueError("Custom script text cannot be empty")
    return normalized


def _segment_custom_script(normalized_text: str) -> tuple[str, ...]:
    """Split normalized text deterministically while preserving every token."""

    words = normalized_text.split(" ")
    if any(len(word) > 240 for word in words):
        raise ValueError("Custom script words must contain at most 240 characters")

    # Dynamic programming keeps normal takes near 8–28 words, prefers sentence
    # endings, and also enforces a hard character bound. Short takes are allowed
    # only when the source itself (or unusually long words) makes them necessary.
    count = len(words)
    scores: list[tuple[int, int, int, int] | None] = [None] * (count + 1)
    next_end: list[int | None] = [None] * (count + 1)
    scores[count] = (0, 0, 0, 0)
    for start in range(count - 1, -1, -1):
        characters = 0
        for end in range(start + 1, min(count, start + 28) + 1):
            characters += len(words[end - 1]) + (1 if end - start > 1 else 0)
            if characters > 240:
                break
            tail = scores[end]
            if tail is None:
                continue
            word_count = end - start
            sentence_end = end == count or bool(
                re.search(r"[.!?][\"')\]]*$", words[end - 1])
            )
            score = (
                tail[0] + int(word_count < 8),
                tail[1] + 1,
                tail[2] + int(not sentence_end),
                tail[3] + (word_count - 18) ** 2,
            )
            if scores[start] is None or score < scores[start]:
                scores[start] = score
                next_end[start] = end
    if scores[0] is None:
        raise ValueError("Custom script cannot be divided into safe recording takes")
    if scores[0][1] > 200:
        raise ValueError("Custom script produces more than 200 recordable segments")

    segments: list[str] = []
    start = 0
    while start < count:
        end = next_end[start]
        if end is None:
            raise RuntimeError("Custom script segmentation failed")
        segments.append(" ".join(words[start:end]))
        start = end

    if " ".join(segments) != normalized_text:
        raise RuntimeError("Custom script segmentation lost source words")
    if any(len(segment.split()) > 28 or len(segment) > 240 for segment in segments):
        raise RuntimeError("Custom script segmentation produced an invalid take")
    return tuple(segments)


@dataclass(frozen=True)
class StorePaths:
    root: Path
    database: Path
    profiles: Path
    recordings: Path
    masters: Path
    training_audio: Path
    datasets: Path
    adapters: Path
    evaluations: Path
    generations: Path

    @classmethod
    def under(cls, root: str | os.PathLike[str]) -> "StorePaths":
        root_path = Path(root).expanduser().resolve()
        recordings = root_path / "profiles" / PROFILE_ID / "recordings"
        return cls(
            root=root_path,
            database=root_path / "state.sqlite",
            profiles=root_path / "profiles",
            recordings=recordings,
            masters=recordings / "masters",
            training_audio=recordings / "training-24k",
            datasets=root_path / "datasets",
            adapters=root_path / "adapters",
            evaluations=root_path / "evaluations",
            generations=root_path / "generations",
        )


def default_store_paths() -> StorePaths:
    return StorePaths.under(Path.home() / "Library" / "Application Support" / "VoiceTrainer")


CUSTOM_SCRIPTS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS custom_scripts (
    id TEXT PRIMARY KEY,
    variant_id TEXT NOT NULL REFERENCES voice_variants(id) ON DELETE RESTRICT,
    title TEXT,
    source_text TEXT NOT NULL,
    source_sha256 TEXT NOT NULL,
    state TEXT NOT NULL DEFAULT 'active' CHECK (state IN ('active', 'complete')),
    item_count INTEGER NOT NULL CHECK (item_count >= 1 AND item_count <= 200),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    completed_at TEXT,
    UNIQUE (id, variant_id)
)
"""

CUSTOM_SCRIPT_ITEMS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS custom_script_items (
    script_id TEXT NOT NULL,
    variant_id TEXT NOT NULL REFERENCES voice_variants(id) ON DELETE RESTRICT,
    prompt_id TEXT NOT NULL UNIQUE REFERENCES prompts(id) ON DELETE RESTRICT,
    position INTEGER NOT NULL CHECK (position >= 0),
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'skipped', 'recorded')),
    recording_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (script_id, prompt_id),
    UNIQUE (script_id, position),
    FOREIGN KEY (script_id, variant_id)
        REFERENCES custom_scripts(id, variant_id) ON DELETE RESTRICT,
    FOREIGN KEY (recording_id, variant_id)
        REFERENCES recordings(id, variant_id) ON DELETE RESTRICT
)
"""

CUSTOM_SCRIPT_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_custom_script_items_variant_pending
ON custom_script_items(variant_id, status, script_id, position)
"""

CUSTOM_SCRIPT_ACTIVE_HASH_INDEX_SQL = """
CREATE UNIQUE INDEX IF NOT EXISTS idx_custom_scripts_active_source
ON custom_scripts(variant_id, source_sha256) WHERE state = 'active'
"""

CUSTOM_SCRIPT_SCHEMA = ";\n".join(
    (
        CUSTOM_SCRIPTS_TABLE_SQL,
        CUSTOM_SCRIPT_ITEMS_TABLE_SQL,
        CUSTOM_SCRIPT_INDEX_SQL,
        CUSTOM_SCRIPT_ACTIVE_HASH_INDEX_SQL,
    )
) + ";\n"


SCHEMA = """
CREATE TABLE IF NOT EXISTS app_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS voice_variants (
    id TEXT PRIMARY KEY,
    slug TEXT NOT NULL COLLATE NOCASE UNIQUE,
    name TEXT NOT NULL,
    description TEXT,
    state TEXT NOT NULL DEFAULT 'active' CHECK (state IN ('active', 'archived')),
    is_default INTEGER NOT NULL DEFAULT 0 CHECK (is_default IN (0, 1)),
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_one_default_voice_variant
ON voice_variants(is_default) WHERE is_default = 1;

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    variant_id TEXT NOT NULL REFERENCES voice_variants(id) ON DELETE RESTRICT,
    mode TEXT NOT NULL CHECK (mode IN ('guided', 'interview', 'hybrid')),
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'complete', 'abandoned')),
    created_at TEXT NOT NULL,
    ended_at TEXT,
    notes TEXT,
    UNIQUE (id, variant_id)
);

CREATE INDEX IF NOT EXISTS idx_sessions_variant_status
ON sessions(variant_id, status, created_at DESC);

CREATE TABLE IF NOT EXISTS prompts (
    id TEXT PRIMARY KEY,
    catalog_version INTEGER NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('guided', 'interview')),
    style TEXT NOT NULL CHECK (style IN ('neutral', 'expressive')),
    split TEXT NOT NULL CHECK (split IN ('train', 'validation', 'test')),
    locked INTEGER NOT NULL CHECK (locked IN (0, 1)),
    ordinal INTEGER NOT NULL,
    text TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS recordings (
    id TEXT PRIMARY KEY,
    variant_id TEXT NOT NULL REFERENCES voice_variants(id) ON DELETE RESTRICT,
    session_id TEXT NOT NULL,
    prompt_id TEXT NOT NULL REFERENCES prompts(id),
    origin TEXT NOT NULL DEFAULT 'voice_trainer_fresh' CHECK (origin = 'voice_trainer_fresh'),
    created_at TEXT NOT NULL,
    master_path TEXT NOT NULL UNIQUE,
    training_path TEXT NOT NULL UNIQUE,
    master_sha256 TEXT NOT NULL,
    training_sha256 TEXT NOT NULL,
    duration_seconds REAL NOT NULL,
    expected_transcript TEXT,
    asr_transcript TEXT,
    confirmed_transcript TEXT,
    transcript_wer REAL,
    transcript_status TEXT NOT NULL CHECK (
        transcript_status IN ('pending', 'verified', 'review_required', 'confirmed')
    ),
    qa_json TEXT NOT NULL,
    qa_passed INTEGER NOT NULL CHECK (qa_passed IN (0, 1)),
    qa_needs_review INTEGER NOT NULL CHECK (qa_needs_review IN (0, 1)),
    playback_confirmed INTEGER NOT NULL DEFAULT 0 CHECK (playback_confirmed IN (0, 1)),
    review_status TEXT NOT NULL DEFAULT 'pending' CHECK (
        review_status IN ('pending', 'approved', 'quarantined', 'rejected')
    ),
    review_note TEXT,
    reviewed_at TEXT,
    UNIQUE (id, variant_id),
    FOREIGN KEY (session_id, variant_id) REFERENCES sessions(id, variant_id) ON DELETE RESTRICT
);

CREATE INDEX IF NOT EXISTS idx_recordings_review ON recordings(review_status, created_at);
CREATE INDEX IF NOT EXISTS idx_recordings_session ON recordings(session_id, prompt_id);
CREATE INDEX IF NOT EXISTS idx_recordings_variant_review
ON recordings(variant_id, review_status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_recordings_variant_session
ON recordings(variant_id, session_id, prompt_id);

CREATE TABLE IF NOT EXISTS dataset_versions (
    id TEXT PRIMARY KEY,
    variant_id TEXT NOT NULL REFERENCES voice_variants(id) ON DELETE RESTRICT,
    name TEXT NOT NULL,
    purpose TEXT NOT NULL CHECK (purpose IN ('pilot', 'full')),
    state TEXT NOT NULL DEFAULT 'frozen' CHECK (state = 'frozen'),
    created_at TEXT NOT NULL,
    manifest_path TEXT NOT NULL UNIQUE,
    manifest_sha256 TEXT NOT NULL,
    recording_count INTEGER NOT NULL,
    duration_seconds REAL NOT NULL,
    catalog_version INTEGER NOT NULL,
    metadata_json TEXT NOT NULL,
    UNIQUE (id, variant_id)
);

CREATE INDEX IF NOT EXISTS idx_datasets_variant_created
ON dataset_versions(variant_id, created_at DESC);

CREATE TABLE IF NOT EXISTS dataset_recordings (
    dataset_id TEXT NOT NULL,
    recording_id TEXT NOT NULL,
    variant_id TEXT NOT NULL REFERENCES voice_variants(id) ON DELETE RESTRICT,
    prompt_id TEXT NOT NULL,
    split TEXT NOT NULL CHECK (split IN ('train', 'validation', 'test')),
    kind TEXT NOT NULL CHECK (kind IN ('guided', 'interview')),
    style TEXT NOT NULL CHECK (style IN ('neutral', 'expressive')),
    training_text TEXT NOT NULL,
    frozen_audio_path TEXT NOT NULL,
    frozen_audio_sha256 TEXT NOT NULL,
    source_master_sha256 TEXT NOT NULL,
    qa_json TEXT NOT NULL,
    PRIMARY KEY (dataset_id, recording_id),
    FOREIGN KEY (dataset_id, variant_id)
        REFERENCES dataset_versions(id, variant_id) ON DELETE RESTRICT,
    FOREIGN KEY (recording_id, variant_id)
        REFERENCES recordings(id, variant_id) ON DELETE RESTRICT
);

CREATE INDEX IF NOT EXISTS idx_dataset_recordings_variant_recording
ON dataset_recordings(variant_id, recording_id);

CREATE TABLE IF NOT EXISTS training_runs (
    id TEXT PRIMARY KEY,
    variant_id TEXT NOT NULL REFERENCES voice_variants(id) ON DELETE RESTRICT,
    dataset_id TEXT NOT NULL,
    status TEXT NOT NULL CHECK (
        status IN ('pending', 'running', 'cancelling', 'cancelled', 'failed', 'complete')
    ),
    progress REAL NOT NULL DEFAULT 0 CHECK (progress >= 0 AND progress <= 1),
    config_json TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    error TEXT,
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    UNIQUE (id, variant_id),
    FOREIGN KEY (dataset_id, variant_id)
        REFERENCES dataset_versions(id, variant_id) ON DELETE RESTRICT
);

CREATE INDEX IF NOT EXISTS idx_training_runs_variant_status
ON training_runs(variant_id, status, created_at DESC);

CREATE TABLE IF NOT EXISTS checkpoints (
    id TEXT PRIMARY KEY,
    variant_id TEXT NOT NULL REFERENCES voice_variants(id) ON DELETE RESTRICT,
    training_run_id TEXT NOT NULL,
    epoch INTEGER NOT NULL,
    artifact_path TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL CHECK (status IN ('candidate', 'promoted', 'rejected')),
    metrics_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (training_run_id, epoch),
    UNIQUE (id, variant_id),
    FOREIGN KEY (training_run_id, variant_id)
        REFERENCES training_runs(id, variant_id) ON DELETE RESTRICT
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_one_promoted_checkpoint_per_variant
ON checkpoints(variant_id) WHERE status = 'promoted';

CREATE INDEX IF NOT EXISTS idx_checkpoints_variant_created
ON checkpoints(variant_id, created_at DESC);

CREATE TABLE IF NOT EXISTS generations (
    id TEXT PRIMARY KEY,
    variant_id TEXT NOT NULL REFERENCES voice_variants(id) ON DELETE RESTRICT,
    checkpoint_id TEXT NOT NULL,
    text TEXT NOT NULL,
    delivery_json TEXT NOT NULL,
    output_path TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL CHECK (status IN ('pending', 'running', 'failed', 'complete')),
    metrics_json TEXT NOT NULL DEFAULT '{}',
    error TEXT,
    created_at TEXT NOT NULL,
    finished_at TEXT,
    UNIQUE (id, variant_id),
    FOREIGN KEY (checkpoint_id, variant_id)
        REFERENCES checkpoints(id, variant_id) ON DELETE RESTRICT
);

CREATE INDEX IF NOT EXISTS idx_generations_variant_checkpoint
ON generations(variant_id, checkpoint_id, created_at DESC);
""" + CUSTOM_SCRIPT_SCHEMA


class VoiceStore:
    """App-friendly persistence API; every operation stays on the local filesystem."""

    def __init__(
        self,
        root: str | os.PathLike[str] | None = None,
        *,
        catalog: PromptCatalog | None = None,
    ) -> None:
        self.paths = StorePaths.under(root) if root is not None else default_store_paths()
        self.catalog = catalog or PromptCatalog.load()

    def _existing_schema_version(self) -> int:
        if not self.paths.database.is_file() or self.paths.database.stat().st_size == 0:
            return 0
        with sqlite3.connect(self.paths.database) as connection:
            user_tables = connection.execute(
                """SELECT COUNT(*) FROM sqlite_master
                   WHERE type='table' AND name NOT LIKE 'sqlite_%'"""
            ).fetchone()[0]
            table = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='app_meta'"
            ).fetchone()
            if table is None:
                if user_tables:
                    raise RuntimeError("Noice City database has no schema metadata")
                return 0
            row = connection.execute(
                "SELECT value FROM app_meta WHERE key='schema_version'"
            ).fetchone()
            if row is None:
                raise RuntimeError("Noice City database schema version is missing")
            return int(row[0])

    @staticmethod
    def _insert_default_variant(connection: sqlite3.Connection) -> None:
        now = _now()
        connection.execute(
            """
            INSERT INTO voice_variants(
                id, slug, name, description, state, is_default, metadata_json,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, 'active', 1, '{}', ?, ?)
            ON CONFLICT(id) DO NOTHING
            """,
            (
                DEFAULT_VARIANT_ID,
                DEFAULT_VARIANT_SLUG,
                DEFAULT_VARIANT_NAME,
                "Default lineage for recordings and persistent named-speaker checkpoints.",
                now,
                now,
            ),
        )
        default = connection.execute(
            "SELECT id FROM voice_variants WHERE is_default=1"
        ).fetchone()
        if default is None:
            connection.execute(
                "UPDATE voice_variants SET is_default=1, updated_at=? WHERE id=?",
                (now, DEFAULT_VARIANT_ID),
            )

    def _migrate_v1_to_v2(self) -> None:
        """Copy a complete v1 snapshot into a new v3 database, then atomically swap it.

        No managed audio, manifest, adapter, evaluation, or generation path is
        rewritten. The original SQLite database is retained as a consolidated
        schema-v1 backup for rollback.
        """

        database = self.paths.database
        temporary = database.with_name(f".{database.name}.schema-v3.{uuid.uuid4().hex}.partial")
        backup = database.with_name(f"{database.name}.schema-v1.backup")
        source: sqlite3.Connection | None = None
        destination: sqlite3.Connection | None = None
        try:
            source = sqlite3.connect(database, timeout=30)
            source.row_factory = sqlite3.Row
            source.execute("PRAGMA busy_timeout = 30000")
            checkpoint = source.execute("PRAGMA wal_checkpoint(FULL)").fetchone()
            if checkpoint is not None and int(checkpoint[0]) != 0:
                raise RuntimeError("Noice City database is busy; stop the local app before migrating")
            source.execute("BEGIN IMMEDIATE")
            if source.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise RuntimeError("Refusing to migrate an invalid schema-v1 SQLite database")
            if source.execute("PRAGMA foreign_key_check").fetchone() is not None:
                raise RuntimeError("Refusing to migrate schema-v1 data with broken relationships")

            destination = sqlite3.connect(temporary)
            destination.row_factory = sqlite3.Row
            destination.execute("PRAGMA foreign_keys = ON")
            destination.executescript(SCHEMA)
            destination.execute("BEGIN IMMEDIATE")
            self._insert_default_variant(destination)

            for row in source.execute("SELECT key, value FROM app_meta WHERE key<>'schema_version'"):
                destination.execute(
                    "INSERT INTO app_meta(key,value) VALUES (?,?)", (row["key"], row["value"])
                )
            destination.execute(
                "INSERT INTO app_meta(key,value) VALUES ('schema_version',?)",
                (str(SCHEMA_VERSION),),
            )
            destination.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

            prompt_columns = "id,catalog_version,kind,style,split,locked,ordinal,text"
            destination.executemany(
                f"INSERT INTO prompts({prompt_columns}) VALUES (?,?,?,?,?,?,?,?)",
                (tuple(row) for row in source.execute(f"SELECT {prompt_columns} FROM prompts")),
            )

            session_columns = "id,mode,status,created_at,ended_at,notes"
            destination.executemany(
                """INSERT INTO sessions(
                    id,variant_id,mode,status,created_at,ended_at,notes
                ) VALUES (?,?,?,?,?,?,?)""",
                (
                    (row["id"], DEFAULT_VARIANT_ID, *(row[name] for name in session_columns.split(",")[1:]))
                    for row in source.execute(f"SELECT {session_columns} FROM sessions")
                ),
            )

            recording_columns = (
                "id,session_id,prompt_id,origin,created_at,master_path,training_path,"
                "master_sha256,training_sha256,duration_seconds,expected_transcript,"
                "asr_transcript,confirmed_transcript,transcript_wer,transcript_status,"
                "qa_json,qa_passed,qa_needs_review,playback_confirmed,review_status,"
                "review_note,reviewed_at"
            )
            source_recording_columns = {
                row["name"] for row in source.execute("PRAGMA table_info(recordings)")
            }
            selected_recording_columns = [
                name for name in recording_columns.split(",") if name in source_recording_columns
            ]
            recording_rows = []
            for row in source.execute(
                f"SELECT {','.join(selected_recording_columns)} FROM recordings"
            ):
                values = dict(row)
                recording_rows.append(
                    tuple(
                        DEFAULT_VARIANT_ID if name == "variant_id"
                        else values.get(name, 0 if name == "playback_confirmed" else None)
                        for name in ("id", "variant_id", *recording_columns.split(",")[1:])
                    )
                )
            destination.executemany(
                f"INSERT INTO recordings(id,variant_id,{','.join(recording_columns.split(',')[1:])}) "
                f"VALUES ({','.join('?' for _ in range(23))})",
                recording_rows,
            )

            dataset_columns = (
                "id,name,purpose,state,created_at,manifest_path,manifest_sha256,"
                "recording_count,duration_seconds,catalog_version,metadata_json"
            )
            destination.executemany(
                f"INSERT INTO dataset_versions(id,variant_id,{','.join(dataset_columns.split(',')[1:])}) "
                f"VALUES ({','.join('?' for _ in range(12))})",
                (
                    (row["id"], DEFAULT_VARIANT_ID, *(row[name] for name in dataset_columns.split(",")[1:]))
                    for row in source.execute(f"SELECT {dataset_columns} FROM dataset_versions")
                ),
            )

            link_columns = (
                "dataset_id,recording_id,prompt_id,split,kind,style,training_text,"
                "frozen_audio_path,frozen_audio_sha256,source_master_sha256,qa_json"
            )
            destination.executemany(
                f"INSERT INTO dataset_recordings(dataset_id,recording_id,variant_id,{','.join(link_columns.split(',')[2:])}) "
                f"VALUES ({','.join('?' for _ in range(12))})",
                (
                    (row["dataset_id"], row["recording_id"], DEFAULT_VARIANT_ID,
                     *(row[name] for name in link_columns.split(",")[2:]))
                    for row in source.execute(f"SELECT {link_columns} FROM dataset_recordings")
                ),
            )

            run_columns = (
                "id,dataset_id,status,progress,config_json,metadata_json,error,"
                "created_at,started_at,finished_at"
            )
            destination.executemany(
                f"INSERT INTO training_runs(id,variant_id,{','.join(run_columns.split(',')[1:])}) "
                f"VALUES ({','.join('?' for _ in range(11))})",
                (
                    (row["id"], DEFAULT_VARIANT_ID, *(row[name] for name in run_columns.split(",")[1:]))
                    for row in source.execute(f"SELECT {run_columns} FROM training_runs")
                ),
            )

            checkpoint_columns = "id,training_run_id,epoch,artifact_path,status,metrics_json,created_at"
            destination.executemany(
                f"INSERT INTO checkpoints(id,variant_id,{','.join(checkpoint_columns.split(',')[1:])}) "
                f"VALUES ({','.join('?' for _ in range(8))})",
                (
                    (row["id"], DEFAULT_VARIANT_ID,
                     *(row[name] for name in checkpoint_columns.split(",")[1:]))
                    for row in source.execute(f"SELECT {checkpoint_columns} FROM checkpoints")
                ),
            )

            generation_columns = (
                "id,checkpoint_id,text,delivery_json,output_path,status,metrics_json,error,"
                "created_at,finished_at"
            )
            destination.executemany(
                f"INSERT INTO generations(id,variant_id,{','.join(generation_columns.split(',')[1:])}) "
                f"VALUES ({','.join('?' for _ in range(11))})",
                (
                    (row["id"], DEFAULT_VARIANT_ID,
                     *(row[name] for name in generation_columns.split(",")[1:]))
                    for row in source.execute(f"SELECT {generation_columns} FROM generations")
                ),
            )

            for table in (
                "sessions", "prompts", "recordings", "dataset_versions",
                "dataset_recordings", "training_runs", "checkpoints", "generations",
            ):
                old_count = source.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                new_count = destination.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                if old_count != new_count:
                    raise RuntimeError(f"schema migration row-count mismatch for {table}")
            if destination.execute("PRAGMA foreign_key_check").fetchone() is not None:
                raise RuntimeError("schema migration produced a foreign-key violation")
            destination.commit()
            if destination.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise RuntimeError("schema migration produced an invalid SQLite database")
            destination.close()
            destination = None

            if not backup.exists():
                backup_temporary = backup.with_name(f".{backup.name}.{uuid.uuid4().hex}.partial")
                # WAL is fully checkpointed and BEGIN IMMEDIATE still excludes
                # every writer, so this is one stable byte-for-byte v1 snapshot.
                shutil.copy2(database, backup_temporary)
                backup_temporary.chmod(0o600)
                with backup_temporary.open("rb") as backup_file:
                    os.fsync(backup_file.fileno())
                os.replace(backup_temporary, backup)
            source.rollback()
            source.close()
            source = None

            with temporary.open("rb") as migrated:
                os.fsync(migrated.fileno())
            temporary.chmod(0o600)
            for suffix in ("-wal", "-shm"):
                database.with_name(database.name + suffix).unlink(missing_ok=True)
            os.replace(temporary, database)
            database.chmod(0o600)
            directory_fd = os.open(database.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except Exception:
            if destination is not None:
                destination.close()
            if source is not None:
                source.rollback()
                source.close()
            temporary.unlink(missing_ok=True)
            raise

    def _migrate_v2_to_v3(self) -> None:
        """Install the additive custom-script schema under one rollback-safe transaction."""

        database = self.paths.database
        backup = database.with_name(f"{database.name}.schema-v2.backup")
        connection = sqlite3.connect(database, timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA busy_timeout = 30000")
            checkpoint = connection.execute("PRAGMA wal_checkpoint(FULL)").fetchone()
            if checkpoint is not None and int(checkpoint[0]) != 0:
                raise RuntimeError("Noice City database is busy; stop the local app before migrating")
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("BEGIN IMMEDIATE")
            if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise RuntimeError("Refusing to migrate an invalid schema-v2 SQLite database")
            if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
                raise RuntimeError("Refusing to migrate schema-v2 data with broken relationships")

            if not backup.exists():
                backup_temporary = backup.with_name(f".{backup.name}.{uuid.uuid4().hex}.partial")
                shutil.copy2(database, backup_temporary)
                backup_temporary.chmod(0o600)
                with backup_temporary.open("rb") as backup_file:
                    os.fsync(backup_file.fileno())
                os.replace(backup_temporary, backup)

            connection.execute(CUSTOM_SCRIPTS_TABLE_SQL)
            connection.execute(CUSTOM_SCRIPT_ITEMS_TABLE_SQL)
            connection.execute(CUSTOM_SCRIPT_INDEX_SQL)
            connection.execute(CUSTOM_SCRIPT_ACTIVE_HASH_INDEX_SQL)
            connection.execute(
                "UPDATE app_meta SET value=? WHERE key='schema_version'",
                (str(SCHEMA_VERSION),),
            )
            connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
                raise RuntimeError("custom-script migration produced a foreign-key violation")
            connection.commit()
            database.chmod(0o600)
            directory_fd = os.open(database.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> "VoiceStore":
        os.umask(0o077)
        for directory in (
            self.paths.root,
            self.paths.profiles,
            self.paths.recordings,
            self.paths.masters,
            self.paths.training_audio,
            self.paths.datasets,
            self.paths.adapters,
            self.paths.evaluations,
            self.paths.generations,
        ):
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            directory.chmod(0o700)
        version = self._existing_schema_version()
        if version == 1:
            self._migrate_v1_to_v2()
        elif version == 2:
            self._migrate_v2_to_v3()
        elif version not in {0, SCHEMA_VERSION}:
            raise RuntimeError(f"Unsupported Noice City schema version: {version}")
        with self._connect() as connection:
            connection.executescript(SCHEMA)
            self._insert_default_variant(connection)
            connection.execute("DROP INDEX IF EXISTS idx_one_promoted_checkpoint")
            connection.execute(
                "INSERT OR REPLACE INTO app_meta(key, value) VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )
            connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            for prompt in self.catalog.prompts:
                connection.execute(
                    """
                    INSERT INTO prompts(id, catalog_version, kind, style, split, locked, ordinal, text)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        catalog_version=excluded.catalog_version,
                        kind=excluded.kind,
                        style=excluded.style,
                        split=excluded.split,
                        locked=excluded.locked,
                        ordinal=excluded.ordinal,
                        text=excluded.text
                    """,
                    (
                        prompt.id,
                        self.catalog.version,
                        prompt.kind,
                        prompt.style,
                        prompt.split,
                        int(prompt.locked),
                        prompt.ordinal,
                        prompt.text,
                    ),
                )
        self.paths.database.chmod(0o600)
        return self

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.paths.database, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        connection.execute("PRAGMA journal_mode = WAL")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _row(row: sqlite3.Row | None, *, json_fields: tuple[str, ...] = ()) -> dict[str, Any] | None:
        if row is None:
            return None
        value = dict(row)
        for field in json_fields:
            value[field] = _json_object(value.get(field))
        for field in ("qa_passed", "qa_needs_review", "playback_confirmed", "locked", "is_default"):
            if field in value:
                value[field] = bool(value[field])
        return value

    @staticmethod
    def _validate_variant_slug(slug: str) -> str:
        value = slug.strip().lower()
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", value):
            raise ValueError("Variant slug must use lowercase letters, numbers, and single hyphens")
        if len(value) > 64:
            raise ValueError("Variant slug must contain at most 64 characters")
        return value

    def _resolve_variant(
        self,
        connection: sqlite3.Connection,
        variant_id: str | None,
        *,
        require_active: bool = False,
    ) -> sqlite3.Row:
        if variant_id is None:
            row = connection.execute(
                "SELECT * FROM voice_variants WHERE is_default=1"
            ).fetchone()
        else:
            row = connection.execute(
                "SELECT * FROM voice_variants WHERE id=?", (variant_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown voice variant: {variant_id or 'default'}")
        if require_active and row["state"] != "active":
            raise ValueError(f"Voice variant is archived: {row['id']}")
        return row

    def create_voice_variant(
        self,
        name: str,
        *,
        slug: str | None = None,
        description: str | None = None,
        make_default: bool = False,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        display_name = name.strip()
        if not display_name or len(display_name) > 80:
            raise ValueError("Variant name must contain 1–80 characters")
        slug_value = self._validate_variant_slug(
            slug or re.sub(r"[^a-z0-9]+", "-", display_name.casefold()).strip("-")
        )
        description_value = description.strip() if description and description.strip() else None
        if description_value and len(description_value) > 500:
            raise ValueError("Variant description must contain at most 500 characters")
        variant_id = f"variant_{uuid.uuid4().hex}"
        now = _now()
        with self._connect() as connection:
            if make_default:
                connection.execute("UPDATE voice_variants SET is_default=0, updated_at=?", (now,))
            try:
                connection.execute(
                    """
                    INSERT INTO voice_variants(
                        id,slug,name,description,state,is_default,metadata_json,created_at,updated_at
                    ) VALUES (?,?,?,?,'active',?,?,?,?)
                    """,
                    (
                        variant_id, slug_value, display_name, description_value,
                        int(make_default), _canonical_json(dict(metadata or {})), now, now,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError(f"Voice variant slug already exists: {slug_value}") from exc
            row = connection.execute(
                "SELECT * FROM voice_variants WHERE id=?", (variant_id,)
            ).fetchone()
        return self._row(row, json_fields=("metadata_json",)) or {}

    def list_voice_variants(self, *, include_archived: bool = True) -> list[dict[str, Any]]:
        query = "SELECT * FROM voice_variants"
        if not include_archived:
            query += " WHERE state='active'"
        query += " ORDER BY is_default DESC, created_at, id"
        with self._connect() as connection:
            rows = connection.execute(query).fetchall()
        return [self._row(row, json_fields=("metadata_json",)) or {} for row in rows]

    def get_voice_variant(self, variant_id: str | None = None) -> dict[str, Any]:
        with self._connect() as connection:
            row = self._resolve_variant(connection, variant_id)
        return self._row(row, json_fields=("metadata_json",)) or {}

    def update_voice_variant(
        self,
        variant_id: str,
        *,
        name: str | None = None,
        description: str | None = None,
        state: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if state is not None and state not in {"active", "archived"}:
            raise ValueError("Variant state must be active or archived")
        with self._connect() as connection:
            existing = self._resolve_variant(connection, variant_id)
            display_name = existing["name"] if name is None else name.strip()
            if not display_name or len(display_name) > 80:
                raise ValueError("Variant name must contain 1–80 characters")
            next_description = existing["description"] if description is None else (
                description.strip() or None
            )
            if next_description and len(next_description) > 500:
                raise ValueError("Variant description must contain at most 500 characters")
            next_state = state or existing["state"]
            if existing["is_default"] and next_state == "archived":
                raise ValueError("Choose another default variant before archiving this one")
            next_metadata = (
                _json_object(existing["metadata_json"]) if metadata is None else dict(metadata)
            )
            connection.execute(
                """UPDATE voice_variants SET name=?,description=?,state=?,metadata_json=?,updated_at=?
                   WHERE id=?""",
                (
                    display_name, next_description, next_state,
                    _canonical_json(next_metadata), _now(), variant_id,
                ),
            )
            row = connection.execute(
                "SELECT * FROM voice_variants WHERE id=?", (variant_id,)
            ).fetchone()
        return self._row(row, json_fields=("metadata_json",)) or {}

    def set_default_voice_variant(self, variant_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            variant = self._resolve_variant(connection, variant_id, require_active=True)
            now = _now()
            connection.execute("UPDATE voice_variants SET is_default=0, updated_at=?", (now,))
            connection.execute(
                "UPDATE voice_variants SET is_default=1, updated_at=? WHERE id=?",
                (now, variant["id"]),
            )
            row = connection.execute(
                "SELECT * FROM voice_variants WHERE id=?", (variant_id,)
            ).fetchone()
        return self._row(row, json_fields=("metadata_json",)) or {}

    @staticmethod
    def _custom_script_gate_seconds(
        connection: sqlite3.Connection,
        variant_id: str,
    ) -> float:
        return float(
            connection.execute(
                """
                SELECT COALESCE(SUM(r.duration_seconds), 0)
                FROM recordings r
                WHERE r.variant_id=? AND r.review_status='approved'
                  AND NOT EXISTS (
                      SELECT 1 FROM custom_script_items i WHERE i.prompt_id=r.prompt_id
                  )
                """,
                (variant_id,),
            ).fetchone()[0]
        )

    @classmethod
    def _require_custom_script_gate(
        cls,
        connection: sqlite3.Connection,
        variant_id: str,
    ) -> float:
        seconds = cls._custom_script_gate_seconds(connection, variant_id)
        if seconds < 120.0:
            raise ValueError(
                "Custom scripts unlock after 120 seconds of approved catalog recordings"
            )
        return seconds

    @staticmethod
    def _refresh_custom_script_state(
        connection: sqlite3.Connection,
        script_id: str,
    ) -> None:
        pending = connection.execute(
            "SELECT COUNT(*) FROM custom_script_items WHERE script_id=? AND status='pending'",
            (script_id,),
        ).fetchone()[0]
        now = _now()
        if pending:
            connection.execute(
                """UPDATE custom_scripts
                   SET state='active', updated_at=?, completed_at=NULL WHERE id=?""",
                (now, script_id),
            )
        else:
            connection.execute(
                """UPDATE custom_scripts
                   SET state='complete', updated_at=?, completed_at=COALESCE(completed_at, ?)
                   WHERE id=?""",
                (now, now, script_id),
            )

    @staticmethod
    def _custom_script_summary(
        connection: sqlite3.Connection,
        script_id: str,
        *,
        created: bool,
    ) -> dict[str, Any]:
        row = connection.execute(
            """
            SELECT s.*,
                COALESCE(SUM(CASE WHEN i.status='pending' THEN 1 ELSE 0 END), 0)
                    AS pending_count,
                COALESCE(SUM(CASE WHEN i.status='skipped' THEN 1 ELSE 0 END), 0)
                    AS skipped_count,
                COALESCE(SUM(CASE WHEN i.status='recorded' THEN 1 ELSE 0 END), 0)
                    AS recorded_count
            FROM custom_scripts s
            LEFT JOIN custom_script_items i ON i.script_id=s.id
            WHERE s.id=?
            GROUP BY s.id
            """,
            (script_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown custom script: {script_id}")
        value = dict(row)
        value["created"] = created
        value["item_count"] = int(value["item_count"])
        for field in ("pending_count", "skipped_count", "recorded_count"):
            value[field] = int(value[field])
        return value

    def create_custom_script(
        self,
        variant_id: str,
        text: str,
        title: str | None = None,
    ) -> dict[str, Any]:
        """Create one immutable, train-only recording pass for arbitrary source text.

        Retrying the same normalized source while its pass has pending items is
        idempotent. Once every item is recorded or skipped, the same source may
        be submitted again to deliberately create a fresh pass.
        """

        source_text = _normalize_custom_script(text)
        segments = _segment_custom_script(source_text)
        source_sha256 = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
        title_value = title.strip() if title and title.strip() else None
        if title_value and len(title_value) > 120:
            raise ValueError("Custom script title must contain at most 120 characters")
        now = _now()
        with self._connect() as connection:
            variant = self._resolve_variant(connection, variant_id, require_active=True)
            self._require_custom_script_gate(connection, variant["id"])
            placeholders = ",".join("?" for _ in segments)
            locked_match = connection.execute(
                f"""
                SELECT p.id FROM prompts p
                WHERE (p.locked=1 OR p.split='test') AND p.text IN ({placeholders})
                  AND NOT EXISTS (
                      SELECT 1 FROM custom_script_items i WHERE i.prompt_id=p.id
                  )
                LIMIT 1
                """,
                segments,
            ).fetchone()
            if locked_match is not None:
                raise ValueError("Custom script cannot duplicate a locked catalog prompt")
            existing = connection.execute(
                """SELECT id FROM custom_scripts
                   WHERE variant_id=? AND source_sha256=? AND state='active'""",
                (variant["id"], source_sha256),
            ).fetchone()
            if existing is not None:
                return self._custom_script_summary(
                    connection, existing["id"], created=False
                )

            script_id = f"script_{uuid.uuid4().hex}"
            connection.execute(
                """
                INSERT INTO custom_scripts(
                    id,variant_id,title,source_text,source_sha256,state,item_count,
                    created_at,updated_at
                ) VALUES (?,?,?,?,?,'active',?,?,?)
                """,
                (
                    script_id,
                    variant["id"],
                    title_value,
                    source_text,
                    source_sha256,
                    len(segments),
                    now,
                    now,
                ),
            )
            for position, segment in enumerate(segments):
                prompt_id = f"custom_prompt_{uuid.uuid4().hex}"
                connection.execute(
                    """
                    INSERT INTO prompts(
                        id,catalog_version,kind,style,split,locked,ordinal,text
                    ) VALUES (?,?,'guided','neutral','train',0,?,?)
                    """,
                    (prompt_id, self.catalog.version, position, segment),
                )
                connection.execute(
                    """
                    INSERT INTO custom_script_items(
                        script_id,variant_id,prompt_id,position,status,created_at,updated_at
                    ) VALUES (?,?,?,?,'pending',?,?)
                    """,
                    (script_id, variant["id"], prompt_id, position, now, now),
                )
            return self._custom_script_summary(connection, script_id, created=True)

    def get_prompt(
        self,
        prompt_id: str,
        variant_id: str | None = None,
    ) -> dict[str, Any]:
        """Return a catalog or custom prompt, enforcing custom variant ownership."""

        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT p.*,
                    i.script_id AS custom_script_id,
                    i.variant_id AS custom_variant_id,
                    i.position AS custom_position,
                    i.status AS custom_status,
                    i.recording_id AS custom_recording_id,
                    s.title AS custom_script_title,
                    s.item_count AS custom_script_item_count
                FROM prompts p
                LEFT JOIN custom_script_items i ON i.prompt_id=p.id
                LEFT JOIN custom_scripts s
                    ON s.id=i.script_id AND s.variant_id=i.variant_id
                WHERE p.id=?
                """,
                (prompt_id,),
            ).fetchone()
        value = self._row(row)
        if value is None:
            raise KeyError(f"Unknown prompt: {prompt_id}")
        is_custom = value["custom_script_id"] is not None
        if is_custom and variant_id is not None and value["custom_variant_id"] != variant_id:
            raise ValueError("Custom prompt belongs to another voice variant")
        value["is_custom"] = is_custom
        value["variant_id"] = value["custom_variant_id"] if is_custom else None
        if is_custom:
            value["custom_position_number"] = int(value["custom_position"]) + 1
            value["custom_script_item_count"] = int(value["custom_script_item_count"])
        return value

    def next_custom_script_prompt(self, variant_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            variant = self._resolve_variant(connection, variant_id, require_active=True)
            self._require_custom_script_gate(connection, variant["id"])
            row = connection.execute(
                """
                SELECT i.prompt_id, i.script_id
                FROM custom_script_items i
                JOIN custom_scripts s
                    ON s.id=i.script_id AND s.variant_id=i.variant_id
                WHERE i.variant_id=? AND i.status='pending'
                ORDER BY s.created_at, s.id, i.position
                LIMIT 1
                """,
                (variant["id"],),
            ).fetchone()
            if row is None:
                return None
            summary = self._custom_script_summary(
                connection, row["script_id"], created=False
            )
            prompt_id = row["prompt_id"]
        prompt = self.get_prompt(prompt_id, variant_id=variant_id)
        prompt["custom_script_pending_count"] = summary["pending_count"]
        prompt["custom_script_recorded_count"] = summary["recorded_count"]
        prompt["custom_script_skipped_count"] = summary["skipped_count"]
        return prompt

    def skip_custom_script_prompt(
        self,
        variant_id: str,
        prompt_id: str,
    ) -> dict[str, Any]:
        with self._connect() as connection:
            variant = self._resolve_variant(connection, variant_id, require_active=True)
            self._require_custom_script_gate(connection, variant["id"])
            row = connection.execute(
                "SELECT script_id,variant_id,status FROM custom_script_items WHERE prompt_id=?",
                (prompt_id,),
            ).fetchone()
            if row is None:
                raise ValueError("Prompt is not a custom-script prompt")
            if row["variant_id"] != variant["id"]:
                raise ValueError("Custom prompt belongs to another voice variant")
            if row["status"] == "recorded":
                raise ValueError("A recorded custom prompt cannot be skipped")
            if row["status"] == "pending":
                connection.execute(
                    """UPDATE custom_script_items SET status='skipped',updated_at=?
                       WHERE prompt_id=? AND variant_id=? AND status='pending'""",
                    (_now(), prompt_id, variant["id"]),
                )
                self._refresh_custom_script_state(connection, row["script_id"])
        return self.get_prompt(prompt_id, variant_id=variant_id)

    def status(self, variant_id: str | None = None) -> dict[str, Any]:
        with self._connect() as connection:
            variant = self._resolve_variant(connection, variant_id)
            summary = connection.execute(
                """
                SELECT
                    COUNT(*) AS recordings,
                    COALESCE(SUM(duration_seconds), 0) AS total_seconds,
                    COALESCE(SUM(CASE WHEN review_status='approved' THEN duration_seconds ELSE 0 END), 0)
                        AS accepted_seconds,
                    COALESCE(SUM(CASE WHEN review_status='pending' THEN 1 ELSE 0 END), 0)
                        AS pending_review
                FROM recordings WHERE variant_id=?
                """,
                (variant["id"],),
            ).fetchone()
            datasets = connection.execute(
                "SELECT COUNT(*) FROM dataset_versions WHERE variant_id=?", (variant["id"],)
            ).fetchone()[0]
            runs = connection.execute(
                """SELECT COUNT(*) FROM training_runs
                   WHERE variant_id=? AND status IN ('pending','running','cancelling')""",
                (variant["id"],),
            ).fetchone()[0]
            promoted = connection.execute(
                "SELECT id FROM checkpoints WHERE variant_id=? AND status='promoted'",
                (variant["id"],),
            ).fetchone()
            custom_scripts = connection.execute(
                "SELECT COUNT(*) FROM custom_scripts WHERE variant_id=?",
                (variant["id"],),
            ).fetchone()[0]
            pending_custom_prompts = connection.execute(
                """SELECT COUNT(*) FROM custom_script_items
                   WHERE variant_id=? AND status='pending'""",
                (variant["id"],),
            ).fetchone()[0]
            custom_script_gate_seconds = self._custom_script_gate_seconds(
                connection, variant["id"]
            )
        return {
            "variant_id": variant["id"],
            "root": str(self.paths.root),
            "recordings": int(summary["recordings"]),
            "total_seconds": float(summary["total_seconds"]),
            "accepted_seconds": float(summary["accepted_seconds"]),
            "accepted_minutes": round(float(summary["accepted_seconds"]) / 60.0, 2),
            "pending_review": int(summary["pending_review"]),
            "dataset_versions": int(datasets),
            "active_training_runs": int(runs),
            "promoted_checkpoint_id": promoted["id"] if promoted else None,
            "custom_scripts": int(custom_scripts),
            "pending_custom_prompts": int(pending_custom_prompts),
            "custom_script_gate_seconds": custom_script_gate_seconds,
            "custom_script_unlocked": custom_script_gate_seconds >= 120.0,
            "prompt_catalog": self.catalog.summary(),
        }

    def create_session(
        self,
        mode: str = "hybrid",
        notes: str | None = None,
        *,
        variant_id: str | None = None,
    ) -> dict[str, Any]:
        if mode not in {"guided", "interview", "hybrid"}:
            raise ValueError("Session mode must be guided, interview, or hybrid")
        session_id = _id("session")
        with self._connect() as connection:
            variant = self._resolve_variant(connection, variant_id, require_active=True)
            connection.execute(
                """INSERT INTO sessions(id, variant_id, mode, created_at, notes)
                   VALUES (?, ?, ?, ?, ?)""",
                (session_id, variant["id"], mode, _now(), notes),
            )
            row = connection.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
        return self._row(row) or {}

    def get_session(
        self,
        session_id: str,
        *,
        variant_id: str | None = None,
    ) -> dict[str, Any]:
        clause = "id=?"
        arguments: tuple[object, ...] = (session_id,)
        if variant_id is not None:
            clause += " AND variant_id=?"
            arguments = (session_id, variant_id)
        with self._connect() as connection:
            row = connection.execute(
                f"SELECT * FROM sessions WHERE {clause}", arguments
            ).fetchone()
        value = self._row(row)
        if value is None:
            raise KeyError(f"Unknown session: {session_id}")
        return value

    def end_session(self, session_id: str, status: str = "complete") -> dict[str, Any]:
        if status not in {"complete", "abandoned"}:
            raise ValueError("Ended session status must be complete or abandoned")
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE sessions SET status=?, ended_at=? WHERE id=? AND status='active'",
                (status, _now(), session_id),
            )
            if not cursor.rowcount:
                raise KeyError(f"No active session: {session_id}")
            row = connection.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
        return self._row(row) or {}

    def next_prompt(
        self,
        mode: str = "hybrid",
        *,
        session_id: str | None = None,
        split: str | None = None,
        exclude_prompt_ids: tuple[str, ...] = (),
    ) -> dict[str, Any] | None:
        if mode not in {"guided", "interview", "hybrid"}:
            raise ValueError("Prompt mode must be guided, interview, or hybrid")
        with self._connect() as connection:
            excluded: set[str] = set()
            counts = {"guided": 0, "interview": 0}
            split_counts = {"train": 0, "validation": 0, "test": 0}
            style_counts = {"neutral": 0, "expressive": 0}
            if session_id:
                session = connection.execute(
                    """SELECT s.status, v.state AS variant_state
                       FROM sessions s JOIN voice_variants v ON v.id=s.variant_id
                       WHERE s.id=?""",
                    (session_id,),
                ).fetchone()
                if session is None:
                    raise KeyError(f"Unknown session: {session_id}")
                if session["status"] != "active":
                    raise ValueError(f"Session is not active: {session_id}")
                if session["variant_state"] != "active":
                    raise ValueError(f"Session voice variant is archived: {session_id}")
                rows = connection.execute(
                    """
                    SELECT r.prompt_id, p.kind, p.split, p.style FROM recordings r
                    JOIN prompts p ON p.id=r.prompt_id WHERE r.session_id=?
                    """,
                    (session_id,),
                ).fetchall()
                excluded = {row["prompt_id"] for row in rows}
                for row in rows:
                    counts[row["kind"]] += 1
                    split_counts[row["split"]] += 1
                    style_counts[row["style"]] += 1
            excluded.update(exclude_prompt_ids)
        kind = mode
        if mode == "hybrid":
            # Four guided takes followed by one natural answer keeps a stable 80/20 mix.
            kind = "interview" if counts["guided"] >= 4 * (counts["interview"] + 1) else "guided"
        if split is None:
            split_position = sum(split_counts.values()) % 10
            split = "train" if split_position < 8 else "validation" if split_position == 8 else "test"
        style = (
            "expressive"
            if style_counts["neutral"] >= 9 * (style_counts["expressive"] + 1)
            else "neutral"
        )
        prompt = self.catalog.select(
            split=split, kind=kind, style=style,
            exclude=excluded,
            chooser_seed=session_id or "global",
        )
        # Small expressive subgroups do not necessarily contain every
        # split/kind combination. Keep the immutable split and requested prompt
        # kind, then make up the 90/10 style deficit on the next eligible take.
        if prompt is None:
            prompt = self.catalog.select(
                split=split,
                kind=kind,
                exclude=excluded,
                chooser_seed=f"{session_id}:style-fallback:{len(excluded)}",
            )
        if prompt is None and excluded:
            prompt = self.catalog.select(
                split=split,
                kind=kind,
                chooser_seed=f"{session_id}:repeat:{len(excluded)}",
            )
        return prompt.to_dict() if prompt else None

    def _assert_managed_recording_path(self, path: Path) -> Path:
        resolved = path.expanduser().resolve(strict=True)
        if not resolved.is_relative_to(self.paths.recordings.resolve()):
            raise ValueError("Only fresh audio created inside Noice City may be registered")
        return resolved

    def add_recording(
        self,
        prompt_id: str,
        *,
        session_id: str | None = None,
        master_bytes: bytes | None = None,
        master_path: str | os.PathLike[str] | None = None,
        train_path: str | os.PathLike[str] | None = None,
        transcript: str | None = None,
        qa: AudioQualityReport | Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Intake one fresh take and preserve both immutable audio representations.

        New callers should pass ``master_bytes``. ``master_path``/``train_path``
        exist for the local capture worker, but are accepted only when both files
        are already within this store's private recording directory.
        """

        if session_id is None:
            prompt = self.get_prompt(prompt_id)
            session_id = self.create_session(
                prompt["kind"],
                variant_id=prompt["variant_id"] if prompt["is_custom"] else None,
            )["id"]
        else:
            session_value = self.get_session(session_id)
            if session_value["status"] != "active":
                raise KeyError(f"No active session: {session_id}")
            prompt = self.get_prompt(prompt_id, variant_id=session_value["variant_id"])
        if prompt["is_custom"] and prompt["custom_status"] != "pending":
            raise ValueError("Custom prompt has already been recorded or skipped")
        recording_id = _id("rec")
        if master_bytes is not None:
            if master_path is not None or train_path is not None:
                raise ValueError("Pass master bytes or managed paths, not both")
            master = self.paths.masters / f"{recording_id}.wav"
            training = self.paths.training_audio / f"{recording_id}.wav"
            quality = save_master_wav(master_bytes, master)
            try:
                derive_training_wav(master, training)
            except Exception:
                master.unlink(missing_ok=True)
                raise
        else:
            if master_path is None or train_path is None:
                raise ValueError("A fresh master upload or both managed audio paths are required")
            master = self._assert_managed_recording_path(Path(master_path))
            training = self._assert_managed_recording_path(Path(train_path))
            quality = analyze_master_wav(master)
        if qa is not None:
            qa_dict = qa.to_dict() if isinstance(qa, AudioQualityReport) else dict(qa)
        else:
            qa_dict = quality.to_dict()
        # Locally calculated audio invariants remain authoritative even if a VAD/ASR
        # worker supplies additional QA fields.
        qa_dict["audio"] = quality.to_dict()
        qa_passed = quality.passed and bool(qa_dict.get("passed", True))
        qa_needs_review = quality.needs_review or bool(qa_dict.get("needs_review", False))
        expected = prompt["text"] if prompt["kind"] == "guided" else None
        transcript = transcript.strip() if transcript and transcript.strip() else None
        wer: float | None = None
        if transcript is None:
            transcript_status = "pending"
        elif prompt["kind"] == "interview":
            transcript_status = "review_required"
        else:
            wer = word_error_rate(prompt["text"], transcript)
            transcript_status = "verified" if wer <= 0.10 else "review_required"
            if wer > 0.10:
                qa_needs_review = True

        try:
            with self._connect() as connection:
                session = connection.execute(
                    """SELECT s.status, s.variant_id, v.state AS variant_state
                       FROM sessions s JOIN voice_variants v ON v.id=s.variant_id
                       WHERE s.id=?""",
                    (session_id,),
                ).fetchone()
                if session is None or session["status"] != "active":
                    raise KeyError(f"No active session: {session_id}")
                if session["variant_state"] != "active":
                    raise ValueError(f"Session voice variant is archived: {session_id}")
                custom_item = connection.execute(
                    """SELECT script_id,variant_id,status FROM custom_script_items
                       WHERE prompt_id=?""",
                    (prompt_id,),
                ).fetchone()
                if custom_item is not None:
                    if custom_item["variant_id"] != session["variant_id"]:
                        raise ValueError("Custom prompt belongs to another voice variant")
                    self._require_custom_script_gate(connection, session["variant_id"])
                    if custom_item["status"] != "pending":
                        raise ValueError("Custom prompt has already been recorded or skipped")
                connection.execute(
                    """
                    INSERT INTO recordings(
                        id, variant_id, session_id, prompt_id, created_at, master_path, training_path,
                        master_sha256, training_sha256, duration_seconds, expected_transcript,
                        asr_transcript, transcript_wer, transcript_status, qa_json, qa_passed,
                        qa_needs_review
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        recording_id,
                        session["variant_id"],
                        session_id,
                        prompt_id,
                        _now(),
                        str(master),
                        str(training),
                        _sha256(master),
                        _sha256(training),
                        quality.metrics.duration_seconds,
                        expected,
                        transcript,
                        wer,
                        transcript_status,
                        _canonical_json(qa_dict),
                        int(qa_passed),
                        int(qa_needs_review),
                    ),
                )
                if custom_item is not None:
                    updated = connection.execute(
                        """UPDATE custom_script_items
                           SET status='recorded',recording_id=?,updated_at=?
                           WHERE prompt_id=? AND variant_id=? AND status='pending'""",
                        (recording_id, _now(), prompt_id, session["variant_id"]),
                    )
                    if updated.rowcount != 1:
                        raise RuntimeError("Custom prompt state changed during recording intake")
                    self._refresh_custom_script_state(connection, custom_item["script_id"])
                row = connection.execute(
                    "SELECT * FROM recordings WHERE id=?", (recording_id,)
                ).fetchone()
        except Exception:
            if master_bytes is not None:
                master.unlink(missing_ok=True)
                training.unlink(missing_ok=True)
            raise
        return self._row(row, json_fields=("qa_json",)) or {}

    def list_recordings(
        self,
        status: str | None = None,
        *,
        variant_id: str | None = None,
    ) -> list[dict[str, Any]]:
        if status is not None and status not in RECORDING_STATUSES:
            raise ValueError(f"Unknown recording status: {status}")
        query = """
            SELECT r.*, p.kind, p.style, p.split, p.text AS prompt_text
            FROM recordings r JOIN prompts p ON p.id=r.prompt_id
        """
        with self._connect() as connection:
            variant = self._resolve_variant(connection, variant_id)
        clauses = ["r.variant_id=?"]
        arguments: list[object] = [variant["id"]]
        if status:
            clauses.append("r.review_status=?")
            arguments.append(status)
        query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY r.created_at DESC"
        with self._connect() as connection:
            rows = connection.execute(query, tuple(arguments)).fetchall()
        return [self._row(row, json_fields=("qa_json",)) or {} for row in rows]

    def get_recording(
        self,
        recording_id: str,
        *,
        variant_id: str | None = None,
    ) -> dict[str, Any]:
        clause = "r.id=?"
        arguments: tuple[object, ...] = (recording_id,)
        if variant_id is not None:
            clause += " AND r.variant_id=?"
            arguments = (recording_id, variant_id)
        with self._connect() as connection:
            row = connection.execute(
                f"""
                SELECT r.*, p.kind, p.style, p.split, p.text AS prompt_text
                FROM recordings r JOIN prompts p ON p.id=r.prompt_id WHERE {clause}
                """,
                arguments,
            ).fetchone()
        value = self._row(row, json_fields=("qa_json",))
        if value is None:
            raise KeyError(f"Unknown recording: {recording_id}")
        return value

    def update_recording(
        self,
        recording_id: str,
        status: str,
        transcript: str | None = None,
        *,
        note: str | None = None,
        override_qa: bool = False,
        playback_confirmed: bool = False,
    ) -> dict[str, Any]:
        if status not in RECORDING_STATUSES:
            raise ValueError(f"Unknown recording status: {status}")
        confirmed = transcript.strip() if transcript and transcript.strip() else None
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM recordings WHERE id=?", (recording_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown recording: {recording_id}")
            custom_item = connection.execute(
                """SELECT script_id,status,recording_id FROM custom_script_items
                   WHERE prompt_id=?""",
                (row["prompt_id"],),
            ).fetchone()
            transcript_status = row["transcript_status"]
            if confirmed:
                transcript_status = "confirmed"
            if status == "approved":
                if custom_item is not None and not (
                    confirmed or row["confirmed_transcript"]
                ):
                    raise ValueError(
                        "Custom-script recordings require an explicitly confirmed transcript"
                    )
                if not row["qa_passed"] and not override_qa:
                    raise ValueError("Failed audio QA requires an explicit override before approval")
                requires_playback = bool(row["qa_needs_review"]) or row["transcript_status"] in {
                    "pending", "review_required"
                }
                if requires_playback and not (row["playback_confirmed"] or playback_confirmed):
                    raise ValueError("Uncertain recordings must be played back before approval")
                if transcript_status in {"pending", "review_required"}:
                    raise ValueError("Playback review and an explicit confirmed transcript are required")
            connection.execute(
                """
                UPDATE recordings SET review_status=?, confirmed_transcript=COALESCE(?, confirmed_transcript),
                    transcript_status=?, playback_confirmed=?, review_note=?, reviewed_at=? WHERE id=?
                """,
                (
                    status,
                    confirmed,
                    transcript_status,
                    int(bool(row["playback_confirmed"]) or playback_confirmed),
                    note,
                    _now(),
                    recording_id,
                ),
            )
            if status in {"quarantined", "rejected"}:
                linked_item = connection.execute(
                    """SELECT script_id FROM custom_script_items
                       WHERE recording_id=? AND status='recorded'""",
                    (recording_id,),
                ).fetchone()
                if linked_item is not None:
                    connection.execute(
                        """UPDATE custom_script_items
                           SET status='pending',recording_id=NULL,updated_at=?
                           WHERE recording_id=? AND status='recorded'""",
                        (_now(), recording_id),
                    )
                    self._refresh_custom_script_state(connection, linked_item["script_id"])
            updated = connection.execute(
                "SELECT * FROM recordings WHERE id=?", (recording_id,)
            ).fetchone()
        return self._row(updated, json_fields=("qa_json",)) or {}

    def bulk_update_recordings(
        self,
        recording_ids: list[str] | tuple[str, ...],
        status: str,
        *,
        transcripts: Mapping[str, str] | None = None,
        confirm_review: bool = False,
        override_failed_qa: bool = False,
        variant_id: str | None = None,
    ) -> dict[str, Any]:
        """Apply one explicit review decision to an exact recording selection."""

        if status not in {"approved", "quarantined", "rejected"}:
            raise ValueError("Bulk status must be approved, quarantined, or rejected")
        unique_ids = list(dict.fromkeys(str(value) for value in recording_ids))
        if not unique_ids:
            raise ValueError("Select at least one recording")
        if len(unique_ids) > 500:
            raise ValueError("A bulk review may contain at most 500 recordings")
        transcript_values = dict(transcripts or {})
        updated: list[dict[str, Any]] = []
        overridden_ids: list[str] = []
        skipped: list[dict[str, str]] = []
        for recording_id in unique_ids:
            try:
                recording = self.get_recording(recording_id, variant_id=variant_id)
            except KeyError:
                skipped.append({"id": recording_id, "reason": "unknown_recording"})
                continue
            transcript: str | None = None
            is_qa_override = False
            if status == "approved":
                if not confirm_review:
                    raise ValueError("Bulk approval requires explicit review confirmation")
                if not recording["qa_passed"] and not override_failed_qa:
                    skipped.append({"id": recording_id, "reason": "failed_audio_qa"})
                    continue
                is_qa_override = not recording["qa_passed"]
                transcript = (
                    transcript_values.get(recording_id)
                    or recording.get("confirmed_transcript")
                    or recording.get("asr_transcript")
                )
                if not transcript or not str(transcript).strip():
                    skipped.append({"id": recording_id, "reason": "missing_transcript"})
                    continue
            try:
                updated_recording = self.update_recording(
                        recording_id,
                        status,
                        transcript=str(transcript) if transcript else None,
                        override_qa=override_failed_qa and status == "approved",
                        playback_confirmed=confirm_review and status == "approved",
                        note=(
                            "Approved in explicit batch review with QA override"
                            if status == "approved" and is_qa_override
                            else f"{status.capitalize()} in explicit batch review"
                        ),
                    )
                updated.append(updated_recording)
                if is_qa_override:
                    overridden_ids.append(recording_id)
            except ValueError as exc:
                skipped.append({"id": recording_id, "reason": str(exc)})
        return {
            "status": status,
            "updated_count": len(updated),
            "overridden_count": len(overridden_ids),
            "overridden_ids": overridden_ids,
            "skipped_count": len(skipped),
            "updated_ids": [row["id"] for row in updated],
            "skipped": skipped,
        }

    def approve_all_pending(self, *, variant_id: str | None = None) -> dict[str, Any]:
        """Backward-compatible explicit batch approval for all pending takes."""

        pending_ids = [
            row["id"] for row in self.list_recordings("pending", variant_id=variant_id)
        ]
        result = self.bulk_update_recordings(
            pending_ids,
            "approved",
            confirm_review=True,
            variant_id=variant_id,
        )
        return {
            **result,
            "approved_count": result["updated_count"],
            "approved_ids": result["updated_ids"],
        }

    def freeze_dataset(
        self,
        name: str,
        *,
        purpose: str = "full",
        minimum_duration_seconds: float | None = None,
        enforce_distribution: bool = True,
        variant_id: str | None = None,
    ) -> dict[str, Any]:
        if purpose not in {"pilot", "full"}:
            raise ValueError("Dataset purpose must be pilot or full")
        minimum = minimum_duration_seconds
        if minimum is None:
            minimum = 120.0 if purpose == "pilot" else 1800.0
        with self._connect() as connection:
            variant_row = self._resolve_variant(connection, variant_id, require_active=True)
            rows = connection.execute(
                """
                SELECT r.*, p.kind, p.style, p.split, p.text AS prompt_text,
                       i.script_id AS custom_script_id
                FROM recordings r
                JOIN prompts p ON p.id=r.prompt_id
                LEFT JOIN custom_script_items i ON i.prompt_id=r.prompt_id
                WHERE r.variant_id=? AND r.review_status='approved'
                ORDER BY r.created_at, r.id
                """,
                (variant_row["id"],),
            ).fetchall()
        if not rows:
            raise ValueError("No approved fresh recordings are available")
        duration = sum(float(row["duration_seconds"]) for row in rows)
        if duration < minimum:
            raise ValueError(
                f"Approved audio is {duration:.1f}s; this {purpose} dataset requires {minimum:.1f}s"
            )
        catalog_rows = [row for row in rows if row["custom_script_id"] is None]
        custom_rows = [row for row in rows if row["custom_script_id"] is not None]
        guided = sum(row["kind"] == "guided" for row in catalog_rows)
        neutral = sum(row["style"] == "neutral" for row in catalog_rows)
        split_counts = {
            split: sum(row["split"] == split for row in rows)
            for split in ("train", "validation", "test")
        }
        catalog_split_counts = {
            split: sum(row["split"] == split for row in catalog_rows)
            for split in ("train", "validation", "test")
        }
        if split_counts["train"] < 1 or split_counts["validation"] < 1 or split_counts["test"] < 2:
            raise ValueError(
                "A frozen dataset needs training and validation takes plus at least two locked test takes"
            )
        if purpose == "full" and enforce_distribution:
            if not catalog_rows:
                raise ValueError("A full dataset needs catalog recordings for coverage checks")
            guided_ratio = guided / len(catalog_rows)
            neutral_ratio = neutral / len(catalog_rows)
            if abs(guided_ratio - 0.80) > 0.05:
                raise ValueError("A full dataset must remain within five points of the 80/20 prompt mix")
            if abs(neutral_ratio - 0.90) > 0.05:
                raise ValueError("A full dataset must remain within five points of the 90/10 delivery mix")
            for split, target in (("train", 0.80), ("validation", 0.10), ("test", 0.10)):
                if abs(catalog_split_counts[split] / len(catalog_rows) - target) > 0.05:
                    raise ValueError(
                        "A full dataset must remain within five points of the 80/10/10 split"
                    )

        dataset_id = _id("dataset")
        dataset_directory = self.paths.datasets / dataset_id
        audio_directory = dataset_directory / "audio"
        dataset_directory.mkdir(parents=True, exist_ok=False, mode=0o700)
        audio_directory.mkdir(mode=0o700)
        entries: list[dict[str, Any]] = []
        try:
            for index, row in enumerate(rows):
                source = self._assert_managed_recording_path(Path(row["training_path"]))
                if _sha256(source) != row["training_sha256"]:
                    raise ValueError(f"Recording derivative changed after intake: {row['id']}")
                target = audio_directory / f"{index:05d}_{row['id']}.wav"
                shutil.copyfile(source, target)
                target.chmod(0o400)
                is_custom = row["custom_script_id"] is not None
                training_text = row["confirmed_transcript"] if is_custom else (
                    row["confirmed_transcript"]
                    or row["expected_transcript"]
                    or row["asr_transcript"]
                )
                if not training_text:
                    reason = (
                        "Custom recording lacks an explicitly confirmed transcript"
                        if is_custom else "Recording lacks a reviewed training transcript"
                    )
                    raise ValueError(f"{reason}: {row['id']}")
                entries.append(
                    {
                        "recording_id": row["id"],
                        "prompt_id": row["prompt_id"],
                        "split": row["split"],
                        "kind": row["kind"],
                        "style": row["style"],
                        "supplemental_custom": is_custom,
                        "custom_script_id": row["custom_script_id"],
                        "training_text": training_text,
                        "audio": str(target.relative_to(dataset_directory)),
                        "audio_sha256": _sha256(target),
                        "source_master_sha256": row["master_sha256"],
                        "duration_seconds": row["duration_seconds"],
                        "qa": _json_object(row["qa_json"]),
                    }
                )
            catalog_count = len(catalog_rows)
            supplemental_custom_duration = sum(
                float(row["duration_seconds"]) for row in custom_rows
            )
            metadata = {
                "guided_ratio": guided / catalog_count if catalog_count else 0.0,
                "interview_ratio": 1 - guided / catalog_count if catalog_count else 0.0,
                "neutral_ratio": neutral / catalog_count if catalog_count else 0.0,
                "expressive_ratio": 1 - neutral / catalog_count if catalog_count else 0.0,
                "splits": catalog_split_counts,
                "overall_splits": split_counts,
                "catalog_recording_count": catalog_count,
                "supplemental_custom_count": len(custom_rows),
                "supplemental_custom_duration_seconds": supplemental_custom_duration,
            }
            manifest = {
                "schema_version": 2,
                "id": dataset_id,
                "variant_id": variant_row["id"],
                "name": name.strip() or dataset_id,
                "purpose": purpose,
                "state": "frozen",
                "created_at": _now(),
                "profile": PROFILE_ID,
                "voice_variant": {
                    "variant_id": variant_row["id"],
                    "slug": variant_row["slug"],
                    "display_name": variant_row["name"],
                    "profile": PROFILE_ID,
                    "language": "english",
                },
                "source_policy": "fresh_voice_trainer_recordings_only",
                "prompt_catalog_version": self.catalog.version,
                "recording_count": len(entries),
                "duration_seconds": duration,
                "supplemental_custom_count": len(custom_rows),
                "supplemental_custom_duration_seconds": supplemental_custom_duration,
                "distribution": metadata,
                "recordings": entries,
            }
            manifest_path = dataset_directory / "manifest.json"
            payload = (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(
                "utf-8"
            )
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=".manifest.", dir=dataset_directory
            )
            with os.fdopen(descriptor, "wb") as output:
                output.write(payload)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary_name, manifest_path)
            manifest_path.chmod(0o400)
            manifest_hash = _sha256(manifest_path)
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO dataset_versions(
                        id, variant_id, name, purpose, created_at, manifest_path, manifest_sha256,
                        recording_count, duration_seconds, catalog_version, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        dataset_id,
                        variant_row["id"],
                        manifest["name"],
                        purpose,
                        manifest["created_at"],
                        str(manifest_path),
                        manifest_hash,
                        len(entries),
                        duration,
                        self.catalog.version,
                        _canonical_json(metadata),
                    ),
                )
                for entry in entries:
                    connection.execute(
                        """
                        INSERT INTO dataset_recordings(
                            dataset_id, recording_id, variant_id, prompt_id, split, kind, style,
                            training_text, frozen_audio_path, frozen_audio_sha256,
                            source_master_sha256, qa_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            dataset_id,
                            entry["recording_id"],
                            variant_row["id"],
                            entry["prompt_id"],
                            entry["split"],
                            entry["kind"],
                            entry["style"],
                            entry["training_text"],
                            str(dataset_directory / entry["audio"]),
                            entry["audio_sha256"],
                            entry["source_master_sha256"],
                            _canonical_json(entry["qa"]),
                        ),
                    )
            return self.dataset(dataset_id)
        except Exception:
            for child in dataset_directory.rglob("*"):
                if child.is_file():
                    child.chmod(0o600)
            shutil.rmtree(dataset_directory, ignore_errors=True)
            raise

    def dataset(
        self,
        dataset_id: str,
        *,
        variant_id: str | None = None,
    ) -> dict[str, Any]:
        clause = "id=?"
        arguments: tuple[object, ...] = (dataset_id,)
        if variant_id is not None:
            clause += " AND variant_id=?"
            arguments = (dataset_id, variant_id)
        with self._connect() as connection:
            row = connection.execute(
                f"SELECT * FROM dataset_versions WHERE {clause}", arguments
            ).fetchone()
        value = self._row(row, json_fields=("metadata_json",))
        if value is None:
            raise KeyError(f"Unknown dataset: {dataset_id}")
        manifest_path = Path(value["manifest_path"])
        value["checksum_valid"] = manifest_path.exists() and _sha256(manifest_path) == value["manifest_sha256"]
        return value

    def list_datasets(self, variant_id: str | None = None) -> list[dict[str, Any]]:
        with self._connect() as connection:
            variant = self._resolve_variant(connection, variant_id)
            ids = [
                row["id"]
                for row in connection.execute(
                    """SELECT id FROM dataset_versions
                       WHERE variant_id=? ORDER BY created_at DESC""",
                    (variant["id"],),
                ).fetchall()
            ]
        return [self.dataset(dataset_id) for dataset_id in ids]

    def create_training_run(
        self, dataset_id: str, config: Mapping[str, Any]
    ) -> dict[str, Any]:
        dataset = self.dataset(dataset_id)
        run_id = _id("run")
        with self._connect() as connection:
            self._resolve_variant(
                connection, str(dataset["variant_id"]), require_active=True
            )
            connection.execute(
                """
                INSERT INTO training_runs(
                    id, variant_id, dataset_id, status, config_json, created_at
                ) VALUES (?, ?, ?, 'pending', ?, ?)
                """,
                (
                    run_id, dataset["variant_id"], dataset_id,
                    _canonical_json(dict(config)), _now(),
                ),
            )
            row = connection.execute("SELECT * FROM training_runs WHERE id=?", (run_id,)).fetchone()
        return self._row(row, json_fields=("config_json", "metadata_json")) or {}

    def update_training_run(
        self,
        run_id: str,
        *,
        status: str | None = None,
        progress: float | None = None,
        error: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if status is not None and status not in RUN_STATUSES:
            raise ValueError(f"Unknown training status: {status}")
        if progress is not None and not 0 <= progress <= 1:
            raise ValueError("Training progress must be between zero and one")
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT * FROM training_runs WHERE id=?", (run_id,)
            ).fetchone()
            if existing is None:
                raise KeyError(f"Unknown training run: {run_id}")
            new_status = status or existing["status"]
            started_at = existing["started_at"] or (_now() if new_status == "running" else None)
            finished_at = existing["finished_at"] or (
                _now() if new_status in {"cancelled", "failed", "complete"} else None
            )
            connection.execute(
                """
                UPDATE training_runs SET status=?, progress=?, error=?, metadata_json=?,
                    started_at=?, finished_at=? WHERE id=?
                """,
                (
                    new_status,
                    float(progress if progress is not None else existing["progress"]),
                    error,
                    _canonical_json(dict(metadata)) if metadata is not None else existing["metadata_json"],
                    started_at,
                    finished_at,
                    run_id,
                ),
            )
            row = connection.execute("SELECT * FROM training_runs WHERE id=?", (run_id,)).fetchone()
        return self._row(row, json_fields=("config_json", "metadata_json")) or {}

    def list_training_runs(
        self,
        status: str | None = None,
        *,
        variant_id: str | None = None,
    ) -> list[dict[str, Any]]:
        if status is not None and status not in RUN_STATUSES:
            raise ValueError(f"Unknown training status: {status}")
        with self._connect() as connection:
            variant = self._resolve_variant(connection, variant_id)
            query = "SELECT * FROM training_runs WHERE variant_id=?"
            arguments: list[object] = [variant["id"]]
            if status is not None:
                query += " AND status=?"
                arguments.append(status)
            query += " ORDER BY created_at DESC"
            rows = connection.execute(query, tuple(arguments)).fetchall()
        return [
            self._row(row, json_fields=("config_json", "metadata_json")) or {}
            for row in rows
        ]

    def get_training_run(
        self,
        run_id: str,
        *,
        variant_id: str | None = None,
    ) -> dict[str, Any]:
        clause = "id=?"
        arguments: tuple[object, ...] = (run_id,)
        if variant_id is not None:
            clause += " AND variant_id=?"
            arguments = (run_id, variant_id)
        with self._connect() as connection:
            row = connection.execute(
                f"SELECT * FROM training_runs WHERE {clause}", arguments
            ).fetchone()
        value = self._row(row, json_fields=("config_json", "metadata_json"))
        if value is None:
            raise KeyError(f"Unknown training run: {run_id}")
        return value

    def add_checkpoint(
        self,
        training_run_id: str,
        epoch: int,
        artifact_path: str | os.PathLike[str],
        metrics: Mapping[str, Any],
        *,
        status: str = "candidate",
    ) -> dict[str, Any]:
        if status not in CHECKPOINT_STATUSES:
            raise ValueError(f"Unknown checkpoint status: {status}")
        checkpoint_id = _id("checkpoint")
        with self._connect() as connection:
            run = connection.execute(
                "SELECT * FROM training_runs WHERE id=?", (training_run_id,)
            ).fetchone()
            if run is None:
                raise KeyError(f"Unknown training run: {training_run_id}")
            if status == "promoted":
                connection.execute(
                    """UPDATE checkpoints SET status='candidate'
                       WHERE variant_id=? AND status='promoted'""",
                    (run["variant_id"],),
                )
            connection.execute(
                """
                INSERT INTO checkpoints(
                    id, variant_id, training_run_id, epoch, artifact_path, status,
                    metrics_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    checkpoint_id,
                    run["variant_id"],
                    training_run_id,
                    epoch,
                    str(Path(artifact_path).expanduser().resolve()),
                    status,
                    _canonical_json(dict(metrics)),
                    _now(),
                ),
            )
            row = connection.execute("SELECT * FROM checkpoints WHERE id=?", (checkpoint_id,)).fetchone()
        return self._row(row, json_fields=("metrics_json",)) or {}

    def list_checkpoints(
        self,
        training_run_id: str | None = None,
        *,
        variant_id: str | None = None,
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM checkpoints WHERE variant_id=?"
        with self._connect() as connection:
            if training_run_id is not None and variant_id is None:
                run = connection.execute(
                    "SELECT variant_id FROM training_runs WHERE id=?", (training_run_id,)
                ).fetchone()
                if run is None:
                    raise KeyError(f"Unknown training run: {training_run_id}")
                resolved_variant_id = run["variant_id"]
            else:
                resolved_variant_id = self._resolve_variant(connection, variant_id)["id"]
        arguments: tuple[object, ...] = (resolved_variant_id,)
        if training_run_id:
            query += " AND training_run_id=?"
            arguments = (resolved_variant_id, training_run_id)
        query += " ORDER BY created_at DESC"
        with self._connect() as connection:
            rows = connection.execute(query, arguments).fetchall()
        return [self._row(row, json_fields=("metrics_json",)) or {} for row in rows]

    def get_checkpoint(
        self,
        checkpoint_id: str,
        *,
        variant_id: str | None = None,
    ) -> dict[str, Any]:
        clause = "id=?"
        arguments: tuple[object, ...] = (checkpoint_id,)
        if variant_id is not None:
            clause += " AND variant_id=?"
            arguments = (checkpoint_id, variant_id)
        with self._connect() as connection:
            row = connection.execute(
                f"SELECT * FROM checkpoints WHERE {clause}", arguments
            ).fetchone()
        value = self._row(row, json_fields=("metrics_json",))
        if value is None:
            raise KeyError(f"Unknown checkpoint: {checkpoint_id}")
        return value

    def update_checkpoint(
        self,
        checkpoint_id: str,
        *,
        status: str | None = None,
        metrics: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if status is not None and status not in CHECKPOINT_STATUSES:
            raise ValueError(f"Unknown checkpoint status: {status}")
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT * FROM checkpoints WHERE id=?", (checkpoint_id,)
            ).fetchone()
            if existing is None:
                raise KeyError(f"Unknown checkpoint: {checkpoint_id}")
            next_status = status or existing["status"]
            next_metrics = _json_object(existing["metrics_json"])
            if metrics is not None:
                next_metrics.update(dict(metrics))
            # Promotion is atomic. The prior artifact is retained and merely
            # returned to candidate status so rollback never loses any files.
            if next_status == "promoted":
                connection.execute(
                    """UPDATE checkpoints SET status='candidate'
                       WHERE variant_id=? AND status='promoted' AND id<>?""",
                    (existing["variant_id"], checkpoint_id),
                )
            connection.execute(
                "UPDATE checkpoints SET status=?, metrics_json=? WHERE id=?",
                (next_status, _canonical_json(next_metrics), checkpoint_id),
            )
            row = connection.execute(
                "SELECT * FROM checkpoints WHERE id=?", (checkpoint_id,)
            ).fetchone()
        return self._row(row, json_fields=("metrics_json",)) or {}

    def add_generation(
        self,
        checkpoint_id: str,
        text: str,
        output_path: str | os.PathLike[str],
        delivery: Mapping[str, Any] | None = None,
        *,
        status: str = "pending",
        metrics: Mapping[str, Any] | None = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        if status not in {"pending", "running", "failed", "complete"}:
            raise ValueError(f"Unknown generation status: {status}")
        if not text.strip():
            raise ValueError("Generation text cannot be empty")
        generation_id = _id("generation")
        finished = _now() if status in {"failed", "complete"} else None
        with self._connect() as connection:
            checkpoint = connection.execute(
                "SELECT * FROM checkpoints WHERE id=?", (checkpoint_id,)
            ).fetchone()
            if checkpoint is None:
                raise KeyError(f"Unknown checkpoint: {checkpoint_id}")
            connection.execute(
                """
                INSERT INTO generations(
                    id, variant_id, checkpoint_id, text, delivery_json, output_path, status,
                    metrics_json, error, created_at, finished_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    generation_id,
                    checkpoint["variant_id"],
                    checkpoint_id,
                    text,
                    _canonical_json(dict(delivery or {})),
                    str(Path(output_path).expanduser().resolve()),
                    status,
                    _canonical_json(dict(metrics or {})),
                    error,
                    _now(),
                    finished,
                ),
            )
            row = connection.execute("SELECT * FROM generations WHERE id=?", (generation_id,)).fetchone()
        return self._row(row, json_fields=("delivery_json", "metrics_json")) or {}

    def list_generations(
        self,
        *,
        variant_id: str | None = None,
        checkpoint_id: str | None = None,
    ) -> list[dict[str, Any]]:
        with self._connect() as connection:
            if checkpoint_id is not None and variant_id is None:
                checkpoint = connection.execute(
                    "SELECT variant_id FROM checkpoints WHERE id=?", (checkpoint_id,)
                ).fetchone()
                if checkpoint is None:
                    raise KeyError(f"Unknown checkpoint: {checkpoint_id}")
                resolved_variant_id = checkpoint["variant_id"]
            else:
                resolved_variant_id = self._resolve_variant(connection, variant_id)["id"]
            query = "SELECT * FROM generations WHERE variant_id=?"
            arguments: list[object] = [resolved_variant_id]
            if checkpoint_id is not None:
                query += " AND checkpoint_id=?"
                arguments.append(checkpoint_id)
            query += " ORDER BY created_at DESC"
            rows = connection.execute(query, tuple(arguments)).fetchall()
        return [
            self._row(row, json_fields=("delivery_json", "metrics_json")) or {}
            for row in rows
        ]

    def get_generation(
        self,
        generation_id: str,
        *,
        variant_id: str | None = None,
    ) -> dict[str, Any]:
        clause = "id=?"
        arguments: tuple[object, ...] = (generation_id,)
        if variant_id is not None:
            clause += " AND variant_id=?"
            arguments = (generation_id, variant_id)
        with self._connect() as connection:
            row = connection.execute(
                f"SELECT * FROM generations WHERE {clause}", arguments
            ).fetchone()
        value = self._row(row, json_fields=("delivery_json", "metrics_json"))
        if value is None:
            raise KeyError(f"Unknown generation: {generation_id}")
        return value
