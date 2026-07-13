from __future__ import annotations

import json
from pathlib import Path

import pytest

from voice_trainer import bootstrap


@pytest.fixture
def isolated_model_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    app_root = tmp_path / "application-support"
    model_root = app_root / "models" / "qwen" / bootstrap.MODEL_REVISION
    asr_root = app_root / "models" / "whisper" / bootstrap.ASR_REVISION
    values = {
        "app_root": app_root,
        "model_root": model_root,
        "model_manifest": model_root / "manifest.json",
        "asr_root": asr_root,
        "asr_manifest": asr_root / "manifest.json",
    }
    monkeypatch.setattr(bootstrap, "APP_ROOT", app_root)
    monkeypatch.setattr(bootstrap, "MODEL_ROOT", model_root)
    monkeypatch.setattr(bootstrap, "MODEL_MANIFEST", values["model_manifest"])
    monkeypatch.setattr(bootstrap, "ASR_ROOT", asr_root)
    monkeypatch.setattr(bootstrap, "ASR_MANIFEST", values["asr_manifest"])
    monkeypatch.setattr(bootstrap, "HF_CACHE_MODEL", tmp_path / "missing-hf-cache")

    def ensure() -> None:
        (app_root / "models").mkdir(parents=True, exist_ok=True, mode=0o700)

    monkeypatch.setattr(bootstrap, "ensure_private_directories", ensure)
    return values


def _write_fake_snapshot(repo_id: str, local_dir: str | Path) -> None:
    root = Path(local_dir)
    required = (
        bootstrap.MODEL_REQUIRED_FILES
        if repo_id == bootstrap.MODEL_REPO
        else bootstrap.ASR_REQUIRED_FILES
    )
    for relative in required:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"fixture:{repo_id}:{relative}".encode())
    (root / "README.md").write_text(f"Fixture for {repo_id}\n", encoding="utf-8")


def test_first_run_downloads_exact_pinned_models_and_restarts_offline(
    isolated_model_paths: dict[str, Path],
) -> None:
    calls: list[tuple[str, str]] = []
    events: list[dict] = []

    def fake_snapshot_download(*, repo_id, revision, local_dir):
        calls.append((repo_id, revision))
        _write_fake_snapshot(repo_id, local_dir)

    installed = bootstrap.install_required_models(
        progress=events.append,
        snapshot_download_fn=fake_snapshot_download,
    )

    assert calls == [
        (bootstrap.MODEL_REPO, bootstrap.MODEL_REVISION),
        (bootstrap.ASR_REPO, bootstrap.ASR_REVISION),
    ]
    assert installed == {
        "qwen": str(isolated_model_paths["model_root"]),
        "whisper": str(isolated_model_paths["asr_root"]),
    }
    assert bootstrap.verify_model(full_hash=True)["revision"] == bootstrap.MODEL_REVISION
    assert bootstrap.verify_asr_model(full_hash=True)["revision"] == bootstrap.ASR_REVISION
    assert {event["component"] for event in events} == {"Qwen", "Whisper"}

    # A restart or repeated setup verifies local manifests and performs no download.
    calls.clear()
    bootstrap.install_required_models(snapshot_download_fn=fake_snapshot_download)
    assert calls == []


@pytest.mark.parametrize("kind", ["qwen", "whisper"])
def test_verification_rejects_empty_or_incomplete_manifests(
    isolated_model_paths: dict[str, Path],
    kind: str,
) -> None:
    if kind == "qwen":
        root = isolated_model_paths["model_root"]
        manifest_path = isolated_model_paths["model_manifest"]
        repo = bootstrap.MODEL_REPO
        revision = bootstrap.MODEL_REVISION
        verify = bootstrap.verify_model
    else:
        root = isolated_model_paths["asr_root"]
        manifest_path = isolated_model_paths["asr_manifest"]
        repo = bootstrap.ASR_REPO
        revision = bootstrap.ASR_REVISION
        verify = bootstrap.verify_asr_model
    root.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "repo": repo,
                "revision": revision,
                "files": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="contains no files"):
        verify(full_hash=False)


def test_failed_download_never_creates_final_model_tree(
    isolated_model_paths: dict[str, Path],
) -> None:
    def incomplete_snapshot(*, repo_id, revision, local_dir):
        Path(local_dir, "config.json").write_text("{}", encoding="utf-8")

    with pytest.raises(RuntimeError, match="incomplete"):
        bootstrap.install_qwen_model(snapshot_download_fn=incomplete_snapshot)

    assert not isolated_model_paths["model_root"].exists()
    assert not isolated_model_paths["model_manifest"].exists()
    assert list((isolated_model_paths["app_root"] / "models").iterdir()) == []


def test_explicit_retry_replaces_an_incomplete_model_without_touching_whisper(
    isolated_model_paths: dict[str, Path],
) -> None:
    calls: list[str] = []

    def fake_snapshot_download(*, repo_id, revision, local_dir):
        calls.append(repo_id)
        _write_fake_snapshot(repo_id, local_dir)

    bootstrap.install_required_models(snapshot_download_fn=fake_snapshot_download)
    calls.clear()
    qwen_config = isolated_model_paths["model_root"] / "config.json"
    qwen_config.write_bytes(qwen_config.read_bytes() + b"corrupt")

    bootstrap.install_required_models(snapshot_download_fn=fake_snapshot_download)

    assert calls == [bootstrap.MODEL_REPO]
    assert bootstrap.verify_model(full_hash=True)["revision"] == bootstrap.MODEL_REVISION
    assert bootstrap.verify_asr_model(full_hash=True)["revision"] == bootstrap.ASR_REVISION
