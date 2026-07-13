from __future__ import annotations

import json
from pathlib import Path

import pytest

from voice_trainer import bootstrap, model_setup


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
    assert bootstrap.ASR_REPO == "openai/whisper-large-v3-turbo"
    assert bootstrap.ASR_REVISION == "41f01f3fe87f28c78e2fbf8b568835947dd65ed9"
    asr_manifest = bootstrap.verify_asr_model(full_hash=True)
    assert asr_manifest["format"] == "huggingface-fp16-safetensors"
    assert bootstrap.ASR_REQUIRED_FILES <= {
        record["path"] for record in asr_manifest["files"]
    }
    assert bootstrap.verify_model(full_hash=True)["revision"] == bootstrap.MODEL_REVISION
    assert asr_manifest["revision"] == bootstrap.ASR_REVISION
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


def test_cache_adoption_creates_an_independent_verified_tree(
    isolated_model_paths: dict[str, Path],
    tmp_path: Path,
) -> None:
    source = tmp_path / "qwen-cache-snapshot"
    _write_fake_snapshot(bootstrap.MODEL_REPO, source)
    original = source / "config.json"

    adopted = bootstrap.adopt_installed_model(source)

    copied = adopted / "config.json"
    assert copied.read_bytes() == original.read_bytes()
    assert copied.stat().st_ino != original.stat().st_ino
    original.write_bytes(b"cache changed after adoption")
    assert copied.read_bytes() != original.read_bytes()
    assert bootstrap.verify_model(full_hash=True)["revision"] == bootstrap.MODEL_REVISION


def test_qwen_alone_makes_the_app_ready_while_whisper_features_remain_separate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(model_setup, "verify_model", lambda *, full_hash: {"ready": True})

    def missing_whisper(*, full_hash):
        del full_hash
        raise FileNotFoundError("Whisper is not installed")

    monkeypatch.setattr(model_setup, "verify_asr_model", missing_whisper)

    status = model_setup.ModelSetupJobs(tmp_path).status()

    assert status["ready"] is True
    assert status["speech_checks_ready"] is False
    assert status["models"]["qwen"]["ready"] is True
    assert status["models"]["whisper"]["ready"] is False
    assert status["status"] == "complete"


def test_whisper_never_replaces_qwen_as_the_required_app_readiness_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_qwen(*, full_hash):
        del full_hash
        raise FileNotFoundError("Qwen is not installed")

    monkeypatch.setattr(model_setup, "verify_model", missing_qwen)
    monkeypatch.setattr(
        model_setup,
        "verify_asr_model",
        lambda *, full_hash: {"ready": not full_hash},
    )

    status = model_setup.ModelSetupJobs(tmp_path).status()

    assert status["ready"] is False
    assert status["speech_checks_ready"] is True
    assert status["models"]["qwen"]["ready"] is False
    assert status["models"]["whisper"]["ready"] is True


class _FakeInstallerProcess:
    _next_pid = 51_000

    def __init__(self, command: list[str], *, require_kill: bool = False) -> None:
        type(self)._next_pid += 1
        self.pid = type(self)._next_pid
        self.command = command
        self.require_kill = require_kill
        self.returncode: int | None = None
        self.terminated = False
        self.killed = False
        self.group_signals: list[int] = []
        staging = Path(command[command.index("--staging") + 1])
        staging.mkdir(parents=True, mode=0o700)
        (staging / "partial-model-file").write_bytes(b"partial")

    def poll(self) -> int | None:
        return self.returncode

    def receive_group_signal(self, signum: int) -> None:
        self.group_signals.append(signum)
        if signum == model_setup.signal.SIGTERM:
            self.terminated = True
            if not self.require_kill:
                self.returncode = -signum
        elif signum == model_setup.signal.SIGKILL:
            self.killed = True
            self.returncode = -signum

    def wait(self, timeout: float | None = None) -> int:
        if self.returncode is None:
            raise model_setup.subprocess.TimeoutExpired(self.command, timeout)
        return self.returncode

def _missing_model(*, full_hash: bool) -> None:
    del full_hash
    raise FileNotFoundError("not installed")


def test_cancel_reaps_worker_and_removes_only_its_job_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app_root = tmp_path / "app"
    cache_root = tmp_path / "cache"
    models_root = app_root / "models"
    installed = models_root / "installed-model" / "keep.bin"
    installed.parent.mkdir(parents=True)
    installed.write_bytes(b"keep")
    unrelated = cache_root / "model-setup" / "model-setup-unrelated"
    unrelated.mkdir(parents=True)
    (unrelated / "keep.log").write_text("keep", encoding="utf-8")
    monkeypatch.setattr(model_setup.config, "APP_ROOT", app_root)
    monkeypatch.setattr(model_setup.config, "CACHE_ROOT", cache_root)
    monkeypatch.setattr(model_setup, "verify_model", _missing_model)
    monkeypatch.setattr(model_setup, "verify_asr_model", _missing_model)
    created: list[_FakeInstallerProcess] = []
    child_environments: list[dict[str, str]] = []

    def fake_popen(command, **kwargs):
        child_environments.append(dict(kwargs["env"]))
        process = _FakeInstallerProcess(command)
        created.append(process)
        return process

    def fake_killpg(pid, signum):
        next(process for process in created if process.pid == pid).receive_group_signal(signum)

    monkeypatch.setattr(model_setup.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(model_setup.os, "killpg", fake_killpg)
    jobs = model_setup.ModelSetupJobs(tmp_path)

    running = jobs.start("qwen")
    assert running["active"] is True
    assert running["cancelable"] is True
    assert "--parent-pid" in created[0].command
    child_environment = child_environments[0]
    assert child_environment["HF_HOME"] == str(cache_root / "huggingface")
    assert child_environment["HF_HUB_CACHE"] == str(cache_root / "huggingface" / "hub")
    assert child_environment["HUGGINGFACE_HUB_CACHE"] == str(
        cache_root / "huggingface" / "hub"
    )
    assert child_environment["HF_XET_CACHE"] == str(cache_root / "huggingface" / "xet")
    assert child_environment["XDG_CACHE_HOME"] == str(cache_root / "xdg")
    assert not {
        "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
        "http_proxy", "https_proxy", "all_proxy",
    }.intersection(child_environment)
    cancelled = jobs.cancel("qwen")

    assert created[0].terminated is True
    assert created[0].returncode == -15
    assert cancelled["status"] == "cancelled"
    assert cancelled["active"] is False
    assert cancelled["cancelable"] is False
    assert installed.read_bytes() == b"keep"
    assert (unrelated / "keep.log").read_text(encoding="utf-8") == "keep"
    assert not any(
        path.name.startswith(".model-setup-")
        for path in models_root.iterdir()
        if path.is_dir()
    )


def test_cancel_escalates_to_kill_and_shutdown_blocks_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app_root = tmp_path / "app"
    cache_root = tmp_path / "cache"
    monkeypatch.setattr(model_setup.config, "APP_ROOT", app_root)
    monkeypatch.setattr(model_setup.config, "CACHE_ROOT", cache_root)
    monkeypatch.setattr(model_setup, "verify_model", _missing_model)
    monkeypatch.setattr(model_setup, "verify_asr_model", _missing_model)
    monkeypatch.setattr(model_setup, "CANCEL_WAIT_SECONDS", 0.05)
    created: list[_FakeInstallerProcess] = []

    def fake_popen(command, **kwargs):
        del kwargs
        process = _FakeInstallerProcess(command, require_kill=True)
        created.append(process)
        return process

    def fake_killpg(pid, signum):
        next(process for process in created if process.pid == pid).receive_group_signal(signum)

    monkeypatch.setattr(model_setup.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(model_setup.os, "killpg", fake_killpg)
    jobs = model_setup.ModelSetupJobs(tmp_path)
    jobs.start("whisper")

    jobs.shutdown()

    assert created[0].terminated is True
    assert created[0].killed is True
    assert created[0].returncode == -9
    assert created[0].group_signals == [
        model_setup.signal.SIGTERM,
        model_setup.signal.SIGKILL,
    ]
    assert jobs.status()["status"] == "cancelled"
    with pytest.raises(RuntimeError, match="shutting down"):
        jobs.start("qwen")


def test_explicit_job_staging_is_cleaned_after_an_interrupted_download(
    isolated_model_paths: dict[str, Path],
) -> None:
    staging = (
        isolated_model_paths["app_root"]
        / "models"
        / ".model-setup-20260713T010203-deadbeef"
    )

    def interrupted_snapshot(*, repo_id, revision, local_dir):
        del repo_id, revision
        Path(local_dir, "partial").write_bytes(b"partial")
        raise RuntimeError("download interrupted")

    with pytest.raises(RuntimeError, match="download interrupted"):
        bootstrap.install_qwen_model(
            snapshot_download_fn=interrupted_snapshot,
            staging_path=staging,
        )

    assert not staging.exists()
    assert not isolated_model_paths["model_root"].exists()


def test_cleanup_refuses_paths_not_owned_by_the_same_setup_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app_root = tmp_path / "app"
    cache_root = tmp_path / "cache"
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "keep"
    sentinel.write_text("keep", encoding="utf-8")
    monkeypatch.setattr(model_setup.config, "APP_ROOT", app_root)
    monkeypatch.setattr(model_setup.config, "CACHE_ROOT", cache_root)
    control = cache_root / "model-setup" / "model-setup-safe"
    control.mkdir(parents=True)

    with pytest.raises(RuntimeError, match="Refusing to clean"):
        model_setup.ModelSetupJobs._cleanup_job_artifacts(control, outside)

    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_static_model_downloads_require_a_complete_local_disclosure() -> None:
    project_root = Path(__file__).resolve().parents[1]
    html = (project_root / "static" / "index.html").read_text(encoding="utf-8")
    javascript = (project_root / "static" / "app.js").read_text(encoding="utf-8")
    local_notices = (project_root / "static" / "MODEL_NOTICES.md").read_text(
        encoding="utf-8"
    )

    assert 'id="model-consent-dialog"' in html
    assert 'href="/static/MODEL_NOTICES.md"' in html
    assert "~/Library/Application Support/VoiceTrainer" in html
    assert 'openModelDownloadConsent("qwen")' in javascript
    assert 'openModelDownloadConsent("whisper")' in javascript
    assert "confirmModelDownload" in javascript
    assert 'id="cancel-model-setup"' in html
    assert 'id="cancel-speech-checks"' in html
    assert 'id="cancel-training-speech-checks"' in html
    assert 'api("/model-setup/cancel"' in javascript

    for expected in (
        bootstrap.MODEL_REPO,
        bootstrap.MODEL_REVISION,
        "Apache-2.0",
        bootstrap.ASR_REPO,
        bootstrap.ASR_REVISION,
        "MIT",
    ):
        assert expected in javascript
        assert expected in local_notices
