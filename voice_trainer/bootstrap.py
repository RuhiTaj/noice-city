from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

from .config import (
    APP_ROOT,
    ASR_MANIFEST,
    ASR_REPO,
    ASR_REVISION,
    ASR_ROOT,
    MODEL_MANIFEST,
    MODEL_REPO,
    MODEL_REVISION,
    MODEL_ROOT,
    ensure_private_directories,
)


HF_CACHE_MODEL = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--mlx-community--Qwen3-TTS-12Hz-1.7B-Base-bf16"
    / "snapshots"
    / MODEL_REVISION
)
MODEL_REQUIRED_FILES = {
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "tokenizer_config.json",
    "speech_tokenizer/config.json",
    "speech_tokenizer/model.safetensors",
}
ASR_REQUIRED_FILES = {"config.json", "model.safetensors", "tokenizer.json"}
FORBIDDEN_SUFFIXES = {".bin", ".pt", ".pth", ".pkl"}
ProgressCallback = Callable[[dict], None]


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _source_files(source: Path) -> list[tuple[Path, Path]]:
    if not source.is_dir():
        raise FileNotFoundError(
            "The installed Qwen3-TTS snapshot was not found at " f"{source}."
        )
    files: list[tuple[Path, Path]] = []
    for candidate in sorted(source.rglob("*")):
        if candidate.is_dir():
            continue
        relative = candidate.relative_to(source)
        if candidate.suffix.lower() in FORBIDDEN_SUFFIXES:
            raise RuntimeError(f"Unsafe model file type: {relative}")
        files.append((candidate.resolve(strict=True), relative))
    present = {str(relative) for _, relative in files}
    missing = MODEL_REQUIRED_FILES - present
    if missing:
        raise RuntimeError(f"Installed model is incomplete: {sorted(missing)}")
    return files


def adopt_installed_model(source: Path = HF_CACHE_MODEL) -> Path:
    """Make an independent, immutable local model tree without network access."""
    ensure_private_directories()
    files = _source_files(source)
    if MODEL_MANIFEST.exists():
        verify_model()
        return MODEL_ROOT

    staging = Path(tempfile.mkdtemp(prefix="model-staging-", dir=APP_ROOT / "models"))
    records: list[dict] = []
    try:
        for original, relative in files:
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            try:
                os.link(original, destination)
            except OSError:
                shutil.copy2(original, destination)
            destination.chmod(0o600)
            records.append(
                {
                    "path": str(relative),
                    "size": destination.stat().st_size,
                    "sha256": sha256_file(destination),
                }
            )

        manifest = {
            "schema_version": 1,
            "repo": MODEL_REPO,
            "revision": MODEL_REVISION,
            "format": "mlx-bf16-safetensors",
            "files": records,
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (staging / "manifest.json").chmod(0o600)
        MODEL_ROOT.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.replace(staging, MODEL_ROOT)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    verify_model()
    return MODEL_ROOT


@contextmanager
def _explicit_hub_access():
    """Temporarily permit Hub traffic only inside an explicit setup process."""
    names = ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE")
    previous = {name: os.environ.get(name) for name in names}
    for name in names:
        os.environ.pop(name, None)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _emit(progress: ProgressCallback | None, **event) -> None:
    if progress is not None:
        progress(event)


def _download_snapshot(
    *,
    repo: str,
    revision: str,
    target: Path,
    required_files: set[str],
    model_format: str,
    component: str,
    progress: ProgressCallback | None = None,
    snapshot_download_fn=None,
) -> Path:
    """Download, validate, checksum, and atomically install one pinned model."""
    ensure_private_directories()
    staging = Path(
        tempfile.mkdtemp(prefix=f"{component}-download-", dir=APP_ROOT / "models")
    )
    try:
        _emit(
            progress,
            component=component,
            phase="downloading",
            message=f"Downloading {component} model files",
        )
        if snapshot_download_fn is None:
            with _explicit_hub_access():
                from huggingface_hub import snapshot_download
                from tqdm.auto import tqdm

                progress_start, progress_end = (
                    (0.08, 0.60) if component == "Qwen" else (0.72, 0.91)
                )

                class SetupProgress(tqdm):
                    _last_emit = 0.0

                    def __init__(self, *args, **kwargs):
                        kwargs["disable"] = True
                        super().__init__(*args, **kwargs)

                    def update(self, amount=1):
                        result = super().update(amount)
                        now = time.monotonic()
                        total = float(self.total or 0)
                        if (
                            str(getattr(self, "desc", "")) == "Downloading bytes"
                            and total > 0
                            and now - type(self)._last_emit >= 0.5
                        ):
                            type(self)._last_emit = now
                            ratio = min(1.0, max(0.0, float(self.n) / total))
                            _emit(
                                progress,
                                component=component,
                                phase="downloading",
                                message=f"Downloading {component} model files",
                                downloaded_bytes=int(self.n),
                                total_bytes=int(total),
                                progress=progress_start + ratio * (progress_end - progress_start),
                            )
                        return result

                snapshot_download(
                    repo_id=repo,
                    revision=revision,
                    local_dir=staging,
                    tqdm_class=SetupProgress,
                )
        else:
            snapshot_download_fn(repo_id=repo, revision=revision, local_dir=staging)
        shutil.rmtree(staging / ".cache", ignore_errors=True)

        _emit(
            progress,
            component=component,
            phase="verifying",
            message=f"Verifying {component} model files",
        )
        records: list[dict] = []
        for path in sorted(staging.rglob("*")):
            if path.is_dir() or ".cache" in path.parts:
                continue
            relative = path.relative_to(staging)
            if path.suffix.lower() in FORBIDDEN_SUFFIXES:
                raise RuntimeError(f"Unsafe {component} model file type: {relative}")
            path.chmod(0o600)
            records.append(
                {
                    "path": str(relative),
                    "size": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        present = {record["path"] for record in records}
        if missing := required_files - present:
            raise RuntimeError(f"Downloaded {component} model is incomplete: {sorted(missing)}")
        manifest = {
            "schema_version": 1,
            "repo": repo,
            "revision": revision,
            "format": model_format,
            "files": records,
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (staging / "manifest.json").chmod(0o600)
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if target.exists():
            raise RuntimeError(
                f"A non-verified {component} model already exists; remove it before repair"
            )
        os.replace(staging, target)
        _emit(
            progress,
            component=component,
            phase="installed",
            message=f"{component} model installed",
        )
        return target
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def install_qwen_model(
    *,
    progress: ProgressCallback | None = None,
    snapshot_download_fn=None,
) -> Path:
    """Download the exact Qwen BF16 snapshot during explicit first-run setup."""
    if MODEL_MANIFEST.exists():
        try:
            verify_model(full_hash=False)
            _emit(progress, component="Qwen", phase="ready", message="Qwen is ready")
            return MODEL_ROOT
        except Exception:
            _emit(
                progress,
                component="Qwen",
                phase="repairing",
                message="Replacing an incomplete Qwen installation",
            )
            shutil.rmtree(MODEL_ROOT, ignore_errors=True)
    elif MODEL_ROOT.exists():
        shutil.rmtree(MODEL_ROOT, ignore_errors=True)
    if HF_CACHE_MODEL.is_dir():
        _emit(
            progress,
            component="Qwen",
            phase="adopting",
            message="Using the verified Qwen snapshot already on this Mac",
        )
        return adopt_installed_model()
    return _download_snapshot(
        repo=MODEL_REPO,
        revision=MODEL_REVISION,
        target=MODEL_ROOT,
        required_files=MODEL_REQUIRED_FILES,
        model_format="mlx-bf16-safetensors",
        component="Qwen",
        progress=progress,
        snapshot_download_fn=snapshot_download_fn,
    )


def verify_model(full_hash: bool = True) -> dict:
    if not MODEL_MANIFEST.is_file():
        raise FileNotFoundError("Local Qwen model is not installed yet.")
    manifest = json.loads(MODEL_MANIFEST.read_text())
    if manifest.get("repo") != MODEL_REPO or manifest.get("revision") != MODEL_REVISION:
        raise RuntimeError("Local model identity does not match the pinned model.")
    records = manifest.get("files")
    if not isinstance(records, list) or not records:
        raise RuntimeError("Local model manifest contains no files.")
    present = {str(record.get("path", "")) for record in records}
    if missing := MODEL_REQUIRED_FILES - present:
        raise RuntimeError(f"Local model manifest is incomplete: {sorted(missing)}")
    for record in records:
        path = (MODEL_ROOT / record["path"]).resolve(strict=True)
        if MODEL_ROOT.resolve() not in path.parents:
            raise RuntimeError(f"Model path escaped its root: {record['path']}")
        if path.stat().st_size != record["size"]:
            raise RuntimeError(f"Model size mismatch: {record['path']}")
        if full_hash and sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"Model checksum mismatch: {record['path']}")
    return manifest


def install_asr_model() -> Path:
    """Download the pinned local QA model during explicit online setup only."""
    ensure_private_directories()
    if ASR_MANIFEST.exists():
        try:
            verify_asr_model(full_hash=False)
            return ASR_ROOT
        except Exception:
            shutil.rmtree(ASR_ROOT, ignore_errors=True)
    elif ASR_ROOT.exists():
        shutil.rmtree(ASR_ROOT, ignore_errors=True)

    _download_snapshot(
        repo=ASR_REPO,
        revision=ASR_REVISION,
        target=ASR_ROOT,
        required_files=ASR_REQUIRED_FILES,
        model_format="mlx-fp16-safetensors",
        component="Whisper",
    )
    verify_asr_model()
    return ASR_ROOT


def install_required_models(
    *,
    progress: ProgressCallback | None = None,
    snapshot_download_fn=None,
) -> dict:
    """Install both pinned model trees; safe to call again after completion."""
    qwen = install_qwen_model(
        progress=progress,
        snapshot_download_fn=snapshot_download_fn,
    )
    if ASR_MANIFEST.exists():
        try:
            verify_asr_model(full_hash=False)
            _emit(progress, component="Whisper", phase="ready", message="Whisper is ready")
            whisper = ASR_ROOT
        except Exception:
            _emit(
                progress,
                component="Whisper",
                phase="repairing",
                message="Replacing an incomplete Whisper installation",
            )
            shutil.rmtree(ASR_ROOT, ignore_errors=True)
            whisper = _download_snapshot(
                repo=ASR_REPO,
                revision=ASR_REVISION,
                target=ASR_ROOT,
                required_files=ASR_REQUIRED_FILES,
                model_format="mlx-fp16-safetensors",
                component="Whisper",
                progress=progress,
                snapshot_download_fn=snapshot_download_fn,
            )
    else:
        if ASR_ROOT.exists():
            shutil.rmtree(ASR_ROOT, ignore_errors=True)
        whisper = _download_snapshot(
            repo=ASR_REPO,
            revision=ASR_REVISION,
            target=ASR_ROOT,
            required_files=ASR_REQUIRED_FILES,
            model_format="mlx-fp16-safetensors",
            component="Whisper",
            progress=progress,
            snapshot_download_fn=snapshot_download_fn,
        )
    verify_model(full_hash=False)
    verify_asr_model(full_hash=False)
    return {"qwen": str(qwen), "whisper": str(whisper)}


def verify_asr_model(full_hash: bool = True) -> dict:
    if not ASR_MANIFEST.is_file():
        raise FileNotFoundError("Local Whisper model is not installed yet.")
    manifest = json.loads(ASR_MANIFEST.read_text())
    if manifest.get("repo") != ASR_REPO or manifest.get("revision") != ASR_REVISION:
        raise RuntimeError("Local ASR model identity does not match the pinned model.")
    records = manifest.get("files")
    if not isinstance(records, list) or not records:
        raise RuntimeError("Local ASR manifest contains no files.")
    present = {str(record.get("path", "")) for record in records}
    if missing := ASR_REQUIRED_FILES - present:
        raise RuntimeError(f"Local ASR manifest is incomplete: {sorted(missing)}")
    for record in records:
        path = (ASR_ROOT / record["path"]).resolve(strict=True)
        if ASR_ROOT.resolve() not in path.parents:
            raise RuntimeError(f"ASR model path escaped its root: {record['path']}")
        if path.stat().st_size != record["size"]:
            raise RuntimeError(f"ASR model size mismatch: {record['path']}")
        if full_hash and sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"ASR model checksum mismatch: {record['path']}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Adopt the installed Qwen model locally")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--download-asr", action="store_true")
    parser.add_argument("--download-models", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Skip full content hashes")
    args = parser.parse_args()
    if args.download_models:
        print(json.dumps(install_required_models(), indent=2))
    elif args.download_asr:
        print(install_asr_model())
    elif args.verify_only:
        verify_model(full_hash=not args.quick)
        verify_asr_model(full_hash=not args.quick)
        print(MODEL_ROOT)
    else:
        print(adopt_installed_model())


if __name__ == "__main__":
    main()
