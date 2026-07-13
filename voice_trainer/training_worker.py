"""Isolated native-MLX worker used by the loopback application.

The web process never imports or retains the multi-gigabyte training model. A
successful worker exits after atomically saving its artifacts; evaluation then
proves that a fresh process can reconstruct them from disk.
"""

from __future__ import annotations

from . import config as app_config

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping

from .training import TrainingCancelled, TrainingConfig, train_named_speaker


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(
        json.dumps(dict(value), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.chmod(0o600)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Noice City isolated MLX worker")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--progress", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--failure", required=True)
    parser.add_argument("--cancel", required=True)
    args = parser.parse_args()

    # This module is an offline runtime boundary. It never performs setup or
    # dependency/model resolution.
    _ = app_config.APP_ROOT
    progress_path = Path(args.progress).expanduser().resolve()
    result_path = Path(args.result).expanduser().resolve()
    failure_path = Path(args.failure).expanduser().resolve()
    cancel_path = Path(args.cancel).expanduser().resolve()
    progress_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    training_config = TrainingConfig.from_dict(json.loads(args.config))

    def progress(event: dict[str, Any]) -> None:
        with progress_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
            output.flush()
            os.fsync(output.fileno())
        progress_path.chmod(0o600)

    try:
        result = train_named_speaker(
            Path(args.model).expanduser().resolve(),
            Path(args.dataset).expanduser().resolve(),
            Path(args.artifacts).expanduser().resolve(),
            config=training_config,
            progress_cb=progress,
            cancel_cb=cancel_path.exists,
        )
        _atomic_json(result_path, result)
        return 0
    except TrainingCancelled as exc:
        _atomic_json(
            failure_path,
            {
                "status": "cancelled",
                "error": str(exc),
                "finished_at": time.time(),
            },
        )
        return 2
    except Exception as exc:
        _atomic_json(
            failure_path,
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "finished_at": time.time(),
            },
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
