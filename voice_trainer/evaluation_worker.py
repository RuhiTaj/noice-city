"""Fresh-process checkpoint/Base evaluation worker."""

from __future__ import annotations

from . import config as app_config

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping

from .evaluation import evaluate_checkpoint, evaluate_unadapted_baseline


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(
        json.dumps(dict(value), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.chmod(0o600)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Noice City fresh-process evaluator")
    parser.add_argument("--mode", choices=("base", "checkpoint"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--artifact")
    parser.add_argument("--variant-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--held-out", action="append", required=True)
    parser.add_argument("--progress", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--failure", required=True)
    args = parser.parse_args()
    _ = app_config.APP_ROOT

    progress_path = Path(args.progress).expanduser().resolve()
    result_path = Path(args.result).expanduser().resolve()
    failure_path = Path(args.failure).expanduser().resolve()
    progress_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)

    def progress(event: Mapping[str, Any]) -> None:
        with progress_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(dict(event), ensure_ascii=False, sort_keys=True) + "\n")
            output.flush()
            os.fsync(output.fileno())
        progress_path.chmod(0o600)

    try:
        arguments = {
            "model_path": Path(args.model).expanduser().resolve(),
            "output_dir": Path(args.output).expanduser().resolve(),
            "held_out_real_audio": [Path(value).expanduser().resolve() for value in args.held_out],
            "progress": progress,
            "verify_model": True,
            "voice_variant_id": args.variant_id,
        }
        if args.mode == "base":
            report = evaluate_unadapted_baseline(**arguments)
        else:
            if not args.artifact:
                raise ValueError("checkpoint evaluation requires --artifact")
            report = evaluate_checkpoint(
                **arguments,
                artifact_path=Path(args.artifact).expanduser().resolve(),
            )
        _atomic_json(
            result_path,
            {
                "status": "complete",
                "mode": args.mode,
                "report_path": str(Path(args.output).expanduser().resolve() / "evaluation.json"),
                "aggregate": report["aggregate"],
            },
        )
        return 0
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
