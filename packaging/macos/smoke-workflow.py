#!/usr/bin/env python3
"""Exercise a packaged Noice City app with synthetic, non-user speech.

This release-only harness talks to the loopback API used by the browser UI. It
uses macOS's built-in Samantha voice to create fresh PCM24 recording fixtures,
then runs the real intake, review, freeze, MLX pilot, evaluation, and generation
paths. Nothing in this script reads or copies a user's recordings or adapters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import wave
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
VARIANT_ID = "variant_ruhi_default"
TARGET_AUDIO_SECONDS = 128.0
POLL_SECONDS = 3.0

INTERVIEW_ANSWERS = {
    "interview-021": (
        "Hyderabad feels familiar to me because its busy streets, warm evenings, "
        "and welcoming conversations always make the city feel alive."
    ),
    "interview-029": (
        "People sometimes mispronounce Tirupati. I say Tirupati naturally, with "
        "the soft Indian consonants I use when speaking at home."
    ),
    "interview-030": (
        "Welcome, everyone! We are genuinely happy you are here, and we cannot "
        "wait to celebrate, laugh, and share a wonderful evening together."
    ),
}

# Required validation/test/interview coverage comes first. Additional guided
# training prompts are appended until the pilot has a safe duration margin.
PROMPT_IDS = [
    "guided-111",
    "guided-114",
    "guided-115",
    "interview-021",
    "interview-029",
    "interview-030",
    *[f"guided-{number:03d}" for number in range(81, 111)],
]


class Client:
    def __init__(self, base_url: str, token: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.headers = {
            "X-VoiceTrainer-Session": token,
            "Origin": self.base_url,
        }

    def request(
        self,
        method: str,
        path: str,
        *,
        data: bytes | None = None,
        content_type: str | None = None,
        timeout: float = 600.0,
    ) -> tuple[bytes, Any]:
        headers = dict(self.headers)
        if content_type:
            headers["Content-Type"] = content_type
        request = urllib.request.Request(
            f"{self.base_url}{path}", data=data, headers=headers, method=method
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read(), response.headers
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"{method} {path} failed with HTTP {exc.code}: {detail}") from exc

    def json(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        payload = None if body is None else json.dumps(body).encode("utf-8")
        raw, _ = self.request(
            method,
            path,
            data=payload,
            content_type="application/json" if payload is not None else None,
        )
        return json.loads(raw)

    def upload_recording(
        self,
        *,
        path: Path,
        prompt_id: str,
        session_id: str,
    ) -> dict[str, Any]:
        boundary = f"NoiceCitySmoke{secrets.token_hex(12)}"
        chunks: list[bytes] = []

        def field(name: str, value: str) -> None:
            chunks.extend(
                (
                    f"--{boundary}\r\n".encode(),
                    f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                    value.encode(),
                    b"\r\n",
                )
            )

        field("prompt_id", prompt_id)
        field("session_id", session_id)
        field("sample_rate", "48000")
        field("bit_depth", "24")
        chunks.extend(
            (
                f"--{boundary}\r\n".encode(),
                b'Content-Disposition: form-data; name="audio"; filename="take.wav"\r\n',
                b"Content-Type: audio/wav\r\n\r\n",
                path.read_bytes(),
                b"\r\n",
                f"--{boundary}--\r\n".encode(),
            )
        )
        raw, _ = self.request(
            "POST",
            "/api/recordings",
            data=b"".join(chunks),
            content_type=f"multipart/form-data; boundary={boundary}",
        )
        return json.loads(raw)


def synthesize(text: str, destination: Path) -> float:
    aiff = destination.with_suffix(".aiff")
    subprocess.run(
        ["/usr/bin/say", "-v", "Samantha", "-r", "125", "-o", str(aiff), text],
        check=True,
    )
    subprocess.run(
        [
            "/usr/bin/afconvert",
            "-f",
            "WAVE",
            "-d",
            "LEI24@48000",
            "-c",
            "1",
            str(aiff),
            str(destination),
        ],
        check=True,
    )
    aiff.unlink()
    with wave.open(str(destination), "rb") as audio:
        if (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) != (48_000, 1, 3):
            raise RuntimeError(f"unexpected recording format: {destination}")
        duration = audio.getnframes() / audio.getframerate()
    if not 2.0 <= duration <= 15.0:
        raise RuntimeError(f"synthetic recording duration is outside intake limits: {duration:.3f}s")
    return duration


def prompt_catalog() -> dict[str, dict[str, Any]]:
    payload = json.loads((ROOT / "assets" / "prompts.json").read_text(encoding="utf-8"))
    return {row["id"]: row for row in payload["prompts"]}


def record_train(client: Client, work_dir: Path) -> dict[str, Any]:
    status = client.json("GET", "/api/status")
    if not status["model_ready"] or not status["asr"]["ready"]:
        raise RuntimeError("Qwen and Whisper must both be verified before the smoke workflow")
    if status["recordings"] or status["datasets"] or status["checkpoints"]:
        raise RuntimeError("smoke workflow requires an empty application-data state")

    session = client.json(
        "POST", "/api/recording-sessions", {"variant_id": VARIANT_ID, "mode": "hybrid"}
    )
    catalog = prompt_catalog()
    recordings: list[dict[str, Any]] = []
    transcripts: dict[str, str] = {}
    duration = 0.0
    for prompt_id in PROMPT_IDS:
        prompt = catalog[prompt_id]
        spoken_text = INTERVIEW_ANSWERS.get(prompt_id, prompt["text"])
        wav_path = work_dir / f"{prompt_id}.wav"
        local_duration = synthesize(spoken_text, wav_path)
        row = client.upload_recording(
            path=wav_path,
            prompt_id=prompt_id,
            session_id=session["id"],
        )
        if not row["qa_passed"]:
            raise RuntimeError(
                f"recording {prompt_id} failed acoustic QA: {row.get('qa_json')}"
            )
        recordings.append(row)
        transcripts[row["id"]] = spoken_text
        duration += float(row["duration_seconds"])
        print(
            f"recorded {prompt_id}: {local_duration:.2f}s, "
            f"ASR={row['transcript_status']}, total={duration:.2f}s",
            flush=True,
        )
        if duration >= TARGET_AUDIO_SECONDS and len(recordings) >= 10:
            break

    if duration < 120.0:
        raise RuntimeError(f"pilot fixture is only {duration:.2f}s")

    review = client.json(
        "POST",
        "/api/recordings/bulk",
        {
            "variant_id": VARIANT_ID,
            "recording_ids": [row["id"] for row in recordings],
            "status": "approved",
            "playback_confirmed_ids": [row["id"] for row in recordings],
            "confirm_playback": False,
            "confirm_transcripts": True,
            "override_failed_qa": False,
            "transcripts": transcripts,
        },
    )
    if review["updated_count"] != len(recordings) or review["skipped_count"]:
        raise RuntimeError(f"bulk review did not approve every fixture: {review}")

    dataset = client.json(
        "POST",
        "/api/datasets/freeze",
        {"variant_id": VARIANT_ID, "name": "release-smoke-pilot", "purpose": "pilot"},
    )
    run = client.json(
        "POST",
        "/api/training-runs",
        {
            "variant_id": VARIANT_ID,
            "dataset_version": dataset["id"],
            "stage": "pilot",
            "label": "release-smoke-pilot",
            "epochs": 1,
        },
    )
    deadline = time.monotonic() + 60 * 45
    while True:
        current = client.json(
            "GET",
            "/api/training-runs/"
            + urllib.parse.quote(run["id"])
            + "?variant_id="
            + urllib.parse.quote(VARIANT_ID),
        )
        print(
            f"training {current['status']}: {current['progress_percent']:.1f}% "
            f"{current.get('message', '')}",
            flush=True,
        )
        if current["status"] in {"complete", "failed", "cancelled"}:
            break
        if time.monotonic() >= deadline:
            raise TimeoutError("pilot training did not finish in 45 minutes")
        time.sleep(POLL_SECONDS)
    if current["status"] != "complete":
        raise RuntimeError(f"pilot training failed: {current}")

    deadline = time.monotonic() + 60 * 60
    checkpoint: dict[str, Any] | None = None
    while True:
        status = client.json("GET", "/api/status")
        checkpoints = status["checkpoints"]
        checkpoint = checkpoints[0] if checkpoints else None
        evaluation_status = (
            checkpoint.get("metrics_json", {}).get("evaluation_status") if checkpoint else None
        )
        print(f"evaluation: {evaluation_status or 'waiting'}", flush=True)
        if checkpoint and evaluation_status in {"awaiting_blind_review", "promoted"}:
            break
        if checkpoint and evaluation_status in {"failed", "rejected"}:
            raise RuntimeError(
                f"checkpoint evaluation did not pass the release smoke: "
                f"{checkpoint['metrics_json']}"
            )
        if time.monotonic() >= deadline:
            raise TimeoutError("checkpoint evaluation did not finish in 60 minutes")
        time.sleep(POLL_SECONDS)

    evaluation = checkpoint["metrics_json"].get("evaluation") or {}
    if not evaluation.get("reference_free"):
        raise RuntimeError(f"checkpoint evaluation was not reference-free: {evaluation}")
    if not evaluation.get("all_intelligible"):
        raise RuntimeError(f"checkpoint evaluation was not fully intelligible: {evaluation}")
    if evaluation.get("locked_prompt_count") != 20 or evaluation.get("intelligible_count") != 20:
        raise RuntimeError(f"checkpoint evaluation did not complete all 20 prompts: {evaluation}")

    result = {
        "variant_id": VARIANT_ID,
        "recording_count": len(recordings),
        "recording_seconds": duration,
        "dataset_id": dataset["id"],
        "dataset_manifest_sha256": dataset["manifest_sha256"],
        "training_run_id": run["id"],
        "checkpoint_id": checkpoint["id"],
        "evaluation_status": checkpoint["metrics_json"]["evaluation_status"],
        "evaluation": evaluation,
    }
    (work_dir / "workflow-result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


def generate(
    client: Client,
    work_dir: Path,
    checkpoint_id: str,
    suffix: str,
    expected_sha256: str | None,
) -> dict[str, Any]:
    text = "Compatibility testing keeps every local voice artifact private and reproducible."
    raw, headers = client.request(
        "POST",
        "/api/generations",
        data=json.dumps(
            {
                "variant_id": VARIANT_ID,
                "text": text,
                "delivery": "natural",
                "checkpoint_version": checkpoint_id,
                "seed": 3407,
            }
        ).encode("utf-8"),
        content_type="application/json",
        timeout=60 * 20,
    )
    output = work_dir / f"generated-{suffix}.wav"
    output.write_bytes(raw)
    with wave.open(str(output), "rb") as audio:
        if audio.getnframes() <= 0 or audio.getnchannels() != 1:
            raise RuntimeError("generation did not return nonempty mono WAV audio")
        sample_rate = audio.getframerate()
        duration = audio.getnframes() / audio.getframerate()
    result = {
        "path": str(output),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "duration_seconds": duration,
        "sample_rate": sample_rate,
        "seed": headers.get("X-VoiceTrainer-Seed"),
        "generation_id": headers.get("X-VoiceTrainer-Generation-Id"),
    }
    if sample_rate != 24_000:
        raise RuntimeError(f"generation sample rate is {sample_rate}, expected 24000")
    if result["seed"] != "3407":
        raise RuntimeError(f"generation returned seed {result['seed']!r}, expected '3407'")
    if expected_sha256 and result["sha256"] != expected_sha256:
        raise RuntimeError(
            "seeded generation changed across fresh processes: "
            f"{result['sha256']} != {expected_sha256}"
        )
    (work_dir / f"generation-{suffix}.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("record-train", "generate"))
    parser.add_argument("--base-url", default="http://127.0.0.1:8787")
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-id")
    parser.add_argument("--suffix", default="after-restart")
    parser.add_argument(
        "--expected-sha256",
        help="Require this exact WAV digest (used for the second fresh-process generation).",
    )
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    token = args.token_file.read_text(encoding="utf-8").strip()
    if len(token) != 64:
        raise SystemExit("invalid local session token")
    client = Client(args.base_url, token)
    if args.phase == "record-train":
        record_train(client, args.work_dir)
    else:
        if not args.checkpoint_id:
            raise SystemExit("--checkpoint-id is required for generate")
        generate(
            client,
            args.work_dir,
            args.checkpoint_id,
            args.suffix,
            args.expected_sha256,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
