from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from voice_trainer.config import ORIGIN
from voice_trainer.security import LocalSessionMiddleware, SESSION_COOKIE


def test_valid_header_establishes_restart_safe_http_only_cookie(monkeypatch) -> None:
    token = "a" * 64
    monkeypatch.setenv("VOICETRAINER_SESSION_TOKEN", token)
    app = FastAPI()
    app.add_middleware(LocalSessionMiddleware)

    @app.get("/api/check")
    def check():
        return {"ok": True}

    @app.post("/api/write")
    def write():
        return {"ok": True}

    with TestClient(app, base_url=ORIGIN) as client:
        response = client.get(
            "/api/check", headers={"X-VoiceTrainer-Session": token}
        )
        assert response.status_code == 200
        assert SESSION_COOKIE in response.cookies

        # A stale header in an old tab can recover through the shared cookie.
        assert client.get(
            "/api/check", headers={"X-VoiceTrainer-Session": "stale"}
        ).status_code == 200
        assert client.post(
            "/api/write",
            headers={"X-VoiceTrainer-Session": "stale", "Origin": ORIGIN},
        ).status_code == 200
        assert client.post(
            "/api/write",
            headers={"X-VoiceTrainer-Session": "stale", "Origin": "http://evil.invalid"},
        ).status_code == 403
