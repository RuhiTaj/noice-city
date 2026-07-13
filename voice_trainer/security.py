from __future__ import annotations

import hmac
import os

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from .config import ORIGIN


SESSION_COOKIE = "voicetrainer_session"

SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self'; style-src 'self'; "
        "font-src 'self'; connect-src 'self'; img-src 'self' data: blob:; "
        "media-src 'self' blob:; object-src 'none'; base-uri 'none'; "
        "frame-ancestors 'none'; form-action 'self'"
    ),
    "Permissions-Policy": "microphone=(self), camera=(), geolocation=()",
    "Referrer-Policy": "no-referrer",
    "X-Content-Type-Options": "nosniff",
    "Cross-Origin-Opener-Policy": "same-origin",
}


class LocalSessionMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self.token = os.environ.get("VOICETRAINER_SESSION_TOKEN", "")
        if len(self.token) < 32:
            raise RuntimeError("VOICETRAINER_SESSION_TOKEN must be set by start.sh")

    async def dispatch(self, request: Request, call_next):
        if request.url.path.startswith("/api/"):
            supplied = request.headers.get("X-VoiceTrainer-Session", "")
            cookie = request.cookies.get(SESSION_COOKIE, "")
            header_valid = hmac.compare_digest(supplied, self.token)
            cookie_valid = hmac.compare_digest(cookie, self.token)
            if not (header_valid or cookie_valid):
                return JSONResponse({"detail": "Invalid local session."}, status_code=403)
            if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
                if request.headers.get("Origin") != ORIGIN:
                    return JSONResponse({"detail": "Invalid request origin."}, status_code=403)
        response = await call_next(request)
        for key, value in SECURITY_HEADERS.items():
            response.headers[key] = value
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
            response.set_cookie(
                SESSION_COOKIE,
                self.token,
                max_age=31_536_000,
                httponly=True,
                samesite="strict",
                path="/",
            )
        return response
