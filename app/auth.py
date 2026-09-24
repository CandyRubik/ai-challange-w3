from __future__ import annotations

import hashlib
import hmac
import os
import time


AUTH_COOKIE_NAME = "ai_challenge_session"
SESSION_TTL_SECONDS = 7 * 24 * 60 * 60


def auth_password() -> str | None:
    value = os.getenv("APP_PASSWORD", "")
    return value if value else None


def production_mode() -> bool:
    return os.getenv("APP_ENV", "development").strip().lower() in {
        "prod", "production",
    }


def cookie_secure() -> bool:
    if production_mode():
        return True
    override = os.getenv("APP_COOKIE_SECURE")
    if override is not None:
        return override.strip().lower() in {"1", "true", "yes"}
    return False


def create_session_cookie(password: str, now: int | None = None) -> str:
    issued_at = str(now if now is not None else int(time.time()))
    signature = hmac.new(
        password.encode("utf-8"), issued_at.encode("ascii"), hashlib.sha256,
    ).hexdigest()
    return f"{issued_at}.{signature}"


def valid_session_cookie(value: str | None, password: str, now: int | None = None) -> bool:
    if not value:
        return False
    try:
        issued_at_text, supplied_signature = value.split(".", 1)
        issued_at = int(issued_at_text)
    except (ValueError, TypeError):
        return False
    current_time = now if now is not None else int(time.time())
    age = current_time - issued_at
    if age < 0 or age > SESSION_TTL_SECONDS:
        return False
    expected_signature = hmac.new(
        password.encode("utf-8"), issued_at_text.encode("ascii"), hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(supplied_signature, expected_signature)
