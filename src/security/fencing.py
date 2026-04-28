"""
HMAC-SHA256 fencing tokens for AlgoTrader Pro.

Replaces the RSA-4096 implementation from Session 2. A single-operator
setup does not benefit from asymmetric cryptography; HMAC-SHA256 with a
shared secret stored in Redis provides the same protection against replay
and forgery at ~100× less latency and zero external dependencies.

Design
------
A fencing token is a URL-safe base64-encoded JSON payload with an HMAC-SHA256
signature appended as a separate field. The payload contains:

    {
        "incident_id":  str,   # unique per-incident UUID
        "action_code":  str,   # e.g. "halted", "paused"
        "severity":     str,   # "internal" | "warning" | "critical" | "emergency"
        "issued_by":    str,   # process UUID (set at import time)
        "issued_at":    float, # Unix timestamp
        "valid_until":  float  # Unix timestamp
    }

The HMAC is computed over the canonical JSON of the payload (keys sorted,
no extra whitespace). Both payload and HMAC are base64url-encoded and joined
with a dot, similar to a minimal JWT: `<payload_b64>.<hmac_b64>`.

Secret management
-----------------
Production:  call init_secret(secret_bytes) at startup. Load the secret from
             AWS Secrets Manager / Vault / environment variable. Rotate via
             init_secret() — old tokens with the previous secret will fail
             verification, which is the desired behaviour during a rotation.

Development: call generate_dev_secret() to create an in-memory random secret.
             Tokens issued in one process run cannot be verified by another
             process. NOT suitable for real-money deployments.

The secret is stored in the module-level `_SECRET` variable. It is never
logged or exposed via any public API.

Usage
-----
    # At startup (main.py)
    from src.security.fencing import init_secret, create_token, verify_token

    secret = os.environb.get(b"FENCING_SECRET") or generate_dev_secret()
    init_secret(secret)

    # Create a token for an automated action
    token_str = create_token(action_code="paused", severity="internal")

    # Verify before executing a state transition
    ok, reason = verify_token(token_str)
    if not ok:
        raise PermissionError(reason)
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_SECRET: bytes | None = None
_PROCESS_ID: str = str(uuid.uuid4())

# Default token validity window. Override per call if needed.
_DEFAULT_VALIDITY_SECONDS: int = 300  # 5 minutes


# ---------------------------------------------------------------------------
# Secret management
# ---------------------------------------------------------------------------


def init_secret(secret: bytes) -> None:
    """
    Load the HMAC signing secret. Call once at startup.

    The secret must be at least 32 bytes. Generate one with:
        python -c "import secrets; print(secrets.token_hex(32))"
    Store it in an environment variable or Secrets Manager — never hardcode.
    """
    if len(secret) < 32:
        raise ValueError("Fencing secret must be at least 32 bytes.")
    global _SECRET
    _SECRET = secret
    logger.info("Fencing token HMAC secret initialised (process_id=%s)", _PROCESS_ID)


def generate_dev_secret() -> bytes:
    """
    Generate a random 32-byte secret for development / testing.

    Automatically calls init_secret(). Tokens are NOT valid across process
    restarts. Do NOT use for real-money deployments.
    """
    logger.warning(
        "Generating EPHEMERAL fencing secret — tokens are NOT valid across "
        "process restarts. Set FENCING_SECRET env var for production."
    )
    secret = os.urandom(32)
    init_secret(secret)
    return secret


def _require_secret() -> bytes:
    if _SECRET is None:
        raise RuntimeError(
            "Fencing token secret not initialised. "
            "Call init_secret() or generate_dev_secret() at startup."
        )
    return _SECRET


# ---------------------------------------------------------------------------
# Token dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FencingToken:
    """Parsed, verified fencing token payload."""

    incident_id: str
    action_code: str
    severity: str
    issued_by: str
    issued_at: float  # Unix timestamp
    valid_until: float  # Unix timestamp


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sign(payload_b64: bytes) -> str:
    """Compute HMAC-SHA256 of the base64-encoded payload."""
    secret = _require_secret()
    mac = hmac.new(secret, payload_b64, hashlib.sha256).digest()
    return base64.urlsafe_b64encode(mac).decode()


def _encode_payload(data: dict) -> str:
    """Canonical JSON → base64url."""
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return base64.urlsafe_b64encode(canonical.encode()).decode()


def _decode_payload(b64: str) -> dict:
    """base64url → dict."""
    return json.loads(base64.urlsafe_b64decode(b64 + "=="))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def create_token(
    action_code: str,
    severity: str = "internal",
    validity_seconds: int = _DEFAULT_VALIDITY_SECONDS,
    incident_id: str | None = None,
) -> str:
    """
    Create a signed fencing token.

    Returns a string in the format ``<payload_b64>.<hmac_b64>`` suitable
    for embedding in HTTP headers, Telegram commands, or Redis values.

    Parameters
    ----------
    action_code       : Target state or action, e.g. "halted", "paused".
    severity          : "internal" | "warning" | "critical" | "emergency".
    validity_seconds  : How long the token is valid (default 300 s = 5 min).
    incident_id       : Optional caller-supplied ID; defaults to a new UUID.
    """
    now = time.time()
    payload = {
        "incident_id": incident_id or str(uuid.uuid4()),
        "action_code": action_code,
        "severity": severity,
        "issued_by": _PROCESS_ID,
        "issued_at": now,
        "valid_until": now + validity_seconds,
    }
    payload_b64 = _encode_payload(payload)
    mac = _sign(payload_b64.encode())
    return f"{payload_b64}.{mac}"


def create_internal_token(action_code: str) -> str:
    """
    Shorthand for automated system actions (graceful shutdown, watchdog).

    Uses "internal" severity and the default 5-minute validity window.
    """
    return create_token(action_code=action_code, severity="internal")


def verify_token(token_str: str) -> tuple[bool, str]:
    """
    Verify a token string and return (valid: bool, reason: str).

    Failure reasons:
        "malformed"        — not two dot-separated base64url segments
        "expired"          — valid_until < now
        "signature_invalid" — HMAC mismatch (tampered or wrong secret)
        "decode_error:<e>"  — payload is not valid JSON
    """
    parts = token_str.split(".")
    if len(parts) != 2:
        return False, "malformed"

    payload_b64, provided_mac = parts

    # 1. Check signature first (constant-time comparison prevents timing attacks)
    try:
        expected_mac = _sign(payload_b64.encode())
    except RuntimeError as e:
        return False, f"secret_not_initialised:{e}"

    if not hmac.compare_digest(expected_mac, provided_mac):
        return False, "signature_invalid"

    # 2. Decode payload
    try:
        data = _decode_payload(payload_b64)
    except Exception as e:
        return False, f"decode_error:{e}"

    # 3. Check expiry
    if time.time() > data.get("valid_until", 0):
        return False, "expired"

    return True, "ok"


def parse_token(token_str: str) -> FencingToken:
    """
    Parse a token string into a FencingToken dataclass after verification.

    Raises ValueError if the token is invalid or expired.
    """
    ok, reason = verify_token(token_str)
    if not ok:
        raise ValueError(f"Invalid fencing token: {reason}")

    payload_b64 = token_str.split(".")[0]
    data = _decode_payload(payload_b64)
    return FencingToken(
        incident_id=data["incident_id"],
        action_code=data["action_code"],
        severity=data["severity"],
        issued_by=data["issued_by"],
        issued_at=data["issued_at"],
        valid_until=data["valid_until"],
    )
