"""
Fencing token creation and verification (RSA-4096 / SHA-256).

Tokens are signed JSON payloads used to authorize safety-critical state
transitions. A token is valid only if:
  1. Its RSA-4096 signature verifies against the loaded public key.
  2. It has not expired (valid_until > now).

Key loading
-----------
Production:  call init_keys(private_pem, public_pem) at startup with bytes
             loaded from AWS Secrets Manager, HashiCorp Vault, or equivalent.
             Never load key material from the filesystem in production.

Development: call generate_ephemeral_keys() to get a throwaway key pair.
             This pair is ephemeral — tokens issued in one process run cannot
             be verified by another process. NOT suitable for real-money trading.

Serialization
-------------
FencingToken.serialize() → URL-safe base64 string for transport over HTTP/Telegram.
FencingToken.from_string(s) → deserialize back to a FencingToken instance.

Security notes
--------------
- PKCS1v15 + SHA-256 is used rather than PSS to simplify cross-language
  interoperability. Switch to PSS for higher security margin if needed.
- The `cryptography` package (>=41.0) is required. Add it to requirements.txt.
"""

from __future__ import annotations

import base64
import json
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Optional

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

logger = logging.getLogger(__name__)

_PRIVATE_KEY: Optional[rsa.RSAPrivateKey] = None
_PUBLIC_KEY: Optional[rsa.RSAPublicKey] = None

# Per-process identity; used as the "issued_by" claim.
_PROCESS_ID: str = str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Key management
# ---------------------------------------------------------------------------


def init_keys(private_pem: bytes, public_pem: bytes) -> None:
    """Load an RSA-4096 key pair from PEM bytes. Call once at startup."""
    global _PRIVATE_KEY, _PUBLIC_KEY
    _PRIVATE_KEY = serialization.load_pem_private_key(private_pem, password=None)
    _PUBLIC_KEY = serialization.load_pem_public_key(public_pem)
    logger.info("Fencing-token RSA key pair loaded (process_id=%s)", _PROCESS_ID)


def generate_ephemeral_keys() -> tuple[bytes, bytes]:
    """
    Generate a throwaway RSA-4096 key pair for development/testing.

    Returns (private_pem, public_pem) as bytes. Automatically calls
    init_keys() so the module is ready to use immediately.

    WARNING: Do NOT use for real-money deployments. The private key exists
    only in memory for the lifetime of this process.
    """
    logger.warning(
        "Generating EPHEMERAL fencing-token keys — "
        "tokens are NOT valid across process restarts. "
        "Load persistent keys via init_keys() for production."
    )
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=4096)
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_pem = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    init_keys(private_pem, public_pem)
    return private_pem, public_pem


# ---------------------------------------------------------------------------
# Token dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FencingToken:
    """
    Immutable, signed authorization token for safety-critical state transitions.

    Fields
    ------
    incident_id    Globally unique ID for this incident/action.
    severity       "warning" | "critical" | "emergency"
    action_code    The specific action authorized: "pause_orders" |
                   "close_only" | "circuit_breaker" | "halted"
    issued_by      process_id of the issuing process.
    issued_at      UTC epoch seconds (float).
    valid_until    UTC epoch seconds (float); token expires after this time.
    signature      Base64-encoded RSA-4096/SHA-256 signature over the
                   canonical JSON of the above fields (sorted keys).
    """

    incident_id: str
    severity: str
    action_code: str
    issued_by: str
    issued_at: float
    valid_until: float
    signature: str

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def _payload_bytes(self) -> bytes:
        """Canonical JSON payload (no signature field) used for signing."""
        payload = {
            "incident_id": self.incident_id,
            "severity": self.severity,
            "action_code": self.action_code,
            "issued_by": self.issued_by,
            "issued_at": self.issued_at,
            "valid_until": self.valid_until,
        }
        return json.dumps(payload, sort_keys=True).encode()

    def serialize(self) -> str:
        """
        Encode the full token (including signature) as a URL-safe base64 string.
        Safe to embed in HTTP query params or Telegram messages.
        """
        full = {
            "incident_id": self.incident_id,
            "severity": self.severity,
            "action_code": self.action_code,
            "issued_by": self.issued_by,
            "issued_at": self.issued_at,
            "valid_until": self.valid_until,
            "signature": self.signature,
        }
        return base64.urlsafe_b64encode(json.dumps(full).encode()).decode()

    @classmethod
    def from_string(cls, token_str: str) -> "FencingToken":
        """
        Deserialize a token produced by serialize().

        Raises ValueError if the string is malformed.
        Does NOT verify the signature — call verify_token() for that.
        """
        try:
            raw = base64.urlsafe_b64decode(token_str.encode())
            data = json.loads(raw)
            return cls(
                incident_id=data["incident_id"],
                severity=data["severity"],
                action_code=data["action_code"],
                issued_by=data["issued_by"],
                issued_at=float(data["issued_at"]),
                valid_until=float(data["valid_until"]),
                signature=data["signature"],
            )
        except (KeyError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"Malformed fencing token: {exc}") from exc


# ---------------------------------------------------------------------------
# Token factory and verification
# ---------------------------------------------------------------------------


def create_token(
    action_code: str,
    severity: str = "critical",
    validity_seconds: int = 300,
    incident_id: Optional[str] = None,
) -> FencingToken:
    """
    Issue a new fencing token signed with the loaded private key.

    Raises RuntimeError if init_keys() has not been called.
    """
    if _PRIVATE_KEY is None:
        raise RuntimeError(
            "Fencing-token private key is not loaded. "
            "Call init_keys() or generate_ephemeral_keys() at startup."
        )

    now = time.time()
    token_id = incident_id or str(uuid.uuid4())

    # Build the unsigned token to get the canonical payload
    unsigned = FencingToken(
        incident_id=token_id,
        severity=severity,
        action_code=action_code,
        issued_by=_PROCESS_ID,
        issued_at=now,
        valid_until=now + validity_seconds,
        signature="",  # placeholder; replaced below
    )

    raw_sig = _PRIVATE_KEY.sign(
        unsigned._payload_bytes(),
        padding.PKCS1v15(),
        hashes.SHA256(),
    )
    sig_b64 = base64.b64encode(raw_sig).decode()

    # Return the final immutable token with the real signature
    import dataclasses

    return dataclasses.replace(unsigned, signature=sig_b64)


def create_internal_token(action_code: str, validity_seconds: int = 60) -> FencingToken:
    """
    Convenience wrapper for system-generated tokens (graceful shutdown,
    auto circuit-breaker). Uses a short TTL (60s default).
    """
    return create_token(
        action_code=action_code,
        severity="emergency",
        validity_seconds=validity_seconds,
        incident_id=f"internal_{action_code}_{uuid.uuid4().hex[:8]}",
    )


def verify_token(token: FencingToken) -> tuple[bool, str]:
    """
    Verify a FencingToken.

    Returns (is_valid: bool, reason: str).
    Checks:
      1. Public key is loaded.
      2. Token has not expired.
      3. RSA signature is valid.

    In HA deployments, add a fourth check that token.issued_by matches
    the current Redis-elected leader ID.
    """
    if _PUBLIC_KEY is None:
        return False, "public_key_not_loaded"

    if time.time() > token.valid_until:
        return False, "token_expired"

    try:
        raw_sig = base64.b64decode(token.signature)
        _PUBLIC_KEY.verify(
            raw_sig,
            token._payload_bytes(),
            padding.PKCS1v15(),
            hashes.SHA256(),
        )
    except InvalidSignature:
        return False, "invalid_signature"
    except Exception as exc:
        logger.error("Unexpected error during token verification: %s", exc)
        return False, f"verification_error:{type(exc).__name__}"

    return True, "valid"
