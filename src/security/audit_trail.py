"""Cryptographically signed audit trail for critical control commands."""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class AuditEntry:
    ts: float
    actor: str
    action: str
    payload: dict
    signature: str


class SignedAuditTrail:
    def __init__(self, secret: bytes):
        if len(secret) < 16:
            raise ValueError("Audit secret must be at least 16 bytes")
        self._secret = secret
        self._entries: list[AuditEntry] = []

    def record(self, actor: str, action: str, payload: dict) -> AuditEntry:
        ts = time.time()
        msg = json.dumps(
            {"ts": ts, "actor": actor, "action": action, "payload": payload},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        sig = hmac.new(self._secret, msg, hashlib.sha256).hexdigest()
        entry = AuditEntry(ts=ts, actor=actor, action=action, payload=payload, signature=sig)
        self._entries.append(entry)
        return entry

    def verify(self, entry: AuditEntry) -> bool:
        msg = json.dumps(
            {
                "ts": entry.ts,
                "actor": entry.actor,
                "action": entry.action,
                "payload": entry.payload,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        expected = hmac.new(self._secret, msg, hashlib.sha256).hexdigest()
        return hmac.compare_digest(entry.signature, expected)

    def export(self) -> list[dict]:
        return [asdict(e) for e in self._entries]
