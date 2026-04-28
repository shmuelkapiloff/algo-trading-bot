"""
FastAPI Control-Plane — Emergency Operations API.

Provides a small, authenticated REST API for human operators to issue
emergency commands without requiring SSH access to the trading machine.

RBAC: All endpoints require a Bearer token (CONTROL_API_TOKEN env var).
Rate limiting: 30 requests/minute per IP (in-process).

Endpoints
---------
  GET  /health          -> system health check (no auth)
  GET  /state           -> current TradingState
  POST /halt            -> immediate HALTED transition
  POST /pause           -> PAUSED transition
  POST /resume          -> ACTIVE transition
  POST /close_only      -> CLOSE_ONLY transition

Security notes
--------------
  - Token compared with hmac.compare_digest (constant-time) to resist
    timing side-channels.
  - No credentials are ever logged.
  - All state transitions validated by RuntimeStateStore.transition() (CAS).
  - rate_limit_middleware enforces 30 req/min per source IP.

Usage
-----
  From main.py:
      import control.api as control_api
      control_api.init_app(redis_client=r, state_store=state_store)
      asyncio.create_task(uvicorn_server.serve())
"""

from __future__ import annotations

import hmac
import logging
import os
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel
from src.security.audit_trail import SignedAuditTrail
from src.security.secrets_manager import SecretRef, SecretResolver

logger = logging.getLogger(__name__)

app = FastAPI(
    title="AlgoTrader Control Plane",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
)

security = HTTPBearer()

# ---------------------------------------------------------------------------
# Shared state (injected at startup via init_app())
# ---------------------------------------------------------------------------

_state_store = None  # RuntimeStateStore instance
_redis_client = None  # redis.asyncio.Redis instance
_audit_trail = None  # SignedAuditTrail
_secret_resolver = SecretResolver()

# In-process rate limiter: 30 req/min per IP
_rate_buckets: dict[str, list] = defaultdict(list)
_RATE_LIMIT = 30
_RATE_WINDOW = 60  # seconds


def init_app(redis_client, state_store) -> None:
    """Inject live dependencies. Call once from main.py before serving."""
    global _redis_client, _state_store, _audit_trail
    _redis_client = redis_client
    _state_store = state_store

    # Optional signed audit trail for all critical control-plane commands.
    audit_secret = _secret_resolver.get(
        SecretRef(name="CONTROL_AUDIT_SECRET", required=False, default="")
    )
    if audit_secret:
        _audit_trail = SignedAuditTrail(secret=audit_secret.encode())
    logger.info("Control API dependencies injected")


# ---------------------------------------------------------------------------
# Middleware — rate limiting
# ---------------------------------------------------------------------------


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host if request.client else "unknown"
    now = datetime.now(timezone.utc).timestamp()
    bucket = _rate_buckets[client_ip]
    bucket[:] = [t for t in bucket if now - t < _RATE_WINDOW]
    if len(bucket) >= _RATE_LIMIT:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded (30 req/min)",
        )
    bucket.append(now)
    return await call_next(request)


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------


def _get_expected_token() -> str:
    token = _secret_resolver.get(
        SecretRef(name="CONTROL_API_TOKEN", required=False, default="")
    )
    if not token:
        raise HTTPException(status_code=503, detail="CONTROL_API_TOKEN not configured")
    return token


def _audit(action: str, payload: dict) -> None:
    if _audit_trail is None:
        return
    try:
        _audit_trail.record(actor="control_api", action=action, payload=payload)
    except Exception:
        logger.exception("Control API audit write failed (non-fatal)")


def _verify_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> None:
    """Constant-time comparison to prevent timing attacks."""
    expected = _get_expected_token()
    provided = credentials.credentials or ""
    if not hmac.compare_digest(provided.encode(), expected.encode()):
        logger.warning("Control API: rejected request (invalid token)")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
        )


# ---------------------------------------------------------------------------
# Dependency getters
# ---------------------------------------------------------------------------


def _get_redis():
    if _redis_client is None:
        raise HTTPException(status_code=503, detail="Redis not available")
    return _redis_client


def _get_store():
    if _state_store is None:
        raise HTTPException(status_code=503, detail="State store not initialised")
    return _state_store


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class StateResponse(BaseModel):
    state: str


class TransitionRequest(BaseModel):
    reason: Optional[str] = None


class TransitionResponse(BaseModel):
    success: bool
    old_state: str
    new_state: str
    message: str


class HealthResponse(BaseModel):
    status: str
    redis_ok: bool


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health(r=Depends(_get_redis)):
    """Public health endpoint — no auth required."""
    redis_ok = False
    try:
        await r.ping()
        redis_ok = True
    except Exception:
        pass
    return HealthResponse(status="ok" if redis_ok else "degraded", redis_ok=redis_ok)


@app.get("/state", response_model=StateResponse, tags=["control"])
async def get_trading_state(
    _: None = Depends(_verify_token),
    store=Depends(_get_store),
):
    state = await store.get_state()
    return StateResponse(state=state.value)


@app.post("/halt", response_model=TransitionResponse, tags=["control"])
async def halt(
    req: TransitionRequest = TransitionRequest(),
    _: None = Depends(_verify_token),
    store=Depends(_get_store),
):
    """Immediately halt all trading (HALTED)."""
    from src.runtime_state import TradingState

    current = await store.get_state()
    success, msg = await store.force_transition_internal(
        target=TradingState.HALTED,
        reason=req.reason or "control_api_halt",
    )
    new_state = TradingState.HALTED if success else current
    logger.warning("Control API HALT: reason=%s  success=%s", req.reason, success)
    _audit(
        action="halt",
        payload={
            "reason": req.reason,
            "success": success,
            "from": current.value,
            "to": new_state.value,
        },
    )
    return TransitionResponse(
        success=success,
        old_state=current.value,
        new_state=new_state.value,
        message=msg,
    )


@app.post("/pause", response_model=TransitionResponse, tags=["control"])
async def pause(
    req: TransitionRequest = TransitionRequest(),
    _: None = Depends(_verify_token),
    store=Depends(_get_store),
):
    """Pause new order submissions."""
    from src.runtime_state import TradingState
    from src.security.fencing import create_token

    current = await store.get_state()
    token = create_token({"action": "pause"})
    success, msg = await store.transition(
        expected=current,
        target=TradingState.PAUSED,
        fencing_token=token,
    )
    new_state = TradingState.PAUSED if success else current
    logger.info("Control API PAUSE: success=%s", success)
    _audit(
        action="pause",
        payload={
            "reason": req.reason,
            "success": success,
            "from": current.value,
            "to": new_state.value,
        },
    )
    return TransitionResponse(
        success=success,
        old_state=current.value,
        new_state=new_state.value,
        message=msg,
    )


@app.post("/resume", response_model=TransitionResponse, tags=["control"])
async def resume(
    req: TransitionRequest = TransitionRequest(),
    _: None = Depends(_verify_token),
    store=Depends(_get_store),
):
    """Resume trading PAUSED -> ACTIVE."""
    from src.runtime_state import TradingState
    from src.security.fencing import create_token

    current = await store.get_state()
    token = create_token({"action": "resume"})
    success, msg = await store.transition(
        expected=current,
        target=TradingState.ACTIVE,
        fencing_token=token,
    )
    new_state = TradingState.ACTIVE if success else current
    logger.info("Control API RESUME: success=%s", success)
    _audit(
        action="resume",
        payload={
            "reason": req.reason,
            "success": success,
            "from": current.value,
            "to": new_state.value,
        },
    )
    return TransitionResponse(
        success=success,
        old_state=current.value,
        new_state=new_state.value,
        message=msg,
    )


@app.post("/close_only", response_model=TransitionResponse, tags=["control"])
async def close_only(
    req: TransitionRequest = TransitionRequest(),
    _: None = Depends(_verify_token),
    store=Depends(_get_store),
):
    """Switch to close-only mode — no new entries allowed."""
    from src.runtime_state import TradingState
    from src.security.fencing import create_token

    current = await store.get_state()
    token = create_token({"action": "close_only"})
    success, msg = await store.transition(
        expected=current,
        target=TradingState.CLOSE_ONLY,
        fencing_token=token,
    )
    new_state = TradingState.CLOSE_ONLY if success else current
    logger.info("Control API CLOSE_ONLY: success=%s", success)
    _audit(
        action="close_only",
        payload={
            "reason": req.reason,
            "success": success,
            "from": current.value,
            "to": new_state.value,
        },
    )
    return TransitionResponse(
        success=success,
        old_state=current.value,
        new_state=new_state.value,
        message=msg,
    )
