"""
AlgoTrader Pro — Main Entry Point (Phase 1 Lean Skeleton).

Bootstrap order
---------------
1. Load environment variables (.env)
2. Configure logging
3. Initialise HMAC fencing secret
4. Connect to Redis
5. Connect to DB (SQLite dev / PostgreSQL prod), create tables
6. Instantiate core services: RuntimeStateStore, OmsLedger
7. Register SIGTERM / SIGINT graceful shutdown handlers
8. Start APScheduler (EOD strategy scan)
9. Run asyncio event loop

Environment variables (required for production)
-----------------------------------------------
DATABASE_URL     SQLAlchemy async URL
                 Dev:  sqlite+aiosqlite:///trading.db
                 Prod: postgresql+asyncpg://user:pass@host:5432/algotrader
REDIS_URL        Redis connection string (default: redis://localhost:6379)
FENCING_SECRET   Hex-encoded 32-byte HMAC secret (generate once, store in Vault)
                 If absent in dev, an ephemeral secret is generated with a warning.

Optional
--------
LOG_LEVEL        DEBUG | INFO | WARNING | ERROR  (default: INFO)
ALPACA_API_KEY   Alpaca paper/live API key
ALPACA_SECRET    Alpaca paper/live API secret
ALPACA_BASE_URL  https://paper-api.alpaca.markets  (paper default)
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from typing import NoReturn

import redis.asyncio as aioredis
from apscheduler.executors.pool import ThreadPoolExecutor
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.analysis.market_regime import MarketRegime
from src.analysis.regime_cache import get_regime_or_recompute
from src.config import get_settings
from src.data.fetcher import MarketDataFetcher
from src.data.models import Base, DATABASE_URL_ENV
from src.data.oms_ledger import OmsLedger
from src.events.bus import EventBus
from src.events.handlers import register_handlers
from src.monitoring.alerts import AlertDispatcher
from src.monitoring.canary_probe import CanaryProbe
from src.monitoring.tca import TcaMonitor
from src.portfolio.manager import PortfolioManager
from src.risk.phase1_suite import build_phase1_gateway
from src.risk.pre_trade_gateway import PreTradeGateway
from src.runtime_state import RuntimeStateStore, TradingState
from src.security.fencing import generate_dev_secret, init_secret
from src.shutdown import ShutdownDependencies, register_shutdown_handlers
from src.signals.mean_reversion import MeanReversionStrategy
from src.signals.momentum import MomentumStrategy

# ---------------------------------------------------------------------------
# Logging setup  (call before any other import that might log)
# ---------------------------------------------------------------------------


def _configure_logging() -> None:
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stdout,
    )
    # Suppress noisy library loggers in production
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("apscheduler").setLevel(logging.INFO)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dependency initialisation helpers
# ---------------------------------------------------------------------------


def _init_fencing_secret() -> None:
    """Load the HMAC fencing secret from env, or generate a dev one."""
    raw = os.getenv("FENCING_SECRET")
    if raw:
        try:
            secret = bytes.fromhex(raw)
        except ValueError:
            logger.critical(
                "FENCING_SECRET is not valid hex. "
                'Generate one with: python -c "import secrets; print(secrets.token_hex(32))"'
            )
            sys.exit(1)
        init_secret(secret)
        logger.info("HMAC fencing secret loaded from environment variable.")
    else:
        generate_dev_secret()
        logger.warning(
            "FENCING_SECRET not set — using ephemeral dev secret. "
            "DO NOT use in production."
        )


async def _init_redis(url: str) -> aioredis.Redis:
    client: aioredis.Redis = aioredis.from_url(
        url,
        decode_responses=False,  # raw bytes — RuntimeStateStore handles decoding
        socket_connect_timeout=5,
        socket_timeout=5,
    )
    try:
        await client.ping()
        logger.info("Redis connected: %s", url)
    except Exception as exc:
        logger.critical("Cannot connect to Redis (%s): %s — aborting.", url, exc)
        sys.exit(1)
    return client


async def _init_db(url: str) -> tuple[object, async_sessionmaker]:
    """Create the async engine, session factory, and ensure tables exist."""
    engine = create_async_engine(url, echo=False, pool_pre_ping=True)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        logger.info("DB tables verified / created: %s", url.split("@")[-1])

    return engine, session_factory


def _build_scheduler() -> AsyncIOScheduler:
    """
    Build the APScheduler instance.

    Jobs are registered separately (strategies/scheduler_jobs.py).
    The scheduler runs strategy scan callbacks in a ThreadPoolExecutor so
    they cannot block the asyncio event loop.
    """
    return AsyncIOScheduler(
        executors={"default": ThreadPoolExecutor(max_workers=2)},
        job_defaults={"coalesce": True, "max_instances": 1},
        timezone="America/New_York",  # NYSE timezone for EOD triggers
    )


# ---------------------------------------------------------------------------
# EOD scan placeholder — replace with real strategy imports in Phase 1
# ---------------------------------------------------------------------------


async def _eod_scan(
    state_store: RuntimeStateStore,
    oms_ledger: OmsLedger,
    fetcher: MarketDataFetcher,
    gateway: PreTradeGateway,
    portfolio: PortfolioManager,
    event_bus: EventBus,
    redis_client,
    strategies: list,
    universe: list[str],
    alpaca_trading_client,
) -> None:
    """
    EOD strategy scan — runs at 15:55 ET Monday–Friday.

    Flow
    ----
    1. Gate check: skip if state is not ACTIVE.
    2. Fetch SPY bars → detect / cache market regime.
    3. Filter universe by ADV (average daily volume).
    4. Fetch OHLCV bars for filtered universe.
    5. Run each strategy against bars + regime → collect SignalIntents.
    6. Size each signal via PortfolioManager.
    7. Run each sized signal through PreTradeGateway.
    8. Submit approved orders to Alpaca.
    9. Record NEW event in OMS ledger.
    10. Publish ORDER_SUBMITTED event to EventBus.
    """
    from datetime import timedelta
    from datetime import datetime as dt
    from src.data.models import OrderEventType
    from src.signals.models import OrderSide
    from src.events import topics as t

    if not await state_store.allows_new_orders():
        state = await state_store.get_state()
        logger.info("EOD scan skipped — trading state is %s", state.value)
        return

    logger.info("EOD scan starting")
    settings = get_settings()
    today = dt.utcnow()
    start = today - timedelta(days=365)  # 1 year of daily bars for indicators

    # ── Step 2: Market regime ────────────────────────────────────────
    try:
        spy_bars = await fetcher.get_daily_bars(
            ["SPY"], start=start, end=today, limit=300
        )
        spy_df = spy_bars.get("SPY")
        if spy_df is None or spy_df.empty:
            logger.error("EOD scan aborted: no SPY bars returned")
            return
        regime = await get_regime_or_recompute(redis_client, spy_df)
        logger.info("Market regime: %s", regime.value)
    except Exception:
        logger.exception("EOD scan: failed to determine market regime — aborting")
        return

    # ── Step 3: Universe filter by ADV ───────────────────────────────
    try:
        filtered = await fetcher.filter_by_adv(
            universe,
            min_adv_usd=(
                settings.risk.min_adv_usd
                if hasattr(settings.risk, "min_adv_usd")
                else 1_000_000
            ),
        )
        logger.info("Universe after ADV filter: %d symbols", len(filtered))
        if not filtered:
            logger.warning("EOD scan: empty universe after ADV filter — done")
            return
    except Exception:
        logger.exception("EOD scan: ADV filter failed — using raw universe")
        filtered = universe

    # ── Step 4: Fetch bars for universe ──────────────────────────────
    try:
        all_bars = await fetcher.get_daily_bars(
            filtered, start=start, end=today, limit=300
        )
    except Exception:
        logger.exception("EOD scan: failed to fetch bars for universe — aborting")
        return

    # ── Steps 5–9: Signal generation → risk → submission ────────────
    submitted = 0
    for strategy in strategies:
        signals = list(strategy.generate_signals(all_bars, regime))
        logger.info("[%s] Generated %d signals", strategy.name, len(signals))

        for signal in signals:
            # Check portfolio pre-conditions
            ok, reason = await portfolio.can_open_position(signal)
            if not ok:
                logger.debug(
                    "[%s] %s skipped: %s", strategy.name, signal.symbol, reason
                )
                continue

            # Get latest price for sizing
            try:
                latest = await fetcher.get_latest_bar(signal.symbol)
                last_price = float(latest.get("close_adj", 0))
            except Exception:
                logger.warning(
                    "Could not get latest price for %s — skipping", signal.symbol
                )
                continue

            if last_price <= 0:
                continue

            # Size the position
            shares = await portfolio.size_signal(signal, last_price)
            if shares <= 0:
                continue

            sized_signal = signal.with_qty(shares)

            # Risk gateway check
            portfolio_state = portfolio.build_portfolio_state()
            gate_result = await gateway.admit_order(sized_signal, portfolio_state)
            if not gate_result.approved:
                logger.info(
                    "[%s] %s rejected by gate %s: %s",
                    strategy.name,
                    signal.symbol,
                    gate_result.gate_name,
                    gate_result.reason,
                )
                continue

            effective_signal = gate_result.modified_signal or sized_signal

            # Submit to Alpaca (paper)
            try:
                from alpaca.trading.requests import MarketOrderRequest
                from alpaca.trading.enums import OrderSide as AlpacaSide, TimeInForce

                alpaca_side = (
                    AlpacaSide.BUY
                    if effective_signal.side == OrderSide.BUY
                    else AlpacaSide.SELL
                )
                order_request = MarketOrderRequest(
                    symbol=effective_signal.symbol,
                    qty=effective_signal.qty,
                    side=alpaca_side,
                    time_in_force=TimeInForce.DAY,
                )
                order = await asyncio.to_thread(
                    alpaca_trading_client.submit_order, order_request
                )
                order_id = str(order.id)

                # Record NEW event
                await oms_ledger.record_event(
                    order_id=order_id,
                    symbol=effective_signal.symbol,
                    event_type=OrderEventType.NEW,
                    payload={
                        "side": effective_signal.side.value,
                        "qty": effective_signal.qty,
                        "strategy_name": effective_signal.strategy_name,
                        "confidence": effective_signal.confidence,
                    },
                    strategy=effective_signal.strategy_name,
                )

                # Publish event
                await event_bus.publish(
                    t.ORDER_SUBMITTED,
                    {
                        "order_id": order_id,
                        "symbol": effective_signal.symbol,
                        "side": effective_signal.side.value,
                        "qty": effective_signal.qty,
                        "strategy_name": effective_signal.strategy_name,
                    },
                )

                submitted += 1
                logger.info(
                    "Order submitted: %s %s x%d (strategy=%s)",
                    effective_signal.side.value.upper(),
                    effective_signal.symbol,
                    effective_signal.qty,
                    effective_signal.strategy_name,
                )

            except Exception:
                logger.exception(
                    "EOD scan: failed to submit order for %s", signal.symbol
                )

    logger.info("EOD scan complete — %d orders submitted", submitted)


# ---------------------------------------------------------------------------
# Main async entrypoint
# ---------------------------------------------------------------------------


async def main() -> None:
    load_dotenv()
    _configure_logging()

    logger.info("AlgoTrader Pro starting up…")

    # ------------------------------------------------------------------
    # 1. Security
    # ------------------------------------------------------------------
    _init_fencing_secret()

    # ------------------------------------------------------------------
    # 2. Infrastructure
    # ------------------------------------------------------------------
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    redis_client = await _init_redis(redis_url)

    db_url = os.getenv(DATABASE_URL_ENV, "sqlite+aiosqlite:///trading.db")
    db_engine, session_factory = await _init_db(db_url)

    # ------------------------------------------------------------------
    # 3. Core services
    # ------------------------------------------------------------------
    settings = get_settings()
    state_store = RuntimeStateStore(redis_client)
    oms_ledger = OmsLedger(session_factory)
    portfolio = PortfolioManager(redis_client, settings)
    await portfolio.load_state()

    # Confirm / set initial trading state
    current_state = await state_store.get_state()
    logger.info("Trading state on startup: %s", current_state.value)

    # ------------------------------------------------------------------
    # 3b. Alpaca trading client (paper by default)
    # ------------------------------------------------------------------
    from alpaca.trading.client import TradingClient

    alpaca_trading_client = TradingClient(
        api_key=settings.alpaca_api_key,
        secret_key=settings.alpaca_secret,
        paper=True,
    )
    fetcher = MarketDataFetcher(
        api_key=settings.alpaca_api_key,
        secret_key=settings.alpaca_secret,
    )

    # ------------------------------------------------------------------
    # 3c. Risk gateway (Phase 1 — see src/risk/phase1_suite.py)
    # ------------------------------------------------------------------
    gateway = build_phase1_gateway(portfolio, settings)

    # ------------------------------------------------------------------
    # 3d. Event bus + handlers
    # ------------------------------------------------------------------
    event_bus = EventBus()

    # Alert dispatcher (Telegram optional — uses log-only mode if not configured)
    alert_dispatcher = AlertDispatcher(
        bot_token=os.getenv("TELEGRAM_BOT_TOKEN"),
        chat_id=os.getenv("TELEGRAM_CHAT_ID"),
    )

    # TCA monitor — records fill quality metrics and triggers circuit breaker
    tca_monitor = TcaMonitor(
        redis_client=redis_client,
        state_store=state_store,
        alert_dispatcher=alert_dispatcher,
    )

    register_handlers(event_bus, oms_ledger, portfolio, tca_monitor=tca_monitor)
    await event_bus.start()

    # ------------------------------------------------------------------
    # 3e. Strategies
    # ------------------------------------------------------------------
    strategies = [
        MomentumStrategy(),
        MeanReversionStrategy(),
    ]

    # ------------------------------------------------------------------
    # 3f. Trading universe (load from config or watchlist file)
    # ------------------------------------------------------------------
    # Phase 1: static watchlist. Replace with dynamic loader in Phase 2.
    universe_file = "config/watchlist.txt"
    import pathlib

    if pathlib.Path(universe_file).exists():
        universe = [
            s.strip().upper()
            for s in pathlib.Path(universe_file).read_text().splitlines()
            if s.strip() and not s.startswith("#")
        ]
        logger.info("Universe loaded: %d symbols from %s", len(universe), universe_file)
    else:
        # Fallback: small default watchlist for smoke testing
        universe = [
            "AAPL",
            "MSFT",
            "AMZN",
            "GOOGL",
            "META",
            "NVDA",
            "JPM",
            "JNJ",
            "V",
            "UNH",
        ]
        logger.warning(
            "No watchlist file found at %s — using default 10-symbol universe",
            universe_file,
        )

    # ------------------------------------------------------------------
    # 4. Scheduler
    # ------------------------------------------------------------------
    scheduler = _build_scheduler()

    # EOD scan: Monday–Friday at 15:55 ET (5 minutes before NYSE close)
    scheduler.add_job(
        func=lambda: asyncio.create_task(
            _eod_scan(
                state_store=state_store,
                oms_ledger=oms_ledger,
                fetcher=fetcher,
                gateway=gateway,
                portfolio=portfolio,
                event_bus=event_bus,
                redis_client=redis_client,
                strategies=strategies,
                universe=universe,
                alpaca_trading_client=alpaca_trading_client,
            )
        ),
        trigger="cron",
        day_of_week="mon-fri",
        hour=15,
        minute=55,
        id="eod_scan",
        replace_existing=True,
    )

    # Canary probe: every 5 minutes during market hours (09:00–15:55 ET, Mon–Fri)
    canary_probe = CanaryProbe(
        fetcher=fetcher,
        redis_client=redis_client,
        alert_dispatcher=alert_dispatcher,
    )
    scheduler.add_job(
        func=lambda: asyncio.create_task(canary_probe.run()),
        trigger="cron",
        day_of_week="mon-fri",
        hour="9-15",
        minute="*/5",
        id="canary_probe",
        replace_existing=True,
    )

    # ------------------------------------------------------------------
    # 5. Signal queue (in-process; Phase 2 will replace with Redis Streams)
    # ------------------------------------------------------------------
    signal_queue: asyncio.Queue = asyncio.Queue(maxsize=256)

    # ------------------------------------------------------------------
    # 6. Graceful shutdown
    # ------------------------------------------------------------------
    loop = asyncio.get_running_loop()

    shutdown_deps = ShutdownDependencies(
        runtime_state=state_store,
        signal_queue=signal_queue,
        oms_ledger_flush=oms_ledger.flush,
        scheduler_shutdown=lambda: scheduler.shutdown(wait=False),
    )
    register_shutdown_handlers(loop, shutdown_deps)

    # ------------------------------------------------------------------
    # 7. Start
    # ------------------------------------------------------------------
    scheduler.start()
    logger.info(
        "AlgoTrader Pro ready. Scheduler running. "
        "Waiting for EOD trigger or shutdown signal."
    )

    # Block until loop.stop() is called by the shutdown handler.
    # In production this means: "wait for SIGTERM / SIGINT."
    stop_event = asyncio.Event()
    loop.add_signal_handler(signal.SIGTERM, stop_event.set)
    loop.add_signal_handler(signal.SIGINT, stop_event.set)
    await stop_event.wait()

    # ------------------------------------------------------------------
    # 8. Teardown (reached after shutdown handler completes)
    # ------------------------------------------------------------------
    logger.info("Stopping event bus…")
    await event_bus.stop()
    logger.info("Closing DB engine and Redis connection…")
    await db_engine.dispose()
    await redis_client.aclose()
    logger.info("AlgoTrader Pro shutdown complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run() -> NoReturn:
    """
    Top-level runner. Wraps asyncio.run() with a clean error boundary.

    Use this when launching from the command line:
        python -m trading_bot.main
    or via a process manager (systemd, supervisor):
        ExecStart=/opt/venv/bin/python -m trading_bot.main
    """
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass  # SIGINT is handled inside main(); this is the clean exit path
    sys.exit(0)


if __name__ == "__main__":
    run()
