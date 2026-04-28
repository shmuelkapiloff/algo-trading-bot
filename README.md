# AlgoTrader Pro

Automated US equities trading bot for paper (and live) trading via Alpaca Markets.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  main.py — async bootstrap + APScheduler EOD scan (15:55 ET)│
│                                                              │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │  Strategies │  │PortfolioMgr  │  │ PreTradeGateway   │  │
│  │  Momentum   │  │Fixed-Frac    │  │ LiquidityGate     │  │
│  │  MeanRevert │  │Sizing        │  │ PortfolioRiskGate │  │
│  └──────┬──────┘  └──────┬───────┘  │ ExecutionReadiness│  │
│         │  SignalIntent  │          └─────────┬─────────┘  │
│         └────────────────┘                    │            │
│                                    ┌──────────▼──────────┐  │
│                                    │  ExecutionRouter     │  │
│                                    │  AlpacaBroker        │  │
│                                    └──────────┬──────────┘  │
│                                               │             │
│  ┌─────────────┐  ┌──────────────┐  ┌────────▼───────────┐ │
│  │  EventBus   │  │  OmsLedger   │  │  OrderStateMachine │ │
│  │  (asyncio)  │  │  (SQLAlchemy)│  │  NEW→SENT→FILLED   │ │
│  └─────────────┘  └──────────────┘  └────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
     │                      │
     │ Redis                │ PostgreSQL / SQLite
     │ (state + cache)      │ (OMS ledger)
     ▼                      ▼
┌──────────┐      ┌──────────────────┐
│ Dashboard│      │ Control API      │
│ Streamlit│      │ FastAPI :8000    │
│ :8501    │      │ (emergency halt) │
└──────────┘      └──────────────────┘
```

## Quick Start (Paper Trading)

### Prerequisites
- Python 3.11+
- Redis (local or Docker)
- Alpaca Markets account (free at alpaca.markets)

### Setup

```bash
# 1. Clone and install
cd trading_bot
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env with your Alpaca API keys

# 3. Apply DB migrations
alembic upgrade head

# 4. Run (paper trading by default)
python -m main
```

### Docker (recommended)

```bash
cp .env.example .env   # fill in API keys
docker compose up -d

# Monitor logs
docker compose logs -f trading-bot

# Open dashboard
open http://localhost:8501
```

## Configuration

All settings live in `config/config.yaml`. Environment variables override file values.

Key settings:
| Setting | Default | Description |
|---|---|---|
| `mode` | `paper` | `paper` or `live` |
| `risk.max_risk_per_trade` | `0.01` | 1% risk per trade |
| `risk.max_positions` | `10` | Max concurrent positions |
| `risk.max_global_open_risk` | `0.02` | 2% total portfolio risk cap |
| `eod_scan_trigger` | `15:55` | EOD scan time (ET) |

## Trading Strategies (Phase 1)

### Momentum
- Entry: Price > EMA50 AND RSI crosses above 35 AND MACD bullish
- Regimes: BULL, SIDEWAYS
- Min confidence: 0.55

### Mean Reversion
- Entry: Price ≤ Bollinger Lower AND RSI < 40 AND price > SMA200 × 0.85
- Regimes: SIDEWAYS, BULL
- Min confidence: 0.60

## Risk Management

### Pre-Trade Gateway (5 gates)
1. **LiquidityGate** — spread + ADV check
2. **PortfolioRiskGate** — sector/concentration limits
3. **ExecutionReadinessGate** — broker health check
4. *(Phase 2)* SignalViabilityGate — EV > 1.5× cost
5. *(Phase 2)* TailRiskGate — ES/VaR scenarios

### Position Sizing (Phase 1 — Fixed-Fractional)
```
shares = (equity × risk_per_trade) / (entry_price × stop_pct)
max_position = equity × absolute_max_position_pct
```

## Control Plane API

Emergency operations via REST (requires `CONTROL_API_TOKEN`):

```bash
# Check status
curl http://localhost:8000/health

# Pause trading
curl -X POST http://localhost:8000/pause \
  -H "Authorization: Bearer $CONTROL_API_TOKEN"

# Emergency halt
curl -X POST http://localhost:8000/halt \
  -H "Authorization: Bearer $CONTROL_API_TOKEN"
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest -v --cov=src
```

84+ tests covering all critical paths:
- Runtime state (CAS, fencing tokens)
- Pre-trade gateway (all 5 gates × all scenarios)
- Market regime detection
- Position sizing
- Momentum + mean-reversion strategies

## Environment Variables

See `.env.example` for the full list. Required:

| Variable | Description |
|---|---|
| `ALPACA_API_KEY` | Alpaca API key |
| `ALPACA_SECRET_KEY` | Alpaca secret key |
| `DATABASE_URL` | `sqlite+aiosqlite:///trading.db` (dev) or `postgresql+asyncpg://...` (prod) |
| `REDIS_URL` | `redis://localhost:6379/0` |
| `FENCING_SECRET` | 32+ byte hex string for HMAC tokens |
| `CONTROL_API_TOKEN` | Bearer token for the control-plane API |

## Project Structure

```
trading_bot/
├── config/               # config.yaml, strategies.yaml, watchlist.txt
├── src/
│   ├── analysis/         # market regime detection + Redis cache
│   ├── data/             # Alpaca fetcher, OMS models, SQLAlchemy
│   ├── events/           # async event bus, handlers, topics
│   ├── execution/        # broker interface, Alpaca adapter, router, reconcile
│   ├── monitoring/       # loguru config, Telegram alerts
│   ├── portfolio/        # position manager, Fixed-Fractional sizer, performance
│   ├── risk/             # PreTradeGateway, 5 risk gates, Phase1 factory
│   ├── security/         # HMAC-SHA256 fencing tokens
│   ├── signals/          # strategy ABC, momentum, mean-reversion
│   ├── config.py         # Pydantic Settings
│   ├── runtime_state.py  # Redis CAS trading state machine
│   └── shutdown.py       # SIGTERM/SIGINT graceful shutdown
├── control/              # FastAPI emergency control API
├── dashboard/            # Streamlit monitoring (read-only)
├── watchdog/             # asyncio event loop heartbeat monitor
├── tests/                # 84+ pytest tests
├── alembic/              # DB migrations
├── Dockerfile
├── docker-compose.yml
├── main.py               # Entry point
└── requirements.txt
```

## Security

- **Paper trading by default** — `ALPACA_PAPER=true` in all configs
- **HMAC-SHA256 fencing tokens** — all state transitions require a signed token
- **Non-root Docker** — runs as UID 1001
- **Constant-time token comparison** — `hmac.compare_digest` in control API
- **No credentials in logs** — explicit redaction throughout

## Phase Roadmap

| Phase | Status | Description |
|---|---|---|
| Phase 1 | ✅ Complete | Fixed-Fractional sizing, 2 strategies, paper trading |
| Phase 2 | Planned | Bayesian Kelly, Redis Streams, VWAP/TWAP execution |
| Phase 3 | Future | Short selling, options, backtesting engine |
