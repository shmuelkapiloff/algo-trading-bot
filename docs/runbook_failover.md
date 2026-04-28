# Broker Failover — Operational Runbook

> **Audience:** On-call engineer / trading system operator  
> **Applies to:** AlgoTrader Pro v1 — Phase 5.6  
> **Last updated:** April 2026

---

## 1. Overview

The system runs a `BrokerHealthMonitor` that polls the primary broker (Alpaca) every 5 seconds.
If 3 consecutive health-checks fail **or** latency exceeds 10 s, the system automatically:

1. Transitions `RuntimeStateStore` → `CLOSE_ONLY`
2. Generates an emergency fencing token
3. Routes all new fills through the secondary broker
4. Sends a Telegram alert

This runbook covers **manual intervention steps** for cases where automatic failover is insufficient.

---

## 2. Automatic Failover (Hands-Off)

| Trigger | Threshold | Action |
|---------|-----------|--------|
| 3 consecutive failed health-checks | 5 s polling interval | Auto-failover to secondary |
| Single health-check latency > 10 s | — | Auto-failover to secondary |
| Cascade: 3 CRITICAL incidents in 60 s | — | Circuit-breaker → EMERGENCY |

No operator action is required. Monitor the Telegram channel for `🔴 BROKER FAILOVER ACTIVATED`.

---

## 3. Manual Failover Trigger

Use this procedure when:
- You want to proactively fail over before a scheduled Alpaca maintenance window.
- Automatic failover is not triggering but you observe degraded fill quality.

### Step 1 — Set Redis broker-mode key

```bash
redis-cli SET algotrader:broker:mode FAILOVER
```

### Step 2 — Force state transition

```python
# In a Python shell or management script:
import asyncio
from trading_bot.src.runtime_state import RuntimeStateStore, TradingState

async def force_failover():
    store = RuntimeStateStore()
    await store.force_transition_internal(
        TradingState.CLOSE_ONLY, reason="manual_failover:maintenance"
    )

asyncio.run(force_failover())
```

### Step 3 — Verify

```bash
redis-cli GET algotrader:broker:mode          # should return FAILOVER
redis-cli GET algotrader:runtime:state        # should return CLOSE_ONLY
```

Check Telegram — you should receive a `🔴 Manual failover activated` alert.

---

## 4. Return to Primary

Performed automatically by `BrokerHealthMonitor._attempt_return_to_primary()` after
**dual-sync validation** (positions + OMS ledger). To force manual return:

### Step 1 — Verify primary is healthy

```bash
# Check Alpaca API status
curl -s https://api.alpaca.markets/v2/account -H "APCA-API-KEY-ID: $APCA_KEY" | jq .status
```

### Step 2 — Clear failover state

```bash
redis-cli SET algotrader:broker:mode PRIMARY
```

### Step 3 — Resume active trading

```python
async def resume():
    store = RuntimeStateStore()
    await store.force_transition_internal(
        TradingState.ACTIVE, reason="manual_return_to_primary"
    )

asyncio.run(resume())
```

### Step 4 — Verify

```bash
redis-cli GET algotrader:runtime:state   # should return ACTIVE
```

---

## 5. Incident Escalation

| State | Who to contact |
|-------|---------------|
| `CLOSE_ONLY` for > 15 min | Check BrokerHealthMonitor logs. Ping #trading-alerts Slack. |
| `HALTED` | Immediate escalation — page on-call lead. |
| OMS ledger mismatch after return | Do not resume. Run reconciliation manually (see §6). |

---

## 6. Manual OMS Reconciliation

If `BrokerHealthMonitor._attempt_return_to_primary()` detects a ledger mismatch and refuses
to return, run:

```bash
cd /path/to/trading_bot
python -m trading_bot.scripts.reconcile_oms --dry-run
# If output looks correct:
python -m trading_bot.scripts.reconcile_oms --apply
```

---

## 7. Fencing Token Audit

All emergency transitions issue a fencing token stored in Redis under:

```
algotrader:fencing:active_token
```

To inspect:

```bash
redis-cli GET algotrader:fencing:active_token | python -m json.tool
```

The `incident_id` field links the token to the incident log in `src/risk/incident_controller.py`.

---

## 8. SLO Thresholds (Reference)

| Metric | SLO | Breach Action |
|--------|-----|--------------|
| Broker health-check latency p95 | < 2 s | WARNING incident |
| Broker health-check latency p99 | < 10 s | Automatic failover |
| Fill reconciliation mismatch rate | < 0.1% per day | CRITICAL incident |
| Order state machine pending count | < 10 | Investigation |
| Time in CLOSE_ONLY state per day | < 30 min | Review |

---

## 9. Drill Checklist (Monthly)

- [ ] Simulate broker outage: kill mock Alpaca → verify auto-failover in < 15 s
- [ ] Verify Telegram alert received within 30 s of outage
- [ ] Verify no new BUY orders submitted in CLOSE_ONLY mode
- [ ] Verify SELL / reduce-only orders still route correctly
- [ ] Return to primary: verify dual-sync passes, state → ACTIVE
- [ ] Review OMS ledger: zero pending events after return
