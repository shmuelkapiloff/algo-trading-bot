# Operational Protocol — Live Deployment Roadmap

This document governs the 9 remaining runtime/operational items in TRADING_BOT_PLAN.md.  
All code is complete. The steps below require live broker access and calendar time.

---

## Phase A — Paper Trading (90–180 days)

### A1. Start paper-trading session

```bash
# Set environment to Alpaca paper mode
export ALPACA_BASE_URL=https://paper-api.alpaca.markets
export ALPACA_PAPER=true

# Launch the bot
python main.py
```

Minimum duration: **90 calendar days** (full quarterly cycle).  
Recommended: **180 days** to cover at least one regime shift.

### A2. Close-only dry-run validation

Before any real capital is used, confirm close-only mode works end-to-end:

```bash
# With bot running in paper mode, trigger CLOSE_ONLY via control API
curl -X POST http://localhost:8000/close_only \
  -H "Authorization: Bearer $CONTROL_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"reason": "pre-live close-only validation"}'
```

Verify:
- All open paper positions are liquidated within one session.
- No new entries are placed after the transition.
- Audit trail entry is created and verifiable (`src/security/audit_trail.py`).

### A3. Daily result analysis

Run after market close each day:

```bash
# Canary gate — replace with actual 30-day metrics once available
python scripts/canary_gate.py \
  --sharpe-30d <SHARPE> \
  --max-dd-pct <MAX_DD>

# Replay historical stress scenarios to confirm stability
python scripts/historical_replay.py
```

Review the SLO dashboard:

```bash
python -m streamlit run dashboard/main.py
# Navigate to Reliability SLO page
```

Log anomalies in `docs/incident_log.md` (create as needed).

### A4. Fine-tune parameters

After 30 days of paper data, evaluate:

| Parameter | Location | Typical range |
|---|---|---|
| `entry_threshold` | `src/signals/signal_generator.py` | 0.5 – 0.7 |
| `position_size_pct` | `src/risk/position_sizer.py` | 0.01 – 0.05 |
| `stop_loss_pct` | `src/risk/stop_loss.py` | 0.01 – 0.03 |
| Walk-forward windows | `backtesting/walk_forward.py` | 63/21 – 126/42 days |

Run walk-forward to validate any change before applying:

```python
from backtesting.walk_forward import WalkForwardOptimizer
opt = WalkForwardOptimizer(train_days=63, test_days=21)
result = opt.run(bars, param_grid, metric_fn)
print(result.summarize())
```

### A5. Regime threshold calibration (30 / 60 / 90 day checkpoints)

At each checkpoint, compare paper results to regime labels produced by `src/analysis/market_regime.py`.  
Adjust thresholds if misclassification rate > 10%:

```python
# src/analysis/market_regime.py
BULL_THRESHOLD = 0.02   # recalibrate if needed
BEAR_THRESHOLD = -0.02
SIDEWAYS_THRESHOLD = 0.01
```

Document each calibration decision with date and rationale in `docs/regime_calibration_log.md`.

---

## Phase B — Canary Rollout

### Pre-requisites

Before moving to Canary 1, **all** of the following must hold over the final 30 paper-trading days:

| KPI | Gate |
|---|---|
| Sharpe ratio (30d) | ≥ 0.5 |
| Max drawdown | ≤ 10% |
| Fill rate | ≥ 95% |
| Latency p95 | ≤ 500 ms |
| Zero CRITICAL incidents | — |

Verify with:

```bash
python scripts/canary_gate.py \
  --sharpe-30d <VALUE> \
  --max-dd-pct <VALUE>
```

### B1. Canary 1 — 10% capital, 10 trading days

1. Obtain **explicit written approval** from account owner.
2. Set capital allocation:
   ```bash
   export CANARY_CAPITAL_PCT=0.10
   export ALPACA_PAPER=false
   ```
3. Run capital ramp drill first:
   ```bash
   python scripts/capital_ramp_drill.py
   ```
4. Launch:
   ```bash
   python main.py
   ```
5. Monitor daily using `python scripts/canary_gate.py` with live metrics.
6. After 10 trading days, run incident drill:
   ```bash
   python scripts/incident_drill.py
   ```

### B2. Canary 2 — 50% capital, after KPI gate

Same approval + gate process as B1.  
KPI gates are identical; all must pass on Canary 1 live data.

```bash
export CANARY_CAPITAL_PCT=0.50
python scripts/capital_ramp_drill.py
python main.py
```

### B3. Full deployment — 100% capital

Requires **explicit written approval** from account owner.  
All prior gates must pass on Canary 2 live data.

```bash
export CANARY_CAPITAL_PCT=1.00
python main.py
```

---

## Phase C — Ongoing Monitoring

### Monthly review checklist

Run on the first trading day of each month:

- [ ] Run `python scripts/historical_replay.py` — all scenarios PASS?
- [ ] Run `python scripts/canary_gate.py` with latest 30d metrics — all gates PASS?
- [ ] Review Grafana SLO dashboard (`observability/grafana/dashboards/algotrader_slo.json`)
- [ ] Check audit trail integrity:
  ```python
  from src.security.audit_trail import AuditTrail
  trail = AuditTrail("audit.log")
  assert trail.verify_chain(), "Audit chain broken — investigate immediately"
  ```
- [ ] Review broker health failover logs for unexpected failovers
- [ ] Check borrow availability for any new short signals
- [ ] Update `docs/regime_calibration_log.md` if thresholds changed
- [ ] File monthly performance report in `docs/performance_reports/YYYY-MM.md`

### Emergency procedures

See [docs/runbook_failover.md](runbook_failover.md) and [docs/runbook_rehearsal_drills.md](runbook_rehearsal_drills.md).

For immediate halt:

```bash
curl -X POST http://localhost:8000/halt \
  -H "Authorization: Bearer $CONTROL_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"reason": "emergency halt"}'
```

---

## Secrets & Security

All API keys must be loaded via `src/security/secrets_manager.py`.  
Never commit credentials. Rotate keys:

- Alpaca API key: every 90 days
- `CONTROL_API_TOKEN`: every 30 days (stored in environment or AWS Secrets Manager)
- Audit HMAC key (`AUDIT_HMAC_KEY`): rotate only when team membership changes; re-sign all past entries

---

## References

| Document | Purpose |
|---|---|
| [runbook_failover.md](runbook_failover.md) | Broker failover procedures |
| [runbook_rehearsal_drills.md](runbook_rehearsal_drills.md) | Drill scripts and pass/fail criteria |
| [../TRADING_BOT_PLAN.md](../../TRADING_BOT_PLAN.md) | Master plan (code items all ✅) |
| [../observability/README.md](../observability/README.md) | OTel + Grafana setup |
