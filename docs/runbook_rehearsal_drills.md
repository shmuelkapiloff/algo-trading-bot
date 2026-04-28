# Rehearsal Runbooks

This runbook defines recurring rehearsal drills required before real-money mode.

## Drill 1: Broker Outage Failover
- Trigger outage simulation (`python scripts/incident_drill.py`)
- Verify auto-failover to CLOSE_ONLY
- Verify return to PRIMARY after healthy checks

## Drill 2: Canary Capital Ramp
- Run `python scripts/capital_ramp_drill.py`
- Validate stage eligibility against paper-trading day count

## Drill 3: Forced Buy-In Simulation
- Simulate borrow recall by adding symbol to `no_borrow_symbols`
- Verify short entries are blocked at universe daily filters
- Verify alert is emitted and action logged in audit trail

## Drill 4: Historical Stress Replay
- Run `python scripts/historical_replay.py`
- Validate deterministic ledger digest output and FILLED terminal states

## Cadence
- Weekly in paper-trading
- After any execution/risk architecture change
