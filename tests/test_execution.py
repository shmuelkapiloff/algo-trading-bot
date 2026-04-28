"""
Tests for src/execution/order_state_machine.py — OrderStateMachine

Covers:
  - register() adds order in NEW state
  - transition() legal path: NEW → SENT → ACK → PARTIAL → FILLED
  - transition() updates filled_qty, fill_price, broker_order_id
  - transition() rejects illegal moves (returns False, state unchanged)
  - transition() on unknown order_id returns False
  - transition() ignores moves FROM terminal state (idempotent)
  - is_terminal() reflects terminal states correctly
  - get_open_orders() excludes terminal orders
  - clear_terminal() removes terminal orders, returns count
  - get_orders_for_symbol() filters by symbol

Run: pytest tests/test_execution.py -v
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

# ---------------------------------------------------------------------------
# Load order_state_machine.py directly to avoid triggering execution/__init__.py
# (which transitively imports AlpacaBroker → alpaca-py, not installed in test env).
# order_state_machine.py has no non-stdlib dependencies.
# Must register in sys.modules *before* exec_module so @dataclass works correctly.
# ---------------------------------------------------------------------------
_OSM_PATH = (
    pathlib.Path(__file__).parent.parent
    / "src"
    / "execution"
    / "order_state_machine.py"
)
_MODULE_NAME = "trading_bot_osm_isolated"
_spec = importlib.util.spec_from_file_location(_MODULE_NAME, _OSM_PATH)
_osm_mod = importlib.util.module_from_spec(_spec)
sys.modules[_MODULE_NAME] = _osm_mod
_spec.loader.exec_module(_osm_mod)

OrderRecord = _osm_mod.OrderRecord
OrderState = _osm_mod.OrderState
OrderStateMachine = _osm_mod.OrderStateMachine
TERMINAL_STATES = _osm_mod.TERMINAL_STATES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _record(order_id: str = "o1", symbol: str = "AAPL") -> OrderRecord:
    return OrderRecord(order_id=order_id, symbol=symbol, strategy_name="test")


# ---------------------------------------------------------------------------
# OrderRecord
# ---------------------------------------------------------------------------


class TestOrderRecord:
    def test_new_record_defaults(self):
        r = _record()
        assert r.state == OrderState.NEW
        assert r.filled_qty == 0.0
        assert r.is_terminal() is False

    def test_is_terminal_for_filled(self):
        r = _record()
        r.state = OrderState.FILLED
        assert r.is_terminal() is True

    def test_is_terminal_for_all_terminals(self):
        for state in TERMINAL_STATES:
            r = _record()
            r.state = state
            assert r.is_terminal() is True

    def test_is_not_terminal_for_active_states(self):
        for state in [OrderState.NEW, OrderState.SENT, OrderState.ACK, OrderState.PARTIAL]:
            r = _record()
            r.state = state
            assert r.is_terminal() is False


# ---------------------------------------------------------------------------
# OrderStateMachine
# ---------------------------------------------------------------------------


class TestOrderStateMachine:
    def _machine_with_order(self, order_id: str = "o1", symbol: str = "AAPL") -> tuple:
        m = OrderStateMachine()
        r = _record(order_id=order_id, symbol=symbol)
        m.register(r)
        return m, r

    # --- register ----------------------------------------------------------

    def test_register_adds_order(self):
        m, r = self._machine_with_order()
        assert m.get("o1") is r

    def test_get_unknown_returns_none(self):
        m = OrderStateMachine()
        assert m.get("nonexistent") is None

    # --- legal transition path --------------------------------------------

    def test_new_to_sent(self):
        m, _ = self._machine_with_order()
        ok = m.transition("o1", OrderState.SENT, broker_order_id="b-123")
        assert ok is True
        assert m.get("o1").state == OrderState.SENT
        assert m.get("o1").broker_order_id == "b-123"

    def test_sent_to_ack(self):
        m, _ = self._machine_with_order()
        m.transition("o1", OrderState.SENT)
        ok = m.transition("o1", OrderState.ACK)
        assert ok is True
        assert m.get("o1").state == OrderState.ACK

    def test_ack_to_partial(self):
        m, _ = self._machine_with_order()
        m.transition("o1", OrderState.SENT)
        m.transition("o1", OrderState.ACK)
        ok = m.transition("o1", OrderState.PARTIAL, filled_qty=5)
        assert ok is True
        assert m.get("o1").state == OrderState.PARTIAL
        assert m.get("o1").filled_qty == 5

    def test_partial_to_filled(self):
        m, _ = self._machine_with_order()
        m.transition("o1", OrderState.SENT)
        m.transition("o1", OrderState.ACK)
        m.transition("o1", OrderState.PARTIAL, filled_qty=5)
        ok = m.transition("o1", OrderState.FILLED, filled_qty=10, fill_price=182.50)
        assert ok is True
        assert m.get("o1").state == OrderState.FILLED
        assert m.get("o1").filled_qty == 10
        assert m.get("o1").fill_price == pytest.approx(182.50)

    def test_ack_directly_to_filled(self):
        m, _ = self._machine_with_order()
        m.transition("o1", OrderState.SENT)
        m.transition("o1", OrderState.ACK)
        ok = m.transition("o1", OrderState.FILLED, filled_qty=10, fill_price=100.0)
        assert ok is True
        assert m.get("o1").state == OrderState.FILLED

    def test_sent_to_rejected(self):
        m, _ = self._machine_with_order()
        m.transition("o1", OrderState.SENT)
        ok = m.transition("o1", OrderState.REJECTED, error="insufficient funds")
        assert ok is True
        assert m.get("o1").state == OrderState.REJECTED
        assert m.get("o1").last_error == "insufficient funds"

    def test_ack_to_canceled(self):
        m, _ = self._machine_with_order()
        m.transition("o1", OrderState.SENT)
        m.transition("o1", OrderState.ACK)
        ok = m.transition("o1", OrderState.CANCELED)
        assert ok is True
        assert m.get("o1").state == OrderState.CANCELED

    def test_ack_to_expired(self):
        m, _ = self._machine_with_order()
        m.transition("o1", OrderState.SENT)
        m.transition("o1", OrderState.ACK)
        ok = m.transition("o1", OrderState.EXPIRED)
        assert ok is True
        assert m.get("o1").state == OrderState.EXPIRED

    # --- illegal transitions -----------------------------------------------

    def test_new_to_filled_is_illegal(self):
        m, _ = self._machine_with_order()
        ok = m.transition("o1", OrderState.FILLED)
        assert ok is False
        assert m.get("o1").state == OrderState.NEW  # state unchanged

    def test_new_to_ack_is_illegal(self):
        m, _ = self._machine_with_order()
        ok = m.transition("o1", OrderState.ACK)
        assert ok is False
        assert m.get("o1").state == OrderState.NEW

    def test_sent_to_partial_is_illegal(self):
        m, _ = self._machine_with_order()
        m.transition("o1", OrderState.SENT)
        ok = m.transition("o1", OrderState.PARTIAL)
        assert ok is False
        assert m.get("o1").state == OrderState.SENT

    def test_filled_to_canceled_is_illegal(self):
        """Terminal state cannot transition to anything."""
        m, _ = self._machine_with_order()
        m.transition("o1", OrderState.SENT)
        m.transition("o1", OrderState.ACK)
        m.transition("o1", OrderState.FILLED)
        ok = m.transition("o1", OrderState.CANCELED)
        assert ok is False
        assert m.get("o1").state == OrderState.FILLED  # unchanged

    def test_canceled_is_idempotent(self):
        m, _ = self._machine_with_order()
        m.transition("o1", OrderState.CANCELED)
        ok = m.transition("o1", OrderState.CANCELED)  # already canceled
        assert ok is False

    # --- unknown order_id -------------------------------------------------

    def test_transition_unknown_order_returns_false(self):
        m = OrderStateMachine()
        ok = m.transition("ghost", OrderState.SENT)
        assert ok is False

    # --- get_open_orders --------------------------------------------------

    def test_get_open_orders_excludes_terminal(self):
        m = OrderStateMachine()
        r1 = _record("o1", "AAPL")
        r2 = _record("o2", "MSFT")
        m.register(r1)
        m.register(r2)
        m.transition("o2", OrderState.SENT)
        m.transition("o2", OrderState.ACK)
        m.transition("o2", OrderState.FILLED)

        open_orders = m.get_open_orders()
        assert len(open_orders) == 1
        assert open_orders[0].order_id == "o1"

    def test_get_open_orders_empty(self):
        m = OrderStateMachine()
        assert m.get_open_orders() == []

    # --- get_orders_for_symbol --------------------------------------------

    def test_get_orders_for_symbol(self):
        m = OrderStateMachine()
        m.register(_record("o1", "AAPL"))
        m.register(_record("o2", "AAPL"))
        m.register(_record("o3", "MSFT"))
        aapl_orders = m.get_orders_for_symbol("AAPL")
        assert len(aapl_orders) == 2
        assert all(r.symbol == "AAPL" for r in aapl_orders)

    def test_get_orders_for_unknown_symbol(self):
        m = OrderStateMachine()
        m.register(_record("o1", "AAPL"))
        assert m.get_orders_for_symbol("NVDA") == []

    # --- clear_terminal ---------------------------------------------------

    def test_clear_terminal_removes_terminal(self):
        m = OrderStateMachine()
        m.register(_record("o1", "AAPL"))
        m.register(_record("o2", "MSFT"))
        m.transition("o1", OrderState.SENT)
        m.transition("o1", OrderState.ACK)
        m.transition("o1", OrderState.FILLED)

        removed = m.clear_terminal()
        assert removed == 1
        assert m.get("o1") is None
        assert m.get("o2") is not None

    def test_clear_terminal_returns_zero_when_none_terminal(self):
        m = OrderStateMachine()
        m.register(_record("o1", "AAPL"))
        assert m.clear_terminal() == 0

    def test_clear_terminal_removes_all_terminal_states(self):
        m = OrderStateMachine()
        for oid, target in [
            ("o1", OrderState.FILLED),
            ("o2", OrderState.CANCELED),
            ("o3", OrderState.REJECTED),
            ("o4", OrderState.EXPIRED),
        ]:
            r = _record(oid)
            m.register(r)
            # Move to terminal
            m.transition(oid, OrderState.SENT)
            if target == OrderState.REJECTED:
                m.transition(oid, target, error="test")
            elif target in (OrderState.CANCELED,):
                m.transition(oid, target)
            else:
                m.transition(oid, OrderState.ACK)
                m.transition(oid, target)

        removed = m.clear_terminal()
        assert removed == 4
        assert m.get_open_orders() == []
