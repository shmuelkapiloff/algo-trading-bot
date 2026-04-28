from __future__ import annotations

import pytest

from trading_bot.scripts.incident_drill import run_incident_drill


@pytest.mark.asyncio
async def test_incident_drill_passes():
    r = await run_incident_drill()
    assert r.failover_triggered is True
    assert r.close_only_transitioned is True
    assert r.returned_to_primary is True
