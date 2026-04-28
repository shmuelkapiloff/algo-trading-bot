"""Tests for src/data/lineage_service.py."""
from __future__ import annotations

import hashlib
from datetime import date

import pandas as pd
import pytest

from src.data.lineage_service import LineageService, DatasetManifest


def _make_bars(symbol: str, closes: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"close_adj": closes}, index=range(len(closes)))


@pytest.fixture
def svc():
    return LineageService(session_factory=None)


class TestLineageService:
    @pytest.mark.asyncio
    async def test_record_snapshot_returns_manifest(self, svc):
        bars = {"AAPL": _make_bars("AAPL", [150.0, 151.0, 152.0])}
        manifest = await svc.record_snapshot(
            as_of_date=date(2024, 1, 15),
            data_source="alpaca",
            bars_by_symbol=bars,
        )
        assert isinstance(manifest, DatasetManifest)
        assert manifest.as_of_date == date(2024, 1, 15)
        assert manifest.data_source == "alpaca"
        assert manifest.snapshot_id != ""
        assert manifest.content_hash != ""

    @pytest.mark.asyncio
    async def test_content_hash_is_sha256(self, svc):
        bars = {"SPY": _make_bars("SPY", [400.0, 401.0])}
        manifest = await svc.record_snapshot(
            as_of_date=date(2024, 1, 15),
            data_source="polygon",
            bars_by_symbol=bars,
        )
        # Hash should be valid hex (64 chars for SHA-256)
        assert len(manifest.content_hash) == 64
        int(manifest.content_hash, 16)  # should not raise

    @pytest.mark.asyncio
    async def test_same_data_same_hash(self, svc):
        bars = {"MSFT": _make_bars("MSFT", [300.0, 301.0, 302.0])}
        m1 = await svc.record_snapshot(date(2024, 1, 10), "alpaca", bars)
        m2 = await svc.record_snapshot(date(2024, 1, 10), "alpaca", bars)
        assert m1.content_hash == m2.content_hash

    @pytest.mark.asyncio
    async def test_different_data_different_hash(self, svc):
        bars1 = {"AAPL": _make_bars("AAPL", [150.0, 151.0])}
        bars2 = {"AAPL": _make_bars("AAPL", [150.0, 152.0])}  # different close
        m1 = await svc.record_snapshot(date(2024, 1, 10), "alpaca", bars1)
        m2 = await svc.record_snapshot(date(2024, 1, 10), "alpaca", bars2)
        assert m1.content_hash != m2.content_hash

    @pytest.mark.asyncio
    async def test_verify_hash_true_for_same_data(self, svc):
        bars = {"AAPL": _make_bars("AAPL", [150.0, 151.0, 152.0])}
        manifest = await svc.record_snapshot(date(2024, 1, 15), "alpaca", bars)
        assert await svc.verify_hash(manifest.snapshot_id, bars) is True

    @pytest.mark.asyncio
    async def test_verify_hash_false_for_tampered_data(self, svc):
        bars = {"AAPL": _make_bars("AAPL", [150.0, 151.0, 152.0])}
        manifest = await svc.record_snapshot(date(2024, 1, 15), "alpaca", bars)
        tampered = {"AAPL": _make_bars("AAPL", [150.0, 151.0, 999.0])}
        assert await svc.verify_hash(manifest.snapshot_id, tampered) is False

    @pytest.mark.asyncio
    async def test_get_manifests_up_to_date(self, svc):
        bars = {"SPY": _make_bars("SPY", [400.0, 401.0])}
        await svc.record_snapshot(date(2024, 1, 10), "alpaca", bars)
        await svc.record_snapshot(date(2024, 1, 15), "alpaca", bars)
        await svc.record_snapshot(date(2024, 1, 20), "alpaca", bars)

        results = await svc.get_manifests_up_to(date(2024, 1, 15))
        assert all(m.as_of_date <= date(2024, 1, 15) for m in results)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_check_source_mismatch_detects_large_divergence(self, svc):
        alpaca_bars = {"AAPL": _make_bars("AAPL", [150.0, 151.0])}
        # Polygon has 2% different closes → above 0.5% tolerance
        polygon_bars = {"AAPL": _make_bars("AAPL", [153.0, 154.0])}
        mismatches = await svc.check_source_mismatch(alpaca_bars, polygon_bars)
        assert "AAPL" in mismatches
        # Value is float (max divergence pct)
        assert mismatches["AAPL"] > 0.005

    @pytest.mark.asyncio
    async def test_check_source_mismatch_no_divergence_within_tolerance(self, svc):
        alpaca_bars = {"SPY": _make_bars("SPY", [400.00, 401.00])}
        # 0.02% difference — within 0.5% tolerance
        polygon_bars = {"SPY": _make_bars("SPY", [400.08, 401.08])}
        mismatches = await svc.check_source_mismatch(alpaca_bars, polygon_bars)
        assert "SPY" not in mismatches

    @pytest.mark.asyncio
    async def test_quality_status_fail_for_empty_symbols(self, svc):
        """Symbols with fewer than 5 bars count as missing."""
        bars_sparse = {f"SYM{i}": pd.DataFrame({"close_adj": [100.0]}) for i in range(100)}
        manifest = await svc.record_snapshot(date(2024, 1, 15), "alpaca", bars_sparse)
        # All sparse → quality should fail or warn
        assert manifest.quality_status in ("warn", "fail")

    @pytest.mark.asyncio
    async def test_quality_status_pass_for_clean_data(self, svc):
        bars = {"SPY": _make_bars("SPY", list(range(100, 200)))}
        manifest = await svc.record_snapshot(date(2024, 1, 15), "alpaca", bars)
        assert manifest.quality_status == "pass"
