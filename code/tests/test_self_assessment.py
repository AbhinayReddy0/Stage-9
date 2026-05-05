"""
tests/test_self_assessment.py — SelfAssessmentEngine Done Criteria
==================================================================
10 test cases covering every acceptance criterion from the build plan.

D1  — stage9_self_assessment row written after every run
D2  — model_health dict has all 5 model name keys
D3  — degradation detected when mape_delta > 3%
D4  — NOT flagged when mape_delta == exactly 3% (strictly greater than)
D5  — high_fallback_alert fires when fallback > 10%
D6  — high_fallback_alert silent at exactly 10%
D7  — pct_auto_proceed computed correctly
D8  — no degradation when model_performance_s9 has no rows
D9  — stage9_self_assessment write is idempotent (ON CONFLICT DO NOTHING)
D10 — non-zero pattern_feedback_failures_count logs a WARNING

No database required — all DB interactions replaced with lightweight doubles.
"""

from __future__ import annotations

import time
import uuid
from unittest.mock import patch

import pytest

from infrastructure.constants import (
    ForecastStatus,
    Model,
    ProcessingTier,
)
from learning.self_assessment import (
    SelfAssessmentEngine,
    SKUResult,
)


# ===========================================================================
# Test doubles
# ===========================================================================

class _FakeCursor:
    """Records execute() calls; fetchall() returns whatever is seeded."""

    def __init__(self, rows=None):
        self._rows = rows or []
        self.executed: list[tuple] = []

    def execute(self, sql, params=None):
        self.executed.append((sql, params))

    def fetchall(self):
        return self._rows

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


class _FakeConn:
    """In-memory connection double. cursor() returns a pre-seeded _FakeCursor."""

    def __init__(self, perf_rows=None):
        # perf_rows: list of (assigned_model, avg_mape_30d, trend, mape_delta, sample_count)
        self._cursor = _FakeCursor(rows=perf_rows or [])
        self.committed = False
        self.rolled_back = False

    def cursor(self):
        return self._cursor

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True


# ===========================================================================
# Builder helpers
# ===========================================================================

def _make_results(
    total: int = 100,
    fallback_count: int = 0,
    forecasted_count: int = 80,
    tier: str = ProcessingTier.FULL,
) -> list[SKUResult]:
    results = []
    for i in range(total):
        results.append(SKUResult(
            sku_id=str(uuid.uuid4()),
            status=(
                ForecastStatus.FORECASTED
                if i < forecasted_count
                else ForecastStatus.NEEDS_ACKNOWLEDGMENT
            ),
            confidence_final=0.75,
            processing_tier=tier,
            assigned_model=Model.SES,
            used_fallback=(i < fallback_count),
            pattern_label="stable",
        ))
    return results


def _perf_rows(model, avg_mape, mape_delta, trend="stable"):
    """One model_performance_s9 row in tuple form."""
    return [(model, avg_mape, trend, mape_delta, 50)]


def _run(perf_rows=None, results=None, pf_failures=0):
    conn = _FakeConn(perf_rows=perf_rows)
    engine = SelfAssessmentEngine()
    return engine.run(
        tenant_id=str(uuid.uuid4()),
        run_id=str(uuid.uuid4()),
        results=results or _make_results(),
        pattern_feedback_failures_count=pf_failures,
        run_start_time=time.time(),
        execution_mode="full",
        conn=conn,
    ), conn


# ===========================================================================
# D1 — row written after every run
# ===========================================================================

def test_d1_assessment_row_written():
    result, conn = _run()
    # Verify INSERT was executed and committed
    inserts = [sql for sql, _ in conn.cursor().executed if "INSERT" in sql]
    assert len(inserts) == 1
    assert conn.committed is True


# ===========================================================================
# D2 — model_health has all 5 model keys
# ===========================================================================

def test_d2_model_health_has_all_five_keys():
    result, _ = _run()
    expected = {Model.NAIVE, Model.CROSTON, Model.PROPHET, Model.HOLTS_LINEAR, Model.SES}
    assert set(result.model_health.keys()) == expected


# ===========================================================================
# D3 — degradation detected when mape_delta > 3% (using 0.05 > 0.03)
# ===========================================================================

def test_d3_degradation_detected_above_threshold():
    rows = _perf_rows(Model.PROPHET, avg_mape=0.20, mape_delta=0.05, trend="degrading")
    result, _ = _run(perf_rows=rows)

    assert Model.PROPHET in result.degrading_models
    assert result.degradation_detected is True
    assert result.model_health[Model.PROPHET].is_degrading is True
    assert any("prophet" in r for r in result.recommendations)


# ===========================================================================
# D4 — NOT flagged when mape_delta == exactly 0.03 (strictly greater than)
# ===========================================================================

def test_d4_degradation_not_flagged_at_exact_threshold():
    rows = _perf_rows(Model.PROPHET, avg_mape=0.15, mape_delta=0.03, trend="stable")
    result, _ = _run(perf_rows=rows)

    assert Model.PROPHET not in result.degrading_models
    assert result.model_health[Model.PROPHET].is_degrading is False
    assert result.degradation_detected is False


# ===========================================================================
# D5 — high_fallback_alert fires when fallback > 10% (11/100)
# ===========================================================================

def test_d5_high_fallback_alert_above_threshold():
    result, _ = _run(results=_make_results(total=100, fallback_count=11))
    assert result.high_fallback_alert is True


# ===========================================================================
# D6 — high_fallback_alert silent at exactly 10% (10/100)
# ===========================================================================

def test_d6_high_fallback_alert_silent_at_threshold():
    result, _ = _run(results=_make_results(total=100, fallback_count=10))
    assert result.high_fallback_alert is False


# ===========================================================================
# D7 — pct_auto_proceed and pct_needs_review correct
# ===========================================================================

def test_d7_pct_auto_proceed_correct():
    result, _ = _run(results=_make_results(total=100, forecasted_count=80))
    assert result.pct_auto_proceed == pytest.approx(0.80, abs=1e-6)
    assert result.pct_needs_review == pytest.approx(0.20, abs=1e-6)


# ===========================================================================
# D8 — no degradation when model_performance_s9 has no rows
# ===========================================================================

def test_d8_no_degradation_when_no_model_perf_rows():
    result, _ = _run(perf_rows=[])

    assert result.degrading_models == []
    assert result.degradation_detected is False
    assert len(result.model_health) == 5
    for entry in result.model_health.values():
        assert entry.avg_mape_30d is None
        assert entry.is_degrading is False
        assert entry.trend == "unknown"


# ===========================================================================
# D9 — idempotent write (ON CONFLICT DO NOTHING — no crash on second call)
# ===========================================================================

def test_d9_write_is_idempotent():
    run_id = str(uuid.uuid4())
    tenant_id = str(uuid.uuid4())
    conn = _FakeConn()
    engine = SelfAssessmentEngine()

    kwargs = dict(
        tenant_id=tenant_id,
        run_id=run_id,
        results=_make_results(),
        pattern_feedback_failures_count=0,
        run_start_time=time.time(),
        execution_mode="full",
        conn=conn,
    )

    engine.run(**kwargs)
    engine.run(**kwargs)  # second call with same run_id must not raise

    inserts = [sql for sql, _ in conn.cursor().executed if "INSERT" in sql]
    # Both calls attempted the INSERT — idempotency is enforced by ON CONFLICT in SQL
    assert len(inserts) == 2


# ===========================================================================
# D10 — non-zero pattern_feedback_failures_count logs a WARNING
# ===========================================================================

def test_d10_pattern_feedback_failures_logs_warning():
    with patch("learning.self_assessment.log") as mock_log:
        _run(pf_failures=2)
        mock_log.warning.assert_called_once()
        args = mock_log.warning.call_args[0]
        assert "pattern_feedback" in args[0]
