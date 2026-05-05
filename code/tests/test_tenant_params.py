"""
Tests for stage9.tenant_params.TenantParams.

Unit tests use an in-memory fake connection.
Integration tests hit stage9_dev; auto-skip if Postgres unreachable.
"""

from __future__ import annotations

import uuid
from decimal import Decimal

import pytest

import infrastructure.seed as seed
from infrastructure.tenant_params import TenantParams, UnknownParamError


# ---------- unit tests ----------

class FakeCursor:
    def __init__(self, select_rows=None) -> None:
        self.select_rows = select_rows or []
        self.executed: list[tuple[str, tuple]] = []
        self.rowcount = 0

    def execute(self, sql: str, params: tuple) -> None:
        self.executed.append((sql, params))
        if sql.lstrip().startswith("SELECT"):
            self._last_select = list(self.select_rows)
            self.rowcount = len(self._last_select)
        elif sql.lstrip().startswith("UPDATE"):
            # Pretend the update matched exactly one row.
            self.rowcount = 1

    def fetchall(self) -> list:
        return self._last_select

    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(self, *exc) -> None:
        return None


class FakeConn:
    def __init__(self, select_rows=None) -> None:
        self.cur = FakeCursor(select_rows)

    def cursor(self) -> FakeCursor:
        return self.cur


def test_load_runs_single_query():
    rows = [("calibration_update_rate", Decimal("0.10")), ("confidence_floor", Decimal("0.30"))]
    conn = FakeConn(select_rows=rows)
    tp = TenantParams.load(str(uuid.uuid4()), conn)
    # Exactly one execute() call — the single SELECT at PRELOADING.
    selects = [e for e in conn.cur.executed if e[0].lstrip().startswith("SELECT")]
    assert len(selects) == 1
    assert len(tp) == 2


def test_get_returns_float():
    tp = TenantParams("t1", {"confidence_floor": Decimal("0.30")})
    value = tp.get("confidence_floor")
    assert value == 0.30
    assert isinstance(value, float)


def test_get_unknown_raises_named_exception():
    tp = TenantParams("t1", {"confidence_floor": Decimal("0.30")})
    with pytest.raises(UnknownParamError, match="not found"):
        tp.get("nonexistent_param")


def test_unknown_param_error_is_not_keyerror():
    # Missing params should raise UnknownParamError, NOT a plain KeyError.
    # Broad `except KeyError` must NOT swallow it.
    tp = TenantParams("t1", {})
    try:
        tp.get("missing")
    except UnknownParamError as e:
        assert not isinstance(e, KeyError)
    else:
        pytest.fail("UnknownParamError was not raised")


def test_update_applies_exponential_smoothing_formula():
    # Formula: new = prior + rate * (evidence - prior)
    # prior = 0.50, rate = 0.10, evidence = 0.60
    # new   = 0.50 + 0.10 * (0.60 - 0.50) = 0.50 + 0.01 = 0.51
    tp = TenantParams(
        "t1",
        {
            "confidence_base_cold_start": Decimal("0.50"),
            "calibration_update_rate":    Decimal("0.10"),
        },
    )
    conn = FakeConn()
    new_value = tp.update("confidence_base_cold_start", 0.60, conn)
    assert new_value == pytest.approx(0.51)


def test_update_refreshes_in_memory_snapshot():
    tp = TenantParams(
        "t1",
        {
            "confidence_base_cold_start": Decimal("0.50"),
            "calibration_update_rate":    Decimal("0.10"),
        },
    )
    conn = FakeConn()
    tp.update("confidence_base_cold_start", 0.60, conn)
    # Subsequent get() sees the updated value, not the stale 0.50.
    assert tp.get("confidence_base_cold_start") == pytest.approx(0.51)


def test_update_unknown_raises():
    tp = TenantParams(
        "t1",
        {"calibration_update_rate": Decimal("0.10")},
    )
    with pytest.raises(UnknownParamError, match="not in snapshot"):
        tp.update("nonexistent", 0.5, FakeConn())


def test_update_without_calibration_rate_raises():
    tp = TenantParams("t1", {"confidence_floor": Decimal("0.30")})
    with pytest.raises(UnknownParamError, match="calibration_update_rate missing"):
        tp.update("confidence_floor", 0.5, FakeConn())


# BV-01 — calibration_update_rate = 0.0 (frozen param)
def test_update_with_zero_rate_leaves_value_unchanged():
    """BV-01: rate=0 → new = prior + 0*(evidence-prior) = prior exactly."""
    tp = TenantParams(
        "t1",
        {
            "confidence_base_cold_start": Decimal("0.50"),
            "calibration_update_rate":    Decimal("0.00"),
        },
    )
    new_value = tp.update("confidence_base_cold_start", 0.80, FakeConn())
    assert new_value == pytest.approx(0.50)
    assert tp.get("confidence_base_cold_start") == pytest.approx(0.50)


# BV-02 — calibration_update_rate = 1.0 (instant replacement)
def test_update_with_rate_one_replaces_value():
    """BV-02: rate=1 → new = prior + 1*(evidence-prior) = evidence exactly."""
    tp = TenantParams(
        "t1",
        {
            "confidence_base_cold_start": Decimal("0.50"),
            "calibration_update_rate":    Decimal("1.00"),
        },
    )
    new_value = tp.update("confidence_base_cold_start", 0.80, FakeConn())
    assert new_value == pytest.approx(0.80)
    assert tp.get("confidence_base_cold_start") == pytest.approx(0.80)


# BV-03 — evidence outside [0, 1] passes through without clamping
def test_update_evidence_above_one_passthrough():
    """BV-03: evidence > 1.0 is accepted; formula is applied as-is (no clamp)."""
    tp = TenantParams(
        "t1",
        {
            "confidence_base_cold_start": Decimal("0.50"),
            "calibration_update_rate":    Decimal("0.10"),
        },
    )
    # 0.50 + 0.10 * (1.50 - 0.50) = 0.60
    new_value = tp.update("confidence_base_cold_start", 1.50, FakeConn())
    assert new_value == pytest.approx(0.60)


# EH-03 — load() with 0 rows returns an empty TenantParams
def test_load_with_zero_rows_returns_empty_tenant_params():
    """EH-03: a tenant that was never seeded returns an empty (not erroring) snapshot."""
    conn = FakeConn(select_rows=[])
    tp = TenantParams.load(str(uuid.uuid4()), conn)
    assert len(tp) == 0


def test_get_on_empty_snapshot_raises_unknown_param_error():
    """EH-03: any get() on an empty snapshot must raise UnknownParamError."""
    tp = TenantParams("t1", {})
    with pytest.raises(UnknownParamError):
        tp.get("confidence_floor")


# SD-04 — update() when the DB row is missing raises UnknownParamError
def test_update_rowcount_zero_raises_unknown_param_error():
    """SD-04: update() raises UnknownParamError when DB UPDATE matches 0 rows."""
    class _ZeroRowCursor:
        executed: list = []
        rowcount: int = 0

        def execute(self, sql, params):
            self.executed.append((sql, params))

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

    class _ZeroRowConn:
        cur = _ZeroRowCursor()

        def cursor(self):
            return self.cur

    tp = TenantParams(
        "t1",
        {
            "confidence_base_cold_start": Decimal("0.50"),
            "calibration_update_rate":    Decimal("0.10"),
        },
    )
    with pytest.raises(UnknownParamError, match="not found in DB"):
        tp.update("confidence_base_cold_start", 0.60, _ZeroRowConn())


# ---------- _assert_params_complete ----------

from handlers.perceiving import _assert_params_complete
from infrastructure.errors import TenantParamNotFoundError
from infrastructure.tenant_params_defaults import VALID_PARAM_NAMES


def test_assert_params_complete_passes_when_all_present():
    values = {name: Decimal("0.5") for name in VALID_PARAM_NAMES}
    params = TenantParams("t1", values)
    _assert_params_complete("t1", params)  # must not raise


def test_assert_params_complete_raises_on_single_missing_param():
    missing = next(iter(sorted(VALID_PARAM_NAMES)))
    values = {name: Decimal("0.5") for name in VALID_PARAM_NAMES if name != missing}
    params = TenantParams("t1", values)
    with pytest.raises(TenantParamNotFoundError, match=missing):
        _assert_params_complete("t1", params)


def test_assert_params_complete_error_includes_tenant_id():
    params = TenantParams("tenant-xyz", {})
    with pytest.raises(TenantParamNotFoundError, match="tenant-xyz"):
        _assert_params_complete("tenant-xyz", params)


def test_assert_params_complete_error_states_missing_count():
    one_present = next(iter(sorted(VALID_PARAM_NAMES)))
    params = TenantParams("t1", {one_present: Decimal("0.5")})
    with pytest.raises(TenantParamNotFoundError) as exc_info:
        _assert_params_complete("t1", params)
    assert str(len(VALID_PARAM_NAMES) - 1) in str(exc_info.value)


def test_assert_params_complete_empty_snapshot_raises():
    params = TenantParams("t1", {})
    with pytest.raises(TenantParamNotFoundError):
        _assert_params_complete("t1", params)


# ---------- integration tests ----------

def _pg_connect_or_skip():
    try:
        import psycopg2
    except ImportError:
        pytest.skip("psycopg2 not installed")
    try:
        from infrastructure.config import DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            dbname=DB_NAME,
            connect_timeout=3,
        )
    except Exception as e:
        pytest.skip(f"Postgres unreachable: {e}")
    return conn


@pytest.fixture
def seeded_tenant():
    conn = _pg_connect_or_skip()
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS tenant_learning_params (
                tenant_id             UUID           NOT NULL,
                param_name            VARCHAR(100)   NOT NULL,
                starting_value        DECIMAL(12,6)  NOT NULL,
                current_value         DECIMAL(12,6)  NOT NULL,
                confidence_in_value   DECIMAL(3,2)   DEFAULT 0.10,
                total_evidence_runs   INTEGER        DEFAULT 0,
                last_updated_run_id   UUID,
                last_updated_at       TIMESTAMP      DEFAULT NOW(),
                PRIMARY KEY (tenant_id, param_name)
            );
            """
        )
    conn.commit()

    tenant_id = str(uuid.uuid4())
    seed.seed_tenant_params(tenant_id, "new", conn=conn)
    conn.commit()

    yield tenant_id, conn

    with conn.cursor() as cur:
        cur.execute("DELETE FROM tenant_learning_params WHERE tenant_id = %s", (tenant_id,))
    conn.commit()
    conn.close()


def test_integration_load_returns_50_params(seeded_tenant):
    tenant_id, conn = seeded_tenant
    tp = TenantParams.load(tenant_id, conn)
    assert len(tp) == 58


def test_integration_get_matches_seeded_values(seeded_tenant):
    tenant_id, conn = seeded_tenant
    tp = TenantParams.load(tenant_id, conn)
    assert tp.get("decision_gate_threshold") == pytest.approx(0.70)
    assert tp.get("confidence_floor") == pytest.approx(0.30)
    assert tp.get("confidence_ceiling") == pytest.approx(0.95)
    assert tp.get("stage8_penalty_threshold") == pytest.approx(0.60)


def test_integration_update_persists_to_db(seeded_tenant):
    tenant_id, conn = seeded_tenant
    tp = TenantParams.load(tenant_id, conn)
    # prior = 0.50, rate = 0.10, evidence = 0.70 → 0.50 + 0.10*(0.70-0.50) = 0.52
    new_value = tp.update("confidence_base_cold_start", 0.70, conn)
    conn.commit()
    assert new_value == pytest.approx(0.52)

    # Round-trip: a fresh load should see the new value in DB.
    tp2 = TenantParams.load(tenant_id, conn)
    assert tp2.get("confidence_base_cold_start") == pytest.approx(0.52)


def test_integration_update_unknown_raises(seeded_tenant):
    tenant_id, conn = seeded_tenant
    tp = TenantParams.load(tenant_id, conn)
    with pytest.raises(UnknownParamError):
        tp.update("nonexistent_param", 0.5, conn)
