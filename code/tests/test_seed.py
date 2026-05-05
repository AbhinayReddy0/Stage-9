"""
Tests for stage9.seed.seed_tenant_params.

Unit tests use a fake cursor/connection — no DB required.
Integration tests hit stage9_dev in local Docker; auto-skip if unreachable.
"""

from __future__ import annotations

import uuid
from decimal import Decimal

import pytest

from infrastructure.errors import UnknownParamError
from infrastructure.seed import seed_tenant_params
from infrastructure.tenant_params_defaults import TENANT_LEARNING_PARAMS_DEFAULTS


# ---------- unit tests (no DB) ----------

class FakeCursor:
    def __init__(self) -> None:
        self.executed: list[tuple[str, tuple]] = []
        self.rowcount = 0

    def execute(self, sql: str, params: tuple) -> None:
        self.executed.append((sql, params))
        if sql.lstrip().upper().startswith("INSERT"):
            self.rowcount = 1

    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(self, *exc) -> None:
        return None


class FakeConn:
    def __init__(self) -> None:
        self.cur = FakeCursor()

    def cursor(self) -> FakeCursor:
        return self.cur


def test_raises_on_missing_conn():
    with pytest.raises(ValueError, match="conn is required"):
        seed_tenant_params(str(uuid.uuid4()), "new")


def test_raises_on_invalid_maturity():
    with pytest.raises(ValueError, match="Invalid tenant_maturity"):
        seed_tenant_params(str(uuid.uuid4()), "veteran", conn=FakeConn())


def test_raises_on_unknown_override_key():
    with pytest.raises(UnknownParamError, match="Unknown param"):
        seed_tenant_params(
            str(uuid.uuid4()),
            "new",
            overrides_dict={"not_a_real_param": 0.5},
            conn=FakeConn(),
        )


def test_inserts_all_rows_on_first_call_unit():
    conn = FakeConn()
    n = seed_tenant_params(str(uuid.uuid4()), "new", conn=conn)
    assert n == len(TENANT_LEARNING_PARAMS_DEFAULTS)
    assert len(conn.cur.executed) == len(TENANT_LEARNING_PARAMS_DEFAULTS)
    sql, _ = conn.cur.executed[0]
    assert "ON CONFLICT (tenant_id, param_name) DO NOTHING" in sql


def test_override_applied_unit():
    conn = FakeConn()
    seed_tenant_params(
        str(uuid.uuid4()),
        "established",
        overrides_dict={"service_level_target": 0.95},
        conn=conn,
    )
    # params tuple is (tenant_id, param_name, starting_value, current_value)
    by_name = {params[1]: params[2] for _, params in conn.cur.executed}
    assert by_name["service_level_target"] == Decimal("0.95")
    # An untouched default should remain at its contract value.
    assert by_name["confidence_base_stable"] == Decimal("0.90")


def test_tenant_maturity_does_not_change_inserted_values():
    # Three exploit_threshold rows must appear with fixed values regardless
    # of the maturity argument passed in.
    for maturity in ("new", "developing", "established"):
        conn = FakeConn()
        seed_tenant_params(str(uuid.uuid4()), maturity, conn=conn)
        by_name = {params[1]: params[2] for _, params in conn.cur.executed}
        assert by_name["exploit_threshold_new"] == Decimal("8.00")
        assert by_name["exploit_threshold_developing"] == Decimal("5.00")
        assert by_name["exploit_threshold_established"] == Decimal("3.00")


# SD-01 — override with value 0.0 must not be silently dropped
def test_override_zero_is_applied_not_dropped():
    """SD-01: 0.0 is falsy in Python; dict.get() must return it, not the default."""
    conn = FakeConn()
    seed_tenant_params(
        str(uuid.uuid4()),
        "established",
        overrides_dict={"service_level_target": 0.0},
        conn=conn,
    )
    by_name = {params[1]: params[2] for _, params in conn.cur.executed}
    assert by_name["service_level_target"] == Decimal("0.0"), (
        "A zero override must not be silently replaced by the default"
    )


# SD-02 — override with value > 1.0 passes through (no range validation)
def test_override_above_one_is_stored_as_given():
    """SD-02: current implementation applies no range check on override values."""
    conn = FakeConn()
    seed_tenant_params(
        str(uuid.uuid4()),
        "established",
        overrides_dict={"service_level_target": 1.5},
        conn=conn,
    )
    by_name = {params[1]: params[2] for _, params in conn.cur.executed}
    assert by_name["service_level_target"] == Decimal("1.5")


# ---------- integration tests (hit stage9_dev via docker) ----------

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
def pg_conn():
    conn = _pg_connect_or_skip()
    # Re-create tenant_learning_params if the down migration wiped it.
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
    yield conn
    conn.rollback()
    conn.close()


def test_integration_first_call_inserts_all_rows(pg_conn):
    tenant_id = str(uuid.uuid4())
    n = seed_tenant_params(tenant_id, "new", conn=pg_conn)
    pg_conn.commit()
    assert n == 58
    with pg_conn.cursor() as cur:
        cur.execute("DELETE FROM tenant_learning_params WHERE tenant_id = %s", (tenant_id,))
    pg_conn.commit()


def test_integration_second_call_is_noop(pg_conn):
    tenant_id = str(uuid.uuid4())
    first = seed_tenant_params(tenant_id, "new", conn=pg_conn)
    pg_conn.commit()
    second = seed_tenant_params(tenant_id, "new", conn=pg_conn)
    pg_conn.commit()
    assert first == 58
    assert second == 0
    with pg_conn.cursor() as cur:
        cur.execute("DELETE FROM tenant_learning_params WHERE tenant_id = %s", (tenant_id,))
    pg_conn.commit()


def test_stage8_confidence_params_seeded(pg_conn):
    """stage8_penalty_threshold and stage8_confidence_threshold must be
    seeded — required by Confidence Engine Step 3."""
    tenant_id = str(uuid.uuid4())
    seed_tenant_params(tenant_id, "new", conn=pg_conn)
    pg_conn.commit()
    with pg_conn.cursor() as cur:
        cur.execute(
            "SELECT param_name, starting_value FROM tenant_learning_params "
            "WHERE tenant_id = %s AND param_name IN "
            "('stage8_penalty_threshold', 'stage8_confidence_threshold') "
            "ORDER BY param_name",
            (tenant_id,),
        )
        rows = {r[0]: float(r[1]) for r in cur.fetchall()}
    assert rows == {
        "stage8_confidence_threshold": 0.65,
        "stage8_penalty_threshold":    0.60,
    }
    with pg_conn.cursor() as cur:
        cur.execute("DELETE FROM tenant_learning_params WHERE tenant_id = %s", (tenant_id,))
    pg_conn.commit()


def test_integration_second_call_different_override_does_not_update(pg_conn):
    """SD-03: ON CONFLICT DO NOTHING — second call with a new override must NOT
    update the row that was inserted on the first call."""
    tenant_id = str(uuid.uuid4())
    seed_tenant_params(tenant_id, "new",
                       overrides_dict={"service_level_target": 0.80}, conn=pg_conn)
    pg_conn.commit()
    second = seed_tenant_params(tenant_id, "new",
                                overrides_dict={"service_level_target": 0.95}, conn=pg_conn)
    pg_conn.commit()
    assert second == 0, "Second call must insert 0 rows (idempotent)"
    with pg_conn.cursor() as cur:
        cur.execute(
            "SELECT current_value FROM tenant_learning_params "
            "WHERE tenant_id = %s AND param_name = 'service_level_target'",
            (tenant_id,),
        )
        row = cur.fetchone()
    assert float(row[0]) == pytest.approx(0.80), (
        "Value must remain at 0.80 — the second call's override must not overwrite it"
    )
    with pg_conn.cursor() as cur:
        cur.execute("DELETE FROM tenant_learning_params WHERE tenant_id = %s", (tenant_id,))
    pg_conn.commit()


def test_integration_override_persisted(pg_conn):
    tenant_id = str(uuid.uuid4())
    seed_tenant_params(
        tenant_id,
        "established",
        overrides_dict={"service_level_target": Decimal("0.95")},
        conn=pg_conn,
    )
    pg_conn.commit()
    with pg_conn.cursor() as cur:
        cur.execute(
            """
            SELECT starting_value, current_value
            FROM tenant_learning_params
            WHERE tenant_id = %s AND param_name = 'service_level_target'
            """,
            (tenant_id,),
        )
        row = cur.fetchone()
    assert row is not None
    assert row[0] == Decimal("0.950000")
    assert row[1] == Decimal("0.950000")
    with pg_conn.cursor() as cur:
        cur.execute("DELETE FROM tenant_learning_params WHERE tenant_id = %s", (tenant_id,))
    pg_conn.commit()
