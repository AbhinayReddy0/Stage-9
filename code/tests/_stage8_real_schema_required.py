"""
Shared guard for the four stage8 audit files: test_stage8_contract.py,
test_read_audit.py, test_rw_audit.py, test_learning_audit.py.

These files are designed to validate a REAL Stage 8 deployment where
stage8.* are BASE TABLEs with uuid sku_id columns. The local development
DB used for the rest of this suite has stage8.* set up as VIEWs over the
varchar-keyed public.* tables (see code/tests/_setup_stage8.sql for the
view definitions).

When views are detected, these audit suites skip at module load with a
clear message, rather than producing dozens of red asserts on the same
underlying schema mismatch.
"""
from __future__ import annotations

import os
import pytest


def skip_if_stage8_uses_views() -> None:
    """Module-level skip — call from `pytest.skip(..., allow_module_level=True)`
    decorator at the top of an audit module."""
    dsn = os.environ.get("STAGE9_TEST_DSN")
    if not dsn:
        pytest.skip("STAGE9_TEST_DSN not set", allow_module_level=True)

    try:
        import psycopg2
    except ImportError:
        pytest.skip("psycopg2 not installed", allow_module_level=True)

    try:
        conn = psycopg2.connect(dsn)
    except Exception as exc:
        pytest.skip(f"DB unavailable: {exc}", allow_module_level=True)

    cur = conn.cursor()
    cur.execute("""
        SELECT
            COUNT(*) FILTER (WHERE table_type = 'BASE TABLE')  AS n_tables,
            COUNT(*) FILTER (WHERE table_type = 'VIEW')        AS n_views
        FROM information_schema.tables
        WHERE table_schema = 'stage8'
    """)
    n_tables, n_views = cur.fetchone()
    cur.close()
    conn.close()

    if n_views > 0 and n_tables == 0:
        pytest.skip(
            f"Stage 8 schema is view-aliased ({n_views} views, 0 base tables) "
            "in this environment — these audit tests target a real Stage 8 "
            "deployment with BASE TABLEs and uuid sku_id columns. "
            "Run them against a production-shaped Stage 8 DB.",
            allow_module_level=True,
        )
