"""
tests/test_batch_writer.py — Atheera Stage 9
=============================================
Unit tests for BatchWriter allowlist enforcement.

Covers:
    - Sacred-write tables are rejected with a descriptive message
    - Unknown tables are rejected with a descriptive message
    - Sacred and unknown tables produce distinct error messages
    - All batchable tables are accepted without error
    - Queue accumulates count correctly across tables
"""

from __future__ import annotations

import pytest

from infrastructure.batch_writer import BatchWriter, _BATCHABLE_TABLES, _SACRED_WRITE_TABLES
from infrastructure.constants import Table


# ---------------------------------------------------------------------------
# Minimal fake connection — queue() never touches the DB, so conn is unused
# during these tests.  We pass a sentinel object to satisfy the constructor.
# ---------------------------------------------------------------------------

class _NullConn:
    """Stand-in connection that should never be called during allowlist tests."""

    def cursor(self):  # pragma: no cover
        raise AssertionError("BatchWriter.queue() must not access the connection")


def _make_writer() -> BatchWriter:
    return BatchWriter(_NullConn())


# ---------------------------------------------------------------------------
# Sacred-write table rejection
# ---------------------------------------------------------------------------

def test_queue_rejects_pattern_feedback_with_sacred_message():
    bw = _make_writer()
    with pytest.raises(ValueError, match="sacred-write"):
        bw.queue(Table.PATTERN_FEEDBACK, {"col": 1})


def test_queue_rejects_cross_agent_signals_with_sacred_message():
    bw = _make_writer()
    with pytest.raises(ValueError, match="sacred-write"):
        bw.queue(Table.CROSS_AGENT_SIGNALS, {"col": 1})


# ---------------------------------------------------------------------------
# Unknown table rejection
# ---------------------------------------------------------------------------

def test_queue_rejects_unknown_table_with_allowlist_message():
    bw = _make_writer()
    with pytest.raises(ValueError, match="allowlist"):
        bw.queue("completely_unknown_table_xyz", {"col": 1})


# ---------------------------------------------------------------------------
# Error message distinctiveness — sacred vs unknown
# ---------------------------------------------------------------------------

def test_sacred_and_unknown_produce_distinct_messages():
    bw = _make_writer()

    sacred_msg = ""
    try:
        bw.queue(Table.PATTERN_FEEDBACK, {"col": 1})
    except ValueError as e:
        sacred_msg = str(e)

    unknown_msg = ""
    try:
        bw.queue("unknown_table_xyz", {"col": 1})
    except ValueError as e:
        unknown_msg = str(e)

    assert sacred_msg != unknown_msg, "Sacred and unknown tables must produce distinct error messages"
    assert "sacred" in sacred_msg.lower()
    assert "allowlist" in unknown_msg.lower()


# ---------------------------------------------------------------------------
# Batchable tables accepted
# ---------------------------------------------------------------------------

def test_queue_accepts_all_batchable_tables():
    bw = _make_writer()
    for table in _BATCHABLE_TABLES:
        bw.queue(table, {"col": 1})  # must not raise


# ---------------------------------------------------------------------------
# Count accumulation
# ---------------------------------------------------------------------------

def test_queue_accumulates_count():
    bw = _make_writer()
    tables = list(_BATCHABLE_TABLES)[:3]  # use first 3 batchable tables
    for i, table in enumerate(tables):
        for _ in range(i + 1):
            bw.queue(table, {"col": 1})
    # 1 + 2 + 3 = 6 total rows
    assert bw.count == 6


def test_queue_count_increments_per_row():
    bw = _make_writer()
    table = next(iter(_BATCHABLE_TABLES))
    for n in range(1, 6):
        bw.queue(table, {"col": n})
        assert bw.count == n


# ---------------------------------------------------------------------------
# Sacred set completeness — both known sacred tables are in the frozenset
# ---------------------------------------------------------------------------

def test_sacred_write_tables_contains_pattern_feedback():
    assert Table.PATTERN_FEEDBACK in _SACRED_WRITE_TABLES


def test_sacred_write_tables_contains_cross_agent_signals():
    assert Table.CROSS_AGENT_SIGNALS in _SACRED_WRITE_TABLES


def test_sacred_and_batchable_are_disjoint():
    assert _SACRED_WRITE_TABLES.isdisjoint(_BATCHABLE_TABLES), \
        "A table cannot be both sacred-write and batchable"
