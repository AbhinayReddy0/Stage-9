"""
unit/orchestration/test_db_utils.py — db_utils helper coverage.

The module exports two things:
    * DBConnection — a runtime-checkable Protocol
    * warn_if_shared_conn — defensive check that warns when callers pass an
      already-in-flight conn into a module that will commit on it
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

_CODE = Path(__file__).resolve().parents[3]
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from infrastructure.db_utils import DBConnection, warn_if_shared_conn


# ---------------------------------------------------------------------------
# DBConnection Protocol
# ---------------------------------------------------------------------------

class TestDBConnectionProtocol:

    def test_object_with_all_three_methods_is_recognised(self):
        class _C:
            def cursor(self): return None
            def commit(self): return None
            def rollback(self): return None
        assert isinstance(_C(), DBConnection)

    def test_object_missing_a_method_is_rejected(self):
        class _NoRollback:
            def cursor(self): return None
            def commit(self): return None
        assert not isinstance(_NoRollback(), DBConnection)

    def test_class_satisfying_protocol_via_duck_typing(self):
        """Plain Python class with the three methods satisfies the runtime
        check without inheriting from anything — that's the point of a
        runtime_checkable Protocol."""
        class _Conn:
            def cursor(self): return object()
            def commit(self): return None
            def rollback(self): return None
        assert isinstance(_Conn(), DBConnection)


# ---------------------------------------------------------------------------
# warn_if_shared_conn
# ---------------------------------------------------------------------------

class TestWarnIfSharedConn:

    def test_idle_status_does_not_warn(self):
        conn = SimpleNamespace(info=SimpleNamespace(transaction_status=0))
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning becomes a raise
            warn_if_shared_conn(conn, label="X")  # passes silently

    def test_in_transaction_status_warns(self):
        """status=2 (INTRANS) is the typical "caller has an open tx"."""
        conn = SimpleNamespace(info=SimpleNamespace(transaction_status=2))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warn_if_shared_conn(conn, label="MyModule")
        assert len(caught) == 1
        assert "MyModule" in str(caught[0].message)
        assert "transaction_status=2" in str(caught[0].message)

    def test_active_status_warns(self):
        """status=1 (ACTIVE — command in flight) also warrants the warning."""
        conn = SimpleNamespace(info=SimpleNamespace(transaction_status=1))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warn_if_shared_conn(conn, label="X")
        assert len(caught) == 1

    def test_inerror_status_warns(self):
        conn = SimpleNamespace(info=SimpleNamespace(transaction_status=3))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warn_if_shared_conn(conn, label="X")
        assert len(caught) == 1
        assert "transaction_status=3" in str(caught[0].message)

    def test_no_info_attr_silent(self):
        """Test fakes / non-psycopg2 drivers may not have .info — never raise."""
        conn = SimpleNamespace()  # no .info
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            warn_if_shared_conn(conn, label="X")  # passes silently

    def test_info_without_transaction_status_silent(self):
        conn = SimpleNamespace(info=SimpleNamespace())  # info but no .transaction_status
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            warn_if_shared_conn(conn, label="X")  # passes silently

    def test_warning_message_contains_actionable_hint(self):
        conn = SimpleNamespace(info=SimpleNamespace(transaction_status=2))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warn_if_shared_conn(conn, label="MyModule")
        msg = str(caught[0].message)
        assert "Pass a dedicated psycopg2 connection" in msg
