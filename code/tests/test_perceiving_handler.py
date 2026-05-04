"""
unit/handlers/test_perceiving_handler.py — perceiving handler coverage.

Production handler is small (1 function, 55 lines):
    1. Loads tenant params via TenantParams.load
    2. Raises TenantParamNotFoundError if no rows
    3. Stores params on RunContext
    4. Peeks at PATTERN_CONFIDENCE signals (read-only)
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Ensure code/ is on path
_CODE = Path(__file__).resolve().parents[3]
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))
if str(_CODE / "handlers") not in sys.path:
    sys.path.insert(0, str(_CODE / "handlers"))

from handlers._context import RunContext, store, remove
from handlers.perceiving import perceiving_handler
from infrastructure.errors import TenantParamNotFoundError
from infrastructure.tenant_params_defaults import VALID_PARAM_NAMES


def _full_params_mock():
    """Mock that satisfies the completeness check: iterating yields every
    param name from VALID_PARAM_NAMES, and len() returns the count."""
    m = MagicMock()
    m.__iter__ = lambda self: iter(VALID_PARAM_NAMES)
    m.__len__ = lambda self: len(VALID_PARAM_NAMES)
    return m


def _ctx(run_id="run-perc"):
    """Create + register a minimal RunContext, return it."""
    ctx = RunContext(tenant_id="t1", run_id=run_id)
    ctx.signal_consumer = MagicMock()
    ctx.signal_consumer.peek.return_value = []
    store(ctx)
    return ctx


def _teardown(ctx):
    remove(ctx.run_id)


class TestPerceivingHandler:

    def test_loads_tenant_params_from_db(self):
        ctx = _ctx("run-1")
        try:
            fake_params = _full_params_mock()
            with patch("handlers.perceiving.TenantParams.load",
                       return_value=fake_params):
                perceiving_handler(tenant_id="t1", run_id="run-1", db=MagicMock())
            assert ctx.params is fake_params
        finally:
            _teardown(ctx)

    def test_raises_when_no_params_seeded(self):
        ctx = _ctx("run-2")
        try:
            empty = MagicMock()
            empty.__len__ = lambda self: 0
            empty.__iter__ = lambda self: iter([])
            with patch("handlers.perceiving.TenantParams.load",
                       return_value=empty):
                with pytest.raises(TenantParamNotFoundError,
                                   match="tenant_learning_params has no rows"):
                    perceiving_handler(tenant_id="t1", run_id="run-2", db=MagicMock())
        finally:
            _teardown(ctx)

    def test_raises_when_some_params_missing(self):
        """New deploy-vs-migration check: incomplete param set must raise
        with a list of the missing param names."""
        ctx = _ctx("run-2b")
        try:
            partial = MagicMock()
            # Missing one param — must fire the completeness guard.
            subset = list(VALID_PARAM_NAMES)[:-1]
            partial.__len__ = lambda self: len(subset)
            partial.__iter__ = lambda self: iter(subset)
            with patch("handlers.perceiving.TenantParams.load",
                       return_value=partial):
                with pytest.raises(TenantParamNotFoundError,
                                   match="introduced in a recent migration"):
                    perceiving_handler(tenant_id="t1", run_id="run-2b",
                                       db=MagicMock())
        finally:
            _teardown(ctx)

    def test_peeks_pattern_confidence_signals(self):
        ctx = _ctx("run-3")
        try:
            fake_params = _full_params_mock()
            with patch("handlers.perceiving.TenantParams.load",
                       return_value=fake_params):
                perceiving_handler(tenant_id="t1", run_id="run-3", db=MagicMock())
            ctx.signal_consumer.peek.assert_called_once()
            args, _ = ctx.signal_consumer.peek.call_args
            assert args[0] == "t1"
        finally:
            _teardown(ctx)

    def test_signal_peek_does_not_consume(self):
        """PEEK contract: signal_consumer.peek is called, not .consume."""
        ctx = _ctx("run-4")
        try:
            fake_params = _full_params_mock()
            with patch("handlers.perceiving.TenantParams.load",
                       return_value=fake_params):
                perceiving_handler(tenant_id="t1", run_id="run-4", db=MagicMock())
            # No call to .consume / .mark_processed
            assert not ctx.signal_consumer.consume.called \
                if hasattr(ctx.signal_consumer, "consume") else True
        finally:
            _teardown(ctx)

    def test_handles_no_pattern_confidence_signals(self):
        """Empty signals list — handler must complete without raising."""
        ctx = _ctx("run-5")
        try:
            ctx.signal_consumer.peek.return_value = []
            fake_params = _full_params_mock()
            with patch("handlers.perceiving.TenantParams.load",
                       return_value=fake_params):
                perceiving_handler(tenant_id="t1", run_id="run-5", db=MagicMock())
            assert ctx.params is not None
        finally:
            _teardown(ctx)

    def test_missing_runcontext_raises_keyerror(self):
        """If preloading_handler didn't run, the registry is empty —
        perceiving must surface that as KeyError (not silently skip)."""
        with pytest.raises(KeyError, match="No RunContext"):
            perceiving_handler(tenant_id="t1", run_id="never-stored", db=MagicMock())
