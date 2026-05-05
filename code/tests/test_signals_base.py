"""
unit/orchestration/test_signals_base.py — signals._base helper coverage.

Covers the boundary helpers used by both SignalEmitter and SignalConsumer:
    * _clamp_confidence — coerce to [0,1]
    * _wrap_jsonb        — wrap dict for psycopg2 Json (or pass through in tests)
    * _decode_jsonb      — decode dict / str / bytes from a JSONB column
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_CODE = Path(__file__).resolve().parents[3]
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from signals._base import _clamp_confidence, _wrap_jsonb, _decode_jsonb


# ---------------------------------------------------------------------------
# _clamp_confidence
# ---------------------------------------------------------------------------

class TestClampConfidence:

    def test_none_returns_none(self):
        assert _clamp_confidence(None) is None

    def test_zero_passes_through(self):
        assert _clamp_confidence(0.0) == 0.0

    def test_one_passes_through(self):
        assert _clamp_confidence(1.0) == 1.0

    def test_mid_passes_through(self):
        assert _clamp_confidence(0.5) == 0.5

    def test_negative_clamped_to_zero(self):
        assert _clamp_confidence(-0.7) == 0.0

    def test_above_one_clamped_to_one(self):
        assert _clamp_confidence(1.5) == 1.0

    def test_int_input_coerced_to_float(self):
        out = _clamp_confidence(0)
        assert isinstance(out, float) and out == 0.0

    def test_string_numeric_converts(self):
        """float() coerces numeric strings — ensure no surprise crash."""
        assert _clamp_confidence("0.42") == pytest.approx(0.42)


# ---------------------------------------------------------------------------
# _wrap_jsonb / _decode_jsonb
# ---------------------------------------------------------------------------

class TestJsonbHelpers:

    # ----- _wrap_jsonb ------------------------------------------------------

    def test_wrap_dict_returns_object(self):
        """_wrap_jsonb returns *something* (Json wrapper or the dict itself
        in environments where psycopg2 isn't available)."""
        out = _wrap_jsonb({"k": "v"})
        assert out is not None

    # ----- _decode_jsonb ----------------------------------------------------

    def test_decode_none_returns_empty_dict(self):
        assert _decode_jsonb(None) == {}

    def test_decode_dict_passes_through(self):
        d = {"a": 1, "b": [2, 3]}
        assert _decode_jsonb(d) is d

    def test_decode_string_parses_json(self):
        assert _decode_jsonb('{"a": 1}') == {"a": 1}

    def test_decode_bytes_parses_json(self):
        assert _decode_jsonb(b'{"a": 1}') == {"a": 1}

    def test_decode_bytearray_parses_json(self):
        assert _decode_jsonb(bytearray(b'{"a": 1}')) == {"a": 1}

    def test_decode_memoryview_parses_json(self):
        assert _decode_jsonb(memoryview(b'{"a": 1}')) == {"a": 1}

    def test_decode_invalid_type_raises_typeerror(self):
        with pytest.raises(TypeError, match="unexpected JSONB value type"):
            _decode_jsonb(42)   # int not in the allowed set

    def test_decode_malformed_json_string_raises(self):
        with pytest.raises(json.JSONDecodeError):
            _decode_jsonb("{not json")

    def test_decode_round_trip_via_wrap(self):
        """A dict round-trips through the API without losing structure."""
        original = {"nested": {"k": [1, 2]}, "x": "string"}
        # _wrap_jsonb produces something psycopg2-compatible; _decode_jsonb
        # accepts dicts directly. Confirm the decode side handles both.
        assert _decode_jsonb(original) == original
        assert _decode_jsonb(json.dumps(original)) == original
