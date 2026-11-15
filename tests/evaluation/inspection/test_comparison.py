"""Comparison logic: exact matching, tolerance, and JSON-safe detail recording."""

import math

import pytest

from simready.evaluation.inspection.comparison import _json_detail, _matches, _record


class TestMatches:
    def test_exact_match(self):
        assert _matches(1.0, 1.0)
        assert _matches("abc", "abc")
        assert _matches(True, True)

    def test_float_tolerance(self):
        assert _matches(1.0, 1.0 + 1e-8)
        assert not _matches(1.0, 1.1)

    def test_nonfinite_rejected(self):
        assert not _matches(float("nan"), float("nan"))
        assert not _matches(float("inf"), float("inf"))
        assert not _matches(1.0, float("nan"))

    def test_dict_matching(self):
        assert _matches({"a": 1, "b": 2}, {"a": 1, "b": 2})
        assert not _matches({"a": 1}, {"a": 1, "b": 2})
        assert not _matches({"a": 1, "b": 2}, {"a": 1})

    def test_list_matching(self):
        assert _matches([1, 2, 3], [1, 2, 3])
        assert not _matches([1, 2], [1, 2, 3])
        assert not _matches([1, 2, 3], [1, 2, 4])

    def test_bool_strict_type(self):
        assert _matches(True, True)
        assert not _matches(True, 1)
        assert not _matches(1, True)

    def test_nested_structures(self):
        assert _matches(
            {"a": [1.0, {"b": 2.0}]},
            {"a": [1.0, {"b": 2.0}]},
        )
        assert not _matches(
            {"a": [1.0, {"b": 2.0}]},
            {"a": [1.0, {"b": 3.0}]},
        )


class TestJsonDetail:
    def test_nonfinite_becomes_string(self):
        result = _json_detail({"x": float("nan")})
        assert result["x"] == "NONFINITE:nan"

    def test_inf_becomes_string(self):
        result = _json_detail({"x": float("inf")})
        assert result["x"] == "NONFINITE:inf"

    def test_nested_nonfinite(self):
        result = _json_detail({"a": [{"b": float("-inf")}]})
        assert result["a"][0]["b"] == "NONFINITE:-inf"

    def test_normal_values_pass_through(self):
        assert _json_detail({"a": 1, "b": [1, 2]}) == {"a": 1, "b": [1, 2]}


class TestRecord:
    def test_record_pass(self):
        checks = []
        _record(checks, "test", True, detail="ok")
        assert checks[0]["status"] == "PASS"
        assert checks[0]["passed"] is True

    def test_record_fail(self):
        checks = []
        _record(checks, "test", False, detail="mismatch")
        assert checks[0]["status"] == "FAIL"
        assert checks[0]["passed"] is False
