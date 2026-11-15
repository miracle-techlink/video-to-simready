"""Direct coverage for primitive validation shared by domain contracts."""

import pytest

from simready.core.validation import finite, indexed, number, require


def test_require_raises_with_message():
    require(True, "unused")
    with pytest.raises(ValueError, match="broken"):
        require(False, "broken")


@pytest.mark.parametrize("value", [0, -1, 2.5])
def test_finite_accepts_plain_numbers(value):
    assert finite(value)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), "1", None, [1], True])
def test_finite_rejects_nonfinite_bool_and_nonnumeric(value):
    assert not finite(value)


def test_indexed_builds_lookup_and_rejects_duplicates():
    rows = [{"id": "a", "v": 1}, {"id": "b", "v": 2}]
    assert indexed(rows, "id", "rows") == {"a": rows[0], "b": rows[1]}
    with pytest.raises(ValueError, match="Duplicate id"):
        indexed(rows + [{"id": "a"}], "id", "rows")


def test_indexed_rejects_non_list_and_missing_key():
    with pytest.raises(ValueError, match="must be a list"):
        indexed({"id": "a"}, "id", "rows")
    with pytest.raises(ValueError, match="nonempty id"):
        indexed([{"id": ""}], "id", "rows")
    with pytest.raises(ValueError, match="nonempty id"):
        indexed([{"name": "a"}], "id", "rows")


@pytest.mark.parametrize("value", [0, 0.5, 10])
def test_number_accepts_in_range(value):
    assert number(value, "score", minimum=0) == value


@pytest.mark.parametrize(
    "value,minimum",
    [(True, 0), ("1", 0), (None, 0), (float("nan"), 0), (float("inf"), 0), (-1, 0)],
)
def test_number_rejects_bool_text_nonfinite_and_below_minimum(value, minimum):
    with pytest.raises(ValueError, match="Invalid score"):
        number(value, "score", minimum=minimum)


def test_number_rejects_nested_structure():
    with pytest.raises(ValueError, match="Invalid"):
        number({"value": 1}, "score")
