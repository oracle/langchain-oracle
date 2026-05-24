"""Test Oracle JSON type normalization."""

from decimal import Decimal, InvalidOperation
from typing import Any


def _normalize_oracle_json_types(obj: Any) -> Any:
    """Convert Oracle-specific types (like Decimal) to standard Python types."""
    if isinstance(obj, dict):
        return {k: _normalize_oracle_json_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_normalize_oracle_json_types(item) for item in obj]
    elif isinstance(obj, Decimal):
        # Convert Decimal to int if it's a whole number, otherwise to float
        try:
            if obj % 1 == 0:
                return int(obj)
            else:
                return float(obj)
        except (InvalidOperation, OverflowError):
            # For very large numbers or special values, convert to float
            return float(obj)
    else:
        return obj


def test_normalize_oracle_json_types():
    """Test that Oracle Decimal types are properly normalized to Python types."""

    # Test basic type conversions
    assert _normalize_oracle_json_types({"step": Decimal("1")}) == {"step": 1}
    assert _normalize_oracle_json_types({"score": Decimal("3.14")}) == {"score": 3.14}
    assert _normalize_oracle_json_types({"active": True}) == {"active": True}
    assert _normalize_oracle_json_types({"name": "test"}) == {"name": "test"}
    assert _normalize_oracle_json_types({"value": None}) == {"value": None}

    # Test nested structures
    assert _normalize_oracle_json_types(
        {"nested": {"count": Decimal("42"), "rate": Decimal("2.5")}}
    ) == {"nested": {"count": 42, "rate": 2.5}}

    # Test lists with mixed types
    assert _normalize_oracle_json_types(
        [Decimal("1"), "text", Decimal("3.14"), True]
    ) == [1, "text", 3.14, True]

    # Test complex nested structure
    assert _normalize_oracle_json_types(
        {
            "metadata": {
                "step": Decimal("1"),
                "scores": [Decimal("1.0"), Decimal("2.5"), Decimal("3")],
                "config": {"max_iter": Decimal("100"), "enabled": True},
            }
        }
    ) == {
        "metadata": {
            "step": 1,
            "scores": [1.0, 2.5, 3],
            "config": {"max_iter": 100, "enabled": True},
        }
    }

    # Test edge cases
    assert _normalize_oracle_json_types({}) == {}
    assert _normalize_oracle_json_types([]) == []
    assert _normalize_oracle_json_types({"zero_int": Decimal("0")}) == {"zero_int": 0}
    assert _normalize_oracle_json_types({"neg_int": Decimal("-42")}) == {"neg_int": -42}
    assert _normalize_oracle_json_types({"neg_float": Decimal("-3.14")}) == {
        "neg_float": -3.14
    }

    # Test very small decimals
    result = _normalize_oracle_json_types({"tiny": Decimal("0.000000000001")})
    assert (
        abs(result["tiny"] - 0.000000000001) < 1e-15
    )  # Use approximate comparison for floats

    # Test that very large numbers convert to float
    result = _normalize_oracle_json_types(
        {"big": Decimal("999999999999999999999999999999")}
    )
    assert isinstance(result["big"], float)
    assert result["big"] == 1e30  # Should be in scientific notation


def test_normalize_preserves_non_decimal_types():
    """Test that non-Decimal types are preserved as-is."""

    # Regular Python types should pass through unchanged
    data = {
        "int": 42,
        "float": 3.14,
        "str": "hello",
        "bool": True,
        "none": None,
        "list": [1, 2, 3],
        "dict": {"a": 1, "b": 2},
    }

    assert _normalize_oracle_json_types(data) == data
