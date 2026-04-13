"""
utils.py — Shared utilities used across the pipeline.
"""


def fmt_dollar(value: float | None, suffix: str = "") -> str:
    """Format a dollar value as $1.23M, $450.0K, or $1,234."""
    if value is None:
        return "N/A"
    if value >= 1_000_000:
        return f"${value / 1_000_000:.2f}M{suffix}"
    if value >= 1_000:
        return f"${value / 1_000:.1f}K{suffix}"
    return f"${value:,.0f}{suffix}"
