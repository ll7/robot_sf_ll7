"""Helpers for splitting scenarios into train/holdout groups."""

from __future__ import annotations

from typing import Any


def split_scenarios(
    scenarios: list[dict[str, Any]] | list[object],
    *,
    default_split: str = "train",
) -> dict[str, list[dict[str, Any]]]:
    """Split scenarios by their ``split`` field.

    Args:
        scenarios: Scenario entries (list of dict-like objects).
        default_split: Split name to use when not specified ("train" or "holdout").

    Returns:
        dict[str, list[dict[str, Any]]]: Mapping from split name to scenario lists.

    Raises:
        ValueError: If an invalid split label is encountered.
    """
    allowed = {"train", "holdout"}
    normalized_default = default_split.strip().lower()
    if normalized_default not in allowed:
        raise ValueError(f"default_split must be one of {sorted(allowed)}")

    splits: dict[str, list[dict[str, Any]]] = {"train": [], "holdout": []}
    for entry in scenarios:
        if not isinstance(entry, dict):
            raise ValueError(f"Scenario entries must be dict-like, got {type(entry)}")
        raw_split = entry.get("split", None)
        split_value = normalized_default if raw_split is None else raw_split
        if not isinstance(split_value, str):
            raise ValueError(f"Scenario split must be a string, got {split_value!r}")
        split_name = split_value.strip().lower()
        if split_name not in allowed:
            raise ValueError(f"Unknown scenario split '{split_name}'. Expected {sorted(allowed)}")
        splits[split_name].append(entry)
    return splits


__all__ = ["split_scenarios"]
