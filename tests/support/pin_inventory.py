"""Fail collection when a required repository pin inventory is empty."""

from collections.abc import Iterable
from typing import TypeVar

T = TypeVar("T")


# Keep this helper importable by the declared Python 3.11 compatibility lane.
def required_pin_inventory(  # noqa: UP047
    values: Iterable[T], *, name: str
) -> tuple[T, ...]:
    """Materialize required witnesses without accepting vacuous parametrization."""
    inventory = tuple(values)
    if not inventory:
        raise ValueError(f"Required pin inventory is empty: {name}")
    return inventory
