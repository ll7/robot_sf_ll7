"""Fail collection when a required repository pin inventory is empty."""

from collections.abc import Iterable


def required_pin_inventory[T](values: Iterable[T], *, name: str) -> tuple[T, ...]:
    """Materialize required witnesses without accepting vacuous parametrization."""
    inventory = tuple(values)
    if not inventory:
        raise ValueError(f"Required pin inventory is empty: {name}")
    return inventory
