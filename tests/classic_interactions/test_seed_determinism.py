"""Deterministic seed ordering test (T006 / FR-002, FR-003).

TDD: Expects run_demo() to produce non-empty episode list with stable seed ordering.
Will fail until implementation returns episode summaries (T016,T017,T019).
"""

from __future__ import annotations

import importlib


def _load():
    """Load.

    Returns:
        Any: Auto-generated placeholder description.
    """
    return importlib.import_module("examples.classic_interactions_pygame")


def test_seed_order_reproducible():
    """Test seed order reproducible.

    Returns:
        Any: Auto-generated placeholder description.
    """
    mod = _load()
    # Force non-dry execution
    if hasattr(mod, "DRY_RUN"):
        original = mod.DRY_RUN
        mod.DRY_RUN = False  # type: ignore
    else:  # pragma: no cover
        original = None

    try:
        eps1 = mod.run_demo()
        eps2 = mod.run_demo()
    finally:
        if hasattr(mod, "DRY_RUN"):
            mod.DRY_RUN = original  # type: ignore

    assert eps1, "Expected non-empty episodes list (TDD failing until implementation)."
    assert eps2, "Expected non-empty episodes list (TDD failing until implementation)."
    seeds1 = [e.get("seed") for e in eps1]
    seeds2 = [e.get("seed") for e in eps2]
    assert seeds1 == seeds2, f"Seed ordering mismatch: {seeds1} vs {seeds2}"
