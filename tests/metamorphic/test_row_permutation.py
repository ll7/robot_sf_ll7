"""Pedestrian-row permutation metamorphism."""

from __future__ import annotations

from tests.metamorphic.support import BASE_MAP, assert_trace_equal, permuted_map, run_episode


def test_row_permutation_preserves_actor_associated_state() -> None:
    """Reordering declarations must not change state after rows are matched by identity."""
    base = run_episode(BASE_MAP)
    permuted = run_episode(permuted_map())
    permuted_indices = {key: index for index, key in enumerate(permuted.row_keys)}
    row_order = tuple(permuted_indices[key] for key in base.row_keys)

    assert_trace_equal(base, permuted, row_order=row_order)
