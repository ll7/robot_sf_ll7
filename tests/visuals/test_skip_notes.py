"""Test skip note constants (T014)."""

from __future__ import annotations

from robot_sf.benchmark.full_classic import visual_constants as vc  # type: ignore


def test_all_notes_membership():
    """TODO docstring. Document this function."""
    expected = {
        vc.NOTE_SMOKE_MODE,
        vc.NOTE_DISABLED,
        vc.NOTE_SIM_VIEW_MISSING,
        vc.NOTE_MOVIEPY_MISSING,
        vc.NOTE_INSUFFICIENT_REPLAY,
        vc.NOTE_FALLBACK_FROM_SIM_VIEW,
    }
    assert expected == vc.ALL_NOTES


def test_renderer_constants():
    """TODO docstring. Document this function."""
    assert vc.RENDERER_SIM_VIEW == "simulation_view"
    assert vc.RENDERER_SYNTHETIC == "synthetic"
