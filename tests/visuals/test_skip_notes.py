"""Test skip note constants (T014)."""

from __future__ import annotations

from robot_sf.benchmark.full_classic import visual_constants as vc  # type: ignore


def test_all_notes_membership():
    expected = {
        vc.NOTE_SMOKE_MODE,
        vc.NOTE_DISABLED,
        vc.NOTE_SIM_VIEW_MISSING,
        vc.NOTE_MOVIEPY_MISSING,
        vc.NOTE_INSUFFICIENT_REPLAY,
    }
    assert vc.ALL_NOTES == expected


def test_renderer_constants():
    assert vc.RENDERER_SIM_VIEW == "simulation_view"
    assert vc.RENDERER_SYNTHETIC == "synthetic"
