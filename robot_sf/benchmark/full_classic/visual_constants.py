"""Constants for visual artifact generation.

Defines canonical renderer identifiers and skip / note codes to ensure
stable manifest values and reduce risk of typos across tests. Kept in a
dedicated module so both internal code and tests can import the same symbols.
"""

from __future__ import annotations

# Renderer identifiers
RENDERER_SIM_VIEW = "simulation_view"
RENDERER_SYNTHETIC = "synthetic"

# Canonical note / skip codes (aligned with data-model + schemas)
NOTE_SMOKE_MODE = "smoke-mode"
NOTE_DISABLED = "disabled"
NOTE_SIM_VIEW_MISSING = "simulation-view-missing"
NOTE_MOVIEPY_MISSING = "moviepy-missing"
NOTE_INSUFFICIENT_REPLAY = "insufficient-replay-state"
NOTE_FALLBACK_FROM_SIM_VIEW = "fallback-from-sim-view"

ALL_NOTES = {
    NOTE_SMOKE_MODE,
    NOTE_DISABLED,
    NOTE_SIM_VIEW_MISSING,
    NOTE_MOVIEPY_MISSING,
    NOTE_INSUFFICIENT_REPLAY,
    NOTE_FALLBACK_FROM_SIM_VIEW,
}

__all__ = [
    "ALL_NOTES",
    "NOTE_DISABLED",
    "NOTE_FALLBACK_FROM_SIM_VIEW",
    "NOTE_INSUFFICIENT_REPLAY",
    "NOTE_MOVIEPY_MISSING",
    "NOTE_SIM_VIEW_MISSING",
    "NOTE_SMOKE_MODE",
    "RENDERER_SIM_VIEW",
    "RENDERER_SYNTHETIC",
]
