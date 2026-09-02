"""Translation metamorphism for the crowd-only environment."""

from __future__ import annotations

import numpy as np

from tests.metamorphic.support import (
    BASE_MAP,
    assert_trace_equal,
    run_episode,
    translated_map,
)


def test_scene_translation_preserves_relative_dynamics() -> None:
    """Translating every scene coordinate translates only position-like outputs."""
    offset = np.asarray((1.0, 1.0), dtype=np.float32)
    base = run_episode(BASE_MAP)
    translated = run_episode(translated_map(offset))

    assert_trace_equal(
        base,
        translated,
        transforms={
            "positions": lambda values: values - offset,
            "goals": lambda values: values - offset,
        },
    )
