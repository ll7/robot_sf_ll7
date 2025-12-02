"""Guidance heuristics tests (T016).

Ensures default_guidance ordering logic behaves as expected for durations
triggering soft vs hard breaches and for >40s reordering heuristic.
"""

from __future__ import annotations

from .guidance import default_guidance


def test_guidance_soft_contains_core_suggestions():
    """Test guidance soft contains core suggestions.

    Returns:
        Any: Auto-generated placeholder description.
    """
    lines = default_guidance(duration_seconds=25.0, breach_type="soft")
    assert lines, "Expected guidance suggestions for soft breach"
    # Core suggestions should appear
    joined = " ".join(lines).lower()
    for kw in ["episode", "horizon", "matrix", "bootstrap"]:
        assert kw in joined, f"Missing guidance keyword: {kw}"
    # Soft threshold reminder phrase
    assert any("soft threshold" in line.lower() for line in lines)


def test_guidance_hard_contains_hard_phrase():
    """Test guidance hard contains hard phrase.

    Returns:
        Any: Auto-generated placeholder description.
    """
    lines = default_guidance(duration_seconds=65.0, breach_type="hard")
    assert any("hard timeout" in line.lower() for line in lines), (
        "Missing hard timeout mitigation phrase"
    )


def test_guidance_large_duration_prioritizes_matrix_and_horizon():
    """Test guidance large duration prioritizes matrix and horizon.

    Returns:
        Any: Auto-generated placeholder description.
    """
    lines = default_guidance(duration_seconds=45.0, breach_type="soft")
    # Expect horizon/matrix suggestions to appear before others due to >40s heuristic
    first_two = " ".join(lines[:2]).lower()
    assert any(k in first_two for k in ["horizon", "matrix"]), (
        "Expected horizon/matrix suggestions prioritized for large overrun"
    )
