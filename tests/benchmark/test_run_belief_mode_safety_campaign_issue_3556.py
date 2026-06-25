"""Tests for issue #3556 belief-mode campaign decision screening."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "benchmark"
    / "run_belief_mode_safety_campaign_issue_3556.py"
)
_SPEC = importlib.util.spec_from_file_location("_issue_3556_campaign", _MODULE_PATH)
assert _SPEC is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)
classify_screened_decision = _MODULE.classify_screened_decision


def _mode(collision_rate: float, near_misses: int) -> dict[str, float | int]:
    """Build the minimal aggregate shape consumed by the decision classifier."""
    return {
        "episodes": 3,
        "collision_rate": collision_rate,
        "total_near_misses": near_misses,
    }


def test_oracle_unsafe_blocks_interpretation_even_without_mode_delta() -> None:
    """A collide-heavy oracle cannot support a dropped-vs-retained safety claim."""
    decision = classify_screened_decision(
        {
            "oracle": _mode(collision_rate=1.0, near_misses=3),
            "uncertain_retained": _mode(collision_rate=1.0, near_misses=3),
            "uncertain_dropped": _mode(collision_rate=1.0, near_misses=3),
        }
    )

    assert decision["decision"] == "inconclusive_oracle_unsafe"
    assert decision["screening_status"] == "oracle_unsafe"
    assert decision["oracle_near_safe"] is False
    assert decision["mode_is_discriminating"] is False


def test_near_safe_nondiscriminating_matrix_stays_inconclusive() -> None:
    """Near-safe oracle alone is not enough when dropped and retained match."""
    decision = classify_screened_decision(
        {
            "oracle": _mode(collision_rate=0.0, near_misses=0),
            "uncertain_retained": _mode(collision_rate=0.0, near_misses=0),
            "uncertain_dropped": _mode(collision_rate=0.0, near_misses=0),
        }
    )

    assert decision["decision"] == "inconclusive"
    assert decision["screening_status"] == "near_safe_nondiscriminating"
    assert decision["oracle_near_safe"] is True
    assert decision["mode_is_discriminating"] is False


def test_near_safe_discriminating_matrix_recommends_revise() -> None:
    """Dropped mode becoming less safe under a near-safe oracle is actionable."""
    decision = classify_screened_decision(
        {
            "oracle": _mode(collision_rate=0.0, near_misses=0),
            "uncertain_retained": _mode(collision_rate=0.0, near_misses=1),
            "uncertain_dropped": _mode(collision_rate=0.0, near_misses=4),
        }
    )

    assert decision["decision"] == "revise"
    assert decision["screening_status"] == "near_safe_discriminating"
    assert decision["oracle_near_safe"] is True
    assert decision["mode_is_discriminating"] is True
