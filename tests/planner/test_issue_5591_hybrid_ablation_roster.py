"""Tests for issue #5591 hybrid portfolio structure-ablation roster.

These are CPU-only configuration-contract tests. They prove the frozen ablation
roster diffs exactly one structural knob from the reference (hybrid_full) arm,
and that the new ``adaptive_switching_enabled`` toggle pins the active head.
They do NOT run the benchmark campaign (that is the SLURM/cheap-lane-outside step).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from robot_sf.planner.hybrid_portfolio import (
    HybridPortfolioAdapter,
    HybridPortfolioConfig,
    build_hybrid_portfolio_build_config,
)
from scripts.validation.build_issue_5591_hybrid_ablation_delta import _arm_summary, _delta

REPO_ROOT = Path(__file__).resolve().parents[2]
ALGO_DIR = REPO_ROOT / "configs" / "algos"

REFERENCE = "hybrid_portfolio_ablation_hybrid_full.yaml"
ABLATION_ARMS = {
    "hybrid_minus_static_escape": "hybrid_portfolio_ablation_hybrid_minus_static_escape.yaml",
    "hybrid_minus_hard_guard": "hybrid_portfolio_ablation_hybrid_minus_hard_guard.yaml",
    "hybrid_minus_adaptive_switching": "hybrid_portfolio_ablation_hybrid_minus_adaptive_switching.yaml",
    "hybrid_minus_all_three": "hybrid_portfolio_ablation_hybrid_minus_all_three.yaml",
}


def _load(name: str) -> dict:
    return yaml.safe_load((ALGO_DIR / name).read_text(encoding="utf-8"))


def _structural_keys(cfg: dict) -> dict:
    """Flatten the structural knobs that the ablation roster toggles."""
    hybrid = cfg.get("hybrid", {})
    risk = cfg.get("risk_dwa", {})
    mppi = cfg.get("mppi_social", {})
    return {
        "hybrid.fallback_on_exception": hybrid.get("fallback_on_exception"),
        "hybrid.adaptive_switching_enabled": hybrid.get("adaptive_switching_enabled"),
        "risk_dwa.progress_escape_enabled": risk.get("progress_escape_enabled"),
        "mppi_social.progress_escape_enabled": mppi.get("progress_escape_enabled"),
    }


def test_reference_arm_enables_every_structural_component() -> None:
    """The reference arm must ship with all three structural mechanisms on."""
    keys = _structural_keys(_load(REFERENCE))
    assert keys == {
        "hybrid.fallback_on_exception": True,
        "hybrid.adaptive_switching_enabled": True,
        "risk_dwa.progress_escape_enabled": True,
        "mppi_social.progress_escape_enabled": True,
    }


def test_single_knob_ablation_arms_differ_by_exactly_one_knob() -> None:
    """Each single-knob arm must differ from reference by exactly one structural knob."""
    ref = _structural_keys(_load(REFERENCE))
    expected_diffs = {
        "hybrid_minus_static_escape": {
            "risk_dwa.progress_escape_enabled",
            "mppi_social.progress_escape_enabled",
        },
        "hybrid_minus_hard_guard": {"hybrid.fallback_on_exception"},
        "hybrid_minus_adaptive_switching": {"hybrid.adaptive_switching_enabled"},
    }
    for arm, fname in ABLATION_ARMS.items():
        if arm == "hybrid_minus_all_three":
            continue
        arm_keys = _structural_keys(_load(fname))
        diffs = {k for k in ref if ref[k] != arm_keys[k]}
        assert diffs == expected_diffs[arm], f"{arm}: unexpected diff {diffs}"


def test_all_three_arm_disables_every_structural_knob() -> None:
    """The sanity-floor arm must disable all three structural mechanisms."""
    arm_keys = _structural_keys(_load(ABLATION_ARMS["hybrid_minus_all_three"]))
    assert arm_keys == {
        "hybrid.fallback_on_exception": False,
        "hybrid.adaptive_switching_enabled": False,
        "risk_dwa.progress_escape_enabled": False,
        "mppi_social.progress_escape_enabled": False,
    }


def test_only_structural_knob_differs_in_single_knob_arms() -> None:
    """No non-structural key may silently drift in a single-knob arm."""
    ref_raw = _load(REFERENCE)
    for arm, fname in ABLATION_ARMS.items():
        if arm == "hybrid_minus_all_three":
            continue
        arm_raw = _load(fname)
        # Compare every key except the structural `hybrid`/`risk_dwa`/`mppi_social` blocks.
        for section in ("risk_dwa", "mppi_social"):
            ref_section = dict(ref_raw.get(section, {}))
            arm_section = dict(arm_raw.get(section, {}))
            ref_section.pop("progress_escape_enabled", None)
            arm_section.pop("progress_escape_enabled", None)
            assert ref_section == arm_section, f"{arm}: {section} drifted"
        ref_hybrid = {
            k: v
            for k, v in ref_raw.get("hybrid", {}).items()
            if k not in ("fallback_on_exception", "adaptive_switching_enabled")
        }
        arm_hybrid = {
            k: v
            for k, v in arm_raw.get("hybrid", {}).items()
            if k not in ("fallback_on_exception", "adaptive_switching_enabled")
        }
        assert ref_hybrid == arm_hybrid, f"{arm}: hybrid block drifted"


def test_adaptive_switching_disabled_pins_initial_head() -> None:
    """With adaptive_switching_enabled=False the active head never switches."""

    class _DummyHead:
        def __init__(self, name: str) -> None:
            self.name = name
            self.calls = 0

        def plan(self, observation: dict) -> tuple[float, float]:
            self.calls += 1
            return (1.0 if self.name == "risk_dwa" else 0.5, 0.0)

        def reset(self) -> None:
            pass

    risk = _DummyHead("risk_dwa")
    orca = _DummyHead("orca")
    pred = _DummyHead("pred")
    mppi = _DummyHead("mppi")
    hybrid = HybridPortfolioAdapter(
        hybrid_config=HybridPortfolioConfig(adaptive_switching_enabled=False),
        risk_dwa=risk,
        orca=orca,
        prediction=pred,
        mppi=mppi,
    )
    # Emergency pedestrian should switch a normal hybrid, but adaptive switching is off.
    obs = {"robot": {"position": (0.0, 0.0)}, "pedestrians": {"positions": [(0.3, 0.0)]}}
    hybrid.plan(obs)
    hybrid.plan(obs)
    assert risk.calls == 2
    assert orca.calls == 0


def test_adaptive_switching_default_is_enabled() -> None:
    """The default config keeps adaptive switching on (backward compatible)."""
    cfg = build_hybrid_portfolio_build_config(_load(REFERENCE))
    assert cfg.hybrid.adaptive_switching_enabled is True


def test_delta_analyzer_skips_unusable_numeric_cells_without_nan() -> None:
    """Blank and non-finite cells must not contaminate summaries or deltas."""
    rows = [
        {
            "collision_rate": "0.2",
            "near_miss_rate": "",
            "comfort_rate": "NaN",
            "time_to_goal_norm": "inf",
            "success_rate": "0.8",
        },
        {
            "collision_rate": "0.4",
            "near_miss_rate": "0.1",
            "comfort_rate": "0.6",
            "time_to_goal_norm": "1.2",
            "success_rate": "bad",
        },
    ]

    summary = _arm_summary(rows)

    assert summary["collision_rate"] == pytest.approx(0.3)
    assert summary["near_miss_rate"] == 0.1
    assert summary["comfort_rate"] == 0.6
    assert summary["time_to_goal_norm"] == 1.2
    assert summary["success_rate"] == 0.8
    assert summary["skipped_numeric_cells"] == 4
    assert summary["skipped_numeric_cells_by_metric"] == {
        "collision_rate": 0,
        "near_miss_rate": 1,
        "comfort_rate": 1,
        "time_to_goal_norm": 1,
        "success_rate": 1,
    }
    assert _delta(summary, summary)["near_miss_rate"] == 0.0


def test_delta_analyzer_uses_null_when_no_finite_value_exists() -> None:
    """An all-invalid metric remains explicitly unavailable, never NaN."""
    empty_metric = _arm_summary([{"success_rate": ""}])
    assert empty_metric["success_rate"] is None
    assert _delta(empty_metric, empty_metric)["success_rate"] is None
