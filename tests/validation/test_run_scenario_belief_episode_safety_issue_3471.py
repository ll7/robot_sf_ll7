"""Tests for the episode-level ScenarioBelief planner-safety experiment (#3471)."""

from __future__ import annotations

import numpy as np

from robot_sf.planner.scenario_belief_adapter import project_scenario_belief_for_planner
from scripts.validation.run_scenario_belief_episode_safety_issue_3471 import (
    MODES,
    UNCERTAINTY_REPRESENTATIONS,
    EpisodeParams,
    _is_commit,
    build_belief_for_mode,
    build_initial_state,
    classify_decision,
    min_separation,
    run_episode,
    run_matrix,
)

_FAST = EpisodeParams(max_steps=40)


def test_all_modes_run_and_emit_expected_metrics():
    """Every belief mode produces the episode metric schema."""
    keys = {
        "mode",
        "collision",
        "min_separation",
        "unsafe_commit_steps",
        "commit_steps",
        "progress",
        "uncertainty_consumed",
        "fail_closed",
    }
    for mode in MODES:
        row = run_episode(mode, seed=101, params=_FAST)
        assert keys <= set(row)
        assert row["mode"] == mode


def test_episode_is_deterministic():
    """Same mode + seed yields identical metrics (excluding wall-clock runtime)."""
    a = run_episode("uncertain_dropped", seed=104, params=_FAST)
    b = run_episode("uncertain_dropped", seed=104, params=_FAST)
    a.pop("runtime_sec")
    b.pop("runtime_sec")
    assert a == b


def test_default_representation_preserves_known_issue_3471_contrast():
    """Default ``belief_drop`` keeps the known #3471 fixture contract."""
    implicit = run_matrix([101, 102], _FAST)
    explicit = run_matrix([101, 102], _FAST, uncertainty_representation="belief_drop")
    implicit.pop("episodes")
    explicit.pop("episodes")
    assert implicit == explicit


def test_each_uncertainty_representation_runs_and_is_reported():
    """Issue #3557 harness parameterizes representation without promoting a claim."""
    for representation in UNCERTAINTY_REPRESENTATIONS:
        row = run_episode(
            "uncertain_dropped",
            seed=101,
            params=_FAST,
            uncertainty_representation=representation,
        )
        assert row["uncertainty_representation"] == representation

        report = run_matrix([101], _FAST, uncertainty_representation=representation)
        assert report["followup_issue"] == 3557
        assert report["uncertainty_representation"] == representation
        assert (
            "not a cross-representation generalization claim"
            in (report["uncertainty_representation_contract"]["claim_boundary"])
        )
        assert set(report["by_mode"]) == set(MODES)


def test_unknown_uncertainty_representation_fails_closed():
    """Unknown representation names fail closed instead of silently reusing belief_drop."""
    try:
        run_matrix([101], _FAST, uncertainty_representation="unknown")
    except ValueError as exc:
        assert "unknown uncertainty representation" in str(exc)
    else:
        raise AssertionError("unknown uncertainty representation was accepted")


def test_retained_matches_oracle():
    """Retaining uncertain agents (gate off) is behaviorally identical to oracle.

    This is the representational-vs-actual separation: degraded belief alone (without the planner
    acting on it) must not change safety outcomes.
    """
    seeds = [101, 102, 103, 104]
    oracle = [run_episode("oracle", s, _FAST) for s in seeds]
    retained = [run_episode("uncertain_retained", s, _FAST) for s in seeds]
    for o, r in zip(oracle, retained, strict=True):
        assert o["collision"] == r["collision"]
        assert o["unsafe_commit_steps"] == r["unsafe_commit_steps"]
        assert o["min_separation"] == r["min_separation"]


def test_dropping_does_not_improve_safety():
    """Dropping uncertain agents must not reduce unsafe commitment vs retaining (hypothesis direction)."""
    seeds = list(range(101, 109))
    retained = sum(
        run_episode("uncertain_retained", s, _FAST)["unsafe_commit_steps"] for s in seeds
    )
    dropped = sum(run_episode("uncertain_dropped", s, _FAST)["unsafe_commit_steps"] for s in seeds)
    assert dropped >= retained


def test_unsupported_planner_fails_closed():
    """An unsupported planner key fails closed (no uncertainty consumed)."""
    state = build_initial_state(101, _FAST)
    belief = build_belief_for_mode(state, "uncertain_dropped", _FAST)
    proj = project_scenario_belief_for_planner(belief, planner_key="totally_unsupported")
    assert proj.compatibility["status"] == "fail_closed"
    assert proj.compatibility["uncertainty_consumed"] is False


def test_degraded_mode_lowers_corridor_existence():
    """Uncertain modes degrade the nearest-corridor agent's existence below the gate threshold."""
    state = build_initial_state(101, _FAST)
    oracle = build_belief_for_mode(state, "oracle", _FAST)
    dropped = build_belief_for_mode(state, "uncertain_dropped", _FAST)
    min_oracle = min(a.existence_probability for a in oracle.agents)
    min_dropped = min(a.existence_probability for a in dropped.agents)
    assert min_dropped < min_oracle
    assert min_dropped < 0.5


def test_is_commit_detects_commit_speed():
    """COMMIT detection keys off the configured commit speed."""
    assert _is_commit(0.95) is True
    assert _is_commit(0.0) is False
    assert _is_commit(0.35) is False


def test_min_separation_subtracts_radii():
    """Minimum separation is surface-to-surface (radii removed)."""
    state = build_initial_state(101, _FAST)
    state.robot_pos = np.array([5.0, 5.0], dtype=np.float32)
    state.ped_pos = np.array([[6.0, 5.0], [9.0, 9.0]], dtype=np.float32)
    sep = min_separation(state, _FAST)
    assert sep == round(1.0 - _FAST.robot_radius - _FAST.ped_radius, 4)


def test_classify_decision_branches():
    """classify_decision maps the dropped-vs-retained contrast to a decision."""
    base = {"uncertainty_consumed_any": True, "collision_rate": 0.0, "worst_min_separation": 1.0}
    worse = {**base, "total_unsafe_commit_steps": 10}
    equal = {**base, "total_unsafe_commit_steps": 0}
    assert (
        classify_decision({"uncertain_retained": equal, "uncertain_dropped": worse})["decision"]
        == "revise"
    )
    assert (
        classify_decision({"uncertain_retained": equal, "uncertain_dropped": equal})["decision"]
        == "inconclusive"
    )
    assert classify_decision({"uncertain_retained": equal})["decision"] == "blocked"
    gate_off = {**equal, "uncertainty_consumed_any": False}
    assert (
        classify_decision({"uncertain_retained": equal, "uncertain_dropped": gate_off})["decision"]
        == "inconclusive"
    )


def test_run_matrix_structure():
    """run_matrix emits all modes, a decision, and the claim boundary."""
    report = run_matrix([101, 102], _FAST)
    assert set(report["by_mode"]) == set(MODES)
    assert report["decision"]["decision"] in {"revise", "continue", "inconclusive", "blocked"}
    assert "not paper-grade" in report["claim_boundary"]
