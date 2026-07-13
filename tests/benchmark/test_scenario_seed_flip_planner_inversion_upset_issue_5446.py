"""Synthetic-fixture tests for the seed-flip / planner-inversion miner (issue #5446).

The fixtures encode *known* answers — a hard seed flip, a stable negative
control, a held-out planner upset, and a non-upset control — and assert the
miner recovers them, reports posterior/interval uncertainty with effective
denominators, estimates planner strength leave-one-scenario-out, records every
exclusion, and consumes (or reports unavailable) the sibling-issue signals
without fabricating them.

Test ids intentionally contain ``scenario``, ``seed``, ``inversion``, and
``upset`` so they match the issue's ``-k`` selector.
"""

from __future__ import annotations

from typing import Any

import pytest

from robot_sf.benchmark.seed_flip_mining import (
    SeedFlipMiningError,
    mine_seed_flip_inversion_candidates,
)


def _row(
    scenario: str,
    planner: str,
    seed: int,
    success: int,
    **overrides: Any,
) -> dict[str, Any]:
    """Build a fully-eligible synthetic result row, with optional field overrides."""
    row: dict[str, Any] = {
        "episode_id": f"{scenario}-{planner}-{seed}",
        "scenario_id": scenario,
        "seed": seed,
        "config_hash": "cfg-abc",
        "repo_commit": "deadbeef",
        "execution_mode": "native",
        "collision_semantics": "typed",
        "scenario_params": {"algo": planner},
        "metrics": {"success": float(success)},
    }
    row.update(overrides)
    return row


def _background_rows() -> list[dict[str, Any]]:
    """Three scenarios where ``strong`` always wins and ``weak`` always loses.

    This makes ``strong`` high-strength and ``weak`` low-strength when estimated
    leave-one-scenario-out, so an upset on a *different* scenario is genuine.
    """
    rows: list[dict[str, Any]] = []
    for scenario in ("s1", "s2", "s3"):
        for seed in (10, 11, 12):
            rows.append(_row(scenario, "strong", seed, 1))
            rows.append(_row(scenario, "weak", seed, 0))
    return rows


def test_scenario_seed_flip_recovered_with_posterior_and_effective_denominator() -> None:
    """A knife-edge cell is recovered with posterior, Wilson interval, and denominator."""
    # A knife-edge cell: same (scenario, planner) flips across seeds.
    rows = _background_rows() + [
        _row("flipper", "weak", 20, 1),
        _row("flipper", "weak", 21, 0),
        _row("flipper", "weak", 22, 1),
        _row("flipper", "weak", 23, 0),
    ]
    manifest = mine_seed_flip_inversion_candidates(rows)
    flips = [c for c in manifest["candidates"] if c["archetype"] == "seed_flip"]
    flip = next(c for c in flips if c["scenario_id"] == "flipper")

    sf = flip["seed_flip"]
    assert sf["raw_success_seeds"] == 2
    assert sf["raw_failure_seeds"] == 2
    assert sf["effective_denominator"] == 4
    assert 0.0 < sf["posterior_mean"] < 1.0
    assert sf["entropy_bits"] > 0.9  # near-maximal for a 50/50 flip
    low, high = sf["interval"]
    assert 0.0 <= low < high <= 1.0
    assert sf["interval_method"] == "wilson_score"
    # Raw per-seed outcomes retained for reproducibility.
    assert flip["reproducibility"]["raw_seed_outcomes"] == {"20": 1, "21": 0, "22": 1, "23": 0}


def test_scenario_stable_cell_is_negative_control_no_seed_flip() -> None:
    """An all-success cell is a negative control and yields no seed-flip candidate."""
    # An all-success cell must NOT be mined as a seed flip.
    rows = _background_rows() + [
        _row("stable", "weak", 30, 1),
        _row("stable", "weak", 31, 1),
        _row("stable", "weak", 32, 1),
    ]
    manifest = mine_seed_flip_inversion_candidates(rows)
    flip_scenarios = {
        c["scenario_id"] for c in manifest["candidates"] if c["archetype"] == "seed_flip"
    }
    assert "stable" not in flip_scenarios


def test_scenario_planner_upset_inversion_uses_heldout_strength() -> None:
    """A planner upset uses leave-one-scenario-out strength and retains paired outcomes."""
    # On `upset_scene` the weak planner wins and the strong planner loses.
    rows = _background_rows() + [
        _row("upset_scene", "weak", 40, 1),
        _row("upset_scene", "weak", 41, 1),
        _row("upset_scene", "strong", 40, 0),
        _row("upset_scene", "strong", 41, 0),
    ]
    manifest = mine_seed_flip_inversion_candidates(rows)
    upsets = [c for c in manifest["candidates"] if c["archetype"] == "planner_upset"]
    upset = next(c for c in upsets if c["scenario_id"] == "upset_scene")

    uo = upset["upset_outcome"]
    assert uo["underdog_planner"] == "weak"
    assert uo["favorite_planner"] == "strong"
    # Held-out strength is estimated from the OTHER scenarios only: weak=0, strong=1.
    assert uo["underdog_heldout_strength"] == pytest.approx(0.0)
    assert uo["favorite_heldout_strength"] == pytest.approx(1.0)
    assert upset["heldout_planner_skill_gap"] == pytest.approx(1.0)
    assert uo["underdog_cell_success"] > uo["favorite_cell_success"]
    # Raw paired outcomes retained for both planners.
    assert set(uo["raw_paired_outcomes"]) == {"weak", "strong"}


def test_scenario_no_inversion_when_favorite_also_wins_cell() -> None:
    """No upset is emitted when the stronger planner also wins the cell."""
    # Negative control: strong wins the cell too -> no upset.
    rows = _background_rows() + [
        _row("no_upset", "weak", 50, 0),
        _row("no_upset", "weak", 51, 0),
        _row("no_upset", "strong", 50, 1),
        _row("no_upset", "strong", 51, 1),
    ]
    manifest = mine_seed_flip_inversion_candidates(rows)
    upset_scenarios = {
        c["scenario_id"] for c in manifest["candidates"] if c["archetype"] == "planner_upset"
    }
    assert "no_upset" not in upset_scenarios


def test_scenario_seed_eligibility_gates_exclude_with_reasons() -> None:
    """Ineligible rows are excluded with concrete, typed reasons."""
    good = _background_rows()
    bad_rows = [
        _row("flip_x", "weak", 60, 1),
        _row("flip_x", "weak", 61, 0, execution_mode="fallback"),  # non-native
        _row("flip_x", "weak", 62, 1, repo_commit=""),  # missing provenance
        _row("flip_x", "weak", 63, 0, collision_semantics="untyped"),  # untyped collision
        _row(
            "flip_x",
            "weak",
            64,
            1,
            release="0.0.2",
            metrics={"total_collision_count": 1.0},
        ),  # withdrawn collision-derived field
    ]
    manifest = mine_seed_flip_inversion_candidates(good + bad_rows)
    reasons = {e["reason"] for e in manifest["exclusions"]}
    assert any(r.startswith("non_native_execution") for r in reasons)
    assert any(r.startswith("missing_provenance") for r in reasons)
    assert any(r.startswith("untyped_collision_semantics") for r in reasons)


def test_scenario_seed_withdrawn_collision_derived_metric_excluded_for_release_0_0_2() -> None:
    """Withdrawn release-0.0.2 collision-derived outcomes are excluded (issue #5097)."""
    metric = "success_rate_collision_gate"
    # Eligible non-0.0.2 rows carrying the same metric keep the miner from failing
    # closed, so the *withdrawn* 0.0.2 rows can be observed in the exclusions.
    eligible = [
        _row("clean", "weak", 80, 1, metrics={metric: 1.0}),
        _row("clean", "weak", 81, 0, metrics={metric: 0.0}),
        _row("clean", "strong", 80, 1, metrics={metric: 1.0}),
    ]
    withdrawn = [
        _row("coll", "weak", 70, 1, release="0.0.2", metrics={metric: 1.0}),
        _row("coll", "weak", 71, 0, release="0.0.2", metrics={metric: 0.0}),
    ]
    manifest = mine_seed_flip_inversion_candidates(eligible + withdrawn, outcome_metric=metric)
    reasons = [e["reason"] for e in manifest["exclusions"]]
    assert any("withdrawn_collision_derived_field" in r for r in reasons)
    # The withdrawn scenario never becomes a candidate.
    assert all(c["scenario_id"] != "coll" for c in manifest["candidates"])


def test_scenario_seed_external_signals_unavailable_then_consumed() -> None:
    """Sibling-issue signals are unavailable when absent and consumed when provided."""
    rows = _background_rows() + [
        _row("flipper", "weak", 20, 1),
        _row("flipper", "weak", 21, 0),
    ]
    # Not provided -> reported unavailable, never fabricated.
    manifest = mine_seed_flip_inversion_candidates(rows)
    flip = next(c for c in manifest["candidates"] if c["scenario_id"] == "flipper")
    assert flip["oracle_regret"]["status"] == "unavailable"
    assert flip["oracle_regret"]["consumes_issue"] == "#5302"
    assert manifest["external_signals"]["quality_diversity"]["provided"] is False

    # Provided -> consumed and linked to its owning issue.
    external = {"oracle_regret": {"flipper::weak": 0.42}}
    manifest2 = mine_seed_flip_inversion_candidates(rows, external=external)
    flip2 = next(
        c
        for c in manifest2["candidates"]
        if c["scenario_id"] == "flipper" and c["archetype"] == "seed_flip"
    )
    assert flip2["oracle_regret"]["status"] == "consumed"
    assert flip2["oracle_regret"]["value"] == 0.42


def test_scenario_seed_archetype_availability_and_pareto_selection() -> None:
    """Archetype availability and Pareto selection are reported over recorded candidates."""
    rows = _background_rows() + [
        # seed flip with a temporal boundary margin -> causal_divergence available.
        _row("flipper", "weak", 20, 1, metrics={"success": 1.0, "temporal_boundary_margin": 0.3}),
        _row("flipper", "weak", 21, 0, metrics={"success": 0.0, "temporal_boundary_margin": 0.1}),
        # upset -> planners disagree -> disagreement_recovery available.
        _row("upset_scene", "weak", 40, 1),
        _row("upset_scene", "weak", 41, 1),
        _row("upset_scene", "strong", 40, 0),
        _row("upset_scene", "strong", 41, 0),
    ]
    manifest = mine_seed_flip_inversion_candidates(rows)
    av = manifest["archetype_availability"]
    assert av["seed_flip"]["available"] is True
    assert av["planner_upset"]["available"] is True
    assert av["causal_divergence"]["available"] is True
    assert av["disagreement_recovery"]["available"] is True

    # Every candidate is recorded before selection; selected are a subset.
    all_ids = {c["candidate_id"] for c in manifest["candidates"]}
    selected = {c["candidate_id"] for c in manifest["candidates"] if c["selected"]}
    assert selected  # non-empty Pareto frontier
    assert selected.issubset(all_ids)
    assert manifest["summary"]["n_selected"] == len(selected)


def test_scenario_seed_fail_closed_on_empty_and_all_ineligible() -> None:
    """The miner fails closed on empty and all-ineligible inputs."""
    with pytest.raises(SeedFlipMiningError):
        mine_seed_flip_inversion_candidates([])
    all_bad = [_row("s", "p", 1, 1, execution_mode="degraded")]
    with pytest.raises(SeedFlipMiningError):
        mine_seed_flip_inversion_candidates(all_bad)


def test_scenario_seed_every_candidate_reports_uncertainty_or_paired_outcomes() -> None:
    """Every candidate reports posterior/interval or retained paired outcomes."""
    rows = _background_rows() + [
        _row("flipper", "weak", 20, 1),
        _row("flipper", "weak", 21, 0),
        _row("upset_scene", "weak", 40, 1),
        _row("upset_scene", "weak", 41, 1),
        _row("upset_scene", "strong", 40, 0),
        _row("upset_scene", "strong", 41, 0),
    ]
    manifest = mine_seed_flip_inversion_candidates(rows)
    for cand in manifest["candidates"]:
        if cand["archetype"] == "seed_flip":
            sf = cand["seed_flip"]
            assert "posterior_mean" in sf and "interval" in sf
            assert sf["effective_denominator"] >= 2
        elif cand["archetype"] == "planner_upset":
            assert cand["heldout_planner_skill_gap"] is not None
            assert "raw_paired_outcomes" in cand["upset_outcome"]
