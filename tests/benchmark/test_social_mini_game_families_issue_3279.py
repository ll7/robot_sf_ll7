"""Tests for the issue #3279 Social Mini-Game scenario families (v0 first cut).

Scope of proof: each declared family is schema-valid, carries a distinct mechanism
label + distinct generator parameters, and produces a *deterministic* generated
scenario through the existing ``generate_scenario`` path. This is diagnostic-smoke
evidence that the families are runnable inputs; it makes no planner-ranking or
benchmark-strength mechanism claim.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from robot_sf.benchmark.scenario_generator import generate_scenario
from robot_sf.benchmark.scenario_schema import validate_scenario_list

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FAMILIES_PATH = (
    _REPO_ROOT / "configs" / "scenarios" / "sets" / "issue_3279_social_mini_game_families_v0.yaml"
)

_EXPECTED_FAMILIES = {
    "doorway",
    "hallway",
    "intersection",
    "blind_corner",
    "crowded_traffic",
}


def _load_families() -> list[dict]:
    """Return the Social Mini-Game scenario matrix as a list of scenario dicts."""
    with _FAMILIES_PATH.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    assert isinstance(payload, list), "scenario matrix must be a top-level YAML list"
    return payload


def test_families_file_is_schema_valid() -> None:
    """The Social Mini-Game matrix must pass the benchmark scenario schema."""
    scenarios = _load_families()
    errors = validate_scenario_list(scenarios)
    assert errors == [], f"schema validation errors: {errors}"


def test_all_five_mini_game_families_present_with_distinct_labels() -> None:
    """Each declared mini-game family appears once with a distinct mechanism label."""
    scenarios = _load_families()
    families = [s["metadata"]["social_mini_game_family"] for s in scenarios]
    mechanism_labels = [s["metadata"]["mechanism_aware_suite_id"] for s in scenarios]

    assert set(families) == _EXPECTED_FAMILIES
    assert len(families) == len(set(families)), "family names must be unique"
    assert len(mechanism_labels) == len(set(mechanism_labels)), "mechanism labels must be unique"


def test_each_family_uses_distinct_generator_parameters() -> None:
    """Families must differ on the (flow, obstacle, density) generator trigger combo."""
    scenarios = _load_families()
    combos = [(s["flow"], s["obstacle"], s["density"]) for s in scenarios]
    assert len(combos) == len(set(combos)), f"families must use distinct param combos: {combos}"


def test_each_family_generates_a_deterministic_runnable_scenario() -> None:
    """Every family produces an identical generated scenario for a fixed seed."""
    scenarios = _load_families()
    for scenario in scenarios:
        seed = scenario["seeds"][0]
        first = generate_scenario(scenario, seed)
        second = generate_scenario(scenario, seed)

        assert first.state.ndim == 2
        assert first.state.shape[1] == 7
        assert first.state.shape[0] >= 1
        assert isinstance(first.obstacles, list)
        # Determinism: same id + seed reproduces the same layout.
        assert np.array_equal(first.state, second.state), (
            f"{scenario['id']} is not deterministic for seed {seed}"
        )

    # Obstacle-bearing families must actually carry obstacle geometry.
    by_id = {s["id"]: s for s in scenarios}
    doorway = generate_scenario(by_id["smg_doorway_v0"], by_id["smg_doorway_v0"]["seeds"][0])
    blind = generate_scenario(by_id["smg_blind_corner_v0"], by_id["smg_blind_corner_v0"]["seeds"][0])
    assert doorway.obstacles, "doorway (bottleneck) must have obstacle geometry"
    assert blind.obstacles, "blind_corner (maze) must have obstacle geometry"


def test_families_carry_diagnostic_only_claim_boundary() -> None:
    """Every family must declare a diagnostic-smoke, not-benchmark claim boundary."""
    scenarios = _load_families()
    for scenario in scenarios:
        metadata = scenario["metadata"]
        assert metadata["evidence_tier"] == "diagnostic_smoke"
        claim = metadata["claim_boundary"].lower()
        assert "not" in claim and "benchmark" in claim
        assert int(metadata["issue"]) == 3279
