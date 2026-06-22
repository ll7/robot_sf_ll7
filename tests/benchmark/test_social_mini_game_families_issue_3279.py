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

from robot_sf.benchmark.aggregate import read_jsonl
from robot_sf.benchmark.scenario_generator import generate_scenario
from robot_sf.benchmark.scenario_schema import validate_scenario_list
from scripts.demo.run_robot_sf_smoke import run_demo

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
_EXPECTED_CONTROL_KEYS = {
    "width_m",
    "occlusion_geometry",
    "start_timing_s",
    "yielding_pressure",
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


def test_each_family_exposes_social_mini_game_controls() -> None:
    """Issue #3423 controls must be visible as first-class manifest metadata."""
    scenarios = _load_families()
    for scenario in scenarios:
        metadata = scenario["metadata"]
        assert "unexposed_parameters" not in metadata
        assert int(metadata["control_exposure_issue"]) == 3423

        controls = metadata["social_mini_game_controls"]
        assert set(controls) == _EXPECTED_CONTROL_KEYS

        width = controls["width_m"]
        assert width["support"] == "documented_equivalent"
        assert width["value_m"] > 0.0
        assert width["equivalent"]

        occlusion = controls["occlusion_geometry"]
        assert occlusion["support"] == "documented_equivalent"
        assert occlusion["value"] in {
            "doorway_bottleneck",
            "none",
            "l_corner_blind_corner",
        }
        assert occlusion["equivalent"]

        timing = controls["start_timing_s"]
        assert timing["support"] == "fixed_seed_equivalent"
        assert timing["value_s"] >= 0.0
        assert "seed" in timing["equivalent"].lower()

        yielding = controls["yielding_pressure"]
        assert yielding["support"] == "documented_equivalent"
        assert yielding["level"] in {"medium", "high"}
        assert yielding["equivalent"]


def test_blind_corner_declares_l_corner_equivalent_and_generates_obstacles() -> None:
    """The blind-corner family should no longer be only a vague maze proxy."""
    scenarios = _load_families()
    blind = next(scenario for scenario in scenarios if scenario["id"] == "smg_blind_corner_v0")

    controls = blind["metadata"]["social_mini_game_controls"]
    assert controls["occlusion_geometry"]["value"] == "l_corner_blind_corner"
    assert "l-corner" in controls["occlusion_geometry"]["equivalent"].lower()
    assert "L-corner" in blind["metadata"]["generator_param_mapping"]

    generated = generate_scenario(blind, blind["seeds"][0])
    vertical_segments = [
        obstacle for obstacle in generated.obstacles if obstacle[0] == obstacle[2]
    ]
    horizontal_segments = [
        obstacle for obstacle in generated.obstacles if obstacle[1] == obstacle[3]
    ]
    assert vertical_segments, "blind-corner equivalent should include vertical occluder geometry"
    assert horizontal_segments, "blind-corner equivalent should include horizontal occluder geometry"


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
    blind = generate_scenario(
        by_id["smg_blind_corner_v0"], by_id["smg_blind_corner_v0"]["seeds"][0]
    )
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


def test_simple_policy_smoke_runs_all_families(tmp_path: Path) -> None:
    """A baseline-safe planner smoke should emit one episode record per mini-game family."""
    output_root = tmp_path / "issue_3279_social_mini_game_smoke"

    summary = run_demo(
        output_root=output_root,
        matrix=_FAMILIES_PATH,
        planners=("simple_policy",),
        horizon=30,
        workers=1,
    )

    planner = summary["planners"][0]
    assert summary["passed"] is True
    assert planner["status"] == "passed"
    assert planner["record_count"] == len(_EXPECTED_FAMILIES)
    assert planner["run_summary"]["total_jobs"] == len(_EXPECTED_FAMILIES)
    assert planner["run_summary"]["written"] == len(_EXPECTED_FAMILIES)

    episodes_path = output_root / "episodes" / "simple_policy.jsonl"
    records = read_jsonl(episodes_path, strict=True)
    scenario_ids = {record["scenario_id"] for record in records}
    canonical_algorithms = {
        record["algorithm_metadata"]["canonical_algorithm"] for record in records
    }

    assert scenario_ids == {scenario["id"] for scenario in _load_families()}
    assert canonical_algorithms == {"goal"}
