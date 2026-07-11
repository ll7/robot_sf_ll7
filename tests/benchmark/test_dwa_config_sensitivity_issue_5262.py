"""Contract checks for the bounded DWA configuration-sensitivity diagnostic (#5262)."""

import csv
import importlib.util
from pathlib import Path

_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts/benchmark/run_dwa_config_sensitivity_issue_5262.py"
)
_SPEC = importlib.util.spec_from_file_location("issue_5262_runner", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
runner = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(runner)
_EVIDENCE_DIR = (
    Path(__file__).resolve().parents[2]
    / "docs/context/evidence/issue_5262_dwa_config_sensitivity_2026-07-11"
)


def test_issue_5262_manifest_is_bounded_and_selects_failure_modes() -> None:
    """The diagnostic stays within 30 fixed-seed CPU episodes and includes both failure modes."""
    manifest = runner.load_manifest(runner.DEFAULT_MANIFEST)
    scenarios = runner.select_scenarios(manifest)

    assert len(manifest["config_points"]) == runner.EXPECTED_CONFIG_POINT_COUNT
    assert len(scenarios) == runner.EXPECTED_SCENARIO_COUNT
    assert len(scenarios) * manifest["seeds_per_scenario"] * len(manifest["config_points"]) == 27
    assert {scenario["metadata"]["archetype"] for scenario in scenarios} == {
        "bottleneck",
        "cross_trap",
        "t_intersection",
    }


def test_issue_5262_config_points_preserve_canonical_baseline_and_vary_all_axes() -> None:
    """The campaign has a canonical reference plus reversible dynamics, tolerance, and objective probes."""
    manifest = runner.load_manifest(runner.DEFAULT_MANIFEST)
    canonical = runner.effective_config(manifest, "canonical")
    faster = runner.effective_config(manifest, "mobility_and_goal")
    progress = runner.effective_config(manifest, "progress_weight_400")

    assert canonical["max_linear_speed"] < faster["max_linear_speed"]
    assert canonical["max_linear_acceleration"] < faster["max_linear_acceleration"]
    assert canonical["goal_tolerance"] < faster["goal_tolerance"]
    assert canonical["progress_weight"] < progress["progress_weight"]
    assert canonical["clearance_weight"] > progress["clearance_weight"]


def test_issue_5262_evidence_rows_match_the_bounded_negative_result() -> None:
    """The durable tables retain all 27 diagnostic rows and no configuration success claim."""
    with (_EVIDENCE_DIR / "dwa_config_sensitivity_per_cell_rows.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        cells = list(csv.DictReader(handle))
    with (_EVIDENCE_DIR / "dwa_config_sensitivity_episode_rows.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        episodes = list(csv.DictReader(handle))

    assert len(cells) == 9
    assert len(episodes) == 27
    assert {row["config_id"] for row in cells} == {
        "canonical",
        "mobility_and_goal",
        "progress_weight_400",
    }
    assert all(row["route_complete"] == "0" for row in cells)
    assert sum(int(row["timeout_event"]) for row in episodes) == 15
    assert sum(int(row["collision_event"]) for row in episodes) == 12
