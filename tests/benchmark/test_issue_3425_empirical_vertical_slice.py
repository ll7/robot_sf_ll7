"""Contract checks for the issue #3425 empirical vertical-slice packet."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

from robot_sf.benchmark.camera_ready._config import _load_campaign_scenarios
from robot_sf.benchmark.camera_ready_campaign import load_campaign_config

REPO_ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN_CONFIG = REPO_ROOT / "configs/benchmarks/issue_3425_empirical_vertical_slice_smoke.yaml"
RESEARCH_MANIFEST = (
    REPO_ROOT / "configs/benchmarks/issue_3425_empirical_vertical_slice_manifest.yaml"
)
PACKET_RUNNER = REPO_ROOT / "scripts/validation/run_research_campaign_manifest.py"


def test_issue_3425_campaign_config_pins_small_baseline_safe_slice() -> None:
    """The launch config must stay small, explicit, and non-paper-facing."""
    cfg = load_campaign_config(CAMPAIGN_CONFIG)

    assert cfg.name == "issue_3425_empirical_vertical_slice_smoke"
    assert cfg.paper_facing is False
    assert cfg.seed_policy.mode == "seed-set"
    assert cfg.seed_policy.seed_set == "eval"
    assert cfg.workers == 1
    assert cfg.horizon == 100
    assert cfg.export_publication_bundle is False

    assert cfg.scenario_candidates.names == (
        "francis2023_blind_corner",
        "francis2023_intersection_no_gesture",
        "francis2023_pedestrian_obstruction",
    )
    assert [(planner.key, planner.algo, planner.benchmark_profile) for planner in cfg.planners] == [
        ("goal", "goal", "baseline-safe"),
        ("social_force", "social_force", "baseline-safe"),
        ("orca", "orca", "baseline-safe"),
    ]
    assert all(planner.planner_group == "core" for planner in cfg.planners)
    assert {planner.key: planner.socnav_missing_prereq_policy for planner in cfg.planners} == {
        "goal": "fail-fast",
        "social_force": "fail-fast",
        "orca": "fail-fast",
    }


def test_issue_3425_campaign_resolves_3x3x3_matrix() -> None:
    """The camera-ready loader should resolve exactly 27 planned rows."""
    cfg = load_campaign_config(CAMPAIGN_CONFIG)
    scenarios = _load_campaign_scenarios(cfg)

    scenario_names = [scenario["name"] for scenario in scenarios]
    assert set(scenario_names) == set(cfg.scenario_candidates.names)
    assert {tuple(scenario["seeds"]) for scenario in scenarios} == {(111, 112, 113)}
    assert len(scenarios) * len(cfg.planners) * 3 == 27


def test_issue_3425_research_manifest_records_metric_blockers_and_rows(
    tmp_path: Path,
) -> None:
    """The handoff manifest should make blockers visible and build dry-run rows."""
    manifest = yaml.safe_load(RESEARCH_MANIFEST.read_text(encoding="utf-8"))

    assert manifest["campaign"]["parent_issue"] == 3425
    assert manifest["campaign"]["evidence_tier"] == "smoke-readiness"
    blocker_issues = {entry["issue"] for entry in manifest["metrics"]["semantics_blockers"]}
    assert blocker_issues == {3724, 3482, 3723, 3699, 3725}
    assert "successful_evidence" not in manifest["row_status_policy"]["fail_closed_values"]

    output_dir = tmp_path / "packet"
    completed = subprocess.run(
        [
            sys.executable,
            str(PACKET_RUNNER),
            str(RESEARCH_MANIFEST),
            "--output-dir",
            str(output_dir),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["campaign_id"] == "issue_3425_empirical_vertical_slice_smoke"
    assert summary["row_status_summary"] == {"diagnostic_only": 27}
