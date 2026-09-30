"""Real multi-seed campaign resume coverage for issue #10025."""

from __future__ import annotations

import json
import shutil
from typing import TYPE_CHECKING

import pytest
import yaml

from robot_sf.benchmark.camera_ready._config import _load_campaign_scenarios
from robot_sf.benchmark.camera_ready._resume_plan import (
    ResumeMismatchError,
    _expected_jobs,
    build_resume_plan,
)
from robot_sf.benchmark.camera_ready_campaign import load_campaign_config, run_campaign
from robot_sf.common.artifact_paths import get_repository_root

if TYPE_CHECKING:
    from pathlib import Path

ARM = "social_force__differential_drive"


@pytest.fixture(scope="module")
def real_campaign(tmp_path_factory):
    """Produce six real rows once, without injecting campaign or runner collaborators."""
    root = tmp_path_factory.mktemp("resume-jobs")
    repo = get_repository_root()
    matrix = root / "scenarios.yaml"
    matrix.write_text(
        yaml.safe_dump(
            {
                "includes": [
                    str(
                        repo
                        / "configs/scenarios/classic_interactions_francis2023_release_0_0_8_v1.yaml"
                    )
                ],
                "select_scenarios": ["francis2023_leave_group", "francis2023_blind_corner"],
            }
        ),
        encoding="utf-8",
    )
    config = {
        "name": "resume-jobs",
        "paper_facing": False,
        "scenario_matrix": str(matrix),
        "seed_policy": {"mode": "fixed-list", "seeds": [1001, 1002, 1003]},
        "snqi_weights": "configs/benchmarks/snqi_weights_camera_ready_v3.json",
        "snqi_baseline": "configs/benchmarks/snqi_baseline_camera_ready_v3.json",
        "snqi_contract": {"enabled": False},
        "workers": 1,
        "horizon": 1,
        "dt": 0.1,
        "resume": True,
        "bootstrap_samples": 20,
        "bootstrap_seed": 1001,
        "kinematics_matrix": ["differential_drive"],
        "export_publication_bundle": False,
        "planners": [
            {
                "key": "social_force",
                "algo": "social_force",
                "benchmark_profile": "baseline-safe",
                "algo_config": "configs/algos/social_force_resolution_independent_v2_kernel_wrapped_v2.yaml",
            }
        ],
    }
    config_path = root / "campaign.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    cfg = load_campaign_config(config_path)
    result = run_campaign(
        cfg, output_root=root, campaign_id="fixture", skip_publication_bundle=True
    )
    campaign = root / "fixture"
    assert (
        json.loads((campaign / "reports" / "campaign_integrity.json").read_text())["status"]
        == "valid"
    )
    assert result is not None
    return cfg, campaign, _load_campaign_scenarios(cfg)


def _plan(cfg, campaign: Path, scenarios):
    """Inspect the same effective scenarios and planner identity used by the campaign."""
    planner = cfg.planners[0]
    return build_resume_plan(
        campaign / "runs",
        planners=[
            {
                "key": planner.key,
                "algo": planner.algo,
                "algo_config_path": str(planner.algo_config_path),
            }
        ],
        kinematics_matrix=["differential_drive"],
        scenarios=scenarios,
    )[0]


def test_partial_arm_runs_only_missing_rows_and_becomes_valid(real_campaign, tmp_path):
    """Four of six real rows must lead to two dispatches, preserving the banked bytes."""
    cfg, original, scenarios = real_campaign
    campaign = tmp_path / "fixture"
    shutil.copytree(original, campaign)
    episodes = campaign / "runs" / ARM / "episodes.jsonl"
    full = episodes.read_bytes().splitlines(keepends=True)
    # Keep a non-prefix subset to prove filtering uses identities, not a row offset.
    banked = [full[index] for index in (0, 2, 3, 5)]
    episodes.write_bytes(b"".join(banked))
    verdict = _plan(cfg, campaign, scenarios)
    assert verdict.expected_total == 6
    assert verdict.verdict == "continue-from-4"
    assert verdict.episodes_remaining == 2

    run_campaign(cfg, output_root=tmp_path, campaign_id="fixture", skip_publication_bundle=True)
    after = episodes.read_bytes().splitlines(keepends=True)
    assert after[:4] == banked
    assert len(after) == 6
    rows = [json.loads(line) for line in after]
    assert {(row["scenario_id"], row["seed"]) for row in rows} == {
        (scenario, seed)
        for scenario in ("francis2023_leave_group", "francis2023_blind_corner")
        for seed in (1001, 1002, 1003)
    }
    summary = json.loads((campaign / "runs" / ARM / "summary.json").read_text())
    assert summary["written"] == 2
    assert (
        json.loads((campaign / "reports" / "campaign_integrity.json").read_text())["status"]
        == "valid"
    )


def test_complete_arm_skips_all_six_identities(real_campaign, tmp_path):
    """A complete multi-seed arm has the right denominator and reuses its exact row bytes."""
    cfg, original, scenarios = real_campaign
    campaign = tmp_path / "fixture"
    shutil.copytree(original, campaign)
    episodes = campaign / "runs" / ARM / "episodes.jsonl"
    before = episodes.read_bytes()
    verdict = _plan(cfg, campaign, scenarios)
    assert verdict.expected_total == 6
    assert verdict.verdict == "skip-complete"
    assert verdict.episodes_remaining == 0
    run_campaign(cfg, output_root=tmp_path, campaign_id="fixture", skip_publication_bundle=True)
    assert episodes.read_bytes() == before
    assert json.loads((campaign / "resume_plan.json").read_text())["arms_skip_complete"] == 1
    assert (
        json.loads((campaign / "reports" / "campaign_integrity.json").read_text())["status"]
        == "valid"
    )


@pytest.mark.parametrize("replacement", ["duplicate", "unexpected"])
def test_duplicate_or_unexpected_rows_cannot_replace_missing_identity(
    real_campaign, tmp_path, replacement
):
    """A full row count cannot hide a duplicate or a foreign job identity."""
    cfg, original, scenarios = real_campaign
    campaign = tmp_path / "fixture"
    shutil.copytree(original, campaign)
    episodes = campaign / "runs" / ARM / "episodes.jsonl"
    rows = episodes.read_bytes().splitlines(keepends=True)
    substitute = rows[0]
    if replacement == "unexpected":
        row = json.loads(rows[-1])
        row["seed"] = 1004
        substitute = (json.dumps(row) + "\n").encode()
    episodes.write_bytes(b"".join(rows[:-1] + [substitute]))
    with pytest.raises(ResumeMismatchError, match=f"{replacement}.*identity"):
        _plan(cfg, campaign, scenarios)


def test_per_scenario_seed_lists_use_map_job_expansion():
    """Map jobs honor distinct seed lists and do not invent repeat jobs ignored by execution."""
    scenarios = [
        {"name": "a", "map_file": "a.svg", "seeds": [1001, 1002], "repeats": 4},
        {"name": "b", "map_file": "b.svg", "seeds": [1003]},
    ]
    assert _expected_jobs(scenarios) == 3


def test_precomputed_count_cannot_hide_missing_jobs(tmp_path):
    """An obsolete supplied denominator cannot authorize skipping real planned jobs."""
    with pytest.raises(ValueError, match="expected_jobs disagrees"):
        build_resume_plan(
            tmp_path,
            planners=[{"key": "goal"}],
            kinematics_matrix=["differential_drive"],
            scenarios=[{"name": "a", "map_file": "a.svg", "seeds": [1001, 1002, 1003]}],
            expected_jobs=1,
        )
