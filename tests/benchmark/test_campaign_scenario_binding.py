"""Signed scenario substitutions must fail campaign and resume admission."""

import json
from copy import deepcopy

import pytest

from robot_sf.benchmark.camera_ready._resume_plan import (
    ResumeMismatchError,
    _validate_resume_runtime_identities,
)
from robot_sf.benchmark.camera_ready._run_state import validate_campaign_integrity
from robot_sf.benchmark.map_runner.map_runner_identity import (
    _scenario_identity_payload,
    _scenario_with_episode_seed_defaults,
)
from robot_sf.benchmark.utils import _config_hash

SCENARIO = {
    "name": "scenario-binding",
    "map_file": "maps/fixture.svg",
    "seeds": [1001],
    "simulation_config": {"max_episode_steps": 600, "peds_have_robot_repulsion": True},
}
PLANNER = {"key": "social_force", "algo": "social_force"}


def _row(scenario):
    params = _scenario_identity_payload(
        _scenario_with_episode_seed_defaults(deepcopy(scenario), seed=1001),
        algo="social_force",
        algo_config={},
        horizon=600,
        dt=0.1,
        record_forces=False,
    )
    return {
        "scenario_id": SCENARIO["name"],
        "seed": 1001,
        "algo": "social_force",
        "scenario_params": params,
        "config_hash": _config_hash(params),
        "git_hash": "commit-a",
        "result_provenance": {
            "scenario_id": SCENARIO["name"],
            "seed": 1001,
            "config_hash": _config_hash(params),
            "repo_commit": "commit-a",
        },
    }


def _write(tmp_path, row):
    path = tmp_path / "episodes.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return path


def _integrity(tmp_path, row):
    path = _write(tmp_path, row)
    return validate_campaign_integrity(
        [
            {
                "status": "ok",
                "planner": PLANNER,
                "episodes_path": str(path),
                "summary": {"episodes_total": 1},
            }
        ],
        scenarios=[SCENARIO],
        resolved_seeds=[1001],
        campaign_root=tmp_path,
        campaign_manifest={"git": {"commit": "commit-a"}},
    )


@pytest.mark.parametrize(
    "field", ["map_file", "max_episode_steps", "peds_have_robot_repulsion", "route_spawn_seed"]
)
def test_signed_scenario_substitution_blocks_integrity_and_resume(tmp_path, field):
    """Re-signing a wrong scenario cannot authorize its release slot or reuse."""
    clean = _row(SCENARIO)
    assert _integrity(tmp_path, clean)["benchmark_success_allowed"] is True
    _validate_resume_runtime_identities(_write(tmp_path, clean), PLANNER, [SCENARIO])
    altered = deepcopy(SCENARIO)
    if field == "map_file":
        altered[field] = "maps/other.svg"
    else:
        altered["simulation_config"][field] = {
            "max_episode_steps": 60,
            "peds_have_robot_repulsion": False,
            "route_spawn_seed": 1002,
        }[field]
    mutant = _row(altered)
    result = _integrity(tmp_path, mutant)
    assert result["benchmark_success_allowed"] is False
    assert any(b["invariant"] == "runtime_identity_mismatch" for b in result["blockers"])
    with pytest.raises(ResumeMismatchError, match="runtime identity mismatch"):
        _validate_resume_runtime_identities(_write(tmp_path, mutant), PLANNER, [SCENARIO])


def test_missing_saved_seed_default_blocks_integrity(tmp_path):
    """Validation must not manufacture an omitted saved stochastic input."""
    row = _row(SCENARIO)
    del row["scenario_params"]["simulation_config"]["route_spawn_seed"]
    row["config_hash"] = row["result_provenance"]["config_hash"] = _config_hash(
        row["scenario_params"]
    )
    assert _integrity(tmp_path, row)["benchmark_success_allowed"] is False


def test_legacy_horizon_annotation_preserves_authored_scenario_identity(tmp_path):
    """Historical accounting restores the authored budget before scenario binding."""
    row = _row(SCENARIO)
    row["scenario_params"]["metadata"] = {
        "scenario_horizon": {"policy": "legacy_runner_cap", "authored_max_episode_steps": 600},
        "campaign_horizon": {"runner_horizon": 60},
    }
    row["scenario_params"]["simulation_config"]["max_episode_steps"] = 60
    row["config_hash"] = row["result_provenance"]["config_hash"] = _config_hash(
        row["scenario_params"]
    )
    # Legacy accounting canonicalizes empty metadata on both sides.
    scenario = {**SCENARIO, "metadata": {}}
    path = _write(tmp_path, row)
    _validate_resume_runtime_identities(path, PLANNER, [scenario])
