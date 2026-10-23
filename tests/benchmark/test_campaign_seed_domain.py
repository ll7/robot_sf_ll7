"""Aggregate admission must not coerce malformed saved seeds into valid slots."""

import json
from copy import deepcopy

import pytest

from robot_sf.benchmark.camera_ready._run_state import validate_campaign_integrity
from robot_sf.benchmark.map_runner.map_runner_identity import (
    _scenario_identity_payload,
    _scenario_with_episode_seed_defaults,
)
from robot_sf.benchmark.utils import _config_hash


@pytest.mark.parametrize("bad_seed", ["1001", 1001.0, 1001.7])
@pytest.mark.parametrize("location", ["row", "provenance"])
def test_saved_seed_type_cannot_alias_an_expected_episode(tmp_path, bad_seed, location):
    """A retyped/truncated seed cannot pass exact coverage or provenance identity."""
    scenario = {"name": "seed-domain", "simulation_config": {}, "seeds": [1001]}
    params = _scenario_identity_payload(
        _scenario_with_episode_seed_defaults(scenario, seed=1001),
        algo="goal",
        algo_config={},
        horizon=600,
        dt=0.1,
        record_forces=False,
    )
    clean = {
        "scenario_id": "seed-domain",
        "seed": 1001,
        "algo": "goal",
        "scenario_params": params,
        "config_hash": _config_hash(params),
        "git_hash": "commit-a",
        "result_provenance": {
            "scenario_id": "seed-domain",
            "seed": 1001,
            "config_hash": _config_hash(params),
            "repo_commit": "commit-a",
        },
    }
    path = tmp_path / "episodes.jsonl"

    def check(row):
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        return validate_campaign_integrity(
            [
                {
                    "status": "ok",
                    "planner": {"key": "goal", "algo": "goal"},
                    "episodes_path": str(path),
                    "summary": {"episodes_total": 1},
                }
            ],
            scenarios=[scenario],
            resolved_seeds=[1001],
            campaign_root=tmp_path,
            campaign_manifest={"git": {"commit": "commit-a"}},
        )

    assert check(clean)["benchmark_success_allowed"] is True
    mutant = deepcopy(clean)
    if location == "row":
        mutant["seed"] = mutant["result_provenance"]["seed"] = bad_seed
    else:
        mutant["result_provenance"]["seed"] = bad_seed
    report = check(mutant)
    assert report["benchmark_success_allowed"] is False
    expected = (
        "episode_identity_mismatch" if location == "row" else "row_provenance_identity_mismatch"
    )
    assert any(b["invariant"] == expected for b in report["blockers"])
