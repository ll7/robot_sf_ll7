"""Selected campaign inputs remain authoritative over fully re-signed producer rows."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from robot_sf.benchmark.camera_ready._config import (
    _apply_fixed_campaign_horizon,
    _scenario_with_kinematics,
)
from robot_sf.benchmark.camera_ready._run_state import validate_campaign_integrity
from robot_sf.benchmark.map_runner.map_runner_episode import _finalize_result_provenance_block
from robot_sf.benchmark.map_runner.map_runner_identity import _compute_map_episode_id
from robot_sf.benchmark.observation_noise import (
    normalize_observation_noise_spec,
    observation_noise_hash,
)
from robot_sf.benchmark.tracking_precision_contract import (
    normalize_tracking_precision_spec,
    tracking_precision_hash,
)
from robot_sf.benchmark.utils import _config_hash, _git_hash_fallback
from tests.benchmark.campaign_identity_support import fixture_campaign, producer_params


def _signed_record(scenario):
    """Use the real producer finalizer, including all identity copies present at each source."""
    params = producer_params(scenario, 1001, algo="social_force", horizon=600, dt=0.1)
    commit = _git_hash_fallback()
    noise = normalize_observation_noise_spec(None)
    precision = normalize_tracking_precision_spec(None)
    row = {
        "scenario_id": "selected-input",
        "seed": 1001,
        "episode_id": _compute_map_episode_id(params, 1001),
        "algo": "social_force",
        "scenario_params": params,
        "config_hash": _config_hash(params),
        "git_hash": commit,
        "observation_noise": noise,
        "observation_noise_hash": observation_noise_hash(noise),
        "tracking_precision": precision,
        "tracking_precision_hash": tracking_precision_hash(precision),
        "provenance": {"config_hash": _config_hash(params), "git_hash": commit},
    }
    _finalize_result_provenance_block(
        row,
        scenario_id="selected-input",
        seed=1001,
        scenario_params=params,
        config=SimpleNamespace(sim_config=SimpleNamespace(time_per_step_in_secs=0.1)),
        horizon_val=600,
        record_forces=False,
        active_observation_mode=params["observation_mode"],
        active_observation_level=params["observation_level"],
        noise_spec=noise,
        tracking_precision_spec=precision,
    )
    return row


@pytest.mark.parametrize("substitution", ["map", "budget"])
def test_resigned_producer_row_cannot_replace_selected_input(tmp_path, substitution):
    """A self-consistent writer row must still match the independently selected scenario."""
    selected = _scenario_with_kinematics(
        {
            "name": "selected-input",
            "map_file": "maps/selected.svg",
            "seeds": [1001],
            "simulation_config": {"max_episode_steps": 600},
        },
        kinematics="differential_drive",
        holonomic_command_mode="vx_vy",
    )
    cfg = fixture_campaign([{"key": "social_force", "algo": "social_force"}], horizon=600, dt=0.1)
    selected = _apply_fixed_campaign_horizon(
        [selected],
        horizon=cfg.horizon,
        horizon_policy=cfg.horizon_policy,
        protocol_version=cfg.protocol_version,
    )[0]
    path = tmp_path / "episodes.jsonl"

    def admit(row):
        path.write_text(json.dumps(row) + "\n")
        return validate_campaign_integrity(
            [
                {
                    "status": "ok",
                    "planner": {
                        "key": "social_force",
                        "algo": "social_force",
                        "kinematics": "differential_drive",
                    },
                    "episodes_path": str(path),
                    "summary": {"episodes_total": 1},
                }
            ],
            scenarios=[selected],
            resolved_seeds=[1001],
            campaign_root=tmp_path,
            campaign_manifest={"git": {"commit": _git_hash_fallback()}},
            cfg=cfg,
        )["benchmark_success_allowed"]

    assert admit(_signed_record(selected)) is True
    substituted = deepcopy(selected)
    if substitution == "map":
        substituted["map_file"] = "maps/substituted.svg"
    else:
        substituted["simulation_config"]["max_episode_steps"] = 60
    assert admit(_signed_record(substituted)) is False
