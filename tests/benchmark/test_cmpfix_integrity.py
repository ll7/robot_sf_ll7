"""External-review counterexamples; pure validation, never environment execution."""

import json

import pytest
import yaml

from robot_sf.benchmark.artifact_publication import _preflight_check_checksums
from robot_sf.benchmark.camera_ready._resume_plan import ResumeMismatchError, build_resume_plan
from robot_sf.benchmark.camera_ready._run_state import validate_campaign_integrity
from robot_sf.benchmark.map_runner.map_runner_identity import _scenario_identity_payload
from robot_sf.benchmark.utils import _config_hash


def _row(scenario, seed, *, algo="social_force", config=None):
    params = _scenario_identity_payload(
        scenario, algo=algo, algo_config=config or {}, horizon=600, dt=0.1, record_forces=False
    )
    params["seed"] = seed
    digest = _config_hash(params)
    return {
        "scenario_id": scenario["name"],
        "seed": seed,
        "algo": algo,
        "scenario_params": params,
        "config_hash": digest,
        "git_hash": "commit-a",
        "algorithm_metadata": {"algorithm": algo, "config_hash": _config_hash(config or {})},
        "result_provenance": {
            "scenario_id": scenario["name"],
            "seed": seed,
            "config_hash": digest,
            "repo_commit": "commit-a",
        },
    }


def _integrity(tmp_path, rows, scenarios, planner=None):
    path = tmp_path / "runs/social_force__differential_drive/episodes.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return validate_campaign_integrity(
        [
            {
                "status": "ok",
                "planner": planner or {"key": "social_force", "algo": "social_force"},
                "episodes_path": str(path),
                "summary": {"episodes_total": len(rows)},
            }
        ],
        scenarios=scenarios,
        resolved_seeds=[1001, 1002],
        campaign_root=tmp_path,
        campaign_manifest={"git": {"commit": "commit-a"}},
    )


def test_cmpfix_equal_count_wrong_episode_set(tmp_path):
    scenario = {"name": "S", "seeds": [1001, 1002]}
    verdict = _integrity(tmp_path, [_row(scenario, 1001), _row(scenario, 1003)], [scenario])
    assert verdict["benchmark_success_allowed"] is False
    blocker = next(b for b in verdict["blockers"] if b["invariant"] == "episode_identity_mismatch")
    assert blocker["details"]["missing_identities"] == [["S", 1002]]
    assert blocker["details"]["unexpected_identities"] == [["S", 1003]]


@pytest.mark.parametrize("corruption", ["algorithm", "config", "recorded_hash"])
def test_cmpfix_wrong_runtime_identity(tmp_path, corruption):
    scenario = {"name": "S", "seeds": [1001]}
    row = _row(
        scenario,
        1001,
        algo="goal" if corruption == "algorithm" else "social_force",
        config={"wrong": True} if corruption == "config" else {},
    )
    if corruption == "recorded_hash":
        row["config_hash"] = row["result_provenance"]["config_hash"] = "wrong-config"
    verdict = _integrity(tmp_path, [row], [scenario])
    assert verdict["benchmark_success_allowed"] is False
    assert any(b["invariant"] == "runtime_identity_mismatch" for b in verdict["blockers"])


def test_cmpfix_resume_refuses_wrong_runtime_identity(tmp_path):
    scenario = {"name": "S", "seeds": [1001], "repeats": 1}
    path = tmp_path / "social_force__differential_drive/episodes.jsonl"
    path.parent.mkdir()
    path.write_text(json.dumps(_row(scenario, 1001, algo="goal", config={"wrong": True})) + "\n")
    (path.parent / "summary.json").write_text('{"written": 1}')
    with pytest.raises(ResumeMismatchError, match="runtime identity"):
        build_resume_plan(
            tmp_path,
            planners=[{"key": "social_force", "algo": "social_force"}],
            kinematics_matrix=["differential_drive"],
            scenarios=[scenario],
        )


def test_cmpfix_scenario_adaptive_configs_remain_valid(tmp_path):
    scenarios = [{"name": "S", "seeds": [1001]}, {"name": "T", "seeds": [1001]}]
    path = tmp_path / "adaptive.yaml"
    path.write_text(
        yaml.safe_dump({"params": {"speed": 1}, "scenario_overrides": {"T": {"speed": 2}}})
    )
    rows = [_row(s, 1001, config={"speed": i}) for i, s in enumerate(scenarios, 1)]
    verdict = _integrity(
        tmp_path,
        rows,
        scenarios,
        {"key": "social_force", "algo": "social_force", "algo_config_path": str(path)},
    )
    assert verdict["blockers"] == []


def test_cmpfix_checksum_root_alias_is_refused(tmp_path):
    root_file = tmp_path / "reports/campaign_table.csv"
    payload_file = tmp_path / "payload/reports/campaign_table.csv"
    for path, value in [(root_file, "success,0.9\n"), (payload_file, "success,0.1\n")]:
        path.parent.mkdir(parents=True)
        path.write_text(value)
    import hashlib

    checksums = tmp_path / "checksums.sha256"
    checksums.write_text(
        f"{hashlib.sha256(root_file.read_bytes()).hexdigest()}  reports/campaign_table.csv\n"
    )
    violations = []
    _preflight_check_checksums(
        tmp_path,
        {
            "files": [
                {
                    "path": "reports/campaign_table.csv",
                    "sha256": hashlib.sha256(payload_file.read_bytes()).hexdigest(),
                }
            ]
        },
        checksums_path=checksums,
        violations=violations,
    )
    assert any("canonical payload path" in v for v in violations)


def test_cmpfix_skip_complete_rechecks_runtime_identity(tmp_path):
    from types import SimpleNamespace

    from robot_sf.benchmark.camera_ready.campaign import _resolve_campaign_planner_batch_result

    scenario = {"name": "S", "seeds": [1001]}
    path = tmp_path / "episodes.jsonl"
    path.write_text(json.dumps(_row(scenario, 1001, algo="goal")) + "\n")
    run = SimpleNamespace(
        episodes_path=path,
        planner_dir=tmp_path,
        kinematics="differential_drive",
        scoped_scenarios=[scenario],
    )
    verdict = SimpleNamespace(
        verdict="skip-complete",
        episodes_path=path,
        prior_summary={"written": 1},
        episodes_found=1,
        expected_total=1,
    )
    planner = SimpleNamespace(key="social_force", algo="social_force", algo_config_path=None)
    with pytest.raises(ResumeMismatchError, match="runtime identity"):
        _resolve_campaign_planner_batch_result(
            None, planner=planner, run=run, resume_verdict=verdict
        )


@pytest.mark.parametrize(
    ("planner", "scenario", "expected"),
    [
        (
            {
                "algo": "guarded_ppo",
                "algo_config_path": "configs/algos/guarded_ppo_camera_ready_cpu.yaml",
            },
            {"name": "dev", "seeds": [1001]},
            ("guarded_ppo", "44a39b76607347cc"),
        ),
        (
            {
                "algo": "hybrid_rule_local_planner",
                "algo_config_path": "configs/policy_search/candidates/scenario_adaptive_hybrid_orca_v2_collision_guard_s30_h600_release.yaml",
            },
            {"name": "francis2023_leave_group", "seeds": [1001]},
            ("orca", "384a28dd063d0e76"),
        ),
    ],
    ids=["flat-config", "nested-orca-handoff"],
)
def test_cmpfix_runtime_config_identity_is_independent_of_cwd(
    tmp_path, monkeypatch, planner, scenario, expected
):
    """Pinned published config digests survive relocation, including candidate base paths."""
    from robot_sf.benchmark.camera_ready._runtime_identity import expected_runtime_identity

    monkeypatch.chdir(tmp_path)
    assert expected_runtime_identity(planner, scenario, 1001) == expected
