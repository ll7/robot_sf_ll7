"""Tests for the issue #4367 campaign runner for the #4205 loop."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts/benchmark/run_issue_4205_codesign_loop_campaign.py"
RESEARCH_CONFIG = (
    REPO_ROOT / "configs/research/issue_4205_static_constriction_codesign_loop_v1.yaml"
)
BENCHMARK_CONFIG = (
    REPO_ROOT / "configs/benchmarks/issue_4205_static_constriction_codesign_loop_v1.yaml"
)

_SPEC = importlib.util.spec_from_file_location("_issue_4367_campaign", SCRIPT)
assert _SPEC is not None
assert _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _hydration_manifest(tmp_path: Path) -> Path:
    """Create a valid tiny hydration manifest for fail-closed tests."""
    checkpoint = tmp_path / "frozen_ppo.zip"
    checkpoint.write_bytes(b"issue-4367-hydrated-checkpoint")
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    config = yaml.safe_load(RESEARCH_CONFIG.read_text(encoding="utf-8"))
    lineage = config["frozen_ppo_lineage"]
    manifest = {
        "schema_version": "robot_sf.issue_4205.frozen_ppo_checkpoint_hydration.v1",
        "issue": 4205,
        "loop_id": config["loop_id"],
        "model_id": lineage["model_id"],
        "algo_config": lineage["algo_config"],
        "algo_config_sha256": lineage["algo_config_sha256"],
        "checkpoint_path": checkpoint.name,
        "checkpoint_sha256": checkpoint_sha256,
        "arms": [
            {
                "key": "ppo_frozen",
                "model_id": lineage["model_id"],
                "algo_config": lineage["algo_config"],
                "checkpoint_sha256": checkpoint_sha256,
                "safety_wrapper": {"enabled": False, "arm_key": "wrapper_off"},
                "cbf_safety_filter": {"enabled": False, "arm_key": "cbf_off"},
            },
            {
                "key": "ppo_frozen_wrapper_on",
                "model_id": lineage["model_id"],
                "algo_config": lineage["algo_config"],
                "checkpoint_sha256": checkpoint_sha256,
                "safety_wrapper": {"enabled": True, "arm_key": "wrapper_on"},
                "cbf_safety_filter": {"enabled": False, "arm_key": "cbf_off"},
            },
            {
                "key": "ppo_frozen_cbf_on",
                "model_id": lineage["model_id"],
                "algo_config": lineage["algo_config"],
                "checkpoint_sha256": checkpoint_sha256,
                "safety_wrapper": {"enabled": False, "arm_key": "wrapper_off"},
                "cbf_safety_filter": {"enabled": True, "arm_key": "cbf_collision_cone_on"},
            },
        ],
    }
    manifest_path = tmp_path / "hydration_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def _fake_run_map_batch(
    scenarios,
    out_path,
    schema_path,
    *,
    algo,
    algo_config_path,
    safety_wrapper,
    cbf_safety_filter,
    **kwargs,
):
    """Write deterministic fake episode rows without rolling episodes."""
    assert kwargs["benchmark_profile"] == "experimental"
    assert Path(kwargs["scenario_path"]) == (
        REPO_ROOT / "configs/scenarios/sets/issue_2588_static_deadlock_controlled_trace.yaml"
    )
    del schema_path, algo_config_path, kwargs
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for scenario in scenarios:
        scenario_id = scenario["name"]
        for seed in scenario["seeds"]:
            records.append(
                {
                    "episode_id": f"{algo}-{scenario_id}-{seed}",
                    "scenario_id": scenario_id,
                    "seed": seed,
                    "success": True,
                    "collision": False,
                    "metrics": {
                        "snqi": 0.8,
                        "deadlock_count": 0,
                        "near_miss_count": 0,
                        "low_progress_window": "absent",
                        "recenter_activation_count": 0,
                        "distance_to_goal_delta": 1.0,
                        "local_minimum_indicator": False,
                        "wrapper_intervention_rate": 0.1 if safety_wrapper["enabled"] else 0.0,
                        "cbf_status_counts": {"active": 1} if cbf_safety_filter["enabled"] else {},
                    },
                }
            )
    out_path.write_text("\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n")
    return {"written": len(records), "failures": [], "out_path": str(out_path)}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_smoke_campaign_requires_hydration_and_writes_contract_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Smoke mode runs one scenario x one seed per arm and writes required tables."""
    monkeypatch.setattr(_MODULE, "run_map_batch", _fake_run_map_batch)
    output_root = tmp_path / "campaign"
    exit_code = _MODULE.main(
        [
            "--config",
            str(BENCHMARK_CONFIG),
            "--hydration-manifest",
            str(_hydration_manifest(tmp_path)),
            "--output-root",
            str(output_root),
            "--smoke",
            "--json",
        ]
    )
    assert exit_code == 0
    metadata = json.loads((output_root / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["mode"] == "smoke"
    assert metadata["benchmark_profile"] == "experimental"
    assert metadata["benchmark_evidence"] is False
    assert metadata["observed_episode_count"] == 3
    assert metadata["scenario_ids"] == ["classic_bottleneck_low"]
    assert metadata["seeds"] == [111]
    assert metadata["arm_resolution_preflight"]["status"] == "passed"
    assert metadata["arm_resolution_preflight"]["arms"]["ppo_frozen_cbf_on"]["status"] == "ok"
    preflight_report = json.loads(
        (output_root / "preflight_report.json").read_text(encoding="utf-8")
    )
    assert preflight_report == metadata["arm_resolution_preflight"]
    assert metadata["preflight_report"].endswith("preflight_report.json")
    assert metadata["arm_execution_order"] == [
        "ppo_frozen_cbf_on",
        "ppo_frozen_wrapper_on",
        "ppo_frozen",
    ]
    assert metadata["phase_0_smoke"] is None
    live_status = json.loads((output_root / "live_arm_status.json").read_text(encoding="utf-8"))
    assert live_status == metadata["live_arm_status"]
    assert live_status["phase"] == "smoke"
    assert list(live_status["arms"]) == [
        "ppo_frozen_cbf_on",
        "ppo_frozen_wrapper_on",
        "ppo_frozen",
    ]
    assert {arm["status"] for arm in live_status["arms"].values()} == {"completed"}
    status_events = [
        json.loads(line)
        for line in (output_root / "arm_status.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [(event["arm"], event["state"]) for event in status_events] == [
        ("ppo_frozen_cbf_on", "running"),
        ("ppo_frozen_cbf_on", "completed"),
        ("ppo_frozen_wrapper_on", "running"),
        ("ppo_frozen_wrapper_on", "completed"),
        ("ppo_frozen", "running"),
        ("ppo_frozen", "completed"),
    ]
    assert {event["phase"] for event in status_events} == {"smoke"}
    per_arm = _read_csv(output_root / "per_arm_metric_table.csv")
    assert [row["arm_key"] for row in per_arm] == [
        "ppo_frozen",
        "ppo_frozen_wrapper_on",
        "ppo_frozen_cbf_on",
    ]
    assert {row["episode_count"] for row in per_arm} == {"1"}
    per_episode = _read_csv(output_root / "per_episode_rows.csv")
    assert len(per_episode) == 3
    assert (output_root / "failure_mode_counts.csv").exists()
    checksums = (output_root / "SHA256SUMS").read_text(encoding="utf-8")
    assert "run_metadata.json" in checksums
    assert "per_arm_metric_table.csv" in checksums


def test_full_campaign_plan_uses_all_preregistered_scenarios_seeds_and_arms(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Full mode consumes the 3 x 3 x 3 pre-registered grid."""
    calls = []

    def recording_runner(*args, **kwargs):
        scenarios = args[0]
        calls.append(
            {
                "scenario_ids": [scenario["name"] for scenario in scenarios],
                "seeds": [scenario["seeds"] for scenario in scenarios],
                "benchmark_profile": kwargs["benchmark_profile"],
                "scenario_path": Path(kwargs["scenario_path"]),
                "safety_wrapper": kwargs["safety_wrapper"],
                "cbf_safety_filter": kwargs["cbf_safety_filter"],
            }
        )
        return _fake_run_map_batch(*args, **kwargs)

    monkeypatch.setattr(_MODULE, "run_map_batch", recording_runner)
    output_root = tmp_path / "campaign"
    result = _MODULE.run_campaign(
        _MODULE.parse_args(
            [
                "--config",
                str(BENCHMARK_CONFIG),
                "--hydration-manifest",
                str(_hydration_manifest(tmp_path)),
                "--output-root",
                str(output_root),
            ]
        )
    )
    assert result["metadata"]["benchmark_evidence"] is True
    assert result["metadata"]["observed_episode_count"] == 27
    assert result["metadata"]["arm_execution_order"] == [
        "ppo_frozen_cbf_on",
        "ppo_frozen_wrapper_on",
        "ppo_frozen",
    ]
    assert result["metadata"]["phase_0_smoke"]["status"] == "passed"
    assert result["metadata"]["phase_0_smoke"]["benchmark_evidence"] is False
    assert result["metadata"]["phase_0_smoke"]["observed_episode_count"] == 3
    assert len(calls) == 6
    assert [call["scenario_ids"] for call in calls[:3]] == [["classic_bottleneck_low"]] * 3
    assert [call["seeds"] for call in calls[:3]] == [[[111]]] * 3
    assert calls[3]["scenario_ids"] == [
        "classic_bottleneck_low",
        "classic_head_on_corridor_low",
        "narrow_passage",
    ]
    assert calls[3]["seeds"] == [[111, 112, 113]] * 3
    assert {call["benchmark_profile"] for call in calls} == {"experimental"}
    assert {call["scenario_path"] for call in calls} == {
        REPO_ROOT / "configs/scenarios/sets/issue_2588_static_deadlock_controlled_trace.yaml"
    }
    assert calls[0]["cbf_safety_filter"]["enabled"] is True
    assert calls[1]["safety_wrapper"]["enabled"] is True
    assert calls[2]["safety_wrapper"]["enabled"] is False
    assert (output_root / "phase_0_smoke" / "live_arm_status.json").exists()
    live_status = json.loads((output_root / "live_arm_status.json").read_text(encoding="utf-8"))
    assert live_status["phase"] == "full"
    assert {arm["status"] for arm in live_status["arms"].values()} == {"completed"}


def test_arm_resolution_preflight_fails_closed_on_missing_hydrated_arm() -> None:
    """Submit contract requires every pre-registered arm resolved before phase-0."""
    arm_runtime = {arm_key: {} for arm_key in _MODULE.EXPECTED_ARM_KEYS}
    hydration = {"arms": list(_MODULE.EXPECTED_ARM_KEYS[:-1])}

    with pytest.raises(_MODULE.ContractError, match="missing_hydration"):
        _MODULE._preflight_campaign_arms(arm_runtime=arm_runtime, hydration=hydration)


def test_arm_resolution_preflight_fails_closed_on_unknown_algorithm() -> None:
    """Submit contract rejects unresolved algorithms before phase-0."""
    arm_runtime = {
        arm_key: {
            "algo": "ppo",
            "algo_config": "configs/baselines/ppo_15m_grid_socnav.yaml",
        }
        for arm_key in _MODULE.EXPECTED_ARM_KEYS
    }
    arm_runtime["ppo_frozen_cbf_on"]["algo"] = "missing_planner"
    hydration = {"arms": list(_MODULE.EXPECTED_ARM_KEYS), "checkpoint_sha256": "abc123"}

    with pytest.raises(_MODULE.ContractError, match="ppo_frozen_cbf_on.*unknown algorithm"):
        _MODULE._preflight_campaign_arms(arm_runtime=arm_runtime, hydration=hydration)


def test_parallel_campaign_uses_spawn_multiprocessing_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Parallel co-design runs pass a spawn context into the map runner."""
    contexts = []

    def recording_runner(*args, **kwargs):
        contexts.append(kwargs["multiprocessing_context"])
        return _fake_run_map_batch(*args, **kwargs)

    monkeypatch.setattr(_MODULE, "run_map_batch", recording_runner)
    _MODULE.run_campaign(
        _MODULE.parse_args(
            [
                "--config",
                str(BENCHMARK_CONFIG),
                "--hydration-manifest",
                str(_hydration_manifest(tmp_path)),
                "--output-root",
                str(tmp_path / "campaign"),
                "--smoke",
                "--workers",
                "2",
            ]
        )
    )

    assert len(contexts) == 3
    assert {context.get_start_method() for context in contexts} == {"spawn"}


def test_missing_hydration_manifest_fails_before_outputs(tmp_path: Path) -> None:
    """Campaign execution is fail-closed without the private hydration manifest."""
    output_root = tmp_path / "campaign"
    exit_code = _MODULE.main(
        [
            "--config",
            str(BENCHMARK_CONFIG),
            "--hydration-manifest",
            str(tmp_path / "missing.json"),
            "--output-root",
            str(output_root),
            "--smoke",
        ]
    )
    assert exit_code == 2
    assert not output_root.exists()


def test_missing_hydrated_checkpoint_fails_before_outputs(tmp_path: Path) -> None:
    """Submit contract rejects a manifest pointing at a missing checkpoint."""
    manifest_path = _hydration_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["checkpoint_path"] = "missing_checkpoint.zip"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output_root = tmp_path / "campaign"

    exit_code = _MODULE.main(
        [
            "--config",
            str(BENCHMARK_CONFIG),
            "--hydration-manifest",
            str(manifest_path),
            "--output-root",
            str(output_root),
            "--smoke",
        ]
    )

    assert exit_code == 2
    assert not output_root.exists()


def test_aggregate_per_arm_tolerates_dict_valued_trace_fields() -> None:
    """Slurm job 13292 regression: trace fields can be dict-valued; set-membership
    against {"", None} raised TypeError (unhashable). Tuple membership must work."""
    rows = [
        {
            "arm_key": "ppo_frozen",
            "success": True,
            "collision": False,
            "near_miss": False,
            "snqi": -0.1,
            "deadlock_count": 0,
            "low_progress_window": {"window_steps": 5, "threshold": 0.1},
            "local_minimum_indicator": None,
            "row_status": "completed",
        }
    ]
    out = _MODULE._aggregate_per_arm(rows)
    assert out[0]["arm_key"] == "ppo_frozen"
    # the dict-valued field counts as a present trace field
    trace_keys = [k for k in out[0] if "trace" in k]
    assert trace_keys and any(out[0][k] in (1, True) for k in trace_keys)
