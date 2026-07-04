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
    assert metadata["benchmark_evidence"] is False
    assert metadata["observed_episode_count"] == 3
    assert metadata["scenario_ids"] == ["classic_bottleneck_low"]
    assert metadata["seeds"] == [111]
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
    assert len(calls) == 3
    assert calls[0]["scenario_ids"] == [
        "classic_bottleneck_low",
        "classic_head_on_corridor_low",
        "narrow_passage",
    ]
    assert calls[0]["seeds"] == [[111, 112, 113]] * 3
    assert calls[0]["safety_wrapper"]["enabled"] is False
    assert calls[1]["safety_wrapper"]["enabled"] is True
    assert calls[2]["cbf_safety_filter"]["enabled"] is True


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
