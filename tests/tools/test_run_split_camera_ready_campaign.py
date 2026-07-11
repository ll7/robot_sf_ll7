"""Tests for per-arm camera-ready execution packets and native aggregation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from scripts.tools import run_split_camera_ready_campaign as runner


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _split_fixture(
    tmp_path: Path,
    planners: tuple[str, ...] = ("goal", "orca"),
    *,
    split_per_planner: bool = True,
) -> Path:
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    parent = tmp_path / "parent.yaml"
    parent.write_text("name: parent\n", encoding="utf-8")
    children = []
    arm_planners = ((planner,) for planner in planners) if split_per_planner else (planners,)
    for index, child_planners in enumerate(arm_planners):
        arm_label = "__".join(child_planners)
        path = split_dir / f"parent__arm_{arm_label}.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "name": f"parent__arm_{arm_label}",
                    "planners": [{"key": planner, "enabled": True} for planner in child_planners],
                    "seed_policy": {"seed_set": "paper_eval_s20"},
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        children.append(
            {
                "filename": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "planner_keys": list(child_planners),
                "arm_index": index,
            }
        )
    manifest = split_dir / "split_manifest.json"
    _write_json(
        manifest,
        {
            "source_config": str(parent),
            "source_sha256": hashlib.sha256(parent.read_bytes()).hexdigest(),
            "children": children,
        },
    )
    return manifest


def _write_campaign(
    root: Path,
    *,
    campaign_id: str,
    config_hash: str,
    planner: str,
    execution_mode: str = "native",
    readiness_status: str | None = None,
    status: str = "ok",
    scenario_hash: str = "matrix-a",
) -> None:
    row = {
        "planner_key": planner,
        "status": status,
        "execution_mode": execution_mode,
        "readiness_status": readiness_status or execution_mode,
        "availability_status": "available" if status == "ok" else "failed",
        "benchmark_success": status == "ok",
        "episodes": 20,
        "success_mean": "0.5000",
        "collisions_mean": "0.1000",
        "snqi_mean": "0.2000",
    }
    _write_json(
        root / "campaign_manifest.json",
        {
            "campaign_id": campaign_id,
            "config_hash": config_hash,
            "seed_policy": {
                "mode": "seed-set",
                "seed_set": "paper_eval_s20",
                "seeds": [],
                "resolved_seeds": [111, 112],
                "seed_sets_path": "configs/benchmarks/seed_sets_v1.yaml",
            },
        },
    )
    _write_json(
        root / "reports/campaign_summary.json",
        {
            "campaign": {
                "campaign_id": campaign_id,
                "scenario_matrix": "configs/scenarios/frozen.yaml",
                "scenario_matrix_hash": scenario_hash,
                "git_hash": "abc123",
                "paper_interpretation_profile": "frozen",
                "paper_profile_version": "v1",
                "observation_noise_hash": "noise-a",
                "snqi_weights_sha256": "weights-a",
                "snqi_baseline_sha256": "baseline-a",
                "kinematics_matrix": ["differential_drive"],
            },
            "planner_rows": [row],
        },
    )


def _packet(tmp_path: Path) -> tuple[Path, dict]:
    packet_path = tmp_path / "packet.json"
    packet = runner.build_execution_packet(
        _split_fixture(tmp_path),
        output_root=tmp_path / "campaigns",
        campaign_prefix="issue5273_s20",
    )
    _write_json(packet_path, packet)
    return packet_path, packet


def test_plan_real_s20_manifest_names_all_child_commands(tmp_path: Path) -> None:
    """The committed S20 split produces one exact command per child."""
    manifest = Path(
        "configs/benchmarks/splits/"
        "paper_experiment_matrix_v1_scenario_horizons_h500_s20/split_manifest.json"
    )
    packet = runner.build_execution_packet(
        manifest,
        output_root=tmp_path / "campaigns",
        campaign_prefix="issue5273_s20",
    )

    assert packet["schema_version"] == runner.PACKET_SCHEMA
    assert packet["status"] == "planned_not_executed"
    assert len(packet["arms"]) == 9
    assert {arm["planner_keys"][0] for arm in packet["arms"]} >= {"goal", "orca", "ppo"}
    assert all("run_camera_ready_benchmark.py" in arm["command"] for arm in packet["arms"])
    assert all("--skip-publication-bundle" in arm["command"] for arm in packet["arms"])


def test_plan_rejects_child_digest_drift(tmp_path: Path) -> None:
    """Execution packets fail closed when a declared child changes."""
    manifest = _split_fixture(tmp_path)
    child = next(manifest.parent.glob("*.yaml"))
    child.write_text("changed: true\n", encoding="utf-8")

    with pytest.raises(ValueError, match="digest mismatch"):
        runner.build_execution_packet(
            manifest,
            output_root=tmp_path / "campaigns",
            campaign_prefix="issue5273",
        )


def test_aggregate_includes_native_and_explicitly_excludes_adapter(tmp_path: Path) -> None:
    """Compatible native rows aggregate while adapter rows remain visible exclusions."""
    packet_path, packet = _packet(tmp_path)
    for arm in packet["arms"]:
        planner = arm["planner_keys"][0]
        _write_campaign(
            Path(arm["campaign_root"]),
            campaign_id=arm["campaign_id"],
            config_hash=arm["config_sha256"][:16],
            planner=planner,
            execution_mode="native" if planner == "goal" else "adapter",
        )

    result = runner.aggregate_execution_packet(packet_path, output_dir=tmp_path / "aggregate")

    assert result["status"] == "native_aggregate_complete"
    assert result["benchmark_success"] is True
    assert result["included_planner_keys"] == ["goal"]
    assert result["excluded_rows"][0]["reason"] == "execution_mode_adapter"
    assert result["excluded_rows"][0]["blocking"] is False
    summary = json.loads(
        (tmp_path / "aggregate/reports/campaign_summary.json").read_text(encoding="utf-8")
    )
    assert [row["planner_key"] for row in summary["planner_rows"]] == ["goal"]
    report = (tmp_path / "aggregate/reports/campaign_report.md").read_text(encoding="utf-8")
    assert "Camera-Ready Benchmark Campaign Report" in report
    assert "goal" in report


def test_aggregate_fails_closed_for_missing_or_failed_arm(tmp_path: Path) -> None:
    """Missing and failed required arms block benchmark-success aggregation."""
    packet_path, packet = _packet(tmp_path)
    arm = packet["arms"][0]
    _write_campaign(
        Path(arm["campaign_root"]),
        campaign_id=arm["campaign_id"],
        config_hash=arm["config_sha256"][:16],
        planner=arm["planner_keys"][0],
        status="failed",
    )

    result = runner.aggregate_execution_packet(packet_path, output_dir=tmp_path / "aggregate")

    assert result["status"] == "blocked"
    assert result["benchmark_success"] is False
    assert {item["reason"] for item in result["excluded_rows"]} == {
        "row_status_failed",
        "missing_campaign_artifacts",
    }
    assert all(item["blocking"] for item in result["excluded_rows"])


def test_aggregate_checks_every_row_in_a_multiplanner_arm(tmp_path: Path) -> None:
    """A failed sibling row blocks even when its arm's first row is native."""
    packet = runner.build_execution_packet(
        _split_fixture(tmp_path, split_per_planner=False),
        output_root=tmp_path / "campaigns",
        campaign_prefix="issue5273_s20",
    )
    packet_path = tmp_path / "packet.json"
    _write_json(packet_path, packet)
    arm = packet["arms"][0]
    campaign_root = Path(arm["campaign_root"])
    _write_campaign(
        campaign_root,
        campaign_id=arm["campaign_id"],
        config_hash=arm["config_sha256"][:16],
        planner="goal",
    )
    summary_path = campaign_root / "reports/campaign_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    failed_row = dict(summary["planner_rows"][0])
    failed_row.update(
        {
            "planner_key": "orca",
            "status": "failed",
            "availability_status": "failed",
            "benchmark_success": False,
        }
    )
    summary["planner_rows"].append(failed_row)
    _write_json(summary_path, summary)

    result = runner.aggregate_execution_packet(packet_path, output_dir=tmp_path / "aggregate")

    assert result["status"] == "blocked"
    assert result["benchmark_success"] is False
    assert result["included_planner_keys"] == ["goal"]
    assert result["excluded_rows"][0]["planner_key"] == "orca"
    assert result["excluded_rows"][0]["reason"] == "row_status_failed"


def test_aggregate_fails_closed_for_contract_mismatch(tmp_path: Path) -> None:
    """Campaigns with different benchmark contracts cannot be combined."""
    packet_path, packet = _packet(tmp_path)
    for index, arm in enumerate(packet["arms"]):
        _write_campaign(
            Path(arm["campaign_root"]),
            campaign_id=arm["campaign_id"],
            config_hash=arm["config_sha256"][:16],
            planner=arm["planner_keys"][0],
            scenario_hash=f"matrix-{index}",
        )

    result = runner.aggregate_execution_packet(packet_path, output_dir=tmp_path / "aggregate")

    assert result["status"] == "blocked"
    mismatch = next(
        item for item in result["excluded_rows"] if item["reason"] == "campaign_contract_mismatch"
    )
    assert mismatch["mismatched_fields"] == ["scenario_matrix_hash"]
    assert mismatch["blocking"] is True


def test_aggregate_rejects_seed_policy_drift_from_packet(tmp_path: Path) -> None:
    """A completed arm cannot silently replace the planned committed seed set."""
    packet_path, packet = _packet(tmp_path)
    for arm in packet["arms"]:
        _write_campaign(
            Path(arm["campaign_root"]),
            campaign_id=arm["campaign_id"],
            config_hash=arm["config_sha256"][:16],
            planner=arm["planner_keys"][0],
        )
    manifest_path = Path(packet["arms"][0]["campaign_root"]) / "campaign_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["seed_policy"]["seed_set"] = "unplanned_seed_set"
    _write_json(manifest_path, manifest)

    result = runner.aggregate_execution_packet(packet_path, output_dir=tmp_path / "aggregate")

    mismatch = next(
        item for item in result["excluded_rows"] if item["reason"] == "seed_policy_mismatch"
    )
    assert mismatch["mismatched_fields"] == ["seed_set"]
    assert result["benchmark_success"] is False
