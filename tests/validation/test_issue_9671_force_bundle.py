"""Adversarial custody checks for the separately produced #9671 force sidecars."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from scripts.validation import check_issue_9671_force_bundle as bundle
from scripts.validation import issue_9671_force_observer_sitecustomize as observer


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _row() -> dict[str, Any]:
    return {
        "episode_id": "classic_doorway_medium--113--example",
        "scenario_id": "classic_doorway_medium",
        "seed": 113,
        "algo": "ppo",
        "scenario_params": {"algo": "ppo"},
        "status": "collision",
        "steps": 1,
        "algorithm_metadata": {
            "simulation_step_trace": {
                "reset": {
                    "robot": {"position": [2.0, 0.0]},
                    "pedestrians": [
                        {"actor_id": "simulator-slot-0", "position": [0.0, 0.0]},
                    ],
                },
                "steps": [
                    {
                        "step": 0,
                        "time_s": 0.1,
                        "robot": {"position": [2.0, 0.0]},
                        "pedestrians": [{"actor_id": "simulator-slot-0", "position": [0.1, 0.0]}],
                        "planner": {"ammv": {"pedestrian_force_vectors": [[1.0, 0.0]]}},
                    }
                ],
            },
        },
    }


def _capture(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "episode_id": row["episode_id"],
        "scenario_id": row["scenario_id"],
        "seed": row["seed"],
        "algo": row["algo"],
        "steps": [
            {
                "step": 0,
                "step_entry_positions": [[0.0, 0.0]],
                "force_input_positions": [[0.0, 0.0]],
                "post_step_positions": [[0.1, 0.0]],
                "total_forces": [[1.0, 0.0]],
                "robot_forces": [[0.2, 0.0]],
                "component_ids": ["ped_robot:robot_0"],
                "component_inputs": [
                    {
                        "positions": [[0.0, 0.0]],
                        "robot_position": [2.0, 0.0],
                        "multipliers": None,
                        "multipliers_applied": False,
                        "config": {
                            "is_active": True,
                            "robot_radius": 1.0,
                            "activation_threshold": 2.0,
                            "force_multiplier": 10.0,
                        },
                        "forces": [[0.2, 0.0]],
                    }
                ],
            }
        ],
    }


def _fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    row = _row()
    archive = tmp_path / "release.tar.gz"
    archive.write_bytes(b"synthetic-release-fixture")
    baseline_raw = tmp_path / "baseline.jsonl"
    new_raw = tmp_path / "new.jsonl"
    for path in (baseline_raw, new_raw):
        path.write_text(json.dumps(row) + "\n")
    baseline_report = tmp_path / "baseline-report.json"
    baseline_report.write_text(
        json.dumps(
            {
                "source_commit": bundle.trace_checker.SOURCE_SHA,
                "release_archive_sha256": _sha(archive),
                "comparison_counts": {"match": 1, "mismatch": 0, "no_release_row": 0},
                "trace_inputs_sha256": {str(baseline_raw): _sha(baseline_raw)},
            }
        )
    )
    observer_path = tmp_path / "sitecustomize.py"
    observer_path.write_bytes(b"synthetic-observer-fixture")
    campaign_items = {}
    for name in ("headon_group", "doorway"):
        config = tmp_path / f"{name}.yaml"
        config.write_text("synthetic-config")
        manifest = tmp_path / f"{name}-campaign.json"
        manifest.write_text("{}")
        startup = tmp_path / f"{name}-startup.json"
        startup.write_text(
            json.dumps(
                {
                    "schema_version": "robot-sf-slurm-startup.v2",
                    "status": "started",
                    "identities": {
                        "job_id": "123" if name == "headon_group" else "124",
                        "campaign": f"test-{name}",
                        "config": bundle.trace_checker.DIAGNOSTIC_CONFIG[name],
                        "public_commit": bundle.trace_checker.SOURCE_SHA,
                        "queue_id": f"queue-{name}",
                        "submission_id": f"submission-{name}",
                    },
                }
            )
        )
        sidecar_dir = tmp_path / f"{name}-sidecars"
        sidecar_dir.mkdir()
        campaign_items[name] = {
            "config": str(config),
            "campaign_manifest": str(manifest),
            "startup_receipt": str(startup),
            "campaign_id": f"test-{name}",
            "job_id": "123" if name == "headon_group" else "124",
            "traces": [str(new_raw)] if name == "headon_group" else [],
            "sidecar_dir": str(sidecar_dir),
        }
    spec = {
        "archive": str(archive),
        "baseline_report": str(baseline_report),
        "baseline_traces": [str(baseline_raw)],
        "observer_path": str(observer_path),
        "observer_sha256": _sha(observer_path),
        "campaigns": campaign_items,
    }
    sidecar = observer.bind_episode(_capture(row), row)
    sidecar["observer_provenance"] = {
        "frozen_source_commit": bundle.trace_checker.SOURCE_SHA,
        "frozen_source_file_sha256": bundle.FROZEN_SOURCE_FILE_SHA256,
        "diagnostic_config_sha256": _sha(Path(campaign_items["headon_group"]["config"])),
        "observer_sha256": spec["observer_sha256"],
        "campaign_id": "test-headon_group",
        "job_id": "123",
        "pid": "456",
    }
    sidecar_path = (
        Path(campaign_items["headon_group"]["sidecar_dir"])
        / f"{row['episode_id']}.robot-force.json"
    )
    sidecar_path.write_text(json.dumps(sidecar))
    key = ("ppo", "classic_doorway_medium", 113)
    report = {
        "diagnostic_inputs": {
            name: {
                "tuples": [key] if name == "headon_group" else [],
                "config_sha256": _sha(Path(item["config"])),
                "effective_config_hash": f"effective-{name}",
                "campaign_manifest_sha256": _sha(Path(item["campaign_manifest"])),
            }
            for name, item in campaign_items.items()
        },
        "release_archive_sha256": _sha(archive),
        "trace_inputs_sha256": {str(new_raw): _sha(new_raw)},
        "producer_manifests_sha256": {},
        "comparisons": [
            {
                "planner": "ppo",
                "scenario": "classic_doorway_medium",
                "seed": 113,
                "comparison": "match",
            }
        ],
        "comparison_counts": {"match": 1, "mismatch": 0, "no_release_row": 0},
    }

    def fake_checked_report(*args: Any, **kwargs: Any) -> dict[str, Any]:
        result = copy.deepcopy(report)
        result["trace_inputs_sha256"] = {str(new_raw): _sha(new_raw)}
        return result

    monkeypatch.setattr(bundle.trace_checker, "check", fake_checked_report)
    return spec, row, sidecar_path


def _build(spec: dict[str, Any]) -> dict[str, Any]:
    return bundle.build_manifest(
        spec,
        expected={("ppo", "classic_doorway_medium", 113)},
        expected_baseline_report_sha256=None,
        expected_baseline_counts={"match": 1, "mismatch": 0, "no_release_row": 0},
        expected_archive_sha256=None,
    )


def test_manifest_binds_exact_sidecar_bytes_and_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec, _, sidecar = _fixture(tmp_path, monkeypatch)
    result = _build(spec)
    assert result["admission_status"] == "candidate"
    assert result["episodes"][0]["release_comparison"] == "match"
    assert result["sidecar_sha256"][str(sidecar)] == _sha(sidecar)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(result))
    assert (
        bundle.validate_manifest(
            spec,
            manifest,
            _sha(manifest),
            expected={("ppo", "classic_doorway_medium", 113)},
            expected_baseline_report_sha256=None,
            expected_baseline_counts={"match": 1, "mismatch": 0, "no_release_row": 0},
            expected_archive_sha256=None,
        )
        == result
    )
    sidecar.write_text(sidecar.read_text() + " ")
    with pytest.raises(ValueError, match="manifest differs"):
        bundle.validate_manifest(
            spec,
            manifest,
            _sha(manifest),
            expected={("ppo", "classic_doorway_medium", 113)},
            expected_baseline_report_sha256=None,
            expected_baseline_counts={"match": 1, "mismatch": 0, "no_release_row": 0},
            expected_archive_sha256=None,
        )


@pytest.mark.parametrize(
    "mutation", ["missing", "extra", "nested_extra", "force_missing", "wrong_job", "wrong_row_hash"]
)
def test_rejects_sidecar_inventory_and_identity_mutations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    spec, _, sidecar = _fixture(tmp_path, monkeypatch)
    if mutation == "missing":
        sidecar.unlink()
    elif mutation == "extra":
        (sidecar.parent / "extra.robot-force.json").write_text("{}")
    elif mutation == "nested_extra":
        nested = sidecar.parent / "nested"
        nested.mkdir()
        (nested / sidecar.name).write_text(sidecar.read_text())
    else:
        data = json.loads(sidecar.read_text())
        if mutation == "force_missing":
            del data["steps"][0]["robot_forces"]
        elif mutation == "wrong_job":
            data["observer_provenance"]["job_id"] = "999"
        else:
            data["episode_record_canonical_sha256"] = "0" * 64
        sidecar.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        _build(spec)


def test_duplicate_raw_episode_identity_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec, row, _ = _fixture(tmp_path, monkeypatch)
    path = Path(spec["campaigns"]["headon_group"]["traces"][0])
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        _build(spec)


def test_rejects_slurm_startup_job_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec, _, _ = _fixture(tmp_path, monkeypatch)
    path = Path(spec["campaigns"]["headon_group"]["startup_receipt"])
    startup = json.loads(path.read_text())
    startup["identities"]["job_id"] = "999"
    path.write_text(json.dumps(startup))
    with pytest.raises(ValueError, match="Slurm startup identity"):
        _build(spec)


def test_changed_outcome_is_retained_as_diagnostic_finding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec, row, sidecar_path = _fixture(tmp_path, monkeypatch)
    row["status"] = "success"
    path = Path(spec["campaigns"]["headon_group"]["traces"][0])
    path.write_text(json.dumps(row) + "\n")
    sidecar = json.loads(sidecar_path.read_text())
    sidecar["episode_record_canonical_sha256"] = bundle._canonical_row_sha(row)
    sidecar_path.write_text(json.dumps(sidecar))
    result = _build(spec)
    assert result["admission_status"] == "diagnostic_mismatch"
    assert result["parity_differences"] == [
        {"tuple": ["ppo", "classic_doorway_medium", 113], "fields": ["status"]}
    ]
