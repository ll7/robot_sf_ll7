"""Adversarial custody checks for the separately produced #9671 force sidecars."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
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
    producer_path = new_raw.with_name(new_raw.name + ".provenance.json")
    producer_path.write_text("synthetic-producer-fixture")
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
        queue_id = f"queue-{name}"
        packet_sha = hashlib.sha256(f"runtime-{name}".encode()).hexdigest()
        nonce = f"nonce-{name}"
        submission_id = (
            "sha256:"
            + hashlib.sha256("\0".join((queue_id, packet_sha, "1", nonce)).encode()).hexdigest()
        )
        launch_packet = tmp_path / f"{name}-launch.yaml"
        launch_packet.write_text(
            "schema: robot-sf-launch-packet.v1\n"
            f"queue_id: {queue_id}\n"
            f"campaign_id: test-{name}\n"
            f"identity:\n  source_sha: {bundle.trace_checker.SOURCE_SHA}\n"
            f"  observer_sha256: {_sha(observer_path)}\n"
            f"  canonical_config_path: {bundle.trace_checker.DIAGNOSTIC_CONFIG[name]}\n"
            f"  canonical_config_sha256: {_sha(config)}\n"
            "  private_ops_runtime_commit: test-private-runtime\n"
        )
        intent_path = tmp_path / f"{name}-intent.json"
        intent_path.write_text(
            json.dumps(
                {
                    "schema": "robot-sf-submission-intent.v1",
                    "queue_id": queue_id,
                    "submission_id": submission_id,
                    "packet_sha256": packet_sha,
                    "campaign": f"test-{name}",
                    "public_commit": bundle.trace_checker.SOURCE_SHA,
                    "private_ops_commit": "test-private-runtime",
                    "attempt": 1,
                    "nonce": nonce,
                }
            )
        )
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
                        "queue_id": queue_id,
                        "submission_id": submission_id,
                        "packet_sha256": packet_sha,
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
            "launch_packet": str(launch_packet),
            "launch_packet_sha256": _sha(launch_packet),
            "packet_sha256": packet_sha,
            "submission_intent_receipt": str(intent_path),
            "submission_intent_sha256": _sha(intent_path),
            "queue_id": queue_id,
            "submission_id": submission_id,
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
        "producer_manifests_sha256": {str(producer_path): _sha(producer_path)},
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
        current_raw = Path(args[1][0])
        current_producer = current_raw.with_name(current_raw.name + ".provenance.json")
        result["trace_inputs_sha256"] = {str(current_raw): _sha(current_raw)}
        result["producer_manifests_sha256"] = {str(current_producer): _sha(current_producer)}
        return result

    monkeypatch.setattr(bundle.trace_checker, "check", fake_checked_report)
    return spec, row, sidecar_path


def _approved_pins(spec: dict[str, Any]) -> dict[str, str]:
    return {name: item["launch_packet_sha256"] for name, item in spec["campaigns"].items()}


def _build(spec: dict[str, Any], approved_pins: dict[str, str] | None = None) -> dict[str, Any]:
    return bundle.build_manifest(
        spec,
        approved_launch_packet_sha256=approved_pins or _approved_pins(spec),
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
    assert result["sidecar_sha256"][result["episodes"][0]["sidecar_path"]] == _sha(sidecar)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(result))
    assert (
        bundle.validate_manifest(
            spec,
            manifest,
            _sha(manifest),
            approved_launch_packet_sha256=_approved_pins(spec),
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
            approved_launch_packet_sha256=_approved_pins(spec),
            expected={("ppo", "classic_doorway_medium", 113)},
            expected_baseline_report_sha256=None,
            expected_baseline_counts={"match": 1, "mismatch": 0, "no_release_row": 0},
            expected_archive_sha256=None,
        )


def test_manifest_survives_cold_tree_relocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec, _, _ = _fixture(tmp_path, monkeypatch)
    manifest = tmp_path / "bundle.json"
    manifest.write_text(json.dumps(_build(spec)))
    relocated = tmp_path.parent / f"{tmp_path.name}-relocated"
    shutil.copytree(tmp_path, relocated)
    moved_spec = json.loads(json.dumps(spec).replace(str(tmp_path), str(relocated)))
    assert (
        bundle.validate_manifest(
            moved_spec,
            relocated / "bundle.json",
            _sha(manifest),
            approved_launch_packet_sha256=_approved_pins(moved_spec),
            expected={("ppo", "classic_doorway_medium", 113)},
            expected_baseline_report_sha256=None,
            expected_baseline_counts={"match": 1, "mismatch": 0, "no_release_row": 0},
            expected_archive_sha256=None,
        )["admission_status"]
        == "candidate"
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


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_packet",
        "forged_packet",
        "missing_reviewed_packet",
        "wrong_queue",
        "wrong_submission",
        "forged_intent",
    ],
)
def test_rejects_unbound_submission_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    spec, _, _ = _fixture(tmp_path, monkeypatch)
    campaign = spec["campaigns"]["headon_group"]
    startup_path = Path(campaign["startup_receipt"])
    startup = json.loads(startup_path.read_text())
    if mutation == "missing_packet":
        del startup["identities"]["packet_sha256"]
    elif mutation == "forged_packet":
        startup["identities"]["packet_sha256"] = "f" * 64
    elif mutation == "missing_reviewed_packet":
        del campaign["packet_sha256"]
    elif mutation == "wrong_queue":
        startup["identities"]["queue_id"] = "unreviewed-queue"
    elif mutation == "wrong_submission":
        startup["identities"]["submission_id"] = "sha256:" + "f" * 64
    else:
        intent_path = Path(campaign["submission_intent_receipt"])
        intent = json.loads(intent_path.read_text())
        intent["queue_id"] = "unreviewed-queue"
        intent_path.write_text(json.dumps(intent))
        campaign["submission_intent_sha256"] = _sha(intent_path)
    startup_path.write_text(json.dumps(startup))
    with pytest.raises(ValueError, match="submission intent identity|submission intent SHA-256"):
        _build(spec)


def test_coherent_observer_spec_sidecar_and_packet_swap_rejects_review_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec, _, sidecar_path = _fixture(tmp_path, monkeypatch)
    approved_pins = _approved_pins(spec)
    observer_path = Path(spec["observer_path"])
    old_observer_sha = spec["observer_sha256"]
    observer_path.write_bytes(b"coherently-forged-observer")
    spec["observer_sha256"] = _sha(observer_path)
    sidecar = json.loads(sidecar_path.read_text())
    sidecar["observer_provenance"]["observer_sha256"] = spec["observer_sha256"]
    sidecar_path.write_text(json.dumps(sidecar))
    for item in spec["campaigns"].values():
        launch_path = Path(item["launch_packet"])
        launch_path.write_text(
            launch_path.read_text().replace(old_observer_sha, spec["observer_sha256"])
        )
        item["launch_packet_sha256"] = _sha(launch_path)
    with pytest.raises(ValueError, match="independently approved review pin"):
        _build(spec, approved_pins)


def test_observer_swap_rejects_unchanged_reviewed_packet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec, _, sidecar_path = _fixture(tmp_path, monkeypatch)
    observer_path = Path(spec["observer_path"])
    observer_path.write_bytes(b"forged-observer")
    spec["observer_sha256"] = _sha(observer_path)
    sidecar = json.loads(sidecar_path.read_text())
    sidecar["observer_provenance"]["observer_sha256"] = spec["observer_sha256"]
    sidecar_path.write_text(json.dumps(sidecar))
    with pytest.raises(ValueError, match="submission intent identity"):
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
