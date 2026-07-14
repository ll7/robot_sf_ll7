"""Tests for the publication-bundle collision-consumer audit."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from robot_sf.benchmark.event_ledger import build_event_ledger
from robot_sf.benchmark.metrics import snqi
from robot_sf.benchmark.snqi_scalarization_sensitivity import (
    load_baseline_mapping,
    load_weight_mapping,
)

if TYPE_CHECKING:
    from collections.abc import Callable

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts/validation/check_release_collision_consumers.py"
WEIGHTS_PATH = ROOT / "configs/benchmarks/snqi_weights_camera_ready_v3.json"
BASELINE_PATH = ROOT / "configs/benchmarks/snqi_baseline_camera_ready_v3.json"
TEST_COMMIT = "a" * 40

_SPEC = importlib.util.spec_from_file_location("check_release_collision_consumers", SCRIPT_PATH)
assert _SPEC is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

audit_bundle = _MODULE.audit_bundle


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, sort_keys=True) + "\n").encode()


def _episode(*, collision: bool, seed: int) -> dict[str, Any]:
    weights = load_weight_mapping(WEIGHTS_PATH)
    baseline = load_baseline_mapping(BASELINE_PATH)
    success = not collision
    total = int(collision)
    metrics: dict[str, Any] = {
        "collisions": total,
        "total_collision_count": total,
        "ped_collision_count": total,
        "obstacle_collision_count": 0,
        "agent_collision_count": 0,
        "success": success,
        "time_to_goal_norm": 0.5 if success else 1.0,
        "near_misses": 0,
        "comfort_exposure": 0.0,
        "force_exceed_events": 0,
        "jerk_mean": 0.0,
        "curvature_mean": 0.0,
    }
    metrics["snqi"] = snqi(metrics, weights, baseline)
    record: dict[str, Any] = {
        "episode_id": f"scenario--{seed}",
        "scenario_id": "scenario",
        "seed": seed,
        "algo": "planner",
        "git_hash": TEST_COMMIT,
        "status": "collision" if collision else "success",
        "termination_reason": "collision" if collision else "success",
        "metrics": metrics,
        "outcome": {
            "collision_event": collision,
            "route_complete": success,
            "timeout_event": False,
        },
    }
    collision_events = None
    if collision:
        collision_events = [
            {
                "collision_partner_type": "pedestrian",
                "collision_partner_id": "ped-1",
                "collision_time": 1.0,
                "relative_speed_at_contact": 0.5,
                "clearance_series_source": "fixture",
                "exact_event_source": "fixture",
            }
        ]
    record["event_ledger"] = build_event_ledger(record, collision_events=collision_events)
    return record


def _add_bytes(archive: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mtime = 0
    info.mode = 0o644
    archive.addfile(info, io.BytesIO(data))


def _bundle(
    tmp_path: Path,
    *,
    mutate_rows: Callable[[list[dict[str, Any]]], None] | None = None,
    tamper_after_signing: bool = False,
) -> tuple[Path, str]:
    rows = [_episode(collision=False, seed=1), _episode(collision=True, seed=2)]
    if mutate_rows is not None:
        mutate_rows(rows)
    weights_sha256 = hashlib.sha256(WEIGHTS_PATH.read_bytes()).hexdigest()
    baseline_sha256 = hashlib.sha256(BASELINE_PATH.read_bytes()).hexdigest()
    payloads = {
        "release/release_manifest.resolved.json": _json_bytes(
            {
                "release_tag": "test-v1",
                "metrics": {
                    "snqi_weights_sha256": weights_sha256,
                    "snqi_baseline_sha256": baseline_sha256,
                },
            }
        ),
        "release/release_result.json": _json_bytes(
            {
                "status": "benchmark_success",
                "evidence_status": "valid",
                "total_episodes": 2,
                "successful_runs": 1,
            }
        ),
        "reports/campaign_summary.json": _json_bytes(
            {
                "campaign": {
                    "status": "benchmark_success",
                    "evidence_status": "valid",
                    "total_episodes": 2,
                    "successful_runs": 1,
                    "snqi_contract_status": "pass",
                    "snqi_weights_sha256": weights_sha256,
                    "snqi_baseline_sha256": baseline_sha256,
                }
            }
        ),
        "runs/planner__differential_drive/episodes.jsonl": b"".join(
            _json_bytes(row) for row in rows
        ),
    }
    signed_hashes = {path: _sha256(data) for path, data in payloads.items()}
    publication_manifest = {
        "publication_channels": {"release_tag": "test-v1"},
        "provenance": {"repository": {"commit": TEST_COMMIT}},
        "files": [
            {"path": path, "sha256": digest} for path, digest in sorted(signed_hashes.items())
        ],
    }
    checksum_bytes = "".join(
        f"{digest}  {path}\n" for path, digest in sorted(signed_hashes.items())
    ).encode()
    if tamper_after_signing:
        payloads["release/release_result.json"] += b" "

    bundle = tmp_path / "fixture.tar.gz"
    root = "fixture"
    with tarfile.open(bundle, "w:gz") as archive:
        _add_bytes(archive, f"{root}/README.md", b"fixture\n")
        _add_bytes(archive, f"{root}/checksums.sha256", checksum_bytes)
        for path, data in payloads.items():
            _add_bytes(archive, f"{root}/payload/{path}", data)
        _add_bytes(
            archive,
            f"{root}/publication_manifest.json",
            _json_bytes(publication_manifest),
        )
    return bundle, hashlib.sha256(bundle.read_bytes()).hexdigest()


def _audit(bundle: Path, digest: str, **overrides: object) -> dict[str, Any]:
    arguments: dict[str, object] = {
        "expected_bundle_sha256": digest,
        "expected_release_tag": "test-v1",
        "expected_rows": 2,
        "expected_arms": 1,
        "expected_rows_per_arm": 2,
    }
    arguments.update(overrides)
    return audit_bundle(bundle, **arguments)


def test_valid_bundle_reconciles_success_and_snqi(tmp_path: Path) -> None:
    """A consistent signed bundle passes every narrow consumer check."""
    bundle, digest = _bundle(tmp_path)

    report = _audit(bundle, digest)

    assert report["status"] == "pass"
    assert report["violation_count"] == 0
    assert report["counts"]["rows"] == 2
    assert report["counts"]["exact_collision_rows"] == 1
    assert report["counts"]["snqi_recomputed_rows"] == 2


def test_collision_marked_success_fails_closed(tmp_path: Path) -> None:
    """An exact collision cannot remain a successful episode."""

    def mark_collision_success(rows: list[dict[str, Any]]) -> None:
        collision_row = rows[1]
        collision_row["metrics"]["success"] = True
        collision_row["metrics"]["snqi"] = snqi(
            collision_row["metrics"],
            load_weight_mapping(WEIGHTS_PATH),
            load_baseline_mapping(BASELINE_PATH),
        )

    bundle, digest = _bundle(tmp_path, mutate_rows=mark_collision_success)

    report = _audit(bundle, digest)

    assert report["status"] == "fail"
    assert report["violation_counts"]["collision_success"] == 1


def test_embedded_checksum_mismatch_fails_closed(tmp_path: Path) -> None:
    """Payload content changed after signing is rejected."""
    bundle, digest = _bundle(tmp_path, tamper_after_signing=True)

    report = _audit(bundle, digest)

    assert report["status"] == "fail"
    assert report["violation_counts"]["payload_sha256"] == 1


def test_expected_cardinality_mismatch_fails_closed(tmp_path: Path) -> None:
    """A truncated or over-complete campaign is rejected."""
    bundle, digest = _bundle(tmp_path)

    report = _audit(bundle, digest, expected_rows=3)

    assert report["status"] == "fail"
    assert report["violation_counts"]["row_count"] == 1


def test_stored_snqi_drift_fails_closed(tmp_path: Path) -> None:
    """A score that does not reproduce under canonical inputs is rejected."""

    def alter_snqi(rows: list[dict[str, Any]]) -> None:
        rows[0]["metrics"]["snqi"] += 0.01

    bundle, digest = _bundle(tmp_path, mutate_rows=alter_snqi)

    report = _audit(bundle, digest)

    assert report["status"] == "fail"
    assert report["violation_counts"]["snqi_recomputation"] == 1


def test_exact_collision_count_disagreement_fails_closed(tmp_path: Path) -> None:
    """A typed exact collision cannot be paired with a zero aggregate."""

    def zero_collision_count(rows: list[dict[str, Any]]) -> None:
        collision_metrics = rows[1]["metrics"]
        collision_metrics["collisions"] = 0
        collision_metrics["total_collision_count"] = 0
        collision_metrics["ped_collision_count"] = 0

    bundle, digest = _bundle(tmp_path, mutate_rows=zero_collision_count)

    report = _audit(bundle, digest)

    assert report["status"] == "fail"
    assert report["violation_counts"]["exact_collision_count"] == 1


def test_cli_writes_pass_report(tmp_path: Path) -> None:
    """The command-line path writes and prints the same passing report."""
    bundle, digest = _bundle(tmp_path)
    output = tmp_path / "report.json"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--bundle",
            str(bundle),
            "--expected-bundle-sha256",
            digest,
            "--expected-release-tag",
            "test-v1",
            "--expected-rows",
            "2",
            "--expected-arms",
            "1",
            "--expected-rows-per-arm",
            "2",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "pass"
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "pass"
