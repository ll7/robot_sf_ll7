"""Focused contract tests for the check-only log-retention helper (#8856)."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import TYPE_CHECKING

import pytest

from scripts.tools.check_log_retention import EXIT_OK, build_report, main

if TYPE_CHECKING:
    from pathlib import Path


def _metadata(data: bytes) -> dict[str, object]:
    return {
        "size_bytes": len(data),
        "line_count": data.count(b"\n") + int(bool(data) and not data.endswith(b"\n")),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _case(
    tmp_path: Path,
    *,
    data: bytes = b"start\nready\nERROR: useful detail\nend\n",
    role: str = "task_stderr",
    task_id: str | None = "0",
    job_state: str = "completed",
    diagnosis: str = "not_required",
    custody: bool = True,
) -> tuple[dict[str, object], Path]:
    root = tmp_path / "logs"
    root.mkdir()
    path = root / "task-0.log"
    path.write_bytes(data)
    entry: dict[str, object] = {
        "path": "task-0.log",
        "role": role,
        "job_id": "job-1",
        "task_id": task_id,
        "encoding": "utf-8",
        "completion": "complete",
        "retention_class": "disposable",
        **_metadata(data),
    }
    manifest: dict[str, object] = {
        "schema": "log_retention_manifest.v1",
        "job": {"job_id": "job-1", "state": job_state, "diagnosis": diagnosis},
        "excerpt_policy": {
            "version": "log_excerpt.v1",
            "max_bytes": 180,
            "first_lines": 1,
            "last_lines": 1,
            "context_lines": 1,
            "max_log_bytes": 4096,
            "loss_declared": True,
        },
        "logs": [entry],
    }
    if custody:
        manifest["custody"] = {
            "status": "verified",
            "durability_class": "cloud_durable",
            "files": [
                {
                    "relative_path": "task-0.log",
                    "sha256": entry["sha256"],
                    "byte_size": entry["size_bytes"],
                    "state": "verified",
                }
            ],
        }
    return manifest, root


def test_valid_log_has_provenance_excerpt_and_storage_estimate(tmp_path: Path) -> None:
    manifest, root = _case(tmp_path)
    report = build_report(manifest, root)

    assert report["status"] == "eligible"
    entry = report["logs"][0]
    assert entry["identity"] == {"job_id": "job-1", "task_id": "0"}
    assert entry["size_bytes"] == manifest["logs"][0]["size_bytes"]
    assert entry["line_count"] == 4
    assert entry["encoding"] == "utf-8"
    assert entry["completion"] == "complete"
    assert entry["sha256"] == manifest["logs"][0]["sha256"]
    assert entry["retention_class"] == "disposable"
    assert entry["excerpt"]["source_location"] == "root-relative:task-0.log"
    assert entry["excerpt"]["source_sha256"] == entry["sha256"]
    assert len(entry["excerpt"]["text"].encode()) <= 180
    assert report["storage_estimate"]["full_log_bytes"] == manifest["logs"][0]["size_bytes"]
    assert report["storage_estimate"]["prune_eligible_bytes"] > 0
    assert report["deletion_plan"] == []


def test_excerpt_is_deterministic_and_sanitized(tmp_path: Path) -> None:
    data = b"first\nkeep\nERROR host=node42 user=alice /scratch/alice/x https://internal.local/a\nkeep\nlast\n"
    manifest, root = _case(tmp_path, data=data)
    first = build_report(manifest, root)
    second = build_report(deepcopy(manifest), root)

    assert first == second
    excerpt = first["logs"][0]["excerpt"]
    assert {item["kind"] for item in excerpt["ranges"]} == {
        "first",
        "error_context",
        "last",
    }
    assert "node42" not in excerpt["text"]
    assert "alice" not in excerpt["text"]
    assert "internal.local" not in excerpt["text"]
    assert "<redacted-" in excerpt["text"]


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        (lambda item: item.update(completion="active"), "active_log"),
        (lambda item: item.update(completion="truncated"), "truncated_log"),
        (lambda item: item.update(), "binary_log"),
        (lambda item: item.update(sha256="0" * 64), "digest_mismatch"),
        (lambda item: item.update(task_id=None), "task_identity_missing"),
    ],
)
def test_invalid_logs_fail_closed(tmp_path: Path, change, reason: str) -> None:
    data = b"safe\n"
    manifest, root = _case(tmp_path, data=data)
    if reason == "binary_log":
        path = root / "task-0.log"
        path.write_bytes(b"safe\n\x00binary\n")
        manifest["logs"][0].update(_metadata(path.read_bytes()))
        manifest["custody"]["files"][0].update(_metadata(path.read_bytes()))
    change(manifest["logs"][0])
    report = build_report(manifest, root)

    assert report["status"] == "blocked"
    assert reason in report["logs"][0]["reasons"]
    assert report["logs"][0]["prune_eligible"] is False
    assert report["storage_estimate"]["prune_eligible_bytes"] == 0


def test_duplicate_log_identity_blocks_all_pruning(tmp_path: Path) -> None:
    manifest, root = _case(tmp_path)
    duplicate = deepcopy(manifest["logs"][0])
    duplicate["path"] = "task-0.log"
    manifest["logs"].append(duplicate)
    report = build_report(manifest, root)

    assert report["status"] == "blocked"
    assert any(problem["code"] == "duplicate_log" for problem in report["problems"])
    assert report["storage_estimate"]["prune_eligible_bytes"] == 0


def test_missing_custody_retains_complete_log_and_zeroes_prune_estimate(tmp_path: Path) -> None:
    manifest, root = _case(tmp_path, custody=False)
    report = build_report(manifest, root)

    assert report["status"] == "blocked"
    assert report["custody"]["verified"] is False
    assert report["storage_estimate"]["prune_eligibility"] == "blocked_durable_custody"
    assert report["storage_estimate"]["prune_eligible_bytes"] == 0
    assert report["logs"][0]["retention_action"] == "retain_until_custody"


def test_failed_job_keeps_full_log_until_diagnosis_and_transfer(tmp_path: Path) -> None:
    manifest, root = _case(tmp_path, job_state="failed", diagnosis="pending")
    report = build_report(manifest, root)

    entry = report["logs"][0]
    assert report["status"] == "eligible"
    assert entry["retention_action"] == "retain_full"
    assert "failed_job_full_retention" in entry["reasons"]
    assert entry["prune_eligible"] is False
    assert report["storage_estimate"]["prune_eligibility"] == "retain_required"


def test_cli_json_smoke(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    manifest, root = _case(tmp_path)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    code = main(["--check", "--manifest", str(path), "--root", str(root), "--format", "json"])

    assert code == EXIT_OK
    assert json.loads(capsys.readouterr().out)["check_only"] is True
