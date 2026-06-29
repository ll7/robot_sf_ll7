"""Tests for the SLURM-to-claim finalizer readiness gate (#3425).

The gate is CPU-only and must fail closed when durable pointers or manifest
linkage are missing. These tests build synthetic queue/manifest/finalizer inputs
in ``tmp_path`` and assert the readiness verdict and exit codes without any SLURM
execution.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "scripts" / "validation" / "preflight_slurm_finalizer.py"

# Import the module directly for fast in-process assertions on the report shape.
sys.path.insert(0, str(REPO_ROOT))
from scripts.validation import preflight_slurm_finalizer as gate  # noqa: E402


def _write_queue(path: Path, *, issue: int = 3425) -> Path:
    """Write a minimal submission queue with one entry."""
    payload = {
        "entries": [
            {"id": "slice_a", "seeds": [101], "status": "completed", "issue": issue},
        ]
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def _write_manifest(path: Path, *, job_id: str = "12345") -> Path:
    """Write a submission manifest with one job linking to the queue entry."""
    payload = {
        "jobs": [
            {
                "queue_id": "slice_a",
                "status": "submitted",
                "slurm_job_id": job_id,
                "seeds": [101],
                "experiment_id": "slice_a_exp",
            }
        ]
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def _write_finalizer(
    path: Path,
    *,
    issue: int = 3425,
    job_id: str = "12345",
    classification: str = "success",
    durable_uri: str | None = "wandb://entity/project/run",
    claim_boundary: str | None = "smoke evidence only",
) -> Path:
    """Write a finalizer manifest matching the finalization schema."""
    payload: dict = {
        "schema_version": "robot-sf-slurm-job-finalization.v1",
        "issue_number": issue,
        "job_id": job_id,
        "classification": classification,
        "artifact_status": "all_required_present",
    }
    if durable_uri is not None:
        payload["durable_uri"] = durable_uri
    if claim_boundary is not None:
        payload["claim_boundary"] = claim_boundary
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _ready_inputs(tmp_path: Path) -> dict:
    """Build a fully ready input set."""
    return {
        "queue_path": _write_queue(tmp_path / "queue.yaml"),
        "submission_manifests": [_write_manifest(tmp_path / "manifest.yaml")],
        "finalizer_manifests": [_write_finalizer(tmp_path / "finalizer.json")],
    }


def _run_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(RUNNER), *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_ready_when_all_inputs_present(tmp_path: Path) -> None:
    """A complete, well-linked input set is ready."""
    report = gate.preflight(**_ready_inputs(tmp_path), generated_at="2026-06-27T00:00:00+00:00")

    assert report["ready"] is True
    assert report["blockers"] == []
    assert report["input_errors"] == []
    statuses = {check["name"]: check["status"] for check in report["checks"]}
    assert statuses["durable_pointer_present"] == "ready"
    assert statuses["finalizer_manifest_linkage"] == "ready"
    assert statuses["claim_boundary_present"] == "ready"


def test_blocks_when_durable_pointer_missing(tmp_path: Path) -> None:
    """A successful finalizer without a durable pointer fails closed."""
    inputs = _ready_inputs(tmp_path)
    _write_finalizer(tmp_path / "finalizer.json", durable_uri=None)

    report = gate.preflight(**inputs)

    assert report["ready"] is False
    blocker_names = {blocker["check"] for blocker in report["blockers"]}
    assert "durable_pointer_present" in blocker_names
    # The blocker carries the smallest external action.
    durable_blocker = next(b for b in report["blockers"] if b["check"] == "durable_pointer_present")
    assert "durable" in durable_blocker["remediation"]


def test_blocks_when_manifest_linkage_missing(tmp_path: Path) -> None:
    """A finalizer whose job id has no manifest job fails closed on linkage."""
    inputs = _ready_inputs(tmp_path)
    # Finalizer reports a different job id than the manifest records.
    _write_finalizer(tmp_path / "finalizer.json", job_id="99999")

    report = gate.preflight(**inputs)

    assert report["ready"] is False
    blocker_names = {blocker["check"] for blocker in report["blockers"]}
    assert "finalizer_manifest_linkage" in blocker_names


def test_blocks_when_issue_traceability_mismatches(tmp_path: Path) -> None:
    """Queue/finalizer issue mismatch blocks public issue traceability."""
    inputs = _ready_inputs(tmp_path)
    _write_finalizer(tmp_path / "finalizer.json", issue=9999)

    report = gate.preflight(**inputs)

    assert report["ready"] is False
    blocker = next(b for b in report["blockers"] if b["check"] == "issue_traceability_matches")
    assert "align queue and finalizer issue numbers" in blocker["remediation"]


def test_blocks_when_queue_issue_traceability_missing(tmp_path: Path) -> None:
    """Missing queue issue metadata blocks public handoff traceability."""
    queue = tmp_path / "queue.yaml"
    queue.write_text(
        yaml.safe_dump(
            {
                "entries": [
                    {"id": "slice_a", "seeds": [101], "status": "completed"},
                ]
            }
        ),
        encoding="utf-8",
    )
    inputs = {
        "queue_path": queue,
        "submission_manifests": [_write_manifest(tmp_path / "manifest.yaml")],
        "finalizer_manifests": [_write_finalizer(tmp_path / "finalizer.json")],
    }

    report = gate.preflight(**inputs)

    assert report["ready"] is False
    blocker = next(b for b in report["blockers"] if b["check"] == "issue_traceability_matches")
    assert "queue entries do not record an issue number" in blocker["detail"]


def test_blocks_when_claim_boundary_missing(tmp_path: Path) -> None:
    """A finalizer without a claim boundary fails closed."""
    inputs = _ready_inputs(tmp_path)
    _write_finalizer(tmp_path / "finalizer.json", claim_boundary=None)

    report = gate.preflight(**inputs)

    assert report["ready"] is False
    blocker_names = {blocker["check"] for blocker in report["blockers"]}
    assert "claim_boundary_present" in blocker_names


def test_blocks_when_finalizer_manifest_absent(tmp_path: Path) -> None:
    """No finalizer manifest blocks with a finalize remediation."""
    report = gate.preflight(
        queue_path=_write_queue(tmp_path / "queue.yaml"),
        submission_manifests=[_write_manifest(tmp_path / "manifest.yaml")],
        finalizer_manifests=[],
    )

    assert report["ready"] is False
    blocker = next(b for b in report["blockers"] if b["check"] == "finalizer_manifests_present")
    assert "slurm_job_finalize.py" in blocker["remediation"]
    # Dependent content checks are skipped, not falsely passed.
    statuses = {check["name"]: check["status"] for check in report["checks"]}
    assert statuses["finalizer_manifest_linkage"] == "skipped"
    assert statuses["durable_pointer_present"] == "skipped"


def test_blocks_when_queue_absent(tmp_path: Path) -> None:
    """Missing queue blocks with a --queue remediation."""
    report = gate.preflight(
        queue_path=None,
        submission_manifests=[_write_manifest(tmp_path / "manifest.yaml")],
        finalizer_manifests=[_write_finalizer(tmp_path / "finalizer.json")],
    )

    assert report["ready"] is False
    blocker = next(b for b in report["blockers"] if b["check"] == "queue_present")
    assert "--queue" in blocker["remediation"]


def test_non_success_finalizer_does_not_require_durable_pointer(tmp_path: Path) -> None:
    """A still-failed finalizer without a durable pointer is not a blocker."""
    inputs = _ready_inputs(tmp_path)
    _write_finalizer(tmp_path / "finalizer.json", classification="failed", durable_uri=None)

    report = gate.preflight(**inputs)

    statuses = {check["name"]: check["status"] for check in report["checks"]}
    assert statuses["durable_pointer_present"] == "ready"


def test_malformed_finalizer_is_input_error(tmp_path: Path) -> None:
    """A finalizer with an unsupported schema is an input error, not a blocker."""
    inputs = _ready_inputs(tmp_path)
    (tmp_path / "finalizer.json").write_text(
        json.dumps({"schema_version": "bogus", "job_id": "12345"}), encoding="utf-8"
    )

    report = gate.preflight(**inputs)

    assert report["ready"] is False
    assert report["input_errors"]
    assert any("unsupported schema_version" in err for err in report["input_errors"])


def test_advisory_evidence_root_does_not_block(tmp_path: Path) -> None:
    """A missing evidence root is advisory and does not flip ready to False."""
    report = gate.preflight(**_ready_inputs(tmp_path), evidence_root=tmp_path / "missing_evidence")

    assert report["ready"] is True
    advisory_names = {advisory["check"] for advisory in report["advisories"]}
    assert "evidence_root_present" in advisory_names


def test_advisory_evidence_root_file_is_not_a_directory(tmp_path: Path) -> None:
    """A file passed as the evidence root is advisory-blocked (reconciler scans a dir)."""
    evidence_file = tmp_path / "evidence.txt"
    evidence_file.write_text("not a directory", encoding="utf-8")

    report = gate.preflight(**_ready_inputs(tmp_path), evidence_root=evidence_file)

    # Advisory only: ready stays True, but the check reports the non-directory.
    assert report["ready"] is True
    advisory_names = {advisory["check"] for advisory in report["advisories"]}
    assert "evidence_root_present" in advisory_names
    root_check = next(c for c in report["checks"] if c["name"] == "evidence_root_present")
    assert root_check["status"] == "blocked"
    assert "not a directory" in root_check["detail"]


def test_cli_exit_codes_ready_blocked_and_input_error(tmp_path: Path) -> None:
    """The CLI returns 0 ready, 1 blocked, and 2 on input error."""
    queue = _write_queue(tmp_path / "queue.yaml")
    manifest = _write_manifest(tmp_path / "manifest.yaml")
    finalizer = _write_finalizer(tmp_path / "finalizer.json")

    ready = _run_cli(
        [
            "--queue",
            str(queue),
            "--submission-manifest",
            str(manifest),
            "--finalizer-manifest",
            str(finalizer),
            "--json",
        ]
    )
    assert ready.returncode == 0, ready.stderr
    payload = json.loads(ready.stdout)
    assert payload["ready"] is True

    # Drop the durable pointer -> blocked (exit 1).
    _write_finalizer(tmp_path / "finalizer.json", durable_uri=None)
    blocked = _run_cli(
        [
            "--queue",
            str(queue),
            "--submission-manifest",
            str(manifest),
            "--finalizer-manifest",
            str(finalizer),
        ]
    )
    assert blocked.returncode == 1
    assert "BLOCKED" in blocked.stdout

    # Corrupt the schema -> input error (exit 2).
    (tmp_path / "finalizer.json").write_text(
        json.dumps({"schema_version": "bogus", "job_id": "12345"}), encoding="utf-8"
    )
    errored = _run_cli(
        [
            "--queue",
            str(queue),
            "--submission-manifest",
            str(manifest),
            "--finalizer-manifest",
            str(finalizer),
        ]
    )
    assert errored.returncode == 2
