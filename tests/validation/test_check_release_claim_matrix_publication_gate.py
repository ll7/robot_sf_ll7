"""Tests for the Issue #2910 release claim matrix publication gate."""

from __future__ import annotations

import json

from scripts.validation.check_release_claim_matrix_publication_gate import (
    EXPECTED_MATRIX_SCHEMA,
    build_gate_report,
    main,
)


def _matrix(rows: list[dict]) -> dict:
    """Build minimal release claim matrix fixture."""

    return {
        "schema_version": EXPECTED_MATRIX_SCHEMA,
        "issue": 3294,
        "rows": rows,
    }


def test_gate_passes_complete_benchmark_evidence_row(tmp_path) -> None:
    """A benchmark-evidence row passes when publication prerequisites are present."""

    artifact = tmp_path / "docs" / "context" / "evidence" / "artifact.json"
    source = tmp_path / "docs" / "context" / "source.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}", encoding="utf-8")
    source.write_text("{}", encoding="utf-8")

    report = build_gate_report(
        _matrix(
            [
                {
                    "row_id": "release_artifact:complete",
                    "classification": "benchmark evidence",
                    "availability_status": "available",
                    "artifact_match": True,
                    "artifact_uri": "docs/context/evidence/artifact.json",
                    "scenario_certification": "scenario_cert.v1:accepted",
                    "source_refs": ["docs/context/source.json"],
                }
            ]
        ),
        repo_root=tmp_path,
    )

    assert report["status"] == "pass"
    assert report["summary"]["blocker_count"] == 0


def test_gate_fails_closed_for_missing_certification_and_output_artifact(tmp_path) -> None:
    """Benchmark-evidence rows fail closed on missing certification and local outputs."""

    source = tmp_path / "docs" / "context" / "source.json"
    source.parent.mkdir(parents=True)
    source.write_text("{}", encoding="utf-8")

    report = build_gate_report(
        _matrix(
            [
                {
                    "row_id": "release_artifact:blocked",
                    "classification": "benchmark evidence",
                    "availability_status": "available",
                    "artifact_match": True,
                    "artifact_uri": "output/local.json",
                    "scenario_certification": "not_available_in_manifest",
                    "source_refs": ["docs/context/source.json"],
                }
            ]
        ),
        repo_root=tmp_path,
    )

    assert report["status"] == "blocked"
    checks = {blocker["check"] for blocker in report["blockers"]}
    assert checks == {"artifact_uri", "scenario_certification"}


def test_gate_rejects_non_benchmark_success_promotion(tmp_path) -> None:
    """Diagnostic and non-claim rows must not set benchmark_success true."""

    report = build_gate_report(
        _matrix(
            [
                {
                    "row_id": "leaderboard:diagnostic",
                    "classification": "diagnostic evidence",
                    "benchmark_success": True,
                }
            ]
        ),
        repo_root=tmp_path,
    )

    assert report["status"] == "blocked"
    assert report["blockers"][0]["check"] == "non_benchmark_promotion"


def test_committed_matrix_remains_blocked_until_certification_is_attached() -> None:
    """Current tracked matrix is an integration surface, not a publication-ready claim."""

    report_json = main(["--json"])

    assert report_json == 1


def test_cli_emits_json_report_for_synthetic_matrix(tmp_path, capsys) -> None:
    """CLI JSON output is deterministic enough for downstream wrappers."""

    artifact = tmp_path / "artifact.json"
    source = tmp_path / "source.json"
    artifact.write_text("{}", encoding="utf-8")
    source.write_text("{}", encoding="utf-8")
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(
        json.dumps(
            _matrix(
                [
                    {
                        "row_id": "release_artifact:complete",
                        "classification": "benchmark evidence",
                        "availability_status": "available",
                        "artifact_match": True,
                        "artifact_uri": "artifact.json",
                        "scenario_certification": "accepted",
                        "source_refs": ["source.json"],
                    }
                ]
            )
        ),
        encoding="utf-8",
    )

    assert main(["--matrix", str(matrix_path), "--repo-root", str(tmp_path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == "release-claim-matrix-publication-gate.v1"
    assert payload["status"] == "pass"
