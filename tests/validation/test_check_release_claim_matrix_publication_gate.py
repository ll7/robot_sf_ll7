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


def test_gate_rejects_absolute_artifact_uri_even_when_file_exists(tmp_path) -> None:
    """Benchmark artifacts must be repo-relative durable files."""

    artifact = tmp_path / "absolute-artifact.json"
    source = tmp_path / "docs" / "context" / "source.json"
    artifact.write_text("{}", encoding="utf-8")
    source.parent.mkdir(parents=True)
    source.write_text("{}", encoding="utf-8")

    report = build_gate_report(
        _matrix(
            [
                {
                    "row_id": "release_artifact:absolute",
                    "classification": "benchmark evidence",
                    "availability_status": "available",
                    "artifact_match": True,
                    "artifact_uri": str(artifact),
                    "scenario_certification": "scenario_cert.v1:accepted",
                    "source_refs": ["docs/context/source.json"],
                }
            ]
        ),
        repo_root=tmp_path,
    )

    assert report["status"] == "blocked"
    assert [blocker["check"] for blocker in report["blockers"]] == ["artifact_uri"]


def test_gate_rejects_source_refs_with_parent_directory_components(tmp_path) -> None:
    """Source refs must not escape or normalize around the repository contract."""

    artifact = tmp_path / "docs" / "context" / "evidence" / "artifact.json"
    source = tmp_path / "source.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}", encoding="utf-8")
    source.write_text("{}", encoding="utf-8")

    report = build_gate_report(
        _matrix(
            [
                {
                    "row_id": "release_artifact:traversal-source",
                    "classification": "benchmark evidence",
                    "availability_status": "available",
                    "artifact_match": True,
                    "artifact_uri": "docs/context/evidence/artifact.json",
                    "scenario_certification": "scenario_cert.v1:accepted",
                    "source_refs": ["docs/../source.json"],
                }
            ]
        ),
        repo_root=tmp_path,
    )

    assert report["status"] == "blocked"
    assert [blocker["check"] for blocker in report["blockers"]] == ["source_refs"]


def test_gate_fails_closed_for_dot_artifact_uri_without_crashing(tmp_path) -> None:
    """A ``"."`` path must block rather than raise ``IndexError`` on empty parts."""

    source = tmp_path / "docs" / "source.json"
    source.parent.mkdir(parents=True)
    source.write_text("{}", encoding="utf-8")

    report = build_gate_report(
        _matrix(
            [
                {
                    "row_id": "release_artifact:dot-path",
                    "classification": "benchmark evidence",
                    "availability_status": "available",
                    "artifact_match": True,
                    "artifact_uri": ".",
                    "scenario_certification": "scenario_cert.v1:accepted",
                    "source_refs": ["docs/source.json"],
                }
            ]
        ),
        repo_root=tmp_path,
    )

    assert report["status"] == "blocked"
    assert [blocker["check"] for blocker in report["blockers"]] == ["artifact_uri"]


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


def test_gate_fails_closed_for_unknown_classification(tmp_path) -> None:
    """Unknown classification labels must not silently bypass strict checks."""

    report = build_gate_report(
        _matrix(
            [
                {
                    "row_id": "release_artifact:unknown-classification",
                    "classification": "nominal benchmark evidence",
                    "benchmark_success": False,
                }
            ]
        ),
        repo_root=tmp_path,
    )

    assert report["status"] == "blocked"
    assert report["blockers"] == [
        {
            "row_id": "release_artifact:unknown-classification",
            "check": "classification",
            "severity": "blocker",
            "reason": "unrecognized classification nominal benchmark evidence",
            "next_action": (
                "Use benchmark evidence or a known non-benchmark classification before publication."
            ),
        }
    ]


def test_committed_matrix_remains_blocked_until_certification_is_attached() -> None:
    """Current tracked matrix is an integration surface, not a publication-ready claim."""

    report_json = main(["--json"])

    assert report_json == 1


def test_cli_emits_json_report_for_synthetic_matrix(tmp_path, capsys) -> None:
    """CLI JSON output is deterministic enough for downstream wrappers."""

    artifact = tmp_path / "docs" / "artifact.json"
    source = tmp_path / "docs" / "source.json"
    artifact.parent.mkdir(parents=True)
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
                        "artifact_uri": "docs/artifact.json",
                        "scenario_certification": "scenario_cert.v1:accepted",
                        "source_refs": ["docs/source.json"],
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


def test_gate_blocks_benchmark_row_with_pending_certification(tmp_path) -> None:
    """A benchmark row with scenario_certification='pending' blocks."""

    artifact = tmp_path / "docs" / "artifact.json"
    source = tmp_path / "docs" / "source.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text("{}", encoding="utf-8")
    source.write_text("{}", encoding="utf-8")

    report = build_gate_report(
        _matrix(
            [
                {
                    "row_id": "release_artifact:pending_cert",
                    "classification": "benchmark evidence",
                    "availability_status": "available",
                    "artifact_match": True,
                    "artifact_uri": "docs/artifact.json",
                    "scenario_certification": "pending",
                    "source_refs": ["docs/source.json"],
                }
            ]
        ),
        repo_root=tmp_path,
    )

    assert report["status"] == "blocked"
    assert any(blocker["check"] == "scenario_certification" for blocker in report["blockers"])


def test_gate_blocks_non_durable_prefix_path(tmp_path) -> None:
    """Paths not starting with a durable prefix block."""

    artifact = tmp_path / "other" / "artifact.json"
    source = tmp_path / "docs" / "source.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    source.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text("{}", encoding="utf-8")
    source.write_text("{}", encoding="utf-8")

    report = build_gate_report(
        _matrix(
            [
                {
                    "row_id": "release_artifact:non_durable_path",
                    "classification": "benchmark evidence",
                    "availability_status": "available",
                    "artifact_match": True,
                    "artifact_uri": "other/artifact.json",
                    "scenario_certification": "scenario_cert.v1:accepted",
                    "source_refs": ["docs/source.json"],
                }
            ]
        ),
        repo_root=tmp_path,
    )

    assert report["status"] == "blocked"
    assert any(blocker["check"] == "artifact_uri" for blocker in report["blockers"])
