"""Tests for local route-efficiency report."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.dev import route_efficiency_report as rer


def _compact_artifacts(*, present: bool = True) -> dict[str, dict]:
    """Build a minimal compact_artifacts block for one attempt."""
    return {
        key: {
            "present": present,
            "path": f"run/{key}.txt",
            "reason": None if present else "missing",
        }
        for key in rer._EXPECTED_ARTIFACT_KEYS
    }


def _manifest(
    attempts: list[dict],
    chosen_index: int = 0,
    *,
    chosen_route: dict | None = None,
) -> dict:
    """Build a minimal routed-worker manifest."""
    chosen = chosen_route or attempts[chosen_index].get("route", {})
    return {
        "schema": "routed_worker_manifest.v1",
        "attempted_routes": attempts,
        "chosen_route": chosen,
        "route_evidence_only": True,
        "warning": "route evidence only",
    }


def test_analyze_single_complete_attempt(tmp_path: Path) -> None:
    """A single complete attempt should yield full artifact rate and zero reroutes."""
    manifest = _manifest(
        [
            {
                "attempt_index": 0,
                "route": {"provider": "qwen"},
                "returncode": 0,
                "failure_class": "none",
                "compact_artifacts": _compact_artifacts(present=True),
            }
        ]
    )

    report = rer.analyze_manifests([manifest])

    assert report["manifests_analyzed"] == 1
    assert report["delegated_attempts"] == 1
    assert report["complete_artifacts"]["count"] == 1
    assert report["complete_artifacts"]["rate"] == 1.0
    assert report["reroutes"] == 0
    assert report["estimated_inspections_avoided"] == 1
    assert report["incomplete_by_provider"] == {}
    assert report["incomplete_by_failure_class"] == {}


def test_analyze_tracked_fixture_manifest() -> None:
    """A tracked fixture should exercise the report's documented manifest input."""
    manifest_path = (
        Path(__file__).resolve().parents[1]
        / "fixtures"
        / "route_efficiency"
        / "routing_manifest_complete.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    report = rer.analyze_manifests([manifest])

    assert report["manifests_analyzed"] == 1
    assert report["delegated_attempts"] == 1
    assert report["complete_artifacts"] == {"count": 1, "rate": 1.0}
    assert report["validation_presence"] == {"present": 1, "success_inferable": 1}


def test_analyze_missing_artifacts_counted_as_incomplete() -> None:
    """Attempts with missing artifacts should feed incomplete breakdowns."""
    manifest = _manifest(
        [
            {
                "attempt_index": 0,
                "route": {"provider": "gemini"},
                "returncode": 2,
                "failure_class": "route-collapse",
                "compact_artifacts": _compact_artifacts(present=False),
            },
            {
                "attempt_index": 1,
                "route": {"provider": "qwen"},
                "returncode": 0,
                "failure_class": "none",
                "compact_artifacts": _compact_artifacts(present=True),
            },
        ]
    )

    report = rer.analyze_manifests([manifest])

    assert report["delegated_attempts"] == 2
    assert report["complete_artifacts"]["count"] == 1
    assert report["complete_artifacts"]["rate"] == 0.5
    assert report["reroutes"] == 1  # second attempt is a reroute
    assert report["incomplete_by_provider"]["gemini"] == 1
    assert report["incomplete_by_failure_class"]["route-collapse"] == 1


def test_route_success_vs_task_acceptance_distinction() -> None:
    """A successful route should not imply task acceptance without snapshot."""
    manifest = _manifest(
        [
            {
                "attempt_index": 0,
                "route": {"provider": "qwen"},
                "returncode": 0,
                "failure_class": "none",
                "compact_artifacts": _compact_artifacts(present=True),
            }
        ]
    )

    report = rer.analyze_manifests([manifest])

    assert report["final_outcome"]["accepted"] is None
    assert "not task acceptance" in report["final_outcome"]["note"]
    assert "not task acceptance" in report["warning"]


def test_snapshot_provides_acceptance_metadata() -> None:
    """When a snapshot provides accepted=True, the report surfaces it."""
    manifest = _manifest(
        [
            {
                "attempt_index": 0,
                "route": {"provider": "qwen"},
                "returncode": 0,
                "failure_class": "none",
                "compact_artifacts": _compact_artifacts(present=True),
            }
        ]
    )
    snapshot = {"accepted": True, "outcome": "merged"}

    report = rer.analyze_manifests([manifest], snapshot=snapshot)

    assert report["final_outcome"]["accepted"] is True
    assert report["final_outcome"]["note"] == "merged"


def test_multiple_manifests_aggregate() -> None:
    """Multiple manifests should sum attempts, reroutes, and incomplete counts."""
    m1 = _manifest(
        [
            {
                "attempt_index": 0,
                "route": {"provider": "qwen"},
                "returncode": 0,
                "failure_class": "none",
                "compact_artifacts": _compact_artifacts(present=True),
            }
        ]
    )
    m2 = _manifest(
        [
            {
                "attempt_index": 0,
                "route": {"provider": "gemini"},
                "returncode": 1,
                "failure_class": "timeout",
                "compact_artifacts": _compact_artifacts(present=False),
            },
            {
                "attempt_index": 1,
                "route": {"provider": "qwen"},
                "returncode": 0,
                "failure_class": "none",
                "compact_artifacts": _compact_artifacts(present=True),
            },
        ]
    )

    report = rer.analyze_manifests([m1, m2])

    assert report["manifests_analyzed"] == 2
    assert report["delegated_attempts"] == 3
    assert report["complete_artifacts"]["count"] == 2
    assert report["reroutes"] == 1
    assert report["incomplete_by_provider"]["gemini"] == 1
    assert report["incomplete_by_failure_class"]["timeout"] == 1


def test_empty_manifest_list() -> None:
    """An empty manifest list should produce a zeroed-out report."""
    report = rer.analyze_manifests([])

    assert report["manifests_analyzed"] == 0
    assert report["delegated_attempts"] == 0
    assert report["complete_artifacts"]["rate"] == 0.0
    assert report["estimated_inspections_avoided"] == 0


def test_markdown_output_compactness(tmp_path: Path) -> None:
    """Markdown output should be shorter than JSON and contain key sections."""
    manifest = _manifest(
        [
            {
                "attempt_index": 0,
                "route": {"provider": "qwen"},
                "returncode": 0,
                "failure_class": "none",
                "compact_artifacts": _compact_artifacts(present=True),
            }
        ]
    )

    report = rer.analyze_manifests([manifest])
    md = rer._format_markdown(report)
    js = json.dumps(report, indent=2, sort_keys=True)

    assert "# Route Efficiency Report" in md
    assert "Delegated attempts" in md
    assert "Complete artifacts" in md
    assert "Reroutes" in md
    assert "not task acceptance" in md
    assert len(md) <= len(js)


def test_json_output_includes_schema_and_warning() -> None:
    """JSON output should always include the schema version and evidence warning."""
    report = rer.analyze_manifests([])
    text = json.dumps(report, indent=2, sort_keys=True)

    assert report["schema"] == "route_efficiency_report.v1"
    assert "route evidence only" in report["warning"]
    assert "route_efficiency_report.v1" in text


def test_validation_success_detection() -> None:
    """Validation success should be detected from artifact result text."""
    manifest = _manifest(
        [
            {
                "attempt_index": 0,
                "route": {"provider": "qwen"},
                "returncode": 0,
                "failure_class": "none",
                "compact_artifacts": {
                    **_compact_artifacts(present=True),
                    "validation": {
                        "present": True,
                        "path": "run/validation.txt",
                        "reason": None,
                        "result": "all tests passed",
                    },
                },
            }
        ]
    )

    report = rer.analyze_manifests([manifest])

    assert report["validation_presence"]["present"] == 1
    assert report["validation_presence"]["success_inferable"] == 1


def test_validation_success_detection_ignores_zero_failed() -> None:
    """Common pytest summaries with zero failures should still infer success."""
    manifest = _manifest(
        [
            {
                "attempt_index": 0,
                "route": {"provider": "qwen"},
                "returncode": 0,
                "failure_class": "none",
                "compact_artifacts": {
                    **_compact_artifacts(present=True),
                    "validation": {
                        "present": True,
                        "path": "run/validation.txt",
                        "reason": None,
                        "result": "21 passed, 0 failed",
                    },
                },
            }
        ]
    )

    report = rer.analyze_manifests([manifest])

    assert report["validation_presence"]["present"] == 1
    assert report["validation_presence"]["success_inferable"] == 1


def test_validation_failure_detection() -> None:
    """Validation failure should be detected but not counted as success."""
    manifest = _manifest(
        [
            {
                "attempt_index": 0,
                "route": {"provider": "qwen"},
                "returncode": 0,
                "failure_class": "none",
                "compact_artifacts": {
                    **_compact_artifacts(present=True),
                    "validation": {
                        "present": True,
                        "path": "run/validation.txt",
                        "reason": None,
                        "result": "2 errors found",
                    },
                },
            }
        ]
    )

    report = rer.analyze_manifests([manifest])

    assert report["validation_presence"]["present"] == 1
    assert report["validation_presence"]["success_inferable"] == 0


def test_validation_failure_detection_avoids_false_positive_substrings() -> None:
    """Unsuccessful or zero-passed summaries should not infer validation success."""
    manifest = _manifest(
        [
            {
                "attempt_index": 0,
                "route": {"provider": "qwen"},
                "returncode": 1,
                "failure_class": "validation",
                "compact_artifacts": {
                    **_compact_artifacts(present=True),
                    "validation": {
                        "present": True,
                        "path": "run/validation.txt",
                        "reason": None,
                        "result": "unsuccessful run: 0 passed",
                    },
                },
            }
        ]
    )

    report = rer.analyze_manifests([manifest])

    assert report["validation_presence"]["present"] == 1
    assert report["validation_presence"]["success_inferable"] == 0


def test_defensive_handling_of_missing_fields() -> None:
    """Manifests with missing or malformed fields should not crash the report."""
    # Completely empty manifest dict.
    report = rer.analyze_manifests([{}])
    assert report["delegated_attempts"] == 0

    # Attempt with no compact_artifacts key.
    manifest = _manifest(
        [
            {
                "attempt_index": 0,
                "route": {"provider": "qwen"},
                "returncode": 0,
            }
        ]
    )
    report = rer.analyze_manifests([manifest])
    assert report["delegated_attempts"] == 1
    assert report["complete_artifacts"]["count"] == 0

    # Attempt with non-dict compact_artifacts.
    manifest2 = _manifest(
        [
            {
                "attempt_index": 0,
                "route": {"provider": "qwen"},
                "returncode": 0,
                "compact_artifacts": "not-a-dict",
            }
        ]
    )
    report2 = rer.analyze_manifests([manifest2])
    assert report2["complete_artifacts"]["count"] == 0


def test_cli_json_output(tmp_path: Path) -> None:
    """CLI should produce valid JSON to stdout."""
    manifest = _manifest(
        [
            {
                "attempt_index": 0,
                "route": {"provider": "qwen"},
                "returncode": 0,
                "failure_class": "none",
                "compact_artifacts": _compact_artifacts(present=True),
            }
        ]
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    exit_code = rer.main([str(manifest_path)])
    assert exit_code == 0


def test_cli_writes_output_file(tmp_path: Path) -> None:
    """CLI --output should write the report to a file."""
    manifest = _manifest(
        [
            {
                "attempt_index": 0,
                "route": {"provider": "qwen"},
                "returncode": 0,
                "failure_class": "none",
                "compact_artifacts": _compact_artifacts(present=True),
            }
        ]
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    out_path = tmp_path / "report.json"

    exit_code = rer.main([str(manifest_path), "--output", str(out_path)])
    assert exit_code == 0
    assert out_path.exists()
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["schema"] == "route_efficiency_report.v1"


def test_cli_markdown_output(tmp_path: Path) -> None:
    """CLI --format markdown should produce Markdown text."""
    manifest = _manifest(
        [
            {
                "attempt_index": 0,
                "route": {"provider": "qwen"},
                "returncode": 0,
                "failure_class": "none",
                "compact_artifacts": _compact_artifacts(present=True),
            }
        ]
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    out_path = tmp_path / "report.md"

    exit_code = rer.main([str(manifest_path), "--format", "markdown", "--output", str(out_path)])
    assert exit_code == 0
    text = out_path.read_text(encoding="utf-8")
    assert "# Route Efficiency Report" in text
