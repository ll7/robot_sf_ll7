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


def test_snapshot_preserves_explicit_falsy_metadata() -> None:
    """Explicit falsy snapshot values should not be replaced by defaults."""
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
    snapshot = {"accepted": 0, "outcome": ""}

    report = rer.analyze_manifests([manifest], snapshot=snapshot)

    assert report["final_outcome"]["accepted"] is False
    assert report["final_outcome"]["note"] == ""


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

    # Non-dict manifests and malformed attempted_routes are ignored defensively.
    report = rer.analyze_manifests([[], {"attempted_routes": "not-a-list"}])  # type: ignore[list-item]
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

    # Non-dict attempts are counted as malformed incomplete attempts.
    manifest3 = {"attempted_routes": ["not-a-dict"]}
    report3 = rer.analyze_manifests([manifest3])  # type: ignore[list-item]
    assert report3["delegated_attempts"] == 1
    assert report3["incomplete_by_failure_class"]["malformed"] == 1


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


def test_cli_rejects_malformed_manifest_json(tmp_path: Path) -> None:
    """CLI loading should reject non-object manifest payloads."""
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(["not-a-dict"]), encoding="utf-8")

    try:
        rer.main([str(manifest_path)])
    except ValueError as exc:
        assert "manifest list entries" in str(exc)
    else:  # pragma: no cover - keeps the failure message clear.
        raise AssertionError("expected malformed manifest list to raise ValueError")


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


def test_routing_recommendations_no_recommendation_single_attempt() -> None:
    """Single attempt should yield no_recommendation due to insufficient data."""
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

    recs = report["routing_recommendations"]
    assert len(recs) == 1
    assert recs[0]["class"] == "no_recommendation"
    assert "insufficient data" in recs[0]["action"]
    assert "not task acceptance" in recs[0]["caveat"]


def test_routing_recommendations_prefer_provider() -> None:
    """Provider with 100% completion and >= 2 attempts should trigger prefer_provider."""
    manifest = _manifest(
        [
            {
                "attempt_index": 0,
                "route": {"provider": "qwen"},
                "returncode": 0,
                "failure_class": "none",
                "compact_artifacts": _compact_artifacts(present=True),
            },
            {
                "attempt_index": 1,
                "route": {"provider": "qwen"},
                "returncode": 0,
                "failure_class": "none",
                "compact_artifacts": _compact_artifacts(present=True),
            },
            {
                "attempt_index": 2,
                "route": {"provider": "gemini"},
                "returncode": 1,
                "failure_class": "timeout",
                "compact_artifacts": _compact_artifacts(present=False),
            },
        ]
    )

    report = rer.analyze_manifests([manifest])

    recs = report["routing_recommendations"]
    prefer = [r for r in recs if r["class"] == "prefer_provider"]
    assert len(prefer) >= 1
    assert "qwen" in prefer[0]["action"]
    assert "2/2" in prefer[0]["action"]
    assert "not task acceptance" in prefer[0]["caveat"]


def test_routing_recommendations_avoid_provider() -> None:
    """Provider with 0% completion should trigger avoid_provider."""
    manifest = _manifest(
        [
            {
                "attempt_index": 0,
                "route": {"provider": "qwen"},
                "returncode": 0,
                "failure_class": "none",
                "compact_artifacts": _compact_artifacts(present=True),
            },
            {
                "attempt_index": 1,
                "route": {"provider": "gemini"},
                "returncode": 1,
                "failure_class": "timeout",
                "compact_artifacts": _compact_artifacts(present=False),
            },
            {
                "attempt_index": 2,
                "route": {"provider": "gemini"},
                "returncode": 1,
                "failure_class": "timeout",
                "compact_artifacts": _compact_artifacts(present=False),
            },
        ]
    )

    report = rer.analyze_manifests([manifest])

    recs = report["routing_recommendations"]
    avoid = [r for r in recs if r["class"] == "avoid_provider"]
    assert len(avoid) >= 1
    assert "gemini" in avoid[0]["action"]
    assert "0/2" in avoid[0]["action"]
    assert "not task acceptance" in avoid[0]["caveat"]


def test_routing_recommendations_avoid_provider_requires_two_attempts() -> None:
    """A single failed attempt should not produce a provider avoidance recommendation."""
    manifest = _manifest(
        [
            {
                "attempt_index": 0,
                "route": {"provider": "qwen"},
                "returncode": 0,
                "failure_class": "none",
                "compact_artifacts": _compact_artifacts(present=True),
            },
            {
                "attempt_index": 1,
                "route": {"provider": "gemini"},
                "returncode": 1,
                "failure_class": "timeout",
                "compact_artifacts": _compact_artifacts(present=False),
            },
        ]
    )

    report = rer.analyze_manifests([manifest])

    recs = report["routing_recommendations"]
    assert not [r for r in recs if r["class"] == "avoid_provider"]
    assert any(r["class"] == "investigate_failure_class" for r in recs)


def test_routing_recommendations_investigate_failure_class() -> None:
    """Dominant failure class should trigger investigate_failure_class."""
    manifest = _manifest(
        [
            {
                "attempt_index": 0,
                "route": {"provider": "qwen"},
                "returncode": 0,
                "failure_class": "none",
                "compact_artifacts": _compact_artifacts(present=True),
            },
            {
                "attempt_index": 1,
                "route": {"provider": "gemini"},
                "returncode": 1,
                "failure_class": "timeout",
                "compact_artifacts": _compact_artifacts(present=False),
            },
            {
                "attempt_index": 2,
                "route": {"provider": "copilot"},
                "returncode": 1,
                "failure_class": "timeout",
                "compact_artifacts": _compact_artifacts(present=False),
            },
            {
                "attempt_index": 3,
                "route": {"provider": "claude"},
                "returncode": 1,
                "failure_class": "other",
                "compact_artifacts": _compact_artifacts(present=False),
            },
        ]
    )

    report = rer.analyze_manifests([manifest])

    recs = report["routing_recommendations"]
    invest = [r for r in recs if r["class"] == "investigate_failure_class"]
    assert len(invest) >= 1
    assert "timeout" in invest[0]["action"]
    assert "not task acceptance" in invest[0]["caveat"]


def test_routing_recommendations_reroute_threshold_met() -> None:
    """Completion rate below 50% should trigger reroute_threshold_met."""
    manifest = _manifest(
        [
            {
                "attempt_index": 0,
                "route": {"provider": "qwen"},
                "returncode": 0,
                "failure_class": "none",
                "compact_artifacts": _compact_artifacts(present=True),
            },
            {
                "attempt_index": 1,
                "route": {"provider": "gemini"},
                "returncode": 1,
                "failure_class": "timeout",
                "compact_artifacts": _compact_artifacts(present=False),
            },
            {
                "attempt_index": 2,
                "route": {"provider": "copilot"},
                "returncode": 1,
                "failure_class": "validation",
                "compact_artifacts": _compact_artifacts(present=False),
            },
        ]
    )

    report = rer.analyze_manifests([manifest])

    recs = report["routing_recommendations"]
    reroute = [r for r in recs if r["class"] == "reroute_threshold_met"]
    assert len(reroute) >= 1
    assert "1/3" in reroute[0]["action"]
    assert "not task acceptance" in reroute[0]["caveat"]


def test_routing_recommendations_no_recommendation_mixed() -> None:
    """Mixed results with no dominant pattern should yield no_recommendation."""
    manifest = _manifest(
        [
            {
                "attempt_index": 0,
                "route": {"provider": "qwen"},
                "returncode": 0,
                "failure_class": "none",
                "compact_artifacts": _compact_artifacts(present=True),
            },
            {
                "attempt_index": 1,
                "route": {"provider": "gemini"},
                "returncode": 0,
                "failure_class": "none",
                "compact_artifacts": _compact_artifacts(present=True),
            },
        ]
    )

    report = rer.analyze_manifests([manifest])

    recs = report["routing_recommendations"]
    assert len(recs) == 1
    assert recs[0]["class"] == "no_recommendation"
    assert "insufficient data" in recs[0]["action"]


def test_routing_recommendations_entry_structure() -> None:
    """Each recommendation entry must have class, action, evidence, caveat keys."""
    manifest = _manifest(
        [
            {
                "attempt_index": 0,
                "route": {"provider": "qwen"},
                "returncode": 0,
                "failure_class": "none",
                "compact_artifacts": _compact_artifacts(present=True),
            },
            {
                "attempt_index": 1,
                "route": {"provider": "gemini"},
                "returncode": 1,
                "failure_class": "timeout",
                "compact_artifacts": _compact_artifacts(present=False),
            },
            {
                "attempt_index": 2,
                "route": {"provider": "gemini"},
                "returncode": 1,
                "failure_class": "timeout",
                "compact_artifacts": _compact_artifacts(present=False),
            },
        ]
    )

    report = rer.analyze_manifests([manifest])

    for rec in report["routing_recommendations"]:
        assert "class" in rec
        assert "action" in rec
        assert "evidence" in rec
        assert "caveat" in rec
        assert "not task acceptance" in rec["caveat"]


def test_markdown_includes_routing_recommendations_section() -> None:
    """Markdown output should include Routing recommendations section when present."""
    manifest = _manifest(
        [
            {
                "attempt_index": 0,
                "route": {"provider": "qwen"},
                "returncode": 0,
                "failure_class": "none",
                "compact_artifacts": _compact_artifacts(present=True),
            },
            {
                "attempt_index": 1,
                "route": {"provider": "gemini"},
                "returncode": 1,
                "failure_class": "timeout",
                "compact_artifacts": _compact_artifacts(present=False),
            },
            {
                "attempt_index": 2,
                "route": {"provider": "gemini"},
                "returncode": 1,
                "failure_class": "timeout",
                "compact_artifacts": _compact_artifacts(present=False),
            },
        ]
    )

    report = rer.analyze_manifests([manifest])
    md = rer._format_markdown(report)

    assert "## Routing recommendations" in md
    assert "avoid_provider" in md
    assert "not task acceptance" in md


_FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "route_efficiency"


def _load_fixture(name: str) -> dict:
    """Load a fixture JSON file from tests/fixtures/route_efficiency/."""
    return json.loads((_FIXTURE_DIR / name).read_text(encoding="utf-8"))


def _manifest_with_source(name: str) -> tuple[dict, str]:
    """Load a fixture and return (manifest, source_label)."""
    manifest = _load_fixture(name)
    return manifest, name


def test_dashboard_single_manifest_equals_single_report() -> None:
    """Dashboard with one manifest should produce consistent overall metrics."""
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

    dashboard = rer.analyze_dashboard([(manifest, "single.json")])
    report = rer.analyze_manifests([manifest])

    assert dashboard["schema"] == "route_efficiency_dashboard.v1"
    assert dashboard["manifests_analyzed"] == 1
    assert dashboard["overall"]["delegated_attempts"] == report["delegated_attempts"]
    assert dashboard["overall"]["complete_artifacts"] == report["complete_artifacts"]
    assert dashboard["overall"]["total_reroutes"] == report["reroutes"]
    assert len(dashboard["per_manifest"]) == 1
    assert dashboard["per_manifest"][0]["source"] == "single.json"
    assert "warning" in dashboard
    assert "not task acceptance" in dashboard["warning"]


def test_dashboard_multiple_manifests_per_manifest_breakdown() -> None:
    """Dashboard with multiple manifests should include per-manifest breakdown."""
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

    dashboard = rer.analyze_dashboard([(m1, "m1.json"), (m2, "m2.json")])

    assert dashboard["manifests_analyzed"] == 2
    assert len(dashboard["per_manifest"]) == 2
    assert dashboard["per_manifest"][0]["source"] == "m1.json"
    assert dashboard["per_manifest"][1]["source"] == "m2.json"
    assert dashboard["per_manifest"][0]["delegated_attempts"] == 1
    assert dashboard["per_manifest"][1]["delegated_attempts"] == 2
    assert dashboard["per_manifest"][0]["complete_artifacts"]["rate"] == 1.0
    assert dashboard["per_manifest"][1]["complete_artifacts"]["rate"] == 0.5


def test_dashboard_provider_trend_time_series() -> None:
    """Dashboard should expose per-provider trend across manifests."""
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
                "route": {"provider": "qwen"},
                "returncode": 0,
                "failure_class": "none",
                "compact_artifacts": _compact_artifacts(present=True),
            },
            {
                "attempt_index": 1,
                "route": {"provider": "gemini"},
                "returncode": 1,
                "failure_class": "timeout",
                "compact_artifacts": _compact_artifacts(present=False),
            },
        ]
    )

    dashboard = rer.analyze_dashboard([(m1, "m1.json"), (m2, "m2.json")])

    trend = dashboard["provider_trend"]
    qwen_trend = [t for t in trend if t["provider"] == "qwen"]
    gemini_trend = [t for t in trend if t["provider"] == "gemini"]
    assert len(qwen_trend) == 1
    assert len(qwen_trend[0]["per_manifest"]) == 2
    assert qwen_trend[0]["per_manifest"][0]["source"] == "m1.json"
    assert qwen_trend[0]["per_manifest"][0]["complete"] == 1
    assert qwen_trend[0]["per_manifest"][1]["source"] == "m2.json"
    assert qwen_trend[0]["per_manifest"][1]["complete"] == 1
    assert len(gemini_trend) == 1
    assert gemini_trend[0]["per_manifest"][0]["source"] == "m2.json"
    assert gemini_trend[0]["per_manifest"][0]["complete"] == 0


def test_dashboard_overall_metrics_match_aggregate() -> None:
    """Dashboard overall metrics should match sum of per-manifest attempts."""
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

    dashboard = rer.analyze_dashboard([(m1, "m1.json"), (m2, "m2.json")])

    overall = dashboard["overall"]
    assert overall["delegated_attempts"] == 3
    assert overall["complete_artifacts"]["count"] == 2
    assert overall["complete_artifacts"]["rate"] == round(2 / 3, 4)
    assert overall["total_reroutes"] == 1
    assert overall["estimated_inspections_avoided"] == 2

    per_total = sum(p["delegated_attempts"] for p in dashboard["per_manifest"])
    per_complete = sum(p["complete_artifacts"]["count"] for p in dashboard["per_manifest"])
    assert overall["delegated_attempts"] == per_total
    assert overall["complete_artifacts"]["count"] == per_complete


def test_dashboard_empty_manifest_list() -> None:
    """Dashboard with empty manifest list should produce zeroed-out report."""
    dashboard = rer.analyze_dashboard([])

    assert dashboard["manifests_analyzed"] == 0
    assert dashboard["overall"]["delegated_attempts"] == 0
    assert dashboard["overall"]["complete_artifacts"]["rate"] == 0.0
    assert dashboard["overall"]["estimated_inspections_avoided"] == 0
    assert dashboard["per_manifest"] == []
    assert dashboard["provider_trend"] == []


def test_dashboard_markdown_output_compactness() -> None:
    """Dashboard markdown should be shorter than JSON and contain key sections."""
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
        ]
    )

    dashboard = rer.analyze_dashboard([(m1, "m1.json"), (m2, "m2.json")])
    md = rer._format_dashboard_markdown(dashboard)
    js = json.dumps(dashboard, indent=2, sort_keys=True)

    assert "# Route Efficiency Dashboard" in md
    assert "Delegated attempts" in md
    assert "Complete artifacts" in md
    assert "Per-manifest breakdown" in md
    assert "Provider trend" in md
    assert "not task acceptance" in md
    assert len(md) <= len(js)


def test_dashboard_cli_json_output(tmp_path: Path) -> None:
    """CLI --dashboard should produce valid JSON dashboard."""
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
        ]
    )
    p1 = tmp_path / "m1.json"
    p2 = tmp_path / "m2.json"
    p1.write_text(json.dumps(m1), encoding="utf-8")
    p2.write_text(json.dumps(m2), encoding="utf-8")

    exit_code = rer.main(["--dashboard", str(p1), str(p2)])
    assert exit_code == 0


def test_dashboard_cli_markdown_output(tmp_path: Path) -> None:
    """CLI --dashboard --format markdown should produce dashboard markdown."""
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
    p1 = tmp_path / "m1.json"
    p1.write_text(json.dumps(m1), encoding="utf-8")
    out_path = tmp_path / "dashboard.md"

    exit_code = rer.main(
        ["--dashboard", "--format", "markdown", "--output", str(out_path), str(p1)]
    )
    assert exit_code == 0
    text = out_path.read_text(encoding="utf-8")
    assert "# Route Efficiency Dashboard" in text
    assert "Per-manifest breakdown" in text


def test_dashboard_no_raw_logs_or_secrets() -> None:
    """Dashboard output should not contain raw logs, secrets, or quota fields."""
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
    dashboard = rer.analyze_dashboard([(m1, "m1.json")])
    text = json.dumps(dashboard, indent=2, sort_keys=True)

    forbidden = ["api_key", "secret", "token", "quota", "raw_log", "stdout", "stderr"]
    for key in forbidden:
        assert key not in text.lower()


def test_dashboard_incomplete_failure_visibility() -> None:
    """Dashboard should surface incomplete and failure class trends."""
    m1 = _manifest(
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
                "route": {"provider": "copilot"},
                "returncode": 1,
                "failure_class": "timeout",
                "compact_artifacts": _compact_artifacts(present=False),
            },
        ]
    )
    m2 = _manifest(
        [
            {
                "attempt_index": 0,
                "route": {"provider": "gemini"},
                "returncode": 1,
                "failure_class": "validation",
                "compact_artifacts": _compact_artifacts(present=False),
            },
        ]
    )

    dashboard = rer.analyze_dashboard([(m1, "m1.json"), (m2, "m2.json")])

    assert "timeout" in dashboard["incomplete_by_failure_class"]
    assert dashboard["incomplete_by_failure_class"]["timeout"] == 2
    assert "validation" in dashboard["incomplete_by_failure_class"]
    assert dashboard["incomplete_by_failure_class"]["validation"] == 1
    assert dashboard["incomplete_by_provider"]["gemini"] == 2
    assert dashboard["incomplete_by_provider"]["copilot"] == 1
    assert dashboard["missing_artifacts"]["result_json"] == 3
    assert dashboard["missing_artifacts"]["validation"] == 3


def test_dashboard_routing_recommendations_have_caveats() -> None:
    """Dashboard routing recommendations should include route-evidence-only caveats."""
    m1 = _manifest(
        [
            {
                "attempt_index": 0,
                "route": {"provider": "qwen"},
                "returncode": 0,
                "failure_class": "none",
                "compact_artifacts": _compact_artifacts(present=True),
            },
            {
                "attempt_index": 1,
                "route": {"provider": "gemini"},
                "returncode": 1,
                "failure_class": "timeout",
                "compact_artifacts": _compact_artifacts(present=False),
            },
            {
                "attempt_index": 2,
                "route": {"provider": "gemini"},
                "returncode": 1,
                "failure_class": "timeout",
                "compact_artifacts": _compact_artifacts(present=False),
            },
        ]
    )

    dashboard = rer.analyze_dashboard([(m1, "m1.json")])
    recs = dashboard["routing_recommendations"]
    assert len(recs) >= 1
    for rec in recs:
        assert "not task acceptance" in rec["caveat"]


def test_dashboard_with_fixture_files() -> None:
    """Dashboard should load and analyze real fixture files correctly."""
    fixtures = [
        "historical_manifest_20260610.json",
        "historical_manifest_20260612.json",
        "historical_manifest_20260614.json",
    ]
    manifests_with_sources = [_manifest_with_source(f) for f in fixtures]

    dashboard = rer.analyze_dashboard(manifests_with_sources)

    assert dashboard["manifests_analyzed"] == 3
    assert dashboard["overall"]["delegated_attempts"] == 6
    assert dashboard["missing_artifacts"]["diffstat"] == 2
    assert len(dashboard["per_manifest"]) == 3
    assert len(dashboard["provider_trend"]) >= 2
    assert "qwen" in [t["provider"] for t in dashboard["provider_trend"]]
    assert "gemini" in [t["provider"] for t in dashboard["provider_trend"]]


def test_dashboard_markdown_includes_recommendations_section() -> None:
    """Dashboard markdown should include routing recommendations when present."""
    m1 = _manifest(
        [
            {
                "attempt_index": 0,
                "route": {"provider": "qwen"},
                "returncode": 0,
                "failure_class": "none",
                "compact_artifacts": _compact_artifacts(present=True),
            },
            {
                "attempt_index": 1,
                "route": {"provider": "gemini"},
                "returncode": 1,
                "failure_class": "timeout",
                "compact_artifacts": _compact_artifacts(present=False),
            },
            {
                "attempt_index": 2,
                "route": {"provider": "gemini"},
                "returncode": 1,
                "failure_class": "timeout",
                "compact_artifacts": _compact_artifacts(present=False),
            },
        ]
    )

    dashboard = rer.analyze_dashboard([(m1, "m1.json")])
    md = rer._format_dashboard_markdown(dashboard)

    assert "## Routing recommendations" in md
    assert "avoid_provider" in md
    assert "not task acceptance" in md


# ---------------------------------------------------------------------------
# Negative-fixture tests (issue-2817): ensure the report cannot overinterpret
# bad delegated outputs.
# ---------------------------------------------------------------------------


def test_negative_fixture_all_attempts_failed() -> None:
    """All attempts failed should yield zero completion and clear missing counts."""
    manifest = _load_fixture("negative_all_attempts_failed.json")
    report = rer.analyze_manifests([manifest])

    assert report["delegated_attempts"] == 3
    assert report["complete_artifacts"]["count"] == 0
    assert report["complete_artifacts"]["rate"] == 0.0
    assert report["validation_presence"]["present"] == 0
    assert report["validation_presence"]["success_inferable"] == 0
    assert report["reroutes"] == 2
    assert report["incomplete_by_provider"]["gemini"] == 2
    assert report["incomplete_by_provider"]["copilot"] == 1
    assert report["incomplete_by_failure_class"]["timeout"] == 2
    assert report["incomplete_by_failure_class"]["route-collapse"] == 1
    expected_missing = len(rer._EXPECTED_ARTIFACT_KEYS) * 3
    assert sum(report["missing_artifacts"].values()) == expected_missing


def test_negative_fixture_partial_artifacts_not_complete() -> None:
    """Partial artifacts should not be counted as complete."""
    manifest = _load_fixture("negative_partial_artifacts.json")
    report = rer.analyze_manifests([manifest])

    assert report["delegated_attempts"] == 2
    assert report["complete_artifacts"]["count"] == 0
    assert report["complete_artifacts"]["rate"] == 0.0
    assert report["validation_presence"]["present"] == 1
    assert report["validation_presence"]["success_inferable"] == 0
    assert report["incomplete_by_failure_class"]["partial-artifacts"] == 1
    assert report["incomplete_by_failure_class"]["validation"] == 1


def test_negative_fixture_malformed_compact_artifacts() -> None:
    """Malformed compact_artifacts should be counted as incomplete, not crash."""
    manifest = _load_fixture("negative_malformed_compact_artifacts.json")
    report = rer.analyze_manifests([manifest])

    assert report["delegated_attempts"] == 3
    assert report["complete_artifacts"]["count"] == 1
    assert report["complete_artifacts"]["rate"] == round(1 / 3, 4)
    assert report["incomplete_by_provider"]["qwen"] == 1
    assert report["incomplete_by_provider"]["gemini"] == 1
    assert report["incomplete_by_failure_class"]["malformed"] == 2


def test_negative_fixture_empty_routes() -> None:
    """Empty attempted_routes should yield zeroed report without error."""
    manifest = _load_fixture("negative_empty_routes.json")
    report = rer.analyze_manifests([manifest])

    assert report["delegated_attempts"] == 0
    assert report["complete_artifacts"]["count"] == 0
    assert report["complete_artifacts"]["rate"] == 0.0
    assert report["incomplete_by_provider"] == {}
    assert report["incomplete_by_failure_class"] == {}
    assert report["missing_artifacts"] == {}


def test_negative_fixture_validation_failure_with_complete_artifacts() -> None:
    """Complete artifacts with failing validation should not count as success."""
    manifest = _load_fixture("negative_validation_failure_complete_artifacts.json")
    report = rer.analyze_manifests([manifest])

    assert report["delegated_attempts"] == 1
    assert report["complete_artifacts"]["count"] == 1
    assert report["complete_artifacts"]["rate"] == 1.0
    assert report["validation_presence"]["present"] == 1
    assert report["validation_presence"]["success_inferable"] == 0


def test_negative_fixture_single_attempt_all_missing() -> None:
    """Single attempt with all artifacts missing should show full missing counts."""
    manifest = _load_fixture("negative_single_attempt_all_missing.json")
    report = rer.analyze_manifests([manifest])

    assert report["delegated_attempts"] == 1
    assert report["complete_artifacts"]["count"] == 0
    assert report["complete_artifacts"]["rate"] == 0.0
    assert report["incomplete_by_provider"]["gemini"] == 1
    assert report["incomplete_by_failure_class"]["route-collapse"] == 1
    for key in rer._EXPECTED_ARTIFACT_KEYS:
        assert report["missing_artifacts"][key] == 1


def test_negative_fixture_reroute_success_not_accepted() -> None:
    """A reroute can complete validation while still lacking task acceptance."""
    manifest = _load_fixture("negative_reroute_success_not_accepted.json")
    report = rer.analyze_manifests([manifest])

    assert report["delegated_attempts"] == 2
    assert report["complete_artifacts"] == {"count": 1, "rate": 0.5}
    assert report["validation_presence"] == {"present": 1, "success_inferable": 1}
    assert report["reroutes"] == 1
    assert report["incomplete_by_provider"]["gemini"] == 1
    assert report["incomplete_by_failure_class"]["missing-artifact-paths"] == 1
    assert report["by_provider"]["qwen"] == {"total": 1, "complete": 1}
    for key in rer._EXPECTED_ARTIFACT_KEYS:
        assert report["missing_artifacts"][key] == 1
    assert report["final_outcome"]["accepted"] is None
    assert "not task acceptance" in report["final_outcome"]["note"]
    assert "not task acceptance" in report["warning"]


def test_negative_fixture_reroute_threshold_triggered() -> None:
    """All-failed manifest should trigger reroute_threshold_met recommendation."""
    manifest = _load_fixture("negative_all_attempts_failed.json")
    report = rer.analyze_manifests([manifest])

    recs = report["routing_recommendations"]
    reroute = [r for r in recs if r["class"] == "reroute_threshold_met"]
    assert len(reroute) >= 1
    assert "0/3" in reroute[0]["action"]
    assert "not task acceptance" in reroute[0]["caveat"]


def test_negative_fixture_avoid_provider_triggered() -> None:
    """Provider with 0% completion across >= 2 attempts should trigger avoid."""
    manifest = _load_fixture("negative_all_attempts_failed.json")
    report = rer.analyze_manifests([manifest])

    recs = report["routing_recommendations"]
    avoid = [r for r in recs if r["class"] == "avoid_provider"]
    assert len(avoid) >= 1
    assert "gemini" in avoid[0]["action"]
    assert "0/2" in avoid[0]["action"]


def test_negative_fixture_dashboard_all_failed() -> None:
    """Dashboard with all-failed fixture should surface zero completion."""
    manifest, source = _manifest_with_source("negative_all_attempts_failed.json")
    dashboard = rer.analyze_dashboard([(manifest, source)])

    assert dashboard["overall"]["delegated_attempts"] == 3
    assert dashboard["overall"]["complete_artifacts"]["count"] == 0
    assert dashboard["overall"]["complete_artifacts"]["rate"] == 0.0
    assert len(dashboard["per_manifest"]) == 1
    assert dashboard["per_manifest"][0]["source"] == source
    assert dashboard["per_manifest"][0]["complete_artifacts"]["rate"] == 0.0


def test_negative_fixture_dashboard_mixed_with_positive() -> None:
    """Dashboard mixing positive and negative fixtures should aggregate correctly."""
    pos, src_pos = _manifest_with_source("routing_manifest_complete.json")
    neg, src_neg = _manifest_with_source("negative_all_attempts_failed.json")
    dashboard = rer.analyze_dashboard([(pos, src_pos), (neg, src_neg)])

    assert dashboard["overall"]["delegated_attempts"] == 4
    assert dashboard["overall"]["complete_artifacts"]["count"] == 1
    assert dashboard["overall"]["complete_artifacts"]["rate"] == 0.25
    assert len(dashboard["per_manifest"]) == 2
    assert dashboard["per_manifest"][0]["complete_artifacts"]["rate"] == 1.0
    assert dashboard["per_manifest"][1]["complete_artifacts"]["rate"] == 0.0
