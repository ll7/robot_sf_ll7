"""Tests for the issue #3294 release claim matrix assembler."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.analysis.assemble_release_claim_matrix_issue_3294 import (
    DEFAULT_ARTIFACT_MANIFEST,
    DEFAULT_LEADERBOARD_GLOB,
    DEFAULT_ODD_COVERAGE,
    DEFAULT_RELEASE_CONFIG,
    DEFAULT_RELEASE_SNAPSHOT,
    DEFAULT_SCENARIO_CERTIFICATION_SUMMARY,
    SourcePaths,
    _scenario_certification_prerequisites,
    _scenario_certification_status,
    build_matrix,
    main,
    render_markdown,
)


def _default_sources() -> SourcePaths:
    """Return default source paths used by the production CLI."""
    return SourcePaths(
        release_snapshot=DEFAULT_RELEASE_SNAPSHOT,
        artifact_manifest=DEFAULT_ARTIFACT_MANIFEST,
        release_config=DEFAULT_RELEASE_CONFIG,
        odd_coverage=DEFAULT_ODD_COVERAGE,
        scenario_certification_summary=DEFAULT_SCENARIO_CERTIFICATION_SUMMARY,
        leaderboard_sidecars=tuple(sorted(Path().glob(DEFAULT_LEADERBOARD_GLOB))),
    )


def test_build_matrix_uses_tracked_sources_and_fail_closed_classifications() -> None:
    """The matrix should assemble release, leaderboard, and ODD rows without output/ evidence."""
    matrix = build_matrix(_default_sources())

    assert matrix["schema_version"] == "release_claim_matrix_issue_3294.v1"
    assert matrix["summary"]["row_count"] >= 10
    assert matrix["summary"]["classification_counts"]["benchmark evidence"] >= 1
    assert matrix["summary"]["classification_counts"]["diagnostic evidence"] >= 1
    assert matrix["summary"]["classification_counts"]["non-claim"] >= 1
    assert all(not str(row["artifact_uri"]).startswith("output/") for row in matrix["rows"])

    release_rows = [row for row in matrix["rows"] if row["section"] == "release_artifact"]
    assert release_rows
    assert all(
        row["scenario_certification"] == "scenario_cert.v1:accepted_reviewed"
        for row in release_rows
    )
    assert all(
        row["publication_suite_policy"]["policy_status"] == "applied" for row in release_rows
    )
    assert all(not row["missing_prerequisites"] for row in release_rows)
    assert all(
        DEFAULT_SCENARIO_CERTIFICATION_SUMMARY.as_posix() in row["source_refs"]
        for row in release_rows
    )

    odd_rows = [row for row in matrix["rows"] if row["section"] == "odd_hazard_coverage"]
    assert odd_rows
    assert all(row["benchmark_success"] is False for row in odd_rows)
    assert any(row["scenario_certification"] == "not_available" for row in odd_rows)


def test_cli_writes_json_and_markdown(tmp_path: Path, monkeypatch) -> None:
    """CLI should write a reviewable JSON matrix and Markdown table."""
    output_dir = tmp_path / "matrix"
    monkeypatch.setattr(
        "sys.argv",
        [
            "assemble_release_claim_matrix_issue_3294.py",
            "--output-dir",
            output_dir.as_posix(),
        ],
    )

    assert main() == 0

    matrix_path = output_dir / "release_claim_matrix.json"
    markdown_path = output_dir / "release_claim_matrix.md"
    payload = json.loads(matrix_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")
    assert payload["summary"]["row_count"] >= 10
    assert "Claim boundary:" in markdown
    assert "fallback" in markdown.lower() or "degraded" in markdown.lower()


def test_render_markdown_keeps_matrix_table_rows_single_line() -> None:
    """Multiline source caveats should not split the Markdown table."""
    markdown = render_markdown(build_matrix(_default_sources()))
    assert "scenario_cert.v1 summary is not publication-accepted" not in markdown
    table_started = False
    for line in markdown.splitlines():
        if line.startswith("| Row |"):
            table_started = True
        if not table_started:
            continue
        if line == "":
            break
        assert line.startswith("| ")


def test_release_matrix_without_publication_policy_remains_blocked() -> None:
    """Without the v0.1 suite policy, scenario certification still fails closed."""

    sources = SourcePaths(
        release_snapshot=DEFAULT_RELEASE_SNAPSHOT,
        artifact_manifest=DEFAULT_ARTIFACT_MANIFEST,
        release_config=DEFAULT_RELEASE_CONFIG,
        odd_coverage=DEFAULT_ODD_COVERAGE,
        scenario_certification_summary=DEFAULT_SCENARIO_CERTIFICATION_SUMMARY,
        publication_suite_policy=None,
        leaderboard_sidecars=tuple(sorted(Path().glob(DEFAULT_LEADERBOARD_GLOB))),
    )
    matrix = build_matrix(sources)
    release_rows = [row for row in matrix["rows"] if row["section"] == "release_artifact"]

    assert release_rows
    assert all(row["scenario_certification"] == "scenario_cert.v1:blocked" for row in release_rows)
    assert all(row["missing_prerequisites"] for row in release_rows)


def test_scenario_certification_status_fails_closed_on_null_and_missing_counts() -> None:
    """Explicit-null status and absent counts must never certify as accepted."""
    # Explicit null publication_gate_status must not coerce to the string "None".
    assert (
        _scenario_certification_status({"publication_gate_status": None})
        == "scenario_cert.v1:blocked"
    )
    # Missing eligibility counts must fail closed (no vacuous accepted).
    assert _scenario_certification_status({}) == "scenario_cert.v1:blocked"
    # Incomplete counts (only one key present) must also fail closed.
    assert (
        _scenario_certification_status({"benchmark_eligibility_counts": {"excluded": 0}})
        == "scenario_cert.v1:blocked"
    )
    # Complete, all-zero counts certify accepted.
    assert (
        _scenario_certification_status(
            {"benchmark_eligibility_counts": {"excluded": 0, "stress_only": 0}}
        )
        == "scenario_cert.v1:accepted"
    )


def test_scenario_certification_prerequisites_normalizes_null_and_missing() -> None:
    """Null blocker must not surface as "None"; missing counts report unavailable."""
    # Explicit-null blocker with missing counts: no "None", explicit unavailable text.
    prereqs = _scenario_certification_prerequisites({"publication_blocker": None})
    assert prereqs == [
        "scenario_cert.v1 summary is not publication-accepted "
        "(benchmark_eligibility_counts missing or incomplete)"
    ]
    # A real blocker string passes through verbatim.
    assert _scenario_certification_prerequisites(
        {"publication_blocker": "2 excluded scenarios remain"}
    ) == ["2 excluded scenarios remain"]
