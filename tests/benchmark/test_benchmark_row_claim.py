"""Tests for benchmark row claim validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from robot_sf.benchmark.benchmark_row_claim import (
    BENCHMARK_ROW_CLAIM_SCHEMA_VERSION,
    BenchmarkRowClaimError,
    load_benchmark_row_claim_schema,
    validate_all_leaderboards,
    validate_benchmark_row_claim,
    validate_leaderboard_claims,
)
from robot_sf.benchmark.cli import cli_main

FIXTURE_ROOT = Path("tests/fixtures/benchmark_row_claim/v1")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    """Write a JSON fixture file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _valid_claim() -> dict[str, object]:
    """Return a minimal valid row claim payload."""
    return {
        "schema_version": BENCHMARK_ROW_CLAIM_SCHEMA_VERSION,
        "suite_id": "test_suite",
        "planner_id": "test_planner",
        "planner_mode": "native",
        "seeds": [1, 2, 3],
        "metrics": {"success": 0.5, "collision": 0.0},
        "row_status": "successful_evidence",
        "exclusions": ["test_fixture"],
        "artifact_uri": "docs/leaderboards/README.md",
        "claim_wording": "Native test row with tracked evidence.",
        "evidence_tier": "benchmark",
        "claim_boundary": "Test fixture only; no benchmark claim.",
    }


def test_load_schema_matches_expected_version() -> None:
    """The loaded schema should declare the expected version constant."""
    schema = load_benchmark_row_claim_schema()
    assert schema["$id"].endswith("benchmark_row_claim.v1.json")
    assert schema["properties"]["schema_version"]["const"] == BENCHMARK_ROW_CLAIM_SCHEMA_VERSION


def test_valid_claim_is_accepted(tmp_path: Path) -> None:
    """A claim that satisfies schema, artifact, and wording policy is accepted."""
    claim = _valid_claim()
    result = validate_benchmark_row_claim(claim)
    assert result["schema_version"] == BENCHMARK_ROW_CLAIM_SCHEMA_VERSION
    assert result["planner_id"] == "test_planner"


def test_missing_required_field_is_rejected() -> None:
    """Schema validation should reject a claim with a missing required field."""
    claim = _valid_claim()
    del claim["planner_mode"]
    with pytest.raises(BenchmarkRowClaimError):
        validate_benchmark_row_claim(claim)


def test_output_artifact_uri_is_rejected(tmp_path: Path) -> None:
    """Worktree-local output/ artifact URIs must fail closed."""
    claim = _valid_claim()
    claim["artifact_uri"] = "output/benchmarks/camera_ready/episodes.jsonl"
    with pytest.raises(BenchmarkRowClaimError, match="output/"):
        validate_benchmark_row_claim(claim)


def test_artifact_uri_resolving_into_output_is_rejected(tmp_path: Path) -> None:
    """Relative traversal must not bypass the output/ path guard."""
    artifact = tmp_path / "output" / "evidence.json"
    _write_json(artifact, {"schema_version": "test.v1"})
    claim = _valid_claim()
    claim["artifact_uri"] = "docs/../output/evidence.json"
    with pytest.raises(BenchmarkRowClaimError, match="output/"):
        validate_benchmark_row_claim(claim, repo_root=tmp_path)


def test_artifact_uri_outside_repo_is_rejected(tmp_path: Path) -> None:
    """Relative traversal must not escape the repository root."""
    outside = tmp_path.parent / "outside-evidence.json"
    _write_json(outside, {"schema_version": "test.v1"})
    claim = _valid_claim()
    claim["artifact_uri"] = "../outside-evidence.json"
    with pytest.raises(BenchmarkRowClaimError, match="within the repository"):
        validate_benchmark_row_claim(claim, repo_root=tmp_path)


def test_missing_artifact_is_rejected(tmp_path: Path) -> None:
    """A referenced artifact that does not exist must be rejected."""
    claim = _valid_claim()
    claim["artifact_uri"] = "docs/leaderboards/does_not_exist.rows.json"
    with pytest.raises(BenchmarkRowClaimError, match="not found"):
        validate_benchmark_row_claim(claim)


def test_artifact_uri_symlink_is_rejected(tmp_path: Path) -> None:
    """A sidecar artifact pointer must not resolve through a direct symlink."""
    target = tmp_path / "evidence.json"
    _write_json(target, {"schema_version": "test.v1"})
    link = tmp_path / "linked-evidence.json"
    link.symlink_to(target)
    claim = _valid_claim()
    claim["artifact_uri"] = str(link.relative_to(tmp_path))
    with pytest.raises(BenchmarkRowClaimError, match="symlink"):
        validate_benchmark_row_claim(claim, repo_root=tmp_path)


def test_fallback_mode_with_success_status_is_rejected() -> None:
    """Fallback mode cannot be classified as successful evidence."""
    claim = _valid_claim()
    claim["planner_mode"] = "fallback"
    with pytest.raises(BenchmarkRowClaimError, match="fallback"):
        validate_benchmark_row_claim(claim)


def test_degraded_mode_with_success_status_is_rejected() -> None:
    """Degraded mode cannot be classified as successful evidence."""
    claim = _valid_claim()
    claim["planner_mode"] = "degraded"
    with pytest.raises(BenchmarkRowClaimError, match="degraded"):
        validate_benchmark_row_claim(claim)


def test_superlative_wording_on_diagnostic_tier_is_rejected() -> None:
    """Superlative/ranking wording is only allowed for paper-facing successful rows."""
    claim = _valid_claim()
    claim["evidence_tier"] = "diagnostic"
    claim["claim_wording"] = "Planner outperforms all baselines on this suite."
    with pytest.raises(BenchmarkRowClaimError, match=r"outperforms|superlative|ranking"):
        validate_benchmark_row_claim(claim)


def test_paper_facing_success_wording_is_accepted(tmp_path: Path) -> None:
    """Paper-facing successful rows may use comparative wording."""
    claim = _valid_claim()
    claim["evidence_tier"] = "paper_facing"
    claim["claim_wording"] = "Planner outperforms the baseline on the benchmark matrix."
    validate_benchmark_row_claim(claim)


def test_proof_wording_on_non_success_status_is_rejected() -> None:
    """Non-success rows cannot use proof or benchmark-readiness wording."""
    claim = _valid_claim()
    claim["row_status"] = "completed_smoke_not_benchmark_evidence"
    claim["evidence_tier"] = "smoke"
    claim["claim_wording"] = "Proves benchmark readiness for the planner."
    with pytest.raises(BenchmarkRowClaimError, match="proof/success wording"):
        validate_benchmark_row_claim(claim)


def test_empty_claim_wording_is_rejected() -> None:
    """An empty claim wording must be rejected."""
    claim = _valid_claim()
    claim["claim_wording"] = "   "
    with pytest.raises(BenchmarkRowClaimError, match="claim_wording"):
        validate_benchmark_row_claim(claim)


def test_leaderboard_sidecar_with_bad_row_is_rejected(tmp_path: Path) -> None:
    """A sidecar containing one bad row reports that row as rejected."""
    artifact = tmp_path / "good_artifact.json"
    _write_json(artifact, {"schema_version": "test.v1"})
    bad_claim = _valid_claim()
    bad_claim["artifact_uri"] = str(artifact.relative_to(tmp_path))

    sidecar = tmp_path / "test.rows.json"
    _write_json(
        sidecar,
        {
            "schema_version": BENCHMARK_ROW_CLAIM_SCHEMA_VERSION,
            "rows": [bad_claim],
        },
    )

    report = validate_leaderboard_claims(sidecar, repo_root=tmp_path)
    assert report["accepted"] == 1
    assert report["rejected"] == 0
    assert report["valid"] is True

    bad_claim["artifact_uri"] = "output/missing.jsonl"
    _write_json(
        sidecar,
        {
            "schema_version": BENCHMARK_ROW_CLAIM_SCHEMA_VERSION,
            "rows": [bad_claim],
        },
    )
    report = validate_leaderboard_claims(sidecar, repo_root=tmp_path)
    assert report["rejected"] == 1
    assert report["valid"] is False


def test_leaderboard_markdown_must_match_sidecar_claim(tmp_path: Path) -> None:
    """Visible Markdown rows must not drift from sidecar row claims."""
    artifact = tmp_path / "artifact.json"
    _write_json(artifact, {"schema_version": "test.v1"})
    claim = _valid_claim()
    claim["artifact_uri"] = str(artifact.relative_to(tmp_path))
    sidecar = tmp_path / "demo.rows.json"
    _write_json(
        sidecar,
        {
            "schema_version": BENCHMARK_ROW_CLAIM_SCHEMA_VERSION,
            "rows": [claim],
        },
    )
    sidecar.with_name("demo.md").write_text(
        "\n".join(
            [
                "| planner | suite | success | collision | near_miss | low_progress | "
                "min_distance | runtime | benchmark_track | evidence_uri | status | "
                "claim_boundary |",
                "| --- | --- | ---: | ---: | ---: | --- | --- | --- | --- | --- | --- | --- |",
                "| `test_planner` | `test_suite` | `0.5` | `0.0` | `0.0` | `0` | "
                "`not_recorded` | `1s` | `test` | [`artifact.json`](artifact.json) | "
                "`successful_evidence` | Overstated visible claim. |",
            ]
        ),
        encoding="utf-8",
    )

    report = validate_leaderboard_claims(sidecar, repo_root=tmp_path)

    assert report["valid"] is False
    assert "Markdown claim_wording" in report["errors"][0]["error"]


def test_leaderboard_markdown_path_uses_rows_json_stem(tmp_path: Path) -> None:
    """A foo.rows.json sidecar should be checked against foo.md, not foo.rows.md."""
    artifact = tmp_path / "artifact.json"
    _write_json(artifact, {"schema_version": "test.v1"})
    claim = _valid_claim()
    claim["artifact_uri"] = str(artifact.relative_to(tmp_path))
    sidecar = tmp_path / "demo.rows.json"
    _write_json(
        sidecar,
        {
            "schema_version": BENCHMARK_ROW_CLAIM_SCHEMA_VERSION,
            "rows": [claim],
        },
    )
    sidecar.with_name("demo.md").write_text(
        "\n".join(
            [
                "| planner | suite | success | collision | near_miss | low_progress | "
                "min_distance | runtime | benchmark_track | evidence_uri | status | "
                "claim_boundary |",
                "| --- | --- | ---: | ---: | ---: | --- | --- | --- | --- | --- | --- | --- |",
                "| `test_planner` | `test_suite` | `0.5` | `0.0` | `0.0` | `0` | "
                "`not_recorded` | `1s` | `test` | [`artifact.json`](artifact.json) | "
                "`successful_evidence` | Native test row with tracked evidence. |",
            ]
        ),
        encoding="utf-8",
    )

    report = validate_leaderboard_claims(sidecar, repo_root=tmp_path)

    assert report["valid"] is True


def test_cli_validate_all_leaderboards_passes() -> None:
    """The CLI --all flag should validate every tracked leaderboard sidecar."""
    exit_code = cli_main(["validate-row-claims", "--all"])
    assert exit_code == 0


def test_cli_validate_single_sidecar_passes() -> None:
    """The CLI should accept a single valid leaderboard sidecar."""
    exit_code = cli_main(["validate-row-claims", "--sidecar", "docs/leaderboards/smoke.rows.json"])
    assert exit_code == 0


def test_cli_rejects_invalid_sidecar(tmp_path: Path) -> None:
    """The CLI should return non-zero for a sidecar with an invalid claim."""
    sidecar = tmp_path / "bad.rows.json"
    _write_json(
        sidecar,
        {
            "schema_version": BENCHMARK_ROW_CLAIM_SCHEMA_VERSION,
            "rows": [
                {
                    "schema_version": BENCHMARK_ROW_CLAIM_SCHEMA_VERSION,
                    "suite_id": "bad",
                    "planner_id": "bad",
                    "planner_mode": "fallback",
                    "seeds": [],
                    "metrics": {},
                    "row_status": "successful_evidence",
                    "exclusions": ["invalid_fixture"],
                    "artifact_uri": "docs/leaderboards/README.md",
                    "claim_wording": "Bad claim.",
                    "evidence_tier": "smoke",
                }
            ],
        },
    )
    exit_code = cli_main(["validate-row-claims", "--sidecar", str(sidecar)])
    assert exit_code == 2


def test_validate_all_leaderboards_finds_sidecars(tmp_path: Path) -> None:
    """validate_all_leaderboards should discover and validate *.rows.json files."""
    artifact = tmp_path / "artifact.json"
    _write_json(artifact, {"schema_version": "test.v1"})
    sidecar = tmp_path / "demo.rows.json"
    claim = _valid_claim()
    claim["artifact_uri"] = str(artifact.relative_to(tmp_path))
    _write_json(
        sidecar,
        {
            "schema_version": BENCHMARK_ROW_CLAIM_SCHEMA_VERSION,
            "rows": [claim],
        },
    )

    report = validate_all_leaderboards(leaderboards_dir=tmp_path, repo_root=tmp_path)
    assert report["overall_valid"] is True
    assert report["leaderboards_checked"] == 1


def test_validate_all_leaderboards_reports_failure(tmp_path: Path) -> None:
    """validate_all_leaderboards should report overall invalid when a sidecar fails."""
    sidecar = tmp_path / "demo.rows.json"
    _write_json(
        sidecar,
        {
            "schema_version": BENCHMARK_ROW_CLAIM_SCHEMA_VERSION,
            "rows": [
                {
                    "schema_version": BENCHMARK_ROW_CLAIM_SCHEMA_VERSION,
                    "suite_id": "bad",
                    "planner_id": "bad",
                    "planner_mode": "fallback",
                    "seeds": [],
                    "metrics": {},
                    "row_status": "successful_evidence",
                    "exclusions": ["invalid_fixture"],
                    "artifact_uri": "docs/leaderboards/README.md",
                    "claim_wording": "Bad claim.",
                    "evidence_tier": "smoke",
                }
            ],
        },
    )

    report = validate_all_leaderboards(leaderboards_dir=tmp_path, repo_root=tmp_path)
    assert report["overall_valid"] is False
    assert report["leaderboards_checked"] == 1
