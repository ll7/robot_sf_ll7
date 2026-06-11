"""Tests for benchmark publication bundle CLI helper."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from scripts.tools import benchmark_publication_bundle

_EVIDENCE_BUNDLE_SCHEMA = (
    Path(__file__).resolve().parents[2]
    / "robot_sf"
    / "benchmark"
    / "schemas"
    / "evidence_bundle.v1.json"
)


def _write(path: Path, payload: str) -> None:
    """Write UTF-8 payload to disk for test fixtures."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def _make_run(run_dir: Path) -> None:
    """Create a minimal run directory fixture."""
    _write(run_dir / "manifest.json", '{"git_hash":"abc123","scenario_matrix_hash":"m1"}')
    _write(run_dir / "run_meta.json", '{"repo":{"remote":"git@github.com:ll7/robot_sf_ll7.git"}}')
    _write(run_dir / "episodes" / "episodes.jsonl", '{"episode_id":"ep-1"}\n')
    _write(run_dir / "aggregates" / "summary.json", '{"success_rate": 1.0}\n')
    _write(run_dir / "reports" / "report.md", "# report\n")


def test_size_report_command_writes_json(tmp_path: Path, capsys) -> None:
    """CLI size-report should emit and optionally persist a JSON payload."""
    root = tmp_path / "benchmarks"
    _make_run(root / "run_1")
    output_json = tmp_path / "size_report.json"

    exit_code = benchmark_publication_bundle.main(
        [
            "size-report",
            "--benchmarks-root",
            str(root),
            "--output-json",
            str(output_json),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert output_json.exists()
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["run_count"] == 1
    stdout_payload = json.loads(captured.out)
    assert stdout_payload["schema_version"] == payload["schema_version"]


def test_export_command_creates_bundle_artifacts(tmp_path: Path, capsys) -> None:
    """CLI export should produce publication bundle directory and archive."""
    run_dir = tmp_path / "benchmarks" / "run_2"
    _make_run(run_dir)
    out_dir = tmp_path / "publication"

    exit_code = benchmark_publication_bundle.main(
        [
            "export",
            "--run-dir",
            str(run_dir),
            "--out-dir",
            str(out_dir),
            "--bundle-name",
            "run_2_bundle",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    assert Path(payload["bundle_dir"]).exists()
    assert Path(payload["manifest_path"]).exists()
    assert Path(payload["checksums_path"]).exists()
    assert Path(payload["archive_path"]).exists()
    assert payload["file_count"] > 0


def _make_evidence_source(source_root: Path) -> None:
    """Create a compact evidence source fixture."""
    _write(source_root / "summary.json", '{"result_classification":"diagnostic-only"}\n')
    _write(source_root / "metric_table.csv", "metric,value\nsuccess_rate,0.5\n")
    _write(source_root / "trace_manifest.yaml", "trace_count: 1\n")
    _write(source_root / "claim_boundary.md", "# Claim Boundary\n\nDiagnostic only.\n")


def _make_dissertation_source(source_root: Path) -> Path:
    """Create a source fixture for dissertation artifact bundle tests."""
    _write(source_root / "tables" / "campaign_table.md", "| planner | status |\n| --- | --- |\n")
    _write(source_root / "figures" / "planner_status.png", "fake png bytes\n")
    spec_path = source_root / "artifact_spec.json"
    spec = {
        "artifacts": [
            {
                "artifact_id": "tab_campaign_table",
                "source_path": "tables/campaign_table.md",
                "source_artifact": "release-backed campaign table candidate",
                "caption_draft": "Campaign table preserving fallback and degraded rows.",
                "claim_boundary": "Formatted table only; not new benchmark evidence.",
                "recommended_manuscript_use": "discussion",
                "fallback_degraded_summary": "Fallback and degraded rows remain visible.",
            },
            {
                "artifact_id": "fig_planner_status",
                "source_path": "figures/planner_status.png",
                "source_artifact": "planner status summary figure candidate",
                "caption_draft": "Planner status counts for export review.",
                "claim_boundary": "Diagnostic status distribution only.",
                "recommended_manuscript_use": "do-not-use",
                "fallback_degraded_summary": "Do not use as performance evidence.",
            },
        ]
    }
    spec_path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    return spec_path


def test_evidence_bundle_command_creates_manifest_and_checksums(tmp_path: Path, capsys) -> None:
    """CLI evidence-bundle should copy selected compact files with provenance metadata."""
    source_root = tmp_path / "evidence_source"
    _make_evidence_source(source_root)
    out_dir = tmp_path / "bundles"

    exit_code = benchmark_publication_bundle.main(
        [
            "evidence-bundle",
            "--source-root",
            str(source_root),
            "--out-dir",
            str(out_dir),
            "--bundle-name",
            "issue_2460_example",
            "--file",
            "summary.json",
            "--file",
            "metric_table.csv",
            "--file",
            "trace_manifest.yaml",
            "--file",
            "claim_boundary.md",
            "--command",
            "uv run python scripts/example.py --config tracked.yaml",
            "--commit",
            "abc123",
            "--claim-boundary",
            "diagnostic_only_not_benchmark_evidence",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    bundle_dir = Path(payload["bundle_dir"])
    manifest_path = Path(payload["manifest_path"])
    checksums_path = Path(payload["checksums_path"])
    assert bundle_dir.exists()
    assert manifest_path.exists()
    assert checksums_path.exists()
    assert payload["file_count"] == 4
    assert (bundle_dir / "payload" / "summary.json").exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema = json.loads(_EVIDENCE_BUNDLE_SCHEMA.read_text(encoding="utf-8"))
    jsonschema.validate(instance=manifest, schema=schema)
    assert manifest["schema_version"] == "evidence_bundle.v1"
    assert manifest["command"] == "uv run python scripts/example.py --config tracked.yaml"
    assert manifest["commit"] == "abc123"
    assert manifest["claim_boundary"] == "diagnostic_only_not_benchmark_evidence"
    assert manifest["policy"]["large_raw_artifacts"] == "excluded"
    assert [entry["path"] for entry in manifest["files"]] == [
        "claim_boundary.md",
        "metric_table.csv",
        "summary.json",
        "trace_manifest.yaml",
    ]
    checksums = checksums_path.read_text(encoding="utf-8")
    assert "summary.json" in checksums
    assert "metric_table.csv" in checksums


def test_evidence_bundle_command_fails_closed_for_missing_file(tmp_path: Path) -> None:
    """Evidence bundles should fail before writing manifests when a requested file is missing."""
    source_root = tmp_path / "evidence_source"
    _make_evidence_source(source_root)

    with pytest.raises(FileNotFoundError):
        benchmark_publication_bundle.main(
            [
                "evidence-bundle",
                "--source-root",
                str(source_root),
                "--out-dir",
                str(tmp_path / "bundles"),
                "--bundle-name",
                "missing_file",
                "--file",
                "summary.json",
                "--file",
                "missing.json",
                "--command",
                "uv run python scripts/example.py",
                "--commit",
                "abc123",
                "--claim-boundary",
                "diagnostic_only",
            ]
        )


def test_dissertation_bundle_command_creates_artifact_manifest(tmp_path: Path, capsys) -> None:
    """CLI dissertation-bundle should emit manifest rows with provenance and caveats."""
    source_root = tmp_path / "publication_candidates"
    spec_path = _make_dissertation_source(source_root)
    out_dir = tmp_path / "dissertation_export"

    exit_code = benchmark_publication_bundle.main(
        [
            "dissertation-bundle",
            "--source-root",
            str(source_root),
            "--out-dir",
            str(out_dir),
            "--bundle-name",
            "campaign_table_bundle",
            "--artifact-spec",
            str(spec_path),
            "--command",
            "uv run python scripts/tools/compile_benchmark_artifacts.py --campaign-root tracked",
            "--commit",
            "abc123",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    bundle_dir = Path(payload["bundle_dir"])
    manifest_path = Path(payload["manifest_path"])
    checksums_path = Path(payload["checksums_path"])
    assert manifest_path.name == "artifact_manifest.json"
    assert (bundle_dir / "payload" / "artifacts" / "tab_campaign_table.md").exists()
    assert (bundle_dir / "payload" / "artifacts" / "fig_planner_status.png").exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "dissertation_artifact_bundle.v1"
    assert manifest["source_commit"] == "abc123"
    assert manifest["generation_command"].startswith("uv run python")
    assert manifest["policy"]["paper_or_benchmark_claim"] == "not_established_by_bundle_alone"
    rows = {row["artifact_id"]: row for row in manifest["artifacts"]}
    table_row = rows["tab_campaign_table"]
    assert table_row["source_artifact"] == "release-backed campaign table candidate"
    assert table_row["recommended_manuscript_use"] == "discussion"
    assert table_row["fallback_degraded_summary"] == "Fallback and degraded rows remain visible."
    assert table_row["claim_boundary"] == "Formatted table only; not new benchmark evidence."
    assert table_row["sha256"] in checksums_path.read_text(encoding="utf-8")


def test_dissertation_bundle_command_rejects_invalid_manuscript_use(tmp_path: Path) -> None:
    """Dissertation artifact rows should fail closed for unsupported manuscript use labels."""
    source_root = tmp_path / "publication_candidates"
    spec_path = _make_dissertation_source(source_root)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec["artifacts"][0]["recommended_manuscript_use"] = "appendix"
    spec_path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="recommended_manuscript_use"):
        benchmark_publication_bundle.main(
            [
                "dissertation-bundle",
                "--source-root",
                str(source_root),
                "--out-dir",
                str(tmp_path / "dissertation_export"),
                "--bundle-name",
                "bad_use",
                "--artifact-spec",
                str(spec_path),
                "--command",
                "uv run python scripts/tools/compile_benchmark_artifacts.py",
                "--commit",
                "abc123",
            ]
        )


def test_dissertation_bundle_command_rejects_null_spec_fields(tmp_path: Path) -> None:
    """JSON null values should not be coerced into string manifest fields."""
    source_root = tmp_path / "publication_candidates"
    spec_path = _make_dissertation_source(source_root)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec["artifacts"][0]["caption_draft"] = None
    spec_path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="cannot be null"):
        benchmark_publication_bundle.main(
            [
                "dissertation-bundle",
                "--source-root",
                str(source_root),
                "--out-dir",
                str(tmp_path / "dissertation_export"),
                "--bundle-name",
                "null_field",
                "--artifact-spec",
                str(spec_path),
                "--command",
                "uv run python scripts/tools/compile_benchmark_artifacts.py",
                "--commit",
                "abc123",
            ]
        )


def test_dissertation_bundle_command_rejects_unsafe_artifact_id(tmp_path: Path) -> None:
    """Artifact IDs become filenames and should allow only portable safe characters."""
    source_root = tmp_path / "publication_candidates"
    spec_path = _make_dissertation_source(source_root)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec["artifacts"][0]["artifact_id"] = "tab:campaign"
    spec_path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid artifact_id"):
        benchmark_publication_bundle.main(
            [
                "dissertation-bundle",
                "--source-root",
                str(source_root),
                "--out-dir",
                str(tmp_path / "dissertation_export"),
                "--bundle-name",
                "unsafe_id",
                "--artifact-spec",
                str(spec_path),
                "--command",
                "uv run python scripts/tools/compile_benchmark_artifacts.py",
                "--commit",
                "abc123",
            ]
        )


def test_dissertation_bundle_command_fails_closed_for_missing_source(tmp_path: Path) -> None:
    """Dissertation bundle export should not write manifests for missing artifact sources."""
    source_root = tmp_path / "publication_candidates"
    spec_path = _make_dissertation_source(source_root)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec["artifacts"][0]["source_path"] = "tables/missing.md"
    spec_path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        benchmark_publication_bundle.main(
            [
                "dissertation-bundle",
                "--source-root",
                str(source_root),
                "--out-dir",
                str(tmp_path / "dissertation_export"),
                "--bundle-name",
                "missing_source",
                "--artifact-spec",
                str(spec_path),
                "--command",
                "uv run python scripts/tools/compile_benchmark_artifacts.py",
                "--commit",
                "abc123",
            ]
        )


def test_dissertation_bundle_command_rejects_symlink_source(tmp_path: Path) -> None:
    """Dissertation bundle export should reject symlink payload sources before resolving paths."""
    source_root = tmp_path / "publication_candidates"
    spec_path = _make_dissertation_source(source_root)
    target = source_root / "tables" / "campaign_table.md"
    link = source_root / "tables" / "campaign_table_link.md"
    link.symlink_to(target)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec["artifacts"][0]["source_path"] = "tables/campaign_table_link.md"
    spec_path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="refuses symlink"):
        benchmark_publication_bundle.main(
            [
                "dissertation-bundle",
                "--source-root",
                str(source_root),
                "--out-dir",
                str(tmp_path / "dissertation_export"),
                "--bundle-name",
                "symlink_source",
                "--artifact-spec",
                str(spec_path),
                "--command",
                "uv run python scripts/tools/compile_benchmark_artifacts.py",
                "--commit",
                "abc123",
            ]
        )


def test_dissertation_bundle_overwrite_replaces_existing_file_path(
    tmp_path: Path,
    capsys,
) -> None:
    """Overwrite should replace a colliding file path instead of calling rmtree on it."""
    source_root = tmp_path / "publication_candidates"
    spec_path = _make_dissertation_source(source_root)
    out_dir = tmp_path / "dissertation_export"
    out_dir.mkdir()
    (out_dir / "file_collision").write_text("old payload\n", encoding="utf-8")

    exit_code = benchmark_publication_bundle.main(
        [
            "dissertation-bundle",
            "--source-root",
            str(source_root),
            "--out-dir",
            str(out_dir),
            "--bundle-name",
            "file_collision",
            "--artifact-spec",
            str(spec_path),
            "--command",
            "uv run python scripts/tools/compile_benchmark_artifacts.py",
            "--commit",
            "abc123",
            "--overwrite",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    assert Path(payload["bundle_dir"]).is_dir()
    assert Path(payload["manifest_path"]).name == "artifact_manifest.json"
