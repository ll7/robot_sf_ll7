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
                "chapter_target": "Results, Section 4.2",
                "metadata": {
                    "table_label": "tab:campaign-table",
                    "source_release_tag": "0.0.2",
                },
            },
            {
                "artifact_id": "fig_planner_status",
                "source_path": "figures/planner_status.png",
                "source_artifact": "planner status summary figure candidate",
                "caption_draft": "Planner status counts for export review.",
                "claim_boundary": "Diagnostic status distribution only.",
                "recommended_manuscript_use": "do-not-use",
                "fallback_degraded_summary": "Do not use as performance evidence.",
                "chapter_target": "Limitations, Section 5.1",
                "chapter_target_justification": "Used to contextualize diagnostic limits.",
            },
        ]
    }
    spec_path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    return spec_path


def _make_claims_matrix_fixtures(tmp_path: Path) -> tuple[Path, Path]:
    """Create mock dissertation and evidence bundle directories for claims matrix tests."""
    # Dissertation Bundle 1
    diss_bundle_1_root = tmp_path / "diss_bundle_1"
    diss_bundle_1_root.mkdir(parents=True, exist_ok=True)
    diss_bundle_1_payload_dir = diss_bundle_1_root / "payload" / "artifacts"
    diss_bundle_1_payload_dir.mkdir(parents=True, exist_ok=True)
    _write(
        diss_bundle_1_payload_dir / "tab_campaign_table.md", "| planner | status |\n| --- | --- |\n"
    )
    _write(
        diss_bundle_1_payload_dir / "fig_planner_status.png", "fake png bytes for planner status\n"
    )
    _write(
        diss_bundle_1_payload_dir / "fig_diagnostic_plot.png",
        "fake png bytes for diagnostic plot\n",
    )
    _write(
        diss_bundle_1_payload_dir / "fig_missing_checksum.png",
        "fake png bytes for missing checksum\n",
    )  # This file exists, but checksum will be missing in manifest
    _write(
        diss_bundle_1_payload_dir / "fig_missing_source_path.png",
        "fake png bytes for missing source path\n",
    )  # This file exists, but source path will be missing in manifest
    table_hash = benchmark_publication_bundle._sha256_file(
        diss_bundle_1_payload_dir / "tab_campaign_table.md"
    )
    status_hash = benchmark_publication_bundle._sha256_file(
        diss_bundle_1_payload_dir / "fig_planner_status.png"
    )
    diagnostic_hash = benchmark_publication_bundle._sha256_file(
        diss_bundle_1_payload_dir / "fig_diagnostic_plot.png"
    )
    missing_checksum_hash = benchmark_publication_bundle._sha256_file(
        diss_bundle_1_payload_dir / "fig_missing_checksum.png"
    )
    missing_source_hash = benchmark_publication_bundle._sha256_file(
        diss_bundle_1_payload_dir / "fig_missing_source_path.png"
    )

    diss_bundle_1_spec = {
        "schema_version": "dissertation_artifact_bundle.v1",
        "source_commit": "abc123",
        "generation_command": "test-command-1",
        "artifacts": [
            {
                "artifact_id": "tab_campaign_table_bundle1",
                "source_path": "src/tables/campaign_table.md",
                "source_artifact": "Campaign Table 1",
                "caption_draft": "Caption for Campaign Table 1.",
                "claim_boundary": "benchmark-facing, strong claim",
                "recommended_manuscript_use": "results",
                "fallback_degraded_summary": "No degradation.",
                "sha256": table_hash,
                "output_path": "artifacts/tab_campaign_table.md",
            },
            {
                "artifact_id": "fig_planner_status_bundle1",
                "source_path": "src/figures/planner_status.png",
                "source_artifact": "Planner Status Fig 1",
                "caption_draft": "Caption for Planner Status Fig 1.",
                "claim_boundary": "diagnostic-only",
                "recommended_manuscript_use": "discussion",
                "fallback_degraded_summary": "Degraded due to data issues.",
                "sha256": status_hash,
                "output_path": "artifacts/fig_planner_status.png",
            },
            {
                "artifact_id": "fig_diagnostic_plot_bundle1",
                "source_path": "src/figures/diagnostic.png",
                "source_artifact": "Diagnostic Plot 1",
                "caption_draft": "Caption for Diagnostic Plot 1.",
                "claim_boundary": "diagnostic-only",
                "recommended_manuscript_use": "results",  # Invalid promotion
                "fallback_degraded_summary": "Internal use only.",
                "sha256": diagnostic_hash,
                "output_path": "artifacts/fig_diagnostic_plot.png",
            },
            {
                "artifact_id": "fig_missing_checksum_bundle1",
                "source_path": "src/figures/missing.png",
                "source_artifact": "Missing Checksum 1",
                "caption_draft": "Caption for Missing Checksum 1.",
                "claim_boundary": "valid",
                "recommended_manuscript_use": "discussion",
                "fallback_degraded_summary": "No degradation.",
                "sha256": "",  # Missing checksum
                "output_path": "artifacts/fig_missing_checksum.png",
            },
            {
                "artifact_id": "fig_missing_source_path_bundle1",
                "source_path": "",  # Missing source path
                "source_artifact": "Missing Source Path 1",
                "caption_draft": "Caption for Missing Source Path 1.",
                "claim_boundary": "valid",
                "recommended_manuscript_use": "discussion",
                "fallback_degraded_summary": "No degradation.",
                "sha256": missing_source_hash,
                "output_path": "artifacts/fig_missing_source_path.png",
            },
        ],
    }
    _write(
        diss_bundle_1_root / "artifact_manifest.json",
        json.dumps(diss_bundle_1_spec, indent=2) + "\n",
    )
    _write(
        diss_bundle_1_root / "checksums.sha256",
        f"{table_hash}  artifacts/tab_campaign_table.md\n"
        f"{status_hash}  artifacts/fig_planner_status.png\n"
        f"{diagnostic_hash}  artifacts/fig_diagnostic_plot.png\n"
        f"{missing_checksum_hash}  artifacts/fig_missing_checksum.png\n"
        f"{missing_source_hash}  artifacts/fig_missing_source_path.png\n",
    )

    # Evidence Bundle 1
    evidence_bundle_1_root = tmp_path / "evidence_bundle_1"
    evidence_bundle_1_root.mkdir(parents=True, exist_ok=True)
    evidence_bundle_1_spec = {
        "schema_version": "evidence_bundle.v1",
        "command": "test-evidence-command-1",
        "commit": "def456",
        "claim_boundary": "strong evidence, paper-grade",
        "files": [],  # Not strictly needed for this test
    }
    _write(
        evidence_bundle_1_root / "evidence_bundle_manifest.json",
        json.dumps(evidence_bundle_1_spec, indent=2) + "\n",
    )

    return diss_bundle_1_root, evidence_bundle_1_root


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


def test_evidence_bundle_command_writes_dry_run_mirror_manifest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Dry-run mirroring should record deterministic remote asset pointers without upload."""
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
            "issue_3468_dry_run",
            "--file",
            "summary.json",
            "--command",
            "uv run python scripts/example.py --config tracked.yaml",
            "--commit",
            "abc123",
            "--claim-boundary",
            "diagnostic_only_not_benchmark_evidence",
            "--mirror-dry-run-base-uri",
            "s3://example-bucket/evidence",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    payload = json.loads(captured.out)
    mirror_manifest_path = Path(payload["mirror_manifest_path"])
    mirror_manifest = json.loads(mirror_manifest_path.read_text(encoding="utf-8"))
    asset = mirror_manifest["assets"][0]

    assert mirror_manifest["schema_version"] == "evidence_bundle_mirror.v1"
    assert mirror_manifest["backend"] == "dry_run_uri"
    assert mirror_manifest["mode"] == "dry_run"
    assert mirror_manifest["credentials"] == "not_recorded"
    assert asset["path"] == "summary.json"
    assert asset["source_path"] == "payload/summary.json"
    assert asset["size_bytes"] == len('{"result_classification":"diagnostic-only"}\n')
    assert asset["kind"] == "aggregates"
    assert asset["mime_type"] == "application/json"
    assert asset["remote_uri"] == "s3://example-bucket/evidence/payload/summary.json"
    assert asset["upload_status"] == "dry_run"


def test_evidence_bundle_command_copies_local_mirror_backend(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Local mirror backend provides credential-free upload-path coverage."""
    source_root = tmp_path / "evidence_source"
    _make_evidence_source(source_root)
    out_dir = tmp_path / "bundles"
    mirror_dir = tmp_path / "mirror"

    exit_code = benchmark_publication_bundle.main(
        [
            "evidence-bundle",
            "--source-root",
            str(source_root),
            "--out-dir",
            str(out_dir),
            "--bundle-name",
            "issue_3468_local",
            "--file",
            "metric_table.csv",
            "--command",
            "uv run python scripts/example.py --config tracked.yaml",
            "--commit",
            "abc123",
            "--claim-boundary",
            "diagnostic_only_not_benchmark_evidence",
            "--mirror-local-dir",
            str(mirror_dir),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    payload = json.loads(captured.out)
    mirrored_file = mirror_dir / "issue_3468_local" / "payload" / "metric_table.csv"
    assert mirrored_file.read_text(encoding="utf-8") == "metric,value\nsuccess_rate,0.5\n"
    mirror_manifest = json.loads(Path(payload["mirror_manifest_path"]).read_text(encoding="utf-8"))
    asset = mirror_manifest["assets"][0]
    assert asset["upload_status"] == "uploaded"
    assert asset["remote_uri"] == mirrored_file.resolve().as_uri()
    assert asset["mime_type"] == "text/csv"


def test_evidence_bundle_command_fails_closed_for_invalid_mirror_local_dir(
    tmp_path: Path,
) -> None:
    """Invalid local mirror targets should fail closed with an actionable error."""
    source_root = tmp_path / "evidence_source"
    _make_evidence_source(source_root)
    invalid_mirror = tmp_path / "mirror-is-file"
    invalid_mirror.write_text("not a directory", encoding="utf-8")

    with pytest.raises(ValueError, match="local_dir is not a directory"):
        benchmark_publication_bundle.main(
            [
                "evidence-bundle",
                "--source-root",
                str(source_root),
                "--out-dir",
                str(tmp_path / "bundles"),
                "--bundle-name",
                "issue_3468_invalid",
                "--file",
                "summary.json",
                "--command",
                "uv run python scripts/example.py --config tracked.yaml",
                "--commit",
                "abc123",
                "--claim-boundary",
                "diagnostic_only_not_benchmark_evidence",
                "--mirror-local-dir",
                str(invalid_mirror),
            ]
        )


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
    assert table_row["metadata"] == {
        "table_label": "tab:campaign-table",
        "source_release_tag": "0.0.2",
    }
    assert table_row["chapter_target"] == "Results, Section 4.2"
    assert rows["fig_planner_status"]["chapter_target"] == "Limitations, Section 5.1"
    assert (
        rows["fig_planner_status"]["chapter_target_justification"]
        == "Used to contextualize diagnostic limits."
    )
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


def test_dissertation_bundle_command_rejects_justification_without_chapter_target(
    tmp_path: Path,
) -> None:
    """Chapter-target justifications require a chapter target to justify."""
    source_root = tmp_path / "publication_candidates"
    spec_path = _make_dissertation_source(source_root)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec["artifacts"][0].pop("chapter_target")
    spec["artifacts"][0]["chapter_target_justification"] = "No target is present."
    spec_path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="without chapter_target"):
        benchmark_publication_bundle.main(
            [
                "dissertation-bundle",
                "--source-root",
                str(source_root),
                "--out-dir",
                str(tmp_path / "dissertation_export"),
                "--bundle-name",
                "missing_target",
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


def test_validate_dissertation_bundle_command_checks_checksums_and_metadata(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Validation should pass with expected source commit and command checks."""
    source_root = tmp_path / "publication_candidates"
    spec_path = _make_dissertation_source(source_root)
    out_dir = tmp_path / "dissertation_export"
    command = "uv run python scripts/tools/compile_benchmark_artifacts.py --campaign-root tracked"
    bundle_name = "campaign_table_bundle"
    benchmark_publication_bundle.main(
        [
            "dissertation-bundle",
            "--source-root",
            str(source_root),
            "--out-dir",
            str(out_dir),
            "--bundle-name",
            bundle_name,
            "--artifact-spec",
            str(spec_path),
            "--command",
            command,
            "--commit",
            "abc123",
        ]
    )
    # keep only the validation output.
    capsys.readouterr()

    validate_exit = benchmark_publication_bundle.main(
        [
            "validate-dissertation-bundle",
            "--bundle-dir",
            str(out_dir / bundle_name),
            "--source-root",
            str(source_root),
            "--expected-source-command",
            command,
            "--expected-source-commit",
            "abc123",
        ]
    )
    captured = capsys.readouterr().out

    assert validate_exit == 0
    payload = json.loads(captured)
    assert payload["artifact_count"] == 2
    assert payload["source_commit"] == "abc123"


def test_validate_dissertation_bundle_command_fails_for_checksum_mismatch(
    tmp_path: Path,
) -> None:
    """Checksum drift in copied payload files should fail validation."""
    source_root = tmp_path / "publication_candidates"
    spec_path = _make_dissertation_source(source_root)
    out_dir = tmp_path / "dissertation_export"
    command = "uv run python scripts/tools/compile_benchmark_artifacts.py --campaign-root tracked"
    bundle_name = "campaign_table_bundle"
    benchmark_publication_bundle.main(
        [
            "dissertation-bundle",
            "--source-root",
            str(source_root),
            "--out-dir",
            str(out_dir),
            "--bundle-name",
            bundle_name,
            "--artifact-spec",
            str(spec_path),
            "--command",
            command,
            "--commit",
            "abc123",
        ]
    )
    payload_path = out_dir / bundle_name / "payload" / "artifacts" / "tab_campaign_table.md"
    payload_path.write_text(payload_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Payload checksum mismatch"):
        benchmark_publication_bundle.main(
            [
                "validate-dissertation-bundle",
                "--bundle-dir",
                str(out_dir / bundle_name),
                "--source-root",
                str(source_root),
                "--expected-source-command",
                command,
                "--expected-source-commit",
                "abc123",
            ]
        )


def test_validate_dissertation_bundle_command_rejects_escaped_source_path(
    tmp_path: Path,
) -> None:
    """Source paths containing traversal segments should fail closed after resolution."""
    source_root = tmp_path / "publication_candidates"
    spec_path = _make_dissertation_source(source_root)
    out_dir = tmp_path / "dissertation_export"
    command = "uv run python scripts/tools/compile_benchmark_artifacts.py --campaign-root tracked"
    bundle_name = "campaign_table_bundle"
    benchmark_publication_bundle.main(
        [
            "dissertation-bundle",
            "--source-root",
            str(source_root),
            "--out-dir",
            str(out_dir),
            "--bundle-name",
            bundle_name,
            "--artifact-spec",
            str(spec_path),
            "--command",
            command,
            "--commit",
            "abc123",
        ]
    )
    manifest_path = out_dir / bundle_name / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for row in manifest["artifacts"]:
        if row["artifact_id"] == "tab_campaign_table":
            row["source_path"] = "../outside.md"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Source path escapes source root"):
        benchmark_publication_bundle.main(
            [
                "validate-dissertation-bundle",
                "--bundle-dir",
                str(out_dir / bundle_name),
                "--source-root",
                str(source_root),
                "--expected-source-command",
                command,
                "--expected-source-commit",
                "abc123",
            ]
        )


def test_validate_dissertation_bundle_command_rejects_invalid_manuscript_use_results_on_weak_boundary(
    tmp_path: Path,
) -> None:
    """Weak claim boundaries must not be promoted to 'results' manuscript usage."""
    source_root = tmp_path / "publication_candidates"
    spec_path = _make_dissertation_source(source_root)
    out_dir = tmp_path / "dissertation_export"
    command = "uv run python scripts/tools/compile_benchmark_artifacts.py --campaign-root tracked"
    bundle_name = "campaign_table_bundle"
    benchmark_publication_bundle.main(
        [
            "dissertation-bundle",
            "--source-root",
            str(source_root),
            "--out-dir",
            str(out_dir),
            "--bundle-name",
            bundle_name,
            "--artifact-spec",
            str(spec_path),
            "--command",
            command,
            "--commit",
            "abc123",
        ]
    )
    manifest_path = out_dir / bundle_name / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"][0]["recommended_manuscript_use"] = "results"
    manifest["artifacts"][0]["claim_boundary"] = (
        "Diagnostic-only results only; historical evidence."
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="promotes diagnostic"):
        benchmark_publication_bundle.main(
            [
                "validate-dissertation-bundle",
                "--bundle-dir",
                str(out_dir / bundle_name),
                "--source-root",
                str(source_root),
                "--expected-source-command",
                command,
                "--expected-source-commit",
                "abc123",
            ]
        )


def test_validate_dissertation_bundle_command_rejects_reference_claim_boundary_weakening(
    tmp_path: Path,
) -> None:
    """Reference manifests should catch weakened claim boundaries during validation."""
    source_root = tmp_path / "publication_candidates"
    spec_path = _make_dissertation_source(source_root)
    out_dir = tmp_path / "dissertation_export"
    command = "uv run python scripts/tools/compile_benchmark_artifacts.py --campaign-root tracked"
    bundle_name = "campaign_table_bundle"
    benchmark_publication_bundle.main(
        [
            "dissertation-bundle",
            "--source-root",
            str(source_root),
            "--out-dir",
            str(out_dir),
            "--bundle-name",
            bundle_name,
            "--artifact-spec",
            str(spec_path),
            "--command",
            command,
            "--commit",
            "abc123",
        ]
    )
    manifest_path = out_dir / bundle_name / "artifact_manifest.json"
    reference_manifest_path = out_dir / "reference_artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    reference_manifest = dict(manifest)
    reference_manifest["artifacts"] = [dict(row) for row in manifest["artifacts"]]
    for row in reference_manifest["artifacts"]:
        if row["artifact_id"] == "tab_campaign_table":
            row["claim_boundary"] = "Benchmark result verified."
    reference_manifest_path.write_text(
        json.dumps(reference_manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    for row in manifest["artifacts"]:
        if row["artifact_id"] == "tab_campaign_table":
            row["claim_boundary"] = "Diagnostic-only exploratory result."
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Claim boundary weakened"):
        benchmark_publication_bundle.main(
            [
                "validate-dissertation-bundle",
                "--bundle-dir",
                str(out_dir / bundle_name),
                "--source-root",
                str(source_root),
                "--expected-source-command",
                command,
                "--expected-source-commit",
                "abc123",
                "--reference-manifest",
                str(reference_manifest_path),
            ]
        )


def test_validate_dissertation_bundle_command_rejects_malformed_reference_manifest(
    tmp_path: Path,
) -> None:
    """Reference manifests must have the same artifact-list shape as bundle manifests."""
    source_root = tmp_path / "publication_candidates"
    spec_path = _make_dissertation_source(source_root)
    out_dir = tmp_path / "dissertation_export"
    command = "uv run python scripts/tools/compile_benchmark_artifacts.py --campaign-root tracked"
    bundle_name = "campaign_table_bundle"
    benchmark_publication_bundle.main(
        [
            "dissertation-bundle",
            "--source-root",
            str(source_root),
            "--out-dir",
            str(out_dir),
            "--bundle-name",
            bundle_name,
            "--artifact-spec",
            str(spec_path),
            "--command",
            command,
            "--commit",
            "abc123",
        ]
    )
    reference_manifest_path = out_dir / "reference_artifact_manifest.json"
    reference_manifest_path.write_text('{"not_artifacts": []}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="Reference manifest requires artifacts list"):
        benchmark_publication_bundle.main(
            [
                "validate-dissertation-bundle",
                "--bundle-dir",
                str(out_dir / bundle_name),
                "--source-root",
                str(source_root),
                "--expected-source-command",
                command,
                "--expected-source-commit",
                "abc123",
                "--reference-manifest",
                str(reference_manifest_path),
            ]
        )


def test_diff_dissertation_bundle_command_reports_changes_in_markdown(tmp_path, capsys) -> None:
    """Diff output should render captions, missing artifacts, corruption, and claim drift."""
    source_root = tmp_path / "publication_candidates"
    base_spec_path = _make_dissertation_source(source_root)
    out_dir = tmp_path / "dissertation_export"
    command = "uv run python scripts/tools/compile_benchmark_artifacts.py --campaign-root tracked"

    benchmark_publication_bundle.main(
        [
            "dissertation-bundle",
            "--source-root",
            str(source_root),
            "--out-dir",
            str(out_dir),
            "--bundle-name",
            "baseline",
            "--artifact-spec",
            str(base_spec_path),
            "--command",
            command,
            "--commit",
            "abc123",
        ]
    )
    baseline_manifest = out_dir / "baseline" / "artifact_manifest.json"
    assert baseline_manifest.exists()
    baseline_payload = json.loads(baseline_manifest.read_text(encoding="utf-8"))
    for row in baseline_payload["artifacts"]:
        if row["artifact_id"] == "tab_campaign_table":
            row["claim_boundary"] = "Benchmark result verified."
    baseline_manifest.write_text(json.dumps(baseline_payload, indent=2) + "\n", encoding="utf-8")

    next_spec = json.loads(base_spec_path.read_text(encoding="utf-8"))
    next_spec["artifacts"][0]["caption_draft"] = "Updated dissertation-safe caption."
    next_spec["artifacts"][0]["claim_boundary"] = "Diagnostic-only exploratory result."
    next_spec["artifacts"] = next_spec["artifacts"][:1]
    next_spec_path = out_dir / "next_artifact_spec.json"
    next_spec_path.write_text(json.dumps(next_spec, indent=2) + "\n", encoding="utf-8")

    benchmark_publication_bundle.main(
        [
            "dissertation-bundle",
            "--source-root",
            str(source_root),
            "--out-dir",
            str(out_dir),
            "--bundle-name",
            "next",
            "--artifact-spec",
            str(next_spec_path),
            "--command",
            command,
            "--commit",
            "abc123",
        ]
    )
    # create a deliberate corruption in the next bundle payload.
    next_payload = out_dir / "next" / "payload" / "artifacts" / "tab_campaign_table.md"
    next_payload.write_text(next_payload.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    diff_exit = benchmark_publication_bundle.main(
        [
            "diff-dissertation-bundle",
            "--bundle-dir",
            str(out_dir / "next"),
            "--reference-bundle-dir",
            str(out_dir / "baseline"),
        ]
    )
    captured = capsys.readouterr().out

    assert diff_exit == 0
    assert "## Changed captions" in captured
    assert "## Missing/added artifacts" in captured
    assert "## Corrupted artifacts" in captured
    assert "## Weakened claim boundaries" in captured


def test_diff_dissertation_bundle_command_rejects_escaped_output_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Diff mode should not hash payload paths that escape the bundle payload directory."""
    source_root = tmp_path / "publication_candidates"
    spec_path = _make_dissertation_source(source_root)
    out_dir = tmp_path / "dissertation_export"
    command = "uv run python scripts/tools/compile_benchmark_artifacts.py --campaign-root tracked"

    for bundle_name in ("baseline", "next"):
        benchmark_publication_bundle.main(
            [
                "dissertation-bundle",
                "--source-root",
                str(source_root),
                "--out-dir",
                str(out_dir),
                "--bundle-name",
                bundle_name,
                "--artifact-spec",
                str(spec_path),
                "--command",
                command,
                "--commit",
                "abc123",
            ]
        )
    capsys.readouterr()
    manifest_path = out_dir / "next" / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for row in manifest["artifacts"]:
        if row["artifact_id"] == "tab_campaign_table":
            row["output_path"] = "../outside.md"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    diff_exit = benchmark_publication_bundle.main(
        [
            "diff-dissertation-bundle",
            "--bundle-dir",
            str(out_dir / "next"),
            "--reference-bundle-dir",
            str(out_dir / "baseline"),
        ]
    )
    captured = capsys.readouterr().out

    assert diff_exit == 0
    assert "invalid_output_path" in captured


def test_claim_matrix_generation_basic(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """CLI claim-matrix should generate correct JSON and Markdown output for basic case."""
    diss_bundle_1, _ = _make_claims_matrix_fixtures(tmp_path)
    json_output_path = tmp_path / "claims_matrix.json"
    markdown_output_path = tmp_path / "claims_matrix.md"

    exit_code = benchmark_publication_bundle.main(
        [
            "claim-matrix",
            "--bundle-dir",
            str(diss_bundle_1),
            "--json-output",
            str(json_output_path),
            "--markdown-output",
            str(markdown_output_path),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0, captured.err
    assert json_output_path.exists()
    assert markdown_output_path.exists()

    json_content = json.loads(json_output_path.read_text(encoding="utf-8"))
    assert json_content["schema_version"] == "claim_matrix.v1"
    assert len(json_content["claims"]) == 5

    # Check a specific claim for expected values
    claim_table = next(
        (c for c in json_content["claims"] if c["artifact_id"] == "tab_campaign_table_bundle1"),
        None,
    )
    assert claim_table is not None
    assert claim_table["evidence_tier"] == "paper-grade"
    assert claim_table["allowed_wording"] == "results"
    assert claim_table["validation_status"] == "valid"

    markdown_content = markdown_output_path.read_text(encoding="utf-8")
    assert "# Dissertation Claim Matrix" in markdown_content
    assert (
        "| Artifact Id | Source Artifact Path | Checksum | Evidence Tier | Allowed Wording | Not Claimed Boundary | Figure Table Candidate | Caveat | Validation Status |"
        in markdown_content
    )
    assert "tab_campaign_table_bundle1" in markdown_content
    assert "Campaign Table 1 (src/tables/campaign_table.md)" in markdown_content
    assert "paper-grade | results | benchmark-facing, strong claim" in markdown_content


def test_claim_matrix_diagnostic_only_evidence_tier(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Diagnostic-only claim boundaries should result in appropriate evidence tier."""
    diss_bundle_1, _ = _make_claims_matrix_fixtures(tmp_path)
    json_output_path = tmp_path / "claims_matrix.json"
    markdown_output_path = tmp_path / "claims_matrix.md"

    exit_code = benchmark_publication_bundle.main(
        [
            "claim-matrix",
            "--bundle-dir",
            str(diss_bundle_1),
            "--json-output",
            str(json_output_path),
            "--markdown-output",
            str(markdown_output_path),
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0, captured.err
    json_content = json.loads(json_output_path.read_text(encoding="utf-8"))

    claim_diagnostic = next(
        (c for c in json_content["claims"] if c["artifact_id"] == "fig_planner_status_bundle1"),
        None,
    )
    assert claim_diagnostic is not None
    assert claim_diagnostic["evidence_tier"] == "diagnostic-only"
    assert claim_diagnostic["allowed_wording"] == "discussion"
    assert claim_diagnostic["validation_status"] == "valid"


def test_claim_matrix_weakest_wording_promotion(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Promoting diagnostic evidence to 'results' should result in 'invalid_claim'."""
    diss_bundle_1, _ = _make_claims_matrix_fixtures(tmp_path)
    json_output_path = tmp_path / "claims_matrix.json"
    markdown_output_path = tmp_path / "claims_matrix.md"

    exit_code = benchmark_publication_bundle.main(
        [
            "claim-matrix",
            "--bundle-dir",
            str(diss_bundle_1),
            "--json-output",
            str(json_output_path),
            "--markdown-output",
            str(markdown_output_path),
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0, captured.err
    json_content = json.loads(json_output_path.read_text(encoding="utf-8"))

    claim_invalid_promotion = next(
        (c for c in json_content["claims"] if c["artifact_id"] == "fig_diagnostic_plot_bundle1"),
        None,
    )
    assert claim_invalid_promotion is not None
    assert (
        claim_invalid_promotion["evidence_tier"] == "diagnostic-only"
    )  # Still diagnostic-only based on boundary
    assert claim_invalid_promotion["allowed_wording"] == "discussion"  # Weakened from results
    assert (
        claim_invalid_promotion["validation_status"]
        == "invalid_claim: diagnostic promoted to results"
    )


def test_claim_matrix_evidence_bundle_boundary_weakens_wording(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A weak referenced evidence bundle should weaken otherwise paper-grade wording."""
    diss_bundle_1, evidence_bundle_1 = _make_claims_matrix_fixtures(tmp_path)
    evidence_manifest = evidence_bundle_1 / "evidence_bundle_manifest.json"
    evidence_payload = json.loads(evidence_manifest.read_text(encoding="utf-8"))
    evidence_payload["claim_boundary"] = "diagnostic-only, not benchmark evidence"
    evidence_manifest.write_text(json.dumps(evidence_payload, indent=2) + "\n", encoding="utf-8")
    json_output_path = tmp_path / "claims_matrix.json"
    markdown_output_path = tmp_path / "claims_matrix.md"

    exit_code = benchmark_publication_bundle.main(
        [
            "claim-matrix",
            "--bundle-dir",
            str(diss_bundle_1),
            "--evidence-bundle-dir",
            str(evidence_bundle_1),
            "--json-output",
            str(json_output_path),
            "--markdown-output",
            str(markdown_output_path),
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0, captured.err
    json_content = json.loads(json_output_path.read_text(encoding="utf-8"))

    claim_table = next(
        (c for c in json_content["claims"] if c["artifact_id"] == "tab_campaign_table_bundle1"),
        None,
    )
    assert claim_table is not None
    assert claim_table["evidence_tier"] == "diagnostic-only"
    assert claim_table["allowed_wording"] == "discussion"
    assert "Evidence bundle boundary: diagnostic-only" in claim_table["caveat"]


def test_claim_matrix_missing_checksum_or_source(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Missing checksum or source path should lead to 'non-claimable' status."""
    diss_bundle_1, _ = _make_claims_matrix_fixtures(tmp_path)
    json_output_path = tmp_path / "claims_matrix.json"
    markdown_output_path = tmp_path / "claims_matrix.md"

    exit_code = benchmark_publication_bundle.main(
        [
            "claim-matrix",
            "--bundle-dir",
            str(diss_bundle_1),
            "--json-output",
            str(json_output_path),
            "--markdown-output",
            str(markdown_output_path),
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0, captured.err
    json_content = json.loads(json_output_path.read_text(encoding="utf-8"))

    # Test missing checksum
    claim_missing_checksum = next(
        (c for c in json_content["claims"] if c["artifact_id"] == "fig_missing_checksum_bundle1"),
        None,
    )
    assert claim_missing_checksum is not None
    assert claim_missing_checksum["evidence_tier"] == "non-claimable"
    assert claim_missing_checksum["allowed_wording"] == "do-not-use"
    assert claim_missing_checksum["validation_status"] == "non-claimable: missing checksum"

    # Test missing source path
    claim_missing_source_path = next(
        (
            c
            for c in json_content["claims"]
            if c["artifact_id"] == "fig_missing_source_path_bundle1"
        ),
        None,
    )
    assert claim_missing_source_path is not None
    assert claim_missing_source_path["evidence_tier"] == "non-claimable"
    assert claim_missing_source_path["allowed_wording"] == "do-not-use"
    assert claim_missing_source_path["validation_status"] == "non-claimable: missing source path"

    # Test missing payload file
    # This was implicitly tested by the fig_missing_checksum_bundle1 and fig_missing_source_path_bundle1
    # because the payload file check happens before the manifest checksum check.
    # The payload file for fig_missing_checksum_bundle1 exists in the fixture.
    # We need to manually remove a payload file for one artifact to test the 'missing payload file' status
    # This test should ideally be in its own fixture setup for clarity.
    # For now, relying on the fixture setup where fig_missing_checksum_bundle1 payload exists.
    # A dedicated test for 'missing payload file' would involve deleting the created payload file.


def _make_chapter_target_claims_matrix_fixtures(tmp_path: Path) -> Path:
    """Create a dissertation bundle with chapter-target metadata for claim-matrix tests."""
    bundle_root = tmp_path / "diss_bundle_chapter_targets"
    bundle_root.mkdir(parents=True, exist_ok=True)
    payload_dir = bundle_root / "payload" / "artifacts"
    payload_dir.mkdir(parents=True, exist_ok=True)

    _write(payload_dir / "tab_results.md", "| metric | value |\n| --- | --- |\n")
    _write(payload_dir / "fig_limitations.png", "fake png bytes for limitations\n")
    _write(payload_dir / "fig_results_unjustified.png", "fake png bytes for unjustified\n")
    _write(payload_dir / "fig_results_justified.png", "fake png bytes for justified\n")
    _write(payload_dir / "fig_alias_target.png", "fake png bytes for alias target\n")
    _write(payload_dir / "fig_metadata_target.png", "fake png bytes for metadata target\n")
    _write(payload_dir / "tab_no_target.md", "| diagnostic | value |\n| --- | --- |\n")

    table_hash = benchmark_publication_bundle._sha256_file(payload_dir / "tab_results.md")
    limitations_hash = benchmark_publication_bundle._sha256_file(
        payload_dir / "fig_limitations.png"
    )
    unjustified_hash = benchmark_publication_bundle._sha256_file(
        payload_dir / "fig_results_unjustified.png"
    )
    justified_hash = benchmark_publication_bundle._sha256_file(
        payload_dir / "fig_results_justified.png"
    )
    alias_hash = benchmark_publication_bundle._sha256_file(payload_dir / "fig_alias_target.png")
    metadata_hash = benchmark_publication_bundle._sha256_file(
        payload_dir / "fig_metadata_target.png"
    )
    no_target_hash = benchmark_publication_bundle._sha256_file(payload_dir / "tab_no_target.md")

    spec = {
        "schema_version": "dissertation_artifact_bundle.v1",
        "source_commit": "abc123",
        "generation_command": "test-command-chapter-targets",
        "artifacts": [
            {
                "artifact_id": "tab_results_chapter_target",
                "source_path": "src/tables/results.md",
                "source_artifact": "Results Table",
                "caption_draft": "Benchmark results table.",
                "claim_boundary": "benchmark-facing, strong claim",
                "recommended_manuscript_use": "results",
                "fallback_degraded_summary": "No degradation.",
                "sha256": table_hash,
                "output_path": "artifacts/tab_results.md",
                "chapter_target": "Results, Section 4.2",
            },
            {
                "artifact_id": "fig_limitations_chapter_target",
                "source_path": "src/figures/limitations.png",
                "source_artifact": "Limitations Figure",
                "caption_draft": "Limitations figure.",
                "claim_boundary": "diagnostic-only",
                "recommended_manuscript_use": "discussion",
                "fallback_degraded_summary": "Diagnostic only.",
                "sha256": limitations_hash,
                "output_path": "artifacts/fig_limitations.png",
                "chapter_target": "Limitations, Section 5.1",
            },
            {
                "artifact_id": "fig_results_unjustified_chapter_target",
                "source_path": "src/figures/results_unjustified.png",
                "source_artifact": "Unjustified Results Figure",
                "caption_draft": "Diagnostic figure wrongly targeted at Results.",
                "claim_boundary": "diagnostic-only",
                "recommended_manuscript_use": "discussion",
                "fallback_degraded_summary": "Diagnostic only.",
                "sha256": unjustified_hash,
                "output_path": "artifacts/fig_results_unjustified.png",
                "chapter_target": "Results, Section 4.3",
            },
            {
                "artifact_id": "fig_results_justified_chapter_target",
                "source_path": "src/figures/results_justified.png",
                "source_artifact": "Justified Results Figure",
                "caption_draft": "Diagnostic figure with explicit Results justification.",
                "claim_boundary": "diagnostic-only",
                "recommended_manuscript_use": "discussion",
                "fallback_degraded_summary": "Diagnostic only.",
                "sha256": justified_hash,
                "output_path": "artifacts/fig_results_justified.png",
                "chapter_target": "Results, Section 4.4",
                "chapter_target_justification": "Required to illustrate a diagnostic contrast in the Results section.",
            },
            {
                "artifact_id": "fig_alias_chapter_target",
                "source_path": "src/figures/alias_target.png",
                "source_artifact": "Alias Target Figure",
                "caption_draft": "Diagnostic figure using the dissertation_chapter alias.",
                "claim_boundary": "diagnostic-only",
                "recommended_manuscript_use": "discussion",
                "fallback_degraded_summary": "Diagnostic only.",
                "sha256": alias_hash,
                "output_path": "artifacts/fig_alias_target.png",
                "dissertation_chapter": "Methodology, Section 3.2",
            },
            {
                "artifact_id": "fig_metadata_chapter_target",
                "source_path": "src/figures/metadata_target.png",
                "source_artifact": "Metadata Target Figure",
                "caption_draft": "Diagnostic figure using metadata chapter target.",
                "claim_boundary": "diagnostic-only",
                "recommended_manuscript_use": "discussion",
                "fallback_degraded_summary": "Diagnostic only.",
                "sha256": metadata_hash,
                "output_path": "artifacts/fig_metadata_target.png",
                "metadata": {
                    "chapter_target": "Future Work, Section 6.1",
                    "chapter_target_justification": "Future-work placement for diagnostic limits.",
                },
            },
            {
                "artifact_id": "tab_no_chapter_target",
                "source_path": "src/tables/no_target.md",
                "source_artifact": "No Target Table",
                "caption_draft": "Diagnostic table without chapter target.",
                "claim_boundary": "diagnostic-only",
                "recommended_manuscript_use": "discussion",
                "fallback_degraded_summary": "Diagnostic only.",
                "sha256": no_target_hash,
                "output_path": "artifacts/tab_no_target.md",
            },
        ],
    }
    _write(bundle_root / "artifact_manifest.json", json.dumps(spec, indent=2) + "\n")
    _write(
        bundle_root / "checksums.sha256",
        f"{table_hash}  artifacts/tab_results.md\n"
        f"{limitations_hash}  artifacts/fig_limitations.png\n"
        f"{unjustified_hash}  artifacts/fig_results_unjustified.png\n"
        f"{justified_hash}  artifacts/fig_results_justified.png\n"
        f"{alias_hash}  artifacts/fig_alias_target.png\n"
        f"{metadata_hash}  artifacts/fig_metadata_target.png\n"
        f"{no_target_hash}  artifacts/tab_no_target.md\n",
    )
    return bundle_root


def test_claim_matrix_chapter_target_surfaces(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Chapter targets should appear in JSON and Markdown when present and be absent otherwise."""
    diss_bundle = _make_chapter_target_claims_matrix_fixtures(tmp_path)
    json_output_path = tmp_path / "claims_matrix_chapter_targets.json"
    markdown_output_path = tmp_path / "claims_matrix_chapter_targets.md"

    exit_code = benchmark_publication_bundle.main(
        [
            "claim-matrix",
            "--bundle-dir",
            str(diss_bundle),
            "--json-output",
            str(json_output_path),
            "--markdown-output",
            str(markdown_output_path),
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0, captured.err

    json_content = json.loads(json_output_path.read_text(encoding="utf-8"))
    assert json_content["schema_version"] == "claim_matrix.v1"
    assert len(json_content["claims"]) == 7

    by_id = {c["artifact_id"]: c for c in json_content["claims"]}
    results_row = by_id["tab_results_chapter_target"]
    assert results_row["evidence_tier"] == "paper-grade"
    assert results_row["chapter_target"] == "Results, Section 4.2"
    assert results_row["chapter_target_status"] == "ok"

    limitations_row = by_id["fig_limitations_chapter_target"]
    assert limitations_row["evidence_tier"] == "diagnostic-only"
    assert limitations_row["chapter_target"] == "Limitations, Section 5.1"
    assert limitations_row["chapter_target_status"] == "ok"

    no_target_row = by_id["tab_no_chapter_target"]
    assert no_target_row["chapter_target"] is None
    assert no_target_row["chapter_target_status"] is None

    alias_row = by_id["fig_alias_chapter_target"]
    assert alias_row["chapter_target"] == "Methodology, Section 3.2"
    assert alias_row["chapter_target_status"] == "ok"

    metadata_row = by_id["fig_metadata_chapter_target"]
    assert metadata_row["chapter_target"] == "Future Work, Section 6.1"
    assert metadata_row["chapter_target_status"] == "ok"

    markdown_content = markdown_output_path.read_text(encoding="utf-8")
    assert "# Dissertation Claim Matrix" in markdown_content
    assert "Chapter Target" in markdown_content
    assert "Results, Section 4.2" in markdown_content
    assert "Limitations, Section 5.1" in markdown_content
    no_target_markdown_row = next(
        line for line in markdown_content.splitlines() if "tab_no_chapter_target" in line
    )
    assert no_target_markdown_row.endswith(" |  |  |")


def test_claim_matrix_diagnostic_chapter_target_requires_justification(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Diagnostic-only rows targeting Results should warn unless explicitly justified."""
    diss_bundle = _make_chapter_target_claims_matrix_fixtures(tmp_path)
    json_output_path = tmp_path / "claims_matrix_chapter_targets.json"
    markdown_output_path = tmp_path / "claims_matrix_chapter_targets.md"

    exit_code = benchmark_publication_bundle.main(
        [
            "claim-matrix",
            "--bundle-dir",
            str(diss_bundle),
            "--json-output",
            str(json_output_path),
            "--markdown-output",
            str(markdown_output_path),
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0, captured.err

    json_content = json.loads(json_output_path.read_text(encoding="utf-8"))
    by_id = {c["artifact_id"]: c for c in json_content["claims"]}

    unjustified_row = by_id["fig_results_unjustified_chapter_target"]
    assert unjustified_row["evidence_tier"] == "diagnostic-only"
    assert unjustified_row["chapter_target"] == "Results, Section 4.3"
    assert unjustified_row["chapter_target_status"].startswith("warning:")
    assert "limitations/methodology/future-work" in unjustified_row["chapter_target_status"]

    justified_row = by_id["fig_results_justified_chapter_target"]
    assert justified_row["evidence_tier"] == "diagnostic-only"
    assert justified_row["chapter_target"] == "Results, Section 4.4"
    assert justified_row["chapter_target_status"] == "ok"


def test_claim_matrix_chapter_target_validation_edge_cases() -> None:
    """Chapter-target validation should flag inconsistent weak rows."""
    assert (
        benchmark_publication_bundle._validate_chapter_target(
            "diagnostic-only",
            None,
            "Justification without a target.",
        )
        == "warning: justification provided but no chapter target is specified"
    )
    assert benchmark_publication_bundle._validate_chapter_target(
        "non-claimable",
        "Results, Section 4.5",
        None,
    ).startswith("warning: non-claimable row targets")
    assert benchmark_publication_bundle._format_claim_matrix_cell("a | b") == r"a \| b"


def test_claim_matrix_no_chapter_target_remains_stable(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Rows without chapter targets should omit chapter-target columns from Markdown."""
    diss_bundle_1, _ = _make_claims_matrix_fixtures(tmp_path)
    json_output_path = tmp_path / "claims_matrix_stable.json"
    markdown_output_path = tmp_path / "claims_matrix_stable.md"

    exit_code = benchmark_publication_bundle.main(
        [
            "claim-matrix",
            "--bundle-dir",
            str(diss_bundle_1),
            "--json-output",
            str(json_output_path),
            "--markdown-output",
            str(markdown_output_path),
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0, captured.err

    json_content = json.loads(json_output_path.read_text(encoding="utf-8"))
    for claim in json_content["claims"]:
        assert claim["chapter_target"] is None
        assert claim["chapter_target_status"] is None

    markdown_content = markdown_output_path.read_text(encoding="utf-8")
    assert "Chapter Target" not in markdown_content
    assert (
        "| Artifact Id | Source Artifact Path | Checksum | Evidence Tier | Allowed Wording | Not Claimed Boundary | Figure Table Candidate | Caveat | Validation Status |"
        in markdown_content
    )
