"""Tests for the release evidence snapshot gate."""

from __future__ import annotations

import hashlib
import json
import subprocess
from typing import TYPE_CHECKING, Any

import yaml

from scripts.tools import release_evidence_snapshot as snapshot

if TYPE_CHECKING:
    from pathlib import Path


def test_release_evidence_snapshot_includes_tracked_catalog_files(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """A tracked artifact catalog should pull in its source, output, and caption files."""
    repo = _init_fixture_repo(tmp_path / "repo")
    monkeypatch.chdir(repo)
    output_json = tmp_path / "snapshot.json"

    exit_code = snapshot.main(
        [
            "--source-ref",
            "HEAD",
            "--no-default-includes",
            "--include",
            "configs/release_smoke.yaml",
            "--artifact-catalog",
            "docs/context/evidence/issue_fixture/artifact_catalog.yaml",
            "--no-auto-artifact-catalogs",
            "--output-json",
            str(output_json),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "release_evidence_snapshot.v0.1"
    assert payload["status"] == "valid"
    assert payload["evidence_classification"] == "diagnostic-only"
    assert payload["source"]["source_commit"]
    assert payload["doi_ready_metadata"]["release_id"] == "fixture_release"

    paths = {entry["path"] for entry in payload["file_manifest"]["files"]}
    assert "configs/release_smoke.yaml" in paths
    assert "docs/context/evidence/issue_fixture/artifact_catalog.yaml" in paths
    assert "docs/context/evidence/issue_fixture/tables/campaign_table.md" in paths
    assert "docs/context/evidence/issue_fixture/figures/status.png" in paths

    catalog = payload["artifact_catalogs"][0]
    assert catalog["status"] == "valid"
    assert {artifact["artifact_id"] for artifact in catalog["principal_artifacts"]} == {
        "tab_campaign_table",
        "fig_status",
    }
    assert payload["fallback_degraded_exclusions"]
    assert payload["config_seed_identifiers"]["seed_policies"] == [
        {
            "path": "configs/release_smoke.yaml",
            "mode": "fixed-list",
            "seed_set": None,
            "seeds": [111, 112],
            "seed_sets_path": "configs/seed_sets.yaml",
        }
    ]

    _assert_repo_path(repo)


def test_release_evidence_snapshot_missing_required_input_fails_closed(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Required missing durable inputs should be explicit and produce a failing exit."""
    repo = _init_fixture_repo(tmp_path / "repo")
    monkeypatch.chdir(repo)
    output_json = tmp_path / "snapshot.json"

    exit_code = snapshot.main(
        [
            "--source-ref",
            "HEAD",
            "--no-default-includes",
            "--artifact-catalog",
            "docs/context/evidence/issue_fixture/artifact_catalog.yaml",
            "--no-auto-artifact-catalogs",
            "--require-input",
            "docs/context/evidence/missing/promoted_table.md",
            "--output-json",
            str(output_json),
        ]
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert exit_code == 2
    assert payload["status"] == "fail_closed"
    assert payload["evidence_classification"] == "diagnostic-only"
    assert payload["missing_inputs"] == [
        {
            "path": "docs/context/evidence/missing/promoted_table.md",
            "status": "missing",
            "reason": "required input is absent from the source ref",
        }
    ]


def test_release_evidence_snapshot_catalog_checksum_mismatch_fails_closed(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Artifact catalogs with stale checksums should not pass the release evidence gate."""
    repo = _init_fixture_repo(tmp_path / "repo")
    monkeypatch.chdir(repo)
    table_path = repo / "docs/context/evidence/issue_fixture/tables/campaign_table.md"
    table_path.write_text(
        "| planner | status |\n| --- | --- |\n| orca | degraded |\n", encoding="utf-8"
    )
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "stale catalog fixture"], cwd=repo, check=True)
    output_json = tmp_path / "snapshot.json"

    exit_code = snapshot.main(
        [
            "--source-ref",
            "HEAD",
            "--no-default-includes",
            "--artifact-catalog",
            "docs/context/evidence/issue_fixture/artifact_catalog.yaml",
            "--no-auto-artifact-catalogs",
            "--output-json",
            str(output_json),
        ]
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert exit_code == 2
    assert payload["status"] == "fail_closed"
    assert payload["artifact_catalogs"][0]["status"] == "invalid"
    assert any(
        issue.startswith("checksum mismatch for docs/context/evidence/issue_fixture/tables/")
        for issue in payload["artifact_catalogs"][0]["issues"]
    )


def test_release_evidence_snapshot_malformed_catalog_fails_closed(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Malformed tracked catalogs should be reported as invalid snapshot inputs."""
    repo = _init_fixture_repo(tmp_path / "repo")
    monkeypatch.chdir(repo)
    catalog_path = repo / "docs/context/evidence/issue_fixture/artifact_catalog.yaml"
    catalog_path.write_text("artifacts: [unterminated\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "malformed catalog fixture"], cwd=repo, check=True)
    output_json = tmp_path / "snapshot.json"

    exit_code = snapshot.main(
        [
            "--source-ref",
            "HEAD",
            "--no-default-includes",
            "--artifact-catalog",
            "docs/context/evidence/issue_fixture/artifact_catalog.yaml",
            "--no-auto-artifact-catalogs",
            "--output-json",
            str(output_json),
        ]
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert exit_code == 2
    assert payload["status"] == "fail_closed"
    assert payload["artifact_catalogs"][0]["status"] == "invalid"
    assert payload["artifact_catalogs"][0]["issues"][0].startswith("could not parse")


def _init_fixture_repo(repo: Path) -> Path:
    """Create a tiny Git repository with a valid tracked artifact catalog."""
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo, check=True)

    config = repo / "configs/release_smoke.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(
        yaml.safe_dump(
            {
                "release_id": "fixture_release",
                "release_tag": "fixture-v0",
                "name": "fixture_release_smoke",
                "paper_facing": True,
                "scenario_matrix": "configs/scenarios/fixture.yaml",
                "seed_policy": {
                    "mode": "fixed-list",
                    "seeds": [111, 112],
                    "seed_sets_path": "configs/seed_sets.yaml",
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    evidence = repo / "docs/context/evidence/issue_fixture"
    (evidence / "sources/reports").mkdir(parents=True)
    (evidence / "tables").mkdir()
    (evidence / "figures").mkdir()
    (evidence / "sources/reports/campaign_table.csv").write_text(
        "planner,status\norca,native\nfallback_planner,fallback\n",
        encoding="utf-8",
    )
    (evidence / "tables/campaign_table.md").write_text(
        "| planner | status |\n| --- | --- |\n| orca | native |\n",
        encoding="utf-8",
    )
    (evidence / "figures/status.png").write_bytes(b"fixture-png")
    (evidence / "captions.md").write_text(
        "Fallback and degraded rows are exclusions, not success evidence.\n",
        encoding="utf-8",
    )

    catalog = {
        "schema_version": "artifact_catalog.v1",
        "catalog_id": "fixture_catalog",
        "artifacts": [
            {
                "artifact_id": "tab_campaign_table",
                "artifact_kind": "table",
                "source_kind": "benchmark_campaign",
                "source_files": [
                    _ref(evidence, "sources/reports/campaign_table.csv"),
                ],
                "outputs": {"md": _ref(evidence, "tables/campaign_table.md")},
                "caption_file": _ref(evidence, "captions.md"),
                "generation_command": "uv run python scripts/tools/compile_benchmark_artifacts.py",
                "generation_commit": "abc123",
                "claim_boundary": (
                    "Diagnostic-only table. Fallback and degraded rows are exclusions, not "
                    "success evidence."
                ),
            },
            {
                "artifact_id": "fig_status",
                "artifact_kind": "figure",
                "source_kind": "benchmark_campaign",
                "source_files": [
                    _ref(evidence, "sources/reports/campaign_table.csv"),
                ],
                "outputs": {"png": _ref(evidence, "figures/status.png")},
                "caption_file": _ref(evidence, "captions.md"),
                "generation_command": "uv run python scripts/tools/compile_benchmark_artifacts.py",
                "generation_commit": "abc123",
                "claim_boundary": (
                    "Diagnostic-only figure. Fallback and degraded rows are exclusions, not "
                    "success evidence."
                ),
            },
        ],
    }
    (evidence / "artifact_catalog.yaml").write_text(
        yaml.safe_dump(catalog, sort_keys=False),
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "fixture"], cwd=repo, check=True)
    return repo


def _ref(base: Path, relative_path: str) -> dict[str, str]:
    """Return an artifact-catalog file reference for a fixture file."""
    path = base / relative_path
    return {"path": relative_path, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _assert_repo_path(repo: Path) -> None:
    """Keep the temp repo object visibly used for type checkers and debuggers."""
    assert repo.exists()
