"""Tests for the publication-bundle preflight gate (issue #5530)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from robot_sf.benchmark.artifact_publication import (
    PublicationPreflightError,
    export_publication_bundle,
    verify_publication_bundle_preflight,
)
from scripts.tools import publication_preflight as preflight_cli


def _write(path: Path, payload: str) -> None:
    """Write UTF-8 text to a path, creating parents as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def _make_valid_run(run_dir: Path) -> None:
    """Create a minimal synthetic benchmark run directory."""
    _write(run_dir / "manifest.json", json.dumps({"git_hash": "abc123"}))
    _write(
        run_dir / "run_meta.json",
        json.dumps({"repo": {"remote": "git@github.com:ll7/robot_sf_ll7.git", "commit": "abc123"}}),
    )
    _write(run_dir / "episodes" / "episodes.jsonl", '{"episode_id":"ep-1"}\n')
    _write(run_dir / "reports" / "report.md", "# Report\n")


def _seed_release_bundle(bundle_dir: Path, *, publication_commit: str = "abc123") -> None:
    """Inject a self-consistent release payload (result + summary) into a bundle.

    The release_result/campaign_summary pair agree on the four reconciliation
    fields, and the campaign_summary provenance commit matches the episode
    software_commit and the (overriden) publication manifest commit.
    """
    payload = bundle_dir / "payload"
    (payload / "release").mkdir(parents=True, exist_ok=True)
    (payload / "reports").mkdir(parents=True, exist_ok=True)
    (payload / "runs" / "orca__holonomic").mkdir(parents=True, exist_ok=True)

    release_result = {
        "status": "benchmark_success",
        "evidence_status": "valid",
        "total_episodes": 1440,
        "successful_runs": 1,
    }
    _write(payload / "release" / "release_result.json", json.dumps(release_result))

    campaign_summary = {
        "campaign": {
            "status": "benchmark_success",
            "evidence_status": "valid",
            "total_episodes": 1440,
            "successful_runs": 1,
        }
    }
    _write(payload / "reports" / "campaign_summary.json", json.dumps(campaign_summary))

    _write(
        payload / "runs" / "orca__holonomic" / "episodes.jsonl",
        json.dumps(
            {
                "episode_id": "ep-1",
                "event_ledger": {"software_commit": publication_commit},
            }
        )
        + "\n",
    )


def _inject_publication_commit(bundle_dir: Path, *, commit: str) -> dict:
    """Rewrite the publication manifest's repository commit and return the manifest.

    ``export_publication_bundle`` already writes ``files`` and ``checksums.sha256``
    with ``payload/``-prefixed paths, so the publication commit is the only field
    that needs updating here to keep the bundle self-consistent.
    """
    manifest_path = bundle_dir / "publication_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.setdefault("provenance", {}).setdefault("repository", {})["commit"] = commit
    if "publication_channels" not in manifest:
        manifest["publication_channels"] = {}
    # Use a resolved DOI/URL so the placeholder check does not fire spuriously.
    manifest["publication_channels"].setdefault("doi", "10.5281/zenodo.1234567")
    manifest["publication_channels"].setdefault(
        "release_url", "https://github.com/ll7/robot_sf_ll7/releases/tag/v1"
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _build_bundle(tmp_path: Path, *, publication_commit: str = "abc123") -> Path:
    """Export a publication bundle and seed a valid release payload into it."""
    run_dir = tmp_path / "benchmarks" / "run_release"
    _make_valid_run(run_dir)
    out_dir = tmp_path / "publication"
    result = export_publication_bundle(
        run_dir,
        out_dir,
        bundle_name="run_release_bundle",
        include_videos=False,
        doi="10.5281/zenodo.1234567",
        repository_url="https://github.com/ll7/robot_sf_ll7",
        release_tag="v1",
        overwrite=True,
    )
    _inject_publication_commit(result.bundle_dir, commit=publication_commit)
    _seed_release_bundle(result.bundle_dir, publication_commit=publication_commit)
    return result.bundle_dir


def test_preflight_passes_on_consistent_bundle(tmp_path: Path) -> None:
    """A self-consistent bundle should pass the preflight with no violations."""
    bundle_dir = _build_bundle(tmp_path)
    report = verify_publication_bundle_preflight(bundle_dir)
    assert report["status"] == "pass"
    assert report["violation_count"] == 0
    assert report["evidence"]["publication_commit"] == "abc123"


def test_preflight_cli_exit_codes(tmp_path: Path) -> None:
    """CLI should exit 0 on pass and 1 on a failing bundle."""
    bundle_dir = _build_bundle(tmp_path)
    assert preflight_cli.main(["--bundle-dir", str(bundle_dir)]) == 0

    # Break reconciliation -> fail.
    summary = bundle_dir / "payload" / "reports" / "campaign_summary.json"
    payload = json.loads(summary.read_text(encoding="utf-8"))
    payload["campaign"]["total_episodes"] = 9999
    summary.write_text(json.dumps(payload), encoding="utf-8")
    assert preflight_cli.main(["--bundle-dir", str(bundle_dir)]) == 1


def test_preflight_cli_reports_malformed_inputs_without_traceback(tmp_path: Path) -> None:
    """Malformed JSON/checksums must remain structured fail-closed CLI failures."""
    bundle_dir = _build_bundle(tmp_path)
    (bundle_dir / "checksums.sha256").write_text("not-a-checksum\n", encoding="utf-8")
    assert preflight_cli.main(["--bundle-dir", str(bundle_dir)]) == 1

    bundle_dir = _build_bundle(tmp_path / "malformed-json")
    (bundle_dir / "payload" / "release" / "release_result.json").write_text(
        "{not-json}\n", encoding="utf-8"
    )
    assert preflight_cli.main(["--bundle-dir", str(bundle_dir)]) == 1


def test_preflight_fails_on_result_summary_disagreement(tmp_path: Path) -> None:
    """Disagreement between release_result and campaign_summary must fail closed."""
    bundle_dir = _build_bundle(tmp_path)
    summary = bundle_dir / "payload" / "reports" / "campaign_summary.json"
    payload = json.loads(summary.read_text(encoding="utf-8"))
    payload["campaign"]["status"] = "unexpected_failure"
    payload["campaign"]["evidence_status"] = "invalid"
    summary.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(PublicationPreflightError, match="release/release_result.json disagrees"):
        verify_publication_bundle_preflight(bundle_dir)


def test_preflight_fails_on_checksum_relative_to_root(tmp_path: Path) -> None:
    """Checksums must verify relative to the bundle root, not payload/.

    Reproduces the 0.0.3 defect: checksums written relative to payload/ make
    ``sha256sum -c checksums.sha256`` fail from the bundle root.
    """
    bundle_dir = _build_bundle(tmp_path)
    manifest = json.loads((bundle_dir / "publication_manifest.json").read_text(encoding="utf-8"))
    files = manifest["files"]
    for entry in files:
        entry["path"] = f"payload/{entry['path']}"
    (bundle_dir / "publication_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    checksums_path = bundle_dir / "checksums.sha256"
    lines = [
        line
        if not line.strip() or line.startswith("#")
        else f"{line.split()[0]}  payload/{line.split()[1]}"
        for line in checksums_path.read_text(encoding="utf-8").splitlines()
    ]
    checksums_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(PublicationPreflightError, match="missing from bundle root"):
        verify_publication_bundle_preflight(bundle_dir)


def test_preflight_fails_when_manifest_digest_disagrees_with_checksums(tmp_path: Path) -> None:
    """Manifest SHA-256 values must agree with the signed checksum file."""
    bundle_dir = _build_bundle(tmp_path)
    manifest_path = bundle_dir / "publication_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][0]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(PublicationPreflightError, match="manifest sha256 disagrees"):
        verify_publication_bundle_preflight(bundle_dir)


def test_preflight_fails_on_commit_mismatch_without_explanation(tmp_path: Path) -> None:
    """Episode software_commit != publication commit must fail without an explanation."""
    bundle_dir = _build_bundle(tmp_path, publication_commit="abc123")
    # Episode commit differs from the publication manifest commit.
    episodes = bundle_dir / "payload" / "runs" / "orca__holonomic" / "episodes.jsonl"
    episodes.write_text(
        json.dumps({"episode_id": "ep-1", "event_ledger": {"software_commit": "deadbeef"}}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(PublicationPreflightError, match="software_commit values"):
        verify_publication_bundle_preflight(bundle_dir)


def test_preflight_allows_commit_mismatch_with_explanation(tmp_path: Path) -> None:
    """Structured commit reconciliation downgrades a documented mismatch to a warning."""
    bundle_dir = _build_bundle(tmp_path, publication_commit="abc123")
    episodes = bundle_dir / "payload" / "runs" / "orca__holonomic" / "episodes.jsonl"
    episodes.write_text(
        json.dumps({"episode_id": "ep-1", "event_ledger": {"software_commit": "deadbeef"}}) + "\n",
        encoding="utf-8",
    )
    manifest_path = bundle_dir / "publication_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.setdefault("provenance", {})["commit_reconciliation"] = {
        "status": "explained",
        "publication_commit": "abc123",
        "runtime_commits": ["deadbeef"],
        "explanation": (
            "Episodes were produced by an earlier runtime commit; the publication commit "
            "carries a reporting-only fix that does not change episode outputs."
        ),
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = verify_publication_bundle_preflight(bundle_dir)
    assert report["status"] == "pass"
    assert any("structured provenance.commit_reconciliation" in w for w in report["warnings"])


def test_preflight_rejects_unstructured_commit_reconciliation(tmp_path: Path) -> None:
    """A top-level prose string cannot waive a runtime/publication mismatch."""
    bundle_dir = _build_bundle(tmp_path, publication_commit="abc123")
    episodes = bundle_dir / "payload" / "runs" / "orca__holonomic" / "episodes.jsonl"
    episodes.write_text(
        json.dumps({"episode_id": "ep-1", "event_ledger": {"software_commit": "deadbeef"}}) + "\n",
        encoding="utf-8",
    )
    manifest_path = bundle_dir / "publication_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["publication_runtime_diff_explanation"] = "prose-only waiver"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(
        PublicationPreflightError, match="structured provenance.commit_reconciliation"
    ):
        verify_publication_bundle_preflight(bundle_dir)


def test_preflight_rejects_malformed_episode_rows(tmp_path: Path) -> None:
    """Malformed rows or missing episode-ledger commits must not be silently ignored."""
    bundle_dir = _build_bundle(tmp_path)
    episodes = bundle_dir / "payload" / "runs" / "orca__holonomic" / "episodes.jsonl"
    episodes.write_text("{not-json}\n", encoding="utf-8")
    with pytest.raises(PublicationPreflightError, match="invalid JSON"):
        verify_publication_bundle_preflight(bundle_dir)

    bundle_dir = _build_bundle(tmp_path / "missing-ledger")
    episodes = bundle_dir / "payload" / "runs" / "orca__holonomic" / "episodes.jsonl"
    episodes.write_text(json.dumps({"episode_id": "ep-1"}) + "\n", encoding="utf-8")
    with pytest.raises(PublicationPreflightError, match="event_ledger"):
        verify_publication_bundle_preflight(bundle_dir)


def test_preflight_fails_on_placeholder_channels(tmp_path: Path) -> None:
    """Unresolved DOI/release-tag placeholders must be rejected."""
    bundle_dir = _build_bundle(tmp_path)
    manifest_path = bundle_dir / "publication_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["publication_channels"]["doi"] = "10.5281/zenodo.<record-id>"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(PublicationPreflightError, match="retains an unresolved placeholder"):
        verify_publication_bundle_preflight(bundle_dir)


def test_preflight_fails_on_ambiguous_goal_timeout_row(tmp_path: Path) -> None:
    """A goal-reached + timeout row without timing evidence must be rejected."""
    bundle_dir = _build_bundle(tmp_path)
    episodes = bundle_dir / "payload" / "runs" / "orca__holonomic" / "episodes.jsonl"
    episodes.write_text(
        json.dumps(
            {
                "episode_id": "ep-2",
                "event_ledger": {
                    "software_commit": "abc123",
                    "exact_events": {"goal_reached": True, "timeout": True, "collision": False},
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(PublicationPreflightError, match="goal_reached\\+timeout"):
        verify_publication_bundle_preflight(bundle_dir)


def test_preflight_accepts_goal_timeout_row_with_timing_evidence(tmp_path: Path) -> None:
    """A goal-reached + timeout row carrying reached_goal_step must pass."""
    bundle_dir = _build_bundle(tmp_path)
    episodes = bundle_dir / "payload" / "runs" / "orca__holonomic" / "episodes.jsonl"
    episodes.write_text(
        json.dumps(
            {
                "episode_id": "ep-2",
                "reached_goal_step": 412,
                "event_ledger": {
                    "software_commit": "abc123",
                    "exact_events": {"goal_reached": True, "timeout": True, "collision": False},
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    report = verify_publication_bundle_preflight(bundle_dir)
    assert report["status"] == "pass"


def test_preflight_rejects_missing_release_inputs_when_required(tmp_path: Path) -> None:
    """When required, missing release_result/campaign_summary must block publication."""
    bundle_dir = _build_bundle(tmp_path)
    (bundle_dir / "payload" / "release" / "release_result.json").unlink()
    (bundle_dir / "payload" / "reports" / "campaign_summary.json").unlink()
    with pytest.raises(PublicationPreflightError, match="release reconciliation inputs missing"):
        verify_publication_bundle_preflight(bundle_dir)


def test_preflight_skips_release_check_when_not_required(tmp_path: Path) -> None:
    """Without the release inputs and require_release_reconciliation=False, it passes."""
    bundle_dir = _build_bundle(tmp_path)
    (bundle_dir / "payload" / "release" / "release_result.json").unlink()
    (bundle_dir / "payload" / "reports" / "campaign_summary.json").unlink()
    report = verify_publication_bundle_preflight(bundle_dir, require_release_reconciliation=False)
    assert report["status"] == "pass"
