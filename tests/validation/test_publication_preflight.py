"""Tests for the publication-bundle preflight gate (issue #5530)."""

from __future__ import annotations

import hashlib
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


# --- Coverage for the remaining fail-closed preflight branches (issue #5691) ---


def _episodes_path(bundle_dir: Path) -> Path:
    """Return the seeded arm episodes ledger path used by the synthetic bundle."""
    return bundle_dir / "payload" / "runs" / "orca__holonomic" / "episodes.jsonl"


def _load_manifest(bundle_dir: Path) -> dict:
    """Read and return the publication manifest dict for mutation."""
    return json.loads((bundle_dir / "publication_manifest.json").read_text(encoding="utf-8"))


def _save_manifest(bundle_dir: Path, manifest: dict) -> None:
    """Write the publication manifest dict back to the bundle."""
    (bundle_dir / "publication_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


@pytest.mark.parametrize(
    "value",
    [
        "output/local_cache",
        "file:///tmp/rel",
        "./relative",
        "../escape",
        "https://localhost:8080/x",
        "10.5281/zenodo.{release_tag}",
    ],
)
def test_preflight_rejects_each_unresolved_channel_placeholder(tmp_path: Path, value: str) -> None:
    """Every documented placeholder shape in publication_channels must fail closed."""
    bundle_dir = _build_bundle(tmp_path)
    manifest = _load_manifest(bundle_dir)
    manifest["publication_channels"]["release_url"] = value
    _save_manifest(bundle_dir, manifest)
    with pytest.raises(PublicationPreflightError, match="retains an unresolved placeholder"):
        verify_publication_bundle_preflight(bundle_dir)


def test_preflight_ignores_empty_channel_value(tmp_path: Path) -> None:
    """An empty publication_channels value is not a placeholder and must not fail."""
    bundle_dir = _build_bundle(tmp_path)
    manifest = _load_manifest(bundle_dir)
    manifest["publication_channels"]["notes"] = ""
    _save_manifest(bundle_dir, manifest)
    report = verify_publication_bundle_preflight(bundle_dir)
    assert report["status"] == "pass"


def test_preflight_warns_when_publication_channels_omitted(tmp_path: Path) -> None:
    """A missing publication_channels block is a warning, not a blocking violation."""
    bundle_dir = _build_bundle(tmp_path)
    manifest = _load_manifest(bundle_dir)
    manifest.pop("publication_channels", None)
    _save_manifest(bundle_dir, manifest)
    report = verify_publication_bundle_preflight(bundle_dir)
    assert report["status"] == "pass"
    assert any("omits publication_channels" in w for w in report["warnings"])


def test_preflight_tolerates_non_dict_publication_channels(tmp_path: Path) -> None:
    """A present but non-dict publication_channels block is neither warning nor violation."""
    bundle_dir = _build_bundle(tmp_path)
    manifest = _load_manifest(bundle_dir)
    manifest["publication_channels"] = "not-a-dict"
    _save_manifest(bundle_dir, manifest)
    report = verify_publication_bundle_preflight(bundle_dir)
    assert report["status"] == "pass"
    assert not any("publication_channels" in w for w in report["warnings"])


def test_preflight_accepts_checksums_with_comments_and_blank_lines(tmp_path: Path) -> None:
    """Comment and blank lines in checksums.sha256 are skipped during parsing."""
    bundle_dir = _build_bundle(tmp_path)
    checksums_path = bundle_dir / "checksums.sha256"
    lines = checksums_path.read_text(encoding="utf-8").splitlines()
    rewritten = ["# header comment", "", *lines, "", "# trailing comment", ""]
    checksums_path.write_text("\n".join(rewritten) + "\n", encoding="utf-8")
    report = verify_publication_bundle_preflight(bundle_dir)
    assert report["status"] == "pass"


@pytest.mark.parametrize(
    ("body", "match"),
    [
        ("abcdef  payload/manifest.json\n", "malformed checksum entry"),
        ("g" * 64 + "  payload/manifest.json\n", "malformed checksum entry"),
        ("# only a comment\n\n# trailing\n", None),
    ],
    ids=["short-digest", "nonhex-digest", "no-entries"],
)
def test_preflight_fails_on_malformed_checksum_file(tmp_path: Path, body: str, match: str) -> None:
    """Malformed checksum lines surface as fail-closed checksum violations."""
    bundle_dir = _build_bundle(tmp_path)
    (bundle_dir / "checksums.sha256").write_text(body, encoding="utf-8")
    with pytest.raises(PublicationPreflightError, match="checksums.sha256 cannot be validated"):
        verify_publication_bundle_preflight(bundle_dir)


def test_preflight_fails_on_duplicate_checksum_path(tmp_path: Path) -> None:
    """Duplicate paths in checksums.sha256 must be rejected."""
    bundle_dir = _build_bundle(tmp_path)
    checksums_path = bundle_dir / "checksums.sha256"
    first_line = checksums_path.read_text(encoding="utf-8").splitlines()[0]
    checksums_path.write_text(first_line + "\n" + first_line + "\n", encoding="utf-8")
    with pytest.raises(PublicationPreflightError, match="duplicate path"):
        verify_publication_bundle_preflight(bundle_dir)


def test_preflight_fails_on_checksum_path_escaping_bundle_root(tmp_path: Path) -> None:
    """A checksum path with ``..`` must not reach outside the bundle root."""
    bundle_dir = _build_bundle(tmp_path)
    checksums_path = bundle_dir / "checksums.sha256"
    first_line = checksums_path.read_text(encoding="utf-8").splitlines()[0]
    escaped = f"{'0' * 64}  ../escape.txt\n"
    checksums_path.write_text(first_line + "\n" + escaped, encoding="utf-8")
    with pytest.raises(PublicationPreflightError, match="escapes bundle root"):
        verify_publication_bundle_preflight(bundle_dir)


def test_preflight_fails_on_checksum_digest_drift(tmp_path: Path) -> None:
    """A checksum that does not recompute against its bundle file must fail."""
    bundle_dir = _build_bundle(tmp_path)
    checksums_path = bundle_dir / "checksums.sha256"
    lines = checksums_path.read_text(encoding="utf-8").splitlines()
    sha, path = lines[0].split(maxsplit=1)
    tampered = list(sha)
    tampered[0] = "1" if tampered[0] == "0" else "0"
    lines[0] = "".join(tampered) + "  " + path
    checksums_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(PublicationPreflightError, match="checksum mismatch at bundle root"):
        verify_publication_bundle_preflight(bundle_dir)


def test_preflight_fails_when_manifest_files_not_a_list(tmp_path: Path) -> None:
    """publication_manifest files must be a list; otherwise fail closed."""
    bundle_dir = _build_bundle(tmp_path)
    manifest = _load_manifest(bundle_dir)
    manifest["files"] = "not-a-list"
    _save_manifest(bundle_dir, manifest)
    with pytest.raises(PublicationPreflightError, match="files must be a list"):
        verify_publication_bundle_preflight(bundle_dir)


def test_preflight_fails_on_non_object_manifest_file_entry(tmp_path: Path) -> None:
    """A non-object entry in manifest files is reported as a violation."""
    bundle_dir = _build_bundle(tmp_path)
    manifest = _load_manifest(bundle_dir)
    manifest["files"] = ["not-an-object", *manifest["files"]]
    _save_manifest(bundle_dir, manifest)
    with pytest.raises(PublicationPreflightError, match="files must contain objects"):
        verify_publication_bundle_preflight(bundle_dir)


def test_preflight_fails_on_manifest_file_entry_missing_path(tmp_path: Path) -> None:
    """A manifest file entry without a non-empty path is rejected."""
    bundle_dir = _build_bundle(tmp_path)
    manifest = _load_manifest(bundle_dir)
    manifest["files"] = [{"sha256": "0" * 64}, *manifest["files"]]
    _save_manifest(bundle_dir, manifest)
    with pytest.raises(PublicationPreflightError, match="path must be a non-empty string"):
        verify_publication_bundle_preflight(bundle_dir)


def test_preflight_fails_on_malformed_manifest_sha256(tmp_path: Path) -> None:
    """A manifest file sha256 of the wrong length is rejected."""
    bundle_dir = _build_bundle(tmp_path)
    manifest = _load_manifest(bundle_dir)
    manifest["files"][0]["sha256"] = "0" * 10
    _save_manifest(bundle_dir, manifest)
    with pytest.raises(PublicationPreflightError, match="sha256 is malformed for file"):
        verify_publication_bundle_preflight(bundle_dir)


def test_preflight_fails_on_duplicate_manifest_file(tmp_path: Path) -> None:
    """Duplicate manifest file entries (same path) are rejected."""
    bundle_dir = _build_bundle(tmp_path)
    manifest = _load_manifest(bundle_dir)
    duplicate = dict(manifest["files"][0])
    manifest["files"].append(duplicate)
    _save_manifest(bundle_dir, manifest)
    with pytest.raises(PublicationPreflightError, match="duplicate publication manifest file"):
        verify_publication_bundle_preflight(bundle_dir)


def test_preflight_fails_on_release_result_not_a_json_object(tmp_path: Path) -> None:
    """A release_result.json that is valid JSON but not an object must fail closed."""
    bundle_dir = _build_bundle(tmp_path)
    (bundle_dir / "payload" / "release" / "release_result.json").write_text(
        "[1, 2]\n", encoding="utf-8"
    )
    with pytest.raises(
        PublicationPreflightError, match="release reconciliation cannot be validated"
    ):
        verify_publication_bundle_preflight(bundle_dir)


def test_preflight_fails_when_campaign_summary_has_no_campaign_block(tmp_path: Path) -> None:
    """A campaign_summary without an object campaign block is rejected."""
    bundle_dir = _build_bundle(tmp_path)
    (bundle_dir / "payload" / "reports" / "campaign_summary.json").write_text(
        json.dumps({"not_a_campaign": True}), encoding="utf-8"
    )
    with pytest.raises(
        PublicationPreflightError, match="campaign block is missing or not an object"
    ):
        verify_publication_bundle_preflight(bundle_dir)


def test_preflight_fails_on_episode_row_that_is_not_an_object(tmp_path: Path) -> None:
    """An episode row that is valid JSON but not an object must be rejected."""
    bundle_dir = _build_bundle(tmp_path)
    _episodes_path(bundle_dir).write_text("[1, 2]\n", encoding="utf-8")
    with pytest.raises(PublicationPreflightError, match="episode row must be an object"):
        verify_publication_bundle_preflight(bundle_dir)


def test_preflight_fails_on_episode_ledger_without_software_commit(tmp_path: Path) -> None:
    """An event_ledger missing its software_commit is rejected, not silently dropped."""
    bundle_dir = _build_bundle(tmp_path)
    _episodes_path(bundle_dir).write_text(
        json.dumps({"episode_id": "ep-1", "event_ledger": {}}) + "\n", encoding="utf-8"
    )
    with pytest.raises(PublicationPreflightError, match="event_ledger.software_commit is required"):
        verify_publication_bundle_preflight(bundle_dir)


def test_preflight_fails_when_no_episode_ledger_files_present(tmp_path: Path) -> None:
    """A payload with no runs/*/episodes.jsonl files reports a provenance violation."""
    bundle_dir = _build_bundle(tmp_path)
    _episodes_path(bundle_dir).unlink()
    with pytest.raises(PublicationPreflightError, match="publication payload contains no runs"):
        verify_publication_bundle_preflight(bundle_dir)


def test_preflight_handles_unreadable_episode_ledger(tmp_path: Path) -> None:
    """An unreadable episode ledger is a fail-closed violation, not a crash."""
    bundle_dir = _build_bundle(tmp_path)
    ledger = _episodes_path(bundle_dir)
    ledger.chmod(0o000)
    try:
        with pytest.raises(PublicationPreflightError, match="cannot read episode ledger"):
            verify_publication_bundle_preflight(bundle_dir)
    finally:
        ledger.chmod(0o644)


def test_preflight_skips_blank_episode_rows(tmp_path: Path) -> None:
    """Blank lines between episode rows are skipped by both ledger passes."""
    bundle_dir = _build_bundle(tmp_path)
    ledger = _episodes_path(bundle_dir)
    original = ledger.read_text(encoding="utf-8")
    ledger.write_text("\n" + original + "\n\n", encoding="utf-8")
    report = verify_publication_bundle_preflight(bundle_dir)
    assert report["status"] == "pass"


def test_preflight_accepts_goal_only_row_without_timeout(tmp_path: Path) -> None:
    """An exact_events row with goal_reached but not timeout is not ambiguous."""
    bundle_dir = _build_bundle(tmp_path)
    _episodes_path(bundle_dir).write_text(
        json.dumps(
            {
                "episode_id": "ep-2",
                "event_ledger": {
                    "software_commit": "abc123",
                    "exact_events": {"goal_reached": True, "timeout": False},
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    report = verify_publication_bundle_preflight(bundle_dir)
    assert report["status"] == "pass"
    assert report["evidence"]["goal_reached_timeout_rows"] == 0


def test_preflight_accepts_goal_timeout_row_with_boundary_note(tmp_path: Path) -> None:
    """A goal_reached+timeout row is acceptable with an explicit boundary note."""
    bundle_dir = _build_bundle(tmp_path)
    _episodes_path(bundle_dir).write_text(
        json.dumps(
            {
                "episode_id": "ep-2",
                "goal_timeout_boundary_note": "reached the goal on the final timeout step",
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


def test_preflight_fails_when_provenance_repository_commit_missing(tmp_path: Path) -> None:
    """A missing provenance.repository.commit is a blocking violation."""
    bundle_dir = _build_bundle(tmp_path)
    manifest = _load_manifest(bundle_dir)
    manifest.pop("provenance", None)
    _save_manifest(bundle_dir, manifest)
    with pytest.raises(PublicationPreflightError, match="provenance.repository.commit is missing"):
        verify_publication_bundle_preflight(bundle_dir)


def test_preflight_fails_when_provenance_repository_block_missing(tmp_path: Path) -> None:
    """A provenance dict without a repository block is rejected."""
    bundle_dir = _build_bundle(tmp_path)
    manifest = _load_manifest(bundle_dir)
    manifest["provenance"] = {}
    _save_manifest(bundle_dir, manifest)
    with pytest.raises(PublicationPreflightError, match="provenance.repository.commit is missing"):
        verify_publication_bundle_preflight(bundle_dir)


def test_preflight_fails_when_repository_commit_is_empty(tmp_path: Path) -> None:
    """An empty repository commit string is treated as missing."""
    bundle_dir = _build_bundle(tmp_path)
    manifest = _load_manifest(bundle_dir)
    manifest.setdefault("provenance", {}).setdefault("repository", {})["commit"] = ""
    _save_manifest(bundle_dir, manifest)
    with pytest.raises(PublicationPreflightError, match="provenance.repository.commit is missing"):
        verify_publication_bundle_preflight(bundle_dir)


def test_preflight_fails_on_missing_manifest(tmp_path: Path) -> None:
    """A missing publication_manifest.json raises immediately."""
    bundle_dir = _build_bundle(tmp_path)
    (bundle_dir / "publication_manifest.json").unlink()
    with pytest.raises(PublicationPreflightError, match="Publication preflight failed: missing"):
        verify_publication_bundle_preflight(bundle_dir)


def test_preflight_fails_on_missing_checksums(tmp_path: Path) -> None:
    """A missing checksums.sha256 raises immediately."""
    bundle_dir = _build_bundle(tmp_path)
    (bundle_dir / "checksums.sha256").unlink()
    with pytest.raises(PublicationPreflightError, match="Publication preflight failed: missing"):
        verify_publication_bundle_preflight(bundle_dir)


def test_preflight_fails_on_invalid_manifest_json(tmp_path: Path) -> None:
    """A malformed publication_manifest.json fails closed with a structured error."""
    bundle_dir = _build_bundle(tmp_path)
    (bundle_dir / "publication_manifest.json").write_text("{not-json}\n", encoding="utf-8")
    with pytest.raises(PublicationPreflightError, match="Publication preflight failed:"):
        verify_publication_bundle_preflight(bundle_dir)


def test_preflight_fails_when_manifest_missing_schema_version(tmp_path: Path) -> None:
    """A manifest without a schema_version is reported as a violation."""
    bundle_dir = _build_bundle(tmp_path)
    manifest = _load_manifest(bundle_dir)
    manifest.pop("schema_version", None)
    _save_manifest(bundle_dir, manifest)
    with pytest.raises(PublicationPreflightError, match="missing schema_version"):
        verify_publication_bundle_preflight(bundle_dir)


# ---------------------------------------------------------------------------
# Issue #5580: per-episode SNQI field vs diagnostics basis (release-gate self-check)
# ---------------------------------------------------------------------------

from robot_sf.benchmark.metrics import snqi as curvature_aware_snqi  # noqa: E402
from robot_sf.benchmark.snqi_scalarization_sensitivity import (  # noqa: E402
    load_baseline_mapping,
    load_weight_mapping,
)
from robot_sf.common.artifact_paths import get_repository_root  # noqa: E402

_SNQI_WEIGHTS_PATH = (
    get_repository_root() / "configs" / "benchmarks" / "snqi_weights_camera_ready_v3.json"
)
_SNQI_BASELINE_PATH = (
    get_repository_root() / "configs" / "benchmarks" / "snqi_baseline_camera_ready_v3.json"
)


def _snqi_episode(*, success: bool, seed: int) -> dict:
    """Build one episode whose stored metrics.snqi is the curvature-aware scalarizer output."""
    weights = load_weight_mapping(_SNQI_WEIGHTS_PATH)
    baseline = load_baseline_mapping(_SNQI_BASELINE_PATH)
    metrics = {
        "collisions": 0 if success else 1,
        "success": success,
        "time_to_goal_norm": 0.5 if success else 1.0,
        "near_misses": 0,
        "comfort_exposure": 0.0,
        "force_exceed_events": 0,
        "jerk_mean": 0.0,
        "curvature_mean": 0.1,
    }
    metrics["snqi"] = curvature_aware_snqi(metrics, weights, baseline_stats=baseline)
    return {
        "episode_id": f"scenario--{seed}",
        "scenario_id": "scenario",
        "seed": seed,
        "event_ledger": {"software_commit": "abc123"},
        "metrics": metrics,
    }


def _seed_snqi_diagnostics(bundle_dir: Path, ordering: list[dict]) -> None:
    """Write payload/reports/snqi_diagnostics.json with canonical weights/baseline provenance."""
    weights_sha256 = hashlib.sha256(_SNQI_WEIGHTS_PATH.read_bytes()).hexdigest()
    baseline_sha256 = hashlib.sha256(_SNQI_BASELINE_PATH.read_bytes()).hexdigest()
    diagnostics = {
        "weights_version": "snqi_weights_camera_ready_v3",
        "baseline_version": "snqi_baseline_camera_ready_v3",
        "weights_sha256": weights_sha256,
        "baseline_sha256": baseline_sha256,
        "planner_ordering": ordering,
    }
    _write(
        bundle_dir / "payload" / "reports" / "snqi_diagnostics.json",
        json.dumps(diagnostics) + "\n",
    )


def _seed_snqi_arm(bundle_dir: Path, arm: str, rows: list[dict]) -> None:
    """Write payload/runs/<arm>/episodes.jsonl for one arm."""
    episodes_path = bundle_dir / "payload" / "runs" / arm / "episodes.jsonl"
    episodes_path.parent.mkdir(parents=True, exist_ok=True)
    episodes_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_preflight_snqi_check_skipped_without_diagnostics(tmp_path: Path) -> None:
    """A bundle without snqi_diagnostics.json skips the SNQI self-check (checked=False)."""
    bundle_dir = _build_bundle(tmp_path)
    report = verify_publication_bundle_preflight(bundle_dir)
    assert report["status"] == "pass"
    assert report["evidence"]["snqi_field_consistency"]["checked"] is False


def test_preflight_snqi_check_passes_on_consistent_bundle(tmp_path: Path) -> None:
    """A bundle whose stored field matches the curvature-aware recompute passes the SNQI gate."""
    bundle_dir = _build_bundle(tmp_path)
    rows = [_snqi_episode(success=True, seed=1), _snqi_episode(success=True, seed=2)]
    _seed_snqi_arm(bundle_dir, "orca__holonomic", rows)
    mean = sum(r["metrics"]["snqi"] for r in rows) / len(rows)
    _seed_snqi_diagnostics(
        bundle_dir,
        [
            {
                "planner_key": "orca",
                "kinematics": "holonomic",
                "episode_count": len(rows),
                "mean_snqi": mean,
                "rank": 1,
            }
        ],
    )

    report = verify_publication_bundle_preflight(bundle_dir)
    assert report["status"] == "pass"
    snqi = report["evidence"]["snqi_field_consistency"]
    assert snqi["checked"] is True
    assert snqi["violation_count"] == 0
    assert snqi["counts"]["rows"] == 2
    assert snqi["counts"]["episode_field_present"] == 2
    assert snqi["ordering"]["field_planner_ordering"] == {"orca::holonomic": 1}


def test_preflight_snqi_check_fails_on_drifted_stored_field(tmp_path: Path) -> None:
    """A stored metrics.snqi that disagrees with the curvature-aware recompute fails closed.

    This is the issue #5580 drift scenario: the per-episode field was baked under a different
    basis than the diagnostics declare, so the recomputation disagrees with the stored value.
    """
    bundle_dir = _build_bundle(tmp_path)
    rows = [_snqi_episode(success=True, seed=1)]
    rows[0]["metrics"]["snqi"] += 0.5  # corrupt the stored field
    _seed_snqi_arm(bundle_dir, "orca__holonomic", rows)
    mean = sum(r["metrics"]["snqi"] for r in rows) / len(rows)
    _seed_snqi_diagnostics(
        bundle_dir,
        [
            {
                "planner_key": "orca",
                "kinematics": "holonomic",
                "episode_count": len(rows),
                "mean_snqi": mean,
                "rank": 1,
            }
        ],
    )

    with pytest.raises(PublicationPreflightError, match="stored SNQI"):
        verify_publication_bundle_preflight(bundle_dir)


def test_preflight_snqi_check_fails_on_ordering_disagreement(tmp_path: Path) -> None:
    """A field-derived arm ordering that disagrees with diagnostics planner_ordering fails.

    Reproduces the #5580 consequence: aggregates built from the per-episode field can elect a
    different SNQI-best arm than the diagnostics ordering when the two surfaces have drifted.
    """
    bundle_dir = _build_bundle(tmp_path)
    good_rows = [_snqi_episode(success=True, seed=1), _snqi_episode(success=True, seed=2)]
    bad_rows = [_snqi_episode(success=False, seed=3)]
    _seed_snqi_arm(bundle_dir, "orca__holonomic", good_rows)
    _seed_snqi_arm(bundle_dir, "social_force__differential_drive", bad_rows)
    good_mean = sum(r["metrics"]["snqi"] for r in good_rows) / len(good_rows)
    bad_mean = bad_rows[0]["metrics"]["snqi"]
    # Diagnostics declare social_force as the winner (rank 1), but the curvature-aware field
    # basis ranks orca first -> the two surfaces disagree and the gate must fail closed.
    _seed_snqi_diagnostics(
        bundle_dir,
        [
            {
                "planner_key": "social_force",
                "kinematics": "differential_drive",
                "episode_count": len(bad_rows),
                "mean_snqi": bad_mean,
                "rank": 1,
            },
            {
                "planner_key": "orca",
                "kinematics": "holonomic",
                "episode_count": len(good_rows),
                "mean_snqi": good_mean,
                "rank": 2,
            },
        ],
    )

    with pytest.raises(PublicationPreflightError, match="arm ordering disagrees"):
        verify_publication_bundle_preflight(bundle_dir)


def test_preflight_snqi_check_fails_on_provenance_sha_mismatch(tmp_path: Path) -> None:
    """Diagnostics declaring a non-canonical weights/baseline sha fails closed."""
    bundle_dir = _build_bundle(tmp_path)
    rows = [_snqi_episode(success=True, seed=1)]
    _seed_snqi_arm(bundle_dir, "orca__holonomic", rows)
    weights_sha256 = hashlib.sha256(_SNQI_WEIGHTS_PATH.read_bytes()).hexdigest()
    baseline_sha256 = hashlib.sha256(_SNQI_BASELINE_PATH.read_bytes()).hexdigest()
    diagnostics = {
        "weights_version": "snqi_weights_camera_ready_v3",
        "baseline_version": "snqi_baseline_camera_ready_v3",
        "weights_sha256": "0" * 64,  # wrong
        "baseline_sha256": baseline_sha256,
        "planner_ordering": [
            {
                "planner_key": "orca",
                "kinematics": "holonomic",
                "episode_count": len(rows),
                "mean_snqi": rows[0]["metrics"]["snqi"],
                "rank": 1,
            }
        ],
    }
    _write(
        bundle_dir / "payload" / "reports" / "snqi_diagnostics.json",
        json.dumps(diagnostics) + "\n",
    )
    # Reference the unused sha to keep the linter honest about intent.
    assert weights_sha256 != "0" * 64

    with pytest.raises(PublicationPreflightError, match="weights_sha256 does not match"):
        verify_publication_bundle_preflight(bundle_dir)
