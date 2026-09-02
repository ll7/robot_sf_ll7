"""Tests for benchmark publication bundle helpers."""

# evidence-writer-exempt: these tests write only synthetic pytest tmp_path fixtures, never
# tracked or durable benchmark evidence.

from __future__ import annotations

import json
import os
import tarfile
from pathlib import Path
from typing import Any

import pytest

from robot_sf.benchmark import artifact_publication as artifact_publication_module
from robot_sf.benchmark.artifact_publication import (
    _SNQI_DEFAULT_BASELINE_NAME,
    _SNQI_DEFAULT_WEIGHTS_NAME,
    PUBLICATION_BUNDLE_SCHEMA_VERSION,
    RELEASE_PUBLICATION_METADATA_SCHEMA_VERSION,
    SIZE_REPORT_SCHEMA_VERSION,
    _build_rights_provenance_statement,
    _check_goal_timeout_boundary,
    _compute_and_emit_badging_artifacts,
    _find_release_sha,
    _preflight_check_release_metadata,
    _resolve_release_publication_metadata,
    _resolve_repo_file,
    _resolve_run_file,
    _snqi_load_canonical_basis,
    discover_run_directories,
    export_publication_bundle,
    list_publication_files,
    measure_artifact_size_ranges,
)


def _write(path: Path, payload: str) -> None:
    """Write UTF-8 text payload to a file, creating parents as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def _make_run(run_dir: Path, *, with_video: bool = True) -> None:
    """Create a minimal synthetic benchmark run directory for publication tests."""
    _write(
        run_dir / "manifest.json", json.dumps({"git_hash": "abc123", "scenario_matrix_hash": "m1"})
    )
    _write(
        run_dir / "run_meta.json",
        json.dumps({"repo": {"remote": "git@github.com:ll7/robot_sf_ll7.git", "commit": "abc123"}}),
    )
    _write(run_dir / "episodes" / "episodes.jsonl", '{"episode_id":"ep-1"}\n')
    _write(run_dir / "aggregates" / "summary.json", '{"success_rate":1.0}\n')
    _write(run_dir / "reports" / "report.md", "# Report\n")
    _write(run_dir / "plots" / "path_efficiency.pdf", "fake-pdf")
    if with_video:
        _write(run_dir / "videos" / "episode_001.mp4", "fake-video")


def test_list_publication_files_respects_video_toggle(tmp_path: Path) -> None:
    """Video toggle should include or exclude video paths from selection."""
    run_dir = tmp_path / "run_a"
    _make_run(run_dir, with_video=True)
    with_videos = list_publication_files(run_dir, include_videos=True)
    without_videos = list_publication_files(run_dir, include_videos=False)

    assert any(path.as_posix().startswith("videos/") for path in with_videos)
    assert not any(path.as_posix().startswith("videos/") for path in without_videos)


def test_goal_timeout_boundary_accepts_exact_signed_provenance_exclusion(
    tmp_path: Path,
) -> None:
    """A complete signed exclusion permits unchanged ambiguous scientific rows."""
    payload = tmp_path / "payload"
    arm = "guarded_ppo__differential_drive"
    episode_id = "scenario--132--identity"
    row = {
        "episode_id": episode_id,
        "event_ledger": {"exact_events": {"goal_reached": True, "timeout": True}},
    }
    _write(payload / "runs" / arm / "episodes.jsonl", json.dumps(row) + "\n")
    _write(
        payload / "run_meta.json",
        json.dumps(
            {
                "goal_timeout_boundary": {
                    "status": "excluded_from_timing_interpretation",
                    "excluded_row_count": 1,
                    "excluded_rows": [{"arm": arm, "episode_id": episode_id}],
                    "raw_episode_rows_unchanged": True,
                    "timing_evidence_fabricated": False,
                    "note": "Exact event ordering is unavailable.",
                    "policy": "Exclude this reviewed row from timing interpretation.",
                }
            }
        ),
    )

    count, rejections = _check_goal_timeout_boundary(payload)

    assert count == 1
    assert rejections == []


@pytest.mark.parametrize("varying_field", ["scenario_id", "seed"])
def test_goal_timeout_boundary_rejects_duplicate_identity_rows(
    tmp_path: Path, varying_field: str
) -> None:
    """One exclusion cannot collapse two rows sharing an arm and episode ID."""
    payload = tmp_path / "payload"
    arm = "guarded_ppo__differential_drive"
    episode_id = "scenario--132--identity"
    first = {
        "episode_id": episode_id,
        "scenario_id": "scenario-a",
        "seed": 1,
        "event_ledger": {"exact_events": {"goal_reached": True, "timeout": True}},
    }
    second = dict(first)
    second[varying_field] = "scenario-b" if varying_field == "scenario_id" else 2
    _write(
        payload / "runs" / arm / "episodes.jsonl",
        json.dumps(first) + "\n" + json.dumps(second) + "\n",
    )
    _write(
        payload / "run_meta.json",
        json.dumps(
            {
                "goal_timeout_boundary": {
                    "status": "excluded_from_timing_interpretation",
                    "excluded_row_count": 1,
                    "excluded_rows": [{"arm": arm, "episode_id": episode_id}],
                    "raw_episode_rows_unchanged": True,
                    "timing_evidence_fabricated": False,
                    "note": "Exact event ordering is unavailable.",
                    "policy": "Exclude reviewed rows from timing interpretation.",
                }
            }
        ),
    )

    _count, rejections = _check_goal_timeout_boundary(payload)

    assert any("duplicate ambiguous goal+timeout identity" in item for item in rejections)


@pytest.mark.parametrize("drift", ["missing", "extra", "mutated"])
def test_goal_timeout_boundary_rejects_incomplete_or_mutating_exclusion(
    tmp_path: Path, drift: str
) -> None:
    """Signed provenance cannot omit, invent, or claim mutation of scientific rows."""
    payload = tmp_path / "payload"
    arm = "guarded_ppo__differential_drive"
    episode_id = "scenario--132--identity"
    row = {
        "episode_id": episode_id,
        "event_ledger": {"exact_events": {"goal_reached": True, "timeout": True}},
    }
    _write(payload / "runs" / arm / "episodes.jsonl", json.dumps(row) + "\n")
    declared = [] if drift == "missing" else [{"arm": arm, "episode_id": episode_id}]
    if drift == "extra":
        declared.append({"arm": arm, "episode_id": "not-ambiguous"})
    _write(
        payload / "run_meta.json",
        json.dumps(
            {
                "goal_timeout_boundary": {
                    "status": "excluded_from_timing_interpretation",
                    "excluded_row_count": len(declared),
                    "excluded_rows": declared,
                    "raw_episode_rows_unchanged": drift != "mutated",
                    "timing_evidence_fabricated": False,
                    "note": "Exact event ordering is unavailable.",
                    "policy": "Exclude reviewed rows from timing interpretation.",
                }
            }
        ),
    )

    _count, rejections = _check_goal_timeout_boundary(payload)

    assert rejections


def test_goal_timeout_exclusion_parser_fails_closed_without_resolved_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An inconsistent row parser result cannot admit an unsigned identity."""
    monkeypatch.setattr(
        artifact_publication_module,
        "_goal_timeout_exclusion_identity",
        lambda *_args, **_kwargs: (None, None),
    )

    identities, errors = artifact_publication_module._parse_goal_timeout_exclusion_rows(
        {"excluded_rows": [{}], "excluded_row_count": 0}
    )

    assert identities == set()
    assert errors == ["run_meta goal-timeout exclusion row 0 has no resolved identity"]


def test_discover_run_directories_returns_leaf_runs(tmp_path: Path) -> None:
    """Discovery should return leaf marker directories and avoid parent duplicates."""
    root = tmp_path / "benchmarks"
    _make_run(root / "seed_holdout" / "ppo")
    _make_run(root / "seed_holdout" / "orca")

    runs = discover_run_directories(root)
    run_names = {path.name for path in runs}
    assert run_names == {"ppo", "orca"}


def test_measure_artifact_size_ranges_reports_schema_and_counts(tmp_path: Path) -> None:
    """Size report should expose schema metadata and per-run totals."""
    root = tmp_path / "benchmarks"
    _make_run(root / "run_small", with_video=False)
    _make_run(root / "run_large", with_video=True)

    report = measure_artifact_size_ranges(root, include_videos=False)
    assert report["schema_version"] == SIZE_REPORT_SCHEMA_VERSION
    assert report["run_count"] == 2
    assert report["distributions"]["total_bytes"]["count"] == 2
    assert not str(report["benchmarks_root"]).startswith("/")
    assert all(not str(run["run_dir"]).startswith("/") for run in report["runs"])


def test_export_publication_bundle_writes_manifest_checksums_and_archive(tmp_path: Path) -> None:
    """Export should emit a DOI-ready bundle directory and compressed archive."""
    run_dir = tmp_path / "benchmarks" / "run_export"
    _make_run(run_dir, with_video=False)
    out_dir = tmp_path / "publication"

    result = export_publication_bundle(
        run_dir,
        out_dir,
        bundle_name="run_export_bundle",
        include_videos=False,
    )

    assert result.bundle_dir.exists()
    assert result.archive_path.exists()
    assert result.manifest_path.exists()
    assert result.checksums_path.exists()
    assert result.file_count > 0
    assert result.total_bytes > 0

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == PUBLICATION_BUNDLE_SCHEMA_VERSION
    assert manifest["totals"]["file_count"] == result.file_count
    assert len(manifest["files"]) == result.file_count
    assert all("sha256" in entry for entry in manifest["files"])

    checksum_lines = [
        line
        for line in result.checksums_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(checksum_lines) == result.file_count
    assert all("  payload/" in line for line in checksum_lines)

    with tarfile.open(result.archive_path, "r:gz") as handle:
        names = handle.getnames()
    assert any(name.endswith("publication_manifest.json") for name in names)
    assert any(name.endswith("checksums.sha256") for name in names)
    assert any("/payload/" in name for name in names)


def test_list_publication_files_skips_symlink_targets(tmp_path: Path) -> None:
    """Symlinked files inside a run must be excluded from publication payloads."""
    run_dir = tmp_path / "run_symlink"
    _make_run(run_dir, with_video=False)
    secret = tmp_path / "secret.txt"
    secret.write_text("secret", encoding="utf-8")
    link = run_dir / "reports" / "leak_link.txt"
    try:
        os.symlink(secret, link)
    except (NotImplementedError, OSError):  # pragma: no cover
        pytest.skip("Symlink creation unavailable on this platform")

    files = list_publication_files(run_dir, include_videos=False)
    assert not any(path.as_posix() == "reports/leak_link.txt" for path in files)


def test_export_publication_bundle_rejects_unsafe_bundle_names(tmp_path: Path) -> None:
    """Absolute and parent-traversal bundle names should be rejected."""
    run_dir = tmp_path / "benchmarks" / "run_invalid_name"
    _make_run(run_dir, with_video=False)
    out_dir = tmp_path / "publication"

    with pytest.raises(ValueError, match="Invalid bundle_name"):
        export_publication_bundle(run_dir, out_dir, bundle_name="/tmp/evil", overwrite=True)

    with pytest.raises(ValueError, match="Invalid bundle_name"):
        export_publication_bundle(run_dir, out_dir, bundle_name="../escape", overwrite=True)


def test_export_bundle_matrix_path_is_repo_relative_from_any_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Relative matrix paths should resolve against repository root, not current cwd."""
    run_dir = tmp_path / "benchmarks" / "run_matrix_path"
    _make_run(run_dir, with_video=False)
    matrix_path = "configs/scenarios/classic_interactions.yaml"
    _write(
        run_dir / "run_meta.json",
        json.dumps(
            {
                "repo": {"remote": "git@github.com:ll7/robot_sf_ll7.git", "commit": "abc123"},
                "matrix_path": matrix_path,
            }
        ),
    )

    cwd_outside_repo = tmp_path / "outside_repo"
    cwd_outside_repo.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(cwd_outside_repo)

    result = export_publication_bundle(
        run_dir,
        tmp_path / "publication",
        bundle_name="run_matrix_bundle",
        include_videos=False,
    )
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["provenance"]["matrix_path"] == matrix_path


def test_export_camera_campaign_bundle_requires_preflight_artifacts(tmp_path: Path) -> None:
    """Camera-ready bundle export must fail when required preflight files are missing."""
    run_dir = tmp_path / "benchmarks" / "camera_campaign_missing_preflight"
    _make_run(run_dir, with_video=False)
    _write(run_dir / "campaign_manifest.json", json.dumps({"schema_version": "camera-ready"}))

    with pytest.raises(ValueError, match="missing required preflight artifacts"):
        export_publication_bundle(
            run_dir,
            tmp_path / "publication",
            bundle_name="missing_preflight_bundle",
            include_videos=False,
        )


def test_export_camera_campaign_bundle_includes_preflight_seed_policy(tmp_path: Path) -> None:
    """Camera-ready bundle manifest should carry preflight paths and seed policy provenance."""
    run_dir = tmp_path / "benchmarks" / "camera_campaign_ok"
    _make_run(run_dir, with_video=False)
    _write(run_dir / "campaign_manifest.json", json.dumps({"schema_version": "camera-ready"}))
    _write(run_dir / "preflight" / "validate_config.json", "{}")
    _write(run_dir / "preflight" / "preview_scenarios.json", "{}")
    _write(
        run_dir / "run_meta.json",
        json.dumps(
            {
                "repo": {"remote": "git@github.com:ll7/robot_sf_ll7.git", "commit": "abc123"},
                "matrix_path": "configs/scenarios/classic_interactions.yaml",
                "seed_policy": {
                    "mode": "fixed-list",
                    "seeds": [1, 2],
                    "resolved_seeds": [1, 2],
                },
                "preflight_artifacts": {
                    "validate_config": "preflight/validate_config.json",
                    "preview_scenarios": "preflight/preview_scenarios.json",
                },
            }
        ),
    )

    result = export_publication_bundle(
        run_dir,
        tmp_path / "publication",
        bundle_name="camera_campaign_bundle",
        include_videos=False,
    )
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    provenance = manifest["provenance"]
    assert provenance["seed_policy"]["mode"] == "fixed-list"
    assert provenance["seed_policy"]["resolved_seeds"] == [1, 2]
    assert provenance["preflight_artifacts"]["validate_config"] == "preflight/validate_config.json"


def test_export_publication_bundle_includes_artifact_badging(tmp_path: Path) -> None:
    """Export bundle manifest should carry artifact badging metadata when provided.

    Without payload re-derivation evidence, caller-supplied ``functional_smoke_status``
    is metadata only and cannot elevate the badge to ``functional`` (issue #4763).
    """
    run_dir = tmp_path / "benchmarks" / "run_badging"
    _make_run(run_dir, with_video=False)
    out_dir = tmp_path / "publication"

    badging = {
        "claimed_level": "functional",
        "checklist_path": "docs/context/evidence/checklist.md",
        "claim_boundary": "reproducible-smoke-run",
        "functional_smoke_status": "passed",
        "reproduction_status": "not_run",
        "known_nondeterminism": ["Thread scheduling"],
    }

    result = export_publication_bundle(
        run_dir,
        out_dir,
        bundle_name="run_badging_bundle",
        include_videos=False,
        doi="10.5281/zenodo.1234567",
        artifact_badging=badging,
    )

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    # Without payload re-derivation, computed level is "available", not "functional",
    # even with a valid DOI and caller-supplied functional_smoke_status (issue #4763).
    assert manifest["artifact_badging"]["claimed_level"] == "available"
    # Caller-supplied status is preserved as metadata.
    assert manifest["artifact_badging"]["functional_smoke_status"] == "passed"
    assert manifest["artifact_badging"]["known_nondeterminism"] == ["Thread scheduling"]

    # Test invalid configuration raises ValueError
    invalid_badging = {
        "claimed_level": "super-reproduced",  # invalid
    }
    with pytest.raises(ValueError, match="Invalid claimed_level"):
        export_publication_bundle(
            run_dir,
            out_dir,
            bundle_name="run_invalid_badging_bundle",
            include_videos=False,
            overwrite=True,
            artifact_badging=invalid_badging,
        )


def test_export_publication_bundle_never_emits_unverified_reproduced(tmp_path: Path) -> None:
    """A caller-asserted ``reproduction_status`` must not elevate the computed badge.

    This slice runs no independent reproduction rerun, so "reproduced" can never
    be earned here. Passing ``reproduction_status="passed"`` must be treated as
    informational only; the computed badge is capped at "available" (fail-closed
    per issues #4681, #4763: no reproduction or functional claim without evidence).
    """
    run_dir = tmp_path / "benchmarks" / "run_repro"
    _make_run(run_dir, with_video=False)
    out_dir = tmp_path / "publication"

    badging = {
        "claimed_level": "reproduced",
        "functional_smoke_status": "passed",
        "reproduction_status": "passed",  # hand-asserted, unverified
    }

    result = export_publication_bundle(
        run_dir,
        out_dir,
        bundle_name="run_repro_bundle",
        include_videos=False,
        doi="10.5281/zenodo.7654321",
        artifact_badging=badging,
    )

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    # Capped at "available" despite hand-asserted reproduced/passed inputs.
    # No payload re-derivation means it can never reach "functional".
    assert manifest["artifact_badging"]["claimed_level"] == "available"
    # The raw status is still carried through as informational metadata.
    assert manifest["artifact_badging"]["reproduction_status"] == "passed"


# ── Direct unit matrix for _compute_and_emit_badging_artifacts ────────────


def _make_minimal_manifest(
    *,
    doi: str | None = None,
    release_url: str | None = None,
    files: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a minimal manifest_payload for badging tests."""
    return {
        "bundle_name": "test-bundle",
        "created_at_utc": "2024-01-01T00:00:00Z",
        "publication_channels": {
            "doi": doi,
            "release_url": release_url,
        },
        "files": files or [],
    }


class TestComputeBadgingMatrix:
    """Direct unit matrix for ``_compute_and_emit_badging_artifacts`` (issue #4763)."""

    def test_no_durable_id_yields_none(self, tmp_path: Path) -> None:
        """No DOI or release URL -> claimed_level is 'none'."""
        bundle_dir = tmp_path / "bundle"
        bundle_dir.mkdir()
        manifest = _make_minimal_manifest(
            files=[{"path": "a.txt", "size_bytes": 1, "sha256": "x", "kind": "misc"}]
        )
        badging = {}

        computed, achieved = _compute_and_emit_badging_artifacts(bundle_dir, manifest, badging)

        assert computed["claimed_level"] == "none"
        assert achieved == "none"

    def test_placeholder_doi_is_not_durable(self, tmp_path: Path) -> None:
        """Placeholders like {release_tag} or <record-id> are rejected as durable IDs."""
        bundle_dir = tmp_path / "bundle"
        bundle_dir.mkdir()
        manifest = _make_minimal_manifest(
            doi="10.5281/zenodo.<record-id>",  # placeholder
            release_url="https://github.com/x/y/releases/tag/{release_tag}",  # placeholder
            files=[{"path": "a.txt", "size_bytes": 1, "sha256": "x", "kind": "misc"}],
        )
        badging = {}

        computed, achieved = _compute_and_emit_badging_artifacts(bundle_dir, manifest, badging)

        assert computed["claimed_level"] == "none"
        assert achieved == "none"

    def test_local_paths_rejected_as_durable(self, tmp_path: Path) -> None:
        """Local output/ file:// ./ ../ and localhost paths are not durable IDs."""
        bundle_dir = tmp_path / "bundle"
        bundle_dir.mkdir()
        for doi_val in [
            "output/artifacts/v1",
            "file:///home/me",
            "./local",
            "../parent",
            "http://localhost:8080",
        ]:
            manifest = _make_minimal_manifest(
                doi=doi_val,
                files=[{"path": "a.txt", "size_bytes": 1, "sha256": "x", "kind": "misc"}],
            )
            _computed, achieved = _compute_and_emit_badging_artifacts(bundle_dir, manifest, {})
            assert achieved == "none", f"DOI {doi_val!r} should not be durable"

    def test_durable_id_no_payload_tables_yields_available(self, tmp_path: Path) -> None:
        """Durable id + files, no campaign_summary/report payload -> 'available'."""
        bundle_dir = tmp_path / "bundle"
        bundle_dir.mkdir()
        payload_dir = bundle_dir / "payload"
        payload_dir.mkdir()
        (payload_dir / "summary.json").write_text("{}", encoding="utf-8")

        manifest = _make_minimal_manifest(
            doi="10.5281/zenodo.1234567",
            files=[{"path": "summary.json", "size_bytes": 2, "sha256": "abc", "kind": "misc"}],
        )
        badging = {"functional_smoke_status": "passed"}  # caller-supplied only

        computed, achieved = _compute_and_emit_badging_artifacts(bundle_dir, manifest, badging)

        assert computed["claimed_level"] == "available"
        assert achieved == "available"

    def test_caller_smoke_passed_without_payload_not_functional(self, tmp_path: Path) -> None:
        """Caller-supplied functional_smoke_status: passed cannot yield 'functional' alone."""
        bundle_dir = tmp_path / "bundle"
        bundle_dir.mkdir()
        (bundle_dir / "payload").mkdir()

        manifest = _make_minimal_manifest(
            doi="10.5281/zenodo.1234567",
            files=[{"path": "a.txt", "size_bytes": 1, "sha256": "x", "kind": "misc"}],
        )
        badging = {"functional_smoke_status": "passed"}

        computed, achieved = _compute_and_emit_badging_artifacts(bundle_dir, manifest, badging)

        assert computed["claimed_level"] == "available"
        assert achieved == "available"

    def test_no_files_yields_none_even_with_doi(self, tmp_path: Path) -> None:
        """A durable id with an empty files list still yields 'none'."""
        bundle_dir = tmp_path / "bundle"
        bundle_dir.mkdir()
        manifest = _make_minimal_manifest(doi="10.5281/zenodo.1234567", files=[])

        computed, achieved = _compute_and_emit_badging_artifacts(bundle_dir, manifest, {})

        assert computed["claimed_level"] == "none"
        assert achieved == "none"

    def test_reproduction_passed_does_not_elevate(self, tmp_path: Path) -> None:
        """reproduction_status: passed alone does not yield 'reproduced' or 'functional'."""
        bundle_dir = tmp_path / "bundle"
        bundle_dir.mkdir()
        (bundle_dir / "payload").mkdir()

        manifest = _make_minimal_manifest(
            doi="10.5281/zenodo.1234567",
            files=[{"path": "a.txt", "size_bytes": 1, "sha256": "x", "kind": "misc"}],
        )
        badging = {"reproduction_status": "passed"}

        computed, achieved = _compute_and_emit_badging_artifacts(bundle_dir, manifest, badging)

        assert computed["claimed_level"] == "available"
        assert achieved == "available"
        # Raw status preserved
        assert computed["reproduction_status"] == "passed"


# ── Durable-id placeholder test for existing test ──────────────────────────


def test_export_publication_bundle_placeholder_doi_yields_none(tmp_path: Path) -> None:
    """Default placeholder DOI should result in 'none' badge level (issue #4763)."""
    run_dir = tmp_path / "benchmarks" / "run_placeholder"
    _make_run(run_dir, with_video=False)
    out_dir = tmp_path / "publication"

    result = export_publication_bundle(
        run_dir,
        out_dir,
        bundle_name="run_placeholder_bundle",
        include_videos=False,
        # Uses default DOI template 10.5281/zenodo.<record-id>
    )

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["artifact_badging"]["claimed_level"] == "none"


def test_release_bundle_stages_cold_verification_metadata_and_raw_policy(tmp_path: Path) -> None:
    """A completed release export carries metadata and pinned SNQI inputs for cold checks."""
    run_dir = tmp_path / "benchmarks" / "release_export"
    _make_run(run_dir, with_video=True)
    _write(
        run_dir / "run_meta.json",
        json.dumps(
            {
                "repo": {
                    "remote": "git@github.com:ll7/robot_sf_ll7.git",
                    "commit": "a" * 40,
                }
            }
        ),
    )
    _write(
        run_dir / "release" / "release_manifest.resolved.json",
        json.dumps(
            {
                "schema_version": "benchmark-release-manifest.v0.2",
                "release_id": "release_export",
                "release_tag": "paper-matrix-v2-h600-s30-2026-08-abc123456789",
                "metrics": {
                    "snqi_weights_path": "configs/benchmarks/snqi_weights_camera_ready_v3.json",
                    "snqi_baseline_path": "configs/benchmarks/snqi_baseline_camera_ready_v3.json",
                },
                "provenance": {
                    "repository_url": "https://github.com/ll7/robot_sf_ll7",
                    "doi": "10.5281/zenodo.1234567",
                    "citation_path": "CITATION.cff",
                },
            }
        ),
    )
    _write(
        run_dir / "release" / "release_result.json",
        json.dumps({"status": "ok", "source_commit": "a" * 40}),
    )

    result = export_publication_bundle(
        run_dir,
        tmp_path / "publication",
        bundle_name="release_export_bundle",
        include_videos=False,
        release_tag="paper-matrix-v2-h600-s30-2026-08-abc123456789",
        doi="10.5281/zenodo.1234567",
    )
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    metadata = manifest["release_metadata"]

    assert metadata["schema_version"] == "benchmark-release-publication-metadata.v1"
    assert metadata["required"] is True
    assert set(metadata["files"]) == {
        "release_manifest",
        "release_result",
        "citation",
        "zenodo_metadata",
        "rights_provenance",
        "snqi_weights",
        "snqi_baseline",
    }
    assert metadata["raw_artifact_policy"]["campaign_output"] == "durable-required"
    assert metadata["cold_verification"]["credentials"] == "not_recorded"
    citation = result.bundle_dir / "payload" / "release_metadata" / "CITATION.cff"
    assert citation.is_file()
    assert (
        citation.read_bytes()
        == (artifact_publication_module.get_repository_root() / "CITATION.cff").read_bytes()
    )
    assert (result.bundle_dir / "payload" / "release_metadata" / "zenodo_metadata.json").is_file()
    rights = result.bundle_dir / "payload" / "release_metadata" / "rights_provenance.md"
    assert "SNQI" in rights.read_text(encoding="utf-8")
    assert "a" * 40 in rights.read_text(encoding="utf-8")
    assert (
        result.bundle_dir / "payload" / "release_metadata" / "snqi" / _SNQI_DEFAULT_WEIGHTS_NAME
    ).is_file()
    assert (
        result.bundle_dir / "payload" / "release_metadata" / "snqi" / _SNQI_DEFAULT_BASELINE_NAME
    ).is_file()
    # Raw episode rows are retained even when optional videos are excluded.
    assert (result.bundle_dir / "payload" / "episodes" / "episodes.jsonl").is_file()
    assert not (result.bundle_dir / "payload" / "videos").exists()

    violations: list[str] = []
    _preflight_check_release_metadata(
        result.bundle_dir / "payload",
        manifest,
        violations=violations,
    )
    assert violations == []


def test_release_bundle_rejects_run_local_reserved_metadata_namespace(tmp_path: Path) -> None:
    """A run-local ``release_metadata/*`` file cannot replace authoritative metadata."""
    run_dir = tmp_path / "benchmarks" / "release_metadata_collision"
    _make_run(run_dir, with_video=False)
    _write(
        run_dir / "release" / "release_manifest.resolved.json",
        json.dumps(
            {
                "metrics": {
                    "snqi_weights_path": "configs/benchmarks/snqi_weights_camera_ready_v3.json",
                    "snqi_baseline_path": "configs/benchmarks/snqi_baseline_camera_ready_v3.json",
                }
            }
        ),
    )
    _write(run_dir / "release" / "release_result.json", "{}\n")
    _write(run_dir / "release_metadata" / "CITATION.cff", "ATTACKER\n")

    with pytest.raises(ValueError, match="Run-local release_metadata paths are reserved"):
        export_publication_bundle(
            run_dir,
            tmp_path / "publication",
            bundle_name="release_metadata_collision_bundle",
            include_videos=False,
        )

    assert not (tmp_path / "publication").exists()


def test_release_bundle_snqi_basis_prefers_pinned_payload_assets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cold verification must not silently fall back to a checkout's SNQI files."""
    payload_dir = tmp_path / "bundle" / "payload"
    weights_path = payload_dir / "release_metadata" / "snqi" / _SNQI_DEFAULT_WEIGHTS_NAME
    baseline_path = payload_dir / "release_metadata" / "snqi" / _SNQI_DEFAULT_BASELINE_NAME
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    weights_path.write_text(
        (Path("configs/benchmarks") / _SNQI_DEFAULT_WEIGHTS_NAME).read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    baseline_path.write_text(
        (Path("configs/benchmarks") / _SNQI_DEFAULT_BASELINE_NAME).read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    missing_checkout = tmp_path / "cold_checkout"
    missing_checkout.mkdir()
    monkeypatch.setattr(
        "robot_sf.benchmark.artifact_publication.get_repository_root", lambda: missing_checkout
    )

    basis = _snqi_load_canonical_basis(payload_dir)
    assert basis["weights_sha256"]
    assert basis["baseline_sha256"]
    assert "rejections" not in basis


def test_release_metadata_path_and_source_helpers_fail_closed(tmp_path: Path) -> None:
    """Metadata helpers accept only repository files and explicit source SHA fields."""
    repo = tmp_path / "repo"
    repo.mkdir()
    tracked = repo / "metadata.json"
    tracked.write_text("{}\n", encoding="utf-8")
    outside = tmp_path / "outside.json"
    outside.write_text("{}\n", encoding="utf-8")

    assert _resolve_repo_file(None, repo_root=repo) is None
    assert _resolve_repo_file("missing.json", repo_root=repo) is None
    assert _resolve_repo_file(str(outside), repo_root=repo) is None
    assert _resolve_repo_file("metadata.json", repo_root=repo) == tracked
    assert _find_release_sha([{"nested": [{"public_source_commit": "A" * 40}]}]) == "a" * 40
    assert _find_release_sha([{"commit": "short"}, ["not-a-mapping"]]) is None


def test_release_bundle_metadata_path_is_strictly_run_local(tmp_path: Path) -> None:
    """Derived erratum metadata may be run-local but cannot escape or use symlinks."""
    run_root = tmp_path / "run"
    metadata = run_root / "release" / "zenodo_metadata.erratum.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_text("{}\n", encoding="utf-8")
    outside = tmp_path / "outside.json"
    outside.write_text("{}\n", encoding="utf-8")
    linked = run_root / "release" / "linked.json"
    linked.symlink_to(outside)

    assert _resolve_run_file("release/zenodo_metadata.erratum.json", run_root=run_root) == metadata
    assert _resolve_run_file("../outside.json", run_root=run_root) is None
    assert _resolve_run_file(str(outside), run_root=run_root) is None
    assert _resolve_run_file("release/linked.json", run_root=run_root) is None


def test_release_rights_statement_uses_safe_defaults() -> None:
    """Sparse metadata still yields explicit, credential-free claim boundaries."""
    statement = _build_rights_provenance_statement(
        resolved_manifest={"release_tag": ""},
        release_result={},
        zenodo_metadata={"metadata": {"creators": [{"name": ""}, "invalid"]}},
    )

    assert "GPL-3.0-only" in statement
    assert "authoritative release creators" in statement
    assert "recorded in release_result.json" in statement
    assert "credentials" not in statement.lower()


def test_release_metadata_resolver_distinguishes_nonrelease_and_malformed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bundle metadata resolution is optional for normal runs and strict for release runs."""
    run_root = tmp_path / "run"
    run_root.mkdir()
    assert _resolve_release_publication_metadata(run_root) is None

    release_dir = run_root / "release"
    release_dir.mkdir()
    (release_dir / "release_manifest.resolved.json").write_text("not-json", encoding="utf-8")
    (release_dir / "release_result.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed"):
        _resolve_release_publication_metadata(run_root)

    (release_dir / "release_manifest.resolved.json").write_text("{}\n", encoding="utf-8")
    empty_repo = tmp_path / "empty-repo"
    empty_repo.mkdir()
    monkeypatch.setattr(
        "robot_sf.benchmark.artifact_publication.get_repository_root", lambda: empty_repo
    )
    with pytest.raises(ValueError, match="missing required inputs"):
        _resolve_release_publication_metadata(run_root)


def test_release_metadata_preflight_reports_schema_roles_and_policy(tmp_path: Path) -> None:
    """Cold verification reports every malformed release-metadata contract surface."""
    payload_dir = tmp_path / "bundle" / "payload"
    payload_dir.mkdir(parents=True)
    violations: list[str] = []
    _preflight_check_release_metadata(payload_dir, {}, violations=violations)
    assert violations == []

    _preflight_check_release_metadata(
        payload_dir, {"release_metadata": "invalid"}, violations=violations
    )
    assert "must be an object" in violations[-1]

    violations.clear()
    _preflight_check_release_metadata(
        payload_dir,
        {
            "release_metadata": {
                "schema_version": "wrong",
                "required": True,
                "files": {},
                "raw_artifact_policy": {},
                "cold_verification": {},
            }
        },
        violations=violations,
    )
    assert any("unsupported schema_version" in item for item in violations)
    assert any("missing required role" in item for item in violations)
    assert any("must retain campaign output" in item for item in violations)
    assert any("credential policy is invalid" in item for item in violations)

    violations.clear()
    _preflight_check_release_metadata(
        payload_dir,
        {"release_metadata": {"files": {"citation": {"path": "outside"}}}},
        violations=violations,
    )
    assert any("invalid payload path" in item for item in violations)


def test_release_metadata_preflight_rejects_generic_reserved_namespace_entry(
    tmp_path: Path,
) -> None:
    """Preflight must not admit a run-selected file as authoritative metadata."""
    payload_dir = tmp_path / "bundle" / "payload"
    citation = payload_dir / "release_metadata" / "CITATION.cff"
    citation.parent.mkdir(parents=True)
    citation.write_text("ATTACKER\n", encoding="utf-8")
    digest = artifact_publication_module._sha256_file(citation)
    manifest = {
        "files": [
            {
                "path": "release_metadata/CITATION.cff",
                "size_bytes": citation.stat().st_size,
                "sha256": digest,
                "kind": "misc",
            }
        ],
        "release_metadata": {
            "schema_version": RELEASE_PUBLICATION_METADATA_SCHEMA_VERSION,
            "files": {
                "citation": {
                    "path": "payload/release_metadata/CITATION.cff",
                    "sha256": digest,
                }
            },
        },
    }

    violations: list[str] = []
    _preflight_check_release_metadata(payload_dir, manifest, violations=violations)

    assert any("not marked as authoritative provenance" in item for item in violations)
