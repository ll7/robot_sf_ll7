"""Regression tests for the public-safe archive projection (issue #5911).

These tests prove the reusable public-evidence export entry point:

- projects absolute private paths to stable ``private-artifact://`` URIs;
- preserves candidate, metric, certification, family, seed, and archive-ID
  values exactly;
- records source and projected SHA-256 plus a path-only transformation record;
- fails closed when a projected archive still contains a private absolute path;
- writes the public bundle through the shared evidence-writer convention so the
  required ``AI-GENERATED NEEDS-REVIEW`` marker is present; and
- produces a bundle that passes the repository docs-evidence integrity check.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from robot_sf.adversarial.disjoint_evaluation import archive_sha256
from robot_sf.adversarial.public_projection import (
    DEFAULT_PUBLIC_SCHEME,
    PUBLIC_PROJECTION_SCHEMA_VERSION,
    PrivatePathLeakError,
    PublicProjectionConfig,
    assert_no_private_paths,
    find_offending_paths,
    project_archive_to_public,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PRIVATE_ROOT = "/home/runner/git/robot_sf_ll7.worktrees/cheap-13518/output"
# Mirrors the module's private anchor set; kept local so the test does not import
# a private symbol while still asserting the scanner covers each anchor class.
_PRIVATE_PATH_ANCHORS = ("/home/", "/root/", "/Users/", "worktrees/")


def _entry(
    *,
    archive_id: str,
    seed: int,
    objective: float,
    paths: dict[str, str],
    family: str = "classic_cross_trap_medium",
) -> dict:
    """Build one archive entry with private absolute paths.

    ``paths`` carries the path-bearing string fields (``bundle``,
    ``scenario_yaml``, ``manifest``, ``record_path``, ``replay_command``) so the
    helper signature stays under the argument-count lint limit while remaining
    readable.
    """
    bundle = paths["bundle"]
    record_path = paths["record_path"]
    return {
        "archive_id": archive_id,
        "source_manifest": paths["manifest"],
        "source_candidate_index": 13,
        "bundle_path": bundle,
        "scenario_yaml_path": paths["scenario_yaml"],
        "trajectory_csv_path": f"{bundle}/trajectory.csv",
        "episode_record_path": record_path,
        "replay_command": paths["replay_command"],
        "objective_value": objective,
        "scenario_family": family,
        "candidate": {
            "goal": {"theta": 0.0, "x": 8.0, "y": 2.0},
            "pedestrian_delay_s": 0.8516135203091186,
            "pedestrian_speed_mps": 1.088802327928866,
            "scenario_seed": seed,
            "spawn_time_s": 1.0,
            "start": {"theta": 0.0, "x": 0.25, "y": 2.0},
        },
        "candidate_certification": {
            "deterministic_replay": {
                "exact_signature_match": True,
                "original_record_path": record_path,
                "original_record_sha256": "85779cde16" * 6 + "a1b2",
            },
            "independent_seed_confirmation": {"confirmed_seeds": 3, "required": 3},
            "status": "passed",
        },
        "certification_status": {"status": "passed", "details": {}},
        "failure_attribution": {
            "primary_failure": "collision",
            "status": "attributed",
            "details": {"termination_reason": "collision"},
        },
        "mechanism_cluster_key": {
            "policy": "goal",
            "primary_failure": "collision",
            "scenario_template": f"{PRIVATE_ROOT}/configs/{family}.yaml",
            "termination_reason": "collision",
        },
    }


def _private_archive() -> dict:
    """Build a synthetic archive payload that embeds private absolute paths."""
    entries = [
        _entry(
            archive_id="failure_0000",
            seed=12796,
            objective=10.0,
            paths={
                "bundle": (
                    f"{PRIVATE_ROOT}/raw/classic_cross_trap_medium/seed_530501/candidate_0013"
                ),
                "scenario_yaml": (
                    f"{PRIVATE_ROOT}/raw/classic_cross_trap_medium/seed_530501/"
                    "candidate_0013/scenario.yaml"
                ),
                "manifest": (
                    f"{PRIVATE_ROOT}/raw/classic_cross_trap_medium/seed_530501/manifest.json"
                ),
                "record_path": (
                    f"{PRIVATE_ROOT}/raw/classic_cross_trap_medium/seed_530501/"
                    "candidate_0013/episode_records.jsonl"
                ),
                "replay_command": (
                    f"uv run robot_sf_bench run --matrix "
                    f"{PRIVATE_ROOT}/raw/classic_cross_trap_medium/seed_530501/"
                    "candidate_0013/scenario.yaml --out "
                    f"{PRIVATE_ROOT}/raw/classic_cross_trap_medium/seed_530501/"
                    "candidate_0013/episode_records_replay.jsonl --algo goal --no-video"
                ),
            },
        ),
    ]
    return {
        "schema_version": "adversarial_failure_archive.v1",
        "created_at": "2026-07-16T22:18:15+00:00",
        "provenance": {
            "campaign_id": "issue5911-fixture-13518",
            "commit": "0" * 40,
            "config_sha256": "1" * 64,
        },
        "config": {
            "source_manifests": [
                f"{PRIVATE_ROOT}/raw/classic_cross_trap_medium/seed_530501/manifest.json"
            ],
        },
        "summary": {
            "archived_failure_count": 1,
            "source_manifest_count": 1,
            "source_candidate_count": 1,
        },
        "entries": entries,
    }


def test_find_offending_paths_locates_every_private_anchor() -> None:
    """The scanner flags strings carrying any private anchor."""
    archive = _private_archive()
    offending = find_offending_paths(archive)
    assert offending, "fixture must contain private paths"
    assert all(any(anchor in value for anchor in _PRIVATE_PATH_ANCHORS) for value in offending)
    # Deterministic, de-duplicated output.
    assert offending == sorted(set(offending))


def test_projection_replaces_private_roots_with_stable_artifact_uris() -> None:
    """Every private absolute path becomes a private-artifact URI."""
    archive = _private_archive()
    result = project_archive_to_public(
        archive,
        config=PublicProjectionConfig(
            private_root=PRIVATE_ROOT,
            scheme=DEFAULT_PUBLIC_SCHEME,
            job_id=13518,
        ),
    )
    assert PRIVATE_ROOT not in json.dumps(result.projected_archive)
    assert "private-artifact://job-13518/" in json.dumps(result.projected_archive)
    # No private anchor survives projection.
    assert assert_no_private_paths(result.projected_archive) == []


def test_projection_preserves_research_values_exactly() -> None:
    """Candidate, metric, certification, family, seed, and archive-ID are unchanged."""
    archive = _private_archive()
    result = project_archive_to_public(
        archive,
        config=PublicProjectionConfig(private_root=PRIVATE_ROOT, job_id=13518),
    )
    source_entry = archive["entries"][0]
    projected_entry = result.projected_archive["entries"][0]

    # Research values are byte-identical.
    assert projected_entry["archive_id"] == source_entry["archive_id"]
    assert projected_entry["objective_value"] == source_entry["objective_value"]
    assert projected_entry["candidate"] == source_entry["candidate"]
    assert projected_entry["scenario_family"] == source_entry["scenario_family"]
    assert projected_entry["failure_attribution"] == source_entry["failure_attribution"]
    assert (
        projected_entry["candidate_certification"]["status"]
        == source_entry["candidate_certification"]["status"]
    )
    assert (
        projected_entry["candidate_certification"]["independent_seed_confirmation"]
        == source_entry["candidate_certification"]["independent_seed_confirmation"]
    )
    # The seed value is preserved exactly (the leak risk for disjointness proofs).
    assert (
        projected_entry["candidate"]["scenario_seed"] == source_entry["candidate"]["scenario_seed"]
    )
    # The deterministic-replay checksum is preserved; only its sibling path changes.
    assert (
        projected_entry["candidate_certification"]["deterministic_replay"]["original_record_sha256"]
        == source_entry["candidate_certification"]["deterministic_replay"]["original_record_sha256"]
    )
    assert (
        projected_entry["candidate_certification"]["deterministic_replay"]["exact_signature_match"]
        is True
    )


def test_projection_records_source_projected_checksums_and_path_only_transform() -> None:
    """The transformation record carries both digests and declares path-only change."""
    archive = _private_archive()
    result = project_archive_to_public(
        archive,
        config=PublicProjectionConfig(private_root=PRIVATE_ROOT, job_id=13518),
    )

    assert result.source_sha256 == archive_sha256(archive)
    assert result.projected_sha256 == archive_sha256(result.projected_archive)
    assert result.source_sha256 != result.projected_sha256

    record = result.projection
    assert record["schema_version"] == PUBLIC_PROJECTION_SCHEMA_VERSION
    assert record["source_archive_sha256"] == result.source_sha256
    assert record["projected_archive_sha256"] == result.projected_sha256
    assert record["candidate_or_metric_values_changed"] is False
    assert record["changed_fields"] == "string path prefixes only"
    assert record["scheme"] == DEFAULT_PUBLIC_SCHEME
    assert record["job_id"] == 13518
    assert record["private_pointer_scheme"] == "private-artifact://job-13518/"
    assert record["replacement_count"] >= 1
    assert record["offending_path_count"] >= 1
    # The private root is hashed, never recorded verbatim.
    assert PRIVATE_ROOT not in json.dumps(record)
    assert len(record["private_root_sha256"]) == 64


def test_projection_counts_repeated_root_occurrences() -> None:
    """The replacement count records each occurrence, including repeats in one string."""
    archive = {"replay_command": f"cp {PRIVATE_ROOT}/a {PRIVATE_ROOT}/b"}
    result = project_archive_to_public(
        archive,
        config=PublicProjectionConfig(private_root=PRIVATE_ROOT, job_id=13518),
    )

    assert result.projection["replacement_count"] == 2
    assert result.projected_archive["replay_command"].count("private-artifact://job-13518/") == 2


def test_projection_does_not_rewrite_a_similar_but_outside_root() -> None:
    """A root-like prefix outside the configured root must fail closed unchanged."""
    archive = {"path": f"{PRIVATE_ROOT}_backup/raw.json"}

    with pytest.raises(PrivatePathLeakError, match="output_backup"):
        project_archive_to_public(
            archive,
            config=PublicProjectionConfig(private_root=PRIVATE_ROOT, job_id=13518),
        )


def test_projection_checksums_match_independently_recomputed_digests() -> None:
    """Recorded digests match freshly recomputed archive_sha256 over both sides."""
    archive = _private_archive()
    result = project_archive_to_public(
        archive,
        config=PublicProjectionConfig(private_root=PRIVATE_ROOT, job_id=13518),
    )
    assert result.source_sha256 == archive_sha256(archive)
    assert result.projected_sha256 == archive_sha256(result.projected_archive)
    # Re-running is idempotent: projecting the already-public archive is a no-op.
    second = project_archive_to_public(
        result.projected_archive,
        config=PublicProjectionConfig(private_root=PRIVATE_ROOT, job_id=13518),
    )
    assert second.projected_sha256 == result.projected_sha256
    assert second.projection["replacement_count"] == 0


def test_auto_detect_private_root_matches_explicit_projection() -> None:
    """Auto-detected root produces a public archive equivalent to the explicit one."""
    archive = _private_archive()
    explicit = project_archive_to_public(
        archive,
        config=PublicProjectionConfig(private_root=PRIVATE_ROOT, job_id=13518),
    )
    auto = project_archive_to_public(
        archive,
        config=PublicProjectionConfig(auto_detect_private_root=True, job_id=13518),
    )
    assert auto.projected_sha256 == explicit.projected_sha256
    assert auto.projection["auto_detected_private_root"] is True
    assert auto.projection["private_root_supplied"] is False


def test_default_config_preserves_path_tail_without_job_id() -> None:
    """The default entry point (config=None, job_id=None) must not lose path data.

    Regression for issue #5911: ``rstrip("/")`` on the ``<scheme>://`` authority
    form stripped the ``//`` separator, so when no job id was supplied (the
    *default* public entry point) the replacement collapsed to
    ``private-artifact:`` and either malformed the URI (path tail survived but
    the ``://`` was lost) or, when an offending value equaled the target root,
    destroyed the whole string and silently dropped the path tail. Both the
    default ``config=None`` path and an explicit no-job config must preserve the
    documented ``<scheme>://`` authority and the surviving path tail.
    """
    archive = {
        "bundle": f"{PRIVATE_ROOT}/raw/seed_1/candidate_0001",
        "scenario": f"{PRIVATE_ROOT}/raw/seed_1/candidate_0001/scenario.yaml",
    }

    # Default entry point: config=None selects auto-detect with job_id=None.
    default = project_archive_to_public(archive)
    projected = default.projected_archive
    projected_json = json.dumps(projected)
    # The ``://`` authority separator survives (no ``private-artifact:`` collapse)
    # and no private path leaks.
    assert "private-artifact://" in projected_json
    assert "private-artifact:" not in projected_json.replace("private-artifact://", "")
    assert PRIVATE_ROOT not in projected_json
    assert assert_no_private_paths(projected) == []
    # The path tail is preserved (no data loss): the surviving component below
    # the auto-detected common root is retained under the artifact URI rather
    # than being destroyed into a bare ``private-artifact:``.
    assert projected["scenario"].endswith("/scenario.yaml")
    assert projected["scenario"].startswith("private-artifact://")

    # Explicit no-job config over the same archive: same authority, tail kept.
    explicit_no_job = project_archive_to_public(
        archive,
        config=PublicProjectionConfig(private_root=PRIVATE_ROOT, job_id=None),
    )
    explicit_json = json.dumps(explicit_no_job.projected_archive)
    assert "private-artifact://" in explicit_json
    assert "raw/seed_1/candidate_0001/scenario.yaml" in explicit_json


def test_projection_value_equal_to_root_is_not_collapsed() -> None:
    """An offending value that equals the target root keeps the ``<scheme>://`` form.

    Regression for issue #5911: a field whose value is exactly the private root
    (so there is no surviving path tail) must project to the bare ``<scheme>://``
    authority rather than collapsing to ``private-artifact:``, which destroyed
    the value and silently lost data.
    """
    archive = {"root_pointer": PRIVATE_ROOT}
    result = project_archive_to_public(
        archive,
        config=PublicProjectionConfig(private_root=PRIVATE_ROOT, job_id=None),
    )
    assert result.projected_archive["root_pointer"] == "private-artifact://"
    assert PRIVATE_ROOT not in json.dumps(result.projected_archive)


def test_projection_fails_closed_when_private_path_remains() -> None:
    """A private path the root cannot reach raises PrivatePathLeakError."""
    archive = _private_archive()
    # Inject a path under a different private root the config does not cover and
    # that the absolute-path fallback cannot fully neutralize (no public tail
    # anchor and a bare home path with no safe tail to keep).
    archive["entries"][0]["unreachable_path"] = "/home/secret_user/credentials.key"
    with pytest.raises(PrivatePathLeakError) as excinfo:
        project_archive_to_public(
            archive,
            config=PublicProjectionConfig(private_root=PRIVATE_ROOT, job_id=13518),
        )
    assert "secret_user" in str(excinfo.value)


def test_assert_no_private_paths_raises_with_actionable_preview() -> None:
    """The fail-closed helper names the offending values."""
    payload = {"x": "/home/leak/path", "y": ["clean", "/Users/foo/bar"]}
    with pytest.raises(PrivatePathLeakError) as excinfo:
        assert_no_private_paths(payload, label="test payload")
    assert "/home/leak/path" in str(excinfo.value)
    assert "/Users/foo/bar" in str(excinfo.value)
    assert excinfo.value.offending_paths == ["/Users/foo/bar", "/home/leak/path"]


def test_projection_of_clean_archive_is_a_noop() -> None:
    """An archive with no private paths projects unchanged with zero replacements."""
    archive = _private_archive()
    # Pre-project once to get a clean archive.
    clean = project_archive_to_public(
        archive,
        config=PublicProjectionConfig(private_root=PRIVATE_ROOT, job_id=13518),
    ).projected_archive
    result = project_archive_to_public(
        clean,
        config=PublicProjectionConfig(private_root=PRIVATE_ROOT, job_id=13518),
    )
    assert result.source_sha256 == result.projected_sha256
    assert result.projection["replacement_count"] == 0
    assert result.offending_paths == []


def test_cli_writes_public_bundle_with_review_marker_and_passes_integrity_check(
    tmp_path: Path,
) -> None:
    """The CLI emits the marker convention and a docs-evidence-integrity-clean bundle."""
    source_path = tmp_path / "source_archive.json"
    source_path.write_text(json.dumps(_private_archive(), sort_keys=True), encoding="utf-8")
    output_dir = tmp_path / "bundle"

    cli = REPO_ROOT / "scripts/dev/export_public_evidence_archive.py"
    result = subprocess.run(
        [
            sys.executable,
            str(cli),
            "--source",
            str(source_path),
            "--output-dir",
            str(output_dir),
            "--private-root",
            PRIVATE_ROOT,
            "--scheme",
            DEFAULT_PUBLIC_SCHEME,
            "--job-id",
            "13518",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "FAIL-CLOSED" not in result.stderr
    assert "values changed:         False" in result.stdout

    archive_out = output_dir / "archive.json"
    record_out = output_dir / "public_projection.json"
    sums_out = output_dir / "SHA256SUMS"
    assert archive_out.is_file()
    assert record_out.is_file()
    assert sums_out.is_file()

    archive_payload = json.loads(archive_out.read_text(encoding="utf-8"))
    record_payload = json.loads(record_out.read_text(encoding="utf-8"))
    sidecar_payload = json.loads(
        (output_dir / "archive.json.review.json").read_text(encoding="utf-8")
    )

    # The projected archive is written verbatim (no inline marker) so its
    # committed bytes match the recorded digest; the AI-GENERATED NEEDS-REVIEW
    # marker is carried by the companion review sidecar (shared convention).
    assert "review_marker" not in archive_payload
    assert record_payload["review_marker"] == "AI-GENERATED NEEDS-REVIEW"
    assert sidecar_payload["schema_version"] == "evidence-review-marker.v1"
    assert sidecar_payload["review_marker"] == "AI-GENERATED NEEDS-REVIEW"
    assert sidecar_payload["preserved_exact_bytes"] is True

    # No private path leaked into the written bundle (all files, including sums).
    for path in output_dir.iterdir():
        text = path.read_text(encoding="utf-8")
        assert PRIVATE_ROOT not in text
        assert "/home/" not in text
        assert "worktrees/" not in text
    assert "private-artifact://job-13518/" in archive_out.read_text(encoding="utf-8")

    # Recorded digests match the canonical digest of the written bytes.
    assert record_payload["projected_archive_sha256"] == archive_sha256(archive_payload)
    # The transformation record is self-consistent.
    assert record_payload["candidate_or_metric_values_changed"] is False
    assert record_payload["changed_fields"] == "string path prefixes only"

    # The bundle passes the repository docs-evidence integrity check when the
    # evidence file is treated as already-registered (we point the checker at
    # the files directly so it does not require catalog registration here).
    integrity = REPO_ROOT / "scripts/dev/check_docs_evidence_integrity.py"
    check = subprocess.run(
        [
            sys.executable,
            str(integrity),
            "--files",
            str(archive_out),
            str(record_out),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )
    assert check.returncode == 0, check.stderr


def test_cli_fails_closed_and_writes_nothing_when_path_remains(tmp_path: Path) -> None:
    """A fail-closed CLI invocation leaves the output directory empty."""
    archive = _private_archive()
    archive["entries"][0]["unreachable_path"] = "/home/secret_user/credentials.key"
    source_path = tmp_path / "source_archive.json"
    source_path.write_text(json.dumps(archive, sort_keys=True), encoding="utf-8")
    output_dir = tmp_path / "bundle"

    cli = REPO_ROOT / "scripts/dev/export_public_evidence_archive.py"
    result = subprocess.run(
        [
            sys.executable,
            str(cli),
            "--source",
            str(source_path),
            "--output-dir",
            str(output_dir),
            "--private-root",
            PRIVATE_ROOT,
            "--job-id",
            "13518",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )
    assert result.returncode == 1, result.stdout
    assert "FAIL-CLOSED" in result.stderr
    # No partial bundle is written on failure.
    assert not output_dir.exists() or not any(output_dir.iterdir())
