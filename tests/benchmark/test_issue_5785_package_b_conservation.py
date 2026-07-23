"""Diagnostic-summary checks for the Issue #5785 Package B 27-cell result.

The committed report supports an internally consistent 27-cell / 42-count diagnostic summary.
It does not contain the raw candidate/replay tree or execution logs, so these tests deliberately
avoid claiming raw-artifact conservation or independent replay verification.
"""

from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path
from typing import TYPE_CHECKING

from robot_sf.benchmark.adversarial_package_b_report import (
    retrieve_package_b_raw_artifacts,
    validate_package_b_report,
    verify_package_b_candidate_replay_inventory,
)
from scripts.tools.verify_package_b_raw_artifacts import main as verify_raw_artifacts_main

if TYPE_CHECKING:
    from collections.abc import Sequence

BUNDLE = Path("docs/context/evidence/issue_5785_package_b_27cell_replication_2026-07-15")
RECORDED_MANIFEST_SHA256 = "9f174f067d23efd374c019702168213a27085dfffa1b0b5bc10adafaa9614e04"

EXPECTED_TOTALS = {"random": 24, "optuna": 18, "coordinate": 0}
EXPECTED_CELLS = 27
EXPECTED_TOTAL_FAILURES = 42
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


def _load_report(bundle: Path) -> dict[str, object]:
    return json.loads((bundle / "report.json").read_text(encoding="utf-8"))


def _totals_by_sampler(rows: Sequence[dict[str, object]]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for row in rows:
        totals[row["sampler"]] = totals.get(row["sampler"], 0) + int(
            row["certified_valid_failure_count"]
        )
    return totals


def _string_values(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [item for child in value.values() for item in _string_values(child)]
    if isinstance(value, list):
        return [item for child in value for item in _string_values(child)]
    return []


def _write_raw_artifact_fixture(tmp_path: Path) -> tuple[Path, Path]:
    """Write a byte-verifiable 4,761-entry archive fixture with both process logs."""
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    archive_root = tmp_path / "source" / "package_b_raw_artifacts"
    raw_tree = archive_root / "worst_case_snqi" / "fixture"
    raw_tree.mkdir(parents=True)
    entries: list[str] = []
    for index in range(4761):
        name = f"worst_case_snqi/fixture/artifact_{index:04d}.json"
        (archive_root / name).write_bytes(b"")
        entries.append(f"{EMPTY_SHA256}  {name}")
    (bundle / "candidate_replay_SHA256SUMS.txt").write_text(
        "\n".join(entries) + "\n", encoding="utf-8"
    )
    logs_dir = archive_root / "logs"
    logs_dir.mkdir()
    stdout_log = logs_dir / "stdout.log"
    stderr_log = logs_dir / "stderr.log"
    stdout_log.write_text("Package B stdout\n", encoding="utf-8")
    stderr_log.write_text("Package B stderr\n", encoding="utf-8")

    archive_path = tmp_path / "package_b_raw_artifacts.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(archive_root, arcname="package_b_raw_artifacts")
    metadata = {
        "schema_version": "package-b-raw-artifact-bundle.v1",
        "archive": {
            "uri": archive_path.as_uri(),
            "sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
            "format": "tar.gz",
        },
        "archive_root": "package_b_raw_artifacts",
        "raw_tree_path": "worst_case_snqi",
        "logs": [
            {
                "stream": "stdout",
                "path": "logs/stdout.log",
                "sha256": hashlib.sha256(stdout_log.read_bytes()).hexdigest(),
            },
            {
                "stream": "stderr",
                "path": "logs/stderr.log",
                "sha256": hashlib.sha256(stderr_log.read_bytes()).hexdigest(),
            },
        ],
    }
    (bundle / "raw_artifact_bundle.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return bundle, raw_tree


def test_bundle_files_present() -> None:
    """All compact artifacts required for the diagnostic summary are committed."""
    required = (
        "report.json",
        "confirmation.json",
        "comparison_table.md",
        "replication_summary.json",
        "SHA256SUMS",
        "candidate_replay_SHA256SUMS.txt",
        "provenance.md",
        "README.md",
    )
    for name in required:
        assert (BUNDLE / name).is_file(), f"missing diagnostic-summary artifact: {name}"


def test_manifest_identity_matches_recorded_sha256() -> None:
    """The reproduced manifest hash equals the SHA-256 recorded in issue #5785."""
    manifest_path = Path("configs/adversarial/issue_3079_package_b_budget_matched.yaml")
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert digest == RECORDED_MANIFEST_SHA256, (
        "manifest identity drift: reproduced hash differs from the recorded SHA-256 anchor"
    )


def test_report_contract_gate_remains_structurally_valid() -> None:
    """The committed summary still passes the existing report contract gate."""
    gate = validate_package_b_report(BUNDLE / "report.json")
    assert gate.ready is True
    assert gate.status == "ready_for_empirical_review"
    assert gate.matrix["observed_row_count"] == EXPECTED_CELLS


def test_population_and_counts_regenerate_from_committed_report() -> None:
    """The committed report regenerates 27 cells and its recorded 42-count sampler split."""
    report = _load_report(BUNDLE)
    rows = [dict(r) for r in report["rows"]]  # type: ignore[arg-type]
    assert len(rows) == EXPECTED_CELLS
    assert sum(int(r["certified_valid_failure_count"]) for r in rows) == EXPECTED_TOTAL_FAILURES
    assert _totals_by_sampler(rows) == EXPECTED_TOTALS
    # These are self-consistency checks over source-report fields, not raw replay verification.
    for row in rows:
        assert int(row["replayable_valid_failure_count"]) == int(
            row["certified_valid_failure_count"]
        )
        assert int(row["fallback_candidate_count"]) == 0
        assert int(row["degraded_candidate_count"]) == 0


def test_confirmation_sidecar_stays_censored() -> None:
    """Confirmation does not count any certified failure as a confirmed discovery.

    The committed sidecar keeps every one of the 42 recorded certified failures in the
    `not_confirmed` state (independent-seed confirmation, deterministic replay, and stable
    mechanism attribution remain deferred to residual issue #6131). No failure is silently
    promoted to a confirmed discovery.
    """
    sidecar = json.loads((BUNDLE / "confirmation.json").read_text(encoding="utf-8"))
    rows = sidecar["rows"]
    assert len(rows) == EXPECTED_CELLS
    confirmed = sum(int(r.get("confirmed_failure_count", 0)) for r in rows)
    unconfirmed = sum(int(r.get("unconfirmed_certified_failure_count", 0)) for r in rows)
    assert confirmed == 0, "no certified failure may be counted as a confirmed discovery"
    assert unconfirmed == EXPECTED_TOTAL_FAILURES
    assert all(r.get("confirmation_status") == "complete" for r in rows)
    # The sidecar is byte-bound to the committed report.
    report_digest = hashlib.sha256((BUNDLE / "report.json").read_bytes()).hexdigest()
    assert sidecar["source_report_sha256"] == report_digest


def test_durable_summary_sha256sums_match() -> None:
    """The committed SHA256SUMS file verifies the four compact summary artifacts."""
    checksums = (BUNDLE / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    expected_files = {
        "report.json",
        "confirmation.json",
        "comparison_table.md",
        "replication_summary.json",
    }
    verified = 0
    for line in checksums:
        digest, _, name = line.partition("  ")
        name = name.strip()
        if name in expected_files and (BUNDLE / name).is_file():
            actual = hashlib.sha256((BUNDLE / name).read_bytes()).hexdigest()
            assert actual == digest, f"SHA256SUMS mismatch for {name}"
            verified += 1
    assert verified == len(expected_files)


def test_replay_tree_checksum_inventory_is_parseable_and_unique() -> None:
    """The producer-recorded digest inventory has portable, unique logical identifiers."""
    lines = (BUNDLE / "candidate_replay_SHA256SUMS.txt").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4761
    names = set()
    for line in lines:
        digest, _, name = line.partition("  ")
        assert len(digest) == 64, "malformed digest in candidate_replay_SHA256SUMS.txt"
        path = Path(name.strip())
        assert not path.is_absolute()
        assert path.parts[0] == "worst_case_snqi"
        assert "output" not in path.parts
        assert "." not in path.parts
        assert ".." not in path.parts
        names.add(path.as_posix())
    assert len(names) == len(lines)


def test_summary_references_are_portable_and_inventory_backed() -> None:
    """Tracked JSON uses portable identifiers and never claims local output as durable proof."""
    report = _load_report(BUNDLE)
    summary = json.loads((BUNDLE / "replication_summary.json").read_text(encoding="utf-8"))
    inventory = {
        line.partition("  ")[2].strip()
        for line in (BUNDLE / "candidate_replay_SHA256SUMS.txt")
        .read_text(encoding="utf-8")
        .splitlines()
    }

    for value in _string_values(report) + _string_values(summary):
        assert not value.startswith("/"), f"absolute artifact reference: {value}"
        assert "output/" not in value, f"ignored-output artifact reference: {value}"

    assert report["artifact_availability"] == "digest_inventory_only_raw_tree_unavailable"
    for row in report["rows"]:  # type: ignore[union-attr]
        row = dict(row)
        manifest_id = str(row["manifest_path"])
        bundle_id = str(row["best_bundle_path"])
        assert manifest_id in inventory
        assert any(name.startswith(f"{bundle_id}/") for name in inventory)


def test_candidate_replay_inventory_verifier_requires_raw_bytes(tmp_path: Path) -> None:
    """A committed inventory alone cannot be counted as verified raw evidence."""
    result = verify_package_b_candidate_replay_inventory(BUNDLE)
    assert result.is_valid is False
    assert result.total_entries == 4761
    assert result.verified_entries == 0
    assert result.missing_entries == 0
    assert result.mismatched_entries == 0
    assert any("raw_tree_dir is required" in error for error in result.errors)

    # Fail closed on corrupted inventory as well as absent raw bytes.
    bad_manifest = tmp_path / "candidate_replay_SHA256SUMS.txt"
    bad_manifest.write_text("invalid_digest_line\n", encoding="utf-8")
    bad_result = verify_package_b_candidate_replay_inventory(tmp_path)
    assert bad_result.is_valid is False
    assert len(bad_result.errors) > 0


def test_raw_artifact_retrieval_verifies_bytes_logs_and_failures(tmp_path: Path) -> None:
    """The retriever verifies archive, raw bytes, both logs, and all failure modes."""
    bundle, _raw_tree = _write_raw_artifact_fixture(tmp_path)
    retrieval = retrieve_package_b_raw_artifacts(bundle, tmp_path / "retrieved")
    assert retrieval.is_valid is True
    assert retrieval.raw_tree_dir is not None
    assert retrieval.verified_log_entries == 2

    verified = verify_package_b_candidate_replay_inventory(bundle, retrieval.raw_tree_dir)
    assert verified.is_valid is True
    assert verified.verified_entries == 4761

    target = Path(retrieval.raw_tree_dir) / "worst_case_snqi" / "fixture" / "artifact_0000.json"
    target.write_bytes(b"corrupted")
    corrupt = verify_package_b_candidate_replay_inventory(bundle, retrieval.raw_tree_dir)
    assert corrupt.is_valid is False
    assert corrupt.mismatched_entries == 1

    target.unlink()
    missing = verify_package_b_candidate_replay_inventory(bundle, retrieval.raw_tree_dir)
    assert missing.is_valid is False
    assert missing.missing_entries == 1

    metadata_path = bundle / "raw_artifact_bundle.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["archive"]["uri"] = (tmp_path / "missing.tar.gz").as_uri()
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    unavailable = retrieve_package_b_raw_artifacts(bundle, tmp_path / "unavailable")
    assert unavailable.is_valid is False
    assert any("could not retrieve raw-artifact archive" in error for error in unavailable.errors)


def test_raw_artifact_cli_retrieves_or_fails_closed(tmp_path: Path) -> None:
    """The CLI retrieves pinned metadata by default and rejects unavailable metadata."""
    bundle, _raw_tree = _write_raw_artifact_fixture(tmp_path)
    assert (
        verify_raw_artifacts_main(
            ["--bundle", str(bundle), "--retrieve-to", str(tmp_path / "cli-retrieved")]
        )
        == 0
    )
    assert verify_raw_artifacts_main(["--bundle", str(tmp_path / "missing-bundle")]) == 1


def test_candidate_replay_inventory_rejects_traversal_components(tmp_path: Path) -> None:
    """A traversal path is rejected before joining it to a raw-tree root."""
    inventory = (BUNDLE / "candidate_replay_SHA256SUMS.txt").read_text(encoding="utf-8")
    first_line, remainder = inventory.split("\n", maxsplit=1)
    digest = first_line.partition("  ")[0]
    (tmp_path / "candidate_replay_SHA256SUMS.txt").write_text(
        f"{digest}  worst_case_snqi/../../outside\n{remainder}", encoding="utf-8"
    )
    result = verify_package_b_candidate_replay_inventory(tmp_path)
    assert result.is_valid is False
    assert any("must not contain traversal components" in error for error in result.errors)


def test_raw_artifact_retrieval_status_is_explicitly_blocked() -> None:
    """Docs do not represent an absent raw tree/log archive as independently verified."""
    readme_text = (BUNDLE / "README.md").read_text(encoding="utf-8")
    provenance_text = (BUNDLE / "provenance.md").read_text(encoding="utf-8")

    for text in (readme_text, provenance_text):
        assert "Raw Artifact Retrieval Status (Issue #6131)" in text
        assert "7ec582b81cdcb871fb4fcb47700338194e7617d5" in text
        assert "9f174f067d23efd374c019702168213a27085dfffa1b0b5bc10adafaa9614e04" in text
        assert "4761" in text
        assert "verify_package_b_raw_artifacts.py" in text
        assert "run_adversarial_package_b.py" in text
        assert "unavailable" in text
        assert "diagnostic-only" in text or "not paper-facing" in text
