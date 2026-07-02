"""Tests for the fail-closed certified-failure-archive readiness checker.

These cover the issue #3275 up-front input gate in isolation. The module imports
no simulation/torch surfaces, so these tests run standalone. Synthetic fixtures
exercise missing/malformed archives and the overlap-metadata / null-test
prerequisite checks the proposal-vs-random runner depends on.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from robot_sf.adversarial.disjoint_evaluation import (
    ARCHIVE_SCHEMA_VERSION,
    ArchiveReadinessReport,
    assess_archive_file_readiness,
    assess_archive_readiness,
)

_CLI_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "tools"
    / "check_adversarial_archive_readiness.py"
)


def _load_cli():
    """Import the standalone CLI checker module from its script path."""
    spec = importlib.util.spec_from_file_location("check_archive_readiness_cli", _CLI_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _entry(family: str, archive_id: str, seed: int) -> dict:
    """Build a minimal, readiness-complete archive entry."""
    return {
        "archive_id": archive_id,
        "cluster_key": family,
        "candidate": {"scenario_seed": seed},
        "failure_attribution": {"primary_failure": "collision"},
    }


def _archive(entries: list[dict]) -> dict:
    """Wrap entries in a schema-tagged archive payload."""
    return {
        "schema_version": ARCHIVE_SCHEMA_VERSION,
        "entries": entries,
        "null_test_manifest": {
            "required_tests": [
                "shuffled_outcome_label_permutation",
                "ranking_permutation",
            ],
            "n_permutations": 1000,
        },
    }


def _ready_archive() -> dict:
    """A two-family archive that satisfies every readiness prerequisite."""
    return _archive(
        [
            _entry("A", "a0", 1),
            _entry("A", "a1", 2),
            _entry("B", "b0", 3),
            _entry("B", "b1", 4),
        ]
    )


def test_ready_archive_passes_all_prerequisites() -> None:
    """A well-formed two-family archive is ready with no blocking reasons."""
    report = assess_archive_readiness(_ready_archive())
    assert isinstance(report, ArchiveReadinessReport)
    assert report.ready is True
    assert report.status == "ready"
    assert report.schema_ok is True
    assert report.entry_count == 4
    assert report.distinct_family_count == 2
    assert report.disjoint_split_possible is True
    assert report.overlap_metadata_ready is True
    assert report.null_test_prerequisites_ready is True
    assert report.blocking_reasons == []
    # The report is JSON-serializable for durable provenance.
    assert json.loads(json.dumps(report.to_dict()))["ready"] is True


def test_missing_null_test_manifest_fails_closed() -> None:
    """Archive readiness requires explicit null-test prerequisites."""
    payload = _ready_archive()
    del payload["null_test_manifest"]

    report = assess_archive_readiness(payload)

    assert report.ready is False
    assert report.null_test_prerequisites_ready is False
    assert "null_test_manifest_missing" in report.blocking_reasons


def test_malformed_null_test_required_tests_fail_closed() -> None:
    """Both expected null tests must be declared before rerun readiness."""
    payload = _ready_archive()
    payload["null_test_manifest"]["required_tests"] = ["ranking_permutation"]

    report = assess_archive_readiness(payload)

    assert report.ready is False
    assert report.null_test_prerequisites_ready is False
    assert (
        "null_test_required_tests_missing:shuffled_outcome_label_permutation"
        in report.blocking_reasons
    )


@pytest.mark.parametrize("n_permutations", [0, True])
def test_invalid_null_test_permutation_count_fails_closed(n_permutations: object) -> None:
    """Null-test policy must declare a positive permutation count."""
    payload = _ready_archive()
    payload["null_test_manifest"]["n_permutations"] = n_permutations

    report = assess_archive_readiness(payload)

    assert report.ready is False
    assert report.null_test_prerequisites_ready is False
    assert "null_test_n_permutations_invalid" in report.blocking_reasons


def test_matching_summary_metadata_stays_ready() -> None:
    """Optional archive summary counts may be present when they match entries."""
    payload = _ready_archive()
    payload["clusters"] = [{"cluster_id": "cluster_a"}, {"cluster_id": "cluster_b"}]
    payload["summary"] = {
        "archived_failure_count": 4,
        "cluster_count": 2,
    }

    report = assess_archive_readiness(payload)

    assert report.ready is True
    assert report.blocking_reasons == []


def test_summary_archived_failure_count_mismatch_fails_closed() -> None:
    """Stale entry counts block readiness before provenance is trusted."""
    payload = _ready_archive()
    payload["summary"] = {"archived_failure_count": 3}

    report = assess_archive_readiness(payload)

    assert report.ready is False
    assert "summary_archived_failure_count_mismatch:declared=3:actual=4" in report.blocking_reasons


def test_summary_cluster_count_mismatch_fails_closed() -> None:
    """Stale cluster counts block certified-archive readiness."""
    payload = _ready_archive()
    payload["clusters"] = [{"cluster_id": "cluster_a"}]
    payload["summary"] = {
        "archived_failure_count": 4,
        "cluster_count": 2,
    }

    report = assess_archive_readiness(payload)

    assert report.ready is False
    assert "summary_cluster_count_mismatch:declared=2:actual=1" in report.blocking_reasons


def test_summary_cluster_count_without_cluster_rows_fails_closed() -> None:
    """A declared cluster count without cluster rows is not trusted."""
    payload = _ready_archive()
    payload["summary"] = {"cluster_count": 2}

    report = assess_archive_readiness(payload)

    assert report.ready is False
    assert "summary_cluster_count_without_clusters" in report.blocking_reasons


def test_malformed_summary_metadata_fails_closed() -> None:
    """Malformed summary metadata is reported rather than ignored."""
    payload = _ready_archive()
    payload["summary"] = {"archived_failure_count": True, "cluster_count": "2"}

    report = assess_archive_readiness(payload)

    assert report.ready is False
    assert "summary_archived_failure_count_not_int" in report.blocking_reasons
    assert "summary_cluster_count_not_int" in report.blocking_reasons


def test_non_object_payload_fails_closed() -> None:
    """A non-dict payload is not ready and never raises."""
    report = assess_archive_readiness([1, 2, 3])
    assert report.ready is False
    assert report.status == "not_ready"
    assert "archive_payload_not_object" in report.blocking_reasons


@pytest.mark.parametrize(
    "payload",
    [
        {"schema_version": ARCHIVE_SCHEMA_VERSION},
        {"schema_version": ARCHIVE_SCHEMA_VERSION, "entries": []},
        {"schema_version": ARCHIVE_SCHEMA_VERSION, "entries": "not-a-list"},
    ],
)
def test_missing_or_empty_entries_fail_closed(payload: dict) -> None:
    """Missing, empty, or non-list entries fail closed."""
    report = assess_archive_readiness(payload)
    assert report.ready is False
    assert "archive_has_no_entries" in report.blocking_reasons


def test_unexpected_schema_version_blocks() -> None:
    """A mismatched schema tag blocks even when entries are otherwise fine."""
    payload = _ready_archive()
    payload["schema_version"] = "some_other_schema.v9"
    report = assess_archive_readiness(payload)
    assert report.schema_ok is False
    assert report.ready is False
    assert any(r.startswith("unexpected_schema_version") for r in report.blocking_reasons)


def test_single_family_cannot_form_disjoint_split() -> None:
    """One scenario family cannot be split, so overlap metadata is not ready."""
    report = assess_archive_readiness(_archive([_entry("A", "a0", 1), _entry("A", "a1", 2)]))
    assert report.ready is False
    assert report.distinct_family_count == 1
    assert report.disjoint_split_possible is False
    assert report.overlap_metadata_ready is False
    assert report.null_test_prerequisites_ready is False
    assert any(r.startswith("insufficient_scenario_families") for r in report.blocking_reasons)
    assert "no_disjoint_split_possible" in report.blocking_reasons


def test_missing_archive_id_breaks_overlap_metadata() -> None:
    """Missing archive ids break the archive-id overlap check prerequisite."""
    entries = [_entry("A", "a0", 1), _entry("B", "b0", 2)]
    del entries[1]["archive_id"]
    report = assess_archive_readiness(_archive(entries))
    assert report.entries_missing_archive_id == 1
    assert report.overlap_metadata_ready is False
    assert report.ready is False
    assert "entries_missing_archive_id:1" in report.blocking_reasons


def test_missing_scenario_seed_breaks_overlap_metadata() -> None:
    """Missing candidate.scenario_seed breaks the seed-overlap prerequisite."""
    entries = [_entry("A", "a0", 1), _entry("B", "b0", 2)]
    entries[0]["candidate"] = {}  # no scenario_seed
    report = assess_archive_readiness(_archive(entries))
    assert report.entries_missing_scenario_seed == 1
    assert report.overlap_metadata_ready is False
    assert report.ready is False
    assert "entries_missing_scenario_seed:1" in report.blocking_reasons


def test_seed_overlap_becomes_readiness_blocker() -> None:
    """Shared scenario seeds across families blocks overlap metadata readiness."""
    archive_entries = [
        _entry("family_a", "a0", 7),
        _entry("family_a", "a1", 8),
        _entry("family_b", "b0", 7),
        _entry("family_b", "b1", 9),
    ]
    report = assess_archive_readiness(_archive(archive_entries), split_seed=0, eval_fraction=0.5)
    assert report.seed_overlap_count == 1
    assert report.overlap_metadata_ready is False
    assert report.ready is False
    assert "seed_overlap:1" in report.blocking_reasons


def test_archive_id_overlap_becomes_readiness_blocker() -> None:
    """Shared archive IDs across families blocks overlap metadata readiness."""
    archive_entries = [
        _entry("family_a", "same_id", 1),
        _entry("family_a", "a1", 2),
        _entry("family_b", "same_id", 3),
        _entry("family_b", "b1", 4),
    ]
    report = assess_archive_readiness(_archive(archive_entries), split_seed=0, eval_fraction=0.5)
    assert report.archive_id_overlap_count == 1
    assert report.overlap_metadata_ready is False
    assert report.ready is False
    assert "archive_id_overlap:1" in report.blocking_reasons


def test_missing_failure_attribution_breaks_null_test_prereq() -> None:
    """Missing failure_attribution blocks the null-test prerequisite."""
    entries = [_entry("A", "a0", 1), _entry("B", "b0", 2)]
    del entries[1]["failure_attribution"]
    report = assess_archive_readiness(_archive(entries))
    assert report.entries_missing_failure_attribution == 1
    assert report.null_test_prerequisites_ready is False
    assert report.ready is False
    assert "entries_missing_failure_attribution:1" in report.blocking_reasons


def test_empty_failure_attribution_breaks_null_test_prereq() -> None:
    """Empty failure_attribution is not enough to support null tests."""
    entries = [_entry("A", "a0", 1), _entry("B", "b0", 2)]
    entries[1]["failure_attribution"] = {}

    report = assess_archive_readiness(_archive(entries))

    assert report.entries_missing_failure_attribution == 1
    assert report.null_test_prerequisites_ready is False
    assert report.ready is False
    assert "entries_missing_failure_attribution:1" in report.blocking_reasons


def test_unknown_family_entries_are_counted_and_block() -> None:
    """Entries with no derivable family fall into the unknown-family bucket."""
    # Second entry has no cluster_key / failure key / manifest -> unknown_family.
    entries = [_entry("A", "a0", 1), {"archive_id": "x", "candidate": {"scenario_seed": 2}}]
    report = assess_archive_readiness(_archive(entries))
    assert report.entries_unknown_family == 1
    assert report.ready is False
    assert "entries_unknown_family:1" in report.blocking_reasons


def test_non_object_entries_are_flagged() -> None:
    """Non-dict entries are counted and block readiness."""
    payload = _ready_archive()
    payload["entries"].append("not-an-entry")
    report = assess_archive_readiness(payload)
    assert report.ready is False
    assert any(r.startswith("non_object_entries") for r in report.blocking_reasons)


# --- File-loading fail-closed behavior ----------------------------------------


def test_file_readiness_none_path_fails_closed() -> None:
    """A ``None`` path is not ready (no synthetic fallback)."""
    report = assess_archive_file_readiness(None)
    assert report.ready is False
    assert "no_archive_path_provided" in report.blocking_reasons


def test_file_readiness_missing_path_fails_closed(tmp_path) -> None:
    """A missing file path fails closed rather than fabricating data."""
    report = assess_archive_file_readiness(tmp_path / "absent.json")
    assert report.ready is False
    assert any(r.startswith("archive_path_missing") for r in report.blocking_reasons)


def test_file_readiness_empty_file_fails_closed(tmp_path) -> None:
    """An empty file fails closed."""
    path = tmp_path / "empty.json"
    path.write_text("", encoding="utf-8")
    report = assess_archive_file_readiness(path)
    assert report.ready is False
    assert any(r.startswith("archive_file_empty") for r in report.blocking_reasons)


def test_file_readiness_malformed_json_fails_closed(tmp_path) -> None:
    """Malformed JSON fails closed with an unreadable reason."""
    path = tmp_path / "bad.json"
    path.write_text("{not valid json", encoding="utf-8")
    report = assess_archive_file_readiness(path)
    assert report.ready is False
    assert any(r.startswith("archive_unreadable") for r in report.blocking_reasons)


def test_file_readiness_round_trip_ready(tmp_path) -> None:
    """A real ready archive written to disk loads and assesses as ready."""
    path = tmp_path / "archive.json"
    path.write_text(json.dumps(_ready_archive()), encoding="utf-8")
    report = assess_archive_file_readiness(path)
    assert report.ready is True
    assert report.status == "ready"


# --- CLI exit-code contract ----------------------------------------------------


def test_cli_exits_zero_when_ready(tmp_path, capsys) -> None:
    """The CLI exits 0 and prints a ready report for a valid archive."""
    cli = _load_cli()
    path = tmp_path / "archive.json"
    path.write_text(json.dumps(_ready_archive()), encoding="utf-8")
    out_path = tmp_path / "report.json"
    exit_code = cli.main(["--archive", str(path), "--output", str(out_path)])
    assert exit_code == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["ready"] is True
    # The optional output file mirrors the printed report.
    assert json.loads(out_path.read_text(encoding="utf-8"))["ready"] is True


def test_cli_exits_nonzero_when_not_ready(tmp_path, capsys) -> None:
    """The CLI fails closed (exit 1) for a missing archive input."""
    cli = _load_cli()
    exit_code = cli.main(["--archive", str(tmp_path / "absent.json")])
    assert exit_code == 1
    printed = json.loads(capsys.readouterr().out)
    assert printed["ready"] is False
