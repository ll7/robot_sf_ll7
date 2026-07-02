"""Tests for the SDD curation readiness preflight (issue #1126).

These exercise the curation-step gate on importer fixtures only. They assert the fail-closed
contract: fixture/proxy annotations are never marked benchmark-promotable, and missing SDD blocks
benchmark evidence regardless of whether the importer could parse a candidate file.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from scripts.tools import manage_external_data, sdd_curation_preflight

if TYPE_CHECKING:
    from pathlib import Path


def _write_sdd_fixture(path: Path) -> None:
    """Write a tiny valid SDD-format annotation fixture (two qualifying Pedestrian tracks)."""
    path.write_text(
        "\n".join(
            [
                "1 0 0 10 10 0 0 0 0 Pedestrian",
                "1 5 0 15 10 5 0 0 0 Pedestrian",
                "1 10 0 20 10 10 0 0 0 Pedestrian",
                "1 15 0 25 10 15 0 0 0 Pedestrian",
                "2 100 100 110 110 0 0 0 0 Pedestrian",
                "2 100 105 110 115 5 0 0 0 Pedestrian",
                "2 100 110 110 120 10 0 0 0 Pedestrian",
                "2 100 115 110 125 15 0 0 0 Pedestrian",
                "3 0 0 10 10 0 1 0 0 Pedestrian",  # lost -> filtered
                "4 0 0 10 10 0 0 0 0 Biker",  # wrong label -> filtered
            ]
        ),
        encoding="utf-8",
    )


def _write_quoted_sdd_fixture(path: Path) -> None:
    """Write tiny valid SDD-format annotation fixture with quoted labels."""
    path.write_text(
        "\n".join(
            [
                '1 0 0 10 10 0 0 0 0 "Pedestrian"',
                '1 5 0 15 10 5 0 0 0 "Pedestrian"',
                '1 10 0 20 10 10 0 0 0 "Pedestrian"',
                '1 15 0 25 10 15 0 0 0 "Pedestrian"',
                '2 100 100 110 110 0 0 0 0 "Pedestrian"',
                '2 100 105 110 115 5 0 0 0 "Pedestrian"',
                '2 100 110 110 120 10 0 0 0 "Pedestrian"',
                '2 100 115 110 125 15 0 0 0 "Pedestrian"',
                '3 0 0 10 10 0 1 0 0 "Pedestrian"',  # lost -> filtered
                '4 0 0 10 10 0 0 0 0 "Biker"',  # wrong label -> filtered
            ]
        ),
        encoding="utf-8",
    )


def test_probe_accepts_user_facing_label_for_quoted_sdd_rows(tmp_path: Path) -> None:
    """Probe should accept unquoted labels for quoted SDD annotation rows."""
    annotations = tmp_path / "annotations.txt"
    _write_quoted_sdd_fixture(annotations)

    probe = sdd_curation_preflight.probe_annotation_file(
        annotations,
        label="Pedestrian",
        min_track_points=4,
        max_pedestrians=4,
    )

    assert probe["label"] == "Pedestrian"
    assert probe["usable_label_points"] == 8
    assert probe["usable_track_count"] == 2
    assert probe["selection_satisfiable"] is True
    assert probe["blockers"] == []


def test_probe_normalizes_legacy_quoted_label_argument(tmp_path: Path) -> None:
    """Probe report should display the canonical unquoted label."""
    annotations = tmp_path / "annotations.txt"
    _write_quoted_sdd_fixture(annotations)

    probe = sdd_curation_preflight.probe_annotation_file(
        annotations,
        label='"Pedestrian"',
        min_track_points=4,
        max_pedestrians=4,
    )

    assert probe["label"] == "Pedestrian"
    assert probe["usable_label_points"] == 8


def test_probe_counts_usable_tracks_after_filtering(tmp_path: Path) -> None:
    """Probe should count only label-matching, non-lost tracks meeting min_track_points."""
    annotations = tmp_path / "annotations.txt"
    _write_sdd_fixture(annotations)

    probe = sdd_curation_preflight.probe_annotation_file(
        annotations, label="Pedestrian", min_track_points=4, max_pedestrians=4
    )
    assert probe["exists"] is True
    assert probe["usable_track_count"] == 2  # tracks 1 and 2; track 3 lost, track 4 wrong label
    assert probe["usable_label_points"] == 8
    assert probe["selection_satisfiable"] is True
    assert probe["blockers"] == []


def test_probe_missing_file_is_blocker(tmp_path: Path) -> None:
    """A missing annotation file fails closed with a blocker, not an exception."""
    probe = sdd_curation_preflight.probe_annotation_file(
        tmp_path / "nope.txt", label="Pedestrian", min_track_points=8, max_pedestrians=4
    )
    assert probe["exists"] is False
    assert probe["selection_satisfiable"] is False
    assert probe["blockers"]


def test_probe_insufficient_track_length_is_blocker(tmp_path: Path) -> None:
    """A label present but with too-short tracks is not curation-satisfiable."""
    annotations = tmp_path / "annotations.txt"
    _write_sdd_fixture(annotations)

    probe = sdd_curation_preflight.probe_annotation_file(
        annotations, label="Pedestrian", min_track_points=100, max_pedestrians=4
    )
    assert probe["usable_track_count"] == 0
    assert probe["selection_satisfiable"] is False
    assert probe["blockers"]


def test_missing_sdd_blocks_benchmark_promotion_even_with_valid_fixture(tmp_path: Path) -> None:
    """Core fail-closed contract: a parseable fixture must NOT become benchmark evidence."""
    annotations = tmp_path / "annotations.txt"
    _write_sdd_fixture(annotations)

    proxy_gate = {
        "mode": manage_external_data.SDD_MODE_PROXY,
        "dataset_backed": False,
        "availability": {"state": "missing"},
        "reason": "SDD is not staged.",
        "staging_dir": str(tmp_path),
    }
    probe = sdd_curation_preflight.probe_annotation_file(
        annotations, label="Pedestrian", min_track_points=4, max_pedestrians=4
    )
    report = sdd_curation_preflight.classify_curation_readiness(proxy_gate, probe)

    assert report["dataset_backed"] is False
    assert report["benchmark_promotion_allowed"] is False
    assert report["evidence_status"] == sdd_curation_preflight.EVIDENCE_PROXY
    assert report["output_classification"] == sdd_curation_preflight.OUTPUT_PROXY_ONLY
    # The fixture parses, so curation may still run as a schema smoke -- just not be promoted.
    assert report["curation_runnable"] is True
    assert report["blockers"]


def test_missing_sdd_without_probe_is_blocked(tmp_path: Path) -> None:
    """With no candidate annotation and no staged SDD, the gate is blocked."""
    proxy_gate = {
        "mode": manage_external_data.SDD_MODE_PROXY,
        "dataset_backed": False,
        "availability": {"state": "missing"},
        "reason": "SDD is not staged.",
        "staging_dir": str(tmp_path),
    }
    report = sdd_curation_preflight.classify_curation_readiness(proxy_gate, None)
    assert report["benchmark_promotion_allowed"] is False
    assert report["evidence_status"] == sdd_curation_preflight.EVIDENCE_PROXY
    assert report["output_classification"] == sdd_curation_preflight.OUTPUT_BLOCKED


def test_dataset_backed_with_valid_probe_is_benchmark_candidate(tmp_path: Path) -> None:
    """A staged/validated SDD plus a satisfiable probe unlocks benchmark candidacy."""
    annotations = tmp_path / "annotations.txt"
    _write_sdd_fixture(annotations)

    backed_gate = {
        "mode": manage_external_data.SDD_MODE_DATASET_BACKED,
        "dataset_backed": True,
        "availability": {"state": "dataset_backed"},
        "reason": "SDD is staged and validated.",
        "staging_dir": str(tmp_path),
    }
    probe = sdd_curation_preflight.probe_annotation_file(
        annotations, label="Pedestrian", min_track_points=4, max_pedestrians=4
    )
    report = sdd_curation_preflight.classify_curation_readiness(backed_gate, probe)

    assert report["dataset_backed"] is True
    assert report["benchmark_promotion_allowed"] is True
    assert report["evidence_status"] == sdd_curation_preflight.EVIDENCE_BENCHMARK_CANDIDATE
    assert (
        report["output_classification"] == sdd_curation_preflight.OUTPUT_BENCHMARK_READY_CANDIDATE
    )
    assert report["blockers"] == []


def test_dataset_backed_with_bad_probe_is_not_promotable(tmp_path: Path) -> None:
    """Even with staged SDD, an unsatisfiable probe must not be benchmark-promotable."""
    annotations = tmp_path / "annotations.txt"
    _write_sdd_fixture(annotations)

    backed_gate = {
        "mode": manage_external_data.SDD_MODE_DATASET_BACKED,
        "dataset_backed": True,
        "availability": {"state": "dataset_backed"},
        "reason": "SDD is staged and validated.",
        "staging_dir": str(tmp_path),
    }
    probe = sdd_curation_preflight.probe_annotation_file(
        annotations, label="Pedestrian", min_track_points=100, max_pedestrians=4
    )
    report = sdd_curation_preflight.classify_curation_readiness(backed_gate, probe)

    assert report["dataset_backed"] is True
    assert report["benchmark_promotion_allowed"] is False
    assert report["evidence_status"] == sdd_curation_preflight.EVIDENCE_BLOCKED
    assert report["output_classification"] == sdd_curation_preflight.OUTPUT_BLOCKED


def test_cli_require_benchmark_ready_fails_closed_when_unstaged(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """The CLI exits non-zero under --require-benchmark-ready when SDD is not staged.

    Forces the unstaged staging gate so the assertion is deterministic regardless of the local
    machine's SDD staging state, while still exercising the real CLI parse/probe/exit-code path.
    """
    annotations = tmp_path / "annotations.txt"
    _write_sdd_fixture(annotations)
    monkeypatch.setattr(
        manage_external_data,
        "resolve_sdd_scenario_prior_mode",
        lambda *, manifest_path=None: {
            "mode": manage_external_data.SDD_MODE_PROXY,
            "dataset_backed": False,
            "availability": {"state": "missing"},
            "reason": "SDD is not staged.",
            "staging_dir": str(tmp_path),
        },
    )
    exit_code = sdd_curation_preflight.main(
        [
            "--annotation",
            str(annotations),
            "--min-track-points",
            "4",
            "--require-benchmark-ready",
            "--json",
        ]
    )
    assert exit_code == 3
    out = capsys.readouterr().out
    assert '"benchmark_promotion_allowed": false' in out
