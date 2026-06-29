"""Tests for issue #3275 failure-archive rerun readiness/leakage checks."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from robot_sf.benchmark.failure_archive_rerun_readiness import (
    BLOCKED,
    DIAGNOSTIC_ONLY,
    READY,
    classify_failure_archive_rerun_readiness,
)
from scripts.validation.check_failure_archive_rerun_readiness import main as readiness_cli_main

if TYPE_CHECKING:
    from pathlib import Path


def _entry(
    archive_id: str,
    *,
    family: str,
    seed: int,
    certified: bool = True,
) -> dict:
    """Build a minimal failure-archive entry fixture."""

    entry = {
        "archive_id": archive_id,
        "cluster_key": {
            "policy": "goal",
            "scenario_template": family,
            "primary_failure": "collision",
            "termination_reason": "collision",
        },
        "candidate": {"scenario_seed": seed},
        "failure_attribution": {
            "primary_failure": "collision",
            "details": {"termination_reason": "collision"},
        },
    }
    if certified:
        entry["certification_metadata"] = {
            "status": "passed",
            "source": "unit-test scenario_cert.v1 fixture",
        }
    return entry


def _archive(path: Path, entries: list[dict]) -> Path:
    """Write an adversarial failure archive fixture."""

    payload = {"schema_version": "adversarial_failure_archive.v1", "entries": entries}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_disjoint_certified_archives_are_ready(tmp_path: Path) -> None:
    """Disjoint archive IDs and certified rerun rows pass the metadata gate."""

    source = _archive(
        tmp_path / "source.json",
        [
            _entry("source_0000", family="family_a", seed=1),
            _entry("source_0001", family="family_a", seed=2),
        ],
    )
    rerun = _archive(
        tmp_path / "rerun.json",
        [
            _entry("rerun_0000", family="family_b", seed=101),
            _entry("rerun_0001", family="family_b", seed=102),
        ],
    )

    readiness = classify_failure_archive_rerun_readiness(source, rerun)

    assert readiness.status == READY
    assert readiness.ready is True
    assert readiness.archive_id_overlap == []
    assert readiness.missing_certification_archive_ids == []
    assert readiness.to_payload()["claim_boundary"].startswith("readiness/leakage check only")


def test_overlapping_archive_ids_block_leakage(tmp_path: Path) -> None:
    """Archive-ID overlap between source and rerun archives blocks readiness."""

    source = _archive(
        tmp_path / "source.json",
        [_entry("shared_failure", family="family_a", seed=1)],
    )
    rerun = _archive(
        tmp_path / "rerun.json",
        [_entry("shared_failure", family="family_b", seed=101)],
    )

    readiness = classify_failure_archive_rerun_readiness(source, rerun)

    assert readiness.status == BLOCKED
    assert readiness.archive_id_overlap == ["shared_failure"]
    assert "archive_id_overlap:1" in readiness.blockers


def test_missing_certification_metadata_blocks_rerun_archive(tmp_path: Path) -> None:
    """Every rerun archive entry must carry certification metadata."""

    source = _archive(
        tmp_path / "source.json",
        [_entry("source_0000", family="family_a", seed=1)],
    )
    rerun = _archive(
        tmp_path / "rerun.json",
        [_entry("rerun_0000", family="family_b", seed=101, certified=False)],
    )

    readiness = classify_failure_archive_rerun_readiness(source, rerun)

    assert readiness.status == BLOCKED
    assert readiness.missing_certification_archive_ids == ["rerun_0000"]
    assert "missing_certification_metadata:1" in readiness.blockers


def test_failed_or_falsy_certification_status_is_invalid(tmp_path: Path) -> None:
    """Explicitly failed or falsy certification status must fail closed, not pass."""

    source = _archive(
        tmp_path / "source.json",
        [_entry("source_0000", family="family_a", seed=1)],
    )
    cases = [
        {"status": "failed", "source": "unit-test"},
        {"status": False},
        {"status": None},
        {"status": ""},
        "",
        {},
    ]
    for index, certification in enumerate(cases):
        entry = _entry(f"rerun_{index:04d}", family="family_b", seed=200 + index)
        entry["certification_metadata"] = certification
        rerun = _archive(tmp_path / f"rerun_{index}.json", [entry])

        readiness = classify_failure_archive_rerun_readiness(source, rerun)

        assert readiness.status == BLOCKED, certification
        assert readiness.invalid_certification_archive_ids == [f"rerun_{index:04d}"], certification
        assert readiness.missing_certification_archive_ids == [], certification
        assert "invalid_certification_status:1" in readiness.blockers, certification


def test_diagnostic_only_output_caps_otherwise_ready_inputs(tmp_path: Path) -> None:
    """Diagnostic-only rerun outputs cannot be promoted to benchmark evidence."""

    source = _archive(
        tmp_path / "source.json",
        [_entry("source_0000", family="family_a", seed=1)],
    )
    rerun = _archive(
        tmp_path / "rerun.json",
        [_entry("rerun_0000", family="family_b", seed=101)],
    )
    rerun_output = tmp_path / "rerun_output.json"
    rerun_output.write_text(
        json.dumps(
            {
                "schema_version": "proposal_model_rerun.v1",
                "result_classification": "diagnostic_only",
                "benchmark_evidence": False,
            }
        ),
        encoding="utf-8",
    )

    readiness = classify_failure_archive_rerun_readiness(
        source,
        rerun,
        rerun_output=rerun_output,
    )

    assert readiness.status == DIAGNOSTIC_ONLY
    assert readiness.ready is False
    assert readiness.blockers == []
    assert readiness.diagnostic_only_outputs == ["result_classification:diagnostic_only"]


def test_cli_exit_codes_and_writes_report(tmp_path: Path, capsys) -> None:
    """CLI returns 0 for ready inputs and writes the JSON payload."""

    source = _archive(
        tmp_path / "source.json",
        [_entry("source_0000", family="family_a", seed=1)],
    )
    rerun = _archive(
        tmp_path / "rerun.json",
        [_entry("rerun_0000", family="family_b", seed=101)],
    )
    output = tmp_path / "readiness.json"

    exit_code = readiness_cli_main(
        [
            "--source-archive",
            str(source),
            "--rerun-archive",
            str(rerun),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == READY
    assert json.loads(capsys.readouterr().out)["status"] == READY
