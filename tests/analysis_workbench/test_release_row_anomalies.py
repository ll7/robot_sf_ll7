"""Synthetic coverage for the release-row anomaly gate (issue #9734)."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

from robot_sf.analysis_workbench.audit_contracts import record_from_dict, record_to_dict
from robot_sf.analysis_workbench.release_row_anomalies import analyze_release_rows, main

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


CONFIG: dict[str, object] = {
    "short_collision_max_steps": 2,
    "same_step_max_steps": 5,
    "max_contact_speed_m_s": 100.0,
    "min_orbit_curvature": 10.0,
    "max_zero_progress_m": 0.05,
    "max_unannotated_findings": 0,
    "baseline_planner": "blind_goal",
    "pedestrian_free_scenarios": ["pedestrian-free"],
    "min_planners_per_cell": 2,
    "min_paired_cells": 1,
    "require_preflight": False,
}


def _row(  # noqa: PLR0913
    scenario_id: str,
    seed: int,
    algo: str,
    *,
    steps: int = 600,
    success: bool = False,
    collision: bool = False,
    timeout: bool = True,
    invalid_run: bool = False,
    observation_ped_count: int | None = None,
    metrics: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build the compact published-row shape used by the release gate."""

    if observation_ped_count is None:
        observation_ped_count = 0 if scenario_id == "pedestrian-free" else 1
    return {
        "episode_id": f"{scenario_id}:{seed}:{algo}",
        "scenario_id": scenario_id,
        "seed": seed,
        "algo": algo,
        "steps": steps,
        "outcome": {
            "route_complete": success,
            "collision_event": collision,
            "timeout_event": timeout,
        },
        "event_ledger": {"exact_events": {"invalid_run": invalid_run}},
        "integrity": {"effective_view": {"observation_ped_count": observation_ped_count}},
        "metrics": dict(metrics or {}),
    }


def _source(*planners: str) -> dict[str, object]:
    return {"release_id": "synthetic-issue-9734", "planner_ids": list(planners)}


def _findings(report: Mapping[str, Any], detector_id: str) -> list[Mapping[str, Any]]:
    return [finding for finding in report["findings"] if finding["detector_id"] == detector_id]


def test_all_release_row_detector_families_flag_synthetic_anomalies() -> None:
    """Each required issue #9734 detector produces a structured finding."""

    contact = _row(
        "teleport-contact",
        113,
        "planner_a",
        collision=True,
        timeout=False,
    )
    contact["event_ledger"]["collision_events"] = [{"relative_speed_at_contact": 710.0}]
    cases = [
        (
            [
                _row("same-step", 111, "planner_a", steps=5),
                _row("same-step", 111, "planner_b", steps=5),
            ],
            _source("planner_a", "planner_b"),
            None,
        ),
        (
            [_row("short-collision", 112, "planner_a", steps=2, collision=True, timeout=False)],
            _source("planner_a"),
            None,
        ),
        ([contact], _source("planner_a"), None),
        (
            [
                _row(
                    "orbiting",
                    114,
                    "planner_a",
                    metrics={
                        "curvature_mean": 10.5,
                        "socnavbench_path_length": 10.0,
                        "robot_displacement_m": 0.01,
                        "deadlock": True,
                    },
                )
            ],
            _source("planner_a"),
            None,
        ),
        (
            [
                _row("pedestrian-free", 115, "blind_goal", success=True, timeout=False),
                _row("pedestrian-free", 115, "social_force"),
            ],
            _source("blind_goal", "social_force"),
            None,
        ),
        (
            [
                _row("universal-failure", 116, "planner_a"),
                _row("universal-failure", 116, "planner_b"),
            ],
            _source("planner_a", "planner_b"),
            None,
        ),
        (
            [
                _row("invalid-mismatch", 117, "planner_a", invalid_run=True),
                _row("invalid-mismatch", 117, "planner_b", invalid_run=True),
            ],
            _source("planner_a", "planner_b"),
            [{"scenario_id": "invalid-mismatch", "seed": 117, "invalid_run": False}],
        ),
    ]
    reports = [
        analyze_release_rows(rows, config=CONFIG, source=source, preflight=preflight)
        for rows, source, preflight in cases
    ]
    detector_ids = {finding["detector_id"] for report in reports for finding in report["findings"]}

    assert {
        "same_step_all_planners",
        "short_collision",
        "impossible_contact_speed",
        "orbit_zero_progress",
        "pedestrian_free_baseline_regression",
        "universal_failure_unannotated",
        "invalid_run_preflight_mismatch",
    } <= detector_ids

    # Findings expose stable cell identity and annotation state.  The release
    # gate itself is report-level, because one threshold applies to all rows.
    for report in reports:
        assert report["gate"]["blocked"] is True
        for finding in report["findings"]:
            assert finding["scenario_id"]
            assert isinstance(finding["seed"], (int, type(None)))
            assert finding["annotated"] is False


def test_pedestrian_free_finding_is_scenario_level() -> None:
    """Baseline dominance is reported once per paired scenario, with no seed."""

    rows = [
        _row("pedestrian-free", 115, "blind_goal", success=True, timeout=False),
        _row("pedestrian-free", 115, "social_force"),
    ]
    report = analyze_release_rows(
        rows,
        config=CONFIG,
        source=_source("blind_goal", "social_force"),
    )

    finding = _findings(report, "pedestrian_free_baseline_regression")[0]
    assert finding["seed"] is None
    assert finding["planner_id"] == "social_force"


def test_flagged_signal_round_trips_through_audit_contract() -> None:
    """Release-row signals use the canonical BA-03 audit-record envelope."""

    report = analyze_release_rows(
        [_row("signal-roundtrip", 125, "planner_a", steps=2, collision=True, timeout=False)],
        config=CONFIG,
        source=_source("planner_a"),
    )

    signal_payload = report["signals"][0]
    signal = record_from_dict(signal_payload)
    assert record_to_dict(signal) == signal_payload
    assert report["detector_registry_digest"]
    assert report["detector_registry"]["detectors"]


def test_pedestrian_free_detector_excludes_nonfree_observations() -> None:
    """A configured scenario is still excluded when rows observe pedestrians."""

    rows = [
        _row(
            "pedestrian-free",
            115,
            "blind_goal",
            success=True,
            timeout=False,
            observation_ped_count=1,
        ),
        _row("pedestrian-free", 115, "social_force", observation_ped_count=1),
    ]
    report = analyze_release_rows(
        rows,
        config=CONFIG,
        source=_source("blind_goal", "social_force"),
    )

    assert not _findings(report, "pedestrian_free_baseline_regression")
    assert report["gate"]["blocked"] is False


def test_root_cause_annotation_removes_a_universal_failure_finding() -> None:
    """A matching annotation is retained in counts and clears the gate."""

    rows = [
        _row("narrow-doorway", 118, "planner_a"),
        _row("narrow-doorway", 118, "planner_b"),
    ]
    annotations = [
        {
            "scenario_id": "narrow-doorway",
            "seed": 118,
            "root_cause": "declared infeasible safe hold",
            "source_ref": "issue-9728",
        }
    ]

    report = analyze_release_rows(
        rows,
        config=CONFIG,
        annotations=annotations,
        source=_source("planner_a", "planner_b"),
    )

    assert not _findings(report, "universal_failure_unannotated")
    assert report["counts"]["annotated_universal_cells"] == 1
    assert report["gate"]["blocked"] is False


def test_unannotated_threshold_allows_configured_count() -> None:
    """The configured unannotated count is a threshold, not a Boolean switch."""

    config = {**CONFIG, "max_unannotated_findings": 1}
    rows = [
        _row("universal-failure", 119, "planner_a"),
        _row("universal-failure", 119, "planner_b"),
    ]

    report = analyze_release_rows(
        rows,
        config=config,
        source=_source("planner_a", "planner_b"),
    )
    finding = _findings(report, "universal_failure_unannotated")[0]

    assert finding["annotated"] is False
    assert report["gate"]["blocked"] is False


def test_finding_id_changes_with_detector_configuration() -> None:
    """Threshold changes cannot reuse a finding identity on the same rows."""

    rows = [_row("short-collision", 126, "planner_a", steps=1, collision=True, timeout=False)]
    source = _source("planner_a")
    first = analyze_release_rows(rows, config=CONFIG, source=source)
    second = analyze_release_rows(
        rows,
        config={**CONFIG, "short_collision_max_steps": 3},
        source=source,
    )

    assert (
        _findings(first, "short_collision")[0]["finding_id"]
        != _findings(second, "short_collision")[0]["finding_id"]
    )


def test_missing_row_metrics_do_not_create_phantom_measurement_findings() -> None:
    """Absent speed/trajectory summaries stay unavailable, never fabricated."""

    rows = [
        _row("missing-metrics", 120, "planner_a"),
        _row("missing-metrics", 120, "planner_b"),
    ]

    report = analyze_release_rows(
        rows,
        config=CONFIG,
        source=_source("planner_a", "planner_b"),
    )
    detector_ids = {
        finding["detector_id"]
        for finding in report["findings"]
        if finding["scenario_id"] == "missing-metrics"
    }

    assert "impossible_contact_speed" not in detector_ids
    assert "orbit_zero_progress" not in detector_ids
    assert report["missingness"]["path_length_unavailable"] == 2


def test_preflight_match_does_not_flag_invalid_run_accounting() -> None:
    """An invalid-run value agreed by the preflight is not an anomaly."""

    rows = [
        _row("accounted-invalid", 121, "planner_a", invalid_run=True),
        _row("accounted-invalid", 121, "planner_b", invalid_run=True),
    ]
    preflight = [{"scenario_id": "accounted-invalid", "seed": 121, "invalid_run": True}]

    report = analyze_release_rows(
        rows,
        config=CONFIG,
        preflight=preflight,
        source=_source("planner_a", "planner_b"),
    )

    assert not _findings(report, "invalid_run_preflight_mismatch")


def test_annotated_invalid_run_mismatch_still_blocks_gate() -> None:
    """Annotations preserve parity findings and cannot clear the release gate."""

    rows = [
        _row("accounting-mismatch", 124, "planner_a", invalid_run=True),
        _row("accounting-mismatch", 124, "planner_b", invalid_run=True),
    ]
    report = analyze_release_rows(
        rows,
        config=CONFIG,
        preflight=[{"scenario_id": "accounting-mismatch", "seed": 124, "invalid_run": False}],
        annotations=[
            {
                "scenario_id": "accounting-mismatch",
                "seed": 124,
                "root_cause": "known accounting discrepancy",
                "source_ref": "issue-9734-test",
            }
        ],
        source=_source("planner_a", "planner_b"),
    )

    parity_findings = _findings(report, "invalid_run_preflight_mismatch")
    assert len(parity_findings) == 1
    assert parity_findings[0]["annotated"] is True
    assert report["gate"]["blocked"] is True
    assert "invalid_run_preflight_mismatch" in report["gate"]["reasons"]


def test_required_preflight_without_input_is_explicitly_unavailable() -> None:
    """A gate requiring preflight reports unavailable evidence and blocks."""

    config = {**CONFIG, "require_preflight": True}
    report = analyze_release_rows(
        [_row("preflight-required", 122, "planner_a")],
        config=config,
        source=_source("planner_a"),
    )

    assert report["preflight_accounting"]["status"] == "unavailable"
    assert report["gate"]["blocked"] is True
    assert "preflight_accounting_unavailable_or_incomplete" in report["gate"]["reasons"]


def _write_extracted_bundle(root: Path, rows: list[dict[str, object]]) -> Path:
    """Write the minimal extracted-bundle layout consumed by the CLI loader."""

    payload = root / "payload"
    for row in rows:
        planner = str(row["algo"])
        path = payload / "runs" / f"{planner}__differential_drive" / "episodes.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    manifest_entries = []
    for path in sorted(payload.rglob("episodes.jsonl")):
        raw = path.read_bytes()
        manifest_entries.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    (root / "publication_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "benchmark-publication-bundle.v2",
                "files": manifest_entries,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return root


def test_cli_writes_json_and_markdown_and_returns_gate_status(tmp_path: Path) -> None:
    """The CLI emits both report formats and uses 1/0/2 gate statuses."""

    rows = [
        _row("cli-universal", 123, "planner_a"),
        _row("cli-universal", 123, "planner_b"),
    ]
    bundle = _write_extracted_bundle(tmp_path / "bundle", rows)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(CONFIG), encoding="utf-8")
    report_json = tmp_path / "report.json"
    report_md = tmp_path / "report.md"

    args = [
        "--bundle",
        str(bundle),
        "--config",
        str(config_path),
        "--output-json",
        str(report_json),
        "--output-markdown",
        str(report_md),
        "--release-gate",
    ]
    assert main(args) == 1
    assert report_json.is_file()
    assert report_md.is_file()
    assert json.loads(report_json.read_text(encoding="utf-8"))["gate"]["blocked"] is True
    assert "Release-row anomaly report" in report_md.read_text(encoding="utf-8")

    annotations_path = tmp_path / "annotations.json"
    annotations_path.write_text(
        json.dumps(
            [
                {
                    "scenario_id": "cli-universal",
                    "seed": 123,
                    "root_cause": "synthetic fixture",
                    "source_ref": "test",
                }
            ]
        ),
        encoding="utf-8",
    )
    assert main(args[:-1] + ["--annotations", str(annotations_path), "--release-gate"]) == 0

    malformed = tmp_path / "malformed.json"
    malformed.write_text("not-json", encoding="utf-8")
    assert main(args[:-1] + ["--config", str(malformed), "--release-gate"]) == 2

    mismatch_json = tmp_path / "digest-mismatch.json"
    mismatch_md = tmp_path / "digest-mismatch.md"
    assert (
        main(
            [
                *args[:-1],
                "--output-json",
                str(mismatch_json),
                "--output-markdown",
                str(mismatch_md),
                "--expected-bundle-sha256",
                "0" * 64,
                "--release-gate",
            ]
        )
        == 2
    )
    assert not mismatch_json.exists()
    assert not mismatch_md.exists()
