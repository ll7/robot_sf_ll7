"""Synthetic coverage for the release-row anomaly gate (issue #9734)."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from robot_sf.analysis_workbench import release_row_anomalies
from robot_sf.analysis_workbench.audit_contracts import (
    Signal,
    canonical_json,
    record_from_dict,
    record_to_dict,
)
from robot_sf.analysis_workbench.audit_store import AuditStore
from robot_sf.analysis_workbench.release_row_anomalies import (
    ReleaseRowError,
    analyze_release_rows,
    handoff_release_row_signals,
    main,
)
from robot_sf.analysis_workbench.release_row_bundle import load_release_rows
from robot_sf.benchmark.event_ledger import EPISODE_EVENT_LEDGER_SCHEMA_VERSION

if TYPE_CHECKING:
    from collections.abc import Mapping


CONFIG: dict[str, object] = {
    "short_collision_max_steps": 2,
    "same_step_max_steps": 5,
    "max_contact_speed_m_s": 100.0,
    "min_orbit_curvature": 10.0,
    "max_zero_progress_m": 0.05,
    "max_unannotated_findings": 0,
    "baseline_planner": "blind_goal",
    "pedestrian_free_scenarios": ["pedestrian-free"],
    "pedestrian_aware_planners": ["social_force"],
    "min_planners_per_cell": 2,
    "min_paired_cells": 1,
    "require_preflight": False,
}
BASIC_CONFIG = {**CONFIG, "pedestrian_free_scenarios": [], "pedestrian_aware_planners": []}
CANDIDATE_CONFIG = {
    **BASIC_CONFIG,
    "baseline_planner": "goal",
    "pedestrian_aware_planners": ["social_force"],
    "collision_metric_contract": "release_0_0_8",
    "collision_roster_status": "frozen",
    "collision_expected_arm_count": 2,
}


@pytest.fixture
def synthetic_collision_roster(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep arithmetic fixtures independent of the 14-arm campaign roster."""

    monkeypatch.setattr(
        release_row_anomalies, "_release_0_0_8_roster", lambda: {"goal", "social_force"}
    )


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
        "metrics": {
            "ped_collision_count": int(collision),
            "obstacle_collision_count": 0,
            "agent_collision_count": 0,
            "total_collision_count": int(collision),
            "collisions": int(collision),
            **dict(metrics or {}),
        },
    }


def _source(*planners: str) -> dict[str, object]:
    return {
        "release_id": "synthetic-issue-9734",
        "planner_ids": list(planners),
        "manifest_sha256": "f" * 64,
        "episode_members": [
            f"payload/runs/{planner}__differential_drive/episodes.jsonl" for planner in planners
        ],
    }


def _with_source_members(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    for row in rows:
        row["_source_member"] = f"payload/runs/{row['algo']}__differential_drive/episodes.jsonl"
    return rows


def _findings(report: Mapping[str, Any], detector_id: str) -> list[Mapping[str, Any]]:
    return [finding for finding in report["findings"] if finding["detector_id"] == detector_id]


def test_updated_legacy_report_keeps_collision_gate_opt_in() -> None:
    """The orbit method revision preserves the legacy collision-gate boundary."""

    report = analyze_release_rows(
        [
            _row("legacy-v1", 1, "goal", steps=100, success=True, timeout=False),
            _row("legacy-v1", 1, "social_force", steps=100, collision=True, timeout=False),
        ],
        source=_source("goal", "social_force"),
    )

    # Golden changed intentionally for orbit v1.1 (stall-window method/missingness).
    assert hashlib.sha256(canonical_json(report).encode()).hexdigest() == (
        "335fee4a91e5921c8df1256e7c1103e31cb2da753975d4cb727cc0e8a63669ee"
    )
    assert "collision_metric_contract" not in report["config"]
    assert "collision_metric_inconsistent" not in {
        item["detector_id"] for item in report["detector_registry"]["detectors"]
    }


def test_collision_metric_gate_blocks_vv5_double_total_and_missing_components(
    synthetic_collision_roster: None,
) -> None:
    """The exact two-row VV-5 mutant fails while its clean control passes."""

    template = json.loads(
        Path("configs/benchmarks/release_row_anomalies_0_0_8.template.json").read_text()
    )
    assert template["collision_metric_contract"] == "release_0_0_8"
    assert template["collision_roster_status"] == "unfrozen_template"
    assert template["pedestrian_aware_planners"] == []
    rows = [
        _row("vv5-fixture", 111, "goal", steps=100, success=True, timeout=False),
        _row("vv5-fixture", 111, "social_force", steps=100, collision=True, timeout=False),
    ]
    rows[0]["status"] = "success"
    rows[1]["status"] = "collision"
    source = _source("goal", "social_force")
    clean = analyze_release_rows(rows, config=CANDIDATE_CONFIG, source=source)
    assert clean["gate"]["blocked"] is False
    assert not _findings(clean, "collision_metric_inconsistent")
    assert clean["missingness"]["typed_collision_ledger_unavailable"] == 2

    doubled = deepcopy(rows)
    doubled[1]["metrics"]["total_collision_count"] = 2
    doubled[1]["metrics"]["collisions"] = 2
    report = analyze_release_rows(doubled, config=CANDIDATE_CONFIG, source=source)
    findings = _findings(report, "collision_metric_inconsistent")
    assert report["gate"]["blocked"] is True
    assert "collision_metric_inconsistent" in report["gate"]["reasons"]
    assert len(findings) == 1
    assert findings[0]["planner_id"] == "social_force"
    assert findings[0]["measured"]["problems"] == ["collision_component_sum_mismatch"]
    tolerated = analyze_release_rows(
        doubled,
        config={**CANDIDATE_CONFIG, "max_unannotated_findings": 10},
        source=source,
    )
    assert tolerated["gate"]["blocked"] is True
    assert tolerated["gate"]["reasons"] == ["collision_metric_inconsistent"]

    alias = deepcopy(rows)
    alias[1]["metrics"]["collisions"] = 2
    report = analyze_release_rows(alias, config=CANDIDATE_CONFIG, source=source)
    assert (
        "collision_alias_mismatch"
        in _findings(report, "collision_metric_inconsistent")[0]["measured"]["problems"]
    )

    missing = deepcopy(rows)
    del missing[1]["metrics"]["agent_collision_count"]
    report = analyze_release_rows(missing, config=CANDIDATE_CONFIG, source=source)
    assert report["gate"]["blocked"] is True
    assert (
        "missing_or_nonfinite_agent_collision_count"
        in _findings(report, "collision_metric_inconsistent")[0]["measured"]["problems"]
    )
    historical_diagnostic = analyze_release_rows(missing, config=BASIC_CONFIG, source=source)
    assert "collision_metric_contract" not in historical_diagnostic["config"]
    assert not _findings(historical_diagnostic, "collision_metric_inconsistent")
    assert historical_diagnostic["detector_registry_digest"] != report["detector_registry_digest"]
    unfrozen = analyze_release_rows(rows, config=template, source=source)
    assert "collision_roster_unfrozen_template" in unfrozen["gate"]["reasons"]
    assert "collision_roster_manifest_mismatch" in unfrozen["gate"]["reasons"]

    wrong_roster = analyze_release_rows(
        rows,
        config={**CANDIDATE_CONFIG, "pedestrian_aware_planners": ["old_hybrid"]},
        source=source,
    )
    assert "collision_roster_manifest_mismatch" in wrong_roster["gate"]["reasons"]


def test_strict_gate_rejects_self_declared_historical_hybrid_roster() -> None:
    """A matching bundle and config cannot override the committed #9751 roster."""

    frozen = release_row_anomalies._release_0_0_8_roster()
    assert frozen is not None
    assert len(frozen) == 14
    assert "hybrid_rule_v4_fast_progress_static_escape" in frozen
    historical = {
        "goal",
        *release_row_anomalies.DEFAULT_CONFIG["pedestrian_aware_planners"],
    }
    assert len(historical) == 14
    assert "hybrid_rule_v3_fast_progress_static_escape" in historical - frozen
    assert "scenario_adaptive_hybrid_orca_v2_collision_guard" in historical - frozen

    def report_for(roster: set[str]) -> dict[str, Any]:
        config = {
            **CANDIDATE_CONFIG,
            "pedestrian_aware_planners": sorted(roster - {"goal"}),
            "collision_expected_arm_count": len(roster),
        }
        rows = [
            _row("roster-fixture", 111, planner, success=True, timeout=False)
            for planner in sorted(roster)
        ]
        return analyze_release_rows(rows, config=config, source=_source(*sorted(roster)))

    assert report_for(frozen)["gate"]["blocked"] is False
    stale = report_for(historical)
    assert stale["gate"]["blocked"] is True
    assert "collision_roster_source_mismatch" in stale["gate"]["reasons"]
    assert "collision_roster_manifest_mismatch" not in stale["gate"]["reasons"]


def test_collision_metric_gate_preserves_large_integer_count_arithmetic(
    synthetic_collision_roster: None,
) -> None:
    """Adjacent counts beyond float precision must never compare equal."""

    count = 2**53
    rows = [
        _row("large-count", 111, "goal", steps=100, success=True, timeout=False),
        _row("large-count", 111, "social_force", steps=100, collision=True, timeout=False),
    ]
    rows[0]["status"] = "success"
    rows[1]["status"] = "collision"
    rows[1]["metrics"].update(
        ped_collision_count=count,
        total_collision_count=count,
        collisions=count,
    )
    source = _source("goal", "social_force")
    clean = analyze_release_rows(rows, config=CANDIDATE_CONFIG, source=source)
    assert clean["gate"]["blocked"] is False
    assert not _findings(clean, "collision_metric_inconsistent")

    small_integral_float = deepcopy(rows)
    small_integral_float[1]["metrics"].update(
        ped_collision_count=1.0,
        total_collision_count=1.0,
        collisions=1.0,
    )
    assert (
        analyze_release_rows(small_integral_float, config=CANDIDATE_CONFIG, source=source)["gate"][
            "blocked"
        ]
        is False
    )

    mutant = deepcopy(rows)
    mutant[1]["metrics"].update(total_collision_count=count + 1, collisions=count + 1)
    report = analyze_release_rows(mutant, config=CANDIDATE_CONFIG, source=source)
    findings = _findings(report, "collision_metric_inconsistent")
    assert report["gate"]["blocked"] is True
    assert len(findings) == 1
    assert findings[0]["measured"]["component_sum"] == count
    assert findings[0]["measured"]["problems"] == ["collision_component_sum_mismatch"]

    typed = deepcopy(rows)
    typed[1]["event_ledger"] = {
        "schema_version": EPISODE_EVENT_LEDGER_SCHEMA_VERSION,
        "exact_events": {"collision": True, "invalid_run": False},
        "reconciliation": {
            "collision_metric_value": count,
            "collision_metric_source": "metrics.total_collision_count",
        },
    }
    assert (
        analyze_release_rows(typed, config=CANDIDATE_CONFIG, source=source)["gate"]["blocked"]
        is False
    )
    typed[1]["event_ledger"]["reconciliation"]["collision_metric_value"] = count + 1
    result = analyze_release_rows(typed, config=CANDIDATE_CONFIG, source=source)
    assert result["gate"]["blocked"] is True
    assert (
        "ledger_collision_metric_mismatch"
        in _findings(result, "collision_metric_inconsistent")[0]["measured"]["problems"]
    )

    for bad_value, expected_problem in (
        (True, "missing_or_nonfinite_ped_collision_count"),
        (float("nan"), "missing_or_nonfinite_ped_collision_count"),
        (-1, "invalid_count_domain_ped_collision_count"),
        (1.5, "invalid_count_domain_ped_collision_count"),
        (float(count), "invalid_count_domain_ped_collision_count"),
    ):
        malformed = deepcopy(rows)
        malformed[1]["metrics"]["ped_collision_count"] = bad_value
        result = analyze_release_rows(malformed, config=CANDIDATE_CONFIG, source=source)
        findings = _findings(result, "collision_metric_inconsistent")
        assert result["gate"]["blocked"] is True
        assert expected_problem in findings[0]["measured"]["problems"]


def test_collision_metric_gate_reconciles_typed_ledger_without_counting_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exact events and sampled counts retain distinct meanings."""

    monkeypatch.setattr(release_row_anomalies, "_release_0_0_8_roster", lambda: {"goal"})

    row = _row("typed-ledger", 111, "goal", steps=100, collision=True, timeout=False)
    row["event_ledger"] = {
        "schema_version": EPISODE_EVENT_LEDGER_SCHEMA_VERSION,
        "exact_events": {"collision": True, "invalid_run": False},
        "collision_events": [{"collision_partner_type": "pedestrian"}] * 2,
        "reconciliation": {
            "collision_metric_value": 1,
            "collision_metric_source": "metrics.total_collision_count",
        },
    }
    source = _source("goal")
    ledger_config = {
        **CANDIDATE_CONFIG,
        "pedestrian_aware_planners": [],
        "collision_expected_arm_count": 1,
    }
    clean = analyze_release_rows([row], config=ledger_config, source=source)
    assert clean["gate"]["blocked"] is False

    stale = deepcopy(row)
    stale["event_ledger"]["reconciliation"]["collision_metric_value"] = 2
    report = analyze_release_rows([stale], config=ledger_config, source=source)
    assert report["gate"]["blocked"] is True
    assert (
        "ledger_collision_metric_mismatch"
        in _findings(report, "collision_metric_inconsistent")[0]["measured"]["problems"]
    )


def test_strict_collision_signal_handoff_accepts_versioned_registry(tmp_path: Path) -> None:
    """The strict collision detector is accepted by its matching BA handoff."""

    rows = _with_source_members(
        [
            _row("strict-handoff", 111, "goal", steps=100, success=True, timeout=False),
            _row("strict-handoff", 111, "social_force", steps=100, collision=True, timeout=False),
        ]
    )
    rows[1]["status"] = "collision"
    rows[1]["metrics"]["total_collision_count"] = 2
    rows[1]["metrics"]["collisions"] = 2
    report = analyze_release_rows(
        rows,
        config=CANDIDATE_CONFIG,
        source=_source("goal", "social_force"),
    )

    assert report["gate"]["blocked"] is True
    assert "collision_metric_inconsistent" in report["gate"]["reasons"]
    assert "collision_metric_inconsistent" in {
        item["detector_id"] for item in report["detector_registry"]["detectors"]
    }
    with AuditStore(tmp_path / "audit-store") as store:
        receipt = handoff_release_row_signals(report, store)
        assert receipt is not None
        assert receipt.committed is True
        persisted = store.get(report["signals"][0]["signal_id"])

    assert isinstance(persisted.record, Signal)
    assert persisted.record.detector_id == "collision_metric_inconsistent"


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
        analyze_release_rows(
            rows,
            config=(
                CONFIG if source["planner_ids"] == ["blind_goal", "social_force"] else BASIC_CONFIG
            ),
            source=source,
            preflight=preflight,
        )
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


def test_empty_pedestrian_free_scenario_list_discovers_observed_free_cells() -> None:
    """An empty scenario filter discovers cells from paired effective observations."""

    rows = [
        _row(
            "auto-detected-free",
            130,
            "blind_goal",
            success=True,
            timeout=False,
            observation_ped_count=0,
        ),
        _row("auto-detected-free", 130, "social_force", observation_ped_count=0),
    ]
    config = {**CONFIG, "pedestrian_free_scenarios": []}
    report = analyze_release_rows(
        rows,
        config=config,
        source=_source("blind_goal", "social_force"),
    )

    finding = _findings(report, "pedestrian_free_baseline_regression")[0]
    assert finding["scenario_id"] == "auto-detected-free"
    assert finding["planner_id"] == "social_force"


def test_auto_discovery_uses_aware_rows_when_baseline_has_no_free_cells() -> None:
    """A candidate's zero-pedestrian row cannot hide a non-free baseline row."""

    rows = [
        _row(
            "inconsistent-free-metadata",
            130,
            "blind_goal",
            success=True,
            timeout=False,
            observation_ped_count=1,
        ),
        _row(
            "inconsistent-free-metadata",
            130,
            "social_force",
            observation_ped_count=0,
        ),
    ]
    config = {**CONFIG, "pedestrian_free_scenarios": []}
    report = analyze_release_rows(
        rows,
        config=config,
        source=_source("blind_goal", "social_force"),
    )

    assert not _findings(report, "pedestrian_free_baseline_regression")
    assert report["missingness"]["pedestrian_free_pair_too_small"] == 1
    assert "pedestrian_comparison_cohort_incomplete" in report["gate"]["reasons"]
    assert report["gate"]["blocked"] is True


def test_auto_discovery_blocks_inconsistent_pedestrian_counts_in_a_partial_cohort() -> None:
    """One mismatched cell blocks even when enough consistent pairs remain."""

    rows = [
        _row(
            "mixed-free-metadata",
            130,
            "blind_goal",
            success=True,
            timeout=False,
            observation_ped_count=1,
        ),
        _row("mixed-free-metadata", 130, "social_force", observation_ped_count=0),
        _row(
            "mixed-free-metadata",
            131,
            "blind_goal",
            success=True,
            timeout=False,
            observation_ped_count=0,
        ),
        _row("mixed-free-metadata", 131, "social_force", observation_ped_count=0),
    ]
    config = {
        **CONFIG,
        "max_unannotated_findings": 5,
        "pedestrian_free_scenarios": [],
    }
    report = analyze_release_rows(
        rows,
        config=config,
        source=_source("blind_goal", "social_force"),
    )

    assert report["missingness"]["pedestrian_free_status_mismatch"] == 1
    assert "pedestrian_comparison_cohort_incomplete" in report["gate"]["reasons"]
    assert report["gate"]["blocked"] is True


def test_auto_discovery_blocks_when_pedestrian_counts_are_unavailable() -> None:
    """Missing metadata cannot make the automatic comparison cohort disappear."""

    rows = [
        _row("unknown-free-metadata", 130, "blind_goal"),
        _row("unknown-free-metadata", 130, "social_force"),
    ]
    for row in rows:
        del row["integrity"]["effective_view"]["observation_ped_count"]
    config = {**CONFIG, "pedestrian_free_scenarios": []}
    report = analyze_release_rows(
        rows,
        config=config,
        source=_source("blind_goal", "social_force"),
    )

    assert report["missingness"]["pedestrian_free_status_unavailable"] == 2
    assert "pedestrian_comparison_cohort_incomplete" in report["gate"]["reasons"]
    assert report["gate"]["blocked"] is True


def test_auto_discovery_skips_comparison_when_all_counts_are_known_nonzero() -> None:
    """A fully observed release with no pedestrian-free rows has no such comparison cohort."""

    rows = [
        _row(
            "known-pedestrian-cohort",
            130,
            "blind_goal",
            success=True,
            timeout=False,
            observation_ped_count=1,
        ),
        _row(
            "known-pedestrian-cohort",
            130,
            "social_force",
            observation_ped_count=1,
        ),
    ]
    config = {**CONFIG, "pedestrian_free_scenarios": []}
    report = analyze_release_rows(
        rows,
        config=config,
        source=_source("blind_goal", "social_force"),
    )

    assert not _findings(report, "pedestrian_free_baseline_regression")
    assert "pedestrian_comparison_cohort_incomplete" not in report["gate"]["reasons"]
    assert report["gate"]["blocked"] is False


def test_same_step_finding_reports_mixed_outcome_and_event_modes() -> None:
    """Shared timing stays visible while mixed terminal modes remain explicit."""

    timeout = _row("mixed-mode", 131, "planner_a", steps=2)
    collision = _row("mixed-mode", 131, "planner_b", steps=2, collision=True, timeout=False)
    collision["event_ledger"]["collision_events"] = [
        {
            "collision_partner_type": "pedestrian",
            "exact_event_source": "runtime.step.meta.is_pedestrian_collision",
            "relative_speed_at_contact": 0.5,
        }
    ]
    report = analyze_release_rows(
        [timeout, collision],
        config=BASIC_CONFIG,
        source=_source("planner_a", "planner_b"),
    )

    finding = _findings(report, "same_step_all_planners")[0]
    mode = finding["measured"]["reported_failure_mode"]
    assert mode["consistency"] == "mixed"
    assert mode["common_signature"] is None
    assert mode["by_planner"]["planner_b"]["collision_partner_types"] == ["pedestrian"]
    assert mode["root_cause_attribution"] == "unavailable_from_release_rows"


def test_same_step_finding_reports_consistent_collision_signature_without_cause_claim() -> None:
    """Matching observed event summaries are descriptive, not causal evidence."""

    rows = []
    for planner in ("planner_a", "planner_b"):
        row = _row("shared-collision", 132, planner, steps=2, collision=True, timeout=False)
        row["status"] = "collision"
        row["event_ledger"]["collision_events"] = [
            {
                "collision_partner_type": "pedestrian",
                "exact_event_source": "runtime.step.meta.is_pedestrian_collision",
                "relative_speed_at_contact": 0.5,
            }
        ]
        rows.append(row)

    report = analyze_release_rows(
        rows,
        config=BASIC_CONFIG,
        source=_source("planner_a", "planner_b"),
    )

    mode = _findings(report, "same_step_all_planners")[0]["measured"]["reported_failure_mode"]
    assert mode["consistency"] == "consistent"
    assert mode["common_signature"]["collision_partner_types"] == ["pedestrian"]
    assert mode["root_cause_attribution"] == "unavailable_from_release_rows"


def test_same_step_finding_marks_collision_signature_partially_observed() -> None:
    """Missing collision events stay visible as missing evidence in the signature."""

    rows = [
        _row("missing-event-summary", 133, planner, steps=2, collision=True, timeout=False)
        for planner in ("planner_a", "planner_b")
    ]
    for row in rows:
        row["status"] = "collision"

    report = analyze_release_rows(
        rows,
        config=BASIC_CONFIG,
        source=_source("planner_a", "planner_b"),
    )

    mode = _findings(report, "same_step_all_planners")[0]["measured"]["reported_failure_mode"]
    assert mode["consistency"] == "partially_observed"
    assert mode["common_signature"]["collision_event_count"] is None
    assert mode["common_signature"]["collision_partner_types"] is None


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
    assert report["missingness"]["pedestrian_free_pair_too_small"] == 1
    assert "pedestrian_comparison_cohort_incomplete" in report["gate"]["reasons"]
    assert report["gate"]["blocked"] is True


def test_pedestrian_free_detector_uses_only_configured_aware_planners() -> None:
    """Unconfigured comparison arms cannot create pedestrian-aware findings."""

    rows = [
        _row("pedestrian-free", 115, "blind_goal", success=True, timeout=False),
        _row("pedestrian-free", 115, "social_force"),
        _row("pedestrian-free", 115, "blind_other"),
    ]
    report = analyze_release_rows(
        rows,
        config=CONFIG,
        source=_source("blind_goal", "social_force", "blind_other"),
    )

    findings = _findings(report, "pedestrian_free_baseline_regression")
    assert [finding["planner_id"] for finding in findings] == ["social_force"]


def test_missing_configured_pedestrian_aware_planner_blocks_gate() -> None:
    """A configured comparison planner missing from the bundle is not silently skipped."""

    config = {**CONFIG, "pedestrian_aware_planners": ["social_force", "missing_planner"]}
    rows = [
        _row("pedestrian-free", 115, "blind_goal", success=True, timeout=False),
        _row("pedestrian-free", 115, "social_force"),
    ]
    report = analyze_release_rows(
        rows,
        config=config,
        source=_source("blind_goal", "social_force"),
    )

    assert report["missingness"]["pedestrian_aware_planner_missing"] == 1
    assert report["coverage"]["incomplete_cells"] == 1
    assert report["coverage"]["incomplete_cell_ids"][0]["missing_planners"] == ["missing_planner"]
    assert "pedestrian_aware_planner_missing" in report["gate"]["reasons"]
    assert report["gate"]["blocked"] is True


def test_missing_configured_aware_planner_blocks_without_pedestrian_free_cohort() -> None:
    """Whole-roster omissions block even when no pedestrian-free pair is discoverable."""

    config = {**CONFIG, "pedestrian_free_scenarios": [], "require_preflight": True}
    scenario = "pedestrian-occupied-only"
    rows = [_row(scenario, 115, "blind_goal", observation_ped_count=1)]
    report = analyze_release_rows(
        rows,
        config=config,
        source=_source("blind_goal"),
        preflight=[{"scenario_id": scenario, "seed": 115, "invalid_run": False}],
    )

    assert report["missingness"]["pedestrian_aware_planner_missing"] == 1
    assert report["coverage"]["incomplete_cells"] == 1
    assert report["coverage"]["incomplete_cell_ids"][0]["missing_planners"] == ["social_force"]
    assert "pedestrian_aware_planner_missing" in report["gate"]["reasons"]
    assert report["gate"]["blocked"] is True


def test_pedestrian_free_cohort_with_no_valid_pairs_blocks_gate() -> None:
    """Configured pedestrian-free comparisons cannot pass without enough valid pairs."""

    rows = [
        _row(
            "pedestrian-free",
            116,
            "blind_goal",
            success=True,
            timeout=False,
            observation_ped_count=0,
        ),
        _row("pedestrian-free", 116, "social_force", observation_ped_count=1),
    ]
    report = analyze_release_rows(
        rows,
        config=CONFIG,
        source=_source("blind_goal", "social_force"),
    )

    assert report["missingness"]["pedestrian_free_pair_too_small"] == 1
    assert "pedestrian_comparison_cohort_incomplete" in report["gate"]["reasons"]
    assert report["gate"]["blocked"] is True


def test_configured_pedestrian_free_scenario_without_baseline_blocks_gate() -> None:
    """A configured comparison scenario with no blind baseline is unavailable."""

    rows = [_row("pedestrian-free", 117, "social_force")]
    report = analyze_release_rows(
        rows,
        config=CONFIG,
        source=_source("social_force"),
    )

    assert report["missingness"]["pedestrian_free_baseline_missing"] == 1
    assert "pedestrian_comparison_cohort_incomplete" in report["gate"]["reasons"]
    assert report["gate"]["blocked"] is True


def test_auto_discovery_blocks_when_baseline_planner_is_absent() -> None:
    """A pedestrian-free candidate cannot disappear when the blind baseline is missing."""

    rows = [_row("auto-detected-free", 137, "social_force", observation_ped_count=0)]
    config = {**CONFIG, "pedestrian_free_scenarios": []}
    report = analyze_release_rows(
        rows,
        config=config,
        source=_source("social_force"),
    )

    assert report["missingness"]["pedestrian_free_baseline_planner_unavailable"] == 1
    assert "pedestrian_baseline_planner_unavailable" in report["gate"]["reasons"]
    assert report["gate"]["blocked"] is True


def test_contact_speed_metric_fallback_is_used_when_events_have_no_speed() -> None:
    """An empty/incomplete event list does not suppress the row metric fallback."""

    row = _row("teleport-contact", 113, "planner_a", collision=True, timeout=False)
    row["event_ledger"]["collision_events"] = []
    row["metrics"]["max_relative_contact_speed_m_s"] = 710.0

    report = analyze_release_rows([row], config=BASIC_CONFIG, source=_source("planner_a"))

    finding = _findings(report, "impossible_contact_speed")[0]
    assert finding["measured"]["max_relative_speed_at_contact_m_s"] == 710.0
    assert finding["measured"]["speed_source"] == "metrics.max_relative_contact_speed_m_s"


def test_non_admissible_execution_rows_are_excluded_and_block_gate() -> None:
    """Fallback rows do not count as benchmark episodes; outcome failures still do."""

    admitted = _row("execution-status", 127, "planner_a", steps=1)
    admitted["status"] = "failure"
    unavailable = _row("execution-status", 127, "planner_b", steps=1)
    unavailable["row_status"] = "fallback"
    report = analyze_release_rows(
        [admitted, unavailable],
        config=BASIC_CONFIG,
        source=_source("planner_a", "planner_b"),
    )

    assert report["execution_admission"]["eligible_rows"] == 1
    assert report["execution_admission"]["unavailable_rows"] == 1
    assert report["counts"]["rows"] == 2
    assert not _findings(report, "same_step_all_planners")
    assert "execution_admission_incomplete" in report["gate"]["reasons"]
    assert "incomplete_planner_cells" in report["gate"]["reasons"]


def test_invalid_top_level_statuses_cannot_bypass_execution_admission() -> None:
    """Only documented terminal outcomes are exempted from execution-status admission."""

    for invalid_status in ("fallback", "degraded", "unavailable", "timeout", "garbage"):
        admitted = _row(f"top-status-{invalid_status}", 134, "planner_a")
        admitted["status"] = "failure"
        invalid = _row(f"top-status-{invalid_status}", 134, "planner_b")
        invalid["status"] = invalid_status
        report = analyze_release_rows(
            [admitted, invalid],
            config=BASIC_CONFIG,
            source=_source("planner_a", "planner_b"),
        )

        assert report["execution_admission"]["eligible_rows"] == 1
        assert report["execution_admission"]["unavailable_rows"] == 1
        assert "execution_admission_incomplete" in report["gate"]["reasons"]


def test_release_row_signals_commit_to_audit_store_idempotently(tmp_path: Path) -> None:
    """The aggregate detector's typed BA-03 signals have a durable Auditor handoff."""

    report = analyze_release_rows(
        _with_source_members(
            [
                _row("auditor-handoff", 128, "planner_a", steps=2, collision=True, timeout=False),
                _row("auditor-handoff", 128, "planner_b", steps=2, collision=True, timeout=False),
            ]
        ),
        config=BASIC_CONFIG,
        source=_source("planner_a", "planner_b"),
    )
    assert len(report["signals"]) > 1

    with AuditStore(tmp_path / "audit-store") as store:
        first = handoff_release_row_signals(report, store)
        report["signals"].reverse()
        second = handoff_release_row_signals(report, store)
        persisted = store.get(report["signals"][0]["signal_id"])

    assert first.operation_id == second.operation_id
    assert first.committed is True and first.replayed is False
    assert second.replayed is True
    assert isinstance(persisted.record, Signal)
    assert persisted.record.signal_id == report["signals"][0]["signal_id"]


def test_release_row_auditor_handoff_binds_signals_to_findings(tmp_path: Path) -> None:
    """A typed but modified BA-03 payload cannot diverge from its report finding."""

    report = analyze_release_rows(
        _with_source_members(
            [_row("auditor-handoff", 135, "planner_a", steps=2, collision=True, timeout=False)]
        ),
        config=CONFIG,
        source=_source("planner_a"),
    )
    report["signals"][0]["measured"]["forged_speed_m_s"] = 999999.0

    with AuditStore(tmp_path / "audit-store") as store:
        try:
            handoff_release_row_signals(report, store)
        except ReleaseRowError as error:
            assert "match report findings" in str(error)
        else:
            raise AssertionError("signal payload divergent from its finding was accepted")
        assert store.list_records(record_type="signal") == []


def test_release_row_auditor_handoff_rejects_unlisted_finding_source_member(
    tmp_path: Path,
) -> None:
    """A finding cannot project a member outside its verified bundle into BA-03."""

    report = analyze_release_rows(
        _with_source_members(
            [_row("auditor-handoff", 135, "planner_a", steps=2, collision=True, timeout=False)]
        ),
        config=CONFIG,
        source=_source("planner_a"),
    )
    finding = report["findings"][0]
    unlisted_member = "payload/runs/unlisted__differential_drive/episodes.jsonl"
    finding["source_members"] = [unlisted_member]
    signal = next(item for item in report["signals"] if item["signal_id"] == finding["finding_id"])
    for evidence in signal["evidence"]:
        if evidence.get("kind") == "source_identity":
            evidence["source_members"] = [unlisted_member]

    with AuditStore(tmp_path / "audit-store") as store:
        try:
            handoff_release_row_signals(report, store)
        except ReleaseRowError as error:
            assert "source members do not match source identity" in str(error)
        else:
            raise AssertionError("an unlisted finding source member was accepted")
        assert store.list_records(record_type="signal") == []


def test_release_row_auditor_handoff_binds_finding_ids_to_source_digest(tmp_path: Path) -> None:
    """Changing the claimed manifest identity invalidates source-bound finding IDs."""

    report = analyze_release_rows(
        _with_source_members(
            [_row("auditor-handoff", 136, "planner_a", steps=2, collision=True, timeout=False)]
        ),
        config=CONFIG,
        source=_source("planner_a"),
    )
    report["source"]["manifest_sha256"] = "e" * 64

    with AuditStore(tmp_path / "audit-store") as store:
        try:
            handoff_release_row_signals(report, store)
        except ReleaseRowError as error:
            assert "source identity" in str(error)
        else:
            raise AssertionError("finding with mismatched source digest was accepted")


def test_release_row_auditor_handoff_rejects_malformed_signal(tmp_path: Path) -> None:
    """Only canonical BA-03 Signal records can be handed to the Auditor store."""

    report = analyze_release_rows(
        _with_source_members(
            [_row("auditor-handoff", 129, "planner_a", steps=2, collision=True, timeout=False)]
        ),
        config=CONFIG,
        source=_source("planner_a"),
    )
    report["signals"] = [{"schema_version": "audit-record.v1", "record_type": "annotation"}]

    with AuditStore(tmp_path / "audit-store") as store:
        try:
            handoff_release_row_signals(report, store)
        except ReleaseRowError as error:
            assert "signal" in str(error).lower()
        else:
            raise AssertionError("malformed signal payload was accepted")


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
            "manifest_sha256": "f" * 64,
            "root_cause": "declared infeasible safe hold",
            "source_ref": "issue-9728",
        }
    ]

    report = analyze_release_rows(
        rows,
        config=BASIC_CONFIG,
        annotations=annotations,
        source=_source("planner_a", "planner_b"),
    )

    assert not _findings(report, "universal_failure_unannotated")
    assert report["counts"]["annotated_universal_cells"] == 1
    assert report["gate"]["blocked"] is False


def test_scenario_annotation_does_not_cross_manifest_sources() -> None:
    """A scenario-level annotation only applies to the manifest it names."""

    rows = [
        _row("narrow-doorway", 118, "planner_a"),
        _row("narrow-doorway", 118, "planner_b"),
    ]
    report = analyze_release_rows(
        rows,
        config=BASIC_CONFIG,
        annotations=[
            {
                "scenario_id": "narrow-doorway",
                "seed": 118,
                "manifest_sha256": "e" * 64,
                "root_cause": "annotation belongs to another release",
                "source_ref": "issue-9728",
            }
        ],
        source=_source("planner_a", "planner_b"),
    )

    finding = _findings(report, "universal_failure_unannotated")[0]
    assert finding["annotated"] is False
    assert report["gate"]["blocked"] is True


def test_unannotated_threshold_allows_configured_count() -> None:
    """The configured unannotated count is a threshold, not a Boolean switch."""

    config = {**BASIC_CONFIG, "max_unannotated_findings": 1}
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
    first = analyze_release_rows(rows, config=BASIC_CONFIG, source=source)
    second = analyze_release_rows(
        rows,
        config={**BASIC_CONFIG, "short_collision_max_steps": 3},
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
        config=BASIC_CONFIG,
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
        config=BASIC_CONFIG,
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
        config=BASIC_CONFIG,
        preflight=[{"scenario_id": "accounting-mismatch", "seed": 124, "invalid_run": False}],
        annotations=[
            {
                "scenario_id": "accounting-mismatch",
                "seed": 124,
                "manifest_sha256": "f" * 64,
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

    config = {**BASIC_CONFIG, "require_preflight": True}
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
    api_rows, source = load_release_rows(bundle)
    api_report = analyze_release_rows(api_rows, config=BASIC_CONFIG, source=source)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(BASIC_CONFIG), encoding="utf-8")
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
    cli_report = json.loads(report_json.read_text(encoding="utf-8"))
    assert cli_report["detector_registry_digest"] == api_report["detector_registry_digest"]
    assert cli_report["findings"] == api_report["findings"]
    markdown = report_md.read_text(encoding="utf-8")
    assert "Release-row anomaly report" in markdown
    assert markdown.index("Diagnostic row signals only") < markdown.index("Detector gate status")
    assert "## Missing evidence" in markdown

    audit_store = tmp_path / "benchmark-auditor"
    assert main(args[:-1] + ["--audit-store", str(audit_store), "--release-gate"]) == 1
    handed_off = json.loads(report_json.read_text(encoding="utf-8"))
    assert handed_off["auditor_handoff"]["status"] == "submitted"
    with AuditStore(audit_store) as store:
        assert len(store.list_records(record_type="signal")) == len(handed_off["signals"])

    annotations_path = tmp_path / "annotations.json"
    annotations_path.write_text(
        json.dumps(
            [
                {
                    "scenario_id": "cli-universal",
                    "seed": 123,
                    "manifest_sha256": source["manifest_sha256"],
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
