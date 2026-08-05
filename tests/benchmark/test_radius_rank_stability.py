"""Tests for radius rank-stability analysis (issue #6643, Gate 3 of #6600)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from robot_sf.benchmark.radius_rank_stability import (
    ANALYSIS_BLOCKED_PENDING_GATE2,
    EXPECTED_PLANNER_ROSTER,
    EXPECTED_ROWS_PER_ARM,
    EXPECTED_SCENARIO_NAMES,
    RADIUS_EVIDENCE_BUNDLE_SCHEMA,
    RADIUS_RANK_STABILITY_SCHEMA,
    REQUIRED_CLAIM_BOUNDARY_PHRASES,
    VERDICT_INVALID,
    VERDICT_NON_IDENTIFIABLE,
    VERDICT_RADIUS_DEPENDENT,
    VERDICT_STABLE,
    _float_keyed,
    _metric_identifiable,
    _radius_key,
    analyze_metric_rank_stability,
    analyze_radius_sensitivity,
    build_evidence_provenance,
    build_missingness_ledger,
    compute_family_transitions,
    compute_paired_changes,
    decide_radius_verdict,
    evidence_tier_for_verdict,
    load_sweep_summary,
    render_propagation_comment,
    render_verdict_comment,
    sweep_summary_available,
    write_evidence_bundle,
)

if TYPE_CHECKING:
    from types import ModuleType

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RADII = (0.5, 0.8, 1.0)
_BASELINE = 1.0


def _load_cli() -> ModuleType:
    """Load the CLI module by path because scripts/benchmark is not a package."""
    module_path = _REPO_ROOT / "scripts/benchmark/analyze_radius_rank_stability_issue_6643.py"
    spec = importlib.util.spec_from_file_location("radius_rank_stability_cli", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# --- synthetic sweep builders ---------------------------------------------


def _accounting(declared: int, present: int, excluded: dict[str, int] | None = None) -> dict:
    return {"declared": declared, "present": present, "excluded_by_reason": excluded or {}}


def _full_accounting() -> dict:
    return {
        radius: _accounting(EXPECTED_ROWS_PER_ARM, EXPECTED_ROWS_PER_ARM)
        for radius in ("0.5", "0.8", "1.0")
    }


def _stable_tables() -> dict:
    """Return metric tables whose planner order is identical at every radius."""
    tables = {
        "1.0": {
            "orca": {"success": 0.9, "typed_collisions": 0.1, "snqi": 0.8},
            "ppo": {"success": 0.7, "typed_collisions": 0.2, "snqi": 0.6},
            "goal": {"success": 0.3, "typed_collisions": 0.5, "snqi": 0.3},
        },
        "0.8": {
            "orca": {"success": 0.88, "typed_collisions": 0.12, "snqi": 0.78},
            "ppo": {"success": 0.68, "typed_collisions": 0.22, "snqi": 0.58},
            "goal": {"success": 0.28, "typed_collisions": 0.52, "snqi": 0.28},
        },
        "0.5": {
            "orca": {"success": 0.85, "typed_collisions": 0.15, "snqi": 0.75},
            "ppo": {"success": 0.65, "typed_collisions": 0.25, "snqi": 0.55},
            "goal": {"success": 0.25, "typed_collisions": 0.55, "snqi": 0.25},
        },
    }
    extra_planners = tuple(
        planner for planner in EXPECTED_PLANNER_ROSTER if planner not in {"orca", "ppo", "goal"}
    )
    for table in tables.values():
        for index, planner in enumerate(extra_planners):
            table[planner] = {
                "success": 0.1 - index * 0.001,
                "typed_collisions": 0.8 + index * 0.001,
                "snqi": 0.1 - index * 0.001,
            }
    return tables


def _flip_tables() -> dict:
    """Return metric tables whose planner order flips at 0.5 m."""
    tables = _stable_tables()
    tables["0.5"] = {
        "orca": {"success": 0.2, "typed_collisions": 0.6, "snqi": 0.2},
        "ppo": {"success": 0.65, "typed_collisions": 0.25, "snqi": 0.55},
        "goal": {"success": 0.9, "typed_collisions": 0.05, "snqi": 0.9},
    }
    extra_planners = tuple(
        planner for planner in EXPECTED_PLANNER_ROSTER if planner not in {"orca", "ppo", "goal"}
    )
    for index, planner in enumerate(extra_planners):
        tables["0.5"][planner] = {
            "success": 0.1 - index * 0.001,
            "typed_collisions": 0.8 + index * 0.001,
            "snqi": 0.1 - index * 0.001,
        }
    return tables


def _complete_family_feasibility() -> dict:
    """Return matched family evidence, including the geometry-sensitive doorway family."""
    return {
        radius: {"narrow_doorway": "feasible", "crossing": "feasible"}
        for radius in ("0.5", "0.8", "1.0")
    }


def _complete_paired_observations(tables: dict) -> dict:
    """Return finite seed-keyed observations for every arm/planner/rank metric."""
    return {
        radius: {
            planner: {
                metric: {
                    str(seed): values.get(metric, 0.0) if isinstance(values, dict) else 0.0
                    for seed in range(111, 141)
                }
                for metric in ("success", "typed_collisions", "snqi")
            }
            for planner, values in table.items()
        }
        for radius, table in tables.items()
    }


def _campaign_provenance(
    config_sha256: str = "f" * 64, canary_receipt_sha256: str = "d" * 64
) -> dict:
    """Return one immutable campaign/config/canary binding for all arms."""
    return {
        radius: {
            "campaign_commit": "c" * 40,
            "config_sha256": config_sha256,
            "gate1_canary_receipt_sha256": canary_receipt_sha256,
        }
        for radius in ("0.5", "0.8", "1.0")
    }


def _bind_summary_evidence(summary: dict, config_path: Path, canary_receipt_path: Path) -> dict:
    """Bind a synthetic summary to its actual config and Gate 1 receipt fixtures."""
    summary["campaign_provenance"] = _campaign_provenance(
        hashlib.sha256(config_path.read_bytes()).hexdigest(),
        hashlib.sha256(canary_receipt_path.read_bytes()).hexdigest(),
    )
    return summary


def _write_gate1_receipt(path: Path, *, go: bool = True) -> None:
    """Write a minimal schema-valid Gate 1 report fixture."""
    surfaces = (
        "simulator_collision_geometry",
        "obstacle_pedestrian_contact_logic",
        "feasibility_oracle",
        "metric_metadata_and_output_rows",
        "planner_inputs",
    )
    path.write_text(
        json.dumps(
            {
                "schema": "radius_binding_canary_report.v1",
                "canary_schema": "radius_binding_canary.v1",
                "campaign": "issue_6600_gate_1",
                "issue": 6641,
                "parent_issue": 6600,
                "radii_m": [0.5, 0.8, 1.0],
                "go": go,
                "verdicts": [
                    {
                        "schema": "radius_binding_canary.v1",
                        "target_radius_m": radius,
                        "go": go,
                        "surfaces": [{"surface": surface, "bound": go} for surface in surfaces],
                    }
                    for radius in (0.5, 0.8, 1.0)
                ],
            }
        ),
        encoding="utf-8",
    )


def _sweep_summary(
    tables: dict,
    accounting: dict | None = None,
    *,
    family: dict | None = None,
    paired: dict | None = None,
) -> dict:
    summary = {
        "schema_version": "issue_6642_radius_sweep_summary.v1",
        "radii_m": list(_RADII),
        "planners": list(EXPECTED_PLANNER_ROSTER),
        "scenario_matrix": "configs/scenarios/classic_interactions_francis2023.yaml",
        "scenario_cells": list(EXPECTED_SCENARIO_NAMES),
        "seeds": list(range(111, 141)),
        "metric_tables": tables,
        "row_accounting": accounting if accounting is not None else _full_accounting(),
    }
    summary["family_feasibility"] = family if family is not None else _complete_family_feasibility()
    summary["paired_observations"] = (
        paired if paired is not None else _complete_paired_observations(tables)
    )
    summary["campaign_provenance"] = _campaign_provenance()
    return summary


# --- primitives ------------------------------------------------------------


def test_radius_key_is_canonical() -> None:
    """Canonical radius keys drop trailing zeros for stable serialization."""
    assert _radius_key(1.0) == "1"
    assert _radius_key(0.5) == "0.5"
    assert _radius_key(0.8) == "0.8"


def test_float_keyed_normalizes_radius_spellings() -> None:
    """String radius keys '1', '1.0', and '0.5' normalize to numeric keys."""
    normalized = _float_keyed({"1.0": "a", "0.5": "b", "not-a-radius": "c"})
    assert normalized == {1.0: "a", 0.5: "b"}


def test_float_keyed_rejects_non_mapping() -> None:
    """Non-mapping input yields an empty normalized mapping."""
    assert _float_keyed(None) == {}
    assert _float_keyed(["1.0"]) == {}


def test_metric_identifiable_requires_variance() -> None:
    """All-tied and insufficient metric values are non-identifiable."""
    tied = {"a": {"m": 0.5}, "b": {"m": 0.5}}
    identifiable, reason = _metric_identifiable(tied, "m")
    assert identifiable is False
    assert reason == "primary_metric_zero_variance"

    single = {"a": {"m": 0.5}}
    identifiable, reason = _metric_identifiable(single, "m")
    assert identifiable is False
    assert reason == "primary_metric_insufficient_finite_values"

    varied = {"a": {"m": 0.5}, "b": {"m": 0.9}}
    assert _metric_identifiable(varied, "m") == (True, None)


# --- Gate 1 canary surface vocabulary -------------------------------------


def test_gate1_canary_surface_vocabulary_matches_real_emitter() -> None:
    """The Gate 3 checker must accept the surface names the Gate 1 canary emits.

    The merged Gate 1 canary (robot_sf/benchmark/radius_binding_canary.py,
    SURFACE_METRIC_METADATA) emits ``metric_metadata_and_output_rows``, not the
    shortened ``metric_metadata``. A mismatch here silently rejects a real passing
    receipt at promotion time, so the expected set is locked to the emitter name.
    """
    from robot_sf.benchmark.radius_rank_stability import GATE1_CANARY_SURFACES

    assert "metric_metadata_and_output_rows" in GATE1_CANARY_SURFACES
    assert "metric_metadata" not in GATE1_CANARY_SURFACES


def test_real_gate1_receipt_shape_passes_checker(tmp_path: Path) -> None:
    """A schema-valid Gate 1 report with the emitter surface names is accepted."""
    from robot_sf.benchmark.radius_rank_stability import _gate1_canary_receipt_is_passing

    receipt_path = tmp_path / "canary.json"
    _write_gate1_receipt(receipt_path)
    assert _gate1_canary_receipt_is_passing(receipt_path) is True


# --- missingness ledger ----------------------------------------------------


def test_missingness_ledger_complete_when_all_rows_reconcile() -> None:
    """Complete accounting with no exclusions yields a complete ledger."""
    ledger = build_missingness_ledger(_sweep_summary(_stable_tables()))
    assert ledger.complete is True
    assert ledger.blocking_reasons == ()
    assert ledger.declared_total == EXPECTED_ROWS_PER_ARM * 3
    assert ledger.present_total == EXPECTED_ROWS_PER_ARM * 3
    assert ledger.excluded_total == 0


def test_missingness_ledger_flags_unaccounted_rows() -> None:
    """Declared rows that are neither present nor excluded block interpretation."""
    accounting = _full_accounting()
    accounting["0.5"] = _accounting(12, 9)
    ledger = build_missingness_ledger(_sweep_summary(_stable_tables(), accounting))
    assert ledger.complete is False
    assert "radius_0.5_incomplete_accounting" in ledger.blocking_reasons


def test_missingness_ledger_flags_excluded_fallback_rows() -> None:
    """Fallback exclusions are disqualifying even when accounting reconciles."""
    accounting = _full_accounting()
    accounting["0.8"] = _accounting(12, 10, {"fallback": 2})
    ledger = build_missingness_ledger(_sweep_summary(_stable_tables(), accounting))
    assert ledger.complete is False
    assert "radius_0.8_excluded_fallback" in ledger.blocking_reasons
    assert ledger.excluded_by_reason["fallback"] == 2


def test_missingness_ledger_flags_missing_accounting() -> None:
    """A radius arm with no accounting record is a blocking gap."""
    accounting = {"0.5": _accounting(12, 12), "0.8": _accounting(12, 12)}
    ledger = build_missingness_ledger(_sweep_summary(_stable_tables(), accounting))
    assert ledger.complete is False
    assert "radius_1_missing_row_accounting" in ledger.blocking_reasons


def test_missingness_ledger_rejects_underdeclared_scope() -> None:
    """A self-consistent tiny row count cannot masquerade as the full campaign."""
    accounting = _full_accounting()
    accounting["0.5"] = _accounting(1, 1)
    ledger = build_missingness_ledger(_sweep_summary(_stable_tables(), accounting))
    assert ledger.complete is False
    assert any("unexpected_declared_row_count:1" in reason for reason in ledger.blocking_reasons)


def test_missingness_ledger_rejects_fake_scenario_scope() -> None:
    """A 48-cell placeholder roster cannot satisfy the fixed campaign scope."""
    summary = _sweep_summary(_stable_tables())
    summary["scenario_cells"][0] = "not_a_release_scenario"
    ledger = build_missingness_ledger(summary)
    assert ledger.complete is False
    assert "scenario_cell_roster_mismatch" in ledger.blocking_reasons


def test_missingness_ledger_rejects_malformed_accounting() -> None:
    """Unknown reasons and negative counts cannot reconcile valid arms."""
    accounting = _full_accounting()
    accounting["0.5"] = _accounting(12, 11, {"unknown": 1})
    accounting["0.8"] = _accounting(12, 12, {"fallback": -1})
    ledger = build_missingness_ledger(_sweep_summary(_stable_tables(), accounting))
    assert ledger.complete is False
    assert "radius_0.5_incomplete_accounting" in ledger.blocking_reasons
    assert any("unknown_exclusion_reason" in reason for reason in ledger.blocking_reasons)
    assert any("invalid_count:fallback" in reason for reason in ledger.blocking_reasons)


def test_missingness_ledger_rejects_missing_metric_arm() -> None:
    """A row-complete summary with a missing metric arm cannot promote a verdict."""
    tables = _stable_tables()
    del tables["0.5"]
    ledger = build_missingness_ledger(_sweep_summary(tables))
    assert ledger.complete is False
    assert "radius_0.5_missing_metric_table" in ledger.blocking_reasons


def test_missingness_ledger_rejects_invalid_metric_row() -> None:
    """An unrecognized planner row cannot be silently discarded from coverage."""
    tables = _stable_tables()
    tables["0.5"]["unexpected"] = None
    ledger = build_missingness_ledger(_sweep_summary(tables))
    assert ledger.complete is False
    assert "radius_0.5_invalid_metric_row" in ledger.blocking_reasons


def test_missingness_ledger_rejects_missing_required_metric() -> None:
    """A missing rank metric is invalid evidence, not merely a tied result."""
    tables = _stable_tables()
    del tables["0.5"]["goal"]["snqi"]
    ledger = build_missingness_ledger(_sweep_summary(tables))
    assert ledger.complete is False
    assert "radius_0.5_planner_goal_invalid_metric:snqi" in ledger.blocking_reasons


def test_missingness_ledger_rejects_incomplete_planner_roster() -> None:
    """Gate 3 cannot promote a subset of the release planner roster."""
    summary = _sweep_summary(_stable_tables())
    summary["planners"] = ["orca", "ppo", "goal"]
    ledger = build_missingness_ledger(summary)
    assert ledger.complete is False
    assert "unexpected_planner_roster" in ledger.blocking_reasons


def test_missingness_ledger_rejects_non_three_radius_summary() -> None:
    """Gate 3 requires all three preregistered radius arms."""
    summary = _sweep_summary(_stable_tables())
    summary["radii_m"] = [1.0]
    ledger = build_missingness_ledger(summary)
    assert ledger.complete is False
    assert any(reason.startswith("unexpected_radius_arms:") for reason in ledger.blocking_reasons)


# --- per-metric rank stability --------------------------------------------


def test_metric_rank_stability_stable_order() -> None:
    """Identical orders give tau 1.0, zero flips, and unchanged top-1."""
    stability = analyze_metric_rank_stability(
        _sweep_summary(_stable_tables()), "success", baseline_radius=_BASELINE
    )
    assert stability.identifiable is True
    assert stability.baseline_ranking[:3] == ["orca", "ppo", "goal"]
    assert len(stability.baseline_ranking) == len(EXPECTED_PLANNER_ROSTER)
    assert stability.kendall_tau_by_radius[0.5] == 1.0
    assert stability.rank_flips_by_radius[0.5] == 0
    assert stability.top1_changed_by_radius[0.5] is False
    assert stability.flipping_radii == ()


def test_metric_rank_stability_detects_flip() -> None:
    """A reversed order at 0.5 m is reported as flips and a top-1 change."""
    stability = analyze_metric_rank_stability(
        _sweep_summary(_flip_tables()), "success", baseline_radius=_BASELINE
    )
    assert stability.rank_flips_by_radius[0.5] == 3
    assert stability.top1_changed_by_radius[0.5] is True
    assert stability.flipping_radii == (0.5,)
    assert stability.kendall_tau_by_radius[0.5] < 1.0


def test_metric_rank_stability_lower_is_better_default() -> None:
    """Typed collisions default to lower-is-better ranking direction."""
    stability = analyze_metric_rank_stability(
        _sweep_summary(_stable_tables()), "typed_collisions", baseline_radius=_BASELINE
    )
    assert stability.higher_is_better is False
    assert stability.baseline_ranking[:3] == ["orca", "ppo", "goal"]


def test_metric_rank_stability_non_identifiable_baseline() -> None:
    """An all-tied baseline metric is non-identifiable with null rank evidence."""
    tables = _stable_tables()
    tables["1.0"] = {planner: {"success": 0.5} for planner in EXPECTED_PLANNER_ROSTER}
    stability = analyze_metric_rank_stability(
        _sweep_summary(tables), "success", baseline_radius=_BASELINE
    )
    assert stability.identifiable is False
    assert stability.baseline_identifiable is False
    assert stability.kendall_tau_by_radius[0.5] is None
    assert stability.rank_flips_by_radius[0.5] is None


# --- paired changes --------------------------------------------------------


def test_paired_changes_without_observations_report_point_delta() -> None:
    """Without per-seed pairs the delta is the table difference with a null interval."""
    summary = _sweep_summary(_stable_tables())
    del summary["paired_observations"]
    changes = compute_paired_changes(
        summary,
        "success",
        baseline_radius=_BASELINE,
        radii=_RADII,
    )
    orca = next(c for c in changes[0.5] if c.planner == "orca")
    assert orca.delta == pytest.approx(-0.05)
    assert orca.ci_low is None
    assert orca.ci_high is None
    assert orca.reason == "no_paired_observations"


def test_paired_changes_with_observations_are_deterministic() -> None:
    """Paired bootstrap intervals are finite and reproducible for a fixed seed."""
    paired = {
        "1.0": {
            "orca": {"success": {"111": 0.9, "112": 0.85, "113": 0.95, "114": 0.9, "115": 0.88}}
        },
        "0.5": {
            "orca": {"success": {"111": 0.8, "112": 0.75, "113": 0.85, "114": 0.8, "115": 0.78}}
        },
    }
    summary = _sweep_summary(_stable_tables(), paired=paired)
    first = compute_paired_changes(
        summary, "success", baseline_radius=_BASELINE, radii=_RADII, seed=123
    )
    second = compute_paired_changes(
        summary, "success", baseline_radius=_BASELINE, radii=_RADII, seed=123
    )
    orca_first = next(c for c in first[0.5] if c.planner == "orca")
    orca_second = next(c for c in second[0.5] if c.planner == "orca")
    assert orca_first.delta == pytest.approx(-0.1)
    assert orca_first.ci_low == pytest.approx(orca_second.ci_low)
    assert orca_first.ci_high == pytest.approx(orca_second.ci_high)
    assert orca_first.ci_low <= orca_first.delta <= orca_first.ci_high
    assert orca_first.n_pairs == 5
    assert orca_first.reason is None


def test_paired_changes_preserve_seed_alignment_when_values_are_nonfinite() -> None:
    """Missing keyed values are dropped pairwise rather than shifting seed alignment."""
    paired = {
        "1.0": {"orca": {"success": {"111": 0.9, "113": 0.95}}},
        "0.5": {"orca": {"success": {"111": 0.8, "112": 0.7, "113": 0.85}}},
    }
    changes = compute_paired_changes(
        _sweep_summary(_stable_tables(), paired=paired),
        "success",
        baseline_radius=_BASELINE,
        radii=_RADII,
    )
    orca = next(change for change in changes[0.5] if change.planner == "orca")
    assert orca.delta == pytest.approx(-0.1)
    assert orca.n_pairs == 2


# --- family transitions ----------------------------------------------------


def test_family_transitions_flag_narrow_doorway() -> None:
    """The narrow-doorway family is flagged and its transition is detected."""
    family = {
        "1.0": {"narrow_doorway": "infeasible", "crossing": "feasible"},
        "0.8": {"narrow_doorway": "feasible", "crossing": "feasible"},
        "0.5": {"narrow_doorway": "feasible", "crossing": "feasible"},
    }
    transitions = compute_family_transitions(
        _sweep_summary(_stable_tables(), family=family),
        baseline_radius=_BASELINE,
        radii=_RADII,
    )
    by_name = {t.family: t for t in transitions}
    doorway = by_name["narrow_doorway"]
    assert doorway.is_narrow_doorway is True
    assert doorway.status_by_radius[1.0] == "infeasible"
    assert doorway.status_by_radius[0.5] == "feasible"
    assert doorway.changed_vs_baseline[0.5] is True
    assert by_name["crossing"].changed_vs_baseline[0.5] is False
    assert by_name["crossing"].is_narrow_doorway is False


# --- verdict decision ------------------------------------------------------


def _stability(metric: str, *, identifiable: bool, flipping: tuple[float, ...] = ()):
    from robot_sf.benchmark.radius_rank_stability import MetricRankStability

    return MetricRankStability(
        metric=metric,
        higher_is_better=True,
        baseline_ranking=["a", "b"],
        baseline_identifiable=identifiable,
        baseline_identifiability_reason=None,
        rankings_by_radius={},
        kendall_tau_by_radius={},
        rank_flips_by_radius={},
        top1_changed_by_radius={},
        identifiable=identifiable,
        identifiability_reason=None if identifiable else "primary_metric_zero_variance",
        flipping_radii=flipping,
    )


def test_verdict_blocked_when_sweep_unavailable() -> None:
    """No sweep yields the pre-analysis gate status, not a scientific verdict."""
    decision = decide_radius_verdict(sweep_available=False, missingness=None, metric_stability=[])
    assert decision.verdict == ANALYSIS_BLOCKED_PENDING_GATE2
    assert decision.is_scientific_verdict is False
    assert decision.interpretation_promoted is False


def test_verdict_invalid_when_missingness_incomplete() -> None:
    """Incomplete accounting forces the invalid-evidence verdict and stops interpretation."""
    accounting = _full_accounting()
    accounting["0.5"] = _accounting(12, 10, {"degraded": 2})
    ledger = build_missingness_ledger(_sweep_summary(_stable_tables(), accounting))
    decision = decide_radius_verdict(
        sweep_available=True,
        missingness=ledger,
        metric_stability=[_stability("success", identifiable=True)],
    )
    assert decision.verdict == VERDICT_INVALID
    assert decision.interpretation_promoted is False


def test_verdict_invalid_takes_precedence_over_non_identifiable() -> None:
    """Fail-closed invalid evidence outranks a non-identifiable ranking."""
    accounting = _full_accounting()
    accounting["0.5"] = _accounting(12, 10, {"missing": 2})
    ledger = build_missingness_ledger(_sweep_summary(_stable_tables(), accounting))
    decision = decide_radius_verdict(
        sweep_available=True,
        missingness=ledger,
        metric_stability=[_stability("success", identifiable=False)],
    )
    assert decision.verdict == VERDICT_INVALID


def test_verdict_non_identifiable_outranks_flip() -> None:
    """A non-identifiable metric outranks a ranking flip on another metric."""
    ledger = build_missingness_ledger(_sweep_summary(_stable_tables()))
    decision = decide_radius_verdict(
        sweep_available=True,
        missingness=ledger,
        metric_stability=[
            _stability("success", identifiable=False),
            _stability("snqi", identifiable=True, flipping=(0.5,)),
        ],
    )
    assert decision.verdict == VERDICT_NON_IDENTIFIABLE
    assert decision.interpretation_promoted is False


def test_verdict_radius_dependent_on_flip() -> None:
    """Any ranking flip versus baseline yields radius_dependent (a valid boundary)."""
    ledger = build_missingness_ledger(_sweep_summary(_stable_tables()))
    decision = decide_radius_verdict(
        sweep_available=True,
        missingness=ledger,
        metric_stability=[_stability("success", identifiable=True, flipping=(0.5,))],
    )
    assert decision.verdict == VERDICT_RADIUS_DEPENDENT
    assert decision.interpretation_promoted is True
    assert any("rank_flip" in reason for reason in decision.reasons)


def test_verdict_stable_when_no_flips() -> None:
    """Identifiable, flip-free rankings across radii yield stable_within_tested_radii."""
    ledger = build_missingness_ledger(_sweep_summary(_stable_tables()))
    decision = decide_radius_verdict(
        sweep_available=True,
        missingness=ledger,
        metric_stability=[_stability("success", identifiable=True)],
    )
    assert decision.verdict == VERDICT_STABLE
    assert decision.interpretation_promoted is True


# --- top-level report ------------------------------------------------------


def test_analyze_radius_sensitivity_blocked_on_none() -> None:
    """A missing sweep summary produces the blocked status and no scientific verdict."""
    report = analyze_radius_sensitivity(None)
    assert report.verdict.verdict == ANALYSIS_BLOCKED_PENDING_GATE2
    assert report.sweep_available is False
    assert report.missingness is None
    assert report.metric_stability == ()


def test_analyze_radius_sensitivity_existing_empty_summary_is_invalid() -> None:
    """An existing but incomplete summary is invalid evidence, not pending Gate 2."""
    report = analyze_radius_sensitivity({})
    assert report.verdict.verdict == VERDICT_INVALID
    assert report.verdict.interpretation_promoted is False


def test_analyze_radius_sensitivity_report_schema() -> None:
    """A complete sweep produces the versioned schema and claim-boundary phrases."""
    report = analyze_radius_sensitivity(_sweep_summary(_stable_tables()))
    payload = report.to_dict()
    assert payload["schema_version"] == RADIUS_RANK_STABILITY_SCHEMA
    assert payload["baseline_radius_m"] == 1.0
    assert payload["radii_m"] == [0.5, 0.8, 1.0]
    assert payload["scenario_cell_count"] == 48
    assert payload["seed_roster"] == list(range(111, 141))
    for phrase in REQUIRED_CLAIM_BOUNDARY_PHRASES:
        assert phrase in payload["claim_boundary"]
    assert report.verdict.verdict == VERDICT_STABLE


def test_analyze_radius_sensitivity_end_to_end_verdicts() -> None:
    """The orchestrator selects the expected verdict for each synthetic sweep."""
    assert (
        analyze_radius_sensitivity(_sweep_summary(_stable_tables())).verdict.verdict
        == VERDICT_STABLE
    )
    assert (
        analyze_radius_sensitivity(_sweep_summary(_flip_tables())).verdict.verdict
        == VERDICT_RADIUS_DEPENDENT
    )


def test_analyze_radius_sensitivity_missing_metric_arm_is_invalid() -> None:
    """Missing metric coverage is invalid evidence, not an apparently stable result."""
    tables = _stable_tables()
    del tables["0.5"]
    report = analyze_radius_sensitivity(_sweep_summary(tables))
    assert report.verdict.verdict == VERDICT_INVALID
    assert report.verdict.interpretation_promoted is False


def test_analyze_radius_sensitivity_single_radius_is_invalid() -> None:
    """A partial radius declaration cannot produce a Gate 3 verdict."""
    summary = _sweep_summary(_stable_tables())
    summary["radii_m"] = [1.0]
    report = analyze_radius_sensitivity(summary)
    assert report.verdict.verdict == VERDICT_INVALID
    assert report.verdict.interpretation_promoted is False


def test_analyze_radius_sensitivity_requires_matched_family_feasibility() -> None:
    """Omitted narrow-doorway feasibility evidence cannot promote a nominal verdict."""
    summary = _sweep_summary(_stable_tables())
    del summary["family_feasibility"]["0.5"]["narrow_doorway"]
    report = analyze_radius_sensitivity(summary)
    assert report.verdict.verdict == VERDICT_INVALID
    assert report.verdict.interpretation_promoted is False
    assert "radius_0.5_missing_narrow_doorway_feasibility" in report.verdict.reasons


def test_analyze_radius_sensitivity_requires_seed_keyed_paired_observations() -> None:
    """An omitted arm/planner/metric pairing cannot promote a nominal verdict."""
    summary = _sweep_summary(_stable_tables())
    del summary["paired_observations"]["0.8"]["orca"]["success"]
    report = analyze_radius_sensitivity(summary)
    assert report.verdict.verdict == VERDICT_INVALID
    assert report.verdict.interpretation_promoted is False
    assert "radius_0.8_planner_orca_paired_metric_mismatch" in report.verdict.reasons


def test_analyze_radius_sensitivity_rejects_missing_or_mixed_campaign_commits() -> None:
    """Gate 3 requires one immutable Gate 2 campaign commit across all radius arms."""
    missing = _sweep_summary(_stable_tables())
    del missing["campaign_provenance"]["0.5"]["campaign_commit"]
    missing_report = analyze_radius_sensitivity(missing)
    assert missing_report.verdict.verdict == VERDICT_INVALID
    assert "radius_0.5_invalid_campaign_commit" in missing_report.verdict.reasons

    mixed = _sweep_summary(_stable_tables())
    mixed["campaign_provenance"]["0.5"]["campaign_commit"] = "e" * 40
    mixed_report = analyze_radius_sensitivity(mixed)
    assert mixed_report.verdict.verdict == VERDICT_INVALID
    assert "mixed_campaign_provenance" in mixed_report.verdict.reasons


def test_analyze_radius_sensitivity_rejects_mismatched_config_or_canary_receipt() -> None:
    """All arm provenance must bind the same campaign config and Gate 1 receipt."""
    summary = _sweep_summary(_stable_tables())
    summary["campaign_provenance"]["0.8"]["config_sha256"] = "a" * 64
    summary["campaign_provenance"]["0.5"]["gate1_canary_receipt_sha256"] = "b" * 64
    report = analyze_radius_sensitivity(summary)
    assert report.verdict.verdict == VERDICT_INVALID
    assert "mixed_campaign_provenance" in report.verdict.reasons


# --- evidence tier ---------------------------------------------------------


def test_evidence_tier_for_verdict() -> None:
    """Only complete identifiable verdicts are nominal benchmark evidence."""
    assert evidence_tier_for_verdict(VERDICT_STABLE) == "nominal_benchmark_radius_sensitivity"
    assert (
        evidence_tier_for_verdict(VERDICT_RADIUS_DEPENDENT)
        == "nominal_benchmark_radius_sensitivity"
    )
    assert evidence_tier_for_verdict(VERDICT_NON_IDENTIFIABLE) == "diagnostic-only"
    assert evidence_tier_for_verdict(VERDICT_INVALID) == "diagnostic-only"
    assert evidence_tier_for_verdict(ANALYSIS_BLOCKED_PENDING_GATE2) == "diagnostic-only"


# --- durable evidence bundle ----------------------------------------------


def test_write_evidence_bundle_writes_checksummed_files(tmp_path: Path) -> None:
    """The bundle writes five files and records matching output checksums."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text("radius: 1.0\n", encoding="utf-8")
    canary_receipt_path = tmp_path / "canary.json"
    _write_gate1_receipt(canary_receipt_path)
    input_path = tmp_path / "sweep.json"
    summary = _bind_summary_evidence(
        _sweep_summary(_stable_tables()), config_path, canary_receipt_path
    )
    input_path.write_text(json.dumps(summary), encoding="utf-8")
    report = analyze_radius_sensitivity(summary)
    provenance = build_evidence_provenance(
        report,
        config_path=str(config_path),
        command="uv run python scripts/benchmark/analyze_radius_rank_stability_issue_6643.py",
        campaign_commit="c" * 40,
        analysis_commit="a" * 40,
        input_paths={
            "sweep_summary.json": input_path,
            "gate1_canary_receipt.json": canary_receipt_path,
        },
        sweep_summary=summary,
    )
    written = write_evidence_bundle(report, provenance, tmp_path)
    assert set(written) == {
        "result.json",
        "report.md",
        "claim_decision.md",
        "analysis_provenance.json",
        "README.md",
    }
    for path in written.values():
        assert path.is_file()

    provenance_payload = json.loads(written["analysis_provenance.json"].read_text())
    assert provenance_payload["schema_version"] == RADIUS_EVIDENCE_BUNDLE_SCHEMA
    assert provenance_payload["evidence_status"] == "nominal_benchmark_radius_sensitivity"
    assert provenance_payload["provenance"]["campaign_commit"] == "c" * 40
    assert provenance_payload["provenance"]["config_sha256"]
    assert provenance_payload["provenance"]["input_sha256"]["sweep_summary.json"]

    import hashlib

    for filename in ("result.json", "report.md", "claim_decision.md"):
        digest = hashlib.sha256(written[filename].read_bytes()).hexdigest()
        assert provenance_payload["output_sha256"][filename] == digest

    result_payload = json.loads(written["result.json"].read_text())
    assert result_payload["schema_version"] == RADIUS_RANK_STABILITY_SCHEMA
    assert result_payload["verdict"]["verdict"] == VERDICT_STABLE


def test_write_evidence_bundle_blocked_is_diagnostic(tmp_path: Path) -> None:
    """A blocked bundle records diagnostic-only tier and does not promote a claim."""
    report = analyze_radius_sensitivity(None)
    provenance = build_evidence_provenance(
        report,
        config_path="configs/benchmarks/radius_sensitivity_v1.yaml",
        command="cmd",
        campaign_commit="c" * 40,
        analysis_commit="a" * 40,
    )
    written = write_evidence_bundle(report, provenance, tmp_path)
    provenance_payload = json.loads(written["analysis_provenance.json"].read_text())
    assert provenance_payload["evidence_status"] == "diagnostic-only"
    assert provenance_payload["verdict"]["interpretation_promoted"] is False


def test_write_promoted_bundle_requires_input_checksums(tmp_path: Path) -> None:
    """A nominal bundle cannot omit its config or Gate 2 input checksum."""
    summary = _sweep_summary(_stable_tables())
    report = analyze_radius_sensitivity(summary)
    provenance = build_evidence_provenance(
        report,
        config_path=str(tmp_path / "missing.yaml"),
        command="cmd",
        campaign_commit="c" * 40,
        analysis_commit="a" * 40,
    )
    with pytest.raises(ValueError, match="config"):
        write_evidence_bundle(report, provenance, tmp_path / "missing-config-bundle")

    config_path = tmp_path / "config.yaml"
    config_path.write_text("radius: 1.0\n", encoding="utf-8")
    canary_receipt_path = tmp_path / "canary.json"
    _write_gate1_receipt(canary_receipt_path)
    summary = _bind_summary_evidence(
        _sweep_summary(_stable_tables()), config_path, canary_receipt_path
    )
    report = analyze_radius_sensitivity(summary)
    provenance = build_evidence_provenance(
        report,
        config_path=str(config_path),
        command="cmd",
        campaign_commit="c" * 40,
        analysis_commit="a" * 40,
        sweep_summary=summary,
    )
    with pytest.raises(ValueError, match="sweep summary"):
        write_evidence_bundle(report, provenance, tmp_path / "missing-input-bundle")


def test_write_promoted_bundle_requires_bound_campaign_config_and_canary(tmp_path: Path) -> None:
    """Caller-provided provenance cannot override the Gate 2 summary binding."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text("radius: 1.0\n", encoding="utf-8")
    canary_receipt_path = tmp_path / "canary.json"
    _write_gate1_receipt(canary_receipt_path)
    summary = _bind_summary_evidence(
        _sweep_summary(_stable_tables()), config_path, canary_receipt_path
    )
    input_path = tmp_path / "sweep.json"
    input_path.write_text(json.dumps(summary), encoding="utf-8")
    report = analyze_radius_sensitivity(summary)
    provenance = build_evidence_provenance(
        report,
        config_path=str(config_path),
        command="cmd",
        campaign_commit="e" * 40,
        input_paths={
            "sweep_summary.json": input_path,
            "gate1_canary_receipt.json": canary_receipt_path,
        },
        sweep_summary=summary,
    )
    with pytest.raises(ValueError, match="campaign/config/canary"):
        write_evidence_bundle(report, provenance, tmp_path / "mismatched-provenance-bundle")


def test_write_promoted_bundle_rejects_failed_gate1_receipt(tmp_path: Path) -> None:
    """A checksum-matching failed canary cannot promote nominal evidence."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text("radius: 1.0\n", encoding="utf-8")
    canary_receipt_path = tmp_path / "canary.json"
    _write_gate1_receipt(canary_receipt_path, go=False)
    summary = _bind_summary_evidence(
        _sweep_summary(_stable_tables()), config_path, canary_receipt_path
    )
    input_path = tmp_path / "sweep.json"
    input_path.write_text(json.dumps(summary), encoding="utf-8")
    report = analyze_radius_sensitivity(summary)
    provenance = build_evidence_provenance(
        report,
        config_path=str(config_path),
        command="cmd",
        campaign_commit="c" * 40,
        input_paths={
            "sweep_summary.json": input_path,
            "gate1_canary_receipt.json": canary_receipt_path,
        },
        sweep_summary=summary,
    )
    with pytest.raises(ValueError, match="passing Gate 1 canary"):
        write_evidence_bundle(report, provenance, tmp_path / "failed-canary-bundle")


# --- verdict propagation ---------------------------------------------------


def test_verdict_comment_carries_claim_boundary() -> None:
    """The #6600 verdict comment names the verdict and the claim boundary."""
    report = analyze_radius_sensitivity(_sweep_summary(_flip_tables()))
    comment = render_verdict_comment(report)
    assert VERDICT_RADIUS_DEPENDENT in comment
    assert "within-simulator radius sensitivity only" in comment
    assert "not sim-to-real evidence" in comment
    assert "Manuscript admission is a separate author step" in comment


def test_blocked_verdict_comment_explains_gate() -> None:
    """The blocked comment explains the pre-analysis gate, not a scientific verdict."""
    report = analyze_radius_sensitivity(None)
    comment = render_verdict_comment(report)
    assert ANALYSIS_BLOCKED_PENDING_GATE2 in comment
    assert "#6642" in comment


def test_propagation_comment_references_parent() -> None:
    """The #3207 propagation comment carries the verdict and claim boundary."""
    report = analyze_radius_sensitivity(_sweep_summary(_stable_tables()))
    comment = render_propagation_comment(report)
    assert "#6600" in comment
    assert VERDICT_STABLE in comment
    assert "not a safety guarantee" in comment


def test_blocked_propagation_comment_does_not_propagate() -> None:
    """A pending Gate 2 status must not be rendered as a #3207 result."""
    comment = render_propagation_comment(analyze_radius_sensitivity(None))
    assert "No radius-axis result is available" in comment
    assert "No validity-boundary" in comment
    assert "recorded verdict" not in comment


# --- sweep summary loading and gate ---------------------------------------


def test_sweep_summary_available(tmp_path: Path) -> None:
    """Availability reflects whether the summary file exists."""
    assert sweep_summary_available(None) is False
    missing = tmp_path / "missing.json"
    assert sweep_summary_available(missing) is False
    missing.write_text("{}", encoding="utf-8")
    assert sweep_summary_available(missing) is True


def test_load_sweep_summary_fails_closed(tmp_path: Path) -> None:
    """Missing, malformed, and non-object summaries all fail closed."""
    with pytest.raises(FileNotFoundError):
        load_sweep_summary(tmp_path / "absent.json")

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("not-json", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_sweep_summary(bad_json)

    non_object = tmp_path / "list.json"
    non_object.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a JSON object"):
        load_sweep_summary(non_object)


# --- CLI -------------------------------------------------------------------


def test_cli_blocked_exits_nonzero(tmp_path: Path) -> None:
    """Omitting the sweep summary fails closed with the blocked exit code."""
    cli = _load_cli()
    exit_code = cli.main(["--output-dir", str(tmp_path / "bundle"), "--json"])
    assert exit_code == cli.EXIT_BLOCKED_PENDING_GATE2
    assert (tmp_path / "bundle" / "result.json").is_file()


def test_cli_verdict_exits_zero(tmp_path: Path) -> None:
    """A complete sweep summary produces a scientific verdict and exit code zero."""
    cli = _load_cli()
    config_path = tmp_path / "config.yaml"
    config_path.write_text("radius: 1.0\n", encoding="utf-8")
    canary_receipt_path = tmp_path / "canary.json"
    _write_gate1_receipt(canary_receipt_path)
    summary_path = tmp_path / "sweep.json"
    summary_path.write_text(
        json.dumps(
            _bind_summary_evidence(_sweep_summary(_flip_tables()), config_path, canary_receipt_path)
        ),
        encoding="utf-8",
    )
    exit_code = cli.main(
        [
            "--sweep-summary",
            str(summary_path),
            "--output-dir",
            str(tmp_path / "bundle"),
            "--config",
            str(config_path),
            "--campaign-commit",
            "c" * 40,
            "--gate1-canary-receipt",
            str(canary_receipt_path),
        ]
    )
    assert exit_code == cli.EXIT_VERDICT_PRODUCED
    payload = json.loads((tmp_path / "bundle" / "result.json").read_text())
    assert payload["verdict"]["verdict"] == VERDICT_RADIUS_DEPENDENT


def test_cli_malformed_summary_exits_error(tmp_path: Path) -> None:
    """A malformed sweep summary is an unexpected error, not a verdict."""
    cli = _load_cli()
    summary_path = tmp_path / "sweep.json"
    summary_path.write_text("not-json", encoding="utf-8")
    exit_code = cli.main(
        ["--sweep-summary", str(summary_path), "--output-dir", str(tmp_path / "bundle")]
    )
    assert exit_code == cli.EXIT_UNEXPECTED_ERROR
