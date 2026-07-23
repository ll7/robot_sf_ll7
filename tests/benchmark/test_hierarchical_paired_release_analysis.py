"""Focused tests for issue #5351's hierarchical paired release analysis engine.

These tests prove the statistical estimators compute the intended values on
deterministic synthetic matched cells.  They are unit-level evidence for the
analysis contract, not benchmark evidence: the real successor release rows
(from #4364) are still absent, so every report keeps the claim gate blocked.
"""

# evidence-writer-exempt: these tests build synthetic EpisodeEventLedger.v2 rows in
# memory to exercise the analysis engine; they do not generate a repository evidence artifact.

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from robot_sf.benchmark.event_ledger import EPISODE_EVENT_LEDGER_SCHEMA_VERSION
from robot_sf.benchmark.hierarchical_paired_release_analysis import (
    CLAIM_GATE_BLOCKED_ANALYSIS_NOT_RUN,
    CLAIM_GATE_BLOCKED_REVIEW_PENDING,
    EVIDENCE_STATUS_NOT_BENCHMARK_EVIDENCE,
    HIERARCHICAL_PAIRED_RELEASE_ANALYSIS_REPORT_SCHEMA,
    AnalysisPolicy,
    HierarchicalPairedReleaseAnalysisError,
    _paired_mcnemar_p_value,
    build_matched_cells_from_ledger_rows,
    censored_completion_time,
    estimate_paired_effects,
    fail_closed_analysis_from_manifest,
    holm_multiplicity,
    normalized_near_miss_exposure,
    practical_effect_classification,
    run_hierarchical_paired_release_analysis,
)
from robot_sf.benchmark.hierarchical_paired_release_inputs import (
    load_hierarchical_paired_release_input_manifest,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MANIFEST_PATH = (
    _REPO_ROOT / "configs/benchmarks/releases/hierarchical_paired_release_analysis_issue_5351.yaml"
)


def _ledger_row(
    *,
    scenario_id: str,
    seed: int,
    planner: str,
    collision: bool = False,
    near_miss: bool = False,
    timeout: bool = False,
    completion_time: float = 0.0,
    exposure: float = 1.0,
) -> dict[str, Any]:
    """Build a minimal ``EpisodeEventLedger.v2`` row for testing."""

    return {
        "schema_version": EPISODE_EVENT_LEDGER_SCHEMA_VERSION,
        "scenario_id": scenario_id,
        "seed": seed,
        "planner": planner,
        "exact_events": {
            "collision": collision,
            "goal_reached": not collision and not timeout,
            "timeout": timeout,
            "invalid_run": False,
        },
        "surrogate_events": {"near_miss": near_miss},
        "provenance": {
            "completion_time": completion_time,
            "exposure": {"time": exposure, "distance": exposure, "opportunity": exposure},
        },
    }


def _two_arm_rows() -> list[dict[str, Any]]:
    """Eight paired scenario-seed cells where planner A collides on all four."""

    rows: list[dict[str, Any]] = []
    for index in range(4):
        scenario = f"scn-{index}"
        rows.append(
            _ledger_row(
                scenario_id=scenario,
                seed=100 + index,
                planner="alpha",
                collision=True,
                near_miss=True,
                completion_time=10.0,
                exposure=2.0,
            )
        )
        rows.append(
            _ledger_row(
                scenario_id=scenario,
                seed=100 + index,
                planner="beta",
                collision=False,
                near_miss=False,
                completion_time=5.0,
                exposure=1.0,
            )
        )
    return rows


def _manifest() -> dict[str, Any]:
    """Load a fresh copy of the checked-in blocked manifest."""

    return load_hierarchical_paired_release_input_manifest(_MANIFEST_PATH)


def _ready_manifest(tmp_path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Create a manifest whose durable synthetic successor rows pass the input gate."""

    rows_path = tmp_path / "docs/context/evidence/release_successor/rows.jsonl"
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    serialized_rows = "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n"
    rows_path.write_text(serialized_rows, encoding="utf-8")
    manifest = _manifest()
    successor_release = manifest["successor_release"]
    assert isinstance(successor_release, dict)
    successor_release.update(
        {
            "release_tag": "v2.0.0",
            "commit": "a" * 40,
            "typed_ledger_rows": rows_path.relative_to(tmp_path).as_posix(),
            "typed_ledger_rows_sha256": hashlib.sha256(rows_path.read_bytes()).hexdigest(),
        }
    )
    return manifest


def test_build_matched_cells_pairs_ledger_rows_and_preserves_outcomes() -> None:
    """Rows with matching scenario/seed across arms pair into cells with copied fields."""

    cells = build_matched_cells_from_ledger_rows(_two_arm_rows(), planner_pair=("alpha", "beta"))

    assert len(cells) == 4
    assert {cell.scenario_id for cell in cells} == {"scn-0", "scn-1", "scn-2", "scn-3"}
    assert all(cell.planner_a == "alpha" and cell.planner_b == "beta" for cell in cells)
    assert [cell.collision_a for cell in cells] == [1, 1, 1, 1]
    assert [cell.collision_b for cell in cells] == [0, 0, 0, 0]
    assert [cell.exposure_a["time"] for cell in cells] == [2.0, 2.0, 2.0, 2.0]
    assert [cell.exposure_b["opportunity"] for cell in cells] == [1.0, 1.0, 1.0, 1.0]


def test_build_matched_cells_rejects_unmatched_or_duplicate_rows() -> None:
    """A partial release cannot silently shrink the denominator."""

    rows = _two_arm_rows()
    # Drop one beta row -> alpha arm has an unpaired cell.
    unpaired = [
        row for row in rows if not (row["scenario_id"] == "scn-0" and row["planner"] == "beta")
    ]
    with pytest.raises(HierarchicalPairedReleaseAnalysisError, match="unmatched successor rows"):
        build_matched_cells_from_ledger_rows(unpaired, planner_pair=("alpha", "beta"))

    duplicate = _two_arm_rows() + [_ledger_row(scenario_id="scn-0", seed=100, planner="alpha")]
    with pytest.raises(HierarchicalPairedReleaseAnalysisError, match="duplicate ledger row"):
        build_matched_cells_from_ledger_rows(duplicate, planner_pair=("alpha", "beta"))


def test_build_matched_cells_rejects_non_ledger_rows() -> None:
    """Rows that are not EpisodeEventLedger.v2 mappings fail closed."""

    with pytest.raises(HierarchicalPairedReleaseAnalysisError, match="schema_version"):
        build_matched_cells_from_ledger_rows(
            [{"scenario_id": "x", "seed": 1, "planner": "alpha"}],
            planner_pair=("alpha", "beta"),
        )
    with pytest.raises(HierarchicalPairedReleaseAnalysisError, match="empty successor row set"):
        build_matched_cells_from_ledger_rows([], planner_pair=("alpha", "beta"))


def test_build_matched_cells_requires_v2_well_formed_event_blocks() -> None:
    """Legacy and malformed event blocks cannot turn into false zero outcomes."""

    legacy = _ledger_row(scenario_id="x", seed=1, planner="alpha")
    legacy["schema_version"] = "EpisodeEventLedger.v1"
    with pytest.raises(HierarchicalPairedReleaseAnalysisError, match="EpisodeEventLedger.v2"):
        build_matched_cells_from_ledger_rows([legacy], planner_pair=("alpha", "beta"))

    malformed = _ledger_row(scenario_id="x", seed=1, planner="alpha")
    malformed["exact_events"] = "not-a-mapping"
    with pytest.raises(
        HierarchicalPairedReleaseAnalysisError, match="exact_events must be a mapping"
    ):
        build_matched_cells_from_ledger_rows([malformed], planner_pair=("alpha", "beta"))

    missing_near_miss = _ledger_row(scenario_id="x", seed=1, planner="alpha")
    missing_near_miss["surrogate_events"] = {}
    with pytest.raises(HierarchicalPairedReleaseAnalysisError, match="near_miss must be a bool"):
        build_matched_cells_from_ledger_rows([missing_near_miss], planner_pair=("alpha", "beta"))


def test_estimate_paired_effects_reports_exact_risk_difference_and_sign() -> None:
    """A constant 4/4 vs 0/4 split yields risk difference +1.0 favouring beta."""

    cells = build_matched_cells_from_ledger_rows(_two_arm_rows(), planner_pair=("alpha", "beta"))
    policy = AnalysisPolicy(bootstrap_samples=200, bootstrap_seed=7)
    effect = estimate_paired_effects(cells, outcome="collision", policy=policy)

    assert effect.n_cells == 4
    assert effect.risk_a == 1.0
    assert effect.risk_b == 0.0
    assert effect.risk_difference == pytest.approx(1.0)
    assert effect.odds_ratio == pytest.approx(1e6)
    # A constant positive shift across every cell -> the lower bound clears zero.
    assert effect.risk_difference_ci_low > 0.0


def test_estimate_paired_effects_identical_arms_yield_zero_difference() -> None:
    """Identical arms produce a risk difference centred on zero."""

    rows: list[dict[str, Any]] = []
    for index in range(6):
        scenario = f"scn-{index}"
        rows.append(_ledger_row(scenario_id=scenario, seed=index, planner="alpha", collision=True))
        rows.append(_ledger_row(scenario_id=scenario, seed=index, planner="beta", collision=True))
    cells = build_matched_cells_from_ledger_rows(rows, planner_pair=("alpha", "beta"))
    policy = AnalysisPolicy(bootstrap_samples=400, bootstrap_seed=3)
    effect = estimate_paired_effects(cells, outcome="collision", policy=policy)

    assert effect.risk_difference == pytest.approx(0.0)
    assert effect.odds_ratio == pytest.approx(1.0)
    # The interval should straddle zero when there is no signal.
    assert effect.risk_difference_ci_low <= 0.0 <= effect.risk_difference_ci_high


def test_estimate_paired_effects_uses_consistent_zero_risk_odds_ratio() -> None:
    """All-zero point and bootstrap odds ratios use the same neutral convention."""

    rows = [
        _ledger_row(scenario_id=f"scn-{index}", seed=index, planner=planner)
        for index in range(4)
        for planner in ("alpha", "beta")
    ]
    cells = build_matched_cells_from_ledger_rows(rows, planner_pair=("alpha", "beta"))
    effect = estimate_paired_effects(
        cells, outcome="collision", policy=AnalysisPolicy(bootstrap_samples=100, bootstrap_seed=8)
    )

    assert effect.odds_ratio == pytest.approx(1.0)
    assert effect.odds_ratio_ci_low == pytest.approx(1.0)
    assert effect.odds_ratio_ci_high == pytest.approx(1.0)


def test_exact_mcnemar_avoids_underflow_for_large_balanced_discordance() -> None:
    """A large balanced discordant set remains non-significant rather than underflowing."""

    rows: list[dict[str, Any]] = []
    for index in range(1200):
        alpha_collision = index < 600
        rows.extend(
            (
                _ledger_row(
                    scenario_id=f"scn-{index}",
                    seed=index,
                    planner="alpha",
                    collision=alpha_collision,
                ),
                _ledger_row(
                    scenario_id=f"scn-{index}",
                    seed=index,
                    planner="beta",
                    collision=not alpha_collision,
                ),
            )
        )
    cells = build_matched_cells_from_ledger_rows(rows, planner_pair=("alpha", "beta"))

    assert _paired_mcnemar_p_value(cells, outcome="collision") == pytest.approx(1.0)


def test_holm_multiplicity_monotonizes_adjusted_p_values_and_rejects_small() -> None:
    """Holm step-down keeps adjusted p-values non-decreasing by rank."""

    decisions = holm_multiplicity([0.01, 0.04, 0.20], alpha=0.05)

    assert [d.raw_p_value for d in decisions] == [0.01, 0.04, 0.20]
    adjusted = [d.adjusted_p_value for d in decisions]
    assert adjusted == sorted(adjusted)
    # Smallest raw p-value: 0.01 * 3 = 0.03 <= 0.05 -> rejected.
    assert decisions[0].adjusted_p_value == pytest.approx(0.03)
    assert decisions[0].rejected is True
    # 0.20 * 1 = 0.20 -> not rejected.
    assert decisions[2].rejected is False


def test_holm_multiplicity_rejects_invalid_alpha_and_empty_input() -> None:
    """The correction fails closed on out-of-range alpha."""

    with pytest.raises(HierarchicalPairedReleaseAnalysisError, match="alpha"):
        holm_multiplicity([0.01], alpha=0.0)
    with pytest.raises(HierarchicalPairedReleaseAnalysisError, match="alpha"):
        holm_multiplicity([0.01], alpha=1.0)
    assert holm_multiplicity([]) == []


def test_practical_effect_classification_distinguishes_practical_and_statistical() -> None:
    """A large shift clears the threshold; a tiny shift is practically null."""

    cells = build_matched_cells_from_ledger_rows(_two_arm_rows(), planner_pair=("alpha", "beta"))
    policy = AnalysisPolicy(bootstrap_samples=500, bootstrap_seed=11, min_risk_difference=0.02)
    strong = estimate_paired_effects(cells, outcome="collision", policy=policy)
    strong_record = practical_effect_classification(strong, policy=policy)
    assert strong_record["verdict"] == "practically_separable"
    assert strong_record["practically_separable"] is True

    # Build a near-null paired effect by hand: tiny positive difference, tiny CI.
    from robot_sf.benchmark.hierarchical_paired_release_analysis import PairedEffect

    tiny = PairedEffect(
        comparison="alpha:beta:collision",
        n_cells=100,
        risk_a=0.501,
        risk_b=0.500,
        risk_difference=0.001,
        risk_difference_ci_low=0.0005,
        risk_difference_ci_high=0.0015,
        odds_a=0.501 / 0.499,
        odds_b=1.0,
        odds_ratio=0.501 / 0.499,
        odds_ratio_ci_low=0.9,
        odds_ratio_ci_high=1.1,
    )
    tiny_record = practical_effect_classification(tiny, policy=policy)
    assert tiny_record["verdict"] == "statistically_separable_practically_null"
    assert tiny_record["statistically_separable"] is True
    assert tiny_record["practically_separable"] is False


def test_censored_completion_time_treats_timeouts_as_censored() -> None:
    """A timed-out arm is flagged censored and clamped to the horizon."""

    rows: list[dict[str, Any]] = []
    for index in range(4):
        scenario = f"scn-{index}"
        rows.append(
            _ledger_row(
                scenario_id=scenario,
                seed=index,
                planner="alpha",
                timeout=index % 2 == 0,
                completion_time=30.0 if index % 2 == 0 else 8.0,
            )
        )
        rows.append(
            _ledger_row(scenario_id=scenario, seed=index, planner="beta", completion_time=6.0)
        )
    cells = build_matched_cells_from_ledger_rows(rows, planner_pair=("alpha", "beta"))
    summaries = censored_completion_time(cells, horizon=20.0)

    by_planner = {s.planner: s for s in summaries}
    assert by_planner["alpha"].n_censored == 2
    assert by_planner["alpha"].censoring_rate == pytest.approx(0.5)
    assert by_planner["alpha"].n_observed == 2
    assert by_planner["alpha"].mean_observed == pytest.approx(8.0)
    assert by_planner["alpha"].median_observed == pytest.approx(8.0)
    assert by_planner["beta"].n_censored == 0
    assert by_planner["beta"].mean_observed == pytest.approx(6.0)


def test_normalized_near_miss_exposure_normalizes_by_exposure_window() -> None:
    """Near-miss counts are divided by total exposure, not summed raw."""

    cells = build_matched_cells_from_ledger_rows(_two_arm_rows(), planner_pair=("alpha", "beta"))
    summaries = normalized_near_miss_exposure(
        cells, policy=AnalysisPolicy(exposure_opportunity=1.0)
    )
    by_planner = {s.planner: s for s in summaries}
    # alpha: 4 near-miss over 4*2.0=8.0 exposure -> 0.5 per unit opportunity.
    assert by_planner["alpha"].total_near_miss == 4
    assert by_planner["alpha"].total_exposure == pytest.approx(8.0)
    assert by_planner["alpha"].normalized_rate == pytest.approx(0.5)
    # beta: 0 near-miss.
    assert by_planner["beta"].normalized_rate == pytest.approx(0.0)


def test_exposure_and_completion_time_fallbacks_fail_closed_or_read_metric_values() -> None:
    """Exposure is required while row-level metric values remain a valid time source."""

    rows = _two_arm_rows()
    rows[0]["provenance"] = {"exposure": {"time": 2.0, "distance": 2.0, "opportunity": 2.0}}
    rows[0]["metrics"] = {"completion_time": {"value": 7.5}}
    cells = build_matched_cells_from_ledger_rows(rows, planner_pair=("alpha", "beta"))
    assert cells[0].completion_time_a == pytest.approx(7.5)

    invalid_exposure = _two_arm_rows()
    invalid_exposure[0]["provenance"]["exposure"] = 0.0
    with pytest.raises(HierarchicalPairedReleaseAnalysisError, match="exposure must be a mapping"):
        build_matched_cells_from_ledger_rows(invalid_exposure, planner_pair=("alpha", "beta"))

    for invalid_time in (None, False, -1.0, float("nan"), float("inf")):
        invalid_completion_time = _two_arm_rows()
        invalid_completion_time[0]["provenance"]["completion_time"] = invalid_time
        with pytest.raises(HierarchicalPairedReleaseAnalysisError, match="completion_time must be"):
            build_matched_cells_from_ledger_rows(
                invalid_completion_time, planner_pair=("alpha", "beta")
            )

    invalid_provenance_with_metric = _two_arm_rows()
    invalid_provenance_with_metric[0]["provenance"]["completion_time"] = float("nan")
    invalid_provenance_with_metric[0]["metrics"] = {"completion_time": {"value": 7.5}}
    with pytest.raises(HierarchicalPairedReleaseAnalysisError, match="completion_time must be"):
        build_matched_cells_from_ledger_rows(
            invalid_provenance_with_metric, planner_pair=("alpha", "beta")
        )


def test_run_analysis_emits_machine_readable_report_with_blocked_claim_gate(tmp_path: Path) -> None:
    """Even on valid rows the claim gate stays blocked pending review."""

    rows = _two_arm_rows()
    manifest = _ready_manifest(tmp_path, rows)
    report = run_hierarchical_paired_release_analysis(
        manifest,
        repo_root=tmp_path,
        successor_rows=rows,
        planner_pairs=[("alpha", "beta")],
        horizon=20.0,
        policy=AnalysisPolicy(bootstrap_samples=200, bootstrap_seed=5),
    )

    assert report["schema_version"] == HIERARCHICAL_PAIRED_RELEASE_ANALYSIS_REPORT_SCHEMA
    assert report["issue"] == 5351
    assert report["evidence_status"] == EVIDENCE_STATUS_NOT_BENCHMARK_EVIDENCE
    assert report["analysis_executed"] is True
    assert report["claim_gate"]["status"] == CLAIM_GATE_BLOCKED_REVIEW_PENDING
    assert report["semantics"] == {
        "benchmark_metrics_changed": False,
        "analysis_executed": True,
        "claim_promotion": "none",
    }
    assert len(report["paired_effects"]) == 3  # collision, near_miss, timeout
    pair = report["paired_effects"][0]
    assert pair["planner_pair"] == ["alpha", "beta"]
    assert pair["outcome"] == "collision"
    assert pair["risk_difference"] == pytest.approx(1.0)
    assert "holm_adjusted_p_value" in pair
    assert report["multiplicity"]["method"] == "holm_step_down"
    assert report["multiplicity"]["family_size"] == 3
    assert report["censored_completion_time"] is not None
    assert report["normalized_near_miss_exposure"] is not None
    assert len(report["sensitivity_analyses"]) == 3
    sensitivity = report["sensitivity_analyses"][0]
    assert sensitivity["planner_pair"] == ["alpha", "beta"]
    assert sensitivity["seed_level"]["n_clusters"] == 4
    assert sensitivity["family_level"]["n_clusters"] == 4
    conformance = {row["id"]: row["status"] for row in report["protocol_conformance"]}
    assert conformance["paired_effects"] == "delivered_analysis_pending_human_review"
    assert conformance["censored_completion_time"] == "delivered_analysis_pending_human_review"
    assert conformance["normalized_near_miss_exposure"] == (
        "delivered_analysis_pending_human_review"
    )
    assert conformance["sensitivity_analyses"] == "delivered_analysis_pending_human_review"
    # Deterministic across runs given the seeded policy.
    again = run_hierarchical_paired_release_analysis(
        manifest,
        repo_root=tmp_path,
        successor_rows=rows,
        planner_pairs=[("alpha", "beta")],
        horizon=20.0,
        policy=AnalysisPolicy(bootstrap_samples=200, bootstrap_seed=5),
    )
    assert again["paired_effects"] == report["paired_effects"]


def test_run_analysis_rejects_empty_rows_and_missing_pairs(tmp_path: Path) -> None:
    """The runner fail-closes on absent input rather than emitting a fake report."""

    rows = _two_arm_rows()
    with pytest.raises(HierarchicalPairedReleaseAnalysisError, match="input readiness"):
        run_hierarchical_paired_release_analysis(
            _manifest(), repo_root=tmp_path, successor_rows=rows, planner_pairs=[("alpha", "beta")]
        )
    for field in ("release_tag", "commit", "typed_ledger_rows", "typed_ledger_rows_sha256"):
        blocked_manifest = _ready_manifest(tmp_path / field, rows)
        blocked_manifest["successor_release"][field] = None
        with pytest.raises(HierarchicalPairedReleaseAnalysisError, match="input readiness"):
            run_hierarchical_paired_release_analysis(
                blocked_manifest,
                repo_root=tmp_path / field,
                successor_rows=rows,
                planner_pairs=[("alpha", "beta")],
            )
    manifest = _ready_manifest(tmp_path, rows)
    with pytest.raises(HierarchicalPairedReleaseAnalysisError, match="empty successor row set"):
        run_hierarchical_paired_release_analysis(
            manifest, repo_root=tmp_path, successor_rows=[], planner_pairs=[("alpha", "beta")]
        )
    with pytest.raises(HierarchicalPairedReleaseAnalysisError, match="no matched cells"):
        run_hierarchical_paired_release_analysis(
            manifest,
            repo_root=tmp_path,
            successor_rows=[_ledger_row(scenario_id="x", seed=1, planner="gamma")],
            planner_pairs=[("alpha", "beta")],
        )
    with pytest.raises(
        HierarchicalPairedReleaseAnalysisError, match="planner_pairs must not be empty"
    ):
        run_hierarchical_paired_release_analysis(
            manifest, repo_root=tmp_path, successor_rows=rows, planner_pairs=[]
        )
    with pytest.raises(HierarchicalPairedReleaseAnalysisError, match="outcomes must not be empty"):
        run_hierarchical_paired_release_analysis(
            manifest,
            repo_root=tmp_path,
            successor_rows=rows,
            planner_pairs=[("alpha", "beta")],
            outcomes=[],
        )
    with pytest.raises(HierarchicalPairedReleaseAnalysisError, match="bootstrap_samples"):
        run_hierarchical_paired_release_analysis(
            manifest,
            repo_root=tmp_path,
            successor_rows=rows,
            planner_pairs=[("alpha", "beta")],
            policy=AnalysisPolicy(bootstrap_samples=0),
        )


def test_run_analysis_emits_summaries_for_every_planner_pair(tmp_path: Path) -> None:
    """Every declared pair receives its own completion and exposure summaries."""

    rows = _two_arm_rows()
    for row in _two_arm_rows():
        copied = dict(row)
        copied["planner"] = "gamma" if row["planner"] == "alpha" else "delta"
        rows.append(copied)
    report = run_hierarchical_paired_release_analysis(
        _ready_manifest(tmp_path, rows),
        repo_root=tmp_path,
        successor_rows=rows,
        planner_pairs=[("alpha", "beta"), ("gamma", "delta")],
        horizon=20.0,
        policy=AnalysisPolicy(bootstrap_samples=100, bootstrap_seed=4),
    )

    assert len(report["censored_completion_time"]) == 4
    assert len(report["normalized_near_miss_exposure"]) == 12
    assert {summary["dimension"] for summary in report["normalized_near_miss_exposure"]} == {
        "time",
        "distance",
        "opportunity",
    }
    pairs = {tuple(summary["planner_pair"]) for summary in report["censored_completion_time"]}
    assert pairs == {("alpha", "beta"), ("gamma", "delta")}


def test_fail_closed_analysis_from_manifest_reports_blocked_analysis_not_run() -> None:
    """Without successor rows the analysis-side report mirrors the input gate."""

    report = fail_closed_analysis_from_manifest(_manifest(), repo_root=_REPO_ROOT)

    assert report["schema_version"] == HIERARCHICAL_PAIRED_RELEASE_ANALYSIS_REPORT_SCHEMA
    assert report["analysis_executed"] is False
    assert report["evidence_status"] == EVIDENCE_STATUS_NOT_BENCHMARK_EVIDENCE
    assert report["claim_gate"]["status"] == CLAIM_GATE_BLOCKED_ANALYSIS_NOT_RUN
    assert report["paired_effects"] == []
    assert report["sensitivity_analyses"] == []
    conformance = {row["id"]: row["status"] for row in report["protocol_conformance"]}
    assert set(conformance.values()) == {"declared_pending_analysis"}
    assert report["semantics"]["claim_promotion"] == "none"
