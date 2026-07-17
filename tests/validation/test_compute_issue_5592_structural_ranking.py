"""Contract tests for the issue #5592 ``constraints_first_structural_rank`` metric."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.validation import compute_issue_5592_structural_ranking as metric

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKET = REPO_ROOT / "configs/benchmarks/issue_5592_cross_matrix_preregistration.yaml"
ROSTER_SIGNATURE = metric._roster_signature(metric._load_packet(PACKET))

PLANNERS_BY_CLASS = {
    "constraint_first_hybrid": [
        "scenario_adaptive_hybrid_orca_v1",
        "hybrid_rule_v3_fast_progress_static_escape",
    ],
    "learned_policy": ["ppo", "guarded_ppo"],
    "predictive": ["prediction_planner", "prediction_mpc", "prediction_mpc_cbf"],
    "baseline_reactive": ["goal", "social_force", "orca", "socnav_sampling", "sacadrl"],
}


def _rows_with_success(success_by_planner: dict[str, float]) -> list[dict[str, str]]:
    """Build a per-planner episode-aggregate row set from success-rate overrides."""
    known_planners = {planner for planners in PLANNERS_BY_CLASS.values() for planner in planners}
    unknown_planners = set(success_by_planner) - known_planners
    if unknown_planners:
        raise KeyError(f"unknown planner override(s): {sorted(unknown_planners)}")
    rows = []
    for planners in PLANNERS_BY_CLASS.values():
        for planner in planners:
            success = success_by_planner.get(planner, 0.5)
            rows.append(
                {
                    "planner_key": planner,
                    "success_rate": str(success),
                    "collision_event_rate": str(round(1.0 - success, 3)),
                    "near_miss_event_rate": "0.1",
                    "timeout_rate": "0.05",
                    "snqi_mean": str(round(0.4 + success * 0.5, 3)),
                }
            )
    return rows


def _write_rows(tmp_path: Path, name: str, rows: list[dict[str, str]]) -> Path:
    path = tmp_path / name
    with path.open("w", encoding="utf-8", newline="") as handle:
        import csv

        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "planner_key",
                "success_rate",
                "collision_event_rate",
                "near_miss_event_rate",
                "timeout_rate",
                "snqi_mean",
                "execution_mode",
                "availability_status",
            ],
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def test_metric_is_defined_and_exports_name() -> None:
    """The pre-registration names this metric; it now resolves to a real implementation."""
    packet = metric._load_packet(PACKET)
    assert packet["comparison_contract"]["metric"] == "constraints_first_structural_rank"


def test_reference_ranking_is_valid_permutation(tmp_path: Path) -> None:
    """A realistic reference matrix produces a complete 1..4 permutation with roster signature."""
    # constraint-first hybrids are best, then learned, predictive, baseline reactive.
    success = {
        "scenario_adaptive_hybrid_orca_v1": 0.95,
        "hybrid_rule_v3_fast_progress_static_escape": 0.92,
        "ppo": 0.80,
        "guarded_ppo": 0.78,
        "prediction_planner": 0.70,
        "prediction_mpc": 0.72,
        "prediction_mpc_cbf": 0.71,
        "goal": 0.40,
        "social_force": 0.45,
        "orca": 0.50,
        "socnav_sampling": 0.42,
        "sacadrl": 0.48,
    }
    rows = _rows_with_success(success)
    out = tmp_path / "reference_ranking.csv"
    ranking = metric.build_ranking_for_matrix(
        packet_path=PACKET,
        episode_rows_path=_write_rows(tmp_path, "ref_rows.csv", rows),
        output_path=out,
    )
    assert set(ranking.keys()) == set(metric.STRUCTURAL_CLASS_ORDER)
    assert set(ranking.values()) == {1, 2, 3, 4}
    assert ranking["constraint_first_hybrid"] == 1
    assert ranking["baseline_reactive"] == 4

    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines[0].split(",") == metric.RANKING_COLUMNS
    for line in lines[1:]:
        assert line.split(",")[2] == ROSTER_SIGNATURE


def test_candidate_ranking_flips_order_when_predictive_improves(tmp_path: Path) -> None:
    """A geometry where predictive beats learned_policy shows a rank flip, not a fixed order."""
    success = {
        "scenario_adaptive_hybrid_orca_v1": 0.95,
        "hybrid_rule_v3_fast_progress_static_escape": 0.90,
        "ppo": 0.55,
        "guarded_ppo": 0.52,
        "prediction_planner": 0.88,
        "prediction_mpc": 0.89,
        "prediction_mpc_cbf": 0.87,
        "goal": 0.40,
        "social_force": 0.43,
        "orca": 0.46,
        "socnav_sampling": 0.41,
        "sacadrl": 0.44,
    }
    rows = _rows_with_success(success)
    out = tmp_path / "candidate_ranking.csv"
    ranking = metric.build_ranking_for_matrix(
        packet_path=PACKET,
        episode_rows_path=_write_rows(tmp_path, "cand_rows.csv", rows),
        output_path=out,
    )
    assert ranking["predictive"] < ranking["learned_policy"]


def test_rejects_fallback_or_degraded_rows(tmp_path: Path) -> None:
    """Fallback/degraded planner rows must not enter the structural ranking."""
    rows = _rows_with_success({})
    rows[0]["execution_mode"] = "fallback"
    out = tmp_path / "out.csv"
    try:
        metric.build_ranking_for_matrix(
            packet_path=PACKET,
            episode_rows_path=_write_rows(tmp_path, "rows.csv", rows),
            output_path=out,
        )
    except metric.RankingMetricError as exc:
        assert "fallback" in str(exc).lower()
    else:
        raise AssertionError("fallback row must fail closed")


def test_rejects_unknown_planner(tmp_path: Path) -> None:
    """A planner outside the preregistered roster cannot be ranked."""
    rows = _rows_with_success({})
    rows[0]["planner_key"] = "unknown_planner"
    out = tmp_path / "out.csv"
    try:
        metric.build_ranking_for_matrix(
            packet_path=PACKET,
            episode_rows_path=_write_rows(tmp_path, "rows.csv", rows),
            output_path=out,
        )
    except metric.RankingMetricError as exc:
        assert "not in preregistered roster" in str(exc)
    else:
        raise AssertionError("unknown planner must fail closed")


def test_rejects_matrix_with_missing_class(tmp_path: Path) -> None:
    """A matrix whose campaign produced no rows for a structural class fails closed."""
    rows = _rows_with_success({})
    rows = [row for row in rows if row["planner_key"] not in ("ppo", "guarded_ppo")]
    out = tmp_path / "out.csv"
    try:
        metric.build_ranking_for_matrix(
            packet_path=PACKET,
            episode_rows_path=_write_rows(tmp_path, "rows.csv", rows),
            output_path=out,
        )
    except metric.RankingMetricError as exc:
        assert "no eligible rows" in str(exc)
    else:
        raise AssertionError("missing-class matrix must fail closed")


def test_metric_output_feedable_to_agreement_builder(
    tmp_path: Path,
) -> None:
    """The metric output CSV is directly consumable by the #5642 agreement builder."""
    from scripts.validation import build_issue_5592_cross_matrix_agreement as builder

    ref_rows = _rows_with_success(
        {
            "scenario_adaptive_hybrid_orca_v1": 0.95,
            "ppo": 0.80,
            "prediction_planner": 0.70,
            "goal": 0.40,
        }
    )
    cand_rows = _rows_with_success(
        {
            "scenario_adaptive_hybrid_orca_v1": 0.93,
            "ppo": 0.55,
            "prediction_planner": 0.88,
            "goal": 0.40,
        }
    )
    ref_ranking = tmp_path / "ref_ranking.csv"
    cand_ranking = tmp_path / "cand_ranking.csv"
    metric.build_ranking_for_matrix(
        packet_path=PACKET,
        episode_rows_path=_write_rows(tmp_path, "ref.csv", ref_rows),
        output_path=ref_ranking,
    )
    metric.build_ranking_for_matrix(
        packet_path=PACKET,
        episode_rows_path=_write_rows(tmp_path, "cand.csv", cand_rows),
        output_path=cand_ranking,
    )

    summary = builder.build_packet(
        packet_path=PACKET,
        reference_ranking_path=ref_ranking,
        candidate_ranking_path=cand_ranking,
        output_dir=tmp_path / "out",
        generated_at="2026-07-17T00:00:00+00:00",
    )
    assert summary["status"] == "ready"
    # Reference: hybrid best; candidate: predictive improved past learned -> a flip exists.
    assert summary["disagreement_row_count"] >= 1


def test_fixture_rejects_unknown_success_override() -> None:
    """Fixture typos must fail instead of silently changing no planner."""
    with pytest.raises(KeyError, match="unknown planner override"):
        _rows_with_success({"typo_planner": 0.9})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("success_rate", "NaN"),
        ("collision_event_rate", "Inf"),
        ("near_miss_event_rate", "-Inf"),
        ("timeout_rate", "not-a-number"),
        ("snqi_mean", "NaN"),
    ],
)
def test_rejects_nonfinite_or_malformed_metric_cells(
    tmp_path: Path, field: str, value: str
) -> None:
    """Non-finite or malformed numeric cells must fail closed before ranking."""
    rows = _rows_with_success({})
    rows[0][field] = value
    with pytest.raises(metric.RankingMetricError, match="invalid or non-finite"):
        metric.build_ranking_for_matrix(
            packet_path=PACKET,
            episode_rows_path=_write_rows(tmp_path, "rows.csv", rows),
            output_path=tmp_path / "out.csv",
        )


def test_rejects_row_missing_required_metric_field(tmp_path: Path) -> None:
    """A row missing a core metric field must fail closed, not impute a best-case value.

    A missing ``collision_event_rate`` must not be silently treated as 0.0 (best), since
    that would mask a data problem and rank the class as collision-free.
    """
    rows = _rows_with_success({})
    del rows[0]["collision_event_rate"]
    out = tmp_path / "out.csv"
    try:
        metric.build_ranking_for_matrix(
            packet_path=PACKET,
            episode_rows_path=_write_rows(tmp_path, "rows.csv", rows),
            output_path=out,
        )
    except metric.RankingMetricError as exc:
        assert "collision_event_rate" in str(exc)
        assert "missing required metric field" in str(exc)
    else:
        raise AssertionError("missing required metric field must fail closed")
