"""Tests for the SNQI per-term normalization inventory (issue #3699).

These tests prove the diagnostic surfaces the mixed raw/baseline-normalized
condition *fail-closed* and that adding the inventory does not change any
``compute_snqi`` output. The key anti-drift guarantee is
``test_inventory_reconstructs_compute_snqi``: the SNQI score rebuilt from the
inventory's term table must equal ``compute_snqi`` exactly, so the inventory's
scaling labels cannot silently diverge from the scoring code they describe.
"""

from __future__ import annotations

import pytest

from robot_sf.benchmark.snqi.compute import compute_snqi
from robot_sf.benchmark.snqi.normalization_inventory import (
    SCALING_BASELINE_NORMALIZED,
    SCALING_RAW,
    SNQI_LEGACY_SCORE_STATUS,
    SNQI_LEGACY_SCORE_VERSION,
    SNQI_TERM_SCALING,
    SNQI_V1_SCORE_STATUS,
    SNQI_V1_TERM_SCALING,
    build_snqi_contribution_diagnostics,
    build_snqi_normalization_inventory,
    build_snqi_version_contract,
    format_normalization_report,
    scaled_term_value,
)

# Baseline stats covering exactly the four count-type penalty terms that
# ``compute_snqi`` routes through ``normalize_metric``.
_BASELINE_STATS = {
    "time_to_goal_norm": {"med": 0.0, "p95": 1.0},
    "collisions": {"med": 1.0, "p95": 5.0},
    "near_misses": {"med": 2.0, "p95": 8.0},
    "comfort_exposure": {"med": 0.0, "p95": 1.0},
    "force_exceed_events": {"med": 0.0, "p95": 4.0},
    "jerk_mean": {"med": 0.1, "p95": 1.0},
}

_WEIGHTS = {
    "w_success": 1.0,
    "w_time": 0.8,
    "w_collisions": 2.0,
    "w_near": 1.0,
    "w_comfort": 0.5,
    "w_force_exceed": 1.5,
    "w_jerk": 0.3,
}

_METRICS = {
    "success": 1.0,
    "time_to_goal_norm": 0.7,
    "collisions": 3.0,
    "near_misses": 4.0,
    "comfort_exposure": 0.25,
    "force_exceed_events": 2.0,
    "jerk_mean": 0.5,
}


def test_inventory_reconstructs_compute_snqi():
    """Rebuilding the score from the inventory equals ``compute_snqi`` exactly.

    This anchors the inventory's scaling labels to the real scoring code: if a
    term's ``scaling``/``sign``/``metric_key``/``default`` ever drifts from
    ``compute_snqi``, this reconstruction diverges and the test fails.
    """
    reconstructed = 0.0
    for term in SNQI_TERM_SCALING:
        value = scaled_term_value(term, _METRICS, _BASELINE_STATS)
        reconstructed += term.sign * _WEIGHTS[term.weight_name] * value

    expected = compute_snqi(_METRICS, _WEIGHTS, _BASELINE_STATS)
    assert reconstructed == pytest.approx(expected)


def test_reconstruction_holds_with_missing_metrics():
    """Per-term defaults match ``compute_snqi`` when metrics are absent."""
    sparse = {"success": 1.0}  # every other term falls back to its default
    reconstructed = sum(
        term.sign * _WEIGHTS[term.weight_name] * scaled_term_value(term, sparse, _BASELINE_STATS)
        for term in SNQI_TERM_SCALING
    )
    expected = compute_snqi(sparse, _WEIGHTS, _BASELINE_STATS)
    assert reconstructed == pytest.approx(expected)


def test_mixed_scale_is_surfaced():
    """The inventory flags that penalty terms span raw and normalized scales."""
    inv = build_snqi_normalization_inventory(_BASELINE_STATS)
    assert inv.mixed_scale is True
    assert inv.is_consistent is False
    raw_penalty = {t.term for t in inv.raw_penalty_terms}
    assert raw_penalty == {"time", "comfort"}
    normalized_penalty = {t.term for t in inv.normalized_penalty_terms}
    assert normalized_penalty == {"collisions", "near", "force_exceed", "jerk"}


def test_unbounded_terms_are_the_raw_penalty_terms():
    """``time`` and ``comfort`` are reported as the unbounded penalty terms."""
    inv = build_snqi_normalization_inventory(_BASELINE_STATS)
    assert {t.term for t in inv.unbounded_terms} == {"time", "comfort"}


def test_raw_terms_are_unbounded_in_behavior():
    """A raw term's scaled value grows without bound; normalized stays in [0, 1].

    This is the behavioral counterpart to the static ``bounded`` labels: it
    confirms the labels describe real ``compute_snqi`` behavior.
    """
    time_term = next(t for t in SNQI_TERM_SCALING if t.term == "time")
    coll_term = next(t for t in SNQI_TERM_SCALING if t.term == "collisions")

    huge = {"time_to_goal_norm": 1000.0, "collisions": 1000.0}
    assert scaled_term_value(time_term, huge, _BASELINE_STATS) == pytest.approx(1000.0)
    # Baseline-normalized term saturates at the clamp upper bound.
    assert scaled_term_value(coll_term, huge, _BASELINE_STATS) == pytest.approx(1.0)


def test_missing_baseline_coverage_is_surfaced_fail_closed():
    """Normalized terms with no median/p95 are reported, not silently passed."""
    partial = {"collisions": {"med": 1.0, "p95": 5.0}}  # near/force/jerk missing
    inv = build_snqi_normalization_inventory(partial)
    missing = {t.metric_key for t in inv.missing_baseline_coverage}
    assert missing == {"near_misses", "force_exceed_events", "jerk_mean"}


def test_no_baseline_stats_reports_all_normalized_uncovered():
    """Without baseline stats, every normalized term is reported as uncovered."""
    inv = build_snqi_normalization_inventory(None)
    missing = {t.metric_key for t in inv.missing_baseline_coverage}
    assert missing == {"near_misses", "force_exceed_events", "jerk_mean", "collisions"}


def test_inventory_does_not_change_compute_snqi_outputs():
    """Building the inventory must not mutate metrics/weights/stats or scores."""
    before = compute_snqi(_METRICS, _WEIGHTS, _BASELINE_STATS)
    _ = build_snqi_normalization_inventory(_BASELINE_STATS)
    after = compute_snqi(_METRICS, _WEIGHTS, _BASELINE_STATS)
    assert before == after


def test_term_table_covers_every_weight():
    """Every SNQI weight has exactly one term; signs match reward/penalty roles."""
    weight_names = [t.weight_name for t in SNQI_TERM_SCALING]
    assert sorted(weight_names) == sorted(_WEIGHTS)
    success = next(t for t in SNQI_TERM_SCALING if t.term == "success")
    assert success.sign == 1
    assert all(t.sign == -1 for t in SNQI_TERM_SCALING if t.term != "success")
    # Scaling values are restricted to the two known regimes.
    assert all(t.scaling in {SCALING_RAW, SCALING_BASELINE_NORMALIZED} for t in SNQI_TERM_SCALING)


def test_to_dict_is_json_serializable_and_self_consistent():
    """The payload round-trips through JSON and reports the mixed-scale flag."""
    import json

    inv = build_snqi_normalization_inventory(_BASELINE_STATS)
    payload = inv.to_dict()
    encoded = json.loads(json.dumps(payload))
    assert encoded["mixed_scale"] is True
    assert set(encoded["raw_penalty_terms"]) == {"time", "comfort"}
    assert encoded["is_consistent"] is False
    status_by_term = {term["term"]: term["normalization_status"] for term in encoded["terms"]}
    assert status_by_term["success"] == "raw_bounded"
    assert status_by_term["time"] == "raw_unbounded"
    assert status_by_term["comfort"] == "raw_unbounded"
    assert status_by_term["collisions"] == "baseline_normalized_bounded"
    basis_by_term = {term["term"]: term["measurement_basis"] for term in encoded["terms"]}
    assert basis_by_term["time"] == "raw time-to-goal ratio"
    assert basis_by_term["comfort"] == "raw accumulated comfort-exposure value"
    assert basis_by_term["collisions"] == "baseline-relative median/p95 clamped value"
    assert encoded["score_version_contract"] == build_snqi_version_contract()


def test_score_version_contract_preserves_legacy_snqi_v0_as_diagnostic_only():
    """Issue #3699 checker labels current SNQI semantics without changing scores."""

    contract = build_snqi_version_contract()

    assert contract["score_version"] == SNQI_LEGACY_SCORE_VERSION == "SNQI-v0"
    assert contract["status"] == SNQI_LEGACY_SCORE_STATUS
    assert contract["diagnostic_only"] is True
    assert contract["mixed_basis_preserved"] is True
    assert contract["score_semantics_changed"] is False
    assert contract["future_bounded_contract"] == "SNQI-v1"
    assert contract["decision_issue"] == 3699
    assert contract["successor_issue"] == 3978


def test_score_version_contract_exposes_snqi_v1_bounded_diagnostic() -> None:
    """SNQI-v1 is an opt-in bounded diagnostic, not a v0 reinterpretation."""
    contract = build_snqi_version_contract("SNQI-v1")

    assert contract["score_version"] == "SNQI-v1"
    assert contract["status"] == SNQI_V1_SCORE_STATUS
    assert contract["diagnostic_only"] is True
    assert contract["mixed_basis_preserved"] is False
    assert contract["score_semantics_changed"] is True
    assert contract["supersedes"] == "SNQI-v0"
    assert contract["decision_issue"] == 3978
    assert contract["legacy_decision_issue"] == 3699


def test_snqi_v1_inventory_penalties_are_bounded_baseline_normalized() -> None:
    """Every SNQI-v1 penalty term uses the comparable baseline-normalized basis."""
    inv = build_snqi_normalization_inventory(_BASELINE_STATS, score_version="SNQI-v1")

    assert inv.score_version == "SNQI-v1"
    assert inv.mixed_scale is False
    assert inv.is_consistent is True
    assert inv.unbounded_terms == []
    assert {t.metric_key for t in inv.normalized_penalty_terms} == {
        "time_to_goal_norm",
        "collisions",
        "near_misses",
        "comfort_exposure",
        "force_exceed_events",
        "jerk_mean",
    }
    assert all(t.scaling == SCALING_BASELINE_NORMALIZED for t in inv.penalty_terms)
    assert all(t.bounded for t in inv.penalty_terms)


def test_snqi_v1_contribution_diagnostics_are_comparable() -> None:
    """SNQI-v1 scaled values and signed penalties stay within weight bounds."""
    metrics = {
        "success": 1.0,
        "time_to_goal_norm": 3.0,
        "collisions": 9.0,
        "near_misses": 9.0,
        "comfort_exposure": 2.0,
        "force_exceed_events": 9.0,
        "jerk_mean": 9.0,
    }
    weights = {term.weight_name: 1.0 for term in SNQI_V1_TERM_SCALING}
    baseline_stats = {
        "time_to_goal_norm": {"med": 0.0, "p95": 1.0},
        "collisions": {"med": 0.0, "p95": 1.0},
        "near_misses": {"med": 0.0, "p95": 1.0},
        "comfort_exposure": {"med": 0.0, "p95": 1.0},
        "force_exceed_events": {"med": 0.0, "p95": 1.0},
        "jerk_mean": {"med": 0.0, "p95": 1.0},
    }

    diagnostics = build_snqi_contribution_diagnostics(
        metrics,
        weights,
        baseline_stats,
        score_version="SNQI-v1",
    )

    assert diagnostics["mixed_basis"] is False
    assert diagnostics["has_weight_bound_exceedance"] is False
    assert diagnostics["normalization_contract"]["status"] == "comparable_weight_basis"
    assert diagnostics["normalization_contract"]["weights_comparable"] is True
    assert diagnostics["normalization_contract"]["weight_bound_exceedance_terms"] == []
    terms = diagnostics["terms"]
    for term in terms:
        if term["signed_contribution"] < 0:
            assert term["scaling"] == SCALING_BASELINE_NORMALIZED
            assert 0.0 <= term["scaled_value"] <= 1.0
            assert abs(term["signed_contribution"]) <= abs(term["weight"])


def test_snqi_v1_contribution_diagnostics_fail_closed_on_missing_baseline() -> None:
    """SNQI-v1 diagnostics reject inputs the scorer would reject."""
    metrics = {
        "success": 1.0,
        "time_to_goal_norm": 3.0,
        "collisions": 9.0,
        "near_misses": 9.0,
        "comfort_exposure": 2.0,
        "force_exceed_events": 9.0,
        "jerk_mean": 9.0,
    }
    weights = {term.weight_name: 1.0 for term in SNQI_V1_TERM_SCALING}
    incomplete_baseline_stats = {
        "collisions": {"med": 0.0, "p95": 1.0},
        "near_misses": {"med": 0.0, "p95": 1.0},
        "comfort_exposure": {"med": 0.0, "p95": 1.0},
        "force_exceed_events": {"med": 0.0, "p95": 1.0},
        "jerk_mean": {"med": 0.0, "p95": 1.0},
    }

    with pytest.raises(ValueError, match="SNQI-v1 baseline_stats missing med/p95"):
        build_snqi_contribution_diagnostics(
            metrics,
            weights,
            incomplete_baseline_stats,
            score_version="SNQI-v1",
        )


def test_snqi_v1_compute_path_is_covered_in_optional_lane() -> None:
    """Optional benchmark lane covers the versioned compute path."""
    metrics = {
        "success": 1.0,
        "time_to_goal_norm": 3.0,
        "collisions": 9.0,
        "near_misses": 9.0,
        "comfort_exposure": 2.0,
        "force_exceed_events": 9.0,
        "jerk_mean": 9.0,
    }
    weights = {term.weight_name: 1.0 for term in SNQI_V1_TERM_SCALING}
    baseline_stats = {
        "time_to_goal_norm": {"med": 0.0, "p95": 1.0},
        "collisions": {"med": 0.0, "p95": 1.0},
        "near_misses": {"med": 0.0, "p95": 1.0},
        "comfort_exposure": {"med": 0.0, "p95": 1.0},
        "force_exceed_events": {"med": 0.0, "p95": 1.0},
        "jerk_mean": {"med": 0.0, "p95": 1.0},
    }

    assert compute_snqi(metrics, weights, baseline_stats, score_version="SNQI-v1") == pytest.approx(
        -5.0
    )
    with pytest.raises(ValueError, match="unknown SNQI score version"):
        compute_snqi(metrics, weights, baseline_stats, score_version="SNQI-vunknown")


def test_format_report_is_human_readable():
    """The text report names the issue and every term."""
    inv = build_snqi_normalization_inventory(_BASELINE_STATS)
    text = format_normalization_report(inv)
    assert "issue #3699" in text
    assert "basis" in text
    assert "SNQI-v0 (legacy_mixed_basis_diagnostic_only)" in text
    assert "raw time-to-goal ratio" in text
    assert "baseline-relative median/p95 clamped value" in text
    for term in SNQI_TERM_SCALING:
        assert term.term in text


def test_format_report_uses_inventory_score_version_contract() -> None:
    """V1 inventory text reports the v1 contract, not the default v0 contract."""
    inv = build_snqi_normalization_inventory(_BASELINE_STATS, score_version="SNQI-v1")

    text = format_normalization_report(inv)

    assert "SNQI-v1 (bounded_baseline_relative_diagnostic_only)" in text
    assert "SNQI-v0 (legacy_mixed_basis_diagnostic_only)" not in text


def test_contribution_diagnostics_reconstruct_snqi_and_flag_raw_dominance():
    """Contribution checker exposes mixed-basis dominance without changing score."""
    metrics = {
        "success": 1.0,
        "time_to_goal_norm": 3.0,
        "collisions": 9.0,
        "near_misses": 9.0,
        "comfort_exposure": 2.0,
        "force_exceed_events": 9.0,
        "jerk_mean": 9.0,
    }
    weights = {term.weight_name: 1.0 for term in SNQI_TERM_SCALING}
    baseline_stats = {
        "collisions": {"med": 0.0, "p95": 1.0},
        "near_misses": {"med": 0.0, "p95": 1.0},
        "force_exceed_events": {"med": 0.0, "p95": 1.0},
        "jerk_mean": {"med": 0.0, "p95": 1.0},
    }

    diagnostics = build_snqi_contribution_diagnostics(metrics, weights, baseline_stats)
    signed_total = sum(term["signed_contribution"] for term in diagnostics["terms"])

    assert diagnostics["diagnostic_only"] is True
    assert diagnostics["mixed_basis"] is True
    assert diagnostics["raw_penalty_terms_dominate"] is True
    assert diagnostics["raw_penalty_absolute_share"] == pytest.approx(0.5)
    assert diagnostics["baseline_normalized_penalty_absolute_share"] == pytest.approx(0.4)
    assert diagnostics["has_weight_bound_exceedance"] is True
    assert {term["term"] for term in diagnostics["weight_bound_exceedances"]} == {"time", "comfort"}
    assert signed_total == pytest.approx(compute_snqi(metrics, weights, baseline_stats))

    contract = diagnostics["normalization_contract"]
    version_contract = diagnostics["score_version_contract"]
    assert contract["schema_version"] == "snqi_normalization_contract.v1"
    assert contract["diagnostic_only"] is True
    assert contract["status"] == "mixed_unbounded_penalty_basis"
    assert contract["weights_comparable"] is False
    assert contract["decision_issue"] == 3699
    assert contract["successor_issue"] == 3978
    assert contract["raw_unbounded_penalty_terms"] == ["time", "comfort"]
    assert contract["baseline_normalized_penalty_terms"] == [
        "collisions",
        "near",
        "force_exceed",
        "jerk",
    ]
    assert contract["raw_penalty_absolute_share"] == pytest.approx(0.5)
    assert contract["baseline_normalized_penalty_absolute_share"] == pytest.approx(0.4)
    assert contract["weight_bound_exceedance_terms"] == ["time", "comfort"]
    assert "bypass normalize_metric" in contract["reasons"][0]
    assert version_contract["score_version"] == "SNQI-v0"
    assert version_contract["score_semantics_changed"] is False

    by_term = {term["term"]: term for term in diagnostics["terms"]}
    assert by_term["time"]["scaled_value"] == pytest.approx(3.0)
    assert by_term["time"]["exceeds_weight_bound"] is True
    assert by_term["comfort"]["scaled_value"] == pytest.approx(2.0)
    assert by_term["comfort"]["exceeds_weight_bound"] is True
    assert by_term["collisions"]["scaled_value"] == pytest.approx(1.0)
    assert by_term["collisions"]["exceeds_weight_bound"] is False
    assert by_term["time"]["normalization_status"] == "raw_unbounded"
    assert by_term["collisions"]["normalization_status"] == "baseline_normalized_bounded"


def test_report_cli_writes_contribution_diagnostics(tmp_path, capsys):
    """Report CLI can attach per-term contribution diagnostics to JSON output."""
    import importlib.util
    import json
    import pathlib

    script_path = (
        pathlib.Path(__file__).resolve().parents[2]
        / "scripts"
        / "benchmark"
        / "snqi_normalization_inventory_report.py"
    )
    spec = importlib.util.spec_from_file_location(
        "snqi_normalization_inventory_report", script_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    metrics_path = tmp_path / "metrics.json"
    weights_path = tmp_path / "weights.json"
    baseline_path = tmp_path / "baseline.json"
    output_path = tmp_path / "inventory.json"
    metrics_path.write_text(json.dumps(_METRICS), encoding="utf-8")
    weights_path.write_text(json.dumps(_WEIGHTS), encoding="utf-8")
    baseline_path.write_text(json.dumps(_BASELINE_STATS), encoding="utf-8")

    exit_code = module.main(
        [
            "--baseline-stats",
            str(baseline_path),
            "--metrics",
            str(metrics_path),
            "--weights",
            str(weights_path),
            "--json-out",
            str(output_path),
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert "Contribution diagnostics:" in captured.out
    assert "Normalization contract: status=mixed_unbounded_penalty_basis" in captured.out
    assert payload["contributions"]["diagnostic_only"] is True
    assert payload["contributions"]["mixed_basis"] is True
    assert payload["contributions"]["normalization_contract"]["weights_comparable"] is False
    signed_total = sum(term["signed_contribution"] for term in payload["contributions"]["terms"])
    assert signed_total == pytest.approx(compute_snqi(_METRICS, _WEIGHTS, _BASELINE_STATS))


def _load_report_module():
    """Import the report CLI script as a module for direct ``main()`` testing."""
    import importlib.util
    import pathlib

    script_path = (
        pathlib.Path(__file__).resolve().parents[2]
        / "scripts"
        / "benchmark"
        / "snqi_normalization_inventory_report.py"
    )
    spec = importlib.util.spec_from_file_location(
        "snqi_normalization_inventory_report", script_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("metrics_text", "expected_stderr"),
    [
        ("{not valid json", "invalid contribution input JSON"),
        ("[1, 2, 3]", "must each contain a JSON object"),
    ],
)
def test_report_cli_rejects_bad_contribution_inputs(
    tmp_path, capsys, metrics_text, expected_stderr
):
    """Malformed or non-object contribution JSON fails closed with exit code 2.

    Without hardening these inputs escaped as tracebacks (JSONDecodeError) or
    crashed inside ``build_snqi_contribution_diagnostics`` (non-mapping payload)
    instead of returning the CLI's intended input-error exit code.
    """
    import json

    module = _load_report_module()
    metrics_path = tmp_path / "metrics.json"
    weights_path = tmp_path / "weights.json"
    metrics_path.write_text(metrics_text, encoding="utf-8")
    weights_path.write_text(json.dumps(_WEIGHTS), encoding="utf-8")

    exit_code = module.main(["--metrics", str(metrics_path), "--weights", str(weights_path)])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert expected_stderr in captured.err


def test_report_cli_rejects_directory_contribution_input(tmp_path, capsys):
    """A directory passed as a metrics path fails closed instead of erroring out."""
    import json

    module = _load_report_module()
    metrics_dir = tmp_path / "metrics_dir"
    metrics_dir.mkdir()
    weights_path = tmp_path / "weights.json"
    weights_path.write_text(json.dumps(_WEIGHTS), encoding="utf-8")

    exit_code = module.main(["--metrics", str(metrics_dir), "--weights", str(weights_path)])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "metrics file not found" in captured.err
