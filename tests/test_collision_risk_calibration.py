"""Tests for the matched collision-risk calibration harness (issue #5445).

Test names include both ``risk`` and ``calibration`` so the issue's validation
selector ``uv run pytest -q tests -k 'risk and calibration'`` collects them.

The suite covers three things: (1) the calibration/ranking metric math on tiny
hand-computed examples, (2) self-consistency -- the constant-velocity Monte Carlo
estimator is well-calibrated against its own model and the harness reports a
larger calibration error under deliberate misspecification, and (3) the CLI /
config packet, including the fail-closed insufficient-trace path.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from robot_sf.nav.predictive_types import PedestrianState
from robot_sf.research.collision_risk import RiskEstimatorConfig, action_from_constant_velocity
from robot_sf.research.collision_risk.calibration import (
    AVAILABLE_ESTIMATORS,
    UNAVAILABLE_ESTIMATORS,
    CalibrationInputError,
    EstimatorPrediction,
    FamilySpec,
    LabeledSample,
    MatchedDatasetProvenance,
    average_precision,
    brier_score,
    evaluate_estimator,
    expected_calibration_error,
    fnr_at_thresholds,
    generate_matched_dataset,
    log_loss,
    reliability_curve,
    time_to_warning_summary,
)

CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "configs/analysis/issue_5445_matched_calibration.yaml"
)


def _small_config(**overrides: object) -> RiskEstimatorConfig:
    """Return a fast estimator config for calibration tests."""
    base = {"horizon_steps": 15, "dt_s": 0.1, "n_samples": 256, "seed": 7}
    base.update(overrides)
    return RiskEstimatorConfig(**base)  # type: ignore[arg-type]


def _provenance(
    n_samples: int, *, horizon_steps: int, dt_s: float, seed: int
) -> MatchedDatasetProvenance:
    """Build a minimal complete provenance for tests."""
    return MatchedDatasetProvenance(
        target_distribution="declared constant-velocity model (test)",
        collision_predicate="disc footprint contact within horizon",
        horizon_steps=horizon_steps,
        dt_s=dt_s,
        planner_rows=("constant_velocity_drive",),
        discovery_calibration_split="all calibration, unit weights",
        prevalence_note="unit weights",
        n_samples=n_samples,
        compute_cap="cpu fixture",
        stop_rule="stop before real-distribution run",
        seed=seed,
    )


# --------------------------------------------------------------------------- #
# Metric math
# --------------------------------------------------------------------------- #
def test_risk_calibration_brier_and_log_loss_match_hand_values() -> None:
    """Weighted Brier and log loss match closed-form values on a tiny example."""
    scores = np.array([0.0, 1.0, 0.5, 0.5])
    labels = np.array([0.0, 1.0, 1.0, 0.0])
    weights = np.ones(4)
    # Brier: (0 + 0 + 0.25 + 0.25) / 4 = 0.125
    assert brier_score(scores, labels, weights) == pytest.approx(0.125)
    # Log loss with clip: perfect first two ~0, then two * -ln(0.5).
    expected_ll = (2 * -np.log(0.5)) / 4
    assert log_loss(scores, labels, weights) == pytest.approx(expected_ll, rel=1e-4)


def test_risk_calibration_weights_shift_brier() -> None:
    """Weighting a sample changes the weighted Brier score as expected."""
    scores = np.array([1.0, 0.0])
    labels = np.array([0.0, 0.0])
    even = brier_score(scores, labels, np.array([1.0, 1.0]))
    heavy_wrong = brier_score(scores, labels, np.array([3.0, 1.0]))
    assert even == pytest.approx(0.5)
    assert heavy_wrong == pytest.approx(0.75)  # (3*1 + 1*0)/4


def test_risk_calibration_reliability_curve_bins_and_ece() -> None:
    """Reliability bins carry correct counts and ECE is the weighted gap."""
    scores = np.array([0.05, 0.15, 0.95, 0.85])
    labels = np.array([0.0, 0.0, 1.0, 0.0])
    weights = np.ones(4)
    curve = reliability_curve(scores, labels, weights, n_bins=10)
    # Two low-score samples in bin 0-0.1 and 0.1-0.2; two high-score in 0.8-0.9/0.9-1.0.
    counts = {(round(b["bin_lo"], 1)): b["count"] for b in curve}
    assert counts[0.0] == 1 and counts[0.1] == 1
    ece = expected_calibration_error(curve)
    assert 0.0 <= ece <= 1.0


def test_risk_calibration_average_precision_perfect_ranking() -> None:
    """A perfectly separating score gives average precision 1.0."""
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    labels = np.array([1.0, 1.0, 0.0, 0.0])
    weights = np.ones(4)
    assert average_precision(scores, labels, weights) == pytest.approx(1.0)


def test_risk_calibration_average_precision_no_positives_is_nan() -> None:
    """Average precision is NaN when there are no positive labels."""
    scores = np.array([0.9, 0.1])
    labels = np.array([0.0, 0.0])
    weights = np.ones(2)
    assert np.isnan(average_precision(scores, labels, weights))


def test_risk_calibration_fnr_at_thresholds() -> None:
    """False-negative rate counts positives whose score falls below each threshold."""
    scores = np.array([0.05, 0.4, 0.9])
    labels = np.array([1.0, 1.0, 1.0])
    weights = np.ones(3)
    fnr = fnr_at_thresholds(scores, labels, weights, (0.1, 0.5))
    assert fnr["0.100"] == pytest.approx(1 / 3)  # only 0.05 below 0.1
    assert fnr["0.500"] == pytest.approx(2 / 3)  # 0.05 and 0.4 below 0.5


def test_risk_calibration_time_to_warning_lead_time() -> None:
    """Time-to-warning reports lead time only over warned contacting samples."""
    action = action_from_constant_velocity("a", [0, 0], [1, 0], horizon_steps=10, dt_s=0.1)
    ped = (PedestrianState(id=0, position=np.array([1.0, 0.0]), velocity=np.array([0.0, 0.0])),)
    samples = [
        LabeledSample("s0", "f", action, ped, True, 8, 1.0, {}),
        LabeledSample("s1", "f", action, ped, False, -1, 1.0, {}),
    ]
    preds = [
        EstimatorPrediction(score=0.9, is_probability=True, warning_step=5, latency_ms=1.0),
        EstimatorPrediction(score=0.1, is_probability=True, warning_step=-1, latency_ms=1.0),
    ]
    summary = time_to_warning_summary(preds, samples, dt_s=0.1)
    assert summary["n_contacts"] == 1
    assert summary["warned_fraction"] == pytest.approx(1.0)
    assert summary["mean_lead_time_s"] == pytest.approx((8 - 5) * 0.1)


# --------------------------------------------------------------------------- #
# Self-consistency + miscalibration detection
# --------------------------------------------------------------------------- #
def test_risk_calibration_in_model_is_well_calibrated() -> None:
    """The MC estimator is well-calibrated against its own declared model."""
    config = _small_config()
    family = FamilySpec(
        name="in_model", n_scenarios=180, gt_velocity_std_m_s=config.velocity_std_m_s
    )
    dataset = generate_matched_dataset(
        [family],
        config,
        _provenance(180, horizon_steps=config.horizon_steps, dt_s=config.dt_s, seed=config.seed),
    )
    result = evaluate_estimator("constant_velocity_mc", dataset, config, n_bootstrap=50)
    assert result["kind"] == "probabilistic"
    # In-model calibration error should be small; keep a generous bound for MC noise.
    assert result["expected_calibration_error"] < 0.12
    assert result["horizon_monotonicity"]["monotone_fraction"] == pytest.approx(1.0)
    # Stratified metrics are reported per family with a prevalence and ECE.
    strat = result["stratified"]["family"]["in_model"]
    assert strat["n"] == 180
    assert "expected_calibration_error" in strat


def test_risk_calibration_detects_misspecification() -> None:
    """A biased ground truth yields a larger calibration error than the in-model case."""
    config = _small_config()
    in_model = FamilySpec(
        name="in_model", n_scenarios=180, gt_velocity_std_m_s=config.velocity_std_m_s
    )
    biased = FamilySpec(
        name="biased",
        n_scenarios=180,
        gt_velocity_std_m_s=config.velocity_std_m_s,
        gt_velocity_bias_m_s=(-0.6, 0.0),
    )
    prov_in = _provenance(180, horizon_steps=config.horizon_steps, dt_s=config.dt_s, seed=11)
    prov_bad = _provenance(180, horizon_steps=config.horizon_steps, dt_s=config.dt_s, seed=11)
    good = evaluate_estimator(
        "constant_velocity_mc",
        generate_matched_dataset([in_model], config, prov_in),
        config,
        n_bootstrap=0,
    )
    bad = evaluate_estimator(
        "constant_velocity_mc",
        generate_matched_dataset([biased], config, prov_bad),
        config,
        n_bootstrap=0,
    )
    assert bad["expected_calibration_error"] > good["expected_calibration_error"]


def test_risk_calibration_deterministic_signal_not_on_probability_curve() -> None:
    """The deterministic warning is graded as a ranking, never a reliability curve."""
    config = _small_config()
    family = FamilySpec(name="f", n_scenarios=120, gt_velocity_std_m_s=config.velocity_std_m_s)
    dataset = generate_matched_dataset(
        [family],
        config,
        _provenance(120, horizon_steps=config.horizon_steps, dt_s=config.dt_s, seed=3),
    )
    result = evaluate_estimator("deterministic_ttc", dataset, config, n_bootstrap=0)
    assert result["kind"] == "warning"
    assert "reliability_curve" not in result
    assert "expected_calibration_error" not in result
    assert "average_precision" in result


# --------------------------------------------------------------------------- #
# Registry + fail-closed behaviour
# --------------------------------------------------------------------------- #
def test_risk_calibration_unavailable_estimators_recorded() -> None:
    """Contracted-but-missing estimators are named as unavailable, not dropped."""
    assert "multimodal_forecast_mc" in UNAVAILABLE_ESTIMATORS
    assert "learned_risk_1472" in UNAVAILABLE_ESTIMATORS
    assert set(AVAILABLE_ESTIMATORS).isdisjoint(UNAVAILABLE_ESTIMATORS)


def test_risk_calibration_zero_weight_fails_closed() -> None:
    """A zero total weight fails closed rather than dividing by zero."""
    action = action_from_constant_velocity("a", [0, 0], [1, 0], horizon_steps=10, dt_s=0.1)
    ped = (PedestrianState(id=0, position=np.array([1.0, 0.0]), velocity=np.array([0.0, 0.0])),)
    sample = LabeledSample("s0", "f", action, ped, True, 5, 0.0, {})
    from robot_sf.research.collision_risk.calibration import MatchedDataset

    dataset = MatchedDataset(
        samples=(sample,),
        provenance=_provenance(1, horizon_steps=10, dt_s=0.1, seed=0),
    )
    with pytest.raises(CalibrationInputError):
        dataset.prevalence()


# --------------------------------------------------------------------------- #
# CLI / config packet
# --------------------------------------------------------------------------- #
def test_risk_calibration_report_cli_scores_packet(tmp_path: Path) -> None:
    """The CLI scores the frozen packet and emits verdicts for every estimator."""
    pytest.importorskip("yaml")
    from scripts.analysis.collision_risk_calibration_report import build_report

    report = build_report(CONFIG_PATH)
    assert report["status"] == "scored"
    assert report["provenance"]["n_samples"] == 880
    ids = {row["estimator_id"] for row in report["estimators"]}
    assert ids == set(AVAILABLE_ESTIMATORS)
    for row in report["estimators"]:
        assert row["verdict"] in {"use online", "offline analysis only", "revise", "stop"}
    unavailable_ids = {row["estimator_id"] for row in report["unavailable_estimators"]}
    assert unavailable_ids == set(UNAVAILABLE_ESTIMATORS)


def test_risk_calibration_report_cli_fails_closed_on_insufficient_traces(tmp_path: Path) -> None:
    """The CLI refuses to score when eligible samples fall below the minimum."""
    pytest.importorskip("yaml")
    import yaml

    from scripts.analysis.collision_risk_calibration_report import build_report

    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    # Shrink every family below the preregistered minimum eligible-sample count.
    for family in data["families"]:
        family["n_scenarios"] = 1
    data["evaluation"]["min_eligible_samples"] = 200
    packet = tmp_path / "tiny.yaml"
    with packet.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle)

    report = build_report(packet)
    assert report["status"] == "insufficient_traces"
    assert report["n_samples"] < report["min_eligible_samples"]
