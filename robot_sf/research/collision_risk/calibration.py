"""Matched calibration evaluation for online collision-risk estimators (issue #5445).

This module compares collision-risk estimators built on the action-conditioned
API from :mod:`robot_sf.research.collision_risk.estimators` (issue #5444) under
**matched inputs**: every estimator scores exactly the same scenario histories,
candidate actions, footprints, and horizons, and is graded against the same
realized collision outcome. It answers the #5445 research question -- which
estimator is calibrated, action-sensitive, timely, and robust enough to support
analysis or planning -- without launching any benchmark campaign.

What is (and is not) claimed
----------------------------
The labelled outcomes here are generated from an **explicit, declared forecast
model** (constant-velocity Gaussian, optionally *misspecified* per scenario
family), not from a full simulator rollout. That makes this **API + fixture
evidence**: it exercises and validates the calibration machinery, demonstrates
self-consistency of the constant-velocity Monte Carlo estimator against its own
declared distribution, and proves that the machinery *detects* miscalibration
when the ground-truth distribution is deliberately mismatched. It is **not**
calibrated benchmark risk for the simulator distribution and never a real-world
risk claim. A real-distribution calibration run requires eligible simulator
traces with action-conditioned labels and is explicitly out of scope until that
compute packet is approved (see the stop rule in issue #5445).

Contract discipline (matches the #5445 comparison contract)
-----------------------------------------------------------
- Probabilistic estimators (constant-velocity MC) are placed on the reliability
  / Brier / log-loss curve.
- Deterministic warnings (TTC / velocity-obstacle / reachability) are graded
  **only** as warnings -- ranking (area under precision-recall), false-negative
  rate at declared thresholds, and time-to-warning. They are never placed on a
  probability calibration curve, because their scores are not probabilities.
- Estimators that do not yet exist in-repo (multimodal forecast MC, the #1472
  learned-risk model) are recorded as ``unavailable`` with a reason rather than
  silently omitted.
- Hard guards remain authoritative; no ``safe`` verdict is ever emitted.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from robot_sf.nav.predictive_types import PedestrianState
from robot_sf.research.collision_risk.estimators import (
    CandidateAction,
    RiskEstimatorConfig,
    action_from_constant_velocity,
    estimate_action_conditioned_risk,
    pedestrian_arrays,
    segment_min_distance,
)
from robot_sf.research.collision_risk.schema import latency_summary_from_samples

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

CALIBRATION_SCHEMA_VERSION = "collision_risk_calibration.v1"

# Estimators exercised on the matched inputs. ``kind`` controls how the estimator
# is graded: "probabilistic" outputs go on the reliability/Brier curve, "warning"
# outputs are graded as rankings/warnings only.
AVAILABLE_ESTIMATORS = ("constant_velocity_mc", "deterministic_ttc")

# Estimators named by the #5445 comparison contract that do not yet exist in this
# repository. They are reported as unavailable (with a reason) so the comparison
# never silently drops a contracted row.
UNAVAILABLE_ESTIMATORS = {
    "multimodal_forecast_mc": (
        "multimodal pedestrian forecast sampler is not implemented in-repo; the "
        "constant-velocity model is the only forecast baseline available (issue #5307 "
        "predictor family not merged)"
    ),
    "learned_risk_1472": (
        "learned collision-risk model from issue #1472 is not merged; no learned "
        "probability surface is available to score"
    ),
}

CLAIM_BOUNDARY = (
    "API + fixture evidence: calibration is measured against a declared forecast "
    "model, not full simulator rollouts. Self-consistency and miscalibration "
    "detection are demonstrated; this is not calibrated benchmark risk for the "
    "simulator distribution and never a real-world risk claim. Hard guards remain "
    "authoritative; no safe verdict is emitted."
)


class CalibrationInputError(ValueError):
    """Raised, fail-closed, when calibration inputs or provenance are insufficient."""


# --------------------------------------------------------------------------- #
# Data types
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LabeledSample:
    """One matched scenario: shared inputs plus a realized collision outcome.

    Attributes:
        scenario_id: Stable identifier for the scenario.
        family: Scenario family (used for stratification and as the ground-truth
            generative-model selector).
        action: The candidate robot action all estimators score.
        pedestrians: Actor states at time ``t`` (shared across estimators).
        realized_contact: Ground-truth label -- whether the drawn realization made
            robot/actor footprints touch within the horizon.
        realized_first_contact_step: Step index of first realized contact, or
            ``-1`` when no contact occurred.
        weight: Importance/prevalence weight for this sample (default 1.0). Used to
            reweight case-control or enriched discovery samples toward the target
            distribution.
        strata: Stratification tags (planner, density bin, prediction model, ...).
    """

    scenario_id: str
    family: str
    action: CandidateAction
    pedestrians: tuple[PedestrianState, ...]
    realized_contact: bool
    realized_first_contact_step: int
    weight: float = 1.0
    strata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class MatchedDatasetProvenance:
    """Preregistered provenance for a matched calibration dataset.

    These fields are the #5445 preregistration items: they must be fixed *before*
    scoring so the comparison cannot be tuned after seeing results.
    """

    target_distribution: str
    collision_predicate: str
    horizon_steps: int
    dt_s: float
    planner_rows: tuple[str, ...]
    discovery_calibration_split: str
    prevalence_note: str
    n_samples: int
    compute_cap: str
    stop_rule: str
    seed: int
    schema_version: str = CALIBRATION_SCHEMA_VERSION

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe mapping of the provenance."""
        return {
            "schema_version": self.schema_version,
            "target_distribution": self.target_distribution,
            "collision_predicate": self.collision_predicate,
            "horizon_steps": self.horizon_steps,
            "dt_s": self.dt_s,
            "planner_rows": list(self.planner_rows),
            "discovery_calibration_split": self.discovery_calibration_split,
            "prevalence_note": self.prevalence_note,
            "n_samples": self.n_samples,
            "compute_cap": self.compute_cap,
            "stop_rule": self.stop_rule,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class MatchedDataset:
    """A set of matched labelled samples plus their preregistered provenance."""

    samples: tuple[LabeledSample, ...]
    provenance: MatchedDatasetProvenance

    def prevalence(self) -> float:
        """Return the weighted base rate of realized contact."""
        weights = np.array([s.weight for s in self.samples], dtype=float)
        labels = np.array([1.0 if s.realized_contact else 0.0 for s in self.samples])
        total = weights.sum()
        if total <= 0.0:
            raise CalibrationInputError("total sample weight must be positive")
        return float((weights * labels).sum() / total)


@dataclass(frozen=True)
class EstimatorPrediction:
    """One estimator's output for one matched sample.

    Attributes:
        score: Ranking/probability score in ``[0, 1]``. For probabilistic
            estimators this is a calibrated-probability candidate; for warning
            estimators it is a monotone danger score used for ranking only.
        is_probability: True when ``score`` may be placed on the reliability/Brier
            calibration curve.
        warning_step: First horizon step at which the estimator raised a warning,
            or ``-1`` when it never warned within the horizon.
        latency_ms: Wall-clock latency of producing this prediction.
    """

    score: float
    is_probability: bool
    warning_step: int
    latency_ms: float


# --------------------------------------------------------------------------- #
# Weighted calibration / ranking metrics
# --------------------------------------------------------------------------- #
def _as_arrays(
    predictions: Sequence[EstimatorPrediction],
    samples: Sequence[LabeledSample],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(scores, labels, weights)`` float arrays for the samples."""
    scores = np.array([p.score for p in predictions], dtype=float)
    labels = np.array([1.0 if s.realized_contact else 0.0 for s in samples], dtype=float)
    weights = np.array([s.weight for s in samples], dtype=float)
    if weights.sum() <= 0.0:
        raise CalibrationInputError("total sample weight must be positive")
    return scores, labels, weights


def brier_score(scores: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> float:
    """Weighted Brier score (mean squared error of probabilistic forecasts).

    Returns:
        The weighted mean squared error between scores and labels.
    """
    return float((weights * (scores - labels) ** 2).sum() / weights.sum())


def log_loss(
    scores: np.ndarray, labels: np.ndarray, weights: np.ndarray, *, eps: float = 1e-6
) -> float:
    """Weighted log loss with probabilities clipped to ``[eps, 1 - eps]``.

    Returns:
        The weighted mean negative log likelihood.
    """
    clipped = np.clip(scores, eps, 1.0 - eps)
    terms = -(labels * np.log(clipped) + (1.0 - labels) * np.log(1.0 - clipped))
    return float((weights * terms).sum() / weights.sum())


def reliability_curve(
    scores: np.ndarray, labels: np.ndarray, weights: np.ndarray, *, n_bins: int
) -> list[dict[str, float]]:
    """Weighted equal-width reliability bins with counts, mean pred, and mean obs.

    Returns:
        One dict per non-empty bin with ``bin_lo``, ``bin_hi``, ``count``,
        ``weight``, ``mean_pred``, and ``mean_obs``.
    """
    if n_bins <= 0:
        raise CalibrationInputError("n_bins must be positive")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Rightmost edge inclusive so score == 1.0 lands in the last bin.
    bin_index = np.clip(np.digitize(scores, edges[1:-1], right=False), 0, n_bins - 1)
    curve: list[dict[str, float]] = []
    for b in range(n_bins):
        mask = bin_index == b
        if not mask.any():
            continue
        w = weights[mask]
        w_sum = float(w.sum())
        if w_sum <= 0.0:
            continue
        curve.append(
            {
                "bin_lo": float(edges[b]),
                "bin_hi": float(edges[b + 1]),
                "count": int(mask.sum()),
                "weight": w_sum,
                "mean_pred": float((w * scores[mask]).sum() / w_sum),
                "mean_obs": float((w * labels[mask]).sum() / w_sum),
            }
        )
    return curve


def expected_calibration_error(curve: list[dict[str, float]]) -> float:
    """Weighted expected calibration error from a reliability curve.

    Returns:
        The weight-averaged absolute gap between mean prediction and mean
        observation across bins, or ``nan`` when the curve is empty.
    """
    total = sum(bin_["weight"] for bin_ in curve)
    if total <= 0.0:
        return float("nan")
    return float(
        sum(bin_["weight"] * abs(bin_["mean_pred"] - bin_["mean_obs"]) for bin_ in curve) / total
    )


def average_precision(scores: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> float:
    """Weighted average precision (area under the precision-recall curve).

    Uses the step-interpolated definition ``AP = sum (R_k - R_{k-1}) * P_k`` over
    thresholds descending through the distinct scores.

    Returns:
        The weighted average precision, or ``nan`` when there are no positives.
    """
    total_pos = float((weights * labels).sum())
    if total_pos <= 0.0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    s_sorted = scores[order]
    l_sorted = labels[order]
    w_sorted = weights[order]

    ap = 0.0
    prev_recall = 0.0
    tp = 0.0
    fp = 0.0
    i = 0
    n = len(s_sorted)
    while i < n:
        # Consume all samples sharing the current threshold together.
        j = i
        while j < n and s_sorted[j] == s_sorted[i]:
            if l_sorted[j] > 0.5:
                tp += w_sorted[j]
            else:
                fp += w_sorted[j]
            j += 1
        recall = tp / total_pos
        precision = tp / (tp + fp) if (tp + fp) > 0.0 else 1.0
        ap += (recall - prev_recall) * precision
        prev_recall = recall
        i = j
    return float(ap)


def fnr_at_thresholds(
    scores: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    thresholds: Sequence[float],
) -> dict[str, float]:
    """Weighted false-negative rate ``P(score < thr | contact)`` per threshold.

    Returns:
        A mapping from formatted threshold to the weighted false-negative rate.
    """
    pos_mask = labels > 0.5
    pos_weight = float((weights * pos_mask).sum())
    result: dict[str, float] = {}
    for thr in thresholds:
        if pos_weight <= 0.0:
            result[f"{thr:.3f}"] = float("nan")
            continue
        missed = float((weights * pos_mask * (scores < thr)).sum())
        result[f"{thr:.3f}"] = missed / pos_weight
    return result


def time_to_warning_summary(
    predictions: Sequence[EstimatorPrediction],
    samples: Sequence[LabeledSample],
    *,
    dt_s: float,
) -> dict[str, float]:
    """Summarize warning timeliness on samples that realized a contact.

    Returns:
        ``warned_fraction`` (share of contacting samples the estimator warned on),
        ``mean_lead_time_s`` (mean seconds between warning and realized contact,
        over warned contacting samples; non-negative means warned in time), and
        ``n_contacts`` (denominator).
    """
    lead_times: list[float] = []
    n_contacts = 0
    n_warned = 0
    for pred, sample in zip(predictions, samples, strict=True):
        if not sample.realized_contact:
            continue
        n_contacts += 1
        if pred.warning_step < 0:
            continue
        n_warned += 1
        lead = (sample.realized_first_contact_step - pred.warning_step) * dt_s
        lead_times.append(lead)
    return {
        "n_contacts": n_contacts,
        "warned_fraction": (n_warned / n_contacts) if n_contacts else float("nan"),
        "mean_lead_time_s": (sum(lead_times) / len(lead_times)) if lead_times else float("nan"),
    }


def _bootstrap_ci(
    values_fn: Callable[[np.ndarray], float],
    n: int,
    *,
    n_boot: int,
    seed: int,
) -> tuple[float, float]:
    """Percentile bootstrap 95% CI of a statistic over sample indices.

    Returns:
        The ``(2.5th, 97.5th)`` percentile interval, or ``(nan, nan)`` when it
        cannot be computed.
    """
    if n_boot <= 0 or n == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stats[b] = values_fn(idx)
    finite = stats[np.isfinite(stats)]
    if finite.size == 0:
        return (float("nan"), float("nan"))
    return (float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5)))


# --------------------------------------------------------------------------- #
# Estimator adapters
# --------------------------------------------------------------------------- #
def _warning_step_from_first_passage(first_passage: tuple[float, ...], threshold: float) -> int:
    """First step where the cumulative first-passage probability crosses ``threshold``.

    Returns:
        The crossing step index, or ``-1`` when the threshold is never reached.
    """
    cumulative = 0.0
    for step, value in enumerate(first_passage):
        cumulative += value
        if cumulative >= threshold:
            return step
    return -1


def _predict_constant_velocity_mc(
    sample: LabeledSample, config: RiskEstimatorConfig, *, warn_threshold: float
) -> EstimatorPrediction:
    """Score a sample with the constant-velocity Monte Carlo estimator (probabilistic).

    Returns:
        The estimator prediction (joint contact probability as the score).
    """
    start_ns = time.perf_counter_ns()
    estimate = estimate_action_conditioned_risk(
        sample.action, sample.pedestrians, config, measure_latency=False
    )
    latency_ms = (time.perf_counter_ns() - start_ns) / 1e6
    warning_step = _warning_step_from_first_passage(
        estimate.first_passage_distribution, warn_threshold
    )
    return EstimatorPrediction(
        score=estimate.joint_contact_probability,
        is_probability=True,
        warning_step=warning_step,
        latency_ms=latency_ms,
    )


def _predict_deterministic_ttc(
    sample: LabeledSample, config: RiskEstimatorConfig, *, clearance_scale_m: float
) -> EstimatorPrediction:
    """Score a sample with the deterministic TTC/clearance warning (ranking only).

    The score is a monotone danger score derived from the noise-free minimum
    clearance -- it is **not** a probability and is never placed on the reliability
    curve. The warning step is the deterministic first-contact step when a contact
    is certain, else the first velocity-obstacle step, else no warning.

    Returns:
        The estimator prediction (monotone danger score; not a probability).
    """
    start_ns = time.perf_counter_ns()
    estimate = estimate_action_conditioned_risk(
        sample.action, sample.pedestrians, config, measure_latency=False
    )
    latency_ms = (time.perf_counter_ns() - start_ns) / 1e6
    det = estimate.deterministic
    clearance = det.min_clearance_m
    if not math.isfinite(clearance):
        score = 0.0
    else:
        # Logistic in negative clearance: overlap (clearance < 0) -> ~1, wide
        # clearance -> ~0. Monotone and bounded in [0, 1].
        score = 1.0 / (1.0 + math.exp(clearance / clearance_scale_m))
    if det.contact_certain:
        warning_step = det.first_contact_step
    elif any(det.velocity_obstacle_flags):
        warning_step = det.min_clearance_step
    else:
        warning_step = -1
    return EstimatorPrediction(
        score=float(score),
        is_probability=False,
        warning_step=int(warning_step),
        latency_ms=latency_ms,
    )


def predict_estimator(
    estimator_id: str,
    samples: Sequence[LabeledSample],
    config: RiskEstimatorConfig,
    *,
    warn_threshold: float = 0.5,
    clearance_scale_m: float = 0.5,
) -> list[EstimatorPrediction]:
    """Produce predictions for every sample with the named available estimator.

    Returns:
        One prediction per sample, in order.
    """
    if estimator_id == "constant_velocity_mc":
        return [
            _predict_constant_velocity_mc(s, config, warn_threshold=warn_threshold) for s in samples
        ]
    if estimator_id == "deterministic_ttc":
        return [
            _predict_deterministic_ttc(s, config, clearance_scale_m=clearance_scale_m)
            for s in samples
        ]
    raise CalibrationInputError(f"unknown or unavailable estimator: {estimator_id!r}")


# --------------------------------------------------------------------------- #
# Ground-truth (matched) dataset generation
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FamilySpec:
    """Ground-truth generative model + scenario ranges for one scenario family.

    The estimator always assumes ``config.velocity_std_m_s`` with zero bias. When
    ``gt_velocity_std_m_s`` or ``gt_velocity_bias_m_s`` differ from that, the
    family is *misspecified* and a well-built calibration harness must report a
    larger calibration error for it.
    """

    name: str
    n_scenarios: int
    gt_velocity_std_m_s: float
    gt_velocity_bias_m_s: tuple[float, float] = (0.0, 0.0)
    n_pedestrians: int = 4
    robot_speed_range: tuple[float, float] = (0.6, 1.4)
    ped_spawn_radius_m: float = 3.5
    ped_lateral_range: tuple[float, float] = (-2.5, 2.5)
    ped_inbound_vx_range: tuple[float, float] = (-0.7, 0.1)
    ped_vy_range: tuple[float, float] = (-0.4, 0.4)
    strata: dict[str, str] = field(default_factory=dict)


def _draw_scenario(
    family: FamilySpec, config: RiskEstimatorConfig, rng: np.random.Generator, index: int
) -> LabeledSample:
    """Draw one scenario (actors + action) and its single realized outcome.

    Returns:
        A labelled sample with the drawn action, actors, and realized contact.
    """
    # Robot drives along +x at a random speed; the action is deterministic.
    robot_speed = rng.uniform(*family.robot_speed_range)
    action = action_from_constant_velocity(
        f"{family.name}:{index}:drive",
        start_position=[0.0, 0.0],
        velocity=[robot_speed, 0.0],
        horizon_steps=config.horizon_steps,
        dt_s=config.dt_s,
    )

    # Pedestrians spawn ahead of the robot and move generally toward it, so that
    # collision likelihood spans a useful range across scenarios.
    pedestrians = []
    for pid in range(family.n_pedestrians):
        px = rng.uniform(0.8, family.ped_spawn_radius_m)
        py = rng.uniform(*family.ped_lateral_range)
        # Nominal (assumed-mean) velocity: drift toward the robot lane.
        vx = rng.uniform(*family.ped_inbound_vx_range)
        vy = rng.uniform(*family.ped_vy_range)
        pedestrians.append(
            PedestrianState(id=pid, position=np.array([px, py]), velocity=np.array([vx, vy]))
        )

    realized_contact, first_step = _realize_contact(action, pedestrians, family, config, rng)
    strata = {"family": family.name, **family.strata}
    return LabeledSample(
        scenario_id=f"{family.name}:{index}",
        family=family.name,
        action=action,
        pedestrians=tuple(pedestrians),
        realized_contact=realized_contact,
        realized_first_contact_step=first_step,
        weight=1.0,
        strata=strata,
    )


def _realize_contact(
    action: CandidateAction,
    pedestrians: Sequence[PedestrianState],
    family: FamilySpec,
    config: RiskEstimatorConfig,
    rng: np.random.Generator,
) -> tuple[bool, int]:
    """Draw one ground-truth realization and test the contact predicate.

    The realized pedestrian velocities are drawn from the *ground-truth* model of
    the family (which may be misspecified relative to the estimator), then rolled
    out at constant velocity. Contact uses the identical segment-min-distance
    predicate as the estimator, so labels and predictions are on the same
    geometry.

    Returns:
        ``(realized_contact, first_contact_step)`` with ``-1`` when no contact.
    """
    robot_xy = action.as_array(horizon_steps=config.horizon_steps)
    ped_pos, ped_vel, radii, _ids = pedestrian_arrays(pedestrians, config)
    radii_sum = radii + config.robot_radius_m
    n_actors = ped_pos.shape[0]
    if n_actors == 0:
        return False, -1

    bias = np.asarray(family.gt_velocity_bias_m_s, dtype=float).reshape(2)
    noise = rng.normal(0.0, family.gt_velocity_std_m_s, size=(n_actors, 2))
    realized_vel = ped_vel + bias[None, :] + noise
    steps = np.arange(config.horizon_steps + 1, dtype=float)
    realized_pos = (
        ped_pos[:, None, :] + steps[None, :, None] * config.dt_s * realized_vel[:, None, :]
    )  # (K, H+1, 2)
    seg_dist = segment_min_distance(robot_xy, realized_pos)  # (K, H)
    contact = seg_dist - radii_sum[:, None] <= 0.0  # (K, H)
    contact_any_step = contact.any(axis=0)  # (H,)
    if not bool(contact_any_step.any()):
        return False, -1
    return True, int(np.argmax(contact_any_step))


def generate_matched_dataset(
    families: Sequence[FamilySpec],
    config: RiskEstimatorConfig,
    provenance: MatchedDatasetProvenance,
) -> MatchedDataset:
    """Generate a reproducible matched labelled dataset from family specs.

    Returns:
        The matched dataset of labelled samples with its provenance.
    """
    rng = np.random.default_rng(provenance.seed)
    samples: list[LabeledSample] = []
    for family in families:
        for index in range(family.n_scenarios):
            samples.append(_draw_scenario(family, config, rng, index))
    if len(samples) != provenance.n_samples:
        raise CalibrationInputError(
            f"generated {len(samples)} samples but provenance declares "
            f"{provenance.n_samples}; keep them in sync"
        )
    return MatchedDataset(samples=tuple(samples), provenance=provenance)


# --------------------------------------------------------------------------- #
# Evaluation + verdict
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class VerdictThresholds:
    """Preregistered thresholds that map metrics to a per-estimator verdict."""

    ece_use_online_max: float = 0.05
    ece_revise_max: float = 0.15
    min_ap_over_prevalence: float = 1.5
    deadline_ms: float = 100.0


def _verdict(
    *,
    kind: str,
    ece: float,
    brier: float,
    baseline_brier: float,
    average_precision_value: float,
    prevalence: float,
    latency_classification: str,
    thresholds: VerdictThresholds,
) -> tuple[str, str]:
    """Return a ``(verdict, rationale)`` from the metrics and preregistered rules.

    Verdicts: ``use online`` / ``offline analysis only`` / ``revise`` / ``stop``.
    """
    online = latency_classification == "online"
    if kind == "probabilistic":
        if not math.isfinite(ece):
            return "revise", "calibration error could not be computed"
        if ece <= thresholds.ece_use_online_max and brier <= baseline_brier:
            if online:
                return (
                    "use online",
                    f"ECE {ece:.3f} <= {thresholds.ece_use_online_max}, Brier {brier:.3f} "
                    f"<= constant-prevalence baseline {baseline_brier:.3f}, latency online",
                )
            return (
                "offline analysis only",
                f"well-calibrated (ECE {ece:.3f}) but latency classified offline_only",
            )
        if ece <= thresholds.ece_revise_max:
            return (
                "revise",
                f"ECE {ece:.3f} in the revise band (<= {thresholds.ece_revise_max}) or Brier "
                f"{brier:.3f} above baseline {baseline_brier:.3f}",
            )
        return (
            "stop",
            f"ECE {ece:.3f} exceeds the revise band {thresholds.ece_revise_max}; miscalibrated",
        )
    # Warning (non-probabilistic) estimator: graded on ranking + timeliness.
    ratio = (
        average_precision_value / prevalence
        if prevalence > 0.0 and math.isfinite(average_precision_value)
        else float("nan")
    )
    if math.isfinite(ratio) and ratio >= thresholds.min_ap_over_prevalence:
        if online:
            return (
                "use online",
                f"warning ranking AP {average_precision_value:.3f} is {ratio:.2f}x the "
                f"prevalence baseline {prevalence:.3f}, latency online",
            )
        return "offline analysis only", "useful ranking but latency offline_only"
    return (
        "revise",
        f"warning ranking AP {average_precision_value:.3f} is not clearly above the "
        f"prevalence baseline {prevalence:.3f}",
    )


def evaluate_estimator(
    estimator_id: str,
    dataset: MatchedDataset,
    config: RiskEstimatorConfig,
    *,
    n_reliability_bins: int = 10,
    fnr_thresholds: Sequence[float] = (0.1, 0.25, 0.5),
    n_bootstrap: int = 200,
    thresholds: VerdictThresholds | None = None,
) -> dict[str, object]:
    """Evaluate one available estimator against the matched dataset.

    Returns:
        A JSON-safe result dict with calibration/ranking/timeliness metrics,
        latency summary, bootstrap intervals, stratification, and a preregistered
        verdict.
    """
    thresholds = thresholds or VerdictThresholds(deadline_ms=config.deadline_ms)
    samples = dataset.samples
    predictions = predict_estimator(estimator_id, samples, config)
    scores, labels, weights = _as_arrays(predictions, samples)
    prevalence = dataset.prevalence()
    kind = "probabilistic" if predictions[0].is_probability else "warning"

    ap = average_precision(scores, labels, weights)
    fnr = fnr_at_thresholds(scores, labels, weights, fnr_thresholds)
    timeliness = time_to_warning_summary(predictions, samples, dt_s=config.dt_s)

    latency = latency_summary_from_samples(
        [p.latency_ms for p in predictions], deadline_ms=config.deadline_ms
    )

    result: dict[str, object] = {
        "estimator_id": estimator_id,
        "kind": kind,
        "available": True,
        "n_samples": len(samples),
        "prevalence": prevalence,
        "average_precision": ap,
        "fnr_at_thresholds": fnr,
        "time_to_warning": timeliness,
        "latency": latency.to_dict(),
        "horizon_monotonicity": _horizon_monotonicity(estimator_id, samples, config),
    }

    baseline_brier = prevalence * (1.0 - prevalence)
    if kind == "probabilistic":
        curve = reliability_curve(scores, labels, weights, n_bins=n_reliability_bins)
        ece = expected_calibration_error(curve)
        brier = brier_score(scores, labels, weights)
        ll = log_loss(scores, labels, weights)

        def _brier_stat(idx: np.ndarray) -> float:
            return brier_score(scores[idx], labels[idx], weights[idx])

        def _ece_stat(idx: np.ndarray) -> float:
            sub_curve = reliability_curve(
                scores[idx], labels[idx], weights[idx], n_bins=n_reliability_bins
            )
            return expected_calibration_error(sub_curve)

        brier_ci = _bootstrap_ci(
            _brier_stat, len(samples), n_boot=n_bootstrap, seed=dataset.provenance.seed + 1
        )
        ece_ci = _bootstrap_ci(
            _ece_stat, len(samples), n_boot=n_bootstrap, seed=dataset.provenance.seed + 2
        )
        result.update(
            {
                "brier_score": brier,
                "brier_ci95": list(brier_ci),
                "baseline_brier": baseline_brier,
                "log_loss": ll,
                "reliability_curve": curve,
                "expected_calibration_error": ece,
                "ece_ci95": list(ece_ci),
            }
        )
        verdict, rationale = _verdict(
            kind=kind,
            ece=ece,
            brier=brier,
            baseline_brier=baseline_brier,
            average_precision_value=ap,
            prevalence=prevalence,
            latency_classification=str(latency.classification),
            thresholds=thresholds,
        )
    else:
        result["note"] = (
            "deterministic warning: graded on ranking (AP), FNR, and timeliness only; "
            "not placed on the probability reliability curve"
        )
        verdict, rationale = _verdict(
            kind=kind,
            ece=float("nan"),
            brier=float("nan"),
            baseline_brier=baseline_brier,
            average_precision_value=ap,
            prevalence=prevalence,
            latency_classification=str(latency.classification),
            thresholds=thresholds,
        )

    result["verdict"] = verdict
    result["verdict_rationale"] = rationale
    result["stratified"] = _stratified_metrics(
        scores,
        labels,
        weights,
        samples,
        kind=kind,
        dimensions=("family", "density", "prediction_model"),
        n_reliability_bins=n_reliability_bins,
        min_stratum=30,
    )
    return result


def _stratified_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    samples: Sequence[LabeledSample],
    *,
    kind: str,
    dimensions: Sequence[str],
    n_reliability_bins: int,
    min_stratum: int,
) -> dict[str, dict[str, dict[str, object]]]:
    """Compute compact calibration/ranking metrics per stratum value.

    For each dimension (family, density, prediction model, ...) samples are
    grouped by their stratum tag. Groups with fewer than ``min_stratum`` samples
    are marked ``unavailable`` (insufficient denominator) rather than reported,
    per the #5445 stratification criterion.

    Returns:
        A nested mapping ``dimension -> stratum_value -> metrics``.
    """
    out: dict[str, dict[str, dict[str, object]]] = {}
    for dim in dimensions:
        groups: dict[str, list[int]] = {}
        for i, sample in enumerate(samples):
            key = sample.strata.get(dim) if dim != "family" else sample.family
            if key is None:
                continue
            groups.setdefault(str(key), []).append(i)
        if not groups:
            continue
        dim_out: dict[str, dict[str, object]] = {}
        for key, idx_list in sorted(groups.items()):
            idx = np.asarray(idx_list, dtype=int)
            if idx.size < min_stratum:
                dim_out[key] = {
                    "n": int(idx.size),
                    "status": "unavailable_insufficient_denominator",
                }
                continue
            s, y, w = scores[idx], labels[idx], weights[idx]
            entry: dict[str, object] = {
                "n": int(idx.size),
                "prevalence": float((w * y).sum() / w.sum()),
                "average_precision": average_precision(s, y, w),
            }
            if kind == "probabilistic":
                curve = reliability_curve(s, y, w, n_bins=n_reliability_bins)
                entry["expected_calibration_error"] = expected_calibration_error(curve)
                entry["brier_score"] = brier_score(s, y, w)
            dim_out[key] = entry
        out[dim] = dim_out
    return out


def _horizon_monotonicity(
    estimator_id: str, samples: Sequence[LabeledSample], config: RiskEstimatorConfig
) -> dict[str, float]:
    """Fraction of samples whose predicted contact CDF is monotone in the horizon.

    Only meaningful for the probabilistic estimator, whose first-passage CDF must
    be non-decreasing; the deterministic warning has no probability CDF and is
    reported as ``not_applicable``.

    Returns:
        A mapping with ``applicable`` and the monotone-CDF ``monotone_fraction``.
    """
    if estimator_id != "constant_velocity_mc":
        return {"applicable": 0.0, "monotone_fraction": float("nan")}
    monotone = 0
    for sample in samples:
        estimate = estimate_action_conditioned_risk(
            sample.action, sample.pedestrians, config, measure_latency=False
        )
        cdf = np.cumsum(estimate.first_passage_distribution)
        if np.all(np.diff(cdf) >= -1e-9):
            monotone += 1
    return {
        "applicable": 1.0,
        "monotone_fraction": monotone / len(samples) if samples else float("nan"),
    }


def action_sensitivity(
    dataset: MatchedDataset,
    config: RiskEstimatorConfig,
    action_pairs: Sequence[tuple[str, str]],
) -> dict[str, object]:
    """Check predicted risk differs in the expected direction across action pairs.

    Each pair is ``(higher_risk_scenario_id, lower_risk_scenario_id)`` sharing the
    same actor state. Evaluated with the constant-velocity MC estimator.

    Returns:
        A mapping with the number of pairs, the fraction ordered as expected, and
        per-pair details.
    """
    by_id = {s.scenario_id: s for s in dataset.samples}
    correct = 0
    evaluated = 0
    details = []
    for higher_id, lower_id in action_pairs:
        if higher_id not in by_id or lower_id not in by_id:
            continue
        evaluated += 1
        higher = predict_estimator("constant_velocity_mc", [by_id[higher_id]], config)[0]
        lower = predict_estimator("constant_velocity_mc", [by_id[lower_id]], config)[0]
        ok = higher.score > lower.score
        correct += int(ok)
        details.append(
            {
                "higher_risk": higher_id,
                "lower_risk": lower_id,
                "higher_score": higher.score,
                "lower_score": lower.score,
                "direction_as_expected": ok,
            }
        )
    return {
        "n_pairs": evaluated,
        "fraction_correct": (correct / evaluated) if evaluated else float("nan"),
        "pairs": details,
    }


__all__ = [
    "AVAILABLE_ESTIMATORS",
    "CALIBRATION_SCHEMA_VERSION",
    "CLAIM_BOUNDARY",
    "UNAVAILABLE_ESTIMATORS",
    "CalibrationInputError",
    "EstimatorPrediction",
    "FamilySpec",
    "LabeledSample",
    "MatchedDataset",
    "MatchedDatasetProvenance",
    "VerdictThresholds",
    "action_sensitivity",
    "average_precision",
    "brier_score",
    "evaluate_estimator",
    "expected_calibration_error",
    "fnr_at_thresholds",
    "generate_matched_dataset",
    "log_loss",
    "predict_estimator",
    "reliability_curve",
    "time_to_warning_summary",
]
