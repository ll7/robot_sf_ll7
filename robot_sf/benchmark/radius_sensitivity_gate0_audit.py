"""Gate 0 post-hoc feasibility audit for the collision-envelope radius campaign (issue #6640).

This module records the *post-hoc-versus-replay boundary* for the radius-sensitivity campaign
defined in parent issue #6600. Gate 0 inspects the frozen ``0.0.3.post1`` release episode rows
and the metric contract, then emits a machine-readable decision that classifies each
radius-sensitivity outcome as either:

- ``re-derivable`` -- the outcome at a new radius is an exact deterministic function of fields
  retained in the frozen release rows plus the radius metadata, under the frozen metric
  semantics, *and* it is not a trajectory-dependent planner/obstacle-contact/feasibility/collision
  outcome; or
- ``replay-required`` -- the outcome depends on the radius-arm trajectory (which differs across
  arms because the collision-envelope radius changes planner behaviour and simulator collision
  geometry), on per-timestep geometry that the aggregate frozen rows do not retain, or on
  effective radius/map provenance that the frozen release does not pin.

The module is deliberately pure and diagnostic. It does **not** run benchmark episodes, does not
change any frozen ``0.0.3.post1`` metric semantics, release config, or manifest, does not run any
production compute, and does not establish a planner ranking or a radius-sensitivity result. It
only records the decision boundary that Gate 1 (binding canary) and Gate 2 (production sweep)
must respect. Because the frozen release provenance does not retain the effective radius or pin
the map asset bytes, this audit intentionally emits no re-derivable outcome.

Stop conditions enforced programmatically by :func:`validate_gate0_decision`:

1. Trajectory-dependent planner behaviour, obstacle contact, feasibility, and collision outcomes
   are always ``replay-required``.
2. A re-derivable outcome requires its source geometry and semantics to be retained exactly.
3. The decision lists every outcome with exactly one classification.
4. Threshold reclassification of static geometry does not, by itself, infer a full radius sweep.
5. An unretained effective radius or unpinned map asset is not enough evidence for a re-derivable
   outcome.

See parent issue #6600 (Gate 0 spec), validity study #3207 (clearance-semantics foundation in
:mod:`robot_sf.benchmark.clearance_semantics`), and the frozen release pointer under
``docs/context/evidence/issue_4364_release_0_0_3_post1/``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from robot_sf.benchmark.constants import COLLISION_DIST, NEAR_MISS_DIST
from robot_sf.evidence.writers import write_json

GATE0_DECISION_SCHEMA = "radius_sensitivity_gate0_decision.v1"
GATE0_REVIEW_MARKER = "AI-GENERATED NEEDS-REVIEW"

# Campaign axis is the robot collision-envelope (planning proxy) radius in metres. The frozen
# ``0.0.3.post1`` baseline arm is 1.0 m; the sensitivity arms are 0.5 m and 0.8 m. See parent
# issue #6600 and the dissertation ruling referenced there.
CAMPAIGN_AXIS = "robot_collision_envelope_radius_m"
CAMPAIGN_BASELINE_ARM_M = 1.0
CAMPAIGN_ARMS_M: tuple[float, ...] = (0.5, 0.8, 1.0)

# Frozen ``0.0.3.post1`` release provenance. Sourced from the tracked artifact pointer and the
# diagnostic-only collision-consumer reconciliation; the episode rows themselves live in the
# attached release bundle, not in git.
FROZEN_RELEASE_TAG = "0.0.3.post1"
FROZEN_RELEASE_EPISODE_ROWS = 20160
FROZEN_RELEASE_ARMS = 14
FROZEN_RELEASE_ROWS_PER_ARM = 1440
FROZEN_RELEASE_EXECUTION_COMMIT = "a307ef276d701f8d14dead1aa0513f44ee97c0b0"
FROZEN_RELEASE_BUNDLE_SHA256 = "9bf6ea35a17ce812f0a9c841c3681bc072dcf7ba8c121cbcf05113b8514f4de1"
FROZEN_RELEASE_POINTER = (
    "docs/context/evidence/issue_4364_release_0_0_3_post1/artifact_pointer.json"
)
FROZEN_RELEASE_COLLISION_RECONCILIATION = (
    "docs/context/evidence/issue_4364_release_0_0_3_post1/collision_reconciliation.json"
)

# Metric contract grounding (see robot_sf/benchmark/metrics.py and constants.py).
#
# The collision-envelope radius binds the robot-pedestrian *clearance* family only:
#   clearance[t, k] = center_distance[t, k] - (robot_radius_m + ped_radius_m)
# Negative clearance is contact; ``human_collisions`` counts timesteps with negative minimum
# clearance, and ``near_misses`` counts timesteps in ``[0, NEAR_MISS_DIST)``.
RADIUS_AWARE_CLEARANCE_METRICS: tuple[str, ...] = (
    "human_collisions",
    "near_misses",
    "min_clearance",
    "mean_clearance",
)
CLEARANCE_FORMULA = "clearance[t,k] = center_distance[t,k] - (robot_radius_m + ped_radius_m)"

# Wall and agent collision predicates use the *fixed* ``COLLISION_DIST`` centre-distance
# threshold against point obstacles/agents; they do not subtract the collision-envelope radius
# in the metric formula. The radius still binds the *simulator* collision geometry during
# trajectory generation, so these counts remain trajectory-dependent across radius arms.
FIXED_THRESHOLD_COLLISION_METRICS: tuple[str, ...] = ("wall_collisions", "agent_collisions")

# Geometry metrics whose formula never references a radius. They are radius-independent for a
# fixed trajectory but still differ across radius arms because the trajectory itself changes.
RADIUS_INDEPENDENT_GEOMETRY_METRICS: tuple[str, ...] = (
    "clearing_distance_min",
    "clearing_distance_avg",
    "min_distance",
    "mean_distance",
    "robot_ped_within_5m_frac",
)

# The collision-envelope radius default is not uniform across the metric contract. This is a
# Gate 0 finding: post-hoc reclassification must confirm the per-row effective radius (from
# retained metadata / frozen config) before use; it must never assume a single default. Gate 1
# (binding canary) resolves the effective runtime binding.
METRICS_EPISODE_DATA_DEFAULT_ROBOT_RADIUS_M = 1.0
METRICS_EPISODE_DATA_DEFAULT_PED_RADIUS_M = 0.4
RUNNER_DEFAULT_ROBOT_RADIUS_M = 0.3
RUNNER_DEFAULT_PED_RADIUS_M = 0.35

CLAIM_BOUNDARY = (
    "radius_sensitivity_gate0_decision_diagnostic_only: a machine-readable decision recording the "
    "post-hoc-versus-replay boundary for the collision-envelope radius campaign. It does not run "
    "benchmark episodes, change frozen 0.0.3.post1 metric semantics, configs, or manifests, run "
    "production compute, or establish a radius-sensitivity result, planner ranking, feasibility "
    "verdict, or paper-facing benchmark evidence. No current outcome qualifies as re-derivable "
    "because the effective radius and map asset provenance are not retained/pinned; this decision "
    "must not be read as a radius sweep."
)

# Outcome categories that are unconditionally replay-required by stop condition #3.
TRAJECTORY_DEPENDENT_CATEGORIES = frozenset(
    {
        "scalar_metric_clearance_family",
        "scalar_metric_fixed_threshold",
        "scalar_metric_radius_independent_geometry",
        "binary_success",
        "aggregate_collision_count",
        "simulator_contact_outcome",
        "trajectory_feasibility_outcome",
        "planner_behaviour_outcome",
        "planner_ranking_outcome",
        "scenario_family_outcome",
        "scalar_metric_trajectory_dependent",
    }
)

RE_DERIVABLE = "re-derivable"
REPLAY_REQUIRED = "replay-required"
VALID_CLASSIFICATIONS = frozenset({RE_DERIVABLE, REPLAY_REQUIRED})


@dataclass(frozen=True)
class RadiusSensitivityOutcome:
    """One radius-sensitivity outcome and its Gate 0 classification."""

    outcome_id: str
    outcome: str
    category: str
    radius_binding: str
    source_geometry_retained_in_frozen_rows: bool
    is_collision_contact_feasibility_or_planner_outcome: bool
    classification: str
    rationale: str
    caveats: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the outcome."""
        payload = asdict(self)
        payload["caveats"] = list(self.caveats)
        return payload


def _clearance_family_outcomes() -> tuple[RadiusSensitivityOutcome, ...]:
    """Radius-aware clearance-family metrics; all replay-required.

    ``min_clearance``/``mean_clearance`` shift linearly with the radius *for one fixed
    trajectory*, but the radius-arm trajectory itself differs across arms, so the cross-arm value
    is not re-derivable from the retained baseline aggregate. ``human_collisions`` and
    ``near_misses`` are threshold counts over a per-timestep clearance distribution that the
    aggregate frozen rows do not retain, so they require the full replay trajectory (see
    ``robot_sf/benchmark/threshold_sensitivity.py``, which recomputes these counts from
    ``replay_steps``/``replay_peds`` and never from aggregate rows).

    Returns:
        Tuple of replay-required radius-aware clearance-family outcomes.
    """
    common_caveat = (
        "Cross-arm value is trajectory-dependent: the radius changes planner behaviour and "
        "simulator collision geometry, so the 0.5 m / 0.8 m arm trajectory differs from the "
        "retained 1.0 m baseline trajectory."
    )
    return (
        RadiusSensitivityOutcome(
            outcome_id="human_collisions_count",
            outcome="Per-episode pedestrian (human) collision count",
            category="scalar_metric_clearance_family",
            radius_binding="clearance_matrix",
            source_geometry_retained_in_frozen_rows=False,
            is_collision_contact_feasibility_or_planner_outcome=True,
            classification=REPLAY_REQUIRED,
            rationale=(
                "Counts timesteps where min clearance < 0. Reclassifying the count at a new radius "
                "needs the per-timestep clearance distribution; only the aggregate count is "
                "retained. Collision outcomes are replay-required by stop condition #3."
            ),
            caveats=(
                common_caveat,
                "threshold_sensitivity.near_miss_count / human_collisions require full replay "
                "trajectories, not aggregate rows.",
            ),
        ),
        RadiusSensitivityOutcome(
            outcome_id="near_misses_count",
            outcome="Per-episode pedestrian near-miss count",
            category="scalar_metric_clearance_family",
            radius_binding="clearance_matrix",
            source_geometry_retained_in_frozen_rows=False,
            is_collision_contact_feasibility_or_planner_outcome=True,
            classification=REPLAY_REQUIRED,
            rationale=(
                "Counts timesteps where 0 <= min clearance < NEAR_MISS_DIST. A threshold-count "
                "over a non-retained per-timestep distribution; not re-derivable from the "
                "retained aggregate."
            ),
            caveats=(common_caveat,),
        ),
        RadiusSensitivityOutcome(
            outcome_id="min_clearance_scalar",
            outcome="Per-episode minimum robot-pedestrian surface clearance",
            category="scalar_metric_clearance_family",
            radius_binding="clearance_matrix",
            source_geometry_retained_in_frozen_rows=False,
            is_collision_contact_feasibility_or_planner_outcome=False,
            classification=REPLAY_REQUIRED,
            rationale=(
                "For one fixed trajectory, min_clearance shifts by -(delta_robot_radius) exactly. "
                "Across radius arms the centre-distance trajectory differs, so the cross-arm value "
                "is not re-derivable from the retained baseline aggregate."
            ),
            caveats=(
                common_caveat,
                "The within-trajectory linear shift is a flown-baseline diagnostic only, not a "
                "cross-arm re-derivation.",
            ),
        ),
        RadiusSensitivityOutcome(
            outcome_id="mean_clearance_scalar",
            outcome="Per-episode mean robot-pedestrian surface clearance",
            category="scalar_metric_clearance_family",
            radius_binding="clearance_matrix",
            source_geometry_retained_in_frozen_rows=False,
            is_collision_contact_feasibility_or_planner_outcome=False,
            classification=REPLAY_REQUIRED,
            rationale=(
                "Mean of per-step minimum clearance; cross-arm value is trajectory-dependent and "
                "not re-derivable from the retained baseline aggregate."
            ),
            caveats=(common_caveat,),
        ),
    )


def _fixed_threshold_collision_outcomes() -> tuple[RadiusSensitivityOutcome, ...]:
    """Wall/agent collision counts use a fixed threshold but are still trajectory-dependent.

    Returns:
        Tuple of replay-required fixed-threshold collision-count outcomes.
    """
    caveat = (
        "The metric formula uses the fixed COLLISION_DIST centre-distance threshold and does not "
        "subtract the collision-envelope radius, but the radius binds the simulator collision "
        "geometry during trajectory generation, so the count still differs across radius arms."
    )
    outcomes: list[RadiusSensitivityOutcome] = []
    for metric, label in (
        ("wall_collisions", "Per-episode wall/obstacle collision count"),
        ("agent_collisions", "Per-episode other-agent collision count"),
    ):
        outcomes.append(
            RadiusSensitivityOutcome(
                outcome_id=f"{metric}_count",
                outcome=label,
                category="scalar_metric_fixed_threshold",
                radius_binding="fixed_threshold_metric_radius_aware_simulator",
                source_geometry_retained_in_frozen_rows=False,
                is_collision_contact_feasibility_or_planner_outcome=True,
                classification=REPLAY_REQUIRED,
                rationale=(
                    f"{metric} uses the fixed {COLLISION_DIST} m centre-distance threshold against "
                    "point obstacles/agents. The count is trajectory-dependent across radius arms "
                    "and is a collision outcome (stop condition #3)."
                ),
                caveats=(caveat,),
            )
        )
    return tuple(outcomes)


def _radius_independent_geometry_outcomes() -> tuple[RadiusSensitivityOutcome, ...]:
    """Geometry metrics with no radius term; cross-arm value still trajectory-dependent.

    Returns:
        Tuple of replay-required radius-independent-geometry metric outcomes.
    """
    rows = (
        ("clearing_distance_min", "Per-episode minimum robot-centre-to-obstacle-point distance"),
        ("clearing_distance_avg", "Per-episode mean robot-centre-to-obstacle-point distance"),
        ("min_distance", "Per-episode minimum robot-pedestrian centre-to-centre distance"),
        ("mean_distance", "Per-episode mean robot-pedestrian centre-to-centre distance"),
        ("robot_ped_within_5m_frac", "Per-episode fraction of steps within 5 m of a pedestrian"),
    )
    outcomes: list[RadiusSensitivityOutcome] = []
    for metric, label in rows:
        cd_caveat = (
            "The retained value is a radius-independent proximity bound on the flown 1.0 m "
            "baseline trajectory only; it does not reconstruct the 0.5 m / 0.8 m arm value."
            if metric.startswith("clearing_distance")
            else (
                "Centre-to-centre distance is radius-independent in formula; the cross-arm value "
                "still differs because the trajectory differs."
            )
        )
        outcomes.append(
            RadiusSensitivityOutcome(
                outcome_id=f"{metric}_scalar",
                outcome=label,
                category="scalar_metric_radius_independent_geometry",
                radius_binding="none_in_formula_trajectory_via_simulator",
                source_geometry_retained_in_frozen_rows=False,
                is_collision_contact_feasibility_or_planner_outcome=False,
                classification=REPLAY_REQUIRED,
                rationale=(
                    f"{metric} has no radius term in its formula, but the value is computed on the "
                    "radius-arm trajectory, which differs across arms; not re-derivable from the "
                    "retained baseline aggregate."
                ),
                caveats=(cd_caveat,),
            )
        )
    return tuple(outcomes)


def _success_and_aggregate_outcomes() -> tuple[RadiusSensitivityOutcome, ...]:
    """Binary success and aggregate collision counts; replay-required.

    Returns:
        Tuple of replay-required success and aggregate collision-count outcomes.
    """
    return (
        RadiusSensitivityOutcome(
            outcome_id="binary_success",
            outcome="Per-episode binary success (reached goal AND zero collisions)",
            category="binary_success",
            radius_binding="collision_gate_clearance_plus_timing_gate",
            source_geometry_retained_in_frozen_rows=False,
            is_collision_contact_feasibility_or_planner_outcome=True,
            classification=REPLAY_REQUIRED,
            rationale=(
                "Success gates on total_collision_count == 0 (radius-aware via human_collisions) "
                "AND reached_goal_step < horizon. The collision gate is replay-required; the "
                "frozen collision reconciliation also warns the bundle lacks reached_goal_step / "
                "horizon inputs to recompute the timing gate."
            ),
            caveats=(
                "The frozen release collision reconciliation records one row with both "
                "goal_reached and timeout and warns the timing-gate inputs are not retained.",
            ),
        ),
        RadiusSensitivityOutcome(
            outcome_id="total_collision_count",
            outcome="Per-episode total collision count (pedestrian + wall + agent)",
            category="aggregate_collision_count",
            radius_binding="aggregate_of_clearance_and_fixed_threshold_counts",
            source_geometry_retained_in_frozen_rows=False,
            is_collision_contact_feasibility_or_planner_outcome=True,
            classification=REPLAY_REQUIRED,
            rationale=(
                "Aggregates human_collisions (radius-aware), wall_collisions and agent_collisions "
                "(fixed threshold). All three are trajectory-dependent collision outcomes."
            ),
            caveats=(),
        ),
        RadiusSensitivityOutcome(
            outcome_id="ped_collision_count",
            outcome="Per-episode pedestrian collision count (alias of human_collisions)",
            category="aggregate_collision_count",
            radius_binding="clearance_matrix",
            source_geometry_retained_in_frozen_rows=False,
            is_collision_contact_feasibility_or_planner_outcome=True,
            classification=REPLAY_REQUIRED,
            rationale=(
                "Stored on the row as ped_collision_count and equal to human_collisions; a "
                "radius-aware collision outcome (stop condition #3)."
            ),
            caveats=(),
        ),
    )


def _trajectory_simulator_planner_outcomes() -> tuple[RadiusSensitivityOutcome, ...]:
    """Trajectory-, simulator-, planner-, and family-level outcomes; all replay-required.

    Returns:
        Tuple of replay-required trajectory/simulator/planner/family outcomes.
    """
    rows = (
        (
            "simulator_obstacle_contact",
            "Simulator-level obstacle (wall) contact during trajectory generation",
            "simulator_contact_outcome",
            "The collision-envelope radius binds simulator collision geometry; contact is a "
            "trajectory-dependent collision/contact outcome (stop condition #3).",
        ),
        (
            "geometric_body_pedestrian_contact",
            "Physical body pedestrian contact (geometric-body clearance <= contact threshold)",
            "simulator_contact_outcome",
            "Physical contact is a trajectory-dependent contact outcome (stop condition #3).",
        ),
        (
            "trajectory_feasibility_traversal_executed",
            "Whether a collision-free traversal was actually executed to the goal",
            "trajectory_feasibility_outcome",
            "Executed-traversal feasibility is trajectory-dependent (stop condition #3); the "
            "static-map margin reclassification does not reconstruct it.",
        ),
        (
            "planner_behaviour_decisions",
            "Planner decisions / control actions along the trajectory",
            "planner_behaviour_outcome",
            "The radius changes planner inputs and behaviour; planner decisions are "
            "trajectory-dependent (stop condition #3).",
        ),
        (
            "planner_rankings_success_typed_collisions_snqi",
            "Planner rankings on success, typed collisions, and SNQI",
            "planner_ranking_outcome",
            "Rankings aggregate radius-arm episodes; re-deriving them requires the per-arm "
            "trajectories (stop condition #3).",
        ),
        (
            "scenario_family_conclusions_transitions",
            "Scenario-family conclusions and transitions (e.g. narrow-doorway family)",
            "scenario_family_outcome",
            "Family-level transitions depend on per-arm episode outcomes and are replay-required.",
        ),
        (
            "snqi_per_episode",
            "Per-episode Social Navigation Quality Index (SNQI)",
            "scalar_metric_trajectory_dependent",
            "SNQI consumes collision/clearance and trajectory metrics; re-deriving the cross-arm "
            "value requires the radius-arm trajectory.",
        ),
        (
            "kinematic_efficiency_metrics",
            (
                "Kinematic / efficiency metrics (path efficiency, speed, jerk, curvature, energy, "
                "force quantiles, etc.)"
            ),
            "scalar_metric_trajectory_dependent",
            "Computed on the radius-arm trajectory; cross-arm values are trajectory-dependent.",
        ),
    )
    outcomes: list[RadiusSensitivityOutcome] = []
    for outcome_id, label, category, rationale in rows:
        outcomes.append(
            RadiusSensitivityOutcome(
                outcome_id=outcome_id,
                outcome=label,
                category=category,
                radius_binding="trajectory_via_simulator_and_planner",
                source_geometry_retained_in_frozen_rows=False,
                is_collision_contact_feasibility_or_planner_outcome=True,
                classification=REPLAY_REQUIRED,
                rationale=rationale,
                caveats=(),
            )
        )
    return tuple(outcomes)


def _provenance_blocked_outcomes() -> tuple[RadiusSensitivityOutcome, ...]:
    """Classify the tempting post-hoc diagnostics as replay-required.

    The prior implementation treated these two diagnostics as re-derivable. The frozen release
    manifest does not record the effective per-row robot/pedestrian radius, and its scenario
    matrix checksum does not pin the referenced map asset bytes. Until those provenance gaps are
    closed, neither diagnostic satisfies the exact-retention rule.

    Returns:
        Tuple of the two provenance-blocked diagnostic outcomes.
    """
    return (
        RadiusSensitivityOutcome(
            outcome_id="retained_radius_and_threshold_parameters",
            outcome=(
                "Retained radius / threshold parameters (robot_radius, ped_radius, COLLISION_DIST, "
                "NEAR_MISS_DIST) recoverable from the frozen config and metric constants"
            ),
            category="metadata_parameter",
            radius_binding="parameter_value",
            source_geometry_retained_in_frozen_rows=False,
            is_collision_contact_feasibility_or_planner_outcome=False,
            classification=REPLAY_REQUIRED,
            rationale=(
                "The frozen metric constants are known, but the effective per-row robot/pedestrian "
                "radius is not retained: the release config does not declare it and the metric "
                "and runner defaults disagree. The effective parameter provenance must be recovered "
                "before any post-hoc reclassification, so this outcome is replay-required."
            ),
            caveats=(
                "The collision-envelope radius default is not uniform across the metric contract "
                f"(metrics.py EpisodeData default robot_radius={METRICS_EPISODE_DATA_DEFAULT_ROBOT_RADIUS_M} m, "
                f"ped_radius={METRICS_EPISODE_DATA_DEFAULT_PED_RADIUS_M} m; runner.py default "
                f"robot_radius={RUNNER_DEFAULT_ROBOT_RADIUS_M} m, ped_radius={RUNNER_DEFAULT_PED_RADIUS_M} m). "
                "Gate 1 must confirm the per-row effective radius before any reclassification.",
                "A metric constant or default value is not evidence of the effective value used by "
                "the frozen release rows.",
            ),
        ),
        RadiusSensitivityOutcome(
            outcome_id="static_map_geometry_feasibility_margin",
            outcome=(
                "Static map-geometry feasibility margin (e.g. doorway gap minus swept diameter "
                "2*radius) on frozen map geometry"
            ),
            category="static_geometry_diagnostic",
            radius_binding="swept_envelope_parameter_on_frozen_geometry",
            source_geometry_retained_in_frozen_rows=False,
            is_collision_contact_feasibility_or_planner_outcome=False,
            classification=REPLAY_REQUIRED,
            rationale=(
                "A static map margin would be a radius-only reparameterisation if the exact map "
                "asset were retained. The release manifest hashes the scenario matrix but does not "
                "pin the referenced map asset bytes, and the frozen episode rows do not contain the "
                "map geometry. The exact geometry provenance must be recovered before this margin "
                "can be classified, so it is replay-required."
            ),
            caveats=(
                "The scenario matrix checksum is not a checksum of its included map assets.",
                "Threshold reclassification of static geometry does NOT, by itself, infer a full "
                "radius sweep (stop condition #4).",
                "A positive static margin would not reconstruct scripted-traversal or planner "
                "feasibility, which remain replay-required (the #5574 0.5 m probe reclassifies the "
                "doorway as solvable yet its scripted traversal still collided).",
            ),
        ),
    )


def build_outcome_registry() -> tuple[RadiusSensitivityOutcome, ...]:
    """Return the full deterministic Gate 0 outcome registry."""
    return (
        *_clearance_family_outcomes(),
        *_fixed_threshold_collision_outcomes(),
        *_radius_independent_geometry_outcomes(),
        *_success_and_aggregate_outcomes(),
        *_trajectory_simulator_planner_outcomes(),
        *_provenance_blocked_outcomes(),
    )


def _campaign_block() -> dict[str, Any]:
    return {
        "axis": CAMPAIGN_AXIS,
        "arms_m": list(CAMPAIGN_ARMS_M),
        "baseline_arm_m": CAMPAIGN_BASELINE_ARM_M,
        "scenario_surface": "classic_interactions_francis2023_48_cells",
        "planners": "complete_14_planner_release_roster",
        "seeds": "paper_eval_s30 (111-140)",
        "horizon_steps": 600,
    }


def _frozen_release_block() -> dict[str, Any]:
    return {
        "release_tag": FROZEN_RELEASE_TAG,
        "episode_rows": FROZEN_RELEASE_EPISODE_ROWS,
        "arms": FROZEN_RELEASE_ARMS,
        "rows_per_arm": FROZEN_RELEASE_ROWS_PER_ARM,
        "execution_commit": FROZEN_RELEASE_EXECUTION_COMMIT,
        "bundle_sha256": FROZEN_RELEASE_BUNDLE_SHA256,
        "artifact_pointer": FROZEN_RELEASE_POINTER,
        "collision_reconciliation_pointer": FROZEN_RELEASE_COLLISION_RECONCILIATION,
        "row_location": (
            "Episode rows live in the attached release bundle, not in git; this audit inspects the "
            "row schema and metric contract, not the bundle bytes."
        ),
    }


def _metric_contract_block() -> dict[str, Any]:
    return {
        "radius_aware_clearance_metrics": list(RADIUS_AWARE_CLEARANCE_METRICS),
        "clearance_formula": CLEARANCE_FORMULA,
        "fixed_threshold_collision_metrics": {
            metric: {"threshold_m": COLLISION_DIST, "uses_radius_in_formula": False}
            for metric in FIXED_THRESHOLD_COLLISION_METRICS
        },
        "radius_independent_geometry_metrics": list(RADIUS_INDEPENDENT_GEOMETRY_METRICS),
        "collision_constants": {
            "COLLISION_DIST_m": COLLISION_DIST,
            "NEAR_MISS_DIST_m": NEAR_MISS_DIST,
        },
        "radius_default_inconsistency": {
            "metrics_episode_data_default_robot_radius_m": METRICS_EPISODE_DATA_DEFAULT_ROBOT_RADIUS_M,
            "metrics_episode_data_default_ped_radius_m": METRICS_EPISODE_DATA_DEFAULT_PED_RADIUS_M,
            "runner_default_robot_radius_m": RUNNER_DEFAULT_ROBOT_RADIUS_M,
            "runner_default_ped_radius_m": RUNNER_DEFAULT_PED_RADIUS_M,
            "finding": (
                "Collision-envelope radius default is not uniform across the metric contract; "
                "Gate 1 must confirm the per-row effective radius before any post-hoc "
                "reclassification."
            ),
        },
        "frozen_provenance_gaps": {
            "effective_robot_and_pedestrian_radius_retained": False,
            "map_asset_bytes_pinned": False,
            "finding": (
                "The frozen release manifest does not declare the effective robot/pedestrian radius "
                "and its scenario matrix checksum does not pin the referenced map asset bytes; "
                "therefore neither radius metadata nor static-map geometry is re-derivable from the "
                "retained release fields."
            ),
        },
        "threshold_sensitivity_requires_replay": (
            "robot_sf/benchmark/threshold_sensitivity.py recomputes near-miss/comfort counts from "
            "full replay trajectories (replay_steps/replay_peds), never from aggregate frozen rows."
        ),
    }


def _rubric_block() -> dict[str, Any]:
    return {
        "re_derivable": (
            "The outcome at a new radius is an exact deterministic function of fields retained in "
            "the frozen release rows plus radius metadata and pinned source assets, under the "
            "frozen metric semantics, "
            "AND it is not a trajectory-dependent planner/obstacle-contact/feasibility/collision "
            "outcome, AND it is not a threshold-count over a non-retained per-timestep distribution."
        ),
        "replay_required": (
            "The outcome depends on the radius-arm trajectory (which differs across arms because "
            "the collision-envelope radius changes planner behaviour and simulator collision "
            "geometry), on per-timestep geometry the aggregate frozen rows do not retain, or on "
            "effective radius/map provenance that the frozen release does not pin."
        ),
        "stop_conditions": [
            "Do not modify frozen 0.0.3.post1 metric semantics or any release config or manifest.",
            "Do not run any production SLURM compute.",
            "Post-hoc reclassification is allowed only for quantities whose source geometry and "
            "semantics are retained exactly; trajectory-dependent planner behaviour, obstacle "
            "contact, feasibility, and collision outcomes must be classified replay-required.",
            "The output lists every outcome as re-derivable or replay-required.",
            "Do not infer a full sweep from threshold reclassification alone.",
            "Do not classify an outcome as re-derivable when effective radius or map asset bytes "
            "are not retained and pinned by the frozen release provenance.",
        ],
    }


def _summary_block(outcomes: tuple[RadiusSensitivityOutcome, ...]) -> dict[str, Any]:
    re_derivable = [o.outcome_id for o in outcomes if o.classification == RE_DERIVABLE]
    replay_required = [o.outcome_id for o in outcomes if o.classification == REPLAY_REQUIRED]
    return {
        "total_outcomes": len(outcomes),
        "re_derivable_count": len(re_derivable),
        "replay_required_count": len(replay_required),
        "re_derivable_outcome_ids": re_derivable,
        "replay_required_outcome_ids": replay_required,
    }


def build_gate0_decision() -> dict[str, Any]:
    """Build the deterministic Gate 0 decision payload.

    The payload is validated in place by :func:`validate_gate0_decision` before it is returned, so
    a stop-condition violation raises rather than emitting an inconsistent decision.

    Returns:
        Validated ``radius_sensitivity_gate0_decision.v1`` decision payload.
    """
    outcomes = build_outcome_registry()
    decision: dict[str, Any] = {
        "schema_version": GATE0_DECISION_SCHEMA,
        "issue": 6640,
        "parent_issue": 6600,
        "validity_study_issue": 3207,
        "gate": "gate0_post_hoc_feasibility_audit",
        "campaign": _campaign_block(),
        "frozen_release": _frozen_release_block(),
        "metric_contract": _metric_contract_block(),
        "classification_rubric": _rubric_block(),
        "outcomes": [o.to_dict() for o in outcomes],
        "summary": _summary_block(outcomes),
        "claim_boundary": CLAIM_BOUNDARY,
        "next_gate": "gate1_binding_canary",
        "review_marker": GATE0_REVIEW_MARKER,
    }
    validate_gate0_decision(decision)
    return decision


# No outcome currently meets the exact-retention rule. Keep this explicit so a future change cannot
# silently turn an unpinned parameter or map source into benchmark evidence.
ALLOWED_RE_DERIVABLE_IDS = frozenset()


def _validate_outcome_entry(entry: Any, seen_ids: set[str]) -> str | None:
    """Validate one outcome entry and return its id when it is re-derivable.

    Returns:
        The outcome id when the entry is classified ``re-derivable``, otherwise ``None``.
    """
    if not isinstance(entry, dict):
        raise ValueError("each outcome must be a dict")
    outcome_id = entry.get("outcome_id")
    if not isinstance(outcome_id, str) or not outcome_id:
        raise ValueError("each outcome needs a non-empty string outcome_id")
    if outcome_id in seen_ids:
        raise ValueError(f"duplicate outcome_id {outcome_id!r}")
    seen_ids.add(outcome_id)

    classification = entry.get("classification")
    if classification not in VALID_CLASSIFICATIONS:
        raise ValueError(
            f"outcome {outcome_id!r} classification must be one of {sorted(VALID_CLASSIFICATIONS)}"
        )

    is_collision_family = bool(entry.get("is_collision_contact_feasibility_or_planner_outcome"))
    category = entry.get("category")
    _assert_replay_required_for_trajectory_outcome(
        outcome_id, classification, is_collision_family, category
    )

    if classification != RE_DERIVABLE:
        return None
    _assert_re_derivable_constraints(entry, outcome_id, is_collision_family)
    return outcome_id


def _assert_replay_required_for_trajectory_outcome(
    outcome_id: str,
    classification: str,
    is_collision_family: bool,
    category: Any,
) -> None:
    """Stop condition #3: trajectory-dependent outcomes are always replay-required."""
    if is_collision_family and classification != REPLAY_REQUIRED:
        raise ValueError(
            f"outcome {outcome_id!r} is a collision/contact/feasibility/planner outcome and "
            f"must be {REPLAY_REQUIRED!r}, got {classification!r}"
        )
    if category in TRAJECTORY_DEPENDENT_CATEGORIES and classification != REPLAY_REQUIRED:
        raise ValueError(
            f"outcome {outcome_id!r} category {category!r} is trajectory-dependent and must be "
            f"{REPLAY_REQUIRED!r}"
        )


def _assert_re_derivable_constraints(
    entry: dict[str, Any], outcome_id: str, is_collision_family: bool
) -> None:
    """Stop condition #2: re-derivable requires retained source geometry and semantics."""
    if not bool(entry.get("source_geometry_retained_in_frozen_rows")):
        raise ValueError(
            f"re-derivable outcome {outcome_id!r} must retain its source geometry exactly"
        )
    if is_collision_family:
        raise ValueError(
            f"re-derivable outcome {outcome_id!r} must not be a collision/contact/"
            "feasibility/planner outcome"
        )


def _assert_narrow_re_derivable_set(re_derivable_ids: list[str]) -> None:
    """Stop condition #5: only explicitly retained outcomes may be re-derivable."""
    extra = set(re_derivable_ids) - ALLOWED_RE_DERIVABLE_IDS
    if extra:
        raise ValueError(
            "re-derivable set contains outcomes without exact retained provenance and would imply "
            "a sweep: "
            f"{sorted(extra)}"
        )


def _assert_frozen_provenance_gaps(decision: dict[str, Any]) -> None:
    """Enforce the provenance blockers that make this Gate 0 decision fail closed."""
    metric_contract = decision.get("metric_contract")
    if not isinstance(metric_contract, dict):
        raise ValueError("decision.metric_contract must be a dict")
    gaps = metric_contract.get("frozen_provenance_gaps")
    if not isinstance(gaps, dict):
        raise ValueError("metric_contract.frozen_provenance_gaps must be a dict")
    for field in (
        "effective_robot_and_pedestrian_radius_retained",
        "map_asset_bytes_pinned",
    ):
        if gaps.get(field) is not False:
            raise ValueError(
                f"metric_contract.frozen_provenance_gaps.{field} must be false for this Gate 0 "
                "decision"
            )


def _assert_summary_consistent(
    decision: dict[str, Any], outcomes: list[dict[str, Any]], re_derivable_ids: list[str]
) -> None:
    """Validate the summary block against the outcome list."""
    summary = decision.get("summary")
    if not isinstance(summary, dict):
        raise ValueError("decision.summary must be a dict")
    if summary.get("re_derivable_count") != len(re_derivable_ids):
        raise ValueError("summary.re_derivable_count does not match the outcomes")
    if summary.get("replay_required_count") != len(outcomes) - len(re_derivable_ids):
        raise ValueError("summary.replay_required_count does not match the outcomes")
    if summary.get("total_outcomes") != len(outcomes):
        raise ValueError("summary.total_outcomes does not match the outcomes")
    replay_required_ids = [
        entry["outcome_id"] for entry in outcomes if entry.get("classification") == REPLAY_REQUIRED
    ]
    if summary.get("re_derivable_outcome_ids") != re_derivable_ids:
        raise ValueError("summary.re_derivable_outcome_ids does not match the outcomes")
    if summary.get("replay_required_outcome_ids") != replay_required_ids:
        raise ValueError("summary.replay_required_outcome_ids does not match the outcomes")


def validate_gate0_decision(decision: dict[str, Any]) -> None:
    """Validate a Gate 0 decision payload against the stop conditions and schema.

    Raises:
        ValueError: if any stop condition or schema invariant is violated.
    """
    if not isinstance(decision, dict):
        raise ValueError("decision must be a dict")
    if decision.get("schema_version") != GATE0_DECISION_SCHEMA:
        raise ValueError(
            f"schema_version must be {GATE0_DECISION_SCHEMA!r}, got {decision.get('schema_version')!r}"
        )
    outcomes = decision.get("outcomes")
    if not isinstance(outcomes, list) or not outcomes:
        raise ValueError("decision.outcomes must be a non-empty list")

    _assert_frozen_provenance_gaps(decision)
    seen_ids: set[str] = set()
    re_derivable_ids: list[str] = []
    for entry in outcomes:
        re_derivable_id = _validate_outcome_entry(entry, seen_ids)
        if re_derivable_id is not None:
            re_derivable_ids.append(re_derivable_id)

    # Stop condition #4: every outcome has exactly one classification (enforced per entry above).
    _assert_narrow_re_derivable_set(re_derivable_ids)
    _assert_summary_consistent(decision, outcomes, re_derivable_ids)


def write_gate0_decision(output_path: str | Path) -> Path:
    """Write the deterministic Gate 0 decision JSON and return its path.

    The written file is canonical and reproducible: running this function always emits the same
    bytes for the same metric contract.

    Returns:
        Path to the written decision JSON.
    """
    decision = build_gate0_decision()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, decision)
    return path


def load_gate0_decision(path: str | Path) -> dict[str, Any]:
    """Load and validate a Gate 0 decision JSON file.

    Returns:
        Validated decision payload.
    """
    decision = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_gate0_decision(decision)
    return decision


__all__ = [
    "CAMPAIGN_ARMS_M",
    "CAMPAIGN_AXIS",
    "CAMPAIGN_BASELINE_ARM_M",
    "CLAIM_BOUNDARY",
    "FIXED_THRESHOLD_COLLISION_METRICS",
    "FROZEN_RELEASE_TAG",
    "GATE0_DECISION_SCHEMA",
    "RADIUS_AWARE_CLEARANCE_METRICS",
    "RADIUS_INDEPENDENT_GEOMETRY_METRICS",
    "REPLAY_REQUIRED",
    "RE_DERIVABLE",
    "RadiusSensitivityOutcome",
    "build_gate0_decision",
    "build_outcome_registry",
    "load_gate0_decision",
    "validate_gate0_decision",
    "write_gate0_decision",
]
