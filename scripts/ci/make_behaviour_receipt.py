#!/usr/bin/env python3
"""Produce behaviour-receipt payloads and compact PR-body headers."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ci import (  # noqa: E402
    behaviour_receipt,
    receipt_arm_execution,
    receipt_outcomes,
    receipt_provenance,
)

DEV_SEED_MIN = 1001
DEV_SEED_MAX = 1200
GATE_SEED_MAX = 1030
DEFAULT_SEEDS = tuple(range(DEV_SEED_MIN, GATE_SEED_MAX + 1))
PLACEHOLDER_REVIEW_URI = "https://example.org/replace-with-independent-exact-head-refute-review"
PLACEHOLDER_EVIDENCE_URI = "https://example.org/replace-with-durable-behaviour-evidence"

# Runtime fields are projected at named transports. The recursive catcher below
# rejects unclassified fault-like fields; only exact source-cited paths exempt them.
EXECUTION_EVIDENCE_LOCATIONS: dict[str, Any] = {
    "fault_key_pattern": r"fallback|degrad|(?:^|_)status(?:$|_)|stop_count|stop_safe|stop_best_effort|safe_stop|execution_mode|decision_label",
    "required_object_fields": {
        ("integrity", "effective_view"): {
            "degraded": "map_runner_view_integrity.py:91 always emits the runtime integrity assertion.",
            "degraded_reason": "map_runner_view_integrity.py:92 always emits its typed reason, including null when healthy.",
        },
        ("algorithm_metadata", "topology_guided_episode"): {
            "configured_fallback_steps": "map_runner_episode.py:1060 always exports actual cumulative command substitution use.",
        },
    },
    "non_runtime_locations": {
        "algorithm_metadata.config.fallback_to_goal": "robot_sf/baselines/ppo.py:95 config permission, not activation.",
        "algorithm_metadata.config.allow_fallback": "robot_sf/benchmark/map_runner_policies/socnav_family.py:422 adapter configuration permission, not actual fallback use.",
        "algorithm_metadata.adapter_impact.status": "robot_sf/benchmark/map_runner/map_runner_episode.py:4013 derived adapter-impact completeness.",
        "algorithm_metadata.ammv_feasibility.status": "robot_sf/benchmark/ammv_feasibility.py:203 derived command-feasibility diagnostic, not controller degradation.",
        "algorithm_metadata.learned_checkpoint_observation_contract.status": "robot_sf/benchmark/algorithm_metadata.py:1401 checkpoint observation-schema provenance.",
        "algorithm_metadata.paired_effect_metric_producer.status": "robot_sf/benchmark/paired_effect_metric_contract.py:1374 optional derived metrics availability.",
        "algorithm_metadata.paired_effect_metric_producer.pairing.identity_status": "robot_sf/benchmark/paired_effect_metric_contract.py:1379 derived pairing identity availability.",
        "algorithm_metadata.paired_effect_metric_producer.fields.false_positive_stop_rate.status": "robot_sf/benchmark/paired_effect_metric_contract.py:1359 field availability, not controller execution.",
        "algorithm_metadata.paired_effect_metric_producer.fields.progress_at_timeout.status": "robot_sf/benchmark/paired_effect_metric_contract.py:1359 field availability, not controller execution.",
        "algorithm_metadata.paired_effect_metric_producer.fields.stop_yield_latency_s.status": "robot_sf/benchmark/paired_effect_metric_contract.py:1359 field availability, not controller execution.",
        "algorithm_metadata.planner_runtime.protective_stop_count": "robot_sf/planner/hybrid_rule_local_planner.py:4056 native protective braking without an alternate planner.",
        "algorithm_metadata.planner_runtime.goal_posterior_avoidance.status": "robot_sf/planner/hybrid_rule_local_planner.py:1414 optional goal-posterior feature availability.",
        "algorithm_metadata.planner_runtime.goal_posterior_avoidance.fallback_or_degraded": "robot_sf/planner/hybrid_rule_local_planner.py:1414 literal false feature-contract declaration.",
        "algorithm_metadata.planner_runtime.proxemic_costmap.status": "robot_sf/planner/hybrid_rule_local_planner.py:569 optional feature availability.",
        "algorithm_metadata.planner_runtime.proxemic_costmap.fallback_or_degraded": "robot_sf/planner/hybrid_rule_local_planner.py:569 literal false feature-contract declaration.",
        "algorithm_metadata.planner_runtime.last_decision.proxemic_costmap.status": "robot_sf/planner/hybrid_rule_local_planner.py:569 optional feature availability in last-decision export.",
        "algorithm_metadata.planner_runtime.last_decision.proxemic_costmap.fallback_or_degraded": "robot_sf/planner/hybrid_rule_local_planner.py:569 literal false feature-contract declaration.",
        "algorithm_metadata.simulation_step_trace.reset.routes.status": "robot_sf/benchmark/map_runner/map_runner_episode.py:4256 optional reset-route provenance.",
        "algorithm_metadata.simulation_step_trace.reset.spawn.status": "robot_sf/benchmark/map_runner/map_runner_episode.py:4221 optional reset-spawn provenance.",
        "algorithm_metadata.topology_guided_episode.fallback_used": "robot_sf/benchmark/map_runner/map_runner_episode.py:1029 optional topology lane availability, NOT candidate command substitution; configured_fallback_steps is the executed-use counter.",
        "algorithm_metadata.topology_guided_episode.fallback_steps": "robot_sf/benchmark/map_runner/map_runner_episode.py:1026 counts non-ok optional topology hypotheses, not executed commands.",
        "algorithm_metadata.topology_guided_episode.fallback_reasons": "robot_sf/benchmark/map_runner/map_runner_episode.py:897 optional lane absence reasons, not executed fallback count.",
        "algorithm_metadata.topology_guided_episode.status_counts": "robot_sf/benchmark/map_runner/map_runner_episode.py:891 optional hypothesis availability histogram.",
        "algorithm_metadata.topology_guided_episode.lane_status_counts": "robot_sf/benchmark/map_runner/map_runner_episode.py:902 optional lane availability histogram.",
        "algorithm_metadata.topology_guided_episode.lane_status_counts.fallback_only": "robot_sf/benchmark/map_runner/map_runner_episode.py:902 histogram label for an unavailable optional lane.",
        "algorithm_metadata.topology_guided_episode.candidate_availability_status_counts": "robot_sf/benchmark/map_runner/map_runner_episode.py:872 optional candidate availability histogram.",
        "algorithm_metadata.topology_guided_episode.candidate_outcome_counts.configured_fallback": "robot_sf/benchmark/map_runner/map_runner_episode.py:876 descriptive outcome histogram; configured_fallback_steps remains runtime.",
        "algorithm_metadata.topology_guided_episode.config.fallback_on_no_candidate": "robot_sf/planner/topology_guided_local_policy.py:628 config eligibility, not activation.",
        "algorithm_metadata.planner_runtime.last_decision.topology_candidate_availability.status": "robot_sf/planner/topology_guided_local_policy.py:1234 candidate geometry availability; fallback_used separately records the executed substitute.",
        "algorithm_metadata.planner_runtime.topology_guided.last_candidate_availability.status": "robot_sf/planner/topology_guided_local_policy.py:1234 same optional candidate availability before last-step export.",
        "algorithm_metadata.planner_runtime.last_decision.topology_candidate_availability.fallback_configured": "robot_sf/planner/topology_guided_local_policy.py:1238 configuration eligibility, not executed fallback_used.",
        "algorithm_metadata.planner_runtime.topology_guided.last_candidate_availability.fallback_configured": "robot_sf/planner/topology_guided_local_policy.py:1238 same configuration eligibility.",
        "event_ledger.surrogate_events.occlusion_near_miss_predicate.status": "robot_sf/benchmark/safety/safety_predicates.py:402 derived safety-predicate truth/availability.",
        "event_ledger.surrogate_events.occlusion_near_miss_predicate.status_reason": "robot_sf/benchmark/safety/safety_predicates.py:403 derived predicate availability explanation.",
        "event_ledger.surrogate_events.occlusion_near_miss_predicate.fields.visibility_evidence_status": "robot_sf/benchmark/safety/safety_predicates.py:413 derived predicate visibility provenance.",
        "safety_predicates.occlusion_near_miss_predicate.status": "robot_sf/benchmark/safety/safety_predicates.py:402 derived safety-predicate truth/availability.",
        "safety_predicates.occlusion_near_miss_predicate.status_reason": "robot_sf/benchmark/safety/safety_predicates.py:403 derived predicate availability explanation.",
        "safety_predicates.occlusion_near_miss_predicate.fields.visibility_evidence_status": "robot_sf/benchmark/safety/safety_predicates.py:413 derived predicate visibility provenance.",
        "interaction_exposure.interaction_exposure_status": "robot_sf/benchmark/interaction_exposure.py:160 derived encounter exposure availability.",
        "metrics.deadlock_stall.status": "robot_sf/benchmark/metrics.py:2383 derived deadlock metric availability.",
        "metrics.distributional_disruption.missing_data.extreme_speed_tier.status": "robot_sf/benchmark/metrics.py:4120 derived missing-cohort metric availability.",
        "metrics.distributional_disruption.missing_data.fast_speed_tier.status": "robot_sf/benchmark/metrics.py:4120 derived missing-cohort metric availability.",
        "metrics.distributional_disruption.missing_data.slow_speed_tier.status": "robot_sf/benchmark/metrics.py:4120 derived missing-cohort metric availability.",
        "metrics.force_sample_stats.status": "robot_sf/benchmark/metrics.py:405 force-sample availability, not controller execution.",
        "metrics.robot_force_metadata.pp_equiv_status": "robot_sf/benchmark/metrics.py:369 derived counterfactual force provenance.",
        "metrics.social_compliance.metrics.comfort_exposure_person_s.status": "robot_sf/benchmark/social_compliance.py:72 derived social-compliance field availability.",
        "metrics.social_compliance.metrics.distributional_inconvenience_p90_p50_gap.status": "robot_sf/benchmark/social_compliance.py:72 derived social-compliance field availability.",
        "metrics.social_compliance.metrics.flow_disruption_delay_s.status": "robot_sf/benchmark/social_compliance.py:72 derived social-compliance field availability.",
        "metrics.social_compliance.metrics.legibility_progress_deficit_m.status": "robot_sf/benchmark/social_compliance.py:72 derived social-compliance field availability.",
        "metrics.social_compliance.metrics.pedestrian_deviation_mean_m.status": "robot_sf/benchmark/social_compliance.py:72 derived social-compliance field availability.",
        "metrics.social_mini_game.rows.*.status": "robot_sf/benchmark/metrics.py:3879 per-game derived metric availability; wildcard is an array index only.",
        "metrics.social_mini_game.status": "robot_sf/benchmark/metrics.py:4051 derived mini-game diagnostic scope.",
        "public_requirement.status": "robot_sf/benchmark/public_requirement_events.py:128 optional public-requirement event availability.",
        "result_provenance.postprocessing.*.status": "robot_sf/benchmark/map_runner/map_runner_episode.py:5218 derived metric postprocessing provenance.",
        "scenario_params.metadata.plausibility.status": "robot_sf/benchmark/map_runner/map_runner_episode.py:4777 transports authored scenario configuration, not controller execution status.",
        "scenario_params.metadata.authoring.status": "robot_sf/benchmark/map_runner/map_runner_episode.py:4777 transports authored scenario drafting/verification metadata, not controller execution status.",
        "scenario_params.metadata.scenario_horizon.status": "robot_sf/benchmark/map_runner/map_runner_episode.py:4777 transports authored scenario-horizon configuration, not controller execution status.",
        "algorithm_metadata.cbf_safety_filter.fallback_rate": "robot_sf/benchmark/safety/cbf_safety_filter_runtime.py:472 derived ratio; fallback_step_count remains runtime.",
        "algorithm_metadata.cbf_safety_filter.fallback_rows_are_success_evidence": "robot_sf/benchmark/safety/cbf_safety_filter_runtime.py:476 literal false evidence-policy declaration.",
        "algorithm_metadata.cbf_safety_filter.status_counts": "robot_sf/benchmark/safety/cbf_safety_filter_runtime.py:473 derived QP status histogram; individual qp_status and fallback_step_count remain runtime.",
        "algorithm_metadata.cbf_safety_filter.status_counts.fallback_infeasible": "robot_sf/benchmark/safety/cbf_safety_filter_runtime.py:447 derived histogram label, not independent execution authority.",
        "algorithm_metadata.cbf_safety_filter.fallback_mode": "robot_sf/benchmark/safety/cbf_safety_filter_runtime.py:80 configured command choice; actual use remains runtime.",
        "metrics.cbf_filter_fallback_rate": "robot_sf/benchmark/map_runner/map_runner_episode.py:4670 derived summary ratio; source runtime counters remain required.",
        "selected_map_identity.status": "robot_sf/benchmark/map_runner/map_runner_identity.py:344 map identity provenance, not controller execution.",
        "spawn_validity.reset_clearance_status": "robot_sf/benchmark/spawn_validity.py:111 reset-clearance measurement availability.",
    },
    "runtime_objects": {
        (
            "pedestrian_model",
        ): "ped_model_sensitivity.py:97/286-305 exports robot algorithm/planner faults; map_runner_episode.py:5088 transports the report.",
        (
            "integrity",
            "effective_view",
        ): "map_runner_view_integrity.py:291 detects a controller extractor blind to observed pedestrians; episode.py:4806 transports it.",
        (): "Episode-level controller execution and fallback reports.",
        ("controller",): "Episode controller's runtime report.",
        ("algorithm_metadata",): "Executed algorithm's runtime report, not its config/contracts.",
        ("algorithm_metadata", "controller"): "Algorithm controller's runtime report.",
        ("algorithm_metadata", "planner_diagnostics"): "Episode controller fault counters.",
        ("algorithm_metadata", "planner_runtime"): "Runtime planner's execution diagnostics.",
        (
            "algorithm_metadata",
            "cbf_safety_filter",
            "steps",
            "*",
        ): "Every recorded episode CBF step's actual fallback_applied report.",
        (
            "algorithm_metadata",
            "planner_runtime",
            "cbf_safety_filter",
        ): "CBF's emitted cumulative fallback counter and decisions.",
        (
            "algorithm_metadata",
            "planner_runtime",
            "cbf_safety_filter",
            "last_decision",
        ): "CBF's last actually executed filter decision.",
        (
            "algorithm_metadata",
            "planner_runtime",
            "cbf_safety_filter",
            "last_decision",
            "fallback_controller_state",
        ): "CBF decision's actual fallback flag; not an optional metric.",
        (
            "algorithm_metadata",
            "shield_stats",
            "last_decision",
        ): "Emitted safety-shield decision preserved by the runtime transport.",
        (
            "algorithm_metadata",
            "shield_stats",
            "last_decision",
            "fallback_controller_state",
        ): "Shield decision's bound fallback-controller report.",
        ("shield_stats", "last_decision"): "Legacy episode-level emitted shield decision.",
        (
            "shield_stats",
            "last_decision",
            "fallback_controller_state",
        ): "Legacy shield decision's runtime fallback state.",
        (
            "algorithm_metadata",
            "cbf_safety_filter",
        ): "Episode CBF runtime summary's fallback step count (configuration fields ignored).",
        (
            "algorithm_metadata",
            "cbf_safety_filter",
            "last_step",
        ): "Last episode-level CBF applied/fallback report.",
        (
            "algorithm_metadata",
            "planner_runtime",
            "fast_pysf_wrapper",
        ): "Force-kernel controller fallback diagnostics before promotion.",
        (
            "algorithm_metadata",
            "fast_pysf_wrapper",
        ): "Promoted social-force wrapper fallback diagnostics.",
        (
            "algorithm_metadata",
            "planner_runtime",
            "last_step",
        ): "Policy-stack's last executed arbitration decision.",
        (
            "algorithm_metadata",
            "planner_runtime",
            "last_decision",
            "topology_candidate_availability",
        ): "Topology's fallback_used records actual configured fallback use, not eligibility.",
        (
            "algorithm_metadata",
            "planner_runtime",
            "topology_guided",
            "last_candidate_availability",
        ): "Sticky final topology candidate fallback-use report.",
        ("algorithm_metadata", "planner_runtime", "last_decision"): "Last controller decision.",
        (
            "algorithm_metadata",
            "planner_runtime",
            "checkpoint_provenance",
        ): "Checkpoint fallback actually used at runtime.",
        (
            "algorithm_metadata",
            "foresight_prediction",
        ): "Prediction fallback actually used by the controller.",
        (
            "algorithm_metadata",
            "planner_runtime",
            "foresight_prediction",
        ): "Runtime prediction fallback before metadata promotion.",
        ("fallback_diagnostics",): "Episode fallback diagnostics.",
        ("algorithm_metadata", "fallback_diagnostics"): "Algorithm fallback diagnostics.",
        (
            "algorithm_metadata",
            "planner_runtime",
            "fallback_diagnostics",
        ): "Planner fallback diagnostics.",
        (
            "algorithm_metadata",
            "fallback_controller_state",
        ): "Bound guarded-controller runtime state.",
        (
            "algorithm_metadata",
            "planner_runtime",
            "fallback_controller_state",
        ): "Bound planner fallback-controller state.",
        (
            "algorithm_metadata",
            "simulation_step_trace",
            "steps",
            "*",
            "planner",
        ): "Every recorded controller step's execution report.",
    },
    "runtime_fields": {
        "status": "Controller/algorithm runtime status, not derived-metric availability.",
        "row_status": "Runtime execution row status.",
        "readiness_status": "Controller runtime readiness.",
        "availability_status": "Executed controller's availability.",
        "execution_mode": "Command execution mode and explicit fallback/degraded mode markers.",
        "fallback": "Explicit runtime fallback flag.",
        "degraded": "Explicit runtime degradation flag.",
        "fallback_triggered": "Sticky runtime fallback activation.",
        "fallback_or_degraded": "Combined runtime fault flag.",
        "fallback_used": "Fallback implementation actually executed.",
        "fallback_applied": "Fallback command actually applied.",
        "fallback_to_another_checkpoint": "Runtime checkpoint substitution.",
        "fallback_to_goal_seeking": "Runtime goal-controller substitution.",
        "fallback_count": "Number of runtime fallback events.",
        "fallback_stop_count": "NMPC solver failures which actually substituted a stop command.",
        "fallback_step_count": "Episode-level CBF steps executing its infeasibility fallback.",
        "fallback_steps": "Number of fallback controller steps.",
        "fallback_events": "Runtime fallback event count.",
        "fallback_reason": "Runtime fallback reason, valid empty only beside an explicit false flag.",
        "degraded_count": "Number of degraded runtime events.",
        "degraded_steps": "Number of degraded controller steps.",
        "step_degraded": "Per-step runtime degradation marker.",
        "degraded_reason": "Typed runtime degradation reason.",
        "degraded_statuses": "Typed list of runtime degradation statuses.",
        "stop_best_effort": "Runtime best-effort stop counter/decision.",
        "decision_label": "Runtime shield/controller decision label.",
        "fallback_controller_state": "Typed, independently bound guarded-controller state.",
        "fallback_diagnostics": "Typed runtime fallback diagnostic object.",
    },
    "runtime_status_fields": {
        "status": "Controller/algorithm execution status at named runtime parents.",
        "row_status": "Runtime execution row status, independent of loader completion.",
        "readiness_status": "Executed controller readiness, not optional feature availability.",
        "availability_status": "Executed controller availability, not derived metrics.",
        "execution_mode": "Runtime fault modes also invalidate command execution.",
        "decision_label": "Fault labels in an executed controller/guard decision.",
    },
    "runtime_fault_statuses": {
        "fallback": "An alternate controller or command executed.",
        "degraded": "The controller reports degraded execution.",
        "not_available": "The runtime controller is not available.",
        "unavailable": "The runtime controller is unavailable.",
        "failed": "ped_model_sensitivity.py:305 exports robot runtime failure; the shared predicate omits this value.",
        "error": "A runtime error is not evidence of healthy controller execution.",
    },
    "counter_objects": {
        (
            "algorithm_metadata",
            "topology_guided_episode",
        ): "map_runner_episode.py:878 increments actual candidate fallback use; :1060 exports the cumulative count.",
        (
            "guard_stats",
        ): "Native guarded decision counters, bound to the original algorithm identity.",
        ("algorithm_metadata", "guard_stats"): "Algorithm's native guarded decision counters.",
        ("shield_stats", "decision_counts"): "Native shield decision counters.",
        (
            "algorithm_metadata",
            "shield_stats",
            "decision_counts",
        ): "Algorithm's native shield decision counters.",
        (
            "algorithm_metadata",
            "planner_runtime",
            "proposal_status_counts",
        ): "Planner proposal fallback/degraded counters.",
        (
            "algorithm_metadata",
            "planner_runtime",
            "last_decision",
            "proposal_status_counts",
        ): "Last controller proposal status counters.",
        (
            "algorithm_metadata",
            "planner_diagnostics",
            "proposal_status_counts",
        ): "Episode proposal status counters.",
    },
    "counter_fields": {
        "configured_fallback_steps": "map_runner_episode.py:878 increments only after candidate_availability.fallback_used; a clean last step cannot erase it.",
        "uncertainty_fallback_configured": "guarded_ppo.py:707 executes the fallback adapter; map_runner.py:1689/1867 counts the executed command.",
        "uncertainty_fallback_stop": "guarded_ppo.py:714 returns an executed stop decision; map_runner.py:1689/1867 counts it.",
        "uncertainty_fallback_slow_down": "guarded_ppo.py:703 returns an executed speed-limited decision; map_runner.py:1689/1867 counts it.",
        "cbf_best_effort": "CBF decision histogram's infeasible best-effort fallback commands.",
        "fallback_count": "Guard/adapter runtime fallback events, never a native decision exception.",
        "fallback_stop_count": "NMPC fallback stops retained in counter reports.",
        "fallback": "Fallback proposal count, not a boolean flag.",
        "degraded": "Degraded proposal count, not a boolean flag.",
        "fallback_safe": "Guarded native safe-fallback decision count.",
        "fallback_best_effort": "Guarded native best-effort fallback decision count.",
        "stop_safe": "Guarded native safe-stop decision count.",
        "stop_best_effort": "Guarded native best-effort stop decision count.",
    },
    "runtime_aliases": {
        "fallback_degraded_status": (
            "controller_export_status",
            "ped_model_sensitivity.py:286-305 derives robot faults from algorithm/planner runtime; :281 permits only native/null as clean evidence.",
        ),
        "load_status": (
            "load_status",
            "socnav_sacadrl.py:228 and socnav_prediction.py:229 report failed/fallback controller asset loading, not optional metric availability.",
        ),
        "qp_status": (
            "qp_status",
            "cbf_safety_filter.py:803/963 selects a fallback command and emits fallback_infeasible at :810/970; redundant reports remain runtime evidence.",
        ),
        "fallback_reason_counts": (
            "reason_counts",
            "Episode accumulator's actually executed fallback events before export as fallback_reasons.",
        ),
        "ever_degraded": ("degraded", "Force-coupled planner's sticky episode degradation flag."),
        "degradation_reasons": (
            "reasons",
            "Force-coupled planner's accumulated runtime degradation reasons.",
        ),
        "fallback_reasons": (
            "reason_counts",
            "Fast-PySF wrapper's actually used fallback reasons and counts.",
        ),
        "fallback_from": (
            "reason",
            "Hybrid portfolio records the failed planner head it substituted.",
        ),
        "fallback_status": (
            "fallback_status",
            "Hybrid-global-RL explicitly reports goal_fallback or fail_closed execution.",
        ),
        "runtime_status": (
            "status",
            "BRNE's sticky runtime failure status independent of dependency provenance.",
        ),
        "last_step_status": ("status", "BRNE's last executed step status."),
        "observation_validation_status": (
            "status",
            "Recurrent learned adapter's runtime request validation status.",
        ),
        "guard_or_fallback_reason": (
            "reason",
            "Learned adapter step's applied guard/fallback reason, not its logging contract.",
        ),
        "proposal_status_counts": (
            "container",
            "Policy-stack counts actual fallback/degraded proposals; checked at named counter paths.",
        ),
    },
    "decision_fault_labels": {
        "cbf_best_effort": "CBF could not produce a feasible command and executed its fallback.",
    },
    "cbf_state_bindings": {
        "CollisionConeCbfSafetyFilter": (
            ("collision_cone", "collision_cone_cbf_v1"),
            "Only these emitted filter/variant pairs bind collision-cone runtime state.",
        ),
        "DynamicParabolicCbfSafetyFilter": (
            ("dynamic_parabolic_cbf_v1",),
            "Only this emitted filter/variant pair binds dynamic-parabolic runtime state.",
        ),
    },
    # Source inventory only, NOT exemptions: moving any of these keys to an
    # unclassified row transport must still fail closed.
    "non_runtime_writers": {
        "fallback_on_error": "robot_sf/baselines/brne.py:93 Baseline configuration permits substitution; runtime status reports actual use.",
        "fallback_to_goal": "robot_sf/baselines/distributional_rl.py:33 Learned baseline build configuration; emitted fallback_reason/status reports actual use.",
        "fallback_on_exception": "robot_sf/planner/hybrid_portfolio.py:42 Hybrid portfolio configuration; runtime fallback_count/from reports actual use.",
        "fallback_on_failure": "robot_sf/planner/visibility_planner.py:33 Visibility planner build configuration; not an executed fallback report.",
        "fallback_sources": "robot_sf/planner/policy_stack_v1.py:72 Policy-stack configured source labeling; actual proposal_status_counts is runtime.",
        "degraded_sources": "robot_sf/planner/policy_stack_v1.py:73 Policy-stack configured source labeling; actual proposal_status_counts is runtime.",
        "fallback_mode": "robot_sf/benchmark/safety/cbf_safety_filter_runtime.py:80 Configured CBF fallback command; fallback_applied/count reports actual execution.",
        "allow_fallback_default": "robot_sf/benchmark/map_runner_policies/socnav_family.py:46 Policy builder's default fallback eligibility, not runtime use.",
        "pass_allow_fallback": "robot_sf/benchmark/map_runner_policies/socnav_family.py:47 Policy builder's constructor plumbing, not runtime use.",
        "uncertainty_fallback_enabled": "robot_sf/planner/guarded_ppo.py:83 Guarded-controller configuration, not execution degradation.",
        "uncertainty_fallback_mode": "robot_sf/planner/guarded_ppo.py:89 Configured native guarded decision mode, not a fallback-use report.",
        "radius_status": "robot_sf/benchmark/safety/prediction_planning_safety.py:307 Offline prediction safety-radius diagnostic, not controller runtime.",
        "coverage_status": "robot_sf/benchmark/safety/safety_margin_calibration.py:218 Offline safety-margin calibration diagnostic, not controller runtime.",
        "step_visibility_status": "robot_sf/benchmark/map_runner/map_runner_episode.py:2267 Optional simulator visibility provenance, not controller runtime.",
        "allow_fallback": "robot_sf/benchmark/map_runner/map_runner.py:1131 Configuration permits fallback; does not report its execution.",
        "allow_goal_fallback": "robot_sf/benchmark/map_runner_policies/hybrid_global_rl.py:52 Hybrid adapter configuration eligibility, not runtime use.",
        "allow_predictor_fallback": "robot_sf/planner/chance_constrained_mpc.py:444 Predictor configuration eligibility, not runtime use.",
        "fallback_to_constant_velocity": "robot_sf/planner/learned_short_horizon_predictor.py:35 Predictor build configuration, not the emitted foresight fallback_used flag.",
        "fallback_to_stop": "robot_sf/planner/chance_constrained_mpc.py:442 Solver configuration eligibility; actual use is fallback_stop_count.",
        "fallback_configured": "robot_sf/planner/topology_guided_local_policy.py:1244 Topology candidate fallback eligibility; actual use is fallback_used.",
        "fallback_on_no_candidate": "robot_sf/planner/topology_guided_local_policy.py:85 Topology configuration; actual use is fallback_used/fallback_count.",
        "fallback_policy": "robot_sf/benchmark/map_runner/map_runner.py:1640 Declarative shield/planner contract, not an executed decision.",
        "fallback_or_degraded_success": "robot_sf/planner/lidar_tracked_agents.py:354 Lidar adapter declares fallback cannot count as success, not a runtime fault flag.",
        "fallback_rows_are_success_evidence": "robot_sf/benchmark/safety/cbf_safety_filter_runtime.py:476 CBF report's evidence-policy declaration, not runtime use.",
        "fallback_statuses": "robot_sf/planner/policy_stack_v1.py:328 Policy-stack status vocabulary in its training/trace contract, not observed statuses.",
        "status_policy": "robot_sf/planner/policy_stack_v1.py:327 Policy-stack declarative status vocabulary, not observed runtime status.",
        "status_reason": "robot_sf/benchmark/safety/safety_predicates.py:403 Explanation accompanying a separately classified status, not an independent execution marker.",
        "checkpoint_status": "robot_sf/benchmark/map_runner_policies/diffusion_policy.py:83 Diffusion checkpoint provenance; actual execution has its own status and actions.",
        "normalizer_status": "robot_sf/benchmark/map_runner_policies/diffusion_policy.py:86 Diffusion normalization provenance, not controller degradation.",
        "default_execution_mode": "robot_sf/benchmark/map_runner_policies/diffusion_policy.py:94 Declared adapter default; actual mode checked at mode_locations.",
        "calibration_status": "robot_sf/planner/safety_shield.py:117 Shield calibration provenance, not an executed controller report.",
        "solver_status": "robot_sf/planner/nmpc_social.py:97 Offline NMPC initialization solver result, not an executed episode stop.",
        "protective_stop_count": "robot_sf/planner/hybrid_rule_local_planner.py:4278 Hybrid's nominal protective braking within its controller, not fallback substitution.",
        "hard_stop_count": "robot_sf/benchmark/safety/safety_wrapper_runtime.py:625 Safety-wrapper intervention outcome, not fallback/degradation of the controller.",
        "analyzed_hard_stop_count": "robot_sf/benchmark/safety/safety_wrapper_runtime.py:627 Offline false-stop diagnostic's sample size, not controller degradation.",
        "topology_fallback_status": "robot_sf/planner/topology_guided_local_policy.py:1559 Optional topology lane availability; actual controller substitution reports fallback_count/used.",
        "topology_fallback_reason": "robot_sf/planner/topology_guided_local_policy.py:1560 Optional topology availability explanation, not an independent command substitution.",
        "topology_lane_status": "robot_sf/planner/topology_guided_local_policy.py:1545 Optional topology hypothesis lane availability, not the selected controller's status.",
        "topology_status": "robot_sf/planner/topology_guided_local_policy.py:1213 Optional route-corridor diagnostic availability, not controller execution status.",
        "status_counts": "robot_sf/benchmark/map_runner/map_runner_episode.py:847 Optional topology/CBF diagnostic status histogram; actual fallback has explicit runtime counters.",
        "candidate_availability_status_counts": "robot_sf/benchmark/map_runner/map_runner_episode.py:853 Optional topology candidate availability histogram.",
        "lane_status_counts": "robot_sf/benchmark/map_runner/map_runner_episode.py:852 Optional topology lane availability histogram.",
        "fallback_rate": "robot_sf/benchmark/safety/cbf_safety_filter_runtime.py:472 Derived CBF metric; raw fallback_step_count remains runtime evidence.",
        "cbf_filter_fallback_rate": "robot_sf/benchmark/map_runner/map_runner_episode.py:4670 Derived CBF metric; raw fallback_step_count remains runtime evidence.",
        "forecast_status": "robot_sf/planner/sipp_lattice.py:2222 Offline SIPP feasibility diagnostic, not executed controller status.",
        "static_status": "robot_sf/planner/sipp_lattice.py:2825 Offline static feasibility comparison, not executed controller status.",
        "route_progress_status": "robot_sf/planner/maneuver_candidates.py:1393 Candidate route progress diagnostic, not execution degradation.",
        "route_projection_status": "robot_sf/planner/maneuver_candidates.py:390 Candidate geometry projection diagnostic, not execution degradation.",
        "mixed_producer_status": "robot_sf/planner/obstacle_features.py:707 Optional predictive-ego-motion feature producer availability.",
        "planner_metadata_status": "robot_sf/benchmark/map_runner/map_runner.py:907 Preflight metadata retrieval status, not a written runtime row.",
        "planner_metadata_fallback_reason": "robot_sf/benchmark/map_runner/map_runner.py:908 Preflight metadata retrieval error, not a controller command.",
        "compatibility_status": "robot_sf/benchmark/map_runner/map_runner.py:3179 Preflight kinematics gate status, not written execution.",
        "collision_status": "robot_sf/benchmark/map_runner/map_runner_metrics.py:147 Episode outcome classification, separate from execution evidence.",
        "visibility_evidence_status": "robot_sf/benchmark/map_runner/map_runner_episode.py:586 Optional trace visibility evidence, not controller execution.",
        "artifact_pointer_status": "robot_sf/benchmark/map_runner/map_runner_provenance.py:59 Artifact provenance availability, not controller execution.",
        "result_manifest_status": "robot_sf/benchmark/map_runner/map_runner.py:2805 Artifact export provenance status, not controller execution.",
        "runtime_binding_status": "robot_sf/benchmark/safety/safety_wrapper_ablation_manifest.py:357 Offline safety-wrapper ablation binding evidence, not a runtime controller status.",
        "evidence_status": "robot_sf/benchmark/safety/safety_wrapper_ablation_manifest.py:302 Offline safety-wrapper ablation evidence availability.",
        "baseline_runtime_status": "robot_sf/benchmark/safety/prediction_planning_safety.py:579 Offline paired safety comparison of another arm, not this controller.",
        "uncertainty_aware_runtime_status": "robot_sf/benchmark/safety/prediction_planning_safety.py:580 Offline paired safety comparison of another arm, not this controller.",
        "waypoint_status": "robot_sf/planner/hybrid_global_rl.py:287 Route-conditioning geometry diagnostic; actual goal substitution is fallback_status.",
    },
    "non_runtime_emissions": {
        (
            "robot_sf/benchmark/forecast/forecast_heavy_model_adapter.py",
            "<module>",
            "degraded_status",
        ): "Forecast-batch dataclass provenance, not an executed controller report.",
        (
            "robot_sf/planner/obstacle_features.py",
            "_predictive_ego_motion_channel_producer_metadata",
            "fallback",
        ): "Describes ego-speed feature derivation, not a fallback controller.",
        (
            "robot_sf/planner/learned_policy_adapter.py",
            "metadata",
            "guard_or_fallback_reason",
        ): "Declarative logging-field description, not a decision report.",
        (
            "robot_sf/planner/policy_stack_v1.py",
            "arbitration_trace_packet",
            "degraded_statuses",
        ): "Declarative training-label vocabulary, not observed controller status.",
        (
            "robot_sf/benchmark/forecast/forecast_heavy_model_adapter.py",
            "to_dict",
            "degraded_status",
        ): "Forecast-batch provenance, not an executed planner/controller status.",
        (
            "robot_sf/benchmark/forecast/forecast_risk_adapter.py",
            "compute_forecast_risk",
            "degraded_status",
        ): "Derived forecast-risk producer provenance, not controller execution.",
        (
            "robot_sf/benchmark/scenario_generation/replay_adapter.py",
            "materialize_generated_scenario",
            "replay_status",
        ): "Scenario materialization contract, not controller execution.",
        (
            "robot_sf/benchmark/forecast/forecast_heavy_model_adapter.py",
            "to_dict",
            "fallback_status",
        ): "Forecast-batch provenance, not controller substitution.",
        (
            "robot_sf/benchmark/forecast/forecast_risk_adapter.py",
            "compute_forecast_risk",
            "fallback_status",
        ): "Derived forecast-risk provenance, not controller substitution.",
        (
            "robot_sf/planner/hybrid_rule_local_planner.py",
            "_proxemic_costmap_metadata",
            "fallback_or_degraded",
        ): "Feature-producer contract declares no fallback, not controller runtime.",
        (
            "robot_sf/planner/hybrid_rule_local_planner.py",
            "_goal_posterior_metadata",
            "fallback_or_degraded",
        ): "Goal-posterior feature contract declares no fallback, not controller runtime.",
        (
            "robot_sf/planner/hybrid_rule_local_planner.py",
            "_proxemic_costmap_metadata",
            "status",
        ): "Optional proxemic-costmap feature availability.",
        (
            "robot_sf/planner/hybrid_rule_local_planner.py",
            "_goal_posterior_metadata",
            "status",
        ): "Optional goal-posterior feature availability.",
        (
            "robot_sf/planner/socnav_orca.py",
            "adapter_trace_summary",
            "status",
        ): "Optional ORCA mechanism telemetry availability.",
        (
            "robot_sf/planner/learned_risk_surface.py",
            "occupancy_meta",
            "status",
        ): "Occupancy artifact/provenance availability.",
        (
            "robot_sf/planner/stream_gap.py",
            "_uncertainty_keep_mask",
            "status",
        ): "Uncertainty feature gate diagnostic; controller execution is reported independently.",
        (
            "robot_sf/planner/stream_gap.py",
            "_fail_closed_uncertainty_gate",
            "status",
        ): "Uncertainty feature gate diagnostic; controller command completeness is required separately.",
        (
            "robot_sf/planner/topology_guided_local_policy.py",
            "_hypotheses_for_observation",
            "status",
        ): "Optional topology hypothesis availability.",
        (
            "robot_sf/planner/topology_guided_local_policy.py",
            "_topology_hypothesis_candidate",
            "status",
        ): "Optional candidate-source availability; actual fallback_used remains runtime.",
        (
            "robot_sf/baselines/brne.py",
            "_summarize_control_candidates",
            "status",
        ): "Optional BRNE mechanism diagnostic candidate availability.",
        (
            "robot_sf/baselines/brne.py",
            "_record_mechanism_step",
            "status",
        ): "Mechanism trace diagnostic; sticky runtime_status/last_step_status checked independently.",
        (
            "robot_sf/benchmark/map_runner/map_runner_env.py",
            "representative_metric_affecting_config",
            "status",
        ): "Configuration provenance extraction status.",
        (
            "robot_sf/benchmark/map_runner/map_runner_identity.py",
            "selected_map_identity_from_runtime_inputs",
            "status",
        ): "Map identity provenance, not controller degradation.",
        (
            "robot_sf/benchmark/map_runner/map_runner_episode.py",
            "_reset_spawn_edge",
            "status",
        ): "Optional reset/spawn provenance availability.",
        (
            "robot_sf/benchmark/map_runner/map_runner_episode.py",
            "_reset_routes_edge",
            "status",
        ): "Optional reset/routes provenance availability, the original false-positive source.",
        (
            "robot_sf/benchmark/map_runner/map_runner_episode.py",
            "_finalize_adapter_impact_metadata",
            "status",
        ): "Derived adapter-impact availability; actual mode is checked independently.",
        (
            "robot_sf/benchmark/map_runner/map_runner_episode.py",
            "_finalize_result_provenance_block",
            "status",
        ): "Export provenance availability, not controller execution.",
        (
            "robot_sf/benchmark/map_runner/map_runner_episode.py",
            "_finalize_trace_metadata",
            "status",
        ): "Trace export availability; actual trace completeness remains a required check.",
        (
            "robot_sf/benchmark/map_runner/map_runner_trace.py",
            "_signal_state_trace_payload",
            "status",
        ): "Optional signal provenance availability.",
        (
            "robot_sf/benchmark/map_runner/map_runner_trace.py",
            "_intent_conditioned_behavior_summary",
            "status",
        ): "Derived actor-intent diagnostic availability.",
        (
            "robot_sf/benchmark/map_runner/map_runner_trace.py",
            "_vru_diagnostic_summary",
            "status",
        ): "Derived VRU diagnostic availability.",
        (
            "robot_sf/benchmark/map_runner/map_runner_metrics.py",
            "summarize_collision_metrics",
            "status",
        ): "Outcome metric availability, separate from controller execution.",
        (
            "robot_sf/benchmark/safety/safety_predicates.py",
            "occlusion_near_miss_predicate",
            "status",
        ): "Derived safety-predicate availability.",
        (
            "robot_sf/benchmark/safety/safety_wrapper_ablation_manifest.py",
            "build_safety_wrapper_ablation_manifest",
            "status",
        ): "Offline ablation evidence assembly status.",
        (
            "robot_sf/benchmark/safety/safety_wrapper_factorial_preregistration.py",
            "build_preregistration_plan",
            "status",
        ): "Offline preregistration planning status.",
        (
            "robot_sf/benchmark/safety/safety_wrapper_factorial_report.py",
            "build_safety_wrapper_factorial_report",
            "status",
        ): "Offline factorial report status.",
        (
            "robot_sf/benchmark/safety/prediction_planning_safety.py",
            "_same_seed_comparison",
            "status",
        ): "Offline comparison evidence availability.",
        (
            "robot_sf/benchmark/safety/prediction_planning_safety.py",
            "_chance_constrained_provenance",
            "status",
        ): "Prediction/provenance evidence availability.",
    },
    "required_values": {
        ("execution_status",): ("written", "Loader confirms the controller row was written."),
        ("algorithm_metadata", "status"): (
            "ok",
            "Executed algorithm must explicitly report success.",
        ),
    },
    "mode_locations": {
        ("execution_mode",): "Episode mode must agree with algorithm/controller mode.",
        (
            "algorithm_metadata",
            "planner_kinematics",
            "execution_mode",
        ): "Primary command-space mode.",
        (
            "algorithm_metadata",
            "execution_mode",
        ): "Legacy runtime mode must not contradict primary mode.",
        (
            "algorithm_metadata",
            "adapter_impact",
            "execution_mode",
        ): "Recorded adapter mode must agree with executed mode.",
    },
    "mode_capabilities": {
        (
            "algorithm_metadata",
            "planner_kinematics",
            "supports_native_commands",
        ): "A declared capability contract must support the executed native command space.",
        (
            "algorithm_metadata",
            "planner_kinematics",
            "supports_adapter_commands",
        ): "A declared capability contract must support the executed adapter command space.",
    },
    "required_evidence": {
        ("algo",): "Algorithm identity, with the legacy algorithm field as an alternative.",
        ("algorithm",): "Legacy algorithm identity alternative; one identity must exist.",
        ("steps",): "Positive integral executed-step count must equal trace length.",
        (
            "algorithm_metadata",
            "simulation_step_trace",
        ): "Complete finite reset/step geometry is required, independent of optional provenance.",
        (
            "algorithm_metadata",
            "simulation_step_trace",
            "schema_version",
        ): "Recorded trace must use a supported execution schema.",
        (
            "algorithm_metadata",
            "simulation_step_trace",
            "dt",
        ): "Executed timestep must be finite and positive.",
        (
            "algorithm_metadata",
            "simulation_step_trace",
            "reset",
            "robot",
            "position",
        ): "Finite initial robot geometry is required.",
        (
            "algorithm_metadata",
            "simulation_step_trace",
            "steps",
            "*",
            "robot",
            "position",
        ): "Finite robot geometry is required at every controller step.",
        (
            "algorithm_metadata",
            "simulation_step_trace",
            "steps",
            "*",
            "pedestrians",
        ): "Every executed step must record its actor list, including an empty list.",
        (
            "algorithm_metadata",
            "simulation_step_trace",
            "steps",
            "*",
            "planner",
            "selected_action",
        ): "Every executed controller step must have a nonempty action object.",
        (
            "controller_executed",
        ): "Optional override cannot deny execution or have a malformed type; absence requires trace/action proof.",
    },
}


@dataclass(frozen=True)
class SweepRun:
    """Recorded sweep execution or submission metadata used by the receipt header."""

    source_sha: str
    output_dir: Path
    artifact_uri: str
    artifact_sha256: str
    job_id: str
    hardware: dict[str, Any] | None = None
    shard_dirs: tuple[Path, ...] = ()


def _record_request(
    output_dir: Path, source_sha: str, job_id: str, node: str, arm_role: str
) -> None:
    (output_dir / "execution_request.json").write_bytes(
        _json_bytes(
            {
                "schema_version": "behaviour-hardware-request.v1",
                "requested_node": node,
                "source_sha": source_sha,
                "arm_role": arm_role,
                "job_id": job_id,
            }
        )
    )


def _shard_dirs(output_dir: Path, job_id: str) -> tuple[Path, ...]:
    jobs = behaviour_receipt.scheduler_job_ids(job_id)
    paths = tuple(
        output_dir if len(jobs) == 1 and output_dir.name == job else output_dir / job
        for job in jobs
    )
    for job, path in zip(jobs, paths, strict=True):
        if path.name != job or path.is_symlink() or not path.is_dir():
            raise behaviour_receipt.HardwareIdentityError(
                "hardware comparison invalid: missing job-ID-keyed shard directory"
            )
    return paths


def _read_hardware(output_dir: Path, source_sha: str, job_id: str) -> dict[str, Any]:
    jobs = behaviour_receipt.scheduler_job_ids(job_id)
    snapshot = behaviour_receipt.scheduler_snapshot(jobs)
    claims = []
    for job, path in zip(jobs, _shard_dirs(output_dir, job_id), strict=True):
        files = {}
        for name in behaviour_receipt.HARDWARE_FILES:
            if name == "execution_scheduler.json":
                raw = _json_bytes(behaviour_receipt.scheduler_snapshot([job]))
            else:
                file = path / name
                if file.is_symlink() or not file.is_file():
                    raise behaviour_receipt.HardwareIdentityError(
                        f"hardware comparison invalid: missing regular {name} claim file"
                    )
                try:
                    raw = file.read_bytes()
                except OSError as exc:
                    raise behaviour_receipt.HardwareIdentityError(
                        f"hardware comparison invalid: unreadable {name} claim file"
                    ) from exc
            try:
                content = raw.decode("utf-8")
            except UnicodeError as exc:
                raise behaviour_receipt.HardwareIdentityError(
                    f"hardware comparison invalid: invalid {name} encoding"
                ) from exc
            files[name] = {"content": content, "sha256": hashlib.sha256(raw).hexdigest()}
        identity = behaviour_receipt.hardware_identity(files)
        if identity["source_sha"] != source_sha or identity["job_id"] != job:
            raise behaviour_receipt.HardwareIdentityError(
                "hardware comparison invalid: shard source or job differs from request"
            )
        claims.append(files)
    files = claims[0]
    request = json.loads(files["execution_request.json"]["content"])
    request["job_id"] = job_id
    for name, payload in (
        ("execution_request.json", request),
        ("execution_scheduler.json", snapshot),
    ):
        raw = _json_bytes(payload)
        files[name] = {"content": raw.decode(), "sha256": hashlib.sha256(raw).hexdigest()}
    behaviour_receipt.hardware_identity(files)
    return files


def _stamp_scheduler_rows(output_dir: Path) -> int:
    """Bind sealed comparison inputs without modifying any producer bytes."""
    job = os.environ.get("SLURM_JOB_ID", "")
    node = os.environ.get("SLURM_JOB_NODELIST", "")
    behaviour_receipt.scheduler_job_ids(job)
    if output_dir.name != job or not node:
        raise behaviour_receipt.HardwareIdentityError(
            "hardware comparison invalid: execution directory or scheduler context missing"
        )
    for path in (
        *output_dir.glob("episodes_*.jsonl"),
        *output_dir.glob("campaigns/**/episodes.jsonl"),
    ):
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        for row in rows:
            if any(key in row for key in ("scheduler_job_id", "scheduler_node")):
                raise behaviour_receipt.HardwareIdentityError(
                    "hardware comparison invalid: row already has an allocation binding"
                )
            behaviour_receipt.validate_source_row_context(
                {**row, "scheduler_job_id": job, "scheduler_node": node}, job, node
            )
    receipt_provenance.write_binding(output_dir, job, node)
    return 0


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, indent=2, sort_keys=True).encode() + b"\n"


def _canonical_digest(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(repo_root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo_root, text=True).strip()


def _full_sha(repo_root: Path, ref: str) -> str:
    return _git(repo_root, "rev-parse", "--verify", f"{ref}^{{commit}}")


def _check_seed_range(seeds: list[int]) -> list[int]:
    if not seeds:
        raise ValueError("at least one development seed is required")
    bad = [seed for seed in seeds if seed < DEV_SEED_MIN or seed > DEV_SEED_MAX]
    if bad:
        raise ValueError(f"seeds outside development range {DEV_SEED_MIN}-{DEV_SEED_MAX}: {bad}")
    unsupported = [seed for seed in seeds if seed > GATE_SEED_MAX]
    if unsupported:
        raise ValueError(
            "the current behaviour gate admits only seeds "
            f"{DEV_SEED_MIN}-{GATE_SEED_MAX}; unsupported seeds: {unsupported}"
        )
    return seeds


def _release_source(repo_root: Path, baseline: str) -> str:
    try:
        return _full_sha(repo_root, f"refs/tags/{baseline}")
    except subprocess.CalledProcessError:
        subprocess.run(
            ["git", "fetch", "origin", f"refs/tags/{baseline}:refs/tags/{baseline}"],
            cwd=repo_root,
            check=True,
        )
        return _full_sha(repo_root, f"refs/tags/{baseline}")


def _sweep_command(
    source_sha: str,
    output_dir: Path,
    seeds: list[int],
    *,
    arms: list[str] | None,
    scenarios: list[str] | None,
    workers: int,
    check_only: bool,
) -> list[str]:
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/validation/run_empty_world_sweep.py",
        "--head-sha",
        source_sha,
        "--suite",
        "both",
        "--seeds",
        *[str(seed) for seed in seeds],
        "--workers",
        str(workers),
        "--output-dir",
        str(output_dir),
    ]
    if arms:
        cmd.extend(["--arms", *arms])
    if scenarios:
        cmd.extend(["--scenarios", *scenarios])
    if check_only:
        cmd.append("--check-only")
    return cmd


def _ensure_worktree(repo_root: Path, work_dir: Path, label: str, source_sha: str) -> Path:
    checkout = work_dir / "worktrees" / label
    if checkout.exists():
        return checkout
    checkout.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "worktree", "add", "--detach", str(checkout), source_sha],
        cwd=repo_root,
        check=True,
    )
    return checkout


def _submit_sbatch_sweep(
    repo_root: Path,
    work_dir: Path,
    label: str,
    source_sha: str,
    seeds: list[int],
    *,
    args: argparse.Namespace,
) -> SweepRun:
    node = args.sbatch_node
    if node is None:
        raise ValueError("--mode sbatch requires --sbatch-node to pair numerical hardware")
    if re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?", node) is None:
        raise ValueError("--sbatch-node must name a single literal node hostname")
    inventory, _ = behaviour_receipt.scheduler_inventory()
    if node not in inventory:
        raise behaviour_receipt.HardwareIdentityError(
            "hardware comparison invalid: requested node is absent from trusted inventory"
        )
    checkout = _ensure_worktree(repo_root, work_dir, label, source_sha)
    output_root = work_dir / "sweeps" / label
    output_root.mkdir(parents=True, exist_ok=True)
    cmd = _sweep_command(
        source_sha,
        Path("__RECEIPT_JOB_DIR__"),
        seeds,
        arms=args.arms,
        scenarios=args.scenarios,
        workers=args.workers,
        check_only=False,
    )
    shell_command = " ".join(
        '"$RECEIPT_JOB_DIR"' if item == "__RECEIPT_JOB_DIR__" else shlex.quote(item) for item in cmd
    )
    request_template = shlex.quote(
        json.dumps(
            {
                "schema_version": "behaviour-hardware-request.v1",
                "requested_node": node,
                "source_sha": source_sha,
                "arm_role": label,
            }
        )
    )
    with tempfile.NamedTemporaryFile(
        mode="w",
        prefix=f"behaviour_receipt_{label}_",
        suffix=".sbatch",
        dir=work_dir,
        delete=False,
        encoding="utf-8",
    ) as handle:
        script = Path(handle.name)
        handle.write(
            "\n".join(
                [
                    "#!/usr/bin/env bash",
                    f"#SBATCH --job-name=behaviour-{label}",
                    f"#SBATCH --output={output_root / 'slurm-%j.out'}",
                    "#SBATCH --nodes=1",
                    f"#SBATCH --nodelist={node}",
                    f"#SBATCH --partition={inventory[node]['node_class']}",
                    f"#SBATCH --cpus-per-task={args.workers}",
                    "#SBATCH --time=24:00:00",
                    "set -euo pipefail",
                    f'RECEIPT_JOB_DIR={shlex.quote(str(output_root))}/"${{SLURM_JOB_ID:?}}"',
                    'mkdir -- "$RECEIPT_JOB_DIR"',
                    f"cd {shlex.quote(str(checkout))}",
                    'hostname > "$RECEIPT_JOB_DIR/execution_node.txt"',
                    'LC_ALL=C lscpu --json > "$RECEIPT_JOB_DIR/execution_cpu.json"',
                    'printf "%s\\n" "${SLURM_JOB_PARTITION:?}" > "$RECEIPT_JOB_DIR/execution_node_class.txt"',
                    f'test "$(cat "$RECEIPT_JOB_DIR/execution_node.txt")" = {shlex.quote(node)}',
                    f"printf '%s\\n' {request_template} > \"$RECEIPT_JOB_DIR/request_template.json\"",
                    f'uv run python {shlex.quote(str(Path(__file__).resolve()))} --record-job-request "$RECEIPT_JOB_DIR"',
                    shell_command,
                    f'uv run python {shlex.quote(str(Path(__file__).resolve()))} --record-scheduler-context "$RECEIPT_JOB_DIR"',
                    "",
                ]
            )
        )
    job_id = subprocess.check_output(["sbatch", "--parsable", str(script)], text=True).strip()
    behaviour_receipt.scheduler_job_ids(job_id)
    output_dir = output_root / job_id
    submissions = output_root / "submissions"
    submissions.mkdir(exist_ok=True)
    manifest = submissions / f"{job_id}.json"
    with manifest.open("xb") as handle:
        handle.write(
            _json_bytes(
                {
                    "schema_version": "behaviour-sweep-submission.v2",
                    "label": label,
                    "source_sha": source_sha,
                    "job_id": job_id,
                    "command": cmd,
                    "script": script.relative_to(work_dir).as_posix(),
                    "requested_node": node,
                }
            )
        )
    return SweepRun(
        source_sha,
        output_dir,
        f"{PLACEHOLDER_EVIDENCE_URI}/{label}/{job_id}",
        _sha256(manifest),
        job_id,
    )


def _existing_sweep(
    source_sha: str,
    output_dir: Path,
    *,
    artifact_uri: str,
    artifact_sha256: str | None,
    job_id: str,
) -> SweepRun:
    if not output_dir.is_dir():
        raise FileNotFoundError(output_dir)
    hardware = _read_hardware(output_dir, source_sha, job_id)
    shards = _shard_dirs(output_dir, job_id)
    for shard in shards:
        _verify_sweep_metadata(shard, source_sha)
    digest = _canonical_digest(
        {
            path.relative_to(output_dir).as_posix(): _sha256(path)
            for shard in shards
            for path in sorted(shard.rglob("*"))
            if path.is_file()
        }
    )
    if artifact_sha256 is not None and artifact_sha256 != digest:
        raise ValueError("supplied sweep digest differs from actual artifact bytes")
    return SweepRun(source_sha, output_dir, artifact_uri, digest, job_id, hardware, shards)


def _verify_sweep_metadata(output_dir: Path, source_sha: str) -> None:
    summaries = sorted(output_dir.glob("episodes_*.jsonl"))
    if not summaries:
        raise ValueError("sweep has no episode suites")
    for summary in summaries:
        suite = summary.stem.removeprefix("episodes_")
        metadata_path = output_dir / f"execution_{suite}.json"
        if not metadata_path.is_file():
            raise ValueError(f"missing authoritative execution metadata for {suite}")
        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        if meta.get("head_sha") != source_sha:
            raise ValueError(f"{suite} recorded SHA differs from claimed source SHA")
        if meta.get("suite") != suite:
            raise ValueError(f"{suite} execution suite identity differs")
        complete = (
            meta.get("complete") is True
            and meta.get("trace_requested") is True
            and meta.get("campaign_execution_status") == "completed"
            and meta.get("exit_code") == 0
            and meta.get("unexpected_failed_runs") == 0
            and all(
                meta.get(key) == []
                for key in (
                    "failed_slots",
                    "missing_slots",
                    "duplicate_slots",
                    "unexpected_slots",
                    "incomplete_trace_slots",
                )
            )
        )
        if not complete:
            raise ValueError(f"{suite} sweep execution is incomplete")
        records = [json.loads(line) for line in summary.read_text().splitlines() if line.strip()]
        counts = [meta.get(key) for key in ("expected_slots", "written_rows", "total_episodes")]
        if not records or any(count != len(records) for count in counts):
            raise ValueError(f"{suite} execution row counts differ")
        seeds = meta.get("seeds")
        if not isinstance(seeds, list) or sorted(_check_seed_range(seeds)) != sorted(
            {row["seed"] for row in records}
        ):
            raise ValueError(f"{suite} execution seed identity differs")
        if any(row.get("execution_status") != "written" for row in records):
            raise ValueError(f"{suite} has unsuccessful row execution")


def _index_raw_episodes(
    path: Path, cache: dict[Path, dict[tuple[Any, Any], list[dict[str, Any]]]]
) -> dict[tuple[Any, Any], list[dict[str, Any]]]:
    """Decode a raw file once, retaining duplicate identities for the existing check."""
    if path in cache:
        return cache[path]
    index: dict[tuple[Any, Any], list[dict[str, Any]]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        identity = (raw.get("scenario_id"), raw.get("seed"))
        try:
            index.setdefault(identity, []).append(raw)
        except TypeError:
            # JSON arrays/objects cannot match a summary's (str, int) identity.
            # The former linear scan ignored these nonmatching identities too.
            continue
    cache[path] = index
    return index


def _validate_row_allocation(summary: dict, raw: dict, job: str | None, node: str | None) -> None:
    if job is None:
        return
    if node is None:
        raise behaviour_receipt.HardwareIdentityError(
            "hardware comparison invalid: missing verified allocation node"
        )
    for context in (summary, raw):
        # Fields absent from sealed producer rows come from the verified binding.
        # Present fields remain claims and must agree with the scheduler.
        context.setdefault("scheduler_job_id", job)
        context.setdefault("scheduler_node", node)
        behaviour_receipt.validate_source_row_context(context, job, node)


def _load_sweep_rows(
    sweep_dir: Path,
    *,
    scheduler_job_id: str | None = None,
    scheduler_node: str | None = None,
    preserve_raw: bool = False,
) -> dict[tuple[str, str, int], dict[str, Any]]:
    """Join summaries to raw traces with an invocation-local index of each file.

    There is no persistent cache: subsequent calls reread the current bytes.
    Allocation reads verify custody before decoding; raw hashing never overlays status.
    """
    receipt_provenance.verify_optional_binding(sweep_dir, scheduler_job_id, scheduler_node)

    rows: dict[tuple[str, str, int], dict[str, Any]] = {}
    raw_indexes: dict[Path, dict[tuple[Any, Any], list[dict[str, Any]]]] = {}
    for path in sorted(sweep_dir.glob("episodes_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row["arm"]), str(row["scenario"]), int(row["seed"]))
            suite = path.stem.removeprefix("episodes_")
            campaign = sweep_dir / "campaigns" / f"empty_world_{suite}"
            source = row.get("source_file")
            if not isinstance(source, str):
                raise ValueError(f"missing raw controller trace source for {key}")
            source_path = (campaign / source).resolve()
            if not source_path.is_relative_to(campaign.resolve()):
                raise ValueError("raw controller trace path escapes the campaign")
            matches = _index_raw_episodes(source_path, raw_indexes).get(key[1:], [])
            if len(matches) != 1 or source_path.parent.name.split("__")[0] != key[0]:
                raise ValueError(f"raw episode identity differs for {key}")
            raw = matches[0]
            _validate_row_allocation(row, raw, scheduler_job_id, scheduler_node)
            if receipt_outcomes.raw_outcomes(raw) != receipt_outcomes.summary_outcomes(row):
                raise ValueError(f"raw episode outcome differs from summary for {key}")
            if key in rows:
                raise ValueError(f"duplicate sweep row {key}")
            rows[key] = (
                raw if preserve_raw else {**raw, "execution_status": row["execution_status"]}
            )
    if not rows:
        raise ValueError(f"no sweep rows found in {sweep_dir}")
    return rows


def _load_run_rows(
    run: SweepRun, arm_role: str, *, preserve_raw: bool = False
) -> dict[tuple[str, str, int], dict[str, Any]]:

    if run.hardware is None:
        raise behaviour_receipt.HardwareIdentityError(
            "hardware comparison invalid: unverified sweep cannot provide rows"
        )
    identity = behaviour_receipt.hardware_identity(run.hardware)
    rows = {}
    for shard in run.shard_dirs:
        part = receipt_provenance.verified_raw_rows(
            shard, shard.name, identity["node"], source_sha=run.source_sha, arm_role=arm_role
        )
        if preserve_raw:
            original = _load_sweep_rows(
                shard,
                scheduler_job_id=shard.name,
                scheduler_node=identity["node"],
                preserve_raw=True,
            )
            if set(original) != set(part):
                raise behaviour_receipt.HardwareIdentityError(
                    "raw execution invalid: reread scope differs"
                )
            preserved = {}
            for key, raw in original.items():
                audit = dict(raw)
                for field in ("execution_status", "scheduler_job_id", "scheduler_node"):
                    if field not in audit and field in part[key]:
                        audit[field] = part[key][field]
                if audit != dict(part[key]):
                    raise behaviour_receipt.HardwareIdentityError(
                        "raw execution invalid: reread source differs"
                    )
                preserved[key] = receipt_arm_execution.carry_execution_proof(part[key], raw)
            part = preserved
        if set(part) & set(rows):
            raise ValueError("duplicate rows across scheduler shards")
        rows.update(part)
    return rows


def _row_success(row: dict[str, Any]) -> bool:
    return receipt_outcomes.raw_outcomes(row)["success"]


def _row_collisions(row: dict[str, Any]) -> int:
    return receipt_outcomes.raw_outcomes(row)["collisions"]


def _runtime_objects(
    value: Any, path: tuple[str, ...], prefix: str = ""
) -> Iterator[tuple[str, Any]]:
    """Enumerate only named locations; malformed present parents remain evidence."""
    if not path:
        yield prefix, value
    elif path[0] == "*":
        if isinstance(value, list):
            for index, child in enumerate(value):
                yield from _runtime_objects(child, path[1:], f"{prefix}[{index}]")
        else:
            yield prefix, value
    elif not isinstance(value, dict):
        yield prefix, value
    elif path[0] in value:
        name = f"{prefix}.{path[0]}" if prefix else path[0]
        yield from _runtime_objects(value[path[0]], path[1:], name)


def _runtime_value_error(view: dict[str, Any], *, counters: bool) -> str | None:
    """Reject malformed present runtime statuses and counter scalars."""
    for key, value in view.items():
        if key in EXECUTION_EVIDENCE_LOCATIONS["runtime_status_fields"]:
            if not isinstance(value, str) or not value.strip():
                return key
        if counters:
            try:
                valid = (
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(value)
                    and value >= 0
                )
            except (OverflowError, ValueError):
                valid = False
            if not valid:
                return key
    return None


def _bound_runtime_view(
    view: dict[str, Any], path: tuple[str, ...], *, counters: bool, original: dict[str, Any]
) -> dict[str, Any]:
    """Preserve existing typed-container and guarded-counter identity checks."""
    if (
        "fallback_reason" in view
        and view.get("fallback") is False
        and view["fallback_reason"] in (None, "")
    ):
        # Wrapper emitters use fallback, not fallback_used, beside an empty reason.
        # Bind the identical false assertion for the shared reason validator.
        view.setdefault("fallback_used", False)
    for key in ("fallback_controller_state", "fallback_diagnostics"):
        if key in view and isinstance(view[key], dict):
            state = view[key]
            if key == "fallback_controller_state" and "filter" in state:
                filter_name = state["filter"]
                binding = (
                    EXECUTION_EVIDENCE_LOCATIONS["cbf_state_bindings"].get(filter_name)
                    if isinstance(filter_name, str)
                    else None
                )
                if (
                    binding is not None
                    and original.get("schema_version") == "shield-decision.v1"
                    and state.get("variant") in binding[0]
                    and type(state.get("fallback")) is bool
                ):
                    # CBF is not guarded PPO. Its typed decision binds this
                    # state, whose direct runtime fields are checked separately.
                    del view[key]
                    continue
                view[key] = None
                continue
            view[key] = {}
    if not counters:
        return view
    if path[-1] == "guard_stats":
        return {"guard_stats": view}
    if path[-1] == "decision_counts":
        return {"shield_stats": {"decision_counts": view}}
    return {"proposal_status_counts": view}


def _runtime_reason_counts_marker(value: Any) -> str | None:
    if not isinstance(value, dict):
        return "invalid"
    for reason, count in value.items():
        if _runtime_value_error({str(reason): count}, counters=True):
            return "invalid"
        if count > 0:
            return "fallback"
    return None


def _controller_export_status_marker(value: Any) -> str | None:
    if value is None or value == "native":
        return None
    return (
        value
        if isinstance(value, str)
        and value in EXECUTION_EVIDENCE_LOCATIONS["runtime_fault_statuses"]
        else "invalid"
    )


def _runtime_alias_value_marker(kind: str, value: Any) -> str | None:
    if kind == "controller_export_status":
        return _controller_export_status_marker(value)
    if kind in {"load_status", "qp_status"}:
        clean = {
            "load_status": ("loaded", "not_attempted"),
            "qp_status": ("pass_through", "filtered", "disabled", "ineligible"),
        }[kind]
        return None if value in clean else "degraded"
    if kind == "degraded":
        return None if value is False else "true" if value is True else "invalid"
    if kind == "reasons":
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            return "invalid"
        return "degraded" if value else None
    if kind == "reason_counts":
        return _runtime_reason_counts_marker(value)
    if kind in {"reason", "fallback_status"}:
        if value is None or value == "":
            return None
        return (
            "invalid"
            if not isinstance(value, str)
            else None
            if value.strip().lower() in {"none", "nominal"}
            else "fallback"
        )
    if kind == "status":
        return (
            None
            if isinstance(value, str) and value.strip().lower() in {"ok", "validated"}
            else "degraded"
        )
    return None


def _runtime_alias_marker(obj: dict[str, Any], *, name: str = "") -> tuple[str, str] | None:
    """Interpret emitter-specific fields only at allowlisted runtime parents."""
    for key, (kind, _) in EXECUTION_EVIDENCE_LOCATIONS["runtime_aliases"].items():
        if key not in obj or kind == "container" or _non_runtime_location(name, key):
            continue
        marker = _runtime_alias_value_marker(kind, obj[key])
        if marker is not None:
            return key, marker
    return None


def _runtime_decision_marker(view: dict[str, Any]) -> tuple[str, str] | None:
    from robot_sf.benchmark.fallback_policy import runtime_fallback_or_degraded_marker

    decision = view.get("decision_label")
    if decision in EXECUTION_EVIDENCE_LOCATIONS["decision_fault_labels"]:
        return "decision_label", "fallback"
    marker = runtime_fallback_or_degraded_marker({"status": decision}) if decision else None
    if marker is not None:
        return "decision_label", marker[1]
    if view.get("shield_stats", {}).get("decision_counts", {}).get("cbf_best_effort", 0) > 0:
        return "cbf_best_effort", "fallback"
    return None


def _runtime_status_marker(view: dict[str, Any]) -> tuple[str, str] | None:
    """Close the shared predicate's vocabulary gap only in the runtime projection."""
    for key in EXECUTION_EVIDENCE_LOCATIONS["runtime_status_fields"]:
        value = view.get(key)
        if isinstance(value, str):
            normalized = value.strip().lower().replace("-", "_")
            if normalized in EXECUTION_EVIDENCE_LOCATIONS["runtime_fault_statuses"]:
                return key, normalized
    return None


def _required_runtime_marker(row: dict[str, Any]) -> tuple[str, str] | None:
    """Present integrity and cumulative-use reports must not omit their assertions."""
    for path, required in EXECUTION_EVIDENCE_LOCATIONS["required_object_fields"].items():
        for name, obj in _runtime_objects(row, path):
            if not isinstance(obj, dict):
                return name, "invalid"
            for key in required:
                if key not in obj:
                    return f"{name}.{key}", "invalid"
    return None


def _controller_runtime_marker(
    row: dict[str, Any], metadata: dict[str, Any], algorithm: str
) -> tuple[str, str] | None:
    """Check explicit runtime transports, then reject unclassified fault-like fields."""
    from robot_sf.benchmark.fallback_policy import (
        runtime_fallback_or_degraded_marker,
    )

    missing = _required_runtime_marker(row)
    if missing is not None:
        return missing

    for group, fields in (
        ("runtime_objects", "runtime_fields"),
        ("counter_objects", "counter_fields"),
    ):
        for path in EXECUTION_EVIDENCE_LOCATIONS[group]:
            for name, obj in _runtime_objects(row, path):
                if not isinstance(obj, dict):
                    return name, "invalid"
                alias_marker = (
                    _runtime_alias_marker(obj, name=name) if group == "runtime_objects" else None
                )
                if alias_marker is not None:
                    return f"{name}.{alias_marker[0]}" if name else alias_marker[0], alias_marker[1]
                view = {
                    key: obj[key]
                    for key in EXECUTION_EVIDENCE_LOCATIONS[fields]
                    if key in obj and not _non_runtime_location(name, key)
                }
                counters = group == "counter_objects"
                error = _runtime_value_error(view, counters=counters)
                if error:
                    return f"{name}.{error}" if name else error, "invalid"
                # Containers retain their typed/bound check; their contents are
                # checked only at separately named runtime locations above.
                view = _bound_runtime_view(view, path, counters=counters, original=obj)
                decision_marker = _runtime_status_marker(view) or _runtime_decision_marker(view)
                if decision_marker is not None:
                    return f"{name}.{decision_marker[0]}" if name else decision_marker[
                        0
                    ], decision_marker[1]
                marker = runtime_fallback_or_degraded_marker(
                    view, expected_algorithm=algorithm, algorithm_metadata=metadata
                )
                if marker is not None:
                    field = (
                        marker[0].rsplit(".", 1)[-1] if group == "counter_objects" else marker[0]
                    )
                    return f"{name}.{field}" if name else field, marker[1]
    return _unclassified_runtime_marker(row)


def _non_runtime_location(name: str, key: str) -> bool:
    path = re.sub(r"\[\d+\]", ".*", name)
    full = (*path.split("."), key) if path else (key,)
    return full in _non_runtime_paths()


@cache
def _non_runtime_paths() -> frozenset[tuple[str, ...]]:
    """Compile the immutable, source-owned policy once per validator process."""
    return frozenset(
        tuple(location.split("."))
        for location in EXECUTION_EVIDENCE_LOCATIONS["non_runtime_locations"]
    )


def _transport_path(path: tuple[str, ...]) -> tuple[str | None, ...]:
    """Normalize declared array indices without conflating literal dictionary keys."""
    return tuple(None if part == "*" else part for part in path)


@cache
def _runtime_policy_index() -> tuple[Any, ...]:
    """Index the fixed runtime vocabulary; never cache decisions about evidence."""
    return (
        re.compile(EXECUTION_EVIDENCE_LOCATIONS["fault_key_pattern"], re.IGNORECASE),
        set(EXECUTION_EVIDENCE_LOCATIONS["runtime_fields"])
        | set(EXECUTION_EVIDENCE_LOCATIONS["runtime_aliases"]),
        set(EXECUTION_EVIDENCE_LOCATIONS["counter_fields"]),
        set(EXECUTION_EVIDENCE_LOCATIONS["mode_locations"])
        | set(EXECUTION_EVIDENCE_LOCATIONS["required_values"]),
        {_transport_path(path) for path in _non_runtime_paths()},
        {_transport_path(path) for path in EXECUTION_EVIDENCE_LOCATIONS["runtime_objects"]},
        {_transport_path(path) for path in EXECUTION_EVIDENCE_LOCATIONS["counter_objects"]},
    )


def _unclassified_runtime_marker(row: dict[str, Any]) -> tuple[str, str] | None:
    """Unknown fault vocabulary is evidence of an unaudited transport, even if false/zero.

    Source-key catalog membership is deliberately insufficient. Dict keys stay
    exact; only array indices normalize to '*'. No subtree is exempted.
    """
    # None represents an actual array index; a literal '*' dictionary key must
    # not acquire the array-only exemption. Keep dotted dictionary keys intact.
    (
        vocabulary,
        runtime_fields,
        counter_fields,
        required,
        optional,
        runtime_objects,
        counter_objects,
    ) = _runtime_policy_index()
    guarded_diagnostics = _GUARDED_DIAGNOSTIC_PATHS if _guarded_mixed_context(row) else frozenset()

    def classified(path: tuple[str | None, ...], key: str) -> bool:
        return (
            (path in runtime_objects and key in runtime_fields)
            or (path in counter_objects and key in counter_fields)
            or (*path, key) in required
            or (*path, key) in optional
            or (*path, key) in guarded_diagnostics
        )

    def walk(value: Any, path: tuple[str | None, ...] = ()) -> tuple[str, str] | None:
        children = []
        if isinstance(value, dict):
            for key, child in value.items():
                full = (*path, key)
                if vocabulary.search(key) and not classified(path, key):
                    return ".".join(
                        "*" if part is None else part for part in full
                    ), "unclassified_degraded"
                children.append((child, full))
        elif isinstance(value, list):
            children = [(child, (*path, None)) for child in value]
        for child, full in children:
            if isinstance(child, (dict, list)):
                marker = walk(child, full)
                if marker is not None:
                    return marker
        return None

    return walk(row)


# These leaves are source-emitted declarations, not runtime fault reports.
# guarded_ppo.py:1243 and safety_shield.py:117 emit exactly not_calibrated;
# map_runner.py:1637-1641 binds the RiskDWA shield configuration. No subtree
# or arbitrary status/value is excluded from the recursive runtime checker.
_GUARDED_DIAGNOSTIC_PATHS = frozenset(
    {
        ("algorithm_metadata", "safety_shield_contract", "calibration_status"),
        ("algorithm_metadata", "safety_shield_contract", "fallback_policy"),
        (
            "algorithm_metadata",
            "planner_runtime",
            "last_decision",
            "prediction",
            "calibration",
            "status",
        ),
        (
            "algorithm_metadata",
            "shield_stats",
            "last_decision",
            "prediction",
            "calibration",
            "status",
        ),
    }
)

_GUARDED_NATIVE_IDENTITIES = {
    ("algorithm_metadata", "planner_runtime", "planner_type"): "GuardedPPOAdapter",
    ("algorithm_metadata", "policy_semantics"): "guarded_policy_network_inference",
    ("event_ledger", "planner"): "guarded_ppo",
    ("scenario_params", "algo"): "guarded_ppo",
    (
        "algorithm_metadata",
        "paired_effect_metric_producer",
        "pairing",
        "key",
        "planner",
    ): "guarded_ppo",
    (
        "algorithm_metadata",
        "planner_runtime",
        "fallback_diagnostics",
        "planner_type",
    ): "RiskDWAPlannerAdapter",
}

# guarded_ppo.py:820/981/991/1062/1098/1158/1166. The reviewed matrix
# has no optional prior or uncertainty branch; unknown decisions refuse.
_GUARDED_DECISION_POLICIES = {
    "goal_reached": "RiskDWAPlannerAdapter",
    "ppo_clear": "RiskDWAPlannerAdapter",
    "ppo_safe": "RiskDWAPlannerAdapter",
    "fallback_safe": "RiskDWAPlannerAdapter",
    "fallback_best_effort": "RiskDWAPlannerAdapter",
    "stop_safe": "stop",
    "stop_best_effort": "stop",
}


def required_execution_mode(algorithm: str, capabilities: dict) -> str:
    """Honor the two reviewed registry defaults; retain other command-mode rules."""
    if (
        algorithm == "guarded_ppo"
        and capabilities.get("execution_mode") == "mixed"
        and capabilities.get("supports_native_commands") is True
        and capabilities.get("supports_adapter_commands") is True
        and capabilities.get("adapter_name") == "guarded_ppo_action_to_unicycle"
    ):
        return "mixed"
    if (
        algorithm == "hybrid_rule_local_planner"
        and capabilities.get("execution_mode") == "adapter"
        and capabilities.get("supports_native_commands") is True
        and capabilities.get("supports_adapter_commands") is True
        and capabilities.get("adapter_name") == "HybridRuleLocalPlannerAdapter"
    ):
        return "adapter"
    return "native" if capabilities["supports_native_commands"] else "adapter"


_HYBRID_NATIVE_IDENTITIES = {
    ("algo",): "hybrid_rule_local_planner",
    ("algorithm_metadata", "algorithm"): "hybrid_rule_local_planner",
    ("algorithm_metadata", "canonical_algorithm"): "hybrid_rule_local_planner",
    ("algorithm_metadata", "planner_contract", "planner_id"): "hybrid_rule_local_planner",
    ("algorithm_metadata", "policy_semantics"): "hybrid_rule_deterministic_local_planner",
    ("event_ledger", "planner"): "hybrid_rule_local_planner",
    ("scenario_params", "algo"): "hybrid_rule_local_planner",
    (
        "algorithm_metadata",
        "paired_effect_metric_producer",
        "pairing",
        "key",
        "planner",
    ): "hybrid_rule_local_planner",
}


def _hybrid_adapter_context(row: dict[str, Any]) -> bool:
    """Bind the registered hybrid adapter to its complete native row identity."""
    from robot_sf.benchmark.algorithm_metadata import enrich_algorithm_metadata

    algorithm = "hybrid_rule_local_planner"
    if (
        any(
            list(_runtime_objects(row, path)) != [(".".join(path), expected)]
            for path, expected in _HYBRID_NATIVE_IDENTITIES.items()
        )
        or ("algorithm" in row and row["algorithm"] != algorithm)
        or required_execution_mode(
            algorithm, enrich_algorithm_metadata(algo=algorithm)["planner_kinematics"]
        )
        != "adapter"
    ):
        return False
    metadata = row["algorithm_metadata"]
    profile = metadata.get("planner_kinematics")
    if not isinstance(profile, dict) or any(
        profile.get(key) != value or type(profile.get(key)) is not type(value)
        for key, value in {
            "execution_mode": "adapter",
            "supports_native_commands": True,
            "supports_adapter_commands": True,
            "adapter_active": True,
            "adapter_name": "HybridRuleLocalPlannerAdapter",
        }.items()
    ):
        return False
    # Native hybrid diagnostics do not emit planner_type. A supplied assertion
    # must agree; absence is not filled with a manufactured adapter identity.
    runtime_identity = list(
        _runtime_objects(row, ("algorithm_metadata", "planner_runtime", "planner_type"))
    )
    return not runtime_identity or runtime_identity == [
        ("algorithm_metadata.planner_runtime.planner_type", "HybridRuleLocalPlannerAdapter")
    ]


def _guarded_mixed_context(row: dict[str, Any]) -> bool:
    """Authenticate the source-defined guarded profile and its exact declarations."""
    from robot_sf.benchmark.algorithm_metadata import enrich_algorithm_metadata
    from robot_sf.benchmark.fallback_policy import is_verified_guarded_ppo

    metadata = row.get("algorithm_metadata")
    if (
        not isinstance(metadata, dict)
        or row.get("algo", row.get("algorithm")) != "guarded_ppo"
        or ("algorithm" in row and row["algorithm"] != "guarded_ppo")
        or not is_verified_guarded_ppo(metadata, expected_algorithm="guarded_ppo")
        or required_execution_mode(
            "guarded_ppo", enrich_algorithm_metadata(algo="guarded_ppo")["planner_kinematics"]
        )
        != "mixed"
    ):
        return False
    if any(
        list(_runtime_objects(row, path)) != [(".".join(path), expected)]
        for path, expected in _GUARDED_NATIVE_IDENTITIES.items()
    ):
        return False
    planner_contract = metadata.get("planner_contract")
    if (
        metadata.get("algorithm") != "ppo"
        or metadata.get("canonical_algorithm") != "guarded_ppo"
        or not isinstance(planner_contract, dict)
        or planner_contract.get("planner_id") != "guarded_ppo"
    ):
        return False
    profile = metadata.get("planner_kinematics")
    if not isinstance(profile, dict) or any(
        profile.get(key) != value or type(profile.get(key)) is not type(value)
        for key, value in {
            "execution_mode": "mixed",
            "supports_native_commands": True,
            "supports_adapter_commands": True,
            "adapter_active": True,
            "adapter_name": "guarded_ppo_action_to_unicycle",
        }.items()
    ):
        return False
    contract = metadata.get("safety_shield_contract")
    if not isinstance(contract, dict) or any(
        contract.get(key) != value
        for key, value in {
            "schema_version": "safety-shield-contract.v1",
            "shield_name": "GuardedPPOAdapter",
            "prediction_source": "short_horizon_rollout",
            "fallback_policy": "RiskDWAPlannerAdapter",
            "calibration_status": "not_calibrated",
        }.items()
    ):
        return False
    decision_identities = []
    for container in ("planner_runtime", "shield_stats"):
        owner = metadata.get(container)
        decision = owner.get("last_decision") if isinstance(owner, dict) else None
        prediction = decision.get("prediction") if isinstance(decision, dict) else None
        if (
            not isinstance(decision, dict)
            or decision.get("schema_version") != "shield-decision.v1"
            or not isinstance(prediction, dict)
            or prediction.get("source") != "short_horizon_rollout"
            or prediction.get("calibration") != {"status": "not_calibrated"}
        ):
            return False
        label = decision.get("decision_label")
        expected_policy = _GUARDED_DECISION_POLICIES.get(label) if isinstance(label, str) else None
        state = decision.get("fallback_controller_state")
        if (
            expected_policy is None
            or not isinstance(state, dict)
            or state.get("policy") != expected_policy
            or state.get("prior_available") is not False
        ):
            return False
        decision_identities.append((label, expected_policy))
    return decision_identities[0] == decision_identities[1]


def _guarded_commands_complete(row: dict[str, Any]) -> bool:
    """Require both finite command axes and the source conversion at every step."""
    try:
        trace = row["algorithm_metadata"]["simulation_step_trace"]
        for step in trace["steps"]:
            planner = step["planner"]
            for name in ("selected_action", "applied_environment_action"):
                command = planner[name]
                if not isinstance(command, dict) or set(command) != {
                    "linear_velocity",
                    "angular_velocity",
                }:
                    return False
                if not all(
                    type(value) in (int, float) and math.isfinite(value)
                    for value in command.values()
                ):
                    return False
            conversion = planner["action_conversion"]
            if (
                not isinstance(conversion, dict)
                or conversion.get("schema_version") != "policy-action-conversion.v1"
                or conversion.get("kind") != "unicycle_velocity_to_acceleration"
            ):
                return False
        return True
    except (KeyError, TypeError, AttributeError, OverflowError, ValueError):
        return False


def _execution_evidence(row: dict[str, Any]) -> dict[str, Any]:
    from robot_sf.benchmark.fallback_policy import resolve_execution_mode
    from scripts.validation.run_empty_world_sweep import _trace_complete

    metadata = row.get("algorithm_metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    algorithm = row.get("algo") or row.get("algorithm")
    mode = resolve_execution_mode(metadata)
    modes = [
        value
        for path in EXECUTION_EVIDENCE_LOCATIONS["mode_locations"]
        for _, value in _runtime_objects(row, path)
    ]
    guarded_mixed = mode == "mixed" and _guarded_mixed_context(row)
    hybrid_adapter = (
        algorithm == "hybrid_rule_local_planner"
        and mode == "adapter"
        and _hybrid_adapter_context(row)
    )
    if algorithm == "hybrid_rule_local_planner" and not hybrid_adapter:
        mode = "unknown"
    if (mode not in {"native", "adapter"} and not guarded_mixed) or any(
        value != mode for value in modes
    ):
        mode = "unknown"
    profile = metadata.get("planner_kinematics") or {}
    if isinstance(profile, dict) and any(
        path[-1] in profile for path in EXECUTION_EVIDENCE_LOCATIONS["mode_capabilities"]
    ):
        if not guarded_mixed and profile.get(f"supports_{mode}_commands") is not True:
            mode = "unknown"
    marker = (
        _controller_runtime_marker(row, metadata, algorithm)
        if isinstance(algorithm, str)
        else ("algorithm", "invalid")
    )
    required = all(
        list(_runtime_objects(row, path)) == [(".".join(path), expected)]
        for path, (expected, _) in EXECUTION_EVIDENCE_LOCATIONS["required_values"].items()
    )
    try:
        trace = metadata.get("simulation_step_trace") or {}
        steps = trace.get("steps") or []
        complete = (
            type(row.get("steps")) is int
            and row["steps"] > 0
            and _trace_complete(row)
            and (not (guarded_mixed or hybrid_adapter) or _guarded_commands_complete(row))
            and all(
                isinstance((step.get("planner") or {}).get("selected_action"), dict)
                and step["planner"]["selected_action"]
                for step in steps
            )
        )
    except (AttributeError, TypeError, ValueError):
        complete = False
    executed = (
        required
        and complete
        and isinstance(algorithm, str)
        and bool(algorithm.strip())
        and row.get("controller_executed", True) is True
    )
    return receipt_arm_execution.carry_execution_proof(
        row,
        {
            "algorithm": algorithm or "unknown",
            "execution_mode": mode,
            "controller_executed": bool(executed),
            "fallback": row.get("fallback") is True
            or (marker is not None and "fallback" in str(marker)),
            "degraded": row.get("degraded") is True or marker is not None,
        },
    )


def build_rows_and_classifications(
    head_rows: dict[tuple[str, str, int], dict[str, Any]],
    baseline_rows: dict[tuple[str, str, int], dict[str, Any]],
    scope: dict[str, Any],
    *,
    classification_class: str,
    evidence_base_uri: str,
    execution_evidence: dict | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Join head/baseline sweep rows into gate payload rows and classifications."""
    rows: list[dict[str, Any]] = []
    classifications: list[dict[str, Any]] = []
    for arm in scope["arms"]:
        for map_id in scope["maps"]:
            for seed in DEFAULT_SEEDS:
                key = (arm, map_id, seed)
                if key not in head_rows:
                    raise ValueError(f"head sweep omitted {key}")
                if key not in baseline_rows:
                    raise ValueError(f"baseline sweep omitted {key}")
                head = head_rows[key]
                baseline = baseline_rows[key]
                success = _row_success(head)
                collisions = _row_collisions(head)
                baseline_success = _row_success(baseline)
                baseline_collisions = _row_collisions(baseline)
                rows.append(
                    {
                        "arm": arm,
                        "map": map_id,
                        "seed": seed,
                        "success": success,
                        "collisions": collisions,
                        "baseline_success": baseline_success,
                        "baseline_collisions": baseline_collisions,
                        "scheduler_job_id": head.get("scheduler_job_id"),
                        "scheduler_node": head.get("scheduler_node"),
                        "baseline_scheduler_job_id": baseline.get("scheduler_job_id"),
                        "baseline_scheduler_node": baseline.get("scheduler_node"),
                        **(
                            _execution_evidence(head)
                            if execution_evidence is None
                            else execution_evidence[key]
                        ),
                    }
                )
                rows[-1] = receipt_arm_execution.carry_execution_proof(
                    head if execution_evidence is None else execution_evidence[key], rows[-1]
                )
                evidence = f"{evidence_base_uri.rstrip('/')}/{arm}/{map_id}/{seed}"
                if baseline_success and not success:
                    classifications.append(
                        {
                            "arm": arm,
                            "map": map_id,
                            "seed": seed,
                            "kind": "success_to_failure",
                            "class": classification_class,
                            "evidence": evidence,
                            "count": 1,
                        }
                    )
                if collisions > baseline_collisions:
                    classifications.append(
                        {
                            "arm": arm,
                            "map": map_id,
                            "seed": seed,
                            "kind": "new_collision",
                            "class": classification_class,
                            "evidence": evidence,
                            "count": collisions - baseline_collisions,
                        }
                    )
    totals = {
        "episodes": len(rows),
        "new_failures": sum(row["kind"] == "success_to_failure" for row in classifications),
        "new_collisions": sum(
            row["count"] for row in classifications if row["kind"] == "new_collision"
        ),
    }
    return rows, classifications, totals


def derive_valid_comparison(
    head_rows: dict,
    baseline_rows: dict,
    scope: dict,
    *,
    classification_class: str,
    evidence_base_uri: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Shared collection/admission trust decision, derived entirely from raw rows."""
    from robot_sf.benchmark.algorithm_metadata import (
        enrich_algorithm_metadata,
    )

    expected = {
        (arm, map_id, seed)
        for arm in scope["arms"]
        for map_id in scope["maps"]
        for seed in DEFAULT_SEEDS
    }
    head_evidence = {}
    for sources in (head_rows, baseline_rows):
        if set(sources) != expected:
            raise behaviour_receipt.HardwareIdentityError(
                "raw execution invalid: scope coverage differs"
            )
        for key, raw in sources.items():
            evidence = _execution_evidence(raw)
            arm, scenario, _seed = key
            algorithm = receipt_arm_execution.expected_algorithm(
                arm,
                scenario,
                scope.get("arm_algorithms", {}).get(arm, arm),
                row=raw,
                source_sha=raw.get("git_hash"),
                repo_root=behaviour_receipt.ROOT,
            )
            capabilities = enrich_algorithm_metadata(algo=algorithm)["planner_kinematics"]
            require_valid_execution(evidence, algorithm, capabilities)
            if sources is head_rows:
                head_evidence[key] = evidence
    return build_rows_and_classifications(
        head_rows,
        baseline_rows,
        scope,
        classification_class=classification_class,
        evidence_base_uri=evidence_base_uri,
        execution_evidence=head_evidence,
    )


def require_valid_execution(evidence: dict, algorithm: str, capabilities: dict) -> None:
    """One execution rule for raw derivation and projected receipt validation."""
    reason = None
    mode = required_execution_mode(algorithm, capabilities)
    if evidence.get("algorithm") != algorithm:
        reason = "row algorithm differs from reviewed arm"
    elif evidence.get("execution_mode") != mode or not (
        capabilities["supports_native_commands"] or capabilities["supports_adapter_commands"]
    ):
        reason = "command execution mode does not satisfy the algorithm registry"
    elif _audit_status([evidence]) == "fail":
        reason = "intended solver/controller did not execute or trace is incomplete"
    elif _audit_status([evidence]) != "pass":
        reason = "fallback or degraded execution cannot satisfy the receipt"
    if reason is not None:
        raise behaviour_receipt.HardwareIdentityError(f"raw execution invalid: {reason}")


def _audit_status(rows: list[dict[str, Any]]) -> str:
    if not rows or any(
        row.get("controller_executed") is not True
        or (
            row.get("execution_mode") not in {"native", "adapter"}
            and not (row.get("execution_mode") == "mixed" and row.get("algorithm") == "guarded_ppo")
        )
        or not row.get("algorithm")
        or row.get("algorithm") == "unknown"
        or type(row.get("fallback")) is not bool
        or type(row.get("degraded")) is not bool
        for row in rows
    ):
        return "fail"
    return "degraded" if any(row["fallback"] or row["degraded"] for row in rows) else "pass"


def write_real_row_audit(
    path: Path,
    *,
    source_sha: str,
    rows: list[dict[str, Any]],
    classifications: list[dict[str, Any]],
) -> str:
    """Write the real-row audit artifact and return its byte digest."""
    audit = {
        "schema_version": "behaviour-real-row-audit.v1",
        "source_sha": source_sha,
        "status": _audit_status(rows),
        "rows": len(rows),
        "controller_executed_rows": sum(row.get("controller_executed") is True for row in rows),
        "fallback_rows": sum(row.get("fallback") is True for row in rows),
        "degraded_rows": sum(row.get("degraded") is True for row in rows),
        "classification_rows": len(classifications),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(audit))
    return _sha256(path)


def _archive_source_bundle(
    shard: Path, bundle: Path, *, extra_files: tuple[Path, ...] = ()
) -> None:
    """Preserve an existing identical evidence bundle; refuse conflicting bytes."""
    files = [
        *receipt_provenance.source_files(shard),
        shard / receipt_provenance.BINDING_FILE,
        *extra_files,
    ]
    expected = {path.relative_to(shard).as_posix(): path.read_bytes() for path in files}
    if bundle.exists():
        with tarfile.open(bundle, "r:gz") as archive:
            actual = {}
            for member in archive:
                stream = archive.extractfile(member) if member.isfile() else None
                if stream is None or member.name in actual:
                    raise behaviour_receipt.HardwareIdentityError(
                        "hardware comparison invalid: provenance digest bundle is invalid"
                    )
                actual[member.name] = stream.read()
        if actual != expected:
            raise behaviour_receipt.HardwareIdentityError(
                "hardware comparison invalid: provenance digest bundle already exists with different bytes"
            )
        return
    bundle.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(bundle, "x:gz") as archive:
        for path in files:
            archive.add(path, arcname=path.relative_to(shard).as_posix(), recursive=False)


def _equivalence_source_artifacts(
    repo_root: Path, receipt_id: str, output_dir: Path, scheduler: SweepRun, baseline: SweepRun
) -> dict:
    """Archive unchanged sealed inputs and the pinned wrapper for independent admission."""
    artifacts = {}
    for side, run in (("head", scheduler), ("baseline", baseline)):
        if run.hardware is None:
            raise behaviour_receipt.HardwareIdentityError(
                "hardware comparison invalid: unverified sweep cannot provide source artifacts"
            )
        identity = behaviour_receipt.hardware_identity(run.hardware)
        references = {}
        for shard in run.shard_dirs:
            receipt_provenance.verify_binding(
                shard, shard.name, identity["node"], source_sha=run.source_sha, arm_role=side
            )
            bundle = output_dir / f"{receipt_id}-sources" / side / f"{shard.name}.tar.gz"
            _archive_source_bundle(
                shard,
                bundle,
                extra_files=(shard / "acquisition_harness.py", shard / "execution_request.json"),
            )
            references[shard.name] = {
                "path": bundle.relative_to(repo_root).as_posix(),
                "sha256": _sha256(bundle),
                "binding": {
                    "path": receipt_provenance.BINDING_FILE,
                    "sha256": _sha256(shard / receipt_provenance.BINDING_FILE),
                },
            }
        artifacts[side] = references
    return artifacts


def write_receipt_and_header(  # noqa: PLR0913
    *,
    repo_root: Path,
    receipt_id: str,
    head_sha: str,
    scheduler: SweepRun,
    baseline: SweepRun,
    baseline_release: str,
    baseline_body_id: str,
    baseline_config_sha256: str,
    baseline_differences: list[str],
    scope: dict[str, Any],
    rows: list[dict[str, Any]],
    classifications: list[dict[str, Any]],
    totals: dict[str, int],
    audit_uri: str,
    audit_sha256: str,
    audit_source_sha: str,
    refute_review_uri: str,
    output_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    """Write the committed rows payload and return the compact PR header."""
    hardware = {"head": scheduler.hardware, "baseline": baseline.hardware}
    identities = behaviour_receipt.validate_hardware_pair(hardware)
    if scheduler.source_sha == baseline.source_sha:
        raise behaviour_receipt.HardwareIdentityError(
            "producer source invalid: head and baseline share a producer commit"
        )
    derived, _, derived_totals = derive_valid_comparison(
        _load_run_rows(scheduler, "head"),
        _load_run_rows(baseline, "baseline"),
        scope,
        classification_class="known_limitation",
        evidence_base_uri="verified-source",
    )
    if rows != derived or totals != derived_totals:
        raise behaviour_receipt.HardwareIdentityError(
            "raw execution invalid: supplied receipt projection differs from sealed sources"
        )
    behaviour_receipt.validate_scheduler_rows(rows, identities)
    source_artifacts = {}
    for side, run in (("head", scheduler), ("baseline", baseline)):
        references = {}
        for shard in run.shard_dirs:
            receipt_provenance.verify_binding(
                shard,
                shard.name,
                identities[side]["node"],
                source_sha=run.source_sha,
                arm_role=side,
            )
            bundle = output_dir / f"{receipt_id}-sources" / side / f"{shard.name}.tar.gz"
            _archive_source_bundle(shard, bundle)
            references[shard.name] = {
                "path": bundle.relative_to(repo_root).as_posix(),
                "sha256": _sha256(bundle),
                "binding": {
                    "path": receipt_provenance.BINDING_FILE,
                    "sha256": _sha256(shard / receipt_provenance.BINDING_FILE),
                },
            }
        source_artifacts[side] = references
    payload = {
        "schema_version": "behaviour-change-rows.v1",
        "hardware": hardware,
        "source_artifacts": source_artifacts,
        "rows": rows,
        "classifications": classifications,
    }
    receipt_path = output_dir / f"{receipt_id}.json"
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    raw = _json_bytes(payload)
    receipt_path.write_bytes(raw)
    relative = receipt_path.relative_to(repo_root).as_posix()
    vehicle_body = {
        "vehicle_id": scope["vehicle_id"],
        "exceptions": scope["exceptions"],
    }
    job_id = scheduler.job_id
    if scheduler.job_id != baseline.job_id:
        job_id = f"{scheduler.job_id}_{baseline.job_id}"
    job_id = job_id.replace(",", "_")
    header = {
        "schema_version": "behaviour-change-receipt-header.v2",
        "head_sha": head_sha,
        "scheduler": {
            "kind": "slurm",
            "job_id": job_id,
            "source_sha": scheduler.source_sha,
        },
        "artifact": {"uri": scheduler.artifact_uri, "sha256": scheduler.artifact_sha256},
        "baseline": {
            "release": baseline_release,
            "source_sha": baseline.source_sha,
            "artifact_uri": baseline.artifact_uri,
            "artifact_sha256": baseline.artifact_sha256,
            "comparison": "development_reconstruction",
            "body_id": baseline_body_id,
            "config_sha256": baseline_config_sha256,
            "differences": baseline_differences,
        },
        "vehicle": {
            "id": scope["vehicle_id"],
            "body_sha256": _canonical_digest(vehicle_body),
        },
        "classifications": {
            "count": len(classifications),
            "sha256": _canonical_digest(classifications),
        },
        "exceptions": scope["exceptions"],
        "totals": totals,
        "interaction_audit": {
            "uri": audit_uri,
            "sha256": audit_sha256,
            "source_sha": audit_source_sha,
        },
        "refute_review": {
            "head_sha": head_sha,
            "verdict": "pending_independent_review",
            "uri": refute_review_uri,
        },
        "scope_sha256": _canonical_digest(scope),
        "rows_artifact": {"path": relative, "sha256": hashlib.sha256(raw).hexdigest()},
    }
    return receipt_path, header


def _load_scope(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head-sha", required=True)
    parser.add_argument(
        "--receipt-route", choices=("full", "default-path-equivalence"), default="full"
    )
    parser.add_argument(
        "--base-ref", default="origin/main", help="Current main for PR-only equivalence"
    )
    parser.add_argument("--tooling-sha", help="Separately pinned acquisition harness source SHA")
    parser.add_argument("--receipt-context", choices=("pr", "release"), default="pr")
    parser.add_argument(
        "--head-source-sha", help="Recorded execution source before receipt-only commits"
    )
    parser.add_argument("--baseline", default="latest")
    parser.add_argument("--repo", default="ll7/robot_sf_ll7")
    parser.add_argument("--receipt-id", required=True)
    parser.add_argument("--mode", choices=("existing", "local", "sbatch"), default="existing")
    parser.add_argument(
        "--sbatch-node",
        help="Required for sbatch: one literal node shared by baseline and head jobs",
    )
    parser.add_argument("--submit-only", action="store_true")
    parser.add_argument(
        "--comparison-only",
        action="store_true",
        help="Private equivalence diagnostic only; never emit a receipt/header",
    )
    parser.add_argument("--head-sweep-dir", type=Path)
    parser.add_argument("--baseline-sweep-dir", type=Path)
    parser.add_argument("--work-dir", type=Path, default=Path("output/behaviour_receipt"))
    parser.add_argument("--output-dir", type=Path, default=Path("receipts/behaviour"))
    parser.add_argument("--scope-path", type=Path, default=behaviour_receipt.SCOPE_PATH)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--arms", nargs="*")
    parser.add_argument("--scenarios", nargs="*")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--head-artifact-uri", default=f"{PLACEHOLDER_EVIDENCE_URI}/head")
    parser.add_argument("--head-artifact-sha256")
    parser.add_argument("--baseline-artifact-uri", default=f"{PLACEHOLDER_EVIDENCE_URI}/baseline")
    parser.add_argument("--baseline-artifact-sha256")
    parser.add_argument("--job-id", default="1")
    parser.add_argument("--baseline-job-id", default="1")
    parser.add_argument("--audit-uri", default=f"{PLACEHOLDER_EVIDENCE_URI}/real-row-audit.json")
    parser.add_argument("--audit-source-sha")
    parser.add_argument("--refute-review-uri", default=PLACEHOLDER_REVIEW_URI)
    parser.add_argument(
        "--classification-class",
        choices=(
            "defect",
            "known_limitation",
            "vehicle_specific_infeasible",
            "measurement_inconsistency",
        ),
        default="known_limitation",
    )
    parser.add_argument("--classification-evidence-base-uri", default=PLACEHOLDER_EVIDENCE_URI)
    parser.add_argument("--baseline-body-id", default="differential_drive_r1m")
    parser.add_argument("--baseline-config-sha256")
    parser.add_argument(
        "--baseline-difference",
        action="append",
        dest="baseline_differences",
        default=[],
    )
    return parser.parse_args(argv)


def _prepare_sweeps(
    args: argparse.Namespace, repo_root: Path, baseline_source: str
) -> tuple[SweepRun, SweepRun]:
    seeds = _check_seed_range(list(args.seeds))
    work_dir = (repo_root / args.work_dir).resolve()
    if args.mode == "existing":
        if args.head_sweep_dir is None or args.baseline_sweep_dir is None:
            raise ValueError("--mode existing requires --head-sweep-dir and --baseline-sweep-dir")
        behaviour_receipt.validate_hardware_pair(
            {
                "head": _read_hardware(
                    args.head_sweep_dir.resolve(),
                    args.head_source_sha or args.head_sha,
                    args.job_id,
                ),
                "baseline": _read_hardware(
                    args.baseline_sweep_dir.resolve(), baseline_source, args.baseline_job_id
                ),
            }
        )
        return (
            _existing_sweep(
                args.head_source_sha or args.head_sha,
                args.head_sweep_dir.resolve(),
                artifact_uri=args.head_artifact_uri,
                artifact_sha256=args.head_artifact_sha256,
                job_id=args.job_id,
            ),
            _existing_sweep(
                baseline_source,
                args.baseline_sweep_dir.resolve(),
                artifact_uri=args.baseline_artifact_uri,
                artifact_sha256=args.baseline_artifact_sha256,
                job_id=args.baseline_job_id,
            ),
        )
    if args.mode == "local":
        raise behaviour_receipt.HardwareIdentityError(
            "hardware comparison invalid: receipt evidence requires a Slurm allocation; use sbatch"
        )
    return (
        _submit_sbatch_sweep(
            repo_root,
            work_dir,
            "head",
            args.head_sha,
            seeds,
            args=args,
        ),
        _submit_sbatch_sweep(
            repo_root,
            work_dir,
            "baseline",
            baseline_source,
            seeds,
            args=args,
        ),
    )


def _sweep_episode_paths(sweep_dir: Path) -> dict[tuple[str, str, int], str]:
    """Locate every projected digest's raw episode in its hashed sweep archive."""
    paths = {}
    for summary in sorted(sweep_dir.glob("episodes_*.jsonl")):
        suite = summary.stem.removeprefix("episodes_")
        for line in summary.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                paths[(row["arm"], row["scenario"], row["seed"])] = (
                    f"campaigns/empty_world_{suite}/{row['source_file']}"
                )
    return paths


def _sweep_execution_statuses(sweep_dir: Path) -> dict[tuple[str, str, int], str]:
    """Keep verified summary transport status separate from untouched raw episode bytes."""
    statuses = {}
    for summary in sorted(sweep_dir.glob("episodes_*.jsonl")):
        for line in summary.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                statuses[(row["arm"], row["scenario"], row["seed"])] = row["execution_status"]
    return statuses


def _run_episode_transport(run: SweepRun) -> tuple[dict, dict]:
    """Join verified job-directory paths and summary status without altering raw rows."""
    paths, statuses = {}, {}
    for shard in run.shard_dirs:
        for key, path in _sweep_episode_paths(shard).items():
            if key in paths:
                raise ValueError("duplicate episode transport across scheduler shards")
            paths[key] = (shard.relative_to(run.output_dir) / path).as_posix()
        statuses.update(_sweep_execution_statuses(shard))
    return paths, statuses


def _verify_equivalence_tooling(
    sweeps: tuple[SweepRun, SweepRun], tooling_sha: str, seeds: list[int], repo_root: Path
) -> tuple[str, str, str]:
    """Verify separately pinned script bytes and the same archived acquisition wrapper."""
    tooling_path = "scripts/validation/run_empty_world_sweep.py"
    tooling_bytes = subprocess.check_output(
        ["git", "show", f"{tooling_sha}:{tooling_path}"], cwd=repo_root
    )
    tooling_digest = hashlib.sha256(tooling_bytes).hexdigest()
    harnesses = set()
    for shard in (directory for sweep in sweeps for directory in sweep.shard_dirs):
        wrapper = shard / "acquisition_harness.py"
        if wrapper.is_symlink() or not wrapper.is_file():
            raise ValueError("equivalence acquisition wrapper must be an archived regular file")
        wrapper_digest = _sha256(wrapper)
        for summary in shard.glob("episodes_*.jsonl"):
            suite = summary.stem.removeprefix("episodes_")
            metadata = json.loads((shard / f"execution_{suite}.json").read_text())
            if (
                metadata.get("tooling_sha") != tooling_sha
                or metadata.get("tooling_path") != tooling_path
                or metadata.get("tooling_file_sha256") != tooling_digest
            ):
                raise ValueError("equivalence acquisition tooling differs from the pinned Git blob")
            harness = metadata.get("private_harness_sha256")
            if not behaviour_receipt._digest(harness) or harness != wrapper_digest:
                raise ValueError("equivalence acquisition wrapper bytes differ from metadata")
            harnesses.add(harness)
            if metadata.get("seeds") != seeds:
                raise ValueError("equivalence acquisition development seed slice differs")
    if len(harnesses) != 1:
        raise ValueError("equivalence base/head acquisition harness differs")
    return tooling_path, tooling_digest, next(iter(harnesses))


def _validate_equivalence_input_json(args: argparse.Namespace) -> None:
    """Decode no source claims before rejecting ambiguous fields in this route."""
    from scripts.ci import default_path_equivalence as equivalence

    for directory, jobs in (
        (args.head_sweep_dir, args.job_id),
        (args.baseline_sweep_dir, args.baseline_job_id),
    ):
        if directory is None:
            raise ValueError("existing equivalence requires both exact-source sweep directories")
        for shard in _shard_dirs(directory.resolve(), jobs):
            equivalence.validate_source_json(shard)


def _validate_equivalence_sources(
    scheduler: SweepRun, baseline: SweepRun, identities: dict, source_context: dict
) -> None:
    """Private comparison uses the same sealed source proof as committed admission."""
    for side, sweep in (("head", scheduler), ("baseline", baseline)):
        for shard in sweep.shard_dirs:
            raw = receipt_provenance.verified_raw_rows(
                shard,
                shard.name,
                identities[side]["node"],
                source_sha=sweep.source_sha,
                arm_role=side,
            )
            behaviour_receipt._equivalence_source_projection(
                shard, raw, source_context, identities[side]
            )


def _make_equivalence_receipt(args: argparse.Namespace, repo_root: Path) -> int:
    """Consume exact-source sweeps acquired once per source on one pinned node.

    Acquisition may use a separately pinned harness. Existing mode intentionally
    requires its recorded identity instead of stamping an unverified identity
    onto a source checkout's own acquisition script.
    """
    from scripts.ci import default_path_equivalence as equivalence

    if args.receipt_context != "pr":
        raise ValueError("equivalence refused: releases require full behaviour receipt")
    if args.mode != "existing" or args.submit_only:
        raise ValueError("equivalence requires existing exact-source, pinned-harness sweeps")
    if args.seeds != list(equivalence.SEEDS):
        raise ValueError("equivalence requires --seeds 1001 1002 1003")
    if not behaviour_receipt._digest(args.tooling_sha, 40):
        raise ValueError("equivalence requires --tooling-sha with exact acquisition source")
    base_source = _full_sha(repo_root, args.base_ref)
    if base_source != behaviour_receipt.current_main_source(args.repo):
        raise ValueError(
            "equivalence freshness refused: recorded baseline must match current remote main"
        )
    execution_source = args.head_source_sha or args.head_sha
    behaviour_receipt._hardware_require(
        execution_source != base_source,
        "producer source invalid: head and baseline share a producer commit",
    )
    changed = _git(
        repo_root, "diff", "--name-only", "--no-renames", base_source, execution_source
    ).splitlines()
    transitions = equivalence.classify_scope(changed, base_source, execution_source)
    _validate_equivalence_input_json(args)
    scheduler, baseline = _prepare_sweeps(args, repo_root, base_source)
    hardware = {"head": scheduler.hardware, "baseline": baseline.hardware}
    identities = behaviour_receipt.validate_hardware_pair(hardware)
    tooling_path, tooling_digest, harness_digest = _verify_equivalence_tooling(
        (scheduler, baseline), args.tooling_sha, args.seeds, repo_root
    )
    scope = _load_scope(args.scope_path)
    equivalence.validate_scope_entries(scope)
    if len(scope["arms"]) != 14 or len(scope["maps"]) != 51:
        raise ValueError("equivalence requires the full reviewed 14-arm/51-slot matrix")
    source_context = {
        "hardware": hardware,
        "tooling_sha": args.tooling_sha,
        "tooling_path": tooling_path,
        "tooling_file_sha256": tooling_digest,
        "harness_sha256": harness_digest,
        "seeds": args.seeds,
    }
    _validate_equivalence_sources(scheduler, baseline, identities, source_context)
    head_paths, head_statuses = _run_episode_transport(scheduler)
    baseline_paths, baseline_statuses = _run_episode_transport(baseline)
    rows = equivalence.comparison_rows(
        _load_run_rows(scheduler, "head", preserve_raw=True),
        _load_run_rows(baseline, "baseline", preserve_raw=True),
        scope,
        head_execution_statuses=head_statuses,
        baseline_execution_statuses=baseline_statuses,
        allocation_identities=identities,
    )
    behaviour_receipt._require(
        len(rows) == 2142, "equivalence refused: expected 2142 paired episodes"
    )
    behaviour_receipt.validate_scheduler_rows(rows, identities)
    for row in rows:
        key = behaviour_receipt._key(row)
        row.update(head_path=head_paths[key], baseline_path=baseline_paths[key])
    changed_rows = [row for row in rows if row["head_sha256"] != row["baseline_sha256"]]
    diagnostic_path = (repo_root / args.work_dir / f"{args.receipt_id}.comparison.json").resolve()
    diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostic_path.write_bytes(
        _json_bytes(
            {
                "schema_version": "default-path-equivalence-diagnostic.v1",
                "head_source_sha": execution_source,
                "baseline_source_sha": base_source,
                "rows": rows,
                "changed_episodes": len(changed_rows),
                "result": "refused" if changed_rows else "equal_pending_independent_review",
            }
        )
    )
    if args.comparison_only:
        print(
            json.dumps(
                {
                    "result": "refused" if changed_rows else "equal",
                    "episodes": len(rows),
                    "changed_episodes": len(changed_rows),
                    "diagnostic_file": diagnostic_path.name,
                }
            )
        )
        return 1 if changed_rows else 0
    if changed_rows:
        raise ValueError(
            f"equivalence refused: changed behavioral byte in {len(changed_rows)} episodes; inspect comparison diagnostic"
        )
    job_id = (
        scheduler.job_id
        if scheduler.job_id == baseline.job_id
        else f"{scheduler.job_id}_{baseline.job_id}"
    ).replace(",", "_")
    source_artifacts = _equivalence_source_artifacts(
        repo_root, args.receipt_id, (repo_root / args.output_dir).resolve(), scheduler, baseline
    )
    receipt = {
        "schema_version": equivalence.VERSION,
        "admission_context": "pull_request",
        "declaration": "default_benchmark_behaviour_unchanged",
        "head_source_sha": execution_source,
        "baseline": {
            "source_sha": base_source,
            "artifact": {"uri": baseline.artifact_uri, "sha256": baseline.artifact_sha256},
        },
        "scheduler": {"source_sha": execution_source, "job_id": job_id},
        "tooling_sha": args.tooling_sha,
        "tooling_path": tooling_path,
        "tooling_file_sha256": tooling_digest,
        "harness_sha256": harness_digest,
        "harness_path": "acquisition_harness.py",
        "artifact": {"uri": scheduler.artifact_uri, "sha256": scheduler.artifact_sha256},
        "scope_sha256": equivalence.digest(scope),
        "projection_sha256": equivalence.digest(equivalence.EXCLUDED_LEAVES),
        "scope_transitions": transitions,
        "seeds": args.seeds,
        "hardware": hardware,
        "source_artifacts": source_artifacts,
        "rows": rows,
        "totals": {"episodes": len(rows), "changed_episodes": 0},
    }
    output = (repo_root / args.output_dir / f"{args.receipt_id}.json").resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(_json_bytes(receipt))
    header = {
        "schema_version": "default-path-equivalence-header.v1",
        "admission_context": "pull_request",
        "head_sha": args.head_sha,
        "refute_review": {
            "head_sha": args.head_sha,
            "verdict": "pending_independent_review",
            "uri": args.refute_review_uri,
            "comparison_sha256": equivalence.digest(rows),
        },
        "rows_artifact": {
            "path": output.relative_to(repo_root).as_posix(),
            "sha256": _sha256(output),
        },
    }
    print(f"wrote {output.relative_to(repo_root)} (pending independent review)", file=sys.stderr)
    print("<!-- default-path-equivalence:v1")
    print(json.dumps(header, indent=2, sort_keys=True))
    print("-->")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the behaviour receipt producer CLI."""
    special = sys.argv[1:] if argv is None else argv
    if len(special) == 2 and special[0] == "--record-scheduler-context":
        return _stamp_scheduler_rows(Path(special[1]))
    if len(special) == 2 and special[0] == "--record-job-request":
        directory = Path(special[1])
        request = json.loads((directory / "request_template.json").read_text())
        _record_request(
            directory,
            request["source_sha"],
            os.environ["SLURM_JOB_ID"],
            request["requested_node"],
            request["arm_role"],
        )
        return 0
    args = _parse_args(argv)
    if args.comparison_only and args.receipt_route != "default-path-equivalence":
        raise ValueError("--comparison-only requires the default-path equivalence route")
    repo_root = ROOT
    head_sha = _full_sha(repo_root, args.head_sha)
    args.head_sha = head_sha
    if args.head_source_sha:
        args.head_source_sha = _full_sha(repo_root, args.head_source_sha)
        behaviour_receipt.require_receipt_only_ancestor(args.head_source_sha, head_sha, "job")
    if args.receipt_route == "default-path-equivalence":
        return _make_equivalence_receipt(args, repo_root)
    baseline_release = (
        behaviour_receipt.latest_release(args.repo) if args.baseline == "latest" else args.baseline
    )
    baseline_source = _release_source(repo_root, baseline_release)
    scheduler, baseline = _prepare_sweeps(args, repo_root, baseline_source)
    if args.submit_only:
        print(json.dumps({"head_job_id": scheduler.job_id, "baseline_job_id": baseline.job_id}))
        return 0
    behaviour_receipt.validate_hardware_pair(
        {"head": scheduler.hardware, "baseline": baseline.hardware}
    )
    scope = _load_scope(args.scope_path)
    if scheduler.source_sha == baseline.source_sha:
        raise behaviour_receipt.HardwareIdentityError(
            "producer source invalid: head and baseline share a producer commit"
        )
    head_rows = _load_run_rows(scheduler, "head")
    baseline_rows = _load_run_rows(baseline, "baseline")
    rows, classifications, totals = derive_valid_comparison(
        head_rows,
        baseline_rows,
        scope,
        classification_class=args.classification_class,
        evidence_base_uri=args.classification_evidence_base_uri,
    )
    audit_path = (repo_root / args.work_dir / "real_row_audit.json").resolve()
    audit_source_sha = args.audit_source_sha or scheduler.source_sha
    audit_sha = write_real_row_audit(
        audit_path,
        source_sha=audit_source_sha,
        rows=[*rows, *[_execution_evidence(row) for row in baseline_rows.values()]],
        classifications=classifications,
    )
    if json.loads(audit_path.read_text())["status"] != "pass":
        raise ValueError(f"real-row audit did not pass; inspect {audit_path}")
    baseline_config_sha = args.baseline_config_sha256 or _canonical_digest(
        {
            "release": baseline_release,
            "source_sha": baseline_source,
            "scope_sha256": _canonical_digest(scope),
        }
    )
    differences = args.baseline_differences or [
        "development reconstruction against the published baseline; reviewer must confirm body/config/source provenance"
    ]
    receipt_path, header = write_receipt_and_header(
        repo_root=repo_root,
        receipt_id=args.receipt_id,
        head_sha=head_sha,
        scheduler=scheduler,
        baseline=baseline,
        baseline_release=baseline_release,
        baseline_body_id=args.baseline_body_id,
        baseline_config_sha256=baseline_config_sha,
        baseline_differences=differences,
        scope=scope,
        rows=rows,
        classifications=classifications,
        totals=totals,
        audit_uri=args.audit_uri,
        audit_sha256=audit_sha,
        audit_source_sha=audit_source_sha,
        refute_review_uri=args.refute_review_uri,
        output_dir=(repo_root / args.output_dir).resolve(),
    )
    print(f"wrote {receipt_path.relative_to(repo_root)}", file=sys.stderr)
    print("<!-- behaviour-change-receipt:v2")
    print(json.dumps(header, indent=2, sort_keys=True))
    print("-->")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except behaviour_receipt.HardwareIdentityError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
