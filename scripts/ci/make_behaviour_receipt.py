#!/usr/bin/env python3
"""Produce behaviour-receipt payloads and compact PR-body headers."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ci import behaviour_receipt  # noqa: E402

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


def _sweep_command(  # noqa: PLR0913
    source_sha: str,
    output_dir: Path,
    seeds: list[int],
    *,
    arms: list[str] | None,
    scenarios: list[str] | None,
    workers: int,
    check_only: bool,
    harness_sha256: str,
    canary: bool = False,
    first_seed_only: bool = False,
) -> list[str]:
    cmd = [
        "uv",
        "run",
        "python",
        "-c",
        "import hashlib, sys; from pathlib import Path; "
        "path = Path(sys.argv.pop(1)).resolve(); expected = sys.argv.pop(1); "
        "source = path.read_bytes(); digest = hashlib.sha256(source).hexdigest(); "
        "digest == expected or sys.exit('sweep harness differs from selected committed source'); "
        "sys.argv[0] = str(path); sys.path.insert(0, str(path.parents[2])); "
        "exec(compile(source, str(path), 'exec'), "
        "{'__name__': '__main__', '__file__': str(path), '_SWEEP_SOURCE_BYTES': source})",
        "scripts/validation/run_empty_world_sweep.py",
        harness_sha256,
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
    if canary:
        cmd.append("--canary")
    if first_seed_only:
        cmd.append("--first-seed-only")
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


def _install_sweep_harness(
    repo_root: Path,
    checkout: Path,
    output_dir: Path,
    runtime_source_sha: str,
    harness_source_sha: str,
) -> None:
    """Pin acquisition tooling without substituting the tested runtime tree.

    The sweep permits this script to differ from its runtime source. Preserve
    both script digests in the artifact: this is a development reconstruction,
    never archived release data.
    """
    relative = "scripts/validation/run_empty_world_sweep.py"
    source = _full_sha(repo_root, harness_source_sha)
    if _full_sha(checkout, "HEAD") != runtime_source_sha:
        raise ValueError("sweep checkout differs from requested runtime source")
    harness = subprocess.check_output(["git", "show", f"{source}:{relative}"], cwd=repo_root)
    original = subprocess.check_output(
        ["git", "show", f"{runtime_source_sha}:{relative}"], cwd=repo_root
    )
    target = checkout / relative
    if target.is_symlink() or not target.is_file():
        raise ValueError("sweep harness must be a regular source file")
    if target.read_bytes() not in (original, harness):
        raise ValueError("refusing to overwrite uncommitted sweep harness")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / "harness.json"
    from scripts.validation.receipt_harness import bundle_contents, install_bundle

    bundle = bundle_contents(repo_root, runtime_source_sha, source)
    record = _json_bytes(
        {
            "schema_version": "behaviour-sweep-harness.v1",
            "runtime_source_sha": runtime_source_sha,
            "harness_source_sha": source,
            "harness_path": relative,
            "harness_sha256": hashlib.sha256(harness).hexdigest(),
            "original_harness_sha256": hashlib.sha256(original).hexdigest(),
            "comparison": "development_reconstruction",
            **(bundle[1] if bundle is not None else {}),
        }
    )
    if manifest.exists() and manifest.read_bytes() != record:
        raise ValueError("existing sweep harness provenance differs; use a fresh work directory")
    if bundle is not None:
        install_bundle(checkout, bundle[0], bundle[1])
    else:
        target.write_bytes(harness)
    manifest.write_bytes(record)


def _run_local_sweep(  # noqa: PLR0913
    repo_root: Path,
    work_dir: Path,
    label: str,
    source_sha: str,
    seeds: list[int],
    *,
    arms: list[str] | None,
    scenarios: list[str] | None,
    workers: int,
    check_only: bool,
    sweep_harness_source_sha: str | None = None,
) -> SweepRun:
    checkout = _ensure_worktree(repo_root, work_dir, label, source_sha)
    output_dir = work_dir / "sweeps" / label
    if sweep_harness_source_sha:
        _install_sweep_harness(
            repo_root, checkout, output_dir, source_sha, sweep_harness_source_sha
        )
    cmd = _sweep_command(
        source_sha,
        output_dir,
        seeds,
        arms=arms,
        scenarios=scenarios,
        workers=workers,
        check_only=check_only,
        harness_sha256=hashlib.sha256(
            subprocess.check_output(
                [
                    "git",
                    "show",
                    f"{sweep_harness_source_sha or source_sha}:scripts/validation/run_empty_world_sweep.py",
                ],
                cwd=repo_root,
            )
        ).hexdigest(),
    )
    subprocess.run(cmd, cwd=checkout, check=True)
    manifest = output_dir / "README.md"
    artifact_sha = _sha256(manifest) if manifest.is_file() else _canonical_digest(cmd)
    return SweepRun(
        source_sha=source_sha,
        output_dir=output_dir,
        artifact_uri=f"{PLACEHOLDER_EVIDENCE_URI}/{label}",
        artifact_sha256=artifact_sha,
        job_id="1",
    )


def _submit_sbatch_sweep(  # noqa: PLR0913
    repo_root: Path,
    work_dir: Path,
    label: str,
    source_sha: str,
    seeds: list[int],
    *,
    arms: list[str] | None,
    scenarios: list[str] | None,
    workers: int,
    sweep_harness_source_sha: str | None = None,
    check_only: bool = False,
    canary: bool = False,
    first_seed_only: bool = False,
) -> SweepRun:
    checkout = _ensure_worktree(repo_root, work_dir, label, source_sha).resolve()
    if canary and workers > 2:
        raise ValueError("receipt canary is limited to 1-2 CPUs")
    output_dir = work_dir / "sweeps" / label
    output_dir.mkdir(parents=True, exist_ok=True)
    if sweep_harness_source_sha:
        _install_sweep_harness(
            repo_root, checkout, output_dir, source_sha, sweep_harness_source_sha
        )
    cmd = _sweep_command(
        source_sha,
        output_dir,
        seeds,
        arms=arms,
        scenarios=scenarios,
        workers=workers,
        check_only=check_only,
        canary=canary,
        first_seed_only=first_seed_only,
        harness_sha256=hashlib.sha256(
            subprocess.check_output(
                [
                    "git",
                    "show",
                    f"{sweep_harness_source_sha or source_sha}:scripts/validation/run_empty_world_sweep.py",
                ],
                cwd=repo_root,
            )
        ).hexdigest(),
    )
    script = work_dir / f"behaviour_receipt_{label}.sbatch"
    script.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                f"#SBATCH --job-name=behaviour-{label}",
                f"#SBATCH --output={output_dir / 'slurm-%j.out'}",
                "#SBATCH --time=" + ("00:14:00" if canary or check_only else "24:00:00"),
                f"#SBATCH --cpus-per-task={workers}",
                "set -euo pipefail",
                f"cd -- {shlex.quote(str(checkout))}",
                "test -f scripts/validation/run_empty_world_sweep.py",
                'export PYTHONPATH="$PWD:$PWD/fast-pysf${PYTHONPATH:+:$PYTHONPATH}"',
                "export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1",
                shlex.join(cmd),
                "",
            ]
        ),
        encoding="utf-8",
    )
    job_id = subprocess.check_output(
        ["sbatch", "--parsable", f"--chdir={checkout}", str(script)], text=True
    ).strip()
    submission = {
        "schema_version": "behaviour-sweep-submission.v1",
        "label": label,
        "source_sha": source_sha,
        "job_id": job_id,
        "command": cmd,
        "script": str(script),
    }
    manifest = output_dir / "submission.json"
    manifest.write_bytes(_json_bytes(submission))
    return SweepRun(
        source_sha=source_sha,
        output_dir=output_dir,
        artifact_uri=f"{PLACEHOLDER_EVIDENCE_URI}/{label}",
        artifact_sha256=_sha256(manifest),
        job_id=job_id,
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
    _verify_sweep_metadata(output_dir, source_sha)
    digest = _canonical_digest(
        {
            path.relative_to(output_dir).as_posix(): _sha256(path)
            for path in sorted(output_dir.rglob("*"))
            if path.is_file()
        }
    )
    if artifact_sha256 is not None and artifact_sha256 != digest:
        raise ValueError("supplied sweep digest differs from actual artifact bytes")
    return SweepRun(source_sha, output_dir, artifact_uri, digest, job_id)


def _sweep_harness_manifest(output_dir: Path, source_sha: str) -> dict[str, Any] | None:
    """Read optional acquisition provenance and bind it to the runtime source."""
    harness_path = output_dir / "harness.json"
    harness = json.loads(harness_path.read_text()) if harness_path.is_file() else None
    if harness is not None and harness.get("runtime_source_sha") != source_sha:
        raise ValueError("sweep harness runtime source differs")
    return harness


def _verify_sweep_metadata(output_dir: Path, source_sha: str) -> None:
    harness = _sweep_harness_manifest(output_dir, source_sha)
    summaries = sorted(output_dir.glob("episodes_*.jsonl"))
    if not summaries:
        raise ValueError("sweep has no episode suites")
    for summary in summaries:
        suite = summary.stem.removeprefix("episodes_")
        metadata_path = output_dir / f"execution_{suite}.json"
        if not metadata_path.is_file():
            raise ValueError(f"missing authoritative execution metadata for {suite}")
        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        _verify_executed_harness(harness, meta, suite)
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


def _verify_executed_harness(harness: dict | None, metadata: dict, suite: str) -> None:
    """Require an execution digest when an explicit acquisition harness was pinned."""
    if harness is not None and metadata.get("sweep_harness_sha256") != harness.get(
        "harness_sha256"
    ):
        raise ValueError(f"{suite} executed sweep harness differs from pinned provenance")


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


def _load_sweep_rows(sweep_dir: Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    """Join summaries to raw traces with an invocation-local index of each file.

    There is no persistent cache: subsequent calls reread the current bytes.
    """
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
            if _row_success(raw) != _row_success(row) or _row_collisions(raw) != _row_collisions(
                row
            ):
                raise ValueError(f"raw episode outcome differs from summary for {key}")
            if key in rows:
                raise ValueError(f"duplicate sweep row {key}")
            rows[key] = {**raw, "execution_status": row["execution_status"]}
    if not rows:
        raise ValueError(f"no sweep rows found in {sweep_dir}")
    return rows


def _row_success(row: dict[str, Any]) -> bool:
    from robot_sf.benchmark.receipt_stages import observation_success

    return observation_success(row)


def _row_collisions(row: dict[str, Any]) -> int:
    from robot_sf.benchmark.receipt_stages import observation_collisions

    return observation_collisions(row)


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
    return full in {
        tuple(location.split("."))
        for location in EXECUTION_EVIDENCE_LOCATIONS["non_runtime_locations"]
    }


def _transport_path(path: tuple[str, ...]) -> tuple[str | None, ...]:
    """Normalize declared array indices without conflating literal dictionary keys."""
    return tuple(None if part == "*" else part for part in path)


def _unclassified_runtime_marker(row: dict[str, Any]) -> tuple[str, str] | None:
    """Unknown fault vocabulary is evidence of an unaudited transport, even if false/zero.

    Source-key catalog membership is deliberately insufficient. Dict keys stay
    exact; only array indices normalize to '*'. No subtree is exempted.
    """
    vocabulary = re.compile(EXECUTION_EVIDENCE_LOCATIONS["fault_key_pattern"], re.IGNORECASE)
    runtime_fields = set(EXECUTION_EVIDENCE_LOCATIONS["runtime_fields"]) | set(
        EXECUTION_EVIDENCE_LOCATIONS["runtime_aliases"]
    )
    counter_fields = set(EXECUTION_EVIDENCE_LOCATIONS["counter_fields"])
    required = set(EXECUTION_EVIDENCE_LOCATIONS["mode_locations"]) | set(
        EXECUTION_EVIDENCE_LOCATIONS["required_values"]
    )

    # None represents an actual array index; a literal '*' dictionary key must
    # not acquire the array-only exemption. Keep dotted dictionary keys intact.
    optional = {
        _transport_path(tuple(location.split(".")))
        for location in EXECUTION_EVIDENCE_LOCATIONS["non_runtime_locations"]
    }
    runtime_objects = {
        _transport_path(path) for path in EXECUTION_EVIDENCE_LOCATIONS["runtime_objects"]
    }
    counter_objects = {
        _transport_path(path) for path in EXECUTION_EVIDENCE_LOCATIONS["counter_objects"]
    }

    def classified(path: tuple[str | None, ...], key: str) -> bool:
        return (
            (path in runtime_objects and key in runtime_fields)
            or (path in counter_objects and key in counter_fields)
            or (*path, key) in required
            or (*path, key) in optional
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
    if mode not in {"native", "adapter"} or any(value != mode for value in modes):
        mode = "unknown"
    profile = metadata.get("planner_kinematics") or {}
    if isinstance(profile, dict) and any(
        path[-1] in profile for path in EXECUTION_EVIDENCE_LOCATIONS["mode_capabilities"]
    ):
        if profile.get(f"supports_{mode}_commands") is not True:
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
    return {
        "algorithm": algorithm or "unknown",
        "execution_mode": mode,
        "controller_executed": bool(executed),
        "fallback": row.get("fallback") is True
        or (marker is not None and "fallback" in str(marker)),
        "degraded": row.get("degraded") is True or marker is not None,
    }


def compare_episode_observation(head: dict, baseline: dict) -> tuple[bool, int, bool, int]:
    """Compare scored observations using the pinned acquisition outcome contract.

    Returns:
        Head/baseline success and collision observations.
    """
    from robot_sf.benchmark.receipt_stages import compare_episode_observation as compare

    return compare(head, baseline)


def build_rows_and_classifications(
    head_rows: dict[tuple[str, str, int], dict[str, Any]],
    baseline_rows: dict[tuple[str, str, int], dict[str, Any]],
    scope: dict[str, Any],
    *,
    classification_class: str,
    evidence_base_uri: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Join head/baseline sweep rows into gate payload rows and classifications."""
    from robot_sf.benchmark.metric_definitions import require_uniform_metric_schema

    require_uniform_metric_schema([*head_rows.values(), *baseline_rows.values()])
    expected = {
        (arm, slot, seed)
        for arm in scope["arms"]
        for slot in scope["maps"]
        for seed in DEFAULT_SEEDS
    }
    if set(head_rows) - expected or set(baseline_rows) - expected:
        raise ValueError("unexpected episode identities in comparison inputs")
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
                success, collisions, baseline_success, baseline_collisions = (
                    compare_episode_observation(head, baseline)
                )
                rows.append(
                    {
                        "arm": arm,
                        "map": map_id,
                        "seed": seed,
                        "success": success,
                        "collisions": collisions,
                        "baseline_success": baseline_success,
                        "baseline_collisions": baseline_collisions,
                        **_execution_evidence(head),
                    }
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


def _audit_status(rows: list[dict[str, Any]]) -> str:
    if not rows or any(
        row.get("controller_executed") is not True
        or row.get("execution_mode") not in {"native", "adapter"}
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
    payload = {
        "schema_version": "behaviour-change-rows.v1",
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
        "--head-source-sha", help="Recorded execution source before receipt-only commits"
    )
    parser.add_argument("--baseline", default="latest")
    parser.add_argument(
        "--sweep-harness-source-sha",
        help="Explicit committed acquisition script for both sources; records reconstruction provenance",
    )
    parser.add_argument("--repo", default="ll7/robot_sf_ll7")
    parser.add_argument("--receipt-id", required=True)
    parser.add_argument("--mode", choices=("existing", "local", "sbatch"), default="existing")
    parser.add_argument("--submit-only", action="store_true")
    parser.add_argument("--first-seed-only", action="store_true")
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
    parser.add_argument(
        "--canary",
        action="store_true",
        help="Tiny scheduler integration probe outside receipt namespace",
    )
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


def _prepare_sweeps(  # noqa: C901 - paired acquisition modes and preflight admission.
    args: argparse.Namespace, repo_root: Path, baseline_source: str
) -> tuple[SweepRun, SweepRun]:
    if args.mode == "local" and (
        getattr(args, "canary", False) or getattr(args, "first_seed_only", False)
    ):
        raise ValueError("bounded canary and first-seed modes require --mode sbatch")
    seeds = _check_seed_range(list(args.seeds))
    work_dir = (repo_root / args.work_dir).resolve()
    if args.mode == "existing":
        if args.head_sweep_dir is None or args.baseline_sweep_dir is None:
            raise ValueError("--mode existing requires --head-sweep-dir and --baseline-sweep-dir")
        if args.sweep_harness_source_sha:
            _verify_requested_harness(args, repo_root, baseline_source)
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
    if not args.check_only and not getattr(args, "canary", False):
        contracts = []
        for label, source in (("head", args.head_sha), ("baseline", baseline_source)):
            path = work_dir / "sweeps" / label / "contract.json"
            if not path.is_file():
                raise ValueError(
                    "paired execution requires both preflight contracts; submit --check-only first"
                )
            contract = json.loads(path.read_bytes())
            if contract.get("source_sha") != source or not contract.get("full_roster"):
                raise ValueError("preflight contract source/full-roster differs")
            contracts.append(contract)
        for field in ("row_schema_sha256", "metric_schema_version"):
            if contracts[0][field] != contracts[1][field]:
                raise ValueError("paired preflight contract row/metric schema mismatch")
        if contracts[0]["execution"].get(
            "metric_semantics_sha256", contracts[0]["execution"].get("metric_definitions_sha256")
        ) != contracts[1]["execution"].get(
            "metric_semantics_sha256", contracts[1]["execution"].get("metric_definitions_sha256")
        ):
            raise ValueError("paired preflight contract metric semantics mismatch")
    if args.mode == "local":
        return (
            _run_local_sweep(
                repo_root,
                work_dir,
                "head",
                args.head_sha,
                seeds,
                arms=args.arms,
                scenarios=args.scenarios,
                workers=args.workers,
                check_only=args.check_only,
                sweep_harness_source_sha=args.sweep_harness_source_sha,
            ),
            _run_local_sweep(
                repo_root,
                work_dir,
                "baseline",
                baseline_source,
                seeds,
                arms=args.arms,
                scenarios=args.scenarios,
                workers=args.workers,
                check_only=args.check_only,
                sweep_harness_source_sha=args.sweep_harness_source_sha,
            ),
        )
    return (
        _submit_sbatch_sweep(
            repo_root,
            work_dir,
            "head",
            args.head_sha,
            seeds,
            arms=args.arms,
            scenarios=args.scenarios,
            workers=args.workers,
            sweep_harness_source_sha=args.sweep_harness_source_sha,
            check_only=args.check_only,
            canary=getattr(args, "canary", False),
            first_seed_only=getattr(args, "first_seed_only", False),
        ),
        _submit_sbatch_sweep(
            repo_root,
            work_dir,
            "baseline",
            baseline_source,
            seeds,
            arms=args.arms,
            scenarios=args.scenarios,
            workers=args.workers,
            sweep_harness_source_sha=args.sweep_harness_source_sha,
            check_only=args.check_only,
            canary=getattr(args, "canary", False),
            first_seed_only=getattr(args, "first_seed_only", False),
        ),
    )


def _verify_requested_harness(
    args: argparse.Namespace, repo_root: Path, baseline_source: str
) -> None:
    """Collection must honor an explicitly requested paired acquisition revision."""
    source = _full_sha(repo_root, args.sweep_harness_source_sha)
    script = subprocess.check_output(
        ["git", "show", f"{source}:scripts/validation/run_empty_world_sweep.py"], cwd=repo_root
    )
    expected_digest = hashlib.sha256(script).hexdigest()
    for output_dir, runtime in (
        (args.head_sweep_dir, args.head_source_sha or args.head_sha),
        (args.baseline_sweep_dir, baseline_source),
    ):
        manifest = _sweep_harness_manifest(output_dir.resolve(), runtime)
        if (
            manifest is None
            or manifest.get("harness_source_sha") != source
            or manifest.get("harness_sha256") != expected_digest
        ):
            raise ValueError("existing sweep does not use the requested committed harness")


def main(argv: list[str] | None = None) -> int:
    """Run the behaviour receipt producer CLI."""
    args = _parse_args(argv)
    repo_root = ROOT
    head_sha = _full_sha(repo_root, args.head_sha)
    args.head_sha = head_sha
    if args.head_source_sha:
        args.head_source_sha = _full_sha(repo_root, args.head_source_sha)
        behaviour_receipt.require_receipt_only_ancestor(args.head_source_sha, head_sha, "job")
    baseline_release = (
        behaviour_receipt.latest_release(args.repo) if args.baseline == "latest" else args.baseline
    )
    baseline_source = _release_source(repo_root, baseline_release)
    scheduler, baseline = _prepare_sweeps(args, repo_root, baseline_source)
    if args.submit_only or args.check_only or args.canary:
        print(json.dumps({"head_job_id": scheduler.job_id, "baseline_job_id": baseline.job_id}))
        return 0
    scope = _load_scope(args.scope_path)
    head_rows = _load_sweep_rows(scheduler.output_dir)
    baseline_rows = _load_sweep_rows(baseline.output_dir)
    rows, classifications, totals = build_rows_and_classifications(
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
    for label, run in (("head", scheduler), ("baseline", baseline)):
        harness_path = run.output_dir / "harness.json"
        if harness_path.is_file():
            harness = json.loads(harness_path.read_text())
            differences.append(
                f"{label} acquisition harness {harness['harness_source_sha']} "
                f"sha256={harness['harness_sha256']}; runtime source {run.source_sha}; "
                "generated width predictive bindings use this runtime source's published main profiles; "
                "development reconstruction, not archived release outcomes"
            )
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
    raise SystemExit(main())
