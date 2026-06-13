"""Mechanism-layer classifier for observation-noise envelope evidence.

Turns per-condition evaluation results from the observation-noise envelope
into mechanism-layer labels that describe *where* in the observation ->
perception -> command -> trajectory pipeline the perturbation had (or did
not have) an effect.

This is diagnostic, rule-based evidence only. It does not infer causality
from a single fixture and does not promote benchmark or paper-facing claims.
"""

from __future__ import annotations

from typing import Any

MECHANISM_LABELS: set[str] = {
    "observation_did_not_affect_selected_source",
    "observation_affected_source_but_not_command",
    "command_changed_but_trajectory_did_not",
    "delay_shifted_stop_timing",
    "occlusion_changed_first_actionable_frame",
    "noise_stayed_below_decision_threshold",
    "scenario_had_no_actionable_conflict",
    "stored_action_proxy_prevents_live_conclusion",
    "diagnostic_only",
    "inconclusive",
}

# Near-field threshold in metres; scenarios farther away lack actionable
# conflict for observation-noise testing.
_NEAR_FIELD_THRESHOLD_M: float = 2.0

# Heuristic boundary for "small" Gaussian noise that is unlikely to cross
# a decision boundary in the default hybrid_rule_v0 planner.
_SMALL_NOISE_STD_M: float = 0.15


def classify_mechanism(
    condition_result: dict[str, Any],
    fixture_meta: dict[str, Any],
    noop_result: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Classify one perturbation condition by mechanism layer.

    Args:
        condition_result: Single condition dict from ``evaluate_condition``.
        fixture_meta: Fixture metadata with at least ``first_visible_step``.
        noop_result: Optional baseline (noop) condition result for command
            comparison.  When *None*, command-comparison labels fall back to
            ``stored_action_proxy_prevents_live_conclusion``.

    Returns:
        ``{"label": <mechanism_label>, "rationale": <str>}``
    """
    spec: dict[str, Any] = condition_result.get("spec") or {}
    first_observed: int | None = condition_result.get("first_observed_step")
    response_delay: int | None = condition_result.get("response_delay_steps")
    closest_dist: float | None = condition_result.get("closest_distance_m")
    action_proxy: dict[str, Any] = condition_result.get("action_proxy_changes") or {}
    missed_total: int = condition_result.get("missed_actor_observations_total", 0)
    occluded_total: int = condition_result.get("occluded_actor_observations_total", 0)
    stop_yield: dict[str, Any] = condition_result.get("stop_yield_feasibility") or {}

    if spec.get("is_noop", False):
        return {
            "label": "diagnostic_only",
            "rationale": (
                "No-perturbation baseline; mechanism classification is not "
                "applicable to the reference trajectory."
            ),
        }

    if first_observed is None:
        if missed_total > 0 or occluded_total > 0:
            return {
                "label": "occlusion_changed_first_actionable_frame",
                "rationale": (
                    "Perturbation completely suppressed the pedestrian "
                    "observation (missed detections or occlusion mask). "
                    "The first actionable frame was changed or never reached."
                ),
            }
        return {
            "label": "inconclusive",
            "rationale": (
                "Pedestrian never observed but no missed-detection or "
                "occlusion signal recorded. Insufficient data for mechanism "
                "classification."
            ),
        }

    # Evaluate "no actionable conflict" after complete missed/occluded
    # observations so the scenario fallback does not mask an observation
    # mechanism when both signals are present.
    if closest_dist is not None and closest_dist > _NEAR_FIELD_THRESHOLD_M:
        return {
            "label": "scenario_had_no_actionable_conflict",
            "rationale": (
                f"Closest robot-pedestrian distance ({closest_dist:.2f} m) "
                f"exceeds near-field threshold ({_NEAR_FIELD_THRESHOLD_M} m). "
                "No actionable conflict for observation-noise testing."
            ),
        }

    if response_delay is not None and response_delay > 0:
        stop_changed = _stop_yield_changed(stop_yield, noop_result)
        rationale_parts = [
            (
                f"Observation arrived {response_delay} step(s) after "
                f"first-visible (step {first_observed})."
            ),
        ]
        if stop_changed:
            rationale_parts.append(
                "Stop/yield feasibility at first observation differs from "
                "baseline, indicating the delay shifted decision timing."
            )
        return {
            "label": "delay_shifted_stop_timing",
            "rationale": " ".join(rationale_parts),
        }

    noise_std = spec.get("position_noise_std_m") or 0.0
    if noise_std > 0.0:
        command_same = _command_unchanged(action_proxy, noop_result)
        if command_same:
            if noise_std <= _SMALL_NOISE_STD_M:
                return {
                    "label": "noise_stayed_below_decision_threshold",
                    "rationale": (
                        f"Gaussian noise (std={noise_std:.2f} m) applied but "
                        "policy command sequence matches baseline. Noise "
                        "stayed below the decision threshold for this "
                        "planner/scenario combination."
                    ),
                }
            return {
                "label": "observation_affected_source_but_not_command",
                "rationale": (
                    f"Gaussian noise (std={noise_std:.2f} m) perturbed the "
                    "observation source but the policy command sequence "
                    "matches baseline. The perturbation did not propagate "
                    "to the command layer."
                ),
            }
        # Command changed, but trajectory comparison requires live replay.
        return {
            "label": "stored_action_proxy_prevents_live_conclusion",
            "rationale": (
                "Noise perturbed the observation and the stored-trace action "
                "proxy differs from baseline, but live replay is required to "
                "determine whether the trajectory actually changed."
            ),
        }

    return {
        "label": "inconclusive",
        "rationale": (
            "Condition has no active perturbation mechanism and is not the "
            "baseline. Insufficient data for mechanism classification."
        ),
    }


def _command_unchanged(
    action_proxy: dict[str, Any],
    noop_result: dict[str, Any] | None,
) -> bool:
    """Return True when the action proxy matches the noop baseline."""
    if noop_result is None:
        return False
    noop_proxy = noop_result.get("action_proxy_changes") or {}
    return (
        action_proxy.get("linear_velocity_changed") == noop_proxy.get("linear_velocity_changed")
        and action_proxy.get("events") == noop_proxy.get("events")
        and action_proxy.get("velocity_range") == noop_proxy.get("velocity_range")
    )


def _stop_yield_changed(
    stop_yield: dict[str, Any],
    noop_result: dict[str, Any] | None,
) -> bool:
    """Return True when stop/yield feasibility differs from noop."""
    if noop_result is None:
        return False
    noop_sy = noop_result.get("stop_yield_feasibility") or {}
    return stop_yield.get("stop_feasible_first_observed") != noop_sy.get(
        "stop_feasible_first_observed"
    ) or stop_yield.get("yield_feasible_first_observed") != noop_sy.get(
        "yield_feasible_first_observed"
    )


def classify_all_conditions(
    condition_results: list[dict[str, Any]],
    fixture_meta: dict[str, Any],
) -> list[dict[str, Any]]:
    """Classify every condition and attach mechanism labels in-place.

    Args:
        condition_results: List of dicts from ``evaluate_condition``.
        fixture_meta: Fixture metadata.

    Returns:
        The same list with an added ``"mechanism"`` key on each entry.
    """
    noop_result: dict[str, Any] | None = None
    for r in condition_results:
        if (r.get("spec") or {}).get("is_noop", False):
            noop_result = r
            break

    for r in condition_results:
        mechanism = classify_mechanism(r, fixture_meta, noop_result)
        r["mechanism"] = mechanism

    return condition_results
