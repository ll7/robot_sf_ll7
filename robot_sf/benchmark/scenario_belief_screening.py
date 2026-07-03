"""Screen ScenarioBelief drop-vs-retain benchmark evidence for issue #3556.

Plain-language summary: this module decides whether a small real-runner
ScenarioBelief contrast is interpretable before anyone treats it as safety
evidence. It is pure report logic: it inspects pinned scenarios, seeds, mode
rows, and oracle safety thresholds without launching benchmark episodes.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

REQUIRED_BELIEF_MODES = ("oracle", "uncertain_retained", "uncertain_dropped")
DECISION_LABELS = (
    "revise",
    "retention_dominates",
    "inconclusive",
    "inconclusive_oracle_unsafe",
    "blocked_no_near_safe_family",
)


def _scenario_id(scenario: Mapping[str, Any]) -> str | None:
    """Return the stable scenario identifier used in benchmark reports."""
    for key in ("name", "id", "scenario_id"):
        value = scenario.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _check(passed: bool, detail: str) -> dict[str, Any]:
    """Build a compact JSON-serializable screening check.

    Returns:
        Mapping with boolean pass status and a human-readable detail string.
    """
    return {"passed": bool(passed), "detail": detail}


def build_seed_sufficiency_handoff(
    *,
    campaign_output_root: Path | str = "output/issue_3556_belief_mode_campaign",
    output_dir: Path | str = "output/issue_3556_seed_sufficiency",
    campaign_id: str = "issue_3556",
    advisory_seed_threshold: int | None = None,
) -> dict[str, Any]:
    """Build the follow-up command for escalating smoke output to seed sufficiency.

    The command is provenance for the next empirical step only. The screening
    module remains pure and does not inspect campaign artifacts or promote the
    current smoke seed fixture to seed-sufficient evidence.

    Returns:
        JSON-serializable handoff status, command argv, and claim boundary.
    """
    command = [
        "uv",
        "run",
        "python",
        "scripts/tools/analyze_seed_sufficiency.py",
        "--campaign-output-root",
        str(campaign_output_root),
        "--campaign-id",
        campaign_id,
        "--output-dir",
        str(output_dir),
    ]
    if advisory_seed_threshold is not None:
        command.extend(["--advisory-seed-threshold", str(advisory_seed_threshold)])

    return {
        "status": "handoff_only",
        "command": command,
        "claim_boundary": (
            "Exporter only: run after durable campaign outputs exist; this field is not "
            "seed-sufficiency evidence."
        ),
    }


def build_input_screening_report(
    *,
    scenarios: Iterable[Mapping[str, Any]],
    seeds: list[int],
    fov_degrees: float,
    required_modes: tuple[str, ...] = REQUIRED_BELIEF_MODES,
    scenario_set: Path | str | None = None,
    launch_packet: Path | str | None = None,
) -> dict[str, Any]:
    """Screen static #3556 campaign inputs before benchmark episodes run.

    Returns:
        JSON-serializable report with ``ready`` and named checks. This report is
        intentionally limited to static inputs; it does not require episode rows.
    """
    scenario_list = [dict(scenario) for scenario in scenarios]
    scenario_ids = [_scenario_id(scenario) for scenario in scenario_list]
    unique_ids = sorted({sid for sid in scenario_ids if sid})
    seeds_pinned = (
        isinstance(seeds, list)
        and bool(seeds)
        and all(isinstance(seed, int) and not isinstance(seed, bool) for seed in seeds)
        and len(set(seeds)) == len(seeds)
    )
    modes_pinned = tuple(required_modes) == REQUIRED_BELIEF_MODES

    out_of_fov_candidates = []
    for scenario in scenario_list:
        visibility = scenario.get("observation_visibility")
        if not isinstance(visibility, Mapping):
            continue
        enabled = bool(visibility.get("enabled", False))
        raw_fov = visibility.get("fov_degrees")
        scenario_fov = float(raw_fov if raw_fov is not None else fov_degrees)
        has_limited_view = enabled and scenario_fov < 360.0
        has_static_occlusion = bool(visibility.get("static_occlusion", False))
        pedestrians = scenario.get("single_pedestrians")
        has_pedestrians = isinstance(pedestrians, list) and bool(pedestrians)
        if has_limited_view and has_static_occlusion and has_pedestrians:
            out_of_fov_candidates.append(_scenario_id(scenario) or "<unnamed>")

    checks = {
        "scenario_ids_pinned": _check(
            len(unique_ids) == len(scenario_list) and bool(unique_ids),
            f"{len(unique_ids)} unique scenario id(s) pinned"
            if len(unique_ids) == len(scenario_list) and unique_ids
            else "scenario set must resolve at least one uniquely named scenario",
        ),
        "seeds_pinned": _check(
            seeds_pinned,
            f"{len(seeds)} unique integer seed(s) pinned"
            if seeds_pinned
            else "seeds must be a non-empty unique integer list",
        ),
        "belief_modes_pinned": _check(
            modes_pinned,
            f"required modes pinned: {list(required_modes)}"
            if modes_pinned
            else f"required modes must be {list(REQUIRED_BELIEF_MODES)}",
        ),
        "out_of_fov_sidecar_contract": _check(
            bool(out_of_fov_candidates),
            "scenario exposes static occlusion, limited FOV, and explicit pedestrians"
            if out_of_fov_candidates
            else "scenario must expose limited-FOV occlusion-bearing pedestrians",
        ),
    }
    failed = [name for name, check in checks.items() if not check["passed"]]
    return {
        "schema_version": "scenario-belief-screening-inputs.v1",
        "issue": 3556,
        "ready": not failed,
        "scenario_set": str(scenario_set) if scenario_set is not None else None,
        "launch_packet": str(launch_packet) if launch_packet is not None else None,
        "scenario_ids": unique_ids,
        "seeds": list(seeds),
        "required_modes": list(required_modes),
        "fov_degrees": fov_degrees,
        "seed_sufficiency_handoff": build_seed_sufficiency_handoff(
            advisory_seed_threshold=len(seeds) if seeds_pinned else None
        ),
        "checks": checks,
        "failed_checks": failed,
        "claim_boundary": (
            "Static screening only: no benchmark episodes rolled and no safety claim promoted."
        ),
    }


def classify_screened_decision(
    by_mode: Mapping[str, Mapping[str, Any]],
    *,
    oracle_near_safe_threshold: float,
) -> dict[str, Any]:
    """Classify a three-mode result using #3556 allowed decision labels.

    Returns:
        Decision payload containing the label, reason, threshold, and deltas.
    """
    oracle = dict(by_mode.get("oracle", {}))
    retained = dict(by_mode.get("uncertain_retained", {}))
    dropped = dict(by_mode.get("uncertain_dropped", {}))
    if not (oracle.get("episodes") and retained.get("episodes") and dropped.get("episodes")):
        return {
            "decision": "blocked_no_near_safe_family",
            "reason": "one or more modes produced no episode rows",
            "screening_status": "missing_mode_rows",
            "oracle_near_safe": False,
            "oracle_near_safe_threshold": oracle_near_safe_threshold,
            "mode_is_discriminating": False,
        }

    oracle_collision_rate = float(oracle.get("collision_rate", 0.0))
    retained_collision_rate = float(retained.get("collision_rate", 0.0))
    dropped_collision_rate = float(dropped.get("collision_rate", 0.0))
    retained_near_misses = int(retained.get("total_near_misses", 0))
    dropped_near_misses = int(dropped.get("total_near_misses", 0))
    coll_delta = dropped_collision_rate - retained_collision_rate
    nm_delta = dropped_near_misses - retained_near_misses
    worse = coll_delta > 0 or nm_delta > 0
    oracle_unsafe = oracle_collision_rate > oracle_near_safe_threshold

    if oracle_unsafe:
        decision = "inconclusive_oracle_unsafe"
        reason = (
            f"oracle baseline itself unsafe (collision_rate {oracle_collision_rate}); "
            "effect not cleanly attributable"
        )
        screening_status = "oracle_unsafe"
    elif worse:
        decision = "revise"
        reason = (
            "dropping uncertain agents raised collisions "
            f"({retained_collision_rate}->{dropped_collision_rate}) and/or near-misses "
            f"(+{nm_delta}) vs retention, with near-safe oracle ({oracle_collision_rate}); "
            "revise/block dropping default"
        )
        screening_status = "near_safe_discriminating"
    elif coll_delta == 0 and nm_delta == 0:
        decision = "inconclusive"
        reason = "no measurable safety difference matrix"
        screening_status = "near_safe_nondiscriminating"
    else:
        decision = "retention_dominates"
        reason = "dropping did not increase unsafe outcomes here"
        screening_status = "near_safe_nonworsening"

    return {
        "decision": decision,
        "reason": reason,
        "screening_status": screening_status,
        "oracle_collision_rate": round(oracle_collision_rate, 4),
        "oracle_near_safe": not oracle_unsafe,
        "oracle_near_safe_threshold": oracle_near_safe_threshold,
        "mode_is_discriminating": worse,
        "collision_rate_delta_dropped_minus_retained": round(coll_delta, 4),
        "near_miss_delta_dropped_minus_retained": nm_delta,
    }


def build_screening_report(
    *,
    scenarios: Iterable[Mapping[str, Any]],
    seeds: list[int],
    by_mode: Mapping[str, Mapping[str, Any]],
    oracle_near_safe_threshold: float,
    fov_degrees: float,
    required_modes: tuple[str, ...] = REQUIRED_BELIEF_MODES,
    scenario_set: Path | str | None = None,
    launch_packet: Path | str | None = None,
) -> dict[str, Any]:
    """Build the final #3556 screening report after mode rows exist.

    Returns:
        JSON-serializable screening report for campaign output.
    """
    input_report = build_input_screening_report(
        scenarios=scenarios,
        seeds=seeds,
        fov_degrees=fov_degrees,
        required_modes=required_modes,
        scenario_set=scenario_set,
        launch_packet=launch_packet,
    )
    missing_modes = [
        mode for mode in required_modes if not dict(by_mode.get(mode, {})).get("episodes")
    ]
    mode_rows_check = _check(
        not missing_modes,
        "all three configured arms produced episode rows"
        if not missing_modes
        else f"missing episode rows for mode(s): {missing_modes}",
    )
    decision = classify_screened_decision(
        by_mode, oracle_near_safe_threshold=oracle_near_safe_threshold
    )
    checks = dict(input_report["checks"])
    checks["mode_rows_complete"] = mode_rows_check
    failed = [name for name, check in checks.items() if not check["passed"]]
    return {
        "schema_version": "scenario-belief-screening-report.v1",
        "issue": 3556,
        "ready": not failed and decision["decision"] != "blocked_no_near_safe_family",
        "scenario_set": input_report["scenario_set"],
        "launch_packet": input_report["launch_packet"],
        "scenario_ids": input_report["scenario_ids"],
        "seeds": input_report["seeds"],
        "required_modes": input_report["required_modes"],
        "fov_degrees": input_report["fov_degrees"],
        "seed_sufficiency_handoff": input_report["seed_sufficiency_handoff"],
        "checks": checks,
        "failed_checks": failed,
        "decision": decision,
        "allowed_decision_labels": list(DECISION_LABELS),
        "claim_boundary": (
            "Screening report for bounded real-runner smoke or campaign output. It does not "
            "promote a paper-grade safety claim."
        ),
    }
