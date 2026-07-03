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

# Report artifacts the seed-sufficiency analyzer (scripts/tools/analyze_seed_sufficiency.py)
# requires under a retained campaign root's ``reports/`` folder before it can run.
REQUIRED_SEED_SUFFICIENCY_REPORTS = (
    "seed_variability_by_scenario.json",
    "seed_episode_rows.csv",
)

# Bounded decision labels for the seed-sufficiency closure resolver. The resolver
# either points the analyzer at a usable retained campaign root or fails closed
# with an explicit missing-artifact blocker; it never fabricates evidence.
SEED_SUFFICIENCY_CLOSURE_LABELS = (
    "resolved_retained_campaign",
    "blocked_missing_retained_campaign_outputs",
)

# Bounded decision labels for the issue #4328 named-candidate closure evaluation.
# A candidate root is closure-usable only when it exists on the current host, holds
# every analyzer-required report, and carries a #3556 ScenarioBelief lineage;
# otherwise the packet fails closed with an explicit, per-candidate blocker.
SEED_SUFFICIENCY_CANDIDATE_LABELS = (
    "resolved_compatible_candidate",
    "blocked_no_compatible_candidate",
)

# Substrings that identify a #3556 ScenarioBelief campaign lineage. The #3556
# seed-sufficiency closure is specific to the ScenarioBelief drop-vs-retain
# contrast; a foreign campaign's seed-sufficiency (for example a long-horizon or
# extended-roster h600 run) would answer a different question, so promoting it as
# #3556 closure evidence would be a provenance overclaim.
SCENARIO_BELIEF_CAMPAIGN_MARKERS = ("3556", "scenario_belief", "belief_mode", "belief")


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


def build_seed_sufficiency_closure_packet(
    *,
    searched_roots: Iterable[Mapping[str, Any]],
    resolved_campaign_root: str | None,
    analyzer_command: list[str],
    analyzer_output_dir: str | None = None,
    analyzer_summary: Mapping[str, Any] | None = None,
    required_report_files: tuple[str, ...] = REQUIRED_SEED_SUFFICIENCY_REPORTS,
) -> dict[str, Any]:
    """Assemble the issue #3556 seed-sufficiency closure packet from probe results.

    This is pure decision logic: it does not touch the filesystem. The caller
    probes each durable search root, records which retained campaign roots were
    found and whether each holds every analyzer-required report file, then passes
    those probe results here. The packet either promotes a resolved retained
    campaign root or fails closed with an explicit missing-artifact blocker, so a
    partially populated or absent retained campaign never reads as evidence.

    Args:
        searched_roots: Per-root probe results. Each mapping should carry a
            ``search_root`` string plus discovery details (``exists``,
            ``campaign_roots_found``, ``usable_campaign_roots``,
            ``missing_report_files``) so the blocker is fully reproducible.
        resolved_campaign_root: The first retained campaign root holding every
            required report file, or ``None`` when none qualified.
        analyzer_command: The exact ``analyze_seed_sufficiency.py`` argv the
            resolver ran (resolved) or would run once inputs exist (blocked).
        analyzer_output_dir: Where analyzer artifacts were written, when resolved.
        analyzer_summary: Compact analyzer result summary (headline claim status,
            seed counts), when resolved.
        required_report_files: Report artifacts each campaign root must contain.

    Returns:
        JSON-serializable closure packet with an explicit evidence status and a
        bounded decision label from ``SEED_SUFFICIENCY_CLOSURE_LABELS``.
    """
    resolved = resolved_campaign_root is not None
    decision_label = (
        "resolved_retained_campaign" if resolved else "blocked_missing_retained_campaign_outputs"
    )
    searched = [dict(root) for root in searched_roots]
    if resolved:
        claim_boundary = (
            "Seed-sufficiency analyzer ran on a resolved retained campaign root. This packet "
            "reports interval-width / rank-stability status only; it does not itself promote a "
            "benchmark or paper-grade safety claim."
        )
        next_action = (
            "Review the analyzer headline rank-stability contract and record the seed-sufficiency "
            "decision on issue #3556."
        )
    else:
        claim_boundary = (
            "No retained issue #3556 campaign root exposing the analyzer-required report files was "
            "found under the searched durable roots, so no seed-sufficiency evidence is promoted."
        )
        next_action = (
            "Restore or point to a retained issue #3556 campaign root that contains "
            f"reports/{required_report_files[0]} and reports/{required_report_files[1]} under one "
            "of the searched roots, then rerun this resolver."
        )
    return {
        "schema_version": "issue_3556_seed_sufficiency_closure.v1",
        "issue": 3556,
        "evidence_status": "promoted" if resolved else "blocked",
        "decision_label": decision_label,
        "allowed_decision_labels": list(SEED_SUFFICIENCY_CLOSURE_LABELS),
        "required_report_files": list(required_report_files),
        "searched_roots": searched,
        "resolved_campaign_root": resolved_campaign_root,
        "analyzer_command": list(analyzer_command),
        "analyzer_output_dir": analyzer_output_dir,
        "analyzer_summary": dict(analyzer_summary) if analyzer_summary is not None else None,
        "claim_boundary": claim_boundary,
        "next_empirical_action": next_action,
        "forbidden_actions_confirmed": {
            "full_benchmark_campaign_run": False,
            "slurm_or_gpu_submission": False,
            "belief_mode_semantic_change": False,
            "paper_or_dissertation_claim_edit": False,
        },
    }


def evaluate_candidate_root_provenance(
    *,
    name: str,
    lineage: str = "",
    markers: tuple[str, ...] = SCENARIO_BELIEF_CAMPAIGN_MARKERS,
) -> dict[str, Any]:
    """Decide whether a named candidate root carries a #3556 ScenarioBelief lineage.

    Pure substring check over the candidate name and its declared lineage. The
    #3556 closure resolves seed-sufficiency for the ScenarioBelief drop-vs-retain
    contrast specifically; a foreign campaign (for example an h600 long-horizon or
    extended-roster run) would yield a seed-sufficiency verdict about a different
    planner/scenario family, so it must not be promoted as #3556 closure evidence.

    Returns:
        Mapping with ``provenance_compatible`` (bool), the ``matched_markers``, and
        a human-readable ``reason``.
    """
    haystack = f"{name} {lineage}".lower()
    matched = [marker for marker in markers if marker in haystack]
    compatible = bool(matched)
    reason = (
        f"name/lineage matches ScenarioBelief marker(s) {matched}"
        if compatible
        else (
            "no #3556 ScenarioBelief lineage marker in name/lineage; a foreign-campaign "
            "seed-sufficiency verdict would answer a different question than the #3556 contrast"
        )
    )
    return {
        "provenance_compatible": compatible,
        "matched_markers": matched,
        "reason": reason,
    }


def evaluate_seed_sufficiency_candidate(
    *,
    name: str,
    root: str,
    exists_on_host: bool,
    missing_report_files: Iterable[str],
    lineage: str = "",
    required_report_files: tuple[str, ...] = REQUIRED_SEED_SUFFICIENCY_REPORTS,
) -> dict[str, Any]:
    """Evaluate one named candidate root against the #3556 seed-sufficiency contract.

    A candidate is closure-usable only when it satisfies every gate: it exists on
    the current host, holds every analyzer-required report file, and carries a
    #3556 ScenarioBelief lineage. This function is pure decision logic; the caller
    is responsible for probing the filesystem and passing the observed facts in.

    Args:
        name: Short candidate identifier (used for the provenance lineage check).
        root: Repository-root-relative campaign root path.
        exists_on_host: Whether ``root`` was found on the current analysis host.
        missing_report_files: Required report files absent under ``root/reports``
            (empty when every required report is present; the full required set
            when the root itself is absent and cannot be probed).
        lineage: Free-text campaign lineage used for the provenance marker check.
        required_report_files: Report artifacts the analyzer needs.

    Returns:
        JSON-serializable per-candidate record with a derived ``compatible`` flag
        and the list of ``blockers`` that prevent closure use.
    """
    missing = list(missing_report_files)
    provenance = evaluate_candidate_root_provenance(name=name, lineage=lineage)
    reports_present = exists_on_host and not missing
    blockers: list[str] = []
    if not exists_on_host:
        blockers.append("root_absent_on_host")
    if missing:
        blockers.append("missing_required_reports")
    if not provenance["provenance_compatible"]:
        blockers.append("provenance_incompatible_with_3556")
    compatible = not blockers
    return {
        "name": name,
        "root": root,
        "lineage": lineage,
        "exists_on_host": bool(exists_on_host),
        "required_report_files": list(required_report_files),
        "missing_report_files": missing,
        "required_reports_present": reports_present,
        "provenance": provenance,
        "compatible": compatible,
        "blockers": blockers,
    }


def build_h600_candidate_closure_packet(
    *,
    candidates: Iterable[Mapping[str, Any]],
    analyzer_command: list[str],
    resolved_candidate: Mapping[str, Any] | None = None,
    analyzer_output_dir: str | None = None,
    analyzer_summary: Mapping[str, Any] | None = None,
    queue_row_request: Mapping[str, Any] | None = None,
    required_report_files: tuple[str, ...] = REQUIRED_SEED_SUFFICIENCY_REPORTS,
) -> dict[str, Any]:
    """Assemble the issue #4328 named-candidate seed-sufficiency closure packet.

    Composes the canonical closure packet (reusing
    :func:`build_seed_sufficiency_closure_packet` for the shared fields and
    forbidden-action attestations) with a per-candidate compatibility manifest.
    The packet promotes the analyzer result only when a fully compatible candidate
    resolved; otherwise it fails closed with a ``blocked_no_compatible_candidate``
    decision and records exactly why each candidate was rejected.

    Args:
        candidates: Per-candidate records from
            :func:`evaluate_seed_sufficiency_candidate`.
        analyzer_command: The exact analyzer argv that ran (resolved) or would run
            once a compatible candidate exists on host (blocked).
        resolved_candidate: The first fully compatible candidate, or ``None``.
        analyzer_output_dir: Where analyzer artifacts were written, when resolved.
        analyzer_summary: Compact analyzer result summary, when resolved.
        queue_row_request: Description of the #3556-specific ScenarioBelief campaign
            that would satisfy the contract, recorded when no candidate qualifies.
        required_report_files: Report artifacts each candidate root must contain.

    Returns:
        JSON-serializable closure packet with a bounded decision label from
        ``SEED_SUFFICIENCY_CANDIDATE_LABELS``.
    """
    candidate_list = [dict(candidate) for candidate in candidates]
    resolved = resolved_candidate is not None
    resolved_root = dict(resolved_candidate)["root"] if resolved else None
    base = build_seed_sufficiency_closure_packet(
        searched_roots=candidate_list,
        resolved_campaign_root=resolved_root,
        analyzer_command=analyzer_command,
        analyzer_output_dir=analyzer_output_dir,
        analyzer_summary=analyzer_summary,
        required_report_files=required_report_files,
    )
    decision_label = (
        "resolved_compatible_candidate" if resolved else "blocked_no_compatible_candidate"
    )
    compatible_candidates = [c["name"] for c in candidate_list if c.get("compatible")]
    # ``candidates`` is the canonical per-root field for this schema; drop the base
    # builder's ``searched_roots`` alias so the packet does not duplicate every
    # candidate record verbatim.
    base.pop("searched_roots", None)
    base.update(
        {
            "schema_version": "issue_4328_h600_candidate_closure.v1",
            "closure_target_issue": 3556,
            "closure_attempt_issue": 4328,
            "decision_label": decision_label,
            "allowed_decision_labels": list(SEED_SUFFICIENCY_CANDIDATE_LABELS),
            "candidates": candidate_list,
            "compatible_candidates": compatible_candidates,
            "resolved_candidate": dict(resolved_candidate) if resolved else None,
            "queue_row_request": dict(queue_row_request) if queue_row_request else None,
        }
    )
    if not resolved:
        base["claim_boundary"] = (
            "No named h600 candidate root satisfied the #3556 seed-sufficiency contract "
            "(existence on host + required reports + ScenarioBelief provenance), so no "
            "seed-sufficiency evidence is promoted."
        )
        base["next_empirical_action"] = (
            "Run a #3556-specific ScenarioBelief drop-vs-retain campaign that emits "
            f"reports/{required_report_files[0]} and reports/{required_report_files[1]}, or, if a "
            "maintainer accepts a foreign h600 root as a proxy, restore that root on the analysis "
            "host and rerun the analyzer command below."
        )
    return base


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
    explicit_null_fov = []
    for scenario in scenario_list:
        visibility = scenario.get("observation_visibility")
        if not isinstance(visibility, Mapping):
            continue
        enabled = bool(visibility.get("enabled", False))
        if "fov_degrees" in visibility and visibility.get("fov_degrees") is None:
            explicit_null_fov.append(_scenario_id(scenario) or "<unnamed>")
            continue
        raw_fov = visibility.get("fov_degrees", fov_degrees)
        scenario_fov = float(raw_fov)
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
        "explicit_null_fov_rejected": _check(
            not explicit_null_fov,
            "no scenario-level null fov_degrees values"
            if not explicit_null_fov
            else f"explicit null fov_degrees in scenario(s): {explicit_null_fov}",
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
