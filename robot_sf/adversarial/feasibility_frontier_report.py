"""Build a provenance-bound, fixture-first report of planner/search rounds.

The input is a persisted ``adversarial-coevolution-evidence.v3`` bundle. This
module summarizes the supplied records; it does not run planners, infer dynamic
feasibility, or turn a finite no-discovery result into a claim of absence.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote

from robot_sf.common.optional_import import try_import

INPUT_SCHEMA_VERSION = "adversarial-coevolution-evidence.v3"
REPORT_SCHEMA_VERSION = "adversarial-feasibility-frontier.v2"
_GIT_SHA = re.compile(r"^(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})$")
_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
_ADMISSIBILITY_VERDICTS = {
    "structurally_invalid",
    "geometric_or_kinodynamic_impossibility",
    "admissible_feasibility_unknown",
    "empirically_feasible",
    "planner_specific_failure",
}
_PLANNER_STATUSES = {"solved", "unsolved", "mixed", "unknown"}
_EVIDENCE_STATUSES = {"complete", "failed", "partial", "missing", "unknown"}
_EXECUTION_MODES = {"native", "adapter", "mixed", "unknown"}
_READINESS_STATUSES = {"native", "adapter", "fallback", "degraded"}
_AVAILABILITY_STATUSES = {"available", "partial-failure", "failed", "not_available"}
_BENCHMARK_READY_STATUSES = {"native", "adapter"}
_CONFIRMED_FEASIBILITY_VERDICTS = {"empirically_feasible", "planner_specific_failure"}
_CANDIDATE_STATUSES = {
    "complete",
    "invalid",
    "failed",
    "partial",
    "missing",
    "unknown",
    "fallback",
    "degraded",
}
_CORPUS_DISPOSITIONS = {
    "admitted",
    "duplicate",
    "pending",
    "rejected",
    "not_applicable",
    "unknown",
}
_EVALUATION_SETS = ("fixed", "regression", "held_out")
_OPTIMIZER_SELECTION_SCHEMA = "frontier-optimizer-selection.v1"
_FALSIFICATION_SOURCE_SCHEMA = "frontier-falsification-source.v3"
_EVALUATION_SOURCE_SCHEMA = "frontier-evaluation-source.v2"
_CORPUS_CASE_STATUS_SCHEMA = "frontier-corpus-case-status.v2"
_ADMISSIBILITY_EVIDENCE_SCHEMA = "scenario_admissibility.v1"
_DIAGNOSTIC_CLAIM_BOUNDARY = "diagnostic_only_not_benchmark_evidence"
_ORACLE_EVIDENCE_SCHEMAS = {
    "scenario_feasibility_oracle.v1",
    "envelope_sensitivity_axis.v1",
    "issue_5574_feasibility_oracle_report.v1",
}
_SEARCH_CANDIDATE_SOURCE_FIELDS = (
    "candidate_id",
    "case_id",
    "scenario_id",
    "scenario_artifact_sha256",
    "evaluation_status",
    "admissibility_verdict",
    "admissibility_evidence_artifact",
    "target_failure_observed",
    "replay_status",
    "corpus_disposition",
    "replay_artifact",
)
_EVALUATION_ROW_SOURCE_FIELDS = (
    "record_id",
    "scenario_id",
    "scenario_seed",
    "evidence_status",
    "execution_mode",
    "readiness_status",
    "availability_status",
    "eligible",
    "success",
    "collision",
    "minimum_clearance",
    "ped_force_q95",
)
_STOP_REASONS = {
    "budget_exhausted",
    "no_new_admissible_counterexample",
    "planner_improvement_stalled",
    "validity_gate_failed",
    "error",
    "manual_stop",
}


class FrontierReportError(ValueError):
    """Raised when persisted evidence cannot support a trustworthy report."""


def planner_color(key: str) -> str:
    """Load the shared figure palette only when rendering needs it."""
    return import_module("robot_sf.benchmark.figures.style").planner_color(key)


def publication_style(*, size: str) -> Any:
    """Load the shared publication style only when rendering needs it."""
    return import_module("robot_sf.benchmark.figures.style").publication_style(size=size)


def build_provenance(**kwargs: Any) -> dict[str, Any]:
    """Load the shared figure provenance helper only when rendering needs it."""
    return import_module("robot_sf.benchmark.figures.provenance").build_provenance(**kwargs)


def save_publication_figure(*args: Any, **kwargs: Any) -> list[Path]:
    """Load the shared figure writer only when rendering needs it."""
    return import_module("robot_sf.benchmark.figures.export").save_publication_figure(
        *args, **kwargs
    )


@dataclass(frozen=True)
class _EvaluationSetContext:
    experiment_id: Any
    round_number: int
    set_name: str
    planner: dict[str, Any]


@dataclass(frozen=True)
class _AdmissibilityContext:
    case_id: Any
    verdict: Any
    scenario_id: Any
    scenario_artifact_sha256: Any
    target_failure_observed: Any
    source_revision: Any
    planner: dict[str, Any]


def build_frontier_report(
    evidence: dict[str, Any],
    *,
    evidence_root: Path,
    input_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate a persisted round bundle and derive report values from its rows.

    Artifact paths in the bundle are relative to ``evidence_root`` and are
    checked for containment, presence, and SHA-256 before the report is built.
    """
    if not isinstance(evidence, dict):
        raise FrontierReportError("evidence root must be a JSON object")
    root = evidence_root.resolve()
    artifacts: dict[str, dict[str, str]] = {}
    _validate_evidence(evidence, root, artifacts)

    round_reports: list[dict[str, Any]] = []
    discovered_counterexamples: dict[str, int] = {}
    historical_case_states = _historical_case_states(evidence["rounds"])
    admitted_unknown_cases: dict[str, int] = {}
    current_case_verdicts: dict[str, str] = {}
    known_confirmed_case_ids: set[str] = set()

    for round_data in evidence["rounds"]:
        number = round_data["round_number"]
        confirmed_before_round = set(known_confirmed_case_ids)
        historical_confirmed_this_round = [
            case_id
            for case_id, state in historical_case_states.items()
            if state["confirmed_counterexample_evidence_round"] == number
        ]
        for case_id, state in historical_case_states.items():
            if state["initial_unknown_round"] == number:
                admitted_unknown_cases.setdefault(case_id, number)
            if state["confirmed_counterexample_evidence_round"] == number:
                known_confirmed_case_ids.add(case_id)
        eval_summary = {
            name: _summarize_evaluation_set(round_data["evaluation_sets"][name])
            for name in _EVALUATION_SETS
        }
        _add_cohort_comparability(eval_summary, round_data, round_reports)
        candidates = round_data["falsification"]["candidates"]
        (
            verified_case_ids,
            repeated_verified_case_ids,
            admitted_case_ids,
            unknown_case_ids,
        ) = _classify_search_candidates(
            candidates,
            number,
            discovered_counterexamples,
            admitted_unknown_cases,
            current_case_verdicts,
            confirmed_before_round,
        )

        observations = round_data["case_observations"]
        loop_feasibility_upgrade_observations = [
            observation
            for observation in observations
            if observation["origin_round"] > 0
            if _is_followup_feasibility_upgrade(number, observation, current_case_verdicts)
        ]
        historical_verified_upgrades = [
            case_id
            for case_id, state in historical_case_states.items()
            if state["feasibility_upgrade_credit_round"] == number
        ]
        unverified_feasibility_upgrades = [
            case_id
            for case_id, state in historical_case_states.items()
            if state["feasibility_upgrade_round"] == number
            and (
                state["first_verified_target_failure_round"] is None
                or state["first_verified_target_failure_round"] > number
            )
        ]
        feasibility_upgrades = [
            observation["case_id"] for observation in loop_feasibility_upgrade_observations
        ] + historical_verified_upgrades
        verified_case_ids.extend(feasibility_upgrades)
        for case_id in feasibility_upgrades:
            discovered_counterexamples.setdefault(case_id, number)
        known_confirmed_case_ids.update(verified_case_ids)
        for observation in observations:
            current_case_verdicts[observation["case_id"]] = observation["admissibility_verdict"]
        state_counts = Counter(item["planner_status"] for item in observations)
        verdict_counts = Counter(item["admissibility_verdict"] for item in observations)
        current_verdict_counts = Counter(current_case_verdicts.values())
        replay_counts = Counter(item["replay_status"] for item in observations)

        current_verified_status = _current_counterexample_status(
            known_confirmed_case_ids, observations
        )
        falsification = round_data["falsification"]
        candidate_status_counts = Counter(item["evaluation_status"] for item in candidates)
        candidate_verdict_counts = Counter(item["admissibility_verdict"] for item in candidates)
        search_verified_case_ids = [
            case_id for case_id in verified_case_ids if case_id not in historical_verified_upgrades
        ]
        no_verified_counterexample = not search_verified_case_ids
        no_discovery_statement = None
        if no_verified_counterexample:
            repeated_clause = (
                f"{len(set(repeated_verified_case_ids))} known corpus case(s) "
                "were replay-verified again. "
                if repeated_verified_case_ids
                else ""
            )
            historical_clause = (
                f"{len(historical_confirmed_this_round)} historical case(s) reached "
                "replay-verified counterexample status from persisted follow-up evidence. "
                if historical_confirmed_this_round
                else ""
            )
            no_discovery_statement = (
                "No new unique replay-verified planner counterexample was recorded by this round's "
                f"finite search budget ({falsification['budget']['candidate_limit']} candidates, "
                f"{falsification['budget']['simulator_invocations']} simulator invocations). "
                f"{repeated_clause}{historical_clause}"
                "This does not establish that no counterexample exists."
            )

        round_reports.append(
            {
                "round_number": number,
                "planner": round_data["planner"],
                "optimization": round_data["optimization"],
                "falsification": {
                    "method": falsification["method"],
                    "objective": falsification["objective"],
                    "target_failure_predicate": falsification["target_failure_predicate"],
                    "search_space_id": falsification["search_space_id"],
                    "seeds": falsification["seeds"],
                    "budget": falsification["budget"],
                    "stop_reason": falsification["stop_reason"],
                    "candidate_record_count": len(candidates),
                    "candidate_status_counts": dict(sorted(candidate_status_counts.items())),
                    "admissibility_verdict_counts": dict(sorted(candidate_verdict_counts.items())),
                    "admitted_case_ids": sorted(set(admitted_case_ids)),
                    "admitted_unknown_feasibility_case_ids": sorted(set(unknown_case_ids)),
                    "verified_counterexample_case_ids": sorted(set(verified_case_ids)),
                    "repeated_verified_counterexample_case_ids": sorted(
                        set(repeated_verified_case_ids)
                    ),
                    "feasibility_upgrades_from_follow_up_case_ids": sorted(
                        set(feasibility_upgrades)
                    ),
                    "feasibility_upgrades_without_verified_counterexample_case_ids": sorted(
                        set(unverified_feasibility_upgrades)
                    ),
                    "historical_counterexamples_confirmed_this_round_case_ids": sorted(
                        historical_confirmed_this_round
                    ),
                    "candidate_records": candidates,
                    "no_verified_counterexample_statement": no_discovery_statement,
                    "artifact": falsification["artifact"],
                },
                "evaluation_sets": eval_summary,
                "case_frontier": {
                    "observation_count": len(observations),
                    "planner_status_counts": {
                        status: state_counts.get(status, 0) for status in sorted(_PLANNER_STATUSES)
                    },
                    "admissibility_verdict_counts": dict(sorted(verdict_counts.items())),
                    "admissibility_partition_counts": {
                        "invalid": verdict_counts.get("structurally_invalid", 0),
                        "infeasible": verdict_counts.get(
                            "geometric_or_kinodynamic_impossibility", 0
                        ),
                        "feasibility_unknown": verdict_counts.get(
                            "admissible_feasibility_unknown", 0
                        ),
                        "empirically_feasible": verdict_counts.get("empirically_feasible", 0),
                        "planner_specific_failure": verdict_counts.get(
                            "planner_specific_failure", 0
                        ),
                    },
                    "replay_status_counts": dict(sorted(replay_counts.items())),
                    "current_admissibility_verdict_counts": dict(
                        sorted(current_verdict_counts.items())
                    ),
                    "verified_counterexamples_cumulative": len(discovered_counterexamples),
                    "confirmed_counterexamples_in_corpus": len(known_confirmed_case_ids),
                    "admitted_unknown_feasibility_cases_cumulative": len(admitted_unknown_cases),
                    "current_unknown_feasibility_case_count": current_verdict_counts.get(
                        "admissible_feasibility_unknown", 0
                    ),
                    "verified_counterexample_status": current_verified_status,
                    "observations": observations,
                },
            }
        )

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "complete",
        "experiment_id": evidence["experiment_id"],
        "input_schema_version": INPUT_SCHEMA_VERSION,
        "input_sha256": input_sha256,
        "source_revision": evidence["source_revision"],
        "evidence_kind": evidence["evidence_kind"],
        "simulator_identity": evidence["simulator_identity"],
        "scenario_space_id": evidence["scenario_space_id"],
        "round_count": len(round_reports),
        "claim_boundary": (
            "Summary of declared finite-budget evidence only; evidence_kind is "
            "caller-declared metadata, and simulator_run is unverified without an "
            "independently verified producer binding. Synthetic fixtures are "
            "implementation-only. No counterexample found under a budget is not proof of "
            "absence; unknown feasibility is preserved; simulator success is not a claim "
            "of real-world safety or global optimality."
        ),
        "source_artifacts": [artifacts[key] for key in sorted(artifacts)],
        "rounds": round_reports,
    }


def render_frontier_markdown(
    report: dict[str, Any], *, evidence_root_relative_to_report: str = "."
) -> str:
    """Render the compact report text from the machine-readable summary."""
    headline = {
        "synthetic_fixture": "Synthetic fixture (implementation-only) feasibility-frontier report",
        "simulator_run": "Declared simulator-run evidence (unverified) feasibility-frontier report",
        "historical_artifact": "Historical-artifact feasibility-frontier report",
    }[report["evidence_kind"]]
    lines = [
        f"# {headline}: {report['experiment_id']}",
        "",
        f"- Source revision: `{report['source_revision']}`",
        f"- Simulator: `{report['simulator_identity']}`",
        f"- Scenario space: `{report['scenario_space_id']}`",
        f"- Evidence kind: `{report['evidence_kind']}`",
        f"- Rounds: {report['round_count']}",
        f"- Input SHA-256: `{report.get('input_sha256') or 'not supplied'}`",
        f"- Input artifact: {_artifact_link(report.get('input_artifact', {}).get('path'), evidence_root_relative_to_report)}",
        "",
        report["claim_boundary"],
        "",
        "Rates use complete, explicitly eligible rows with benchmark-ready availability and "
        "the relevant outcome recorded. Success and collision use separate denominators. "
        "Execution mode, runtime readiness, availability, excluded rows, and missing outcomes "
        "remain visible in the accounting below.",
        "",
        "| Round | Planner/config | Optimization method and budget | Search method and budget | Held-out success | Regression success | Confirmed counterexamples in corpus |",
        "|---:|---|---|---|---:|---:|---:|",
    ]
    for item in report["rounds"]:
        optimization = item["optimization"]
        search = item["falsification"]
        held_out = item["evaluation_sets"]["held_out"]
        regression = item["evaluation_sets"]["regression"]
        planner = item["planner"]
        budget = optimization["budget"]
        search_budget = search["budget"]
        lines.append(
            "| {round_number} | `{planner_id}` / `{config}` | {method}: {completed}/{limit} "
            "proposals, {calls} simulator calls | {search_method}: {completed_search}/"
            "{limit_search} candidates, {search_calls} simulator calls | {held} | {reg} | {cumulative} |".format(
                round_number=item["round_number"],
                planner_id=planner["planner_id"],
                config=planner["config_identity_sha256"][:12],
                method=optimization["method"],
                completed=budget["proposals_completed"],
                limit=budget["proposal_limit"],
                calls=budget["simulator_invocations"],
                search_method=search["method"],
                completed_search=search_budget["candidates_completed"],
                limit_search=search_budget["candidate_limit"],
                search_calls=search_budget["simulator_invocations"],
                held=_format_rate(held_out),
                reg=_format_rate(regression),
                cumulative=item["case_frontier"]["confirmed_counterexamples_in_corpus"],
            )
        )
    for item in report["rounds"]:
        lines.extend(
            [
                "",
                f"## Round {item['round_number']}",
                "",
                f"- Planner: `{item['planner']['planner_id']}`; config SHA-256 "
                f"`{item['planner']['config_identity_sha256']}`; source revision "
                f"`{item['planner']['source_revision']}`.",
                f"- Optimization objective: `{item['optimization']['objective']}`; "
                f"method `{item['optimization']['method']}`; seeds "
                f"`{item['optimization']['seeds']}`; budget `{item['optimization']['budget']}`.",
                f"- Falsification objective: `{item['falsification']['objective']}`; "
                f"method `{item['falsification']['method']}`; seeds "
                f"`{item['falsification']['seeds']}`; budget "
                f"`{item['falsification']['budget']}`; stop reason "
                f"`{item['falsification']['stop_reason']}`.",
                "- Evaluation accounting:",
            ]
        )
        for set_name in _EVALUATION_SETS:
            summary = item["evaluation_sets"][set_name]
            lines.append(
                f"  - `{set_name}`: {summary['eligible_episode_count']}/"
                f"{summary['expected_episode_count']} expected rows benchmark-eligible; "
                f"{summary['reported_episode_count']} reported, "
                f"identity accounting `{summary['identity_accounting_status']}` "
                f"(ID manifest SHA-256 `{summary['expected_episode_ids_sha256'][:12]}`, "
                f"scenario/seed manifest SHA-256 "
                f"`{summary['scenario_seed_manifest_sha256'][:12]}`; cohort comparison "
                f"`{summary['cohort_comparison_status']}`; success-rate comparison "
                f"`{summary['success_rate_comparison_status']}`); "
                f"{summary['missing_record_count']} expected rows missing "
                f"`{summary['missing_record_ids']}`, "
                f"{summary['unexpected_record_count']} unexpected; "
                f"{_format_rate(summary)} success; {_format_collision_rate(summary)} collision; "
                f"evidence statuses `{summary['evidence_status_counts']}`, execution modes "
                f"`{summary['execution_mode_counts']}`, readiness "
                f"`{summary['readiness_status_counts']}`, availability "
                f"`{summary['availability_status_counts']}`; missing success outcomes "
                f"`{summary['missing_success_record_ids']}`, missing collision outcomes "
                f"`{summary['missing_collision_record_ids']}`; excluded records "
                f"`{summary['excluded_record_ids']}`."
            )
        lines.extend(
            [
                "- Case frontier counts:",
                f"  - Planner status: `{item['case_frontier']['planner_status_counts']}`.",
                f"  - Admissibility verdict: `{item['case_frontier']['admissibility_verdict_counts']}`.",
                f"  - Replay status: `{item['case_frontier']['replay_status_counts']}`.",
                f"  - Feasibility/admissibility partition: "
                f"`{item['case_frontier']['admissibility_partition_counts']}`.",
                f"  - New verified counterexamples discovered in this loop (cumulative): "
                f"{item['case_frontier']['verified_counterexamples_cumulative']}; "
                f"confirmed counterexamples known to the corpus: "
                f"{item['case_frontier']['confirmed_counterexamples_in_corpus']}.",
                f"  - Currently unknown-feasibility cases: "
                f"{item['case_frontier']['current_unknown_feasibility_case_count']}; "
                f"cases ever admitted with unknown feasibility: "
                f"{item['case_frontier']['admitted_unknown_feasibility_cases_cumulative']}.",
            ]
        )
        statement = item["falsification"]["no_verified_counterexample_statement"]
        if statement:
            lines.extend(["", statement])
        case_rows = item["case_frontier"]["observations"]
        if case_rows:
            lines.extend(
                [
                    "",
                    "| Case | Origin round/candidate | Admissibility | Planner status | Replay status | Evidence | Search artifact | Replay artifact |",
                    "|---|---|---|---|---|---|---|---|",
                ]
            )
            for case in case_rows:
                replay_artifact = case.get("replay_artifact")
                replay_path = (
                    _artifact_link(replay_artifact["path"], evidence_root_relative_to_report)
                    if replay_artifact
                    else "unavailable"
                )
                search_path = _artifact_link(
                    case["origin_search_artifact"]["path"], evidence_root_relative_to_report
                )
                lines.append(
                    f"| `{case['case_id']}` | {case['origin_round']} / "
                    f"`{case['origin_candidate_id'] or 'historical'}` | "
                    f"`{case['admissibility_verdict']}` | `{case['planner_status']}` | "
                    f"`{case['replay_status']}` | `{case['evidence_status']}` | "
                    f"`{search_path}` | `{replay_path}` |"
                )
        candidate_rows = item["falsification"]["candidate_records"]
        if candidate_rows:
            lines.extend(
                [
                    "",
                    "Search candidate accounting (including invalid, failed, and unresolved rows):",
                    "",
                    "| Candidate | Evaluation | Admissibility | Target failure | Replay | Corpus disposition | Case |",
                    "|---|---|---|---|---|---|---|",
                ]
            )
            for candidate in candidate_rows:
                lines.append(
                    f"| `{candidate['candidate_id']}` | `{candidate['evaluation_status']}` | "
                    f"`{candidate['admissibility_verdict']}` | "
                    f"`{candidate['target_failure_observed']}` | "
                    f"`{candidate['replay_status']}` | `{candidate['corpus_disposition']}` | "
                    f"`{candidate.get('case_id') or 'none'}` |"
                )
    lines.extend(["", "## Evidence artifacts", ""])
    for artifact in report["source_artifacts"]:
        lines.append(
            f"- {_artifact_link(artifact['path'], evidence_root_relative_to_report)} — `{artifact['sha256']}`; "
            f"{artifact['role']}; source revision `{artifact['source_revision']}`."
        )
    lines.append("")
    return "\n".join(lines)


def _plot_evaluation_performance(
    axis: Any, rounds: list[dict[str, Any]], x_values: list[int]
) -> None:
    """Plot rates as points and connect only adjacent, cohort-comparable rounds."""
    for set_name, label in (
        ("fixed", "Fixed"),
        ("regression", "Regression"),
        ("held_out", "Held out"),
    ):
        values = [item["evaluation_sets"][set_name]["success_rate"] for item in rounds]
        color = planner_color(f"frontier_{set_name}")
        axis.plot(
            x_values,
            [float("nan") if value is None else value for value in values],
            marker="o",
            linestyle="None",
            color=color,
            label=label,
        )
        for index in range(1, len(rounds)):
            current_summary = rounds[index]["evaluation_sets"][set_name]
            previous_value = values[index - 1]
            current_value = values[index]
            if (
                current_summary["success_rate_comparable_to_previous_round"] is True
                and previous_value is not None
                and current_value is not None
            ):
                axis.plot(
                    x_values[index - 1 : index + 1],
                    [previous_value, current_value],
                    linewidth=1.8,
                    color=color,
                    label="_nolegend_",
                )
        for x, item in zip(x_values, rounds, strict=True):
            summary = item["evaluation_sets"][set_name]
            value = summary["success_rate"]
            if value is not None:
                marker = (
                    "*"
                    if summary["success_rate_comparison_status"]
                    in {"non_comparable_cohort", "non_comparable_success_sample"}
                    else ""
                )
                axis.annotate(
                    f"{summary['successes']}/{summary['success_denominator']}{marker}",
                    (x, value),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                    fontsize=7,
                )
    has_changed_cohort = any(
        item["evaluation_sets"][set_name]["success_rate_comparison_status"]
        in {"non_comparable_cohort", "non_comparable_success_sample"}
        for item in rounds
        for set_name in _EVALUATION_SETS
    )
    if has_changed_cohort:
        axis.text(
            0.01,
            0.01,
            "* cohort or eligible success sample changed; trend segment omitted",
            transform=axis.transAxes,
            fontsize=7,
            va="bottom",
        )


def write_frontier_figure(report: dict[str, Any], output_base: Path) -> list[Path]:
    """Write the performance/discovery figure with the repository figure helpers."""
    plt = try_import("matplotlib.pyplot")
    if plt is None:  # pragma: no cover - environment-specific dependency guard
        raise FrontierReportError("matplotlib is required to render the frontier figure")

    rounds = report["rounds"]
    x_values = [item["round_number"] for item in rounds]
    with publication_style(size="double"):
        figure, axes = plt.subplots(2, 1, figsize=(7.0, 7.1), constrained_layout=True)
        performance_axis, cases_axis = axes
        _plot_evaluation_performance(performance_axis, rounds, x_values)
        performance_axis.set_ylim(-0.05, 1.05)
        performance_axis.set_xticks(x_values)
        performance_axis.set_xlabel("Round")
        performance_axis.set_ylabel("Success fraction")
        performance_axis.set_title("Planner evaluation by case set")
        performance_axis.grid(axis="y", alpha=0.25)
        performance_axis.legend(frameon=False, ncol=3, loc="lower right")

        cumulative_known = [
            item["case_frontier"]["confirmed_counterexamples_in_corpus"] for item in rounds
        ]
        current_unknown = [
            item["case_frontier"]["current_unknown_feasibility_case_count"] for item in rounds
        ]
        solved = [
            item["case_frontier"]["verified_counterexample_status"]["solved"] for item in rounds
        ]
        unsolved = [
            item["case_frontier"]["verified_counterexample_status"]["unsolved"] for item in rounds
        ]
        mixed = [
            item["case_frontier"]["verified_counterexample_status"]["mixed"] for item in rounds
        ]
        unknown_planner = [
            item["case_frontier"]["verified_counterexample_status"]["unknown"] for item in rounds
        ]
        not_observed = [
            item["case_frontier"]["verified_counterexample_status"]["not_observed_this_round"]
            for item in rounds
        ]
        invalid = [
            item["falsification"]["admissibility_verdict_counts"].get("structurally_invalid", 0)
            for item in rounds
        ]
        infeasible = [
            item["falsification"]["admissibility_verdict_counts"].get(
                "geometric_or_kinodynamic_impossibility", 0
            )
            for item in rounds
        ]
        cases_axis.plot(
            x_values,
            cumulative_known,
            marker="o",
            linewidth=1.8,
            color=planner_color("frontier_verified_counterexamples"),
            label="Confirmed counterexamples known to corpus",
        )
        cases_axis.plot(
            x_values,
            solved,
            marker="s",
            linewidth=1.8,
            color=planner_color("frontier_solved_counterexamples"),
            label="Known cases solved by current planner",
        )
        cases_axis.plot(
            x_values,
            unsolved,
            marker="x",
            linewidth=1.8,
            color=planner_color("frontier_unsolved_counterexamples"),
            label="Known cases still failing for current planner",
        )
        cases_axis.plot(
            x_values,
            mixed,
            marker="D",
            linewidth=1.5,
            linestyle=":",
            color=planner_color("frontier_mixed_counterexamples"),
            label="Known cases with mixed planner outcomes",
        )
        cases_axis.plot(
            x_values,
            unknown_planner,
            marker="*",
            linewidth=1.5,
            linestyle=":",
            color=planner_color("frontier_unknown_planner_status"),
            label="Known cases with unknown planner outcome",
        )
        cases_axis.plot(
            x_values,
            not_observed,
            marker=".",
            linewidth=1.5,
            linestyle=":",
            color=planner_color("frontier_not_observed_cases"),
            label="Known cases not observed in round",
        )
        cases_axis.plot(
            x_values,
            current_unknown,
            marker="^",
            linewidth=1.8,
            linestyle="--",
            color=planner_color("frontier_unknown_feasibility"),
            label="Cases currently with unknown feasibility",
        )
        cases_axis.plot(
            x_values,
            invalid,
            marker="v",
            linewidth=1.4,
            linestyle="--",
            color=planner_color("frontier_invalid_candidates"),
            label="Structurally invalid search candidates (round)",
        )
        cases_axis.plot(
            x_values,
            infeasible,
            marker="P",
            linewidth=1.4,
            linestyle="--",
            color=planner_color("frontier_infeasible_candidates"),
            label="Geometric/kinodynamic impossibilities (round)",
        )
        cases_axis.set_xticks(x_values)
        cases_axis.set_xlabel("Round")
        cases_axis.set_ylabel("Case/candidate count")
        cases_axis.set_title("Counterexample memory, planner status, and scenario classification")
        cases_axis.grid(axis="y", alpha=0.25)
        cases_axis.legend(frameon=False, loc="best", fontsize=7, ncol=2)
        figure_evidence_label = {
            "synthetic_fixture": "Synthetic Fixture evidence",
            "simulator_run": "Declared Simulator Run evidence (unverified)",
            "historical_artifact": "Historical Artifact evidence",
        }[report["evidence_kind"]]
        figure_title = f"{figure_evidence_label} — Finite-budget planner–falsifier frontier"
        figure.suptitle(figure_title)

        provenance = build_provenance(
            source_artifacts=[
                {"path": item["path"], "hash": item["sha256"]}
                for item in (
                    [*report["source_artifacts"], report["input_artifact"]]
                    if report.get("input_artifact")
                    else report["source_artifacts"]
                )
            ],
            generator_command="build_adversarial_feasibility_frontier_report",
            figure_formats=["png", "pdf"],
            claim_boundary=report["claim_boundary"],
        )
        provenance.update(
            {
                "source_revision": report["source_revision"],
                "evidence_kind": report["evidence_kind"],
                "figure_title": figure_title,
            }
        )
        try:
            return save_publication_figure(
                figure,
                output_base,
                formats=("png", "pdf"),
                provenance=provenance,
                timestamp=datetime(2000, 1, 1, tzinfo=UTC),
            )
        finally:
            plt.close(figure)


def write_frontier_report(
    input_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Load and validate persisted evidence, then write JSON, Markdown, and figure."""
    input_path = input_path.resolve()
    payload_bytes = input_path.read_bytes()
    try:
        evidence = json.loads(payload_bytes)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise FrontierReportError(f"could not parse evidence JSON at {input_path}") from exc
    digest = hashlib.sha256(payload_bytes).hexdigest()
    report = build_frontier_report(
        evidence,
        evidence_root=input_path.parent,
        input_sha256=digest,
    )
    report["input_artifact"] = {
        "path": input_path.name,
        "sha256": digest,
        "source_revision": report["source_revision"],
        "role": "frontier-report-input",
        "schema_version": INPUT_SCHEMA_VERSION,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_outputs = (
        "frontier_report.json",
        "frontier_report.md",
        "frontier.png",
        "frontier.pdf",
        "frontier.provenance.json",
    )
    existing_outputs = [name for name in expected_outputs if (output_dir / name).exists()]
    if existing_outputs:
        raise FrontierReportError(
            "report output already exists; choose a new output directory: "
            + ", ".join(existing_outputs)
        )
    (output_dir / "frontier_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    relative_evidence_root = os.path.relpath(input_path.parent, output_dir.resolve()).replace(
        os.sep, "/"
    )
    (output_dir / "frontier_report.md").write_text(
        render_frontier_markdown(report, evidence_root_relative_to_report=relative_evidence_root),
        encoding="utf-8",
    )
    write_frontier_figure(report, output_dir / "frontier")
    return report


def _validate_evidence(  # noqa: C901 - preserve independent top-level evidence diagnostics.
    evidence: dict[str, Any], evidence_root: Path, artifacts: dict[str, dict[str, str]]
) -> None:
    errors: list[str] = []
    if evidence.get("schema_version") != INPUT_SCHEMA_VERSION:
        errors.append(f"schema_version must be {INPUT_SCHEMA_VERSION!r}")
    if not _is_allowed(
        evidence.get("evidence_kind"),
        {"synthetic_fixture", "simulator_run", "historical_artifact"},
    ):
        errors.append(
            "evidence_kind must be synthetic_fixture, simulator_run, or historical_artifact"
        )
    for field in ("experiment_id", "simulator_identity", "scenario_space_id"):
        _require_text(evidence, field, errors, "evidence")
    _require_sha(evidence.get("source_revision"), "evidence.source_revision", _GIT_SHA, errors)
    rounds = evidence.get("rounds")
    if not isinstance(rounds, list) or len(rounds) < 2:
        errors.append("rounds must contain at least two persisted rounds")
        rounds = []
    expected_numbers = list(range(1, len(rounds) + 1))
    actual_numbers = [
        item.get("round_number") if isinstance(item, dict) else None for item in rounds
    ]
    if actual_numbers != expected_numbers:
        errors.append("round_number values must be contiguous and start at 1")

    round_candidates: dict[int, dict[str, dict[str, Any]]] = {}
    round_search_artifacts: dict[int, Any] = {}
    for round_data in rounds:
        if not isinstance(round_data, dict):
            errors.append("each round must be an object")
            continue
        number = round_data.get("round_number")
        if not isinstance(number, int) or isinstance(number, bool) or number < 1:
            errors.append("round_number must be a positive integer")
            continue
        _validate_round(
            round_data,
            number,
            evidence.get("experiment_id"),
            evidence_root,
            artifacts,
            errors,
        )
        falsification = round_data.get("falsification")
        candidates = falsification.get("candidates", []) if isinstance(falsification, dict) else []
        if isinstance(falsification, dict):
            round_search_artifacts[number] = falsification.get("artifact")
        round_candidates[number] = {
            item.get("case_id"): item
            for item in candidates
            if isinstance(item, dict)
            and item.get("corpus_disposition") == "admitted"
            and isinstance(item.get("case_id"), str)
        }

    _validate_case_dispositions(rounds, errors)
    _validate_case_observation_origins(rounds, round_candidates, round_search_artifacts, errors)
    _validate_evaluation_cohorts(rounds, errors)

    if errors:
        raise FrontierReportError("invalid frontier evidence:\n- " + "\n- ".join(errors))


def _validate_case_observation_origins(  # noqa: C901 - history-linked origin checks share ordered state.
    rounds: list[Any],
    round_candidates: dict[int, dict[str, dict[str, Any]]],
    round_search_artifacts: dict[int, Any],
    errors: list[str],
) -> None:
    latest_observations: dict[str, dict[str, Any]] = {}
    case_origins: dict[str, tuple[int, Any, Any, Any]] = {}
    for round_data in rounds:
        if not isinstance(round_data, dict) or not isinstance(round_data.get("round_number"), int):
            continue
        current_round = round_data["round_number"]
        for observation in round_data.get("case_observations", []):
            if not isinstance(observation, dict):
                continue
            origin_round = observation.get("origin_round")
            case_id = observation.get("case_id")
            _validate_case_origin_identity(current_round, observation, case_origins, errors)
            if not isinstance(case_id, str):
                continue
            if isinstance(origin_round, int) and origin_round > 0:
                _validate_discovered_case_observation(
                    current_round,
                    origin_round,
                    observation,
                    round_candidates,
                    round_search_artifacts,
                    latest_observations,
                    errors,
                )
            elif origin_round == 0:
                previous = latest_observations.get(case_id)
                if previous is not None:
                    _validate_followup_case_observation(
                        current_round,
                        observation,
                        previous["admissibility_verdict"],
                        errors,
                    )
            elif origin_round != 0:
                errors.append(f"round {current_round} case {case_id!r} has invalid origin_round")
            if isinstance(case_id, str):
                latest_observations[case_id] = observation


def _validate_case_origin_identity(
    current_round: int,
    observation: dict[str, Any],
    case_origins: dict[str, tuple[int, Any, Any, Any]],
    errors: list[str],
) -> None:
    case_id = observation.get("case_id")
    origin_round = observation.get("origin_round")
    if (
        not isinstance(case_id, str)
        or not isinstance(origin_round, int)
        or isinstance(origin_round, bool)
    ):
        return
    identity = (
        origin_round,
        observation.get("origin_candidate_id"),
        observation.get("scenario_id"),
        observation.get("scenario_artifact_sha256"),
    )
    previous_identity = case_origins.setdefault(case_id, identity)
    if previous_identity != identity:
        errors.append(
            f"round {current_round} case {case_id!r} changes its origin round or candidate identity"
        )


def _validate_discovered_case_observation(
    current_round: int,
    origin_round: int,
    observation: dict[str, Any],
    round_candidates: dict[int, dict[str, dict[str, Any]]],
    round_search_artifacts: dict[int, Any],
    latest_observations: dict[str, dict[str, Any]],
    errors: list[str],
) -> None:
    case_id = observation.get("case_id")
    if origin_round > current_round:
        errors.append(f"round {current_round} case {case_id!r} has invalid origin_round")
        return
    origin_candidate = round_candidates.get(origin_round, {}).get(case_id)
    if origin_candidate is None:
        errors.append(
            f"round {current_round} case {case_id!r} refers to no admitted discovery "
            f"in origin round {origin_round}"
        )
        return
    if observation.get("origin_candidate_id") != origin_candidate.get("candidate_id"):
        errors.append(
            f"round {current_round} case {case_id!r} origin candidate does not match "
            "its discovery record"
        )
        return
    if any(
        observation.get(field) != origin_candidate.get(field)
        for field in ("scenario_id", "scenario_artifact_sha256")
    ):
        errors.append(
            f"round {current_round} case {case_id!r} scenario identity does not match "
            "its discovery candidate"
        )
        return
    expected_search = round_search_artifacts.get(origin_round)
    if not _same_artifact_reference(observation.get("origin_search_artifact"), expected_search):
        errors.append(
            f"round {current_round} case {case_id!r} origin search artifact does not match "
            f"round {origin_round}'s checksummed falsification source"
        )
    if not _same_artifact_reference(
        observation.get("replay_artifact"), origin_candidate.get("replay_artifact")
    ):
        errors.append(
            f"round {current_round} case {case_id!r} replay artifact does not match "
            "its origin candidate replay"
        )
    if current_round == origin_round:
        _validate_same_round_case_observation(current_round, observation, origin_candidate, errors)
        observation_admissibility = observation.get("admissibility_evidence_artifact")
        candidate_admissibility = origin_candidate.get("admissibility_evidence_artifact")
        if (
            observation_admissibility is not None
            or candidate_admissibility is not None
            or observation.get("admissibility_verdict") in _CONFIRMED_FEASIBILITY_VERDICTS
        ) and not _same_artifact_reference(observation_admissibility, candidate_admissibility):
            errors.append(
                f"round {current_round} case {case_id!r} admissibility evidence does not "
                "match its origin candidate"
            )
        return
    previous = latest_observations.get(case_id)
    previous_verdict = (
        previous["admissibility_verdict"]
        if previous is not None
        else origin_candidate.get("admissibility_verdict")
    )
    _validate_followup_case_observation(current_round, observation, previous_verdict, errors)


def _same_artifact_reference(left: Any, right: Any) -> bool:
    """Compare the complete stable identity of two declared artifact references."""
    fields = ("path", "sha256", "source_revision", "role", "schema_version")
    return (
        isinstance(left, dict)
        and isinstance(right, dict)
        and all(left.get(field) == right.get(field) for field in fields)
    )


def _validate_same_round_case_observation(
    current_round: int,
    observation: dict[str, Any],
    origin_candidate: dict[str, Any],
    errors: list[str],
) -> None:
    case_id = observation.get("case_id")
    if observation.get("admissibility_verdict") != origin_candidate.get("admissibility_verdict"):
        errors.append(
            f"round {current_round} case {case_id!r} conflicts with its "
            "same-round discovery admissibility verdict"
        )
    if not _same_artifact_reference(
        observation.get("admissibility_evidence_artifact"),
        origin_candidate.get("admissibility_evidence_artifact"),
    ):
        errors.append(
            f"round {current_round} case {case_id!r} admissibility evidence does not match "
            "its origin candidate"
        )
    if observation.get("replay_status") != origin_candidate.get("replay_status"):
        errors.append(
            f"round {current_round} case {case_id!r} conflicts with its "
            "same-round discovery replay status"
        )
    if (
        origin_candidate.get("target_failure_observed") is True
        and observation.get("planner_status") == "solved"
    ):
        errors.append(
            f"round {current_round} case {case_id!r} is marked solved despite "
            "the same-round observed target failure"
        )


def _validate_followup_case_observation(
    current_round: int,
    observation: dict[str, Any],
    previous_verdict: str,
    errors: list[str],
) -> None:
    observed_verdict = observation.get("admissibility_verdict")
    if observed_verdict == previous_verdict:
        return
    evidence = observation.get("admissibility_evidence_artifact")
    valid_upgrade = (
        previous_verdict == "admissible_feasibility_unknown"
        and _is_allowed(observed_verdict, _CONFIRMED_FEASIBILITY_VERDICTS)
        and observation.get("evidence_status") == "complete"
        and isinstance(evidence, dict)
        and evidence.get("role") == "admissibility-evidence"
    )
    if not valid_upgrade:
        errors.append(
            f"round {current_round} case {observation.get('case_id')!r} has an unsupported "
            "admissibility-verdict transition"
        )


def _is_followup_feasibility_upgrade(
    number: int, observation: dict[str, Any], latest_verdicts: dict[str, str]
) -> bool:
    """Return whether this observation first upgrades the latest unknown verdict."""
    origin_round = observation["origin_round"]
    if origin_round >= number:
        return False
    if observation["admissibility_verdict"] not in _CONFIRMED_FEASIBILITY_VERDICTS:
        return False
    return latest_verdicts.get(observation["case_id"]) == "admissible_feasibility_unknown"


def _historical_case_states(rounds: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Return round-indexed evidence milestones for origin-zero historical cases."""
    states: dict[str, dict[str, Any]] = {}
    for round_data in rounds:
        number = round_data["round_number"]
        for observation in round_data["case_observations"]:
            if observation["origin_round"] != 0:
                continue
            case_id = observation["case_id"]
            verdict = observation["admissibility_verdict"]
            state = states.setdefault(
                case_id,
                {
                    "first_observed_round": number,
                    "initial_unknown_round": (
                        number if verdict == "admissible_feasibility_unknown" else None
                    ),
                    "first_confirmed_round": None,
                    "feasibility_upgrade_round": None,
                    "first_verified_target_failure_round": None,
                    "latest_verdict": None,
                },
            )
            previous_verdict = state["latest_verdict"]
            if (
                state["first_confirmed_round"] is None
                and verdict in _CONFIRMED_FEASIBILITY_VERDICTS
            ):
                state["first_confirmed_round"] = number
            if (
                state["feasibility_upgrade_round"] is None
                and previous_verdict == "admissible_feasibility_unknown"
                and verdict in _CONFIRMED_FEASIBILITY_VERDICTS
            ):
                state["feasibility_upgrade_round"] = number
            if (
                state["first_verified_target_failure_round"] is None
                and observation["replay_status"] == "verified"
                and observation["planner_status"] in {"unsolved", "mixed"}
            ):
                state["first_verified_target_failure_round"] = number
            state["latest_verdict"] = verdict
    for state in states.values():
        confirmed_round = state["first_confirmed_round"]
        failure_round = state["first_verified_target_failure_round"]
        upgrade_round = state["feasibility_upgrade_round"]
        state["confirmed_counterexample_evidence_round"] = (
            max(confirmed_round, failure_round)
            if confirmed_round is not None and failure_round is not None
            else None
        )
        state["feasibility_upgrade_credit_round"] = (
            max(upgrade_round, failure_round)
            if upgrade_round is not None and failure_round is not None
            else None
        )
    return states


def _validate_round(  # noqa: C901, PLR0912, PLR0915 - retain every round-local blocker for diagnosis.
    data: dict[str, Any],
    number: int,
    experiment_id: Any,
    evidence_root: Path,
    artifacts: dict[str, dict[str, str]],
    errors: list[str],
) -> None:
    prefix = f"rounds[{number - 1}]"
    _require_sha(data.get("source_revision"), f"{prefix}.source_revision", _GIT_SHA, errors)
    planner = data.get("planner")
    if not isinstance(planner, dict):
        errors.append(f"{prefix}.planner must be an object")
        planner = {}
    for field in ("planner_id",):
        _require_text(planner, field, errors, f"{prefix}.planner")
    _require_sha(
        planner.get("config_identity_sha256"),
        f"{prefix}.planner.config_identity_sha256",
        _SHA256,
        errors,
    )
    _require_sha(
        planner.get("source_revision"), f"{prefix}.planner.source_revision", _GIT_SHA, errors
    )

    optimization = data.get("optimization")
    if not isinstance(optimization, dict):
        errors.append(f"{prefix}.optimization must be an object")
        optimization = {}
    for field in ("method", "objective", "selection_rule"):
        _require_text(optimization, field, errors, f"{prefix}.optimization")
    _validate_seeds(optimization.get("seeds"), f"{prefix}.optimization.seeds", errors)
    _validate_artifact_role(
        optimization.get("artifact"),
        f"{prefix}.optimization.artifact",
        "optimization",
        errors,
    )
    _validate_budget(
        optimization.get("budget"),
        f"{prefix}.optimization.budget",
        limit_key="proposal_limit",
        completed_key="proposals_completed",
        errors=errors,
    )
    optimization_source = _validate_artifact(
        optimization.get("artifact"),
        f"{prefix}.optimization.artifact",
        evidence_root,
        artifacts,
        errors,
        parse_json=True,
    )
    _validate_source_identity(
        optimization_source,
        optimization.get("artifact"),
        expected_schema=_OPTIMIZER_SELECTION_SCHEMA,
        experiment_id=experiment_id,
        round_number=number,
        source_revision=data.get("source_revision"),
        prefix=f"{prefix}.optimization.artifact",
        errors=errors,
    )
    if isinstance(optimization_source, dict):
        if optimization_source.get("selected_planner_id") != planner.get("planner_id"):
            errors.append(
                f"{prefix}.optimization artifact selected planner does not match round planner"
            )
        if not _same_text_identity(
            optimization_source.get("selected_config_identity_sha256"),
            planner.get("config_identity_sha256"),
        ):
            errors.append(
                f"{prefix}.optimization artifact selected config does not match round planner"
            )

    falsification = data.get("falsification")
    if not isinstance(falsification, dict):
        errors.append(f"{prefix}.falsification must be an object")
        falsification = {}
    for field in (
        "method",
        "objective",
        "target_failure_predicate",
        "search_space_id",
        "stop_reason",
    ):
        _require_text(falsification, field, errors, f"{prefix}.falsification")
    if not _is_allowed(falsification.get("stop_reason"), _STOP_REASONS):
        errors.append(f"{prefix}.falsification.stop_reason is unsupported")
    _validate_seeds(falsification.get("seeds"), f"{prefix}.falsification.seeds", errors)
    _validate_artifact_role(
        falsification.get("artifact"),
        f"{prefix}.falsification.artifact",
        "falsification-search",
        errors,
    )
    _validate_budget(
        falsification.get("budget"),
        f"{prefix}.falsification.budget",
        limit_key="candidate_limit",
        completed_key="candidates_completed",
        errors=errors,
    )
    falsification_source = _validate_artifact(
        falsification.get("artifact"),
        f"{prefix}.falsification.artifact",
        evidence_root,
        artifacts,
        errors,
        parse_json=True,
    )
    _validate_source_identity(
        falsification_source,
        falsification.get("artifact"),
        expected_schema=_FALSIFICATION_SOURCE_SCHEMA,
        experiment_id=experiment_id,
        round_number=number,
        source_revision=data.get("source_revision"),
        prefix=f"{prefix}.falsification.artifact",
        errors=errors,
    )
    if isinstance(falsification_source, dict):
        if falsification_source.get("target_planner_id") != planner.get("planner_id"):
            errors.append(
                f"{prefix}.falsification artifact target planner does not match round planner"
            )
        if not _same_text_identity(
            falsification_source.get("target_config_identity_sha256"),
            planner.get("config_identity_sha256"),
        ):
            errors.append(
                f"{prefix}.falsification artifact target config does not match round planner"
            )
    candidates = falsification.get("candidates")
    if not isinstance(candidates, list):
        errors.append(f"{prefix}.falsification.candidates must be an array")
        candidates = []
    seen_candidates: set[str] = set()
    admitted_by_case: dict[str, dict[str, Any]] = {}
    for idx, candidate in enumerate(candidates):
        candidate_prefix = f"{prefix}.falsification.candidates[{idx}]"
        if not isinstance(candidate, dict):
            errors.append(f"{candidate_prefix} must be an object")
            continue
        candidate_id = candidate.get("candidate_id")
        _require_text(candidate, "candidate_id", errors, candidate_prefix)
        if isinstance(candidate_id, str):
            if candidate_id in seen_candidates:
                errors.append(f"{candidate_prefix}.candidate_id is duplicated")
            seen_candidates.add(candidate_id)
        if not _is_allowed(candidate.get("evaluation_status"), _CANDIDATE_STATUSES):
            errors.append(f"{candidate_prefix}.evaluation_status is unsupported")
        if not _is_allowed(candidate.get("admissibility_verdict"), _ADMISSIBILITY_VERDICTS):
            errors.append(f"{candidate_prefix}.admissibility_verdict is unsupported")
        if candidate.get("target_failure_observed") not in (True, False, None):
            errors.append(f"{candidate_prefix}.target_failure_observed must be boolean or null")
        if not _is_allowed(
            candidate.get("replay_status"),
            {"verified", "unavailable", "mismatch", "not_attempted", "unknown"},
        ):
            errors.append(f"{candidate_prefix}.replay_status is unsupported")
        if not _is_allowed(candidate.get("corpus_disposition"), _CORPUS_DISPOSITIONS):
            errors.append(f"{candidate_prefix}.corpus_disposition is unsupported")
        case_id = candidate.get("case_id")
        corpus_disposition = candidate.get("corpus_disposition")
        if corpus_disposition == "admitted":
            if not isinstance(case_id, str) or not case_id:
                errors.append(f"{candidate_prefix} admitted record requires case_id")
            elif case_id in admitted_by_case:
                errors.append(f"{candidate_prefix} duplicates an admitted case_id")
            else:
                admitted_by_case[case_id] = candidate
            if candidate.get("evaluation_status") != "complete":
                errors.append(f"{candidate_prefix} admitted record requires complete evaluation")
            if candidate.get("target_failure_observed") is not True:
                errors.append(
                    f"{candidate_prefix} admitted record requires observed target failure"
                )
            if candidate.get("replay_status") != "verified":
                errors.append(f"{candidate_prefix} admitted record requires verified replay")
            if _is_allowed(
                candidate.get("admissibility_verdict"),
                {"structurally_invalid", "geometric_or_kinodynamic_impossibility"},
            ):
                errors.append(f"{candidate_prefix} excluded scenario cannot be admitted")
        elif case_id is not None and (not isinstance(case_id, str) or not case_id):
            errors.append(f"{candidate_prefix}.case_id must be non-empty text or null")
        elif corpus_disposition == "duplicate" and not isinstance(case_id, str):
            errors.append(f"{candidate_prefix} duplicate record requires case_id")
        if corpus_disposition in {"admitted", "duplicate"}:
            _require_text(candidate, "scenario_id", errors, candidate_prefix)
            _require_sha(
                candidate.get("scenario_artifact_sha256"),
                f"{candidate_prefix}.scenario_artifact_sha256",
                _SHA256,
                errors,
            )
        admissibility_artifact = candidate.get("admissibility_evidence_artifact")
        if corpus_disposition in {"admitted", "duplicate"}:
            _validate_artifact_role(
                admissibility_artifact,
                f"{candidate_prefix}.admissibility_evidence_artifact",
                "admissibility-evidence",
                errors,
            )
            admissibility_source = _validate_artifact(
                admissibility_artifact,
                f"{candidate_prefix}.admissibility_evidence_artifact",
                evidence_root,
                artifacts,
                errors,
                parse_json=True,
            )
            _validate_admissibility_evidence_source(
                admissibility_source,
                admissibility_artifact,
                context=_AdmissibilityContext(
                    case_id=case_id,
                    verdict=candidate.get("admissibility_verdict"),
                    scenario_id=candidate.get("scenario_id"),
                    scenario_artifact_sha256=candidate.get("scenario_artifact_sha256"),
                    target_failure_observed=candidate.get("target_failure_observed"),
                    source_revision=data.get("source_revision"),
                    planner=planner,
                ),
                prefix=f"{candidate_prefix}.admissibility_evidence_artifact",
                errors=errors,
            )
        elif admissibility_artifact is not None:
            _validate_artifact_role(
                admissibility_artifact,
                f"{candidate_prefix}.admissibility_evidence_artifact",
                "admissibility-evidence",
                errors,
            )
            admissibility_source = _validate_artifact(
                admissibility_artifact,
                f"{candidate_prefix}.admissibility_evidence_artifact",
                evidence_root,
                artifacts,
                errors,
                parse_json=True,
            )
            _validate_admissibility_evidence_source(
                admissibility_source,
                admissibility_artifact,
                context=_AdmissibilityContext(
                    case_id=case_id,
                    verdict=candidate.get("admissibility_verdict"),
                    scenario_id=candidate.get("scenario_id"),
                    scenario_artifact_sha256=candidate.get("scenario_artifact_sha256"),
                    target_failure_observed=candidate.get("target_failure_observed"),
                    source_revision=data.get("source_revision"),
                    planner=planner,
                ),
                prefix=f"{candidate_prefix}.admissibility_evidence_artifact",
                errors=errors,
            )
        replay_artifact = candidate.get("replay_artifact")
        if candidate.get("replay_status") == "verified" or replay_artifact is not None:
            _validate_artifact_role(
                replay_artifact, f"{candidate_prefix}.replay_artifact", "replay", errors
            )
            _validate_artifact(
                replay_artifact,
                f"{candidate_prefix}.replay_artifact",
                evidence_root,
                artifacts,
                errors,
            )
    budget = falsification.get("budget")
    if isinstance(budget, dict) and isinstance(budget.get("candidates_completed"), int):
        if budget["candidates_completed"] != len(candidates):
            errors.append(
                f"{prefix}.falsification candidate records ({len(candidates)}) do not match "
                f"completed evaluations ({budget['candidates_completed']})"
            )
    if isinstance(falsification_source, dict) and falsification_source.get(
        "candidate_records"
    ) != _source_projection(candidates, _SEARCH_CANDIDATE_SOURCE_FIELDS):
        errors.append(
            f"{prefix}.falsification candidates do not match the checksummed search source"
        )

    evaluation_sets = data.get("evaluation_sets")
    if not isinstance(evaluation_sets, dict):
        errors.append(f"{prefix}.evaluation_sets must be an object")
        evaluation_sets = {}
    for set_name in _EVALUATION_SETS:
        evaluation_context = _EvaluationSetContext(
            experiment_id=experiment_id,
            round_number=number,
            set_name=set_name,
            planner=planner,
        )
        _validate_evaluation_set(
            evaluation_sets.get(set_name),
            f"{prefix}.evaluation_sets.{set_name}",
            f"{set_name}-evaluation",
            evaluation_context,
            evidence_root,
            artifacts,
            errors,
        )

    observations = data.get("case_observations")
    if not isinstance(observations, list):
        errors.append(f"{prefix}.case_observations must be an array")
        observations = []
    seen_cases: set[str] = set()
    for idx, observation in enumerate(observations):
        obs_prefix = f"{prefix}.case_observations[{idx}]"
        if not isinstance(observation, dict):
            errors.append(f"{obs_prefix} must be an object")
            continue
        _require_text(observation, "case_id", errors, obs_prefix)
        _require_text(observation, "scenario_id", errors, obs_prefix)
        _require_sha(
            observation.get("scenario_artifact_sha256"),
            f"{obs_prefix}.scenario_artifact_sha256",
            _SHA256,
            errors,
        )
        case_id = observation.get("case_id")
        if isinstance(case_id, str):
            if case_id in seen_cases:
                errors.append(f"{obs_prefix}.case_id is duplicated within this round")
            seen_cases.add(case_id)
        origin_round = observation.get("origin_round")
        if not isinstance(origin_round, int) or isinstance(origin_round, bool):
            errors.append(f"{obs_prefix}.origin_round must be an integer")
        origin_candidate_id = observation.get("origin_candidate_id")
        if origin_round == 0:
            if origin_candidate_id is not None:
                errors.append(f"{obs_prefix} historical origin must have null origin_candidate_id")
        else:
            _require_text(observation, "origin_candidate_id", errors, obs_prefix)
        if not _is_allowed(observation.get("planner_status"), _PLANNER_STATUSES):
            errors.append(f"{obs_prefix}.planner_status is unsupported")
        if not _is_allowed(observation.get("admissibility_verdict"), _ADMISSIBILITY_VERDICTS):
            errors.append(f"{obs_prefix}.admissibility_verdict is unsupported")
        if not _is_allowed(observation.get("evidence_status"), _EVIDENCE_STATUSES):
            errors.append(f"{obs_prefix}.evidence_status is unsupported")
        if (
            observation.get("evidence_status") != "complete"
            and observation.get("planner_status") != "unknown"
        ):
            errors.append(f"{obs_prefix} incomplete evidence must retain planner_status=unknown")
        replay_status = observation.get("replay_status")
        if not _is_allowed(
            replay_status,
            {"verified", "unavailable", "mismatch", "not_attempted", "unknown"},
        ):
            errors.append(f"{obs_prefix}.replay_status is unsupported")
        expected_roles = {
            "origin_search_artifact": "falsification-search",
            "corpus_artifact": "corpus",
        }
        for key, expected_role in expected_roles.items():
            _validate_artifact_role(
                observation.get(key), f"{obs_prefix}.{key}", expected_role, errors
            )
            source = _validate_artifact(
                observation.get(key),
                f"{obs_prefix}.{key}",
                evidence_root,
                artifacts,
                errors,
                parse_json=key == "corpus_artifact",
            )
            if key == "corpus_artifact":
                _validate_source_identity(
                    source,
                    observation.get(key),
                    expected_schema=_CORPUS_CASE_STATUS_SCHEMA,
                    experiment_id=experiment_id,
                    round_number=number,
                    source_revision=data.get("source_revision"),
                    prefix=f"{obs_prefix}.corpus_artifact",
                    errors=errors,
                )
                _validate_case_status_source(source, observation, planner, obs_prefix, errors)
        replay_artifact = observation.get("replay_artifact")
        if replay_status == "verified" or replay_artifact is not None:
            _validate_artifact_role(
                replay_artifact, f"{obs_prefix}.replay_artifact", "replay", errors
            )
            _validate_artifact(
                replay_artifact,
                f"{obs_prefix}.replay_artifact",
                evidence_root,
                artifacts,
                errors,
            )
        admissibility_evidence = observation.get("admissibility_evidence_artifact")
        if admissibility_evidence is not None:
            _validate_artifact_role(
                admissibility_evidence,
                f"{obs_prefix}.admissibility_evidence_artifact",
                "admissibility-evidence",
                errors,
            )
            admissibility_source = _validate_artifact(
                admissibility_evidence,
                f"{obs_prefix}.admissibility_evidence_artifact",
                evidence_root,
                artifacts,
                errors,
                parse_json=True,
            )
            _validate_admissibility_evidence_source(
                admissibility_source,
                admissibility_evidence,
                context=_AdmissibilityContext(
                    case_id=observation.get("case_id"),
                    verdict=observation.get("admissibility_verdict"),
                    scenario_id=observation.get("scenario_id"),
                    scenario_artifact_sha256=observation.get("scenario_artifact_sha256"),
                    target_failure_observed=None,
                    source_revision=data.get("source_revision"),
                    planner=planner,
                ),
                prefix=f"{obs_prefix}.admissibility_evidence_artifact",
                errors=errors,
            )
        else:
            errors.append(
                f"{obs_prefix}.admissibility_evidence_artifact is required for every case observation"
            )


def _validate_evaluation_set(  # noqa: C901, PLR0912, PLR0915 - report all independent row-evidence defects.
    value: Any,
    prefix: str,
    expected_artifact_role: str,
    context: _EvaluationSetContext,
    evidence_root: Path,
    artifacts: dict[str, dict[str, str]],
    errors: list[str],
) -> None:
    experiment_id = context.experiment_id
    round_number = context.round_number
    planner = context.planner
    if not isinstance(value, dict):
        errors.append(f"{prefix} must be an object")
        return
    expected = value.get("expected_episode_count")
    if not isinstance(expected, int) or isinstance(expected, bool) or expected < 0:
        errors.append(f"{prefix}.expected_episode_count must be a non-negative integer")
    _validate_artifact_role(
        value.get("artifact"), f"{prefix}.artifact", expected_artifact_role, errors
    )
    source = _validate_artifact(
        value.get("artifact"),
        f"{prefix}.artifact",
        evidence_root,
        artifacts,
        errors,
        parse_json=True,
    )
    _validate_source_identity(
        source,
        value.get("artifact"),
        expected_schema=_EVALUATION_SOURCE_SCHEMA,
        experiment_id=experiment_id,
        round_number=round_number,
        source_revision=planner.get("source_revision"),
        prefix=f"{prefix}.artifact",
        errors=errors,
    )
    expected_ids = value.get("expected_episode_ids")
    if (
        not isinstance(expected_ids, list)
        or (
            isinstance(expected, int)
            and not isinstance(expected, bool)
            and expected > 0
            and not expected_ids
        )
        or any(not isinstance(item, str) or not item for item in expected_ids)
        or len(expected_ids) != len(set(expected_ids))
    ):
        errors.append(
            f"{prefix} identity accounting unknown: expected_episode_ids must be an array "
            "of unique stable IDs"
        )
        expected_ids = []
    expected_identities = value.get("expected_episode_identities")
    identity_by_record_id: dict[str, tuple[str, int]] = {}
    if not isinstance(expected_identities, list):
        errors.append(
            f"{prefix}.expected_episode_identities must be an array bound to expected IDs"
        )
        expected_identities = []
    for idx, identity in enumerate(expected_identities):
        identity_prefix = f"{prefix}.expected_episode_identities[{idx}]"
        if not isinstance(identity, dict):
            errors.append(f"{identity_prefix} must be an object")
            continue
        for field in ("record_id", "scenario_id"):
            _require_text(identity, field, errors, identity_prefix)
        record_id = identity.get("record_id")
        scenario_id = identity.get("scenario_id")
        scenario_seed = identity.get("scenario_seed")
        if not isinstance(scenario_seed, int) or isinstance(scenario_seed, bool):
            errors.append(f"{identity_prefix}.scenario_seed must be an integer")
        if isinstance(record_id, str) and isinstance(scenario_id, str):
            if record_id in identity_by_record_id:
                errors.append(f"{identity_prefix}.record_id is duplicated")
            else:
                identity_by_record_id[record_id] = (scenario_id, scenario_seed)
    identity_ids = [item.get("record_id") for item in expected_identities if isinstance(item, dict)]
    if identity_ids != expected_ids:
        errors.append(
            f"{prefix}.expected_episode_identities must match expected_episode_ids in order"
        )
    if len(expected_identities) != len(expected_ids):
        errors.append(
            f"{prefix}.expected_episode_identities count does not match expected_episode_ids"
        )
    if (
        isinstance(expected, int)
        and not isinstance(expected, bool)
        and expected != len(expected_ids)
    ):
        errors.append(f"{prefix}.expected_episode_count does not match expected_episode_ids")
    episodes = value.get("episodes")
    if not isinstance(episodes, list):
        errors.append(f"{prefix}.episodes must be an array")
        return
    seen: set[str] = set()
    for idx, episode in enumerate(episodes):
        row_prefix = f"{prefix}.episodes[{idx}]"
        if not isinstance(episode, dict):
            errors.append(f"{row_prefix} must be an object")
            continue
        _require_text(episode, "record_id", errors, row_prefix)
        record_id = episode.get("record_id")
        if isinstance(record_id, str):
            if record_id in seen:
                errors.append(f"{row_prefix}.record_id is duplicated")
            seen.add(record_id)
        _require_text(episode, "scenario_id", errors, row_prefix)
        scenario_id = episode.get("scenario_id")
        scenario_seed = episode.get("scenario_seed")
        if not isinstance(scenario_seed, int) or isinstance(scenario_seed, bool):
            errors.append(f"{row_prefix}.scenario_seed must be an integer")
        if isinstance(record_id, str) and isinstance(scenario_id, str):
            if identity_by_record_id.get(record_id) != (scenario_id, scenario_seed):
                errors.append(
                    f"{row_prefix} scenario/seed identity does not match the expected manifest"
                )
        if not _is_allowed(episode.get("evidence_status"), _EVIDENCE_STATUSES):
            errors.append(f"{row_prefix}.evidence_status is unsupported")
        if not _is_allowed(episode.get("execution_mode"), _EXECUTION_MODES):
            errors.append(f"{row_prefix}.execution_mode is unsupported")
        if not _is_allowed(episode.get("readiness_status"), _READINESS_STATUSES):
            errors.append(f"{row_prefix}.readiness_status is unsupported")
        if not _is_allowed(episode.get("availability_status"), _AVAILABILITY_STATUSES):
            errors.append(f"{row_prefix}.availability_status is unsupported")
        if episode.get("eligible") not in (True, False, None):
            errors.append(f"{row_prefix}.eligible must be boolean or null")
        for field in ("success", "collision"):
            if episode.get(field) not in (True, False, None):
                errors.append(f"{row_prefix}.{field} must be boolean or null")
        for field in ("minimum_clearance", "ped_force_q95"):
            metric_value = episode.get(field)
            if metric_value is not None and (
                not isinstance(metric_value, (int, float))
                or isinstance(metric_value, bool)
                or not math.isfinite(metric_value)
            ):
                errors.append(f"{row_prefix}.{field} must be a finite number or null")
    _validate_evaluation_source_rows(source, value, episodes, context, prefix, errors)


def _validate_case_status_source(
    source: Any,
    observation: dict[str, Any],
    planner: dict[str, Any],
    prefix: str,
    errors: list[str],
) -> None:
    """Bind each corpus status artifact to its case and round observation."""
    if not isinstance(source, dict):
        return
    expected_fields = (
        "case_id",
        "origin_round",
        "origin_candidate_id",
        "scenario_id",
        "scenario_artifact_sha256",
        "planner_status",
        "admissibility_verdict",
        "replay_status",
        "evidence_status",
    )
    mismatched = [field for field in expected_fields if source.get(field) != observation.get(field)]
    if mismatched:
        errors.append(
            f"{prefix}.corpus_artifact case/status identity does not match the observation: "
            f"{mismatched}"
        )
    if not _same_artifact_reference(
        source.get("admissibility_evidence_artifact"),
        observation.get("admissibility_evidence_artifact"),
    ):
        errors.append(
            f"{prefix}.corpus_artifact admissibility evidence does not match the observation"
        )
    if source.get("planner_id") != planner.get("planner_id"):
        errors.append(f"{prefix}.corpus_artifact planner does not match the round planner")
    if not _same_text_identity(
        source.get("config_identity_sha256"), planner.get("config_identity_sha256")
    ):
        errors.append(f"{prefix}.corpus_artifact config does not match the round planner")


def _validate_evaluation_source_rows(
    source: Any,
    value: dict[str, Any],
    episodes: list[Any],
    context: _EvaluationSetContext,
    prefix: str,
    errors: list[str],
) -> None:
    if not isinstance(source, dict):
        errors.append(f"{prefix} identity accounting unknown: checksummed source is unavailable")
        return
    if source.get("evaluation_set") != context.set_name:
        errors.append(f"{prefix}.artifact evaluation_set does not match its report section")
    if source.get("planner_id") != context.planner.get("planner_id"):
        errors.append(f"{prefix}.artifact planner does not match the round planner")
    if not _same_text_identity(
        source.get("config_identity_sha256"),
        context.planner.get("config_identity_sha256"),
    ):
        errors.append(f"{prefix}.artifact config does not match the round planner")
    expected_ids = value.get("expected_episode_ids")
    if source.get("expected_episode_ids") != expected_ids:
        errors.append(
            f"{prefix} identity accounting unknown: expected episode IDs do not match "
            "the checksummed evaluation source"
        )
    expected_identities = value.get("expected_episode_identities")
    if source.get("expected_episode_identities") != expected_identities:
        errors.append(
            f"{prefix} scenario/seed manifest does not match the checksummed evaluation source"
        )
    source_rows = source.get("episodes")
    row_projection = _source_projection(episodes, _EVALUATION_ROW_SOURCE_FIELDS)
    if source_rows != row_projection:
        errors.append(f"{prefix} rows do not match the checksummed evaluation source")
    observed_ids = [episode.get("record_id") for episode in episodes if isinstance(episode, dict)]
    expected_id_set = (
        {item for item in expected_ids if isinstance(item, str)}
        if isinstance(expected_ids, list)
        else set()
    )
    unexpected_ids = [
        record_id
        for record_id in observed_ids
        if isinstance(record_id, str) and record_id not in expected_id_set
    ]
    if unexpected_ids:
        errors.append(
            f"{prefix} identity accounting unknown: unexpected evaluation row IDs "
            f"{unexpected_ids} are absent from expected_episode_ids"
        )


def _evaluation_identity_sets(
    round_data: dict[str, Any], round_number: Any, errors: list[str]
) -> tuple[dict[str, set[tuple[str, int]]], dict[str, set[str]]]:
    """Collect valid scenario/seed and record IDs while reporting per-set duplicates."""
    pair_sets: dict[str, set[tuple[str, int]]] = {}
    record_sets: dict[str, set[str]] = {}
    evaluation_sets = round_data.get("evaluation_sets")
    if not isinstance(evaluation_sets, dict):
        return pair_sets, record_sets
    for set_name in _EVALUATION_SETS:
        evaluation = evaluation_sets.get(set_name)
        identities = (
            evaluation.get("expected_episode_identities") if isinstance(evaluation, dict) else None
        )
        if not isinstance(identities, list):
            continue
        pairs: list[tuple[str, int]] = []
        record_ids: set[str] = set()
        for identity in identities:
            if not isinstance(identity, dict):
                continue
            record_id = identity.get("record_id")
            scenario_id = identity.get("scenario_id")
            scenario_seed = identity.get("scenario_seed")
            if (
                isinstance(record_id, str)
                and isinstance(scenario_id, str)
                and isinstance(scenario_seed, int)
                and not isinstance(scenario_seed, bool)
            ):
                pairs.append((scenario_id, scenario_seed))
                record_ids.add(record_id)
        if len(pairs) != len(set(pairs)):
            errors.append(
                f"round {round_number} {set_name} scenario/seed identities must be unique"
            )
        pair_sets[set_name] = set(pairs)
        record_sets[set_name] = record_ids
    return pair_sets, record_sets


def _validate_evaluation_cohorts(rounds: list[Any], errors: list[str]) -> None:
    """Reject split leakage and require a stable fixed evaluation cohort."""
    previous_fixed: set[tuple[str, int]] | None = None
    historical_held_out: set[tuple[str, int]] = set()
    historical_held_out_record_ids: set[str] = set()
    for round_data in rounds:
        if not isinstance(round_data, dict):
            continue
        round_number = round_data.get("round_number")
        pair_sets, record_sets = _evaluation_identity_sets(round_data, round_number, errors)
        for exposed_set in ("fixed", "regression"):
            shared_pairs = pair_sets.get("held_out", set()) & pair_sets.get(exposed_set, set())
            shared_record_ids = record_sets.get("held_out", set()) & record_sets.get(
                exposed_set, set()
            )
            shared_historical_pairs = historical_held_out & pair_sets.get(exposed_set, set())
            shared_historical_ids = historical_held_out_record_ids & record_sets.get(
                exposed_set, set()
            )
            if (
                shared_pairs
                or shared_record_ids
                or shared_historical_pairs
                or shared_historical_ids
            ):
                errors.append(
                    f"round {round_number} held_out overlaps {exposed_set} evaluation identities"
                )
        historical_held_out.update(pair_sets.get("held_out", set()))
        historical_held_out_record_ids.update(record_sets.get("held_out", set()))
        fixed = pair_sets.get("fixed")
        if fixed is not None:
            if previous_fixed is not None and fixed != previous_fixed:
                errors.append(
                    f"round {round_number} fixed evaluation scenario/seed cohort changed; "
                    "fixed-cohort comparisons require identical identities across rounds"
                )
            previous_fixed = fixed


def _scenario_seed_manifest_digest(evaluation: dict[str, Any]) -> str:
    identities = evaluation.get("expected_episode_identities", [])
    pairs = sorted(
        {
            (item["scenario_id"], item["scenario_seed"])
            for item in identities
            if isinstance(item, dict)
            and isinstance(item.get("scenario_id"), str)
            and isinstance(item.get("scenario_seed"), int)
            and not isinstance(item.get("scenario_seed"), bool)
        }
    )
    payload = json.dumps(pairs, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _scenario_seed_rows_digest(rows: list[dict[str, Any]]) -> str:
    pairs = sorted((row["scenario_id"], row["scenario_seed"]) for row in rows)
    payload = json.dumps(pairs, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _add_cohort_comparability(
    evaluation_summaries: dict[str, dict[str, Any]],
    round_data: dict[str, Any],
    prior_round_reports: list[dict[str, Any]],
) -> None:
    previous_sets = prior_round_reports[-1]["evaluation_sets"] if prior_round_reports else None
    for set_name, summary in evaluation_summaries.items():
        digest = _scenario_seed_manifest_digest(round_data["evaluation_sets"][set_name])
        summary["scenario_seed_manifest_sha256"] = digest
        if previous_sets is None:
            summary["cohort_comparable_to_previous_round"] = None
            summary["cohort_comparison_status"] = "baseline"
            summary["success_rate_comparable_to_previous_round"] = None
            summary["success_rate_comparison_status"] = "baseline"
            continue
        comparable = digest == previous_sets[set_name]["scenario_seed_manifest_sha256"]
        summary["cohort_comparable_to_previous_round"] = comparable
        summary["cohort_comparison_status"] = (
            "comparable" if comparable else "non_comparable_cohort"
        )
        success_sample_comparable = (
            summary["success_rate_sample_sha256"]
            == previous_sets[set_name]["success_rate_sample_sha256"]
        )
        success_comparable = comparable and success_sample_comparable
        summary["success_rate_comparable_to_previous_round"] = success_comparable
        if not comparable:
            summary["success_rate_comparison_status"] = "non_comparable_cohort"
        elif not success_sample_comparable:
            summary["success_rate_comparison_status"] = "non_comparable_success_sample"
        else:
            summary["success_rate_comparison_status"] = "comparable"


def _validate_budget(
    value: Any,
    prefix: str,
    *,
    limit_key: str,
    completed_key: str,
    errors: list[str],
) -> None:
    if not isinstance(value, dict):
        errors.append(f"{prefix} must be an object")
        return
    keys = (limit_key, completed_key, "simulator_invocations")
    values: dict[str, int] = {}
    for key in keys:
        item = value.get(key)
        if not isinstance(item, int) or isinstance(item, bool) or item < 0:
            errors.append(f"{prefix}.{key} must be a non-negative integer")
        else:
            values[key] = item
    if values.get(completed_key, 0) > values.get(limit_key, 0):
        errors.append(f"{prefix}.{completed_key} exceeds {limit_key}")
    if values.get(limit_key) == 0:
        errors.append(f"{prefix}.{limit_key} must be positive")


def _validate_seeds(value: Any, prefix: str, errors: list[str]) -> None:
    if not isinstance(value, list) or not value:
        errors.append(f"{prefix} must be a non-empty array of integer seeds")
        return
    if any(not isinstance(seed, int) or isinstance(seed, bool) for seed in value):
        errors.append(f"{prefix} must contain only integer seeds")
    elif len(value) != len(set(value)):
        errors.append(f"{prefix} must not contain duplicate seeds")


def _source_projection(rows: Any, fields: tuple[str, ...]) -> list[dict[str, Any]] | None:
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        return None
    return [{field: row.get(field) for field in fields} for row in rows]


def _same_text_identity(left: Any, right: Any) -> bool:
    return isinstance(left, str) and isinstance(right, str) and left.lower() == right.lower()


def _is_allowed(value: Any, choices: set[str]) -> bool:
    """Check a categorical value without hashing malformed JSON values."""
    return isinstance(value, str) and value in choices


def _validate_source_identity(
    payload: Any,
    artifact: Any,
    *,
    expected_schema: str,
    experiment_id: Any,
    round_number: int,
    source_revision: Any,
    prefix: str,
    errors: list[str],
) -> None:
    if not isinstance(payload, dict):
        return
    if payload.get("schema_version") != expected_schema:
        errors.append(f"{prefix} content schema_version must be {expected_schema!r}")
    if not isinstance(artifact, dict) or artifact.get("schema_version") != expected_schema:
        errors.append(f"{prefix}.schema_version must be {expected_schema!r}")
    if payload.get("experiment_id") != experiment_id:
        errors.append(f"{prefix} experiment_id does not match the evidence bundle")
    if payload.get("round_number") != round_number:
        errors.append(f"{prefix} round_number does not match its enclosing round")
    if not _same_text_identity(payload.get("source_revision"), source_revision):
        errors.append(f"{prefix} source_revision does not match the enclosing round")
    if not isinstance(artifact, dict) or not _same_text_identity(
        artifact.get("source_revision"), source_revision
    ):
        errors.append(f"{prefix}.source_revision does not match the enclosing round")


def _validate_artifact(  # noqa: C901, PLR0912 - preserve path/hash/schema findings in one validation pass.
    value: Any,
    prefix: str,
    evidence_root: Path,
    artifacts: dict[str, dict[str, str]],
    errors: list[str],
    *,
    parse_json: bool = False,
) -> Any | None:
    if not isinstance(value, dict):
        errors.append(f"{prefix} must be an artifact reference object")
        return None
    relative_path = value.get("path")
    if not isinstance(relative_path, str) or not relative_path.strip():
        errors.append(f"{prefix}.path is missing")
        return None
    posix_path = PurePosixPath(relative_path)
    if posix_path.is_absolute() or ".." in posix_path.parts:
        errors.append(f"{prefix}.path must stay relative to the evidence bundle")
        return None
    actual_path = (evidence_root / Path(*posix_path.parts)).resolve()
    try:
        actual_path.relative_to(evidence_root)
    except ValueError:
        errors.append(f"{prefix}.path resolves outside the evidence bundle")
        return None
    if not actual_path.is_file():
        errors.append(f"{prefix}.path does not exist: {relative_path}")
        return None
    expected_sha = value.get("sha256")
    _require_sha(expected_sha, f"{prefix}.sha256", _SHA256, errors)
    source_revision = value.get("source_revision")
    _require_sha(source_revision, f"{prefix}.source_revision", _GIT_SHA, errors)
    _require_text(value, "role", errors, prefix)
    _require_text(value, "schema_version", errors, prefix)
    if not isinstance(expected_sha, str) or not _SHA256.fullmatch(expected_sha):
        return None
    payload: Any | None = None
    content: bytes | None = None
    try:
        if parse_json:
            content = actual_path.read_bytes()
            actual_sha = hashlib.sha256(content).hexdigest()
        else:
            actual_sha = _sha256_file(actual_path)
    except OSError as exc:
        errors.append(f"{prefix} could not read artifact: {exc}")
        return None
    if actual_sha.lower() != expected_sha.lower():
        errors.append(f"{prefix}.sha256 does not match {relative_path}")
        return None
    if parse_json and content is not None:
        try:
            payload = json.loads(content)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            errors.append(f"{prefix} could not parse artifact JSON: {exc}")
            return None
        if not isinstance(payload, dict):
            errors.append(f"{prefix} JSON artifact must contain an object")
            payload = None
        elif value.get("schema_version") != payload.get("schema_version"):
            errors.append(f"{prefix}.schema_version does not match the artifact content")
    artifact = {
        "path": relative_path,
        "sha256": actual_sha,
        "source_revision": source_revision,
        "role": value.get("role"),
        "schema_version": value.get("schema_version"),
    }
    previous = artifacts.get(relative_path)
    if previous is not None and previous != artifact:
        errors.append(f"artifact {relative_path!r} has inconsistent provenance declarations")
    else:
        artifacts[relative_path] = artifact
    return payload


def _validate_artifact_role(value: Any, prefix: str, expected_role: str, errors: list[str]) -> None:
    """Require references used as a specific evidence type to declare that role."""
    if isinstance(value, dict) and value.get("role") != expected_role:
        errors.append(f"{prefix}.role must be {expected_role!r}")


def _validate_admissibility_evidence_source(  # noqa: C901 - report independent binding defects.
    source: Any,
    artifact: Any,
    context: _AdmissibilityContext,
    prefix: str,
    errors: list[str],
) -> None:
    """Bind the #9651 verdict artifact to its case and require support for confirmation."""
    if not isinstance(source, dict):
        return
    _validate_admissibility_record_shape(source, artifact, prefix, errors)
    if source.get("case_id") != context.case_id:
        errors.append(f"{prefix}.case_id does not match the candidate or observation")
    if source.get("verdict") != context.verdict:
        errors.append(f"{prefix}.verdict does not match the candidate or observation")
    if source.get("scenario_id") != context.scenario_id:
        errors.append(f"{prefix}.scenario_id does not match the candidate or observation")
    if not isinstance(artifact, dict) or not _same_text_identity(
        artifact.get("source_revision"), context.source_revision
    ):
        errors.append(f"{prefix}.source_revision does not match the enclosing round")
    expected_disposition = (
        "reject"
        if context.verdict in {"structurally_invalid", "geometric_or_kinodynamic_impossibility"}
        else "retain"
    )
    if source.get("search_disposition") != expected_disposition:
        errors.append(f"{prefix}.search_disposition conflicts with its verdict")
    evidence = source.get("evidence")
    if not isinstance(evidence, dict):
        errors.append(f"{prefix}.evidence must be an object")
        return
    scenario_identity = evidence.get("scenario_artifact_identity")
    _validate_scenario_artifact_identity(
        scenario_identity,
        context.scenario_artifact_sha256,
        prefix,
        errors,
    )
    if context.verdict in _CONFIRMED_FEASIBILITY_VERDICTS:
        _validate_confirmed_admissibility_support(
            source,
            evidence,
            context,
            prefix,
            errors,
        )
        target_outcome = source.get("target_planner_outcome")
        if isinstance(context.target_failure_observed, bool) and target_outcome in {
            "route_completed",
            "route_incomplete",
        }:
            if (target_outcome == "route_incomplete") is not context.target_failure_observed:
                errors.append(
                    f"{prefix}.target_planner_outcome does not match the candidate failure record"
                )


def _validate_admissibility_record_shape(  # noqa: C901 - report independent schema-field defects.
    source: dict[str, Any], artifact: Any, prefix: str, errors: list[str]
) -> None:
    """Check required #9651 record fields before binding its evidence to a case."""
    required = {
        "schema_version",
        "case_id",
        "scenario_id",
        "verdict",
        "target_planner_outcome",
        "search_disposition",
        "reason_codes",
        "assumptions",
        "evidence",
    }
    missing = sorted(required - source.keys())
    extra = sorted(source.keys() - required)
    if missing:
        errors.append(f"{prefix} is missing required fields: {missing}")
    if extra:
        errors.append(f"{prefix} contains unsupported fields: {extra}")
    if not isinstance(artifact, dict) or artifact.get("schema_version") != (
        _ADMISSIBILITY_EVIDENCE_SCHEMA
    ):
        errors.append(f"{prefix}.schema_version must be {_ADMISSIBILITY_EVIDENCE_SCHEMA!r}")
    if source.get("schema_version") != _ADMISSIBILITY_EVIDENCE_SCHEMA:
        errors.append(f"{prefix} must contain a {_ADMISSIBILITY_EVIDENCE_SCHEMA} record")
    scenario_id = source.get("scenario_id")
    if not isinstance(scenario_id, str) or not scenario_id.strip():
        errors.append(f"{prefix}.scenario_id must be non-empty text to bind the case")
    if not _is_allowed(source.get("verdict"), _ADMISSIBILITY_VERDICTS):
        errors.append(f"{prefix}.verdict is unsupported")
    if source.get("target_planner_outcome") not in {
        "not_evaluated",
        "unavailable",
        "route_completed",
        "route_incomplete",
    }:
        errors.append(f"{prefix}.target_planner_outcome is unsupported")
    reason_codes = source.get("reason_codes")
    if (
        not isinstance(reason_codes, list)
        or not reason_codes
        or any(not isinstance(item, str) or not item.strip() for item in reason_codes)
    ):
        errors.append(f"{prefix}.reason_codes must contain non-empty text")
    elif len(reason_codes) != len(set(reason_codes)):
        errors.append(f"{prefix}.reason_codes must be unique")
    if not isinstance(source.get("assumptions"), dict):
        errors.append(f"{prefix}.assumptions must be an object")
    if not isinstance(source.get("evidence"), dict):
        errors.append(f"{prefix}.evidence must be an object")


def _validate_scenario_artifact_identity(
    identity: Any,
    expected_sha256: Any,
    prefix: str,
    errors: list[str],
) -> None:
    """Bind the producer's captured scenario bytes to the search and corpus identity."""
    if not isinstance(identity, dict):
        errors.append(f"{prefix}.evidence.scenario_artifact_identity must be an object")
        return
    actual_sha256 = identity.get("sha256")
    if (
        identity.get("status") != "available"
        or not isinstance(actual_sha256, str)
        or _SHA256.fullmatch(actual_sha256) is None
        or not isinstance(expected_sha256, str)
        or actual_sha256.lower() != expected_sha256.lower()
    ):
        errors.append(
            f"{prefix}.scenario_artifact_identity does not match the candidate or corpus scenario"
        )
    requires_effective = identity.get("requires_effective_input_binding")
    if not isinstance(requires_effective, bool):
        errors.append(
            f"{prefix}.scenario_artifact_identity.requires_effective_input_binding must be boolean"
        )
    elif requires_effective:
        effective_sha256 = identity.get("effective_input_sha256")
        if not isinstance(effective_sha256, str) or _SHA256.fullmatch(effective_sha256) is None:
            errors.append(f"{prefix}.scenario_artifact_identity.effective_input_sha256 is required")


def _validate_confirmed_admissibility_support(  # noqa: C901 - keep verdict proof diagnostics explicit.
    source: dict[str, Any],
    evidence: dict[str, Any],
    context: _AdmissibilityContext,
    prefix: str,
    errors: list[str],
) -> None:
    """Require producer-shaped runs bound to this round before confirming feasibility."""
    verdict = source.get("verdict")
    reason_codes = source.get("reason_codes")
    runs = {name: evidence.get(f"{name}_execution") for name in ("reference", "target", "replay")}
    valid_runs: dict[str, bool] = {}
    for role, run in runs.items():
        if run is None:
            valid_runs[role] = False
            continue
        valid_runs[role] = _validate_admissibility_execution(
            run,
            role=role,
            context=context,
            scenario_identity=evidence.get("scenario_artifact_identity"),
            prefix=f"{prefix}.evidence.{role}_execution",
            errors=errors,
        )
    target_run = runs["target"]
    target_observation_bound = _validate_target_planner_observation(
        evidence.get("target_planner_observation"),
        context=context,
        scenario_identity=evidence.get("scenario_artifact_identity"),
        prefix=f"{prefix}.evidence.target_planner_observation",
        errors=errors,
    )
    target_bound = valid_runs["target"] or target_observation_bound
    target_observation = evidence.get("target_planner_observation")
    if target_observation_bound and isinstance(target_observation, dict):
        expected_outcome = (
            "route_completed"
            if target_observation.get("route_complete") is True
            else "route_incomplete"
        )
        if source.get("target_planner_outcome") != expected_outcome:
            errors.append(f"{prefix}.target_planner_outcome conflicts with its target observation")
    if target_bound and isinstance(target_run, dict):
        expected_outcome = (
            "route_completed" if target_run.get("route_complete") else "route_incomplete"
        )
        if source.get("target_planner_outcome") != expected_outcome:
            errors.append(f"{prefix}.target_planner_outcome conflicts with its target execution")

    if verdict == "empirically_feasible":
        completed_execution = any(
            valid_runs[role]
            and isinstance(runs[role], dict)
            and runs[role].get("route_complete") is True
            for role in ("reference", "target", "replay")
        )
        oracle_support = _validate_admissibility_oracle_support(
            source,
            evidence,
            context.scenario_id,
            context.scenario_artifact_sha256,
            prefix,
            errors,
        )
        if not target_bound:
            errors.append(f"{prefix} lacks a complete target-planner execution bound to this round")
        if not completed_execution and not oracle_support:
            errors.append(f"{prefix} does not contain evidence supporting empirical feasibility")
    elif verdict == "planner_specific_failure":
        reference = runs["reference"]
        target = runs["target"]
        replay = runs["replay"]
        matched_failure = (
            valid_runs["reference"]
            and valid_runs["target"]
            and valid_runs["replay"]
            and isinstance(reference, dict)
            and isinstance(target, dict)
            and isinstance(replay, dict)
            and _admissibility_runs_share_case(reference, target)
            and _admissibility_replay_matches_target(replay, target)
            and reference.get("route_complete") is True
            and target.get("route_complete") is False
            and replay.get("route_complete") is False
            and reference.get("planner_id") != target.get("planner_id")
            and "matched_reference_target_failure_reproduced_by_replay" in (reason_codes or [])
        )
        if not matched_failure:
            errors.append(f"{prefix} does not contain matched planner-specific failure evidence")


def _admissibility_runs_share_case(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Match the producer's reference/target scenario, environment, and seed identity."""
    fields = (
        "case_id",
        "scenario_id",
        "scenario_variant",
        "scenario_sha256",
        "robot_model_sha256",
        "simulator_config_sha256",
        "environment_sha256",
        "source_commit",
        "seed",
        "horizon_steps",
    )
    return all(left.get(field) == right.get(field) for field in fields)


def _admissibility_replay_matches_target(replay: dict[str, Any], target: dict[str, Any]) -> bool:
    """Require the deterministic replay to match all target case and episode bindings."""
    return (
        _admissibility_runs_share_case(replay, target)
        and replay.get("planner_id") == target.get("planner_id")
        and replay.get("planner_config_sha256") == target.get("planner_config_sha256")
        and replay.get("planner_checkpoint_sha256") == target.get("planner_checkpoint_sha256")
        and replay.get("episode_id") == target.get("episode_id")
        and replay.get("source_episodes_jsonl_sha256") == target.get("source_episodes_jsonl_sha256")
        and replay.get("determinism_check_status") == "pass"
        and replay.get("resimulated") is True
    )


def _validate_admissibility_execution(  # noqa: C901, PLR0912 - report producer-field defects.
    run: Any,
    role: str,
    context: _AdmissibilityContext,
    scenario_identity: Any,
    prefix: str,
    errors: list[str],
) -> bool:
    """Validate the normalized execution record emitted by scenario_admissibility.v1."""
    start_errors = len(errors)
    if not isinstance(run, dict):
        errors.append(f"{prefix} must be an object")
        return False
    required = {
        "case_id",
        "scenario_id",
        "scenario_variant",
        "planner_id",
        "run_status",
        "fallback_or_degraded",
        "route_complete",
        "seed",
        "horizon_steps",
        "scenario_sha256",
        "robot_model_sha256",
        "simulator_config_sha256",
        "planner_config_sha256",
        "planner_checkpoint_sha256",
        "environment_sha256",
        "source_commit",
        "evidence_ref",
    }
    if role in {"target", "replay"}:
        required.update({"episode_id", "source_episodes_jsonl_sha256"})
    missing = sorted(required - run.keys())
    if missing:
        errors.append(f"{prefix} is missing producer fields: {missing}")
    if run.get("case_id") != context.case_id or run.get("scenario_id") != context.scenario_id:
        errors.append(f"{prefix} case/scenario identity does not match its case record")
    if run.get("scenario_variant") != "original" or run.get("run_status") != "ok":
        errors.append(f"{prefix} is not a completed original-scenario run")
    if run.get("fallback_or_degraded") is not False:
        errors.append(f"{prefix}.fallback_or_degraded must be false")
    if not isinstance(run.get("planner_id"), str) or not run["planner_id"].strip():
        errors.append(f"{prefix}.planner_id must be non-empty text")
    if not isinstance(run.get("evidence_ref"), str) or not run["evidence_ref"].strip():
        errors.append(f"{prefix}.evidence_ref must be non-empty text")
    if not isinstance(run.get("source_commit"), str) or not _same_text_identity(
        run.get("source_commit"), context.source_revision
    ):
        errors.append(f"{prefix}.source_commit does not match the enclosing round")
    for field in (
        "scenario_sha256",
        "robot_model_sha256",
        "simulator_config_sha256",
        "planner_config_sha256",
        "environment_sha256",
    ):
        value = run.get(field)
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            errors.append(f"{prefix}.{field} must be a SHA-256 digest")
    if not _same_text_identity(run.get("scenario_sha256"), context.scenario_artifact_sha256):
        errors.append(f"{prefix}.scenario_sha256 does not match the case scenario artifact")
    checkpoint = run.get("planner_checkpoint_sha256")
    if not (
        isinstance(checkpoint, str)
        and (
            _SHA256.fullmatch(checkpoint) is not None
            or (
                checkpoint == "not_applicable"
                and run.get("planner_id") in {"goal", "orca", "social_force"}
            )
        )
    ):
        errors.append(f"{prefix}.planner_checkpoint_sha256 is unsupported")
    seed, horizon = run.get("seed"), run.get("horizon_steps")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        errors.append(f"{prefix}.seed must be a non-negative integer")
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon <= 0:
        errors.append(f"{prefix}.horizon_steps must be a positive integer")
    if not isinstance(run.get("route_complete"), bool):
        errors.append(f"{prefix}.route_complete must be boolean")
    if role in {"target", "replay"}:
        if not isinstance(run.get("episode_id"), str) or not run["episode_id"].strip():
            errors.append(f"{prefix}.episode_id must be non-empty text")
        if (
            not isinstance(run.get("source_episodes_jsonl_sha256"), str)
            or _SHA256.fullmatch(run["source_episodes_jsonl_sha256"]) is None
        ):
            errors.append(f"{prefix}.source_episodes_jsonl_sha256 must be a SHA-256 digest")
    identity = scenario_identity if isinstance(scenario_identity, dict) else {}
    if identity.get("requires_effective_input_binding") is True:
        expected_effective = identity.get("effective_input_sha256")
        actual_effective = run.get("effective_input_sha256")
        if (
            not isinstance(expected_effective, str)
            or _SHA256.fullmatch(expected_effective) is None
            or not isinstance(actual_effective, str)
            or _SHA256.fullmatch(actual_effective) is None
            or not _same_text_identity(actual_effective, expected_effective)
            or run.get("effective_input_identity_stable") is not True
        ):
            errors.append(f"{prefix} effective scenario input identity is missing or mismatched")
    if role in {"target", "replay"} and (
        run.get("planner_id") != context.planner.get("planner_id")
        or not _same_text_identity(
            run.get("planner_config_sha256"), context.planner.get("config_identity_sha256")
        )
    ):
        errors.append(f"{prefix} planner/config does not match the enclosing round")
    if role == "replay" and (
        run.get("determinism_check_status") != "pass" or run.get("resimulated") is not True
    ):
        errors.append(f"{prefix} must pass replay determinism and simulator resimulation")
    return len(errors) == start_errors


def _validate_target_planner_observation(  # noqa: C901 - collect case/planner binding defects.
    observation: Any,
    context: _AdmissibilityContext,
    scenario_identity: Any,
    prefix: str,
    errors: list[str],
) -> bool:
    """Bind a post-search target observation to this case and planner configuration."""
    if not isinstance(observation, dict) or observation.get("status") != "available":
        return False
    start_errors = len(errors)
    if observation.get("scenario_id") != context.scenario_id:
        errors.append(f"{prefix}.scenario_id does not match the case")
    if observation.get("planner_id") != context.planner.get("planner_id"):
        errors.append(f"{prefix}.planner_id does not match the round planner")
    if not _same_text_identity(observation.get("source_commit"), context.source_revision):
        errors.append(f"{prefix}.source_commit does not match the enclosing round")
    if not _same_text_identity(
        observation.get("planner_config_hash"), context.planner.get("config_identity_sha256")
    ):
        errors.append(f"{prefix}.planner_config_hash does not match the round planner")
    runtime_identity = observation.get("runtime_input_identity")
    producer_identity = scenario_identity if isinstance(scenario_identity, dict) else {}
    if (
        not isinstance(runtime_identity, dict)
        or runtime_identity.get("status") != "available"
        or not _same_text_identity(
            runtime_identity.get("source_artifact_sha256"), context.scenario_artifact_sha256
        )
    ):
        errors.append(f"{prefix}.runtime_input_identity does not match the case scenario")
    elif producer_identity.get("requires_effective_input_binding") is True and not (
        _same_text_identity(
            runtime_identity.get("effective_input_sha256"),
            producer_identity.get("effective_input_sha256"),
        )
    ):
        errors.append(f"{prefix}.runtime_input_identity effective inputs do not match")
    if not isinstance(observation.get("episode_id"), str) or not observation["episode_id"].strip():
        errors.append(f"{prefix}.episode_id must be non-empty text")
    seed = observation.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        errors.append(f"{prefix}.seed must be a non-negative integer")
    if not isinstance(observation.get("route_complete"), bool):
        errors.append(f"{prefix}.route_complete must be boolean")
    if observation.get("reason_code") is not None:
        errors.append(f"{prefix}.reason_code must be null for an available outcome")
    expected_outcome = (
        "route_completed" if observation.get("route_complete") is True else "route_incomplete"
    )
    return len(errors) == start_errors and expected_outcome in {
        "route_completed",
        "route_incomplete",
    }


def _validate_admissibility_oracle_support(
    source: dict[str, Any],
    evidence: dict[str, Any],
    scenario_id: Any,
    scenario_artifact_sha256: Any,
    prefix: str,
    errors: list[str],
) -> bool:
    """Accept only a producer-bound actor-free feasible rollout, never a bare oracle label."""
    assumptions = source.get("assumptions")
    oracle_assumptions = (
        assumptions.get("feasibility_oracle") if isinstance(assumptions, dict) else None
    )
    raw = evidence.get("feasibility_evidence")
    if (
        not isinstance(oracle_assumptions, dict)
        or oracle_assumptions.get("empirical_scope")
        != "named_actor_free_rollout_of_original_static_case"
        or not isinstance(raw, dict)
        or raw.get("schema_version") not in _ORACLE_EVIDENCE_SCHEMAS
    ):
        return False
    selected: dict[str, Any] | None = None
    report = raw
    if raw.get("schema_version") == "issue_5574_feasibility_oracle_report.v1":
        cells = raw.get("cells")
        matches = (
            [
                cell
                for cell in cells
                if isinstance(cell, dict) and cell.get("scenario_id") == scenario_id
            ]
            if isinstance(cells, list)
            else []
        )
        if len(matches) != 1:
            return False
        selected = matches[0]
    elif raw.get("schema_version") == "envelope_sensitivity_axis.v1":
        selected = raw
    else:
        selected = raw
    nominal = selected.get("nominal_verdict", selected)
    if not isinstance(nominal, dict):
        return False
    source_records = (report, selected, nominal)
    for record in source_records:
        if record.get("source_artifact_sha256") != scenario_artifact_sha256:
            return False
        if record.get("source_artifact_identity_stable") is not True:
            return False
    completion = nominal.get("completion")
    geometric = nominal.get("geometric")
    assumptions_match = (
        oracle_assumptions.get("scenario_id") == scenario_id
        and oracle_assumptions.get("source_artifact_sha256") == scenario_artifact_sha256
    )
    completion_steps = (
        completion.get("min_completion_steps") if isinstance(completion, dict) else None
    )
    horizon = completion.get("horizon_steps") if isinstance(completion, dict) else None
    termination = completion.get("termination_reason") if isinstance(completion, dict) else None
    completion_valid = (
        isinstance(completion, dict)
        and completion.get("route_completion_feasible") is True
        and completion.get("status") == "passed"
        and completion.get("blocker") is None
        and completion.get("fallback_or_degraded") is False
        and completion.get("observed_route_completion_feasible") is True
        and completion.get("fallback_marker") is None
        and completion.get("rollout_blocker") is None
        and isinstance(completion_steps, int)
        and not isinstance(completion_steps, bool)
        and completion_steps > 0
        and isinstance(horizon, int)
        and not isinstance(horizon, bool)
        and horizon >= completion_steps
        and completion.get("completion_horizon_margin_steps") == horizon - completion_steps
        and termination
        in {
            "success",
            "goal_reached",
            "route_complete",
            "completed",
            "route_follow_reached_destination",
        }
    )
    rollout_seed = selected.get("rollout_seed", report.get("rollout_seed"))
    rollout_algo = selected.get("rollout_algo", report.get("rollout_algo"))
    manifest = selected.get("scenario_manifest", report.get("scenario_manifest"))
    return (
        assumptions_match
        and nominal.get("schema_version") == "scenario_feasibility_oracle.v1"
        and nominal.get("scenario_id") == scenario_id
        and nominal.get("status") == "feasible"
        and nominal.get("feasible") is True
        and nominal.get("claim_boundary") == _DIAGNOSTIC_CLAIM_BOUNDARY
        and isinstance(geometric, dict)
        and geometric.get("route_geometrically_feasible") is True
        and completion_valid
        and isinstance(rollout_algo, str)
        and bool(rollout_algo.strip())
        and isinstance(rollout_seed, int)
        and not isinstance(rollout_seed, bool)
        and rollout_seed >= 0
        and isinstance(manifest, str)
        and bool(manifest.strip())
        and report.get("claim_boundary", nominal.get("claim_boundary"))
        == _DIAGNOSTIC_CLAIM_BOUNDARY
    )


def _summarize_evaluation_set(data: dict[str, Any]) -> dict[str, Any]:
    eligible_rows = [
        row
        for row in data["episodes"]
        if row["evidence_status"] == "complete"
        and row["execution_mode"] in {"native", "adapter", "mixed"}
        and row["readiness_status"] in _BENCHMARK_READY_STATUSES
        and row["availability_status"] == "available"
        and row["eligible"] is True
    ]
    success_rows = [row for row in eligible_rows if isinstance(row["success"], bool)]
    collision_rows = [row for row in eligible_rows if isinstance(row["collision"], bool)]
    statuses = Counter(row["evidence_status"] for row in data["episodes"])
    execution_modes = Counter(row["execution_mode"] for row in data["episodes"])
    readiness_statuses = Counter(row["readiness_status"] for row in data["episodes"])
    availability_statuses = Counter(row["availability_status"] for row in data["episodes"])
    success_count = sum(row["success"] for row in success_rows)
    collision_count = sum(row["collision"] for row in collision_rows)
    reported_count = len(data["episodes"])
    expected_count = data["expected_episode_count"]
    expected_ids = data["expected_episode_ids"]
    reported_ids = {row["record_id"] for row in data["episodes"]}
    missing_ids = [record_id for record_id in expected_ids if record_id not in reported_ids]
    expected_id_set = set(expected_ids)
    unexpected_ids = [
        row["record_id"] for row in data["episodes"] if row["record_id"] not in expected_id_set
    ]
    return {
        "expected_episode_count": expected_count,
        "expected_episode_ids_sha256": hashlib.sha256(
            json.dumps(expected_ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "reported_episode_count": reported_count,
        "missing_record_count": len(missing_ids),
        "missing_record_ids": missing_ids,
        "unexpected_record_count": len(unexpected_ids),
        "unexpected_record_ids": unexpected_ids,
        "eligible_episode_count": len(eligible_rows),
        "success_denominator": len(success_rows),
        "success_rate_sample_sha256": _scenario_seed_rows_digest(success_rows),
        "collision_denominator": len(collision_rows),
        "excluded_episode_count": reported_count - len(eligible_rows),
        "accounting_complete": not missing_ids and not unexpected_ids,
        "identity_accounting_status": "verified",
        "evidence_status_counts": dict(sorted(statuses.items())),
        "execution_mode_counts": dict(sorted(execution_modes.items())),
        "readiness_status_counts": dict(sorted(readiness_statuses.items())),
        "availability_status_counts": dict(sorted(availability_statuses.items())),
        "eligibility_conflict_count": sum(
            row["eligible"] is True
            and (
                row["evidence_status"] != "complete"
                or row["execution_mode"] not in {"native", "adapter", "mixed"}
                or row["readiness_status"] not in _BENCHMARK_READY_STATUSES
                or row["availability_status"] != "available"
            )
            for row in data["episodes"]
        ),
        "excluded_record_ids": [
            row["record_id"] for row in data["episodes"] if row not in eligible_rows
        ],
        "missing_success_record_ids": [
            row["record_id"] for row in eligible_rows if not isinstance(row["success"], bool)
        ],
        "missing_collision_record_ids": [
            row["record_id"] for row in eligible_rows if not isinstance(row["collision"], bool)
        ],
        "successes": success_count,
        "collisions": collision_count,
        "success_rate": success_count / len(success_rows) if success_rows else None,
        "collision_rate": collision_count / len(collision_rows) if collision_rows else None,
        "minimum_clearance_min": min(
            (
                row["minimum_clearance"]
                for row in eligible_rows
                if row.get("minimum_clearance") is not None
            ),
            default=None,
        ),
        "ped_force_q95_mean": _mean(
            row["ped_force_q95"] for row in eligible_rows if row.get("ped_force_q95") is not None
        ),
        "artifact": data["artifact"],
    }


def _is_verified_counterexample(candidate: dict[str, Any]) -> bool:
    return (
        candidate.get("corpus_disposition") == "admitted"
        and isinstance(candidate.get("case_id"), str)
        and candidate.get("evaluation_status") == "complete"
        and candidate.get("target_failure_observed") is True
        and candidate.get("replay_status") == "verified"
        and candidate.get("admissibility_verdict")
        in {"empirically_feasible", "planner_specific_failure"}
    )


def _classify_search_candidates(
    candidates: list[dict[str, Any]],
    round_number: int,
    discovered_counterexamples: dict[str, int],
    admitted_unknown_cases: dict[str, int],
    current_case_verdicts: dict[str, str],
    confirmed_before_round: set[str],
) -> tuple[list[str], list[str], list[str], list[str]]:
    verified: list[str] = []
    repeated: list[str] = []
    admitted: list[str] = []
    unknown: list[str] = []
    for candidate in candidates:
        case_id = candidate.get("case_id")
        if candidate["corpus_disposition"] == "admitted" and isinstance(case_id, str):
            admitted.append(case_id)
            verdict = candidate["admissibility_verdict"]
            current_case_verdicts.setdefault(case_id, verdict)
            if verdict == "admissible_feasibility_unknown":
                unknown.append(case_id)
                admitted_unknown_cases.setdefault(case_id, round_number)
            if _is_verified_counterexample(candidate):
                if case_id in discovered_counterexamples:
                    repeated.append(case_id)
                else:
                    verified.append(case_id)
                    discovered_counterexamples[case_id] = round_number
        elif _is_repeated_verified_counterexample(candidate) and case_id in confirmed_before_round:
            repeated.append(candidate["case_id"])
    return verified, repeated, admitted, unknown


def _validate_case_dispositions(rounds: list[Any], errors: list[str]) -> None:
    """Keep stable case IDs unique while allowing explicit references to known cases."""
    known_case_ids: set[str] = set()
    for round_data in rounds:
        if not isinstance(round_data, dict):
            continue
        observations = round_data.get("case_observations", [])
        if isinstance(observations, list):
            known_case_ids.update(
                item["case_id"]
                for item in observations
                if isinstance(item, dict)
                and item.get("origin_round") == 0
                and isinstance(item.get("case_id"), str)
            )

    for round_data in rounds:
        if not isinstance(round_data, dict):
            continue
        current_round = round_data.get("round_number")
        falsification = round_data.get("falsification")
        candidates = falsification.get("candidates", []) if isinstance(falsification, dict) else []
        if isinstance(candidates, list):
            _validate_round_case_dispositions(current_round, candidates, known_case_ids, errors)


def _validate_round_case_dispositions(
    current_round: Any,
    candidates: list[Any],
    known_case_ids: set[str],
    errors: list[str],
) -> None:
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        case_id = candidate.get("case_id")
        disposition = candidate.get("corpus_disposition")
        if disposition == "duplicate" and (
            not isinstance(case_id, str) or case_id not in known_case_ids
        ):
            errors.append(
                f"round {current_round} duplicate candidate must refer to a previously "
                "known corpus case"
            )
        elif disposition == "admitted" and isinstance(case_id, str):
            if case_id in known_case_ids:
                errors.append(
                    f"round {current_round} re-admits known case {case_id!r}; use "
                    "corpus_disposition=duplicate"
                )
            else:
                known_case_ids.add(case_id)


def _is_repeated_verified_counterexample(candidate: dict[str, Any]) -> bool:
    """Return whether a search candidate replayed a counterexample already in the corpus."""
    return (
        candidate.get("corpus_disposition") == "duplicate"
        and isinstance(candidate.get("case_id"), str)
        and candidate.get("evaluation_status") == "complete"
        and candidate.get("target_failure_observed") is True
        and candidate.get("replay_status") == "verified"
        and candidate.get("admissibility_verdict")
        in {"empirically_feasible", "planner_specific_failure"}
    )


def _current_counterexample_status(
    known_case_ids: set[str], observations: list[dict[str, Any]]
) -> dict[str, int]:
    current = {item["case_id"]: item for item in observations if item["case_id"] in known_case_ids}
    return {
        "solved": sum(item["planner_status"] == "solved" for item in current.values()),
        "unsolved": sum(item["planner_status"] == "unsolved" for item in current.values()),
        "mixed": sum(item["planner_status"] == "mixed" for item in current.values()),
        "unknown": sum(item["planner_status"] == "unknown" for item in current.values()),
        "not_observed_this_round": len(known_case_ids - set(current)),
    }


def _format_rate(summary: dict[str, Any]) -> str:
    rate = summary["success_rate"]
    if rate is None:
        return "unknown (0 success outcomes recorded)"
    return f"{rate:.3f} ({summary['successes']}/{summary['success_denominator']})"


def _artifact_link(path: Any, evidence_root_relative_to_report: str) -> str:
    if not isinstance(path, str) or not path:
        return "library input"
    relative_path = PurePosixPath(evidence_root_relative_to_report, path).as_posix()
    target = quote(relative_path, safe="/._-")
    return f"[`{path}`](<{target}>)"


def _format_collision_rate(summary: dict[str, Any]) -> str:
    rate = summary["collision_rate"]
    if rate is None:
        return "unknown (0 collision outcomes recorded)"
    return f"{rate:.3f} ({summary['collisions']}/{summary['collision_denominator']})"


def _require_text(data: dict[str, Any], key: str, errors: list[str], prefix: str) -> None:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{prefix}.{key} must be non-empty text")


def _require_sha(value: Any, label: str, pattern: re.Pattern[str], errors: list[str]) -> None:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        errors.append(f"{label} must be a full hexadecimal revision/digest")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mean(values: Any) -> float | None:
    items = list(values)
    return sum(items) / len(items) if items else None


__all__ = [
    "INPUT_SCHEMA_VERSION",
    "REPORT_SCHEMA_VERSION",
    "FrontierReportError",
    "build_frontier_report",
    "render_frontier_markdown",
    "write_frontier_figure",
    "write_frontier_report",
]
