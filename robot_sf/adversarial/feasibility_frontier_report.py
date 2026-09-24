"""Build a provenance-bound, fixture-first report of planner/search rounds.

The input is a persisted ``adversarial-coevolution-evidence.v1`` bundle. This
module summarizes the supplied records; it does not run planners, infer dynamic
feasibility, or turn a finite no-discovery result into a claim of absence.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote

from robot_sf.benchmark.figures.export import save_publication_figure
from robot_sf.benchmark.figures.style import planner_color, publication_style

INPUT_SCHEMA_VERSION = "adversarial-coevolution-evidence.v1"
REPORT_SCHEMA_VERSION = "adversarial-feasibility-frontier.v1"
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
_EXECUTION_MODES = {"normal", "fallback", "degraded", "unknown"}
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
_CORPUS_DISPOSITIONS = {"admitted", "pending", "rejected", "not_applicable", "unknown"}
_EVALUATION_SETS = ("fixed", "regression", "held_out")
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
    admitted_unknown_cases: dict[str, int] = {}

    for round_data in evidence["rounds"]:
        number = round_data["round_number"]
        eval_summary = {
            name: _summarize_evaluation_set(round_data["evaluation_sets"][name])
            for name in _EVALUATION_SETS
        }
        candidates = round_data["falsification"]["candidates"]
        verified_case_ids: list[str] = []
        admitted_case_ids: list[str] = []
        unknown_case_ids: list[str] = []
        for candidate in candidates:
            case_id = candidate.get("case_id")
            if candidate["corpus_disposition"] == "admitted" and isinstance(case_id, str):
                admitted_case_ids.append(case_id)
                if candidate["admissibility_verdict"] == "admissible_feasibility_unknown":
                    unknown_case_ids.append(case_id)
                if _is_verified_counterexample(candidate):
                    verified_case_ids.append(case_id)
                    discovered_counterexamples.setdefault(case_id, number)
                if candidate["admissibility_verdict"] == "admissible_feasibility_unknown":
                    admitted_unknown_cases.setdefault(case_id, number)

        observations = round_data["case_observations"]
        state_counts = Counter(item["planner_status"] for item in observations)
        verdict_counts = Counter(item["admissibility_verdict"] for item in observations)
        replay_counts = Counter(item["replay_status"] for item in observations)

        current_verified_status = _current_counterexample_status(
            set(discovered_counterexamples), observations
        )
        falsification = round_data["falsification"]
        candidate_status_counts = Counter(item["evaluation_status"] for item in candidates)
        candidate_verdict_counts = Counter(item["admissibility_verdict"] for item in candidates)
        no_verified_counterexample = not verified_case_ids
        no_discovery_statement = None
        if no_verified_counterexample:
            no_discovery_statement = (
                "No new replay-verified planner counterexample was recorded by this round's "
                f"finite search budget ({falsification['budget']['candidate_limit']} candidates, "
                f"{falsification['budget']['simulator_invocations']} simulator invocations). "
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
                    "verified_counterexamples_cumulative": len(discovered_counterexamples),
                    "admitted_unknown_feasibility_cases_cumulative": len(admitted_unknown_cases),
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
            "Summary of declared finite-budget evidence only; synthetic fixtures are "
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
    lines = [
        f"# Empirical feasibility frontier: {report['experiment_id']}",
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
        "Success and collision rates use only complete, normal-mode, explicitly eligible "
        "episode rows with the relevant outcome recorded. Excluded and missing rows remain "
        "visible in the accounting below.",
        "",
        "| Round | Planner/config | Optimization method and budget | Search method and budget | Held-out success | Regression success | Verified counterexamples (cumulative) |",
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
                cumulative=item["case_frontier"]["verified_counterexamples_cumulative"],
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
                f"{summary['expected_episode_count']} expected rows eligible; "
                f"{_format_rate(summary)} success; {_format_collision_rate(summary)} collision; "
                f"evidence statuses `{summary['evidence_status_counts']}`, execution modes "
                f"`{summary['execution_mode_counts']}`; excluded records "
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


def write_frontier_figure(report: dict[str, Any], output_base: Path) -> list[Path]:
    """Write the performance/discovery figure with the repository figure helpers."""
    try:
        plt = importlib.import_module("matplotlib.pyplot")
    except ImportError as exc:  # pragma: no cover - environment-specific dependency guard
        raise FrontierReportError("matplotlib is required to render the frontier figure") from exc

    rounds = report["rounds"]
    x_values = [item["round_number"] for item in rounds]
    with publication_style(size="double"):
        figure, axes = plt.subplots(2, 1, figsize=(7.0, 7.1), constrained_layout=True)
        performance_axis, cases_axis = axes
        for set_name, label in (
            ("fixed", "Fixed"),
            ("regression", "Regression"),
            ("held_out", "Held out"),
        ):
            values = [item["evaluation_sets"][set_name]["success_rate"] for item in rounds]
            performance_axis.plot(
                x_values,
                [float("nan") if value is None else value for value in values],
                marker="o",
                linewidth=1.8,
                color=planner_color(f"frontier_{set_name}"),
                label=label,
            )
            for x, item in zip(x_values, rounds, strict=True):
                summary = item["evaluation_sets"][set_name]
                value = summary["success_rate"]
                if value is not None:
                    performance_axis.annotate(
                        f"{summary['successes']}/{summary['eligible_episode_count']}",
                        (x, value),
                        xytext=(0, 5),
                        textcoords="offset points",
                        ha="center",
                        fontsize=7,
                    )
        performance_axis.set_ylim(-0.05, 1.05)
        performance_axis.set_xticks(x_values)
        performance_axis.set_xlabel("Round")
        performance_axis.set_ylabel("Success fraction")
        performance_axis.set_title("Planner evaluation by case set")
        performance_axis.grid(axis="y", alpha=0.25)
        performance_axis.legend(frameon=False, ncol=3, loc="lower right")

        cumulative_known = [
            item["case_frontier"]["verified_counterexamples_cumulative"] for item in rounds
        ]
        cumulative_unknown = [
            item["case_frontier"]["admitted_unknown_feasibility_cases_cumulative"]
            for item in rounds
        ]
        solved = [
            item["case_frontier"]["verified_counterexample_status"]["solved"] for item in rounds
        ]
        cases_axis.plot(
            x_values,
            cumulative_known,
            marker="o",
            linewidth=1.8,
            color=planner_color("frontier_verified_counterexamples"),
            label="Verified failure cases discovered (cumulative)",
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
            cumulative_unknown,
            marker="^",
            linewidth=1.8,
            linestyle="--",
            color=planner_color("frontier_unknown_feasibility"),
            label="Admitted cases with unknown feasibility (cumulative)",
        )
        cases_axis.set_xticks(x_values)
        cases_axis.set_xlabel("Round")
        cases_axis.set_ylabel("Case count")
        cases_axis.set_title("Counterexample memory and current planner status")
        cases_axis.grid(axis="y", alpha=0.25)
        cases_axis.legend(frameon=False, loc="best", fontsize=8)
        figure.suptitle("Finite-budget planner–falsifier frontier")

        provenance = {
            "source_artifacts": [
                {"path": item["path"], "hash": item["sha256"]}
                for item in (
                    [*report["source_artifacts"], report["input_artifact"]]
                    if report.get("input_artifact")
                    else report["source_artifacts"]
                )
            ],
            "repo_commit": report["source_revision"],
            "generator_command": "build_adversarial_feasibility_frontier_report",
            "claim_boundary": report["claim_boundary"],
            "figure_formats": ["png", "pdf"],
        }
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


def _validate_evidence(  # noqa: C901, PLR0912 - collect independent provenance blockers in one pass.
    evidence: dict[str, Any], evidence_root: Path, artifacts: dict[str, dict[str, str]]
) -> None:
    errors: list[str] = []
    if evidence.get("schema_version") != INPUT_SCHEMA_VERSION:
        errors.append(f"schema_version must be {INPUT_SCHEMA_VERSION!r}")
    if evidence.get("evidence_kind") not in {
        "synthetic_fixture",
        "simulator_run",
        "historical_artifact",
    }:
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

    round_candidates: dict[int, dict[str, str | None]] = {}
    for round_data in rounds:
        if not isinstance(round_data, dict):
            errors.append("each round must be an object")
            continue
        number = round_data.get("round_number")
        if not isinstance(number, int) or isinstance(number, bool) or number < 1:
            errors.append("round_number must be a positive integer")
            continue
        _validate_round(round_data, number, evidence_root, artifacts, errors)
        falsification = round_data.get("falsification")
        candidates = falsification.get("candidates", []) if isinstance(falsification, dict) else []
        round_candidates[number] = {
            item.get("case_id"): item.get("candidate_id")
            for item in candidates
            if isinstance(item, dict)
            and item.get("corpus_disposition") == "admitted"
            and isinstance(item.get("case_id"), str)
        }

    for round_data in rounds:
        if not isinstance(round_data, dict) or not isinstance(round_data.get("round_number"), int):
            continue
        current_round = round_data["round_number"]
        for observation in round_data.get("case_observations", []):
            if not isinstance(observation, dict):
                continue
            origin_round = observation.get("origin_round")
            case_id = observation.get("case_id")
            candidate_id = observation.get("origin_candidate_id")
            if isinstance(origin_round, int) and origin_round > 0 and origin_round <= current_round:
                known_candidate_id = round_candidates.get(origin_round, {}).get(case_id)
                if known_candidate_id is None:
                    errors.append(
                        f"round {current_round} case {case_id!r} refers to no admitted discovery "
                        f"in origin round {origin_round}"
                    )
                elif candidate_id != known_candidate_id:
                    errors.append(
                        f"round {current_round} case {case_id!r} origin candidate does not match "
                        "its discovery record"
                    )
            elif origin_round != 0:
                errors.append(f"round {current_round} case {case_id!r} has invalid origin_round")
    if errors:
        raise FrontierReportError("invalid frontier evidence:\n- " + "\n- ".join(errors))


def _validate_round(  # noqa: C901, PLR0912, PLR0915 - retain every round-local blocker for diagnosis.
    data: dict[str, Any],
    number: int,
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
    _validate_budget(
        optimization.get("budget"),
        f"{prefix}.optimization.budget",
        limit_key="proposal_limit",
        completed_key="proposals_completed",
        errors=errors,
    )
    _validate_artifact(
        optimization.get("artifact"),
        f"{prefix}.optimization.artifact",
        evidence_root,
        artifacts,
        errors,
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
    if falsification.get("stop_reason") not in _STOP_REASONS:
        errors.append(f"{prefix}.falsification.stop_reason is unsupported")
    _validate_seeds(falsification.get("seeds"), f"{prefix}.falsification.seeds", errors)
    _validate_budget(
        falsification.get("budget"),
        f"{prefix}.falsification.budget",
        limit_key="candidate_limit",
        completed_key="candidates_completed",
        errors=errors,
    )
    _validate_artifact(
        falsification.get("artifact"),
        f"{prefix}.falsification.artifact",
        evidence_root,
        artifacts,
        errors,
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
        if candidate.get("evaluation_status") not in _CANDIDATE_STATUSES:
            errors.append(f"{candidate_prefix}.evaluation_status is unsupported")
        if candidate.get("admissibility_verdict") not in _ADMISSIBILITY_VERDICTS:
            errors.append(f"{candidate_prefix}.admissibility_verdict is unsupported")
        if candidate.get("target_failure_observed") not in (True, False, None):
            errors.append(f"{candidate_prefix}.target_failure_observed must be boolean or null")
        if candidate.get("replay_status") not in {
            "verified",
            "unavailable",
            "mismatch",
            "not_attempted",
            "unknown",
        }:
            errors.append(f"{candidate_prefix}.replay_status is unsupported")
        if candidate.get("corpus_disposition") not in _CORPUS_DISPOSITIONS:
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
            if candidate.get("admissibility_verdict") in {
                "structurally_invalid",
                "geometric_or_kinodynamic_impossibility",
            }:
                errors.append(f"{candidate_prefix} excluded scenario cannot be admitted")
        elif case_id is not None and (not isinstance(case_id, str) or not case_id):
            errors.append(f"{candidate_prefix}.case_id must be non-empty text or null")
        replay_artifact = candidate.get("replay_artifact")
        if candidate.get("replay_status") == "verified":
            _validate_artifact(
                replay_artifact,
                f"{candidate_prefix}.replay_artifact",
                evidence_root,
                artifacts,
                errors,
            )
        elif replay_artifact is not None:
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

    evaluation_sets = data.get("evaluation_sets")
    if not isinstance(evaluation_sets, dict):
        errors.append(f"{prefix}.evaluation_sets must be an object")
        evaluation_sets = {}
    for set_name in _EVALUATION_SETS:
        _validate_evaluation_set(
            evaluation_sets.get(set_name),
            f"{prefix}.evaluation_sets.{set_name}",
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
        if observation.get("planner_status") not in _PLANNER_STATUSES:
            errors.append(f"{obs_prefix}.planner_status is unsupported")
        if observation.get("admissibility_verdict") not in _ADMISSIBILITY_VERDICTS:
            errors.append(f"{obs_prefix}.admissibility_verdict is unsupported")
        if observation.get("evidence_status") not in _EVIDENCE_STATUSES:
            errors.append(f"{obs_prefix}.evidence_status is unsupported")
        if (
            observation.get("evidence_status") != "complete"
            and observation.get("planner_status") != "unknown"
        ):
            errors.append(f"{obs_prefix} incomplete evidence must retain planner_status=unknown")
        replay_status = observation.get("replay_status")
        if replay_status not in {"verified", "unavailable", "mismatch", "not_attempted", "unknown"}:
            errors.append(f"{obs_prefix}.replay_status is unsupported")
        for key in ("origin_search_artifact", "corpus_artifact"):
            _validate_artifact(
                observation.get(key), f"{obs_prefix}.{key}", evidence_root, artifacts, errors
            )
        replay_artifact = observation.get("replay_artifact")
        if replay_status == "verified" or replay_artifact is not None:
            _validate_artifact(
                replay_artifact,
                f"{obs_prefix}.replay_artifact",
                evidence_root,
                artifacts,
                errors,
            )


def _validate_evaluation_set(  # noqa: C901 - preserve independent row-level evidence findings.
    value: Any,
    prefix: str,
    evidence_root: Path,
    artifacts: dict[str, dict[str, str]],
    errors: list[str],
) -> None:
    if not isinstance(value, dict):
        errors.append(f"{prefix} must be an object")
        return
    expected = value.get("expected_episode_count")
    if not isinstance(expected, int) or isinstance(expected, bool) or expected < 0:
        errors.append(f"{prefix}.expected_episode_count must be a non-negative integer")
    _validate_artifact(
        value.get("artifact"), f"{prefix}.artifact", evidence_root, artifacts, errors
    )
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
        if episode.get("evidence_status") not in _EVIDENCE_STATUSES:
            errors.append(f"{row_prefix}.evidence_status is unsupported")
        if episode.get("execution_mode") not in _EXECUTION_MODES:
            errors.append(f"{row_prefix}.execution_mode is unsupported")
        if episode.get("eligible") not in (True, False, None):
            errors.append(f"{row_prefix}.eligible must be boolean or null")
        for field in ("success", "collision"):
            if episode.get(field) not in (True, False, None):
                errors.append(f"{row_prefix}.{field} must be boolean or null")
        for field in ("minimum_clearance", "ped_force_q95"):
            value = episode.get(field)
            if value is not None and (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
            ):
                errors.append(f"{row_prefix}.{field} must be a finite number or null")


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


def _validate_artifact(
    value: Any,
    prefix: str,
    evidence_root: Path,
    artifacts: dict[str, dict[str, str]],
    errors: list[str],
) -> None:
    if not isinstance(value, dict):
        errors.append(f"{prefix} must be an artifact reference object")
        return
    relative_path = value.get("path")
    if not isinstance(relative_path, str) or not relative_path.strip():
        errors.append(f"{prefix}.path is missing")
        return
    posix_path = PurePosixPath(relative_path)
    if posix_path.is_absolute() or ".." in posix_path.parts:
        errors.append(f"{prefix}.path must stay relative to the evidence bundle")
        return
    actual_path = (evidence_root / Path(*posix_path.parts)).resolve()
    try:
        actual_path.relative_to(evidence_root)
    except ValueError:
        errors.append(f"{prefix}.path resolves outside the evidence bundle")
        return
    if not actual_path.is_file():
        errors.append(f"{prefix}.path does not exist: {relative_path}")
        return
    expected_sha = value.get("sha256")
    _require_sha(expected_sha, f"{prefix}.sha256", _SHA256, errors)
    source_revision = value.get("source_revision")
    _require_sha(source_revision, f"{prefix}.source_revision", _GIT_SHA, errors)
    _require_text(value, "role", errors, prefix)
    _require_text(value, "schema_version", errors, prefix)
    if not isinstance(expected_sha, str) or not _SHA256.fullmatch(expected_sha):
        return
    actual_sha = _sha256_file(actual_path)
    if actual_sha.lower() != expected_sha.lower():
        errors.append(f"{prefix}.sha256 does not match {relative_path}")
        return
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


def _summarize_evaluation_set(data: dict[str, Any]) -> dict[str, Any]:
    eligible_rows = [
        row
        for row in data["episodes"]
        if row["evidence_status"] == "complete"
        and row["execution_mode"] == "normal"
        and row["eligible"] is True
        and isinstance(row["success"], bool)
        and isinstance(row["collision"], bool)
    ]
    statuses = Counter(row["evidence_status"] for row in data["episodes"])
    execution_modes = Counter(row["execution_mode"] for row in data["episodes"])
    success_count = sum(row["success"] for row in eligible_rows)
    collision_count = sum(row["collision"] for row in eligible_rows)
    denominator = len(eligible_rows)
    reported_count = len(data["episodes"])
    expected_count = data["expected_episode_count"]
    return {
        "expected_episode_count": expected_count,
        "reported_episode_count": reported_count,
        "missing_record_count": max(0, expected_count - reported_count),
        "unexpected_record_count": max(0, reported_count - expected_count),
        "eligible_episode_count": denominator,
        "excluded_episode_count": reported_count - denominator,
        "accounting_complete": reported_count == expected_count,
        "evidence_status_counts": dict(sorted(statuses.items())),
        "execution_mode_counts": dict(sorted(execution_modes.items())),
        "eligibility_conflict_count": sum(
            row["eligible"] is True and row["execution_mode"] != "normal"
            for row in data["episodes"]
        ),
        "excluded_record_ids": [
            row["record_id"] for row in data["episodes"] if row not in eligible_rows
        ],
        "successes": success_count,
        "collisions": collision_count,
        "success_rate": success_count / denominator if denominator else None,
        "collision_rate": collision_count / denominator if denominator else None,
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
        return "unknown (0 eligible episodes)"
    return f"{rate:.3f} ({summary['successes']}/{summary['eligible_episode_count']})"


def _artifact_link(path: Any, evidence_root_relative_to_report: str) -> str:
    if not isinstance(path, str) or not path:
        return "library input"
    relative_path = PurePosixPath(evidence_root_relative_to_report, path).as_posix()
    target = quote(relative_path, safe="/._-")
    return f"[`{path}`](<{target}>)"


def _format_collision_rate(summary: dict[str, Any]) -> str:
    rate = summary["collision_rate"]
    if rate is None:
        return "unknown (0 eligible episodes)"
    return f"{rate:.3f} ({summary['collisions']}/{summary['eligible_episode_count']})"


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
