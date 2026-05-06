#!/usr/bin/env python3
"""Summarize the registered policy-search planner portfolio."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import yaml

STAGE_ORDER = (
    "smoke",
    "nominal_sanity",
    "stress_slice",
    "full_matrix",
    "leader_collision_slice_h500",
    "full_matrix_h500",
    "robustness_extension",
)
STAGE_RANK = {stage: index for index, stage in enumerate(STAGE_ORDER)}
PROMOTION_SCALE_STAGES = {
    "full_matrix",
    "full_matrix_h500",
    "robustness_extension",
}


@dataclass(frozen=True)
class CandidateReport:
    """Metrics parsed from one tracked policy-search Markdown report."""

    candidate: str
    stage: str
    path: Path
    report_date: date | None
    decision: str | None
    algorithm: str | None
    scenario_matrix: str | None
    summary_json: str | None
    git_commit: str | None
    episodes: int | None
    success_rate: float | None
    collision_rate: float | None
    near_miss_rate: float | None
    mean_min_distance: float | None
    mean_avg_speed: float | None
    failure_counts: dict[str, int]
    family_runs: list[str]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    today = datetime.now(UTC).date().isoformat()
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("docs/context/policy_search/candidate_registry.yaml"),
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("docs/context/policy_search/reports"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path(f"docs/context/policy_search/portfolio_overview_{today}.md"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(f"docs/context/policy_search/portfolio_overview_{today}.json"),
    )
    parser.add_argument(
        "--list-candidates",
        action="store_true",
        help="Print implemented candidate names and exit.",
    )
    parser.add_argument(
        "--all-implemented",
        action="store_true",
        help="When listing candidates, ignore required_stages and print every implemented entry.",
    )
    parser.add_argument(
        "--stage",
        choices=STAGE_ORDER,
        help="When listing candidates, keep candidates whose required stages include this stage.",
    )
    return parser.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"Expected YAML mapping: {path}")
    return payload


def _parse_optional_float(raw: str) -> float | None:
    value = raw.strip().strip("`")
    if not value or value.lower() in {"n/a", "nan", "none"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _parse_optional_int(raw: str) -> int | None:
    value = raw.strip().strip("`")
    if not value or value.lower() in {"n/a", "nan", "none"}:
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def _section_after(lines: list[str], heading: str) -> list[str]:
    try:
        start = next(index for index, line in enumerate(lines) if line.strip() == heading)
    except StopIteration:
        return []
    body: list[str] = []
    for line in lines[start + 1 :]:
        if line.startswith("## "):
            break
        body.append(line)
    return body


def _first_nonempty(lines: list[str]) -> str | None:
    for line in lines:
        stripped = line.strip()
        if stripped:
            return stripped.strip("`")
    return None


def _parse_metadata(lines: list[str], label: str) -> str | None:
    prefix = f"- {label}:"
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(prefix):
            return stripped.removeprefix(prefix).strip().strip("`")
    return None


def _report_name_parts(path: Path) -> tuple[date | None, str, str] | None:
    stem = path.stem
    match = re.match(r"^(\d{4}-\d{2}-\d{2})_(.+)$", stem)
    if match is None:
        return None
    try:
        report_date = date.fromisoformat(match.group(1))
    except ValueError:
        report_date = None
    rest = match.group(2)
    for stage in sorted(STAGE_ORDER, key=len, reverse=True):
        suffix = f"_{stage}"
        if rest.endswith(suffix):
            return report_date, rest[: -len(suffix)], stage
    return None


def _parse_aggregate(lines: list[str]) -> dict[str, Any]:
    for index, line in enumerate(lines):
        if not line.strip().startswith("| Episodes | Success | Collision | Near Miss |"):
            continue
        if index + 2 >= len(lines):
            return {}
        cells = [cell.strip() for cell in lines[index + 2].strip().strip("|").split("|")]
        if len(cells) < 6:
            return {}
        return {
            "episodes": _parse_optional_int(cells[0]),
            "success_rate": _parse_optional_float(cells[1]),
            "collision_rate": _parse_optional_float(cells[2]),
            "near_miss_rate": _parse_optional_float(cells[3]),
            "mean_min_distance": _parse_optional_float(cells[4]),
            "mean_avg_speed": _parse_optional_float(cells[5]),
        }
    return {}


def _parse_failure_counts(lines: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for line in _section_after(lines, "## Failure Taxonomy"):
        stripped = line.strip()
        if not stripped.startswith("- `"):
            continue
        match = re.match(r"^- `([^`]+)`: `?([0-9]+)`?", stripped)
        if match:
            counts[match.group(1)] = int(match.group(2))
    return counts


def _parse_family_runs(lines: list[str]) -> list[str]:
    rows: list[str] = []
    for line in _section_after(lines, "## Family Override Runs"):
        stripped = line.strip()
        if stripped.startswith("- "):
            rows.append(stripped.removeprefix("- "))
    return rows


def parse_report(path: Path) -> CandidateReport | None:
    """Parse one policy-search Markdown report."""

    parts = _report_name_parts(path)
    if parts is None:
        return None
    report_date, candidate, stage = parts
    lines = path.read_text(encoding="utf-8").splitlines()
    eval_lines = _section_after(lines, "## Evaluation Scope")
    aggregate = _parse_aggregate(lines)
    return CandidateReport(
        candidate=candidate,
        stage=stage,
        path=path,
        report_date=report_date,
        decision=_first_nonempty(_section_after(lines, "## Decision")),
        algorithm=_parse_metadata(eval_lines, "Algorithm"),
        scenario_matrix=_parse_metadata(eval_lines, "Scenario matrix"),
        summary_json=_parse_metadata(eval_lines, "Summary JSON"),
        git_commit=_parse_metadata(eval_lines, "Git commit"),
        episodes=aggregate.get("episodes"),
        success_rate=aggregate.get("success_rate"),
        collision_rate=aggregate.get("collision_rate"),
        near_miss_rate=aggregate.get("near_miss_rate"),
        mean_min_distance=aggregate.get("mean_min_distance"),
        mean_avg_speed=aggregate.get("mean_avg_speed"),
        failure_counts=_parse_failure_counts(lines),
        family_runs=_parse_family_runs(lines),
    )


def _candidate_entries(registry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw = registry.get("candidates")
    if not isinstance(raw, dict):
        raise TypeError("Registry is missing a 'candidates' mapping")
    return {str(name): (value if isinstance(value, dict) else {}) for name, value in raw.items()}


def _candidate_names_for_stage(
    entries: dict[str, dict[str, Any]],
    stage: str | None,
    *,
    all_implemented: bool = False,
) -> list[str]:
    names: list[str] = []
    for name, entry in sorted(entries.items()):
        if str(entry.get("status", "")).strip() != "implemented":
            continue
        if stage is not None and not all_implemented:
            required = entry.get("required_stages")
            if not isinstance(required, list) or stage not in {str(item) for item in required}:
                continue
        names.append(name)
    return names


def _best_report(reports: list[CandidateReport]) -> CandidateReport | None:
    if not reports:
        return None
    return max(
        reports,
        key=lambda report: (
            STAGE_RANK.get(report.stage, -1),
            report.report_date or date.min,
            report.success_rate if report.success_rate is not None else -1.0,
            -(report.collision_rate if report.collision_rate is not None else 999.0),
        ),
    )


def _strongest_report(reports: list[CandidateReport]) -> CandidateReport | None:
    comparable = [
        report
        for report in reports
        if report.success_rate is not None and report.collision_rate is not None
    ]
    if not comparable:
        return _best_report(reports)
    return max(
        comparable,
        key=lambda report: (
            report.success_rate or 0.0,
            -(report.collision_rate or 0.0),
            STAGE_RANK.get(report.stage, -1),
        ),
    )


def _fmt(value: float | int | None, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def _candidate_hypothesis(entry: dict[str, Any]) -> str:
    hypothesis = " ".join(str(entry.get("hypothesis", "")).split())
    return hypothesis.rstrip(".")


def _evidence_strengths(report: CandidateReport) -> list[str]:
    strengths: list[str] = []
    success = report.success_rate
    collision = report.collision_rate
    near = report.near_miss_rate

    if success is not None and success >= 0.9:
        strengths.append("very high route-completion rate")
    elif success is not None and success >= 0.8:
        strengths.append("strong route-completion rate")
    elif success is not None and success >= 0.3:
        strengths.append("moderate progress relative to weak baselines")
    elif success is not None:
        strengths.append("limited current success")

    if collision is not None and collision <= 0.025:
        strengths.append("low collision rate")
    elif collision is not None and collision <= 0.06:
        strengths.append("bounded collision rate")
    elif collision is not None:
        strengths.append("collision rate still needs work")

    if near is not None and near <= 0.5:
        strengths.append("low near-miss exposure")
    if report.family_runs:
        strengths.append("scenario-specific overrides are visible in the evidence")
    return strengths


def _failure_note(report: CandidateReport) -> str:
    if not report.failure_counts:
        return ""
    top_failure, top_count = max(report.failure_counts.items(), key=lambda item: item[1])
    return f" Main remaining failure mode: `{top_failure}` ({top_count})."


def _explain_candidate(entry: dict[str, Any], report: CandidateReport | None) -> str:
    hypothesis = _candidate_hypothesis(entry)
    if report is None:
        return hypothesis or "No tracked evaluation report yet."

    strengths = _evidence_strengths(report)
    prefix = ", ".join(strengths) if strengths else "tracked evidence exists"
    failure_note = _failure_note(report)
    if hypothesis:
        return f"{prefix}; design idea: {hypothesis}.{failure_note}"
    return f"{prefix}.{failure_note}"


def _report_record(report: CandidateReport | None) -> dict[str, Any] | None:
    if report is None:
        return None
    return {
        "candidate": report.candidate,
        "stage": report.stage,
        "report": report.path.as_posix(),
        "date": report.report_date.isoformat() if report.report_date else None,
        "decision": report.decision,
        "algorithm": report.algorithm,
        "scenario_matrix": report.scenario_matrix,
        "summary_json": report.summary_json,
        "git_commit": report.git_commit,
        "episodes": report.episodes,
        "success_rate": report.success_rate,
        "collision_rate": report.collision_rate,
        "near_miss_rate": report.near_miss_rate,
        "mean_min_distance": report.mean_min_distance,
        "mean_avg_speed": report.mean_avg_speed,
        "failure_counts": report.failure_counts,
        "family_runs": report.family_runs,
    }


def build_overview(
    *,
    registry_path: Path,
    reports_dir: Path,
) -> dict[str, Any]:
    """Build a JSON-compatible overview from registry and Markdown reports."""

    registry = _load_yaml(registry_path)
    entries = _candidate_entries(registry)
    reports_by_candidate: dict[str, list[CandidateReport]] = {name: [] for name in entries}
    for path in sorted(reports_dir.glob("*.md")):
        report = parse_report(path)
        if report is None:
            continue
        reports_by_candidate.setdefault(report.candidate, []).append(report)

    rows: list[dict[str, Any]] = []
    for name, entry in sorted(entries.items()):
        reports = reports_by_candidate.get(name, [])
        best = _best_report(reports)
        strongest = _strongest_report(reports)
        rows.append(
            {
                "candidate": name,
                "status": entry.get("status"),
                "family": entry.get("family"),
                "training_required": bool(entry.get("training_required", False)),
                "promotion_gate": entry.get("promotion_gate"),
                "candidate_config_path": entry.get("candidate_config_path"),
                "required_stages": entry.get("required_stages", []),
                "report_count": len(reports),
                "best_evidence": _report_record(best),
                "strongest_observed": _report_record(strongest),
                "analysis": _explain_candidate(entry, best),
            }
        )
    rows.sort(
        key=lambda row: (
            -STAGE_RANK.get((row.get("best_evidence") or {}).get("stage", ""), -1),
            -float((row.get("best_evidence") or {}).get("success_rate") or -1.0),
            float((row.get("best_evidence") or {}).get("collision_rate") or 999.0),
            str(row.get("candidate")),
        )
    )
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "registry": registry_path.as_posix(),
        "reports_dir": reports_dir.as_posix(),
        "stage_order": list(STAGE_ORDER),
        "candidate_count": len(entries),
        "rows": rows,
    }


def _write_markdown(overview: dict[str, Any], output_md: Path) -> None:
    rows = overview["rows"]
    leaders = [
        row
        for row in rows
        if row.get("best_evidence")
        and (row["best_evidence"].get("success_rate") is not None)
        and (row["best_evidence"].get("collision_rate") is not None)
    ][:8]
    missing = [row for row in rows if not row.get("best_evidence")]
    needs_full = [
        row
        for row in rows
        if row.get("best_evidence")
        and row["best_evidence"].get("stage") not in PROMOTION_SCALE_STAGES
    ]

    lines = [
        "# Policy Search Portfolio Overview",
        "",
        f"Generated: `{overview['generated_at']}`",
        "",
        "This note summarizes the registered policy-search candidates and the latest tracked reports.",
        "It is an evidence map, not a paper-facing benchmark claim. Stages are not equally strong:",
        "`full_matrix_h500` and `full_matrix` are promotion-scale full-matrix diagnostics,",
        "`stress_slice` is narrower, and `nominal_sanity`/`smoke` are local gates.",
        "",
        "## Current Leaders",
        "",
        "| Candidate | Family | Evidence Stage | Decision | Episodes | Success | Collision | Near Miss | Report |",
        "|---|---|---|---|---:|---:|---:|---:|---|",
    ]
    for row in leaders:
        evidence = row["best_evidence"]
        lines.append(
            "| {candidate} | {family} | {stage} | {decision} | {episodes} | {success} | "
            "{collision} | {near} | `{report}` |".format(
                candidate=row["candidate"],
                family=row.get("family") or "n/a",
                stage=evidence.get("stage") or "n/a",
                decision=evidence.get("decision") or "n/a",
                episodes=_fmt(evidence.get("episodes")),
                success=_fmt(evidence.get("success_rate")),
                collision=_fmt(evidence.get("collision_rate")),
                near=_fmt(evidence.get("near_miss_rate")),
                report=evidence.get("report") or "n/a",
            )
        )

    lines.extend(
        [
            "",
            "## Why The Best Candidates Look Good",
            "",
        ]
    )
    for row in leaders[:5]:
        lines.append(f"- `{row['candidate']}`: {row['analysis']}")

    lines.extend(
        [
            "",
            "## All Registered Candidates",
            "",
            "| Candidate | Status | Family | Required Stages | Best Stage | Success | Collision | Analysis |",
            "|---|---|---|---|---|---:|---:|---|",
        ]
    )
    for row in rows:
        evidence = row.get("best_evidence") or {}
        required = ", ".join(str(item) for item in row.get("required_stages", [])) or "n/a"
        lines.append(
            "| {candidate} | {status} | {family} | {required} | {stage} | {success} | "
            "{collision} | {analysis} |".format(
                candidate=row["candidate"],
                status=row.get("status") or "n/a",
                family=row.get("family") or "n/a",
                required=required,
                stage=evidence.get("stage") or "not_run",
                success=_fmt(evidence.get("success_rate")),
                collision=_fmt(evidence.get("collision_rate")),
                analysis=str(row.get("analysis", "")).replace("|", "/"),
            )
        )

    lines.extend(["", "## Coverage Gaps", ""])
    if missing:
        lines.append("Candidates with no tracked report:")
        for row in missing:
            lines.append(f"- `{row['candidate']}` ({row.get('family') or 'n/a'})")
    else:
        lines.append("- Every registered candidate has at least one tracked report.")
    if needs_full:
        lines.append("")
        lines.append("Candidates that still need `full_matrix` evidence before broad comparison:")
        for row in needs_full:
            evidence = row["best_evidence"]
            lines.append(
                f"- `{row['candidate']}`: best current stage `{evidence.get('stage')}` "
                f"with success `{_fmt(evidence.get('success_rate'))}` and collision "
                f"`{_fmt(evidence.get('collision_rate'))}`"
            )

    lines.extend(
        [
            "",
            "## Reproduction Commands",
            "",
            "List candidates registered for a stage:",
            "",
            "```bash",
            "uv run python scripts/tools/summarize_policy_search_portfolio.py \\",
            "  --list-candidates --stage full_matrix",
            "```",
            "",
            "Submit a SLURM array for a stage:",
            "",
            "```bash",
            "scripts/dev/sbatch_policy_search_sweep.sh --stage full_matrix --dry-run",
            "scripts/dev/sbatch_policy_search_sweep.sh --stage full_matrix --throttle 2",
            "```",
            "",
            "Refresh this overview after new reports land:",
            "",
            "```bash",
            "uv run python scripts/tools/summarize_policy_search_portfolio.py \\",
            f"  --output-md {output_md.as_posix()}",
            "```",
        ]
    )
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    """Run the requested inventory or overview operation."""

    args = parse_args()
    registry = _load_yaml(args.registry)
    entries = _candidate_entries(registry)
    if args.list_candidates:
        for name in _candidate_names_for_stage(
            entries,
            args.stage,
            all_implemented=bool(args.all_implemented),
        ):
            print(name)
        return 0

    overview = build_overview(registry_path=args.registry, reports_dir=args.reports_dir)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(overview, indent=2), encoding="utf-8")
    _write_markdown(overview, args.output_md)
    print(json.dumps({"json": str(args.output_json), "markdown": str(args.output_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
