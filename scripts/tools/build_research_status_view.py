#!/usr/bin/env python3
"""Build a read-only research-status coverage view from a campaign result store."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from robot_sf.benchmark.identity.hash_utils import load_json as _load_json
from scripts.tools.campaign_result_store import read_parquet_frame, validate_result_store

if TYPE_CHECKING:
    import pandas as pd

SCHEMA_VERSION = "research-status-view.v1"
DEFAULT_SUITE_CONFIG = Path("configs/benchmarks/issue_3059_research_engine_suite_v0.yaml")
DEFAULT_PLANNER_MATRIX = Path("configs/benchmarks/planner_readiness_matrix_v1.yaml")
VALID_COVERAGE_STATUSES = frozenset({"native", "adapter"})
FAIL_CLOSED_STATUSES = frozenset({"fallback", "degraded"})
EXCLUDED_OR_LIMITED_STATUSES = frozenset({"diagnostic_only", "unavailable", "failed"})
DATE_PATTERN = re.compile(r"(?P<date>\d{4}-\d{2}-\d{2})")
NULL_IDENTIFIER_TEXT = frozenset({"", "none", "null", "nan", "<na>"})


@dataclass(frozen=True, slots=True)
class SuiteFamily:
    """Configured scenario-family denominator for coverage accounting."""

    family_id: str
    scenario_ids: tuple[str, ...]


def build_research_status_view(
    result_store: Path,
    *,
    suite_config: Path = DEFAULT_SUITE_CONFIG,
    planner_matrix: Path = DEFAULT_PLANNER_MATRIX,
    seed_set: str | None = None,
) -> dict[str, Any]:
    """Build the research-status coverage payload from canonical result-store rows."""
    validation = validate_result_store(result_store)
    if not validation.ok:
        raise ValueError("; ".join(validation.errors))

    frame = read_parquet_frame(result_store / "episodes.parquet")
    summary = _load_json(result_store / "summary.json")
    suite_payload = _load_yaml(suite_config)
    planner_payload = _load_yaml(planner_matrix)
    seed_budget_name, expected_seeds = _load_seed_budget(suite_payload, seed_set=seed_set)
    families = _load_suite_families(suite_payload)
    planners = _known_planners(planner_payload, frame)

    coverage_matrix: dict[str, dict[str, dict[str, Any]]] = {}
    gaps: list[dict[str, str]] = []
    for planner in planners:
        coverage_matrix[planner] = {}
        for family in families:
            cell = _build_cell(
                frame,
                planner=planner,
                family=family,
                expected_seeds=expected_seeds,
            )
            coverage_matrix[planner][family.family_id] = cell
            gaps.extend(_cell_gaps(planner=planner, family_id=family.family_id, cell=cell))

    return {
        "schema_version": SCHEMA_VERSION,
        "view_status": "read_only_coverage_inventory",
        "claim_boundary": (
            "coverage inventory only; fallback, degraded, denominator-invalid, diagnostic-only, "
            "failed, and unavailable cells are not valid benchmark coverage"
        ),
        "input": {
            "result_store": _display_path(result_store),
            "suite_config": _display_path(suite_config),
            "planner_matrix": _display_path(planner_matrix),
            "study_id": summary.get("study_id"),
            "source_commit": summary.get("source_commit"),
            "episode_count": int(summary.get("episode_count", len(frame)) or 0),
            "run_ids": list(summary.get("run_ids", [])),
        },
        "seed_budget_name": seed_budget_name,
        "expected_seeds": expected_seeds,
        "scenario_families": [
            {"family_id": family.family_id, "scenario_ids": list(family.scenario_ids)}
            for family in families
        ],
        "planners": planners,
        "valid_coverage_row_statuses": sorted(VALID_COVERAGE_STATUSES),
        "fail_closed_row_statuses": sorted(FAIL_CLOSED_STATUSES),
        "excluded_or_limited_row_statuses": sorted(EXCLUDED_OR_LIMITED_STATUSES),
        "coverage_matrix": coverage_matrix,
        "gaps": gaps,
        "limitations": [
            "latest_run_date is derived from ISO dates embedded in documented run_id values; "
            "it is null when run ids do not carry a date",
            "coverage requires the configured suite scenario/seed denominator to be complete "
            "with native or adapter rows and no excluded statuses in the cell",
        ],
    }


def render_markdown(payload: dict[str, Any]) -> str:
    """Render a research-status coverage payload as Markdown."""
    lines = [
        "# Research Status Coverage View",
        "",
        f"- Schema: `{payload['schema_version']}`",
        f"- Status: `{payload['view_status']}`",
        f"- Claim boundary: {payload['claim_boundary']}",
        f"- Seed budget: `{payload['seed_budget_name']}` "
        f"({len(payload.get('expected_seeds', []))} seeds)",
        "",
        "## Coverage Matrix",
        "",
        (
            "| planner | scenario_family | coverage_status | evidence_grade | "
            "seed_budget | latest_run_date | row_status_breakdown |"
        ),
        "|---|---|---|---|---|---|---|",
    ]
    coverage_matrix = payload.get("coverage_matrix", {})
    for planner in sorted(coverage_matrix):
        family_cells = coverage_matrix[planner]
        for family_id in sorted(family_cells):
            cell = family_cells[family_id]
            seed_budget = cell.get("seed_budget", {})
            seed_text = (
                f"{seed_budget.get('valid_seed_count', 0)}/"
                f"{seed_budget.get('expected_seed_count', 0)}"
            )
            status_text = _format_status_counts(cell.get("row_status_breakdown", {}))
            lines.append(
                "| "
                f"{planner} | {family_id} | {cell['coverage_status']} | "
                f"{cell['evidence_grade']} | {seed_text} | "
                f"{cell.get('latest_run_date') or 'unknown'} | {status_text} |"
            )

    lines.extend(
        [
            "",
            "## Gaps",
            "",
            "| planner | scenario_family | gap_type | reason |",
            "|---|---|---|---|",
        ]
    )
    gaps = payload.get("gaps", [])
    if gaps:
        for gap in gaps:
            lines.append(
                "| "
                f"{gap['planner']} | {gap['scenario_family']} | "
                f"{gap['gap_type']} | {gap['reason']} |"
            )
    else:
        lines.append("| NA | NA | none | no gaps detected under the configured denominator |")

    lines.extend(["", "## Limitations", ""])
    for limitation in payload.get("limitations", []):
        lines.append(f"- {limitation}")
    lines.append("")
    return "\n".join(lines)


def _build_cell(
    frame: pd.DataFrame,
    *,
    planner: str,
    family: SuiteFamily,
    expected_seeds: list[int],
) -> dict[str, Any]:
    """Build one planner x scenario-family coverage cell."""
    rows = frame[
        (frame["planner"].astype(str) == planner)
        & (frame["scenario_family"].astype(str) == family.family_id)
    ]
    expected_pairs = {
        (scenario_id, int(seed)) for scenario_id in family.scenario_ids for seed in expected_seeds
    }
    valid_rows = rows[rows["row_status"].astype(str).isin(VALID_COVERAGE_STATUSES)]
    valid_pairs = {
        pair
        for row in valid_rows.itertuples(index=False)
        if (pair := _scenario_seed_pair(row.scenario_id, row.seed)) is not None
    }
    present_pairs = {
        pair
        for row in rows.itertuples(index=False)
        if (pair := _scenario_seed_pair(row.scenario_id, row.seed)) is not None
    }
    valid_seeds = sorted({seed for _, seed in valid_pairs if seed in expected_seeds})
    extra_seeds = sorted({seed for _, seed in present_pairs if seed not in expected_seeds})
    missing_pairs = sorted(expected_pairs - valid_pairs)
    denominator_valid = not missing_pairs
    status_counts = _status_counts(rows)
    fail_closed_reasons = _fail_closed_reasons(
        status_counts=status_counts,
        denominator_valid=denominator_valid,
    )
    coverage_status, evidence_grade, valid_coverage = _coverage_classification(
        status_counts=status_counts,
        denominator_valid=denominator_valid,
    )
    return {
        "planner": planner,
        "scenario_family": family.family_id,
        "scenario_ids": list(family.scenario_ids),
        "expected_episode_count": len(expected_pairs),
        "observed_episode_count": len(rows),
        "valid_episode_count": len(valid_rows),
        "valid_pair_count": len(valid_pairs & expected_pairs),
        "denominator_valid": denominator_valid,
        "valid_coverage": valid_coverage,
        "coverage_status": coverage_status,
        "evidence_grade": evidence_grade,
        "latest_run_date": _latest_run_date(rows),
        "latest_run_id": _latest_run_id(rows),
        "row_status_breakdown": status_counts,
        "seed_budget": {
            "expected_seed_count": len(expected_seeds),
            "valid_seed_count": len(valid_seeds),
            "expected_seeds": expected_seeds,
            "valid_seeds": valid_seeds,
            "missing_seeds": sorted(set(expected_seeds) - set(valid_seeds)),
            "extra_seeds": extra_seeds,
        },
        "missing_denominator_pairs": [
            {"scenario_id": scenario_id, "seed": seed} for scenario_id, seed in missing_pairs
        ],
        "fail_closed_reasons": fail_closed_reasons,
    }


def _coverage_classification(
    *,
    status_counts: dict[str, int],
    denominator_valid: bool,
) -> tuple[str, str, bool]:
    """Classify one cell without promoting invalid rows to coverage."""
    if not status_counts:
        return "missing", "missing", False
    if any(status in status_counts for status in FAIL_CLOSED_STATUSES):
        return "fail_closed_fallback_or_degraded", "fail_closed", False
    if not denominator_valid:
        return "fail_closed_denominator_invalid", "fail_closed", False
    if any(status in status_counts for status in EXCLUDED_OR_LIMITED_STATUSES):
        return "excluded_or_limited", "excluded_or_limited", False
    return "covered", "nominal_benchmark_candidate", True


def _fail_closed_reasons(
    *,
    status_counts: dict[str, int],
    denominator_valid: bool,
) -> list[str]:
    """Return stable fail-closed reasons for a cell."""
    reasons: list[str] = []
    if any(status in status_counts for status in FAIL_CLOSED_STATUSES):
        reasons.append("fallback_or_degraded_rows_present")
    if not denominator_valid:
        reasons.append("denominator_invalid")
    if any(status in status_counts for status in EXCLUDED_OR_LIMITED_STATUSES):
        reasons.append("excluded_or_limited_rows_present")
    return reasons


def _cell_gaps(*, planner: str, family_id: str, cell: dict[str, Any]) -> list[dict[str, str]]:
    """Return gap records for one coverage cell."""
    status = cell["coverage_status"]
    if status == "covered":
        return []
    if status == "missing":
        return [
            {
                "planner": planner,
                "scenario_family": family_id,
                "gap_type": "missing_cell",
                "reason": "no result-store rows for known planner and suite scenario family",
            }
        ]
    gaps: list[dict[str, str]] = []
    if "fallback_or_degraded_rows_present" in cell.get("fail_closed_reasons", []):
        gaps.append(
            {
                "planner": planner,
                "scenario_family": family_id,
                "gap_type": "fail_closed_status",
                "reason": "fallback or degraded rows are present and cannot count as coverage",
            }
        )
    if "denominator_invalid" in cell.get("fail_closed_reasons", []):
        gaps.append(
            {
                "planner": planner,
                "scenario_family": family_id,
                "gap_type": "denominator_invalid",
                "reason": "native or adapter rows do not cover every configured scenario/seed pair",
            }
        )
    if "excluded_or_limited_rows_present" in cell.get("fail_closed_reasons", []):
        gaps.append(
            {
                "planner": planner,
                "scenario_family": family_id,
                "gap_type": "excluded_or_limited_status",
                "reason": "diagnostic-only, unavailable, or failed rows cannot count as coverage",
            }
        )
    return gaps


def _load_seed_budget(payload: dict[str, Any], *, seed_set: str | None) -> tuple[str, list[int]]:
    """Load the requested suite seed budget."""
    seed_policy = payload.get("seed_policy", {})
    candidates: list[dict[str, Any]] = []
    pilot = seed_policy.get("pilot_set")
    if isinstance(pilot, dict):
        candidates.append(pilot)
    candidates.extend(seed_policy.get("escalation_sets", []) or [])
    if not candidates:
        raise ValueError("suite config does not define seed_policy pilot_set or escalation_sets")

    requested = seed_set.lower() if seed_set else None
    for candidate in candidates:
        name = str(candidate.get("name", "unnamed"))
        if requested is None or name.lower() == requested:
            seeds = [int(seed) for seed in candidate.get("seeds", [])]
            if not seeds:
                raise ValueError(f"suite seed set {name!r} does not define seeds")
            return name, seeds
    available = ", ".join(str(candidate.get("name", "unnamed")) for candidate in candidates)
    raise ValueError(f"seed set {seed_set!r} not found in suite config; available: {available}")


def _load_suite_families(payload: dict[str, Any]) -> list[SuiteFamily]:
    """Load scenario-family denominators from the suite config."""
    families: list[SuiteFamily] = []
    for item in payload.get("scenario_families", []) or []:
        family_id = _clean_identifier(item.get("family_id"))
        scenario_ids = tuple(
            scenario_id
            for value in item.get("scenario_ids", []) or []
            if (scenario_id := _clean_identifier(value)) is not None
        )
        if not family_id or not scenario_ids:
            raise ValueError("suite scenario_families entries require family_id and scenario_ids")
        families.append(SuiteFamily(family_id=family_id, scenario_ids=scenario_ids))
    if not families:
        raise ValueError("suite config does not define scenario_families")
    return families


def _known_planners(payload: dict[str, Any], frame: pd.DataFrame) -> list[str]:
    """Return the known planner set plus any observed unregistered planners."""
    planners = {
        planner_id
        for row in payload.get("rows", []) or []
        if isinstance(row, dict) and (planner_id := _clean_identifier(row.get("planner_id")))
    }
    observed = {
        planner_id
        for value in frame["planner"].tolist()
        if (planner_id := _clean_identifier(value)) is not None
    }
    planners.update(observed)
    if not planners:
        raise ValueError("planner matrix does not define rows[].planner_id")
    return sorted(planners)


def _clean_identifier(value: Any) -> str | None:
    """Return a non-null, stripped identifier string."""
    if value is None:
        return None
    if isinstance(value, Real) and not isinstance(value, bool) and not math.isfinite(float(value)):
        return None
    text = str(value).strip()
    if text.lower() in NULL_IDENTIFIER_TEXT:
        return None
    return text


def _scenario_seed_pair(scenario_id: Any, seed: Any) -> tuple[str, int] | None:
    """Return a valid scenario/seed pair, skipping null or non-finite values."""
    scenario = _clean_identifier(scenario_id)
    if scenario is None:
        return None
    try:
        seed_float = float(seed)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(seed_float) or not seed_float.is_integer():
        return None
    return scenario, int(seed_float)


def _status_counts(rows: pd.DataFrame) -> dict[str, int]:
    """Count row statuses in stable lexical order."""
    if rows.empty:
        return {}
    counts = rows["row_status"].astype(str).value_counts().to_dict()
    return {status: int(counts[status]) for status in sorted(counts)}


def _latest_run_date(rows: pd.DataFrame) -> str | None:
    """Return the latest ISO date embedded in documented run_id values."""
    dates: list[str] = []
    for run_id in _run_ids(rows):
        match = DATE_PATTERN.search(run_id)
        if match:
            dates.append(match.group("date"))
    return max(dates) if dates else None


def _latest_run_id(rows: pd.DataFrame) -> str | None:
    """Return the latest run id in stable lexical order."""
    run_ids = _run_ids(rows)
    return max(run_ids) if run_ids else None


def _run_ids(rows: pd.DataFrame) -> list[str]:
    """Return stable run ids from a cell frame."""
    if rows.empty:
        return []
    return sorted({str(value) for value in rows["run_id"].dropna().tolist()})


def _format_status_counts(counts: dict[str, int]) -> str:
    """Format row-status counts for compact Markdown cells."""
    if not counts:
        return "none"
    return ", ".join(f"{status}:{count}" for status, count in counts.items())


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return payload


def _display_path(path: Path) -> str:
    """Return a compact display path when possible."""
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _write_outputs(payload: dict[str, Any], *, output_json: Path, output_md: Path) -> None:
    """Write JSON and Markdown coverage artifacts."""
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md.write_text(render_markdown(payload), encoding="utf-8")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-store", type=Path, required=True)
    parser.add_argument(
        "--output-json", "--json-output", dest="output_json", type=Path, required=True
    )
    parser.add_argument("--output-md", "--markdown", dest="output_md", type=Path, required=True)
    parser.add_argument("--suite-config", type=Path, default=DEFAULT_SUITE_CONFIG)
    parser.add_argument("--planner-matrix", type=Path, default=DEFAULT_PLANNER_MATRIX)
    parser.add_argument(
        "--seed-set",
        default=None,
        help="Configured seed set name to use from the suite; defaults to the pilot set.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = _parse_args(argv)
    try:
        payload = build_research_status_view(
            args.result_store,
            suite_config=args.suite_config,
            planner_matrix=args.planner_matrix,
            seed_set=args.seed_set,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    _write_outputs(payload, output_json=args.output_json, output_md=args.output_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
