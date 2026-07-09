#!/usr/bin/env python3
"""Run a tiny paired planner pilot over materialized scenario perturbations.

Routes the compact evidence summary through
``robot_sf.scenario_certification.criticality_summary`` v1 helpers
(``build_criticality_summary_from_pilot``,
``criticality_summary_to_dict``, ``validate_criticality_summary``)
instead of maintaining duplicate row-status/completed-pair summary logic
here. The raw local ``summary.json`` under ``output/`` retains the
original schema for backward compatibility.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from robot_sf.benchmark.identity.hash_utils import read_jsonl as _load_jsonl
from robot_sf.benchmark.map_runner import run_map_batch
from robot_sf.scenario_certification import materialize_perturbation_pilot_matrix
from robot_sf.scenario_certification.criticality_summary import (
    build_criticality_summary_from_pilot,
    criticality_summary_to_dict,
    validate_criticality_summary,
)
from robot_sf.training.scenario_loader import load_scenarios

SCHEMA_VERSION = "scenario_perturbation_criticality_pilot.v1"
_EPISODE_SCHEMA = Path("robot_sf/benchmark/schemas/episode.schema.v1.json")
_DEFAULT_CANDIDATE_REGISTRY = Path("docs/context/policy_search/candidate_registry.yaml")


@dataclass(frozen=True)
class PlannerRunSpec:
    """Resolved planner execution contract for one pilot label."""

    label: str
    algo: str
    source: str
    algo_config_path: Path | None = None


def _build_parser() -> argparse.ArgumentParser:
    """Build the pilot CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="Scenario perturbation manifest YAML.")
    parser.add_argument(
        "--materialized-output-dir",
        type=Path,
        required=True,
        help="Ignored output/ directory for generated scenario matrix and route overrides.",
    )
    parser.add_argument(
        "--pilot-output-dir",
        type=Path,
        required=True,
        help="Ignored output/ directory for raw pilot JSONL and local summary.",
    )
    parser.add_argument(
        "--planner",
        action="append",
        dest="planners",
        help=(
            "Planner algorithm or policy-search candidate key to run. Repeat for multiple "
            "planners. Defaults to goal and orca."
        ),
    )
    parser.add_argument(
        "--planner-candidate-registry",
        type=Path,
        default=_DEFAULT_CANDIDATE_REGISTRY,
        help=(
            "Policy-search candidate registry used to resolve planner keys with "
            "candidate_config_path entries."
        ),
    )
    parser.add_argument(
        "--seed-limit",
        type=int,
        default=1,
        help="Positive cap on seeds per materialized variant for this local pilot.",
    )
    parser.add_argument("--horizon", type=int, default=80)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--evidence-summary",
        type=Path,
        help="Optional compact tracked JSON summary path. Raw records are never copied here.",
    )
    return parser


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    """Load a YAML mapping from disk."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return payload


def resolve_planner_run_spec(
    planner: str,
    *,
    candidate_registry_path: Path = _DEFAULT_CANDIDATE_REGISTRY,
) -> PlannerRunSpec:
    """Resolve a CLI planner token to a raw algo or policy-search candidate config."""
    if not candidate_registry_path.exists():
        return PlannerRunSpec(label=planner, algo=planner, source="raw_algo")

    registry = _load_yaml_mapping(candidate_registry_path)
    candidates = registry.get("candidates")
    candidate = candidates.get(planner) if isinstance(candidates, dict) else None
    if not isinstance(candidate, dict):
        return PlannerRunSpec(label=planner, algo=planner, source="raw_algo")

    candidate_config_raw = candidate.get("candidate_config_path")
    if not isinstance(candidate_config_raw, str) or not candidate_config_raw.strip():
        return PlannerRunSpec(label=planner, algo=planner, source="raw_algo")

    candidate_config_path = Path(candidate_config_raw)
    if not candidate_config_path.exists() and not candidate_config_path.is_absolute():
        candidate_config_path = candidate_registry_path.parent / candidate_config_path
    candidate_config = _load_yaml_mapping(candidate_config_path)
    algo = candidate_config.get("algo")
    if not isinstance(algo, str) or not algo.strip():
        raise ValueError(
            f"Policy-search candidate {planner!r} in {candidate_config_path} lacks algo"
        )
    return PlannerRunSpec(
        label=planner,
        algo=algo,
        algo_config_path=candidate_config_path,
        source="policy_search_candidate",
    )


def _planner_output_stem(label: str) -> str:
    """Return a filesystem-safe output stem for a planner label."""
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in label)


def _scenario_metadata(scenarios: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Return perturbation metadata keyed by materialized scenario id."""
    metadata: dict[str, dict[str, Any]] = {}
    for scenario in scenarios:
        scenario_id = str(scenario.get("scenario_id") or scenario.get("name") or "")
        raw = scenario.get("metadata")
        perturbation = raw.get("scenario_perturbation") if isinstance(raw, dict) else None
        if not isinstance(perturbation, dict):
            continue
        metadata[scenario_id] = {
            "source_scenario_id": str(perturbation.get("source_scenario_id") or ""),
            "variant_id": str(perturbation.get("variant_id") or scenario_id),
            "family": str(perturbation.get("family") or ""),
            "benchmark_evidence_status": str(perturbation.get("benchmark_evidence_status") or ""),
            "evidence_boundary": str(perturbation.get("evidence_boundary") or ""),
            "perturbation_summary": perturbation.get("perturbation_summary") or {},
        }
    return metadata


def _old_format_summarize(pair_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate one pair-table subset for the legacy raw summary format."""
    status_counts: dict[str, int] = defaultdict(int)
    delta_values: dict[str, list[float]] = defaultdict(list)
    for row in pair_rows:
        status_counts[str(row.get("pair_status") or "unknown")] += 1
        if row.get("pair_status") != "completed":
            continue
        for field in (
            "success_delta",
            "collision_delta",
            "timeout_delta",
            "min_distance_delta",
        ):
            value = row.get(field)
            if value is not None:
                delta_values[field].append(float(value))
    return {
        "pairs": len(pair_rows),
        "status_counts": dict(sorted(status_counts.items())),
        "mean_deltas_completed_pairs": {
            field: sum(values) / len(values)
            for field, values in sorted(delta_values.items())
            if values
        },
    }


def _old_format_grouped_summaries(
    pair_rows: list[dict[str, Any]],
    *,
    field: str,
) -> dict[str, dict[str, Any]]:
    """Aggregate pair rows by one categorical field (legacy raw format)."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pair_rows:
        key = str(row.get(field) or "unknown")
        grouped[key].append(row)
    return {key: _old_format_summarize(rows) for key, rows in sorted(grouped.items())}


def _old_format_pair_summary(pair_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Build legacy-format pair summary (status_counts) from v1 pair rows."""
    summary = _old_format_summarize(pair_rows)
    summary["by_planner"] = _old_format_grouped_summaries(pair_rows, field="planner")
    summary["by_source_scenario"] = _old_format_grouped_summaries(
        pair_rows,
        field="source_scenario_id",
    )
    summary["by_perturbation_family"] = _old_format_grouped_summaries(
        pair_rows,
        field="perturbed_family",
    )
    return summary


def _write_markdown_from_v1(v1_payload: dict[str, Any], path: Path) -> None:
    """Write a compact Markdown summary from the v1 evidence payload."""
    pair_summary = v1_payload["pair_summary"]
    lines = [
        "# Scenario Perturbation Criticality Pilot",
        "",
        "## Boundary",
        "",
        "Diagnostic local pilot only. Raw JSONL remains ignored local output; tracked evidence uses the criticality_summary.v1 schema.",
        "",
        "## Aggregate",
        "",
        f"- Planners: {', '.join(v1_payload['planners'])}",
        f"- Materialized variants: {v1_payload['materialization']['variant_count']}",
        f"- Pair rows: {pair_summary['pairs']}",
        "- Perturbed row status counts: "
        f"`{json.dumps(pair_summary['row_status_counts'], sort_keys=True)}`",
        "",
        "## Mean Deltas For Completed Pairs",
        "",
    ]
    deltas = pair_summary["mean_deltas_completed_pairs"]
    if deltas:
        for field, value in deltas.items():
            lines.append(f"- `{field}`: `{value:.4f}`")
    else:
        lines.append("- none")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_validated_criticality_summary_payload(  # noqa: PLR0913
    *,
    records_by_planner: dict[str, list[dict[str, Any]]],
    scenario_metadata: dict[str, dict[str, Any]],
    manifest: str,
    manifest_id: str,
    planners: list[str],
    horizon: int,
    dt: float,
    seed_limit: int,
    materialization: dict[str, Any],
    planner_runs: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build and validate the compact criticality_summary.v1 evidence payload."""
    v1_summary = build_criticality_summary_from_pilot(
        records_by_planner=records_by_planner,
        scenario_metadata=scenario_metadata,
        manifest=manifest,
        manifest_id=manifest_id,
        planners=planners,
        horizon=horizon,
        dt=dt,
        seed_limit=seed_limit,
        materialization=materialization,
        planner_runs=planner_runs,
    )
    v1_payload = criticality_summary_to_dict(v1_summary)
    validate_criticality_summary(v1_payload)
    return v1_payload, v1_summary.pair_rows


def main() -> int:
    """Run the scenario perturbation pilot.

    Builds the evidence summary through criticality_summary v1 helpers
    (build_criticality_summary_from_pilot → criticality_summary_to_dict
    → validate_criticality_summary). The local raw ``summary.json``
    retains the legacy format for backward compatibility under ``output/``.
    """
    args = _build_parser().parse_args()
    planners = args.planners or ["goal", "orca"]
    planner_specs = [
        resolve_planner_run_spec(planner, candidate_registry_path=args.planner_candidate_registry)
        for planner in planners
    ]
    materialized = materialize_perturbation_pilot_matrix(
        args.manifest,
        output_dir=args.materialized_output_dir,
        seed_limit=args.seed_limit,
    )
    matrix_path = Path(materialized.scenario_matrix_path)
    scenarios = load_scenarios(matrix_path)
    metadata = _scenario_metadata(scenarios)

    args.pilot_output_dir.mkdir(parents=True, exist_ok=True)
    records_by_planner: dict[str, list[dict[str, Any]]] = {}
    planner_runs: dict[str, dict[str, Any]] = {}
    for planner_spec in planner_specs:
        jsonl_path = (
            args.pilot_output_dir / f"{_planner_output_stem(planner_spec.label)}.episodes.jsonl"
        )
        if jsonl_path.exists():
            jsonl_path.unlink()
        batch_summary = run_map_batch(
            matrix_path,
            jsonl_path,
            schema_path=_EPISODE_SCHEMA,
            algo=planner_spec.algo,
            algo_config_path=(
                planner_spec.algo_config_path.as_posix()
                if planner_spec.algo_config_path is not None
                else None
            ),
            horizon=args.horizon,
            dt=args.dt,
            workers=args.workers,
            resume=False,
            benchmark_profile="experimental",
        )
        records = _load_jsonl(jsonl_path)
        records_by_planner[planner_spec.label] = records
        planner_runs[planner_spec.label] = {
            "algo": planner_spec.algo,
            "algo_config_path": (
                planner_spec.algo_config_path.as_posix()
                if planner_spec.algo_config_path is not None
                else None
            ),
            "source": planner_spec.source,
            "jsonl_path": jsonl_path.as_posix(),
            "episodes": len(records),
            "batch_summary": batch_summary,
        }

    # Build the v1 evidence summary via the criticality_summary helpers
    v1_materialization = {
        "schema_version": materialized.schema_version,
        "manifest_id": materialized.manifest_id,
        "included_variants": list(materialized.included_variants),
        "excluded_variants": list(materialized.excluded_variants),
        "variant_count": len(materialized.included_variants),
        "local_artifact_boundary": (
            "materialized scenario matrix, route overrides, and raw episode JSONL "
            "remain ignored local outputs reproducible from the tracked manifest and command"
        ),
    }
    v1_planner_runs: dict[str, dict[str, Any]] = {}
    for label, run in planner_runs.items():
        v1_planner_runs[label] = {
            "algo": run["algo"],
            "algo_config_path": run["algo_config_path"],
            "source": run["source"],
            "episodes": run["episodes"],
        }
    v1_payload, pair_rows = build_validated_criticality_summary_payload(
        records_by_planner=records_by_planner,
        scenario_metadata=metadata,
        manifest=args.manifest.as_posix(),
        manifest_id=materialized.manifest_id,
        planners=[s.label for s in planner_specs],
        horizon=args.horizon,
        dt=args.dt,
        seed_limit=args.seed_limit,
        materialization=v1_materialization,
        planner_runs=v1_planner_runs,
    )

    # Legacy raw summary (backward compatible format)
    old_materialization = {
        "schema_version": materialized.schema_version,
        "manifest_id": materialized.manifest_id,
        "scenario_matrix_path": materialized.scenario_matrix_path,
        "summary_path": materialized.summary_path,
        "included_variants": list(materialized.included_variants),
        "excluded_variants": list(materialized.excluded_variants),
        "variant_count": len(materialized.included_variants),
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "manifest": args.manifest.as_posix(),
        "planners": [planner_spec.label for planner_spec in planner_specs],
        "horizon": args.horizon,
        "dt": args.dt,
        "seed_limit": args.seed_limit,
        "materialization": old_materialization,
        "planner_runs": planner_runs,
        "pair_summary": _old_format_pair_summary(pair_rows),
        "pair_rows": pair_rows,
        "claim_boundary": (
            "diagnostic local pilot only; not benchmark-strength or paper-facing evidence"
        ),
    }
    summary_path = args.pilot_output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown_from_v1(v1_payload, args.pilot_output_dir / "summary.md")

    if args.evidence_summary is not None:
        args.evidence_summary.parent.mkdir(parents=True, exist_ok=True)
        args.evidence_summary.write_text(
            json.dumps(v1_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    print(
        json.dumps(
            {
                "summary": summary_path.as_posix(),
                "evidence_summary": (
                    args.evidence_summary.as_posix() if args.evidence_summary is not None else None
                ),
                "pair_summary": summary["pair_summary"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
