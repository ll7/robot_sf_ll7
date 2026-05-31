#!/usr/bin/env python3
"""Run a tiny paired planner pilot over materialized scenario perturbations."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from robot_sf.benchmark.map_runner import run_map_batch
from robot_sf.scenario_certification import materialize_perturbation_pilot_matrix
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


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSONL rows from a benchmark episode file."""
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


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


def _nested_strings(value: Any) -> list[str]:
    """Return lower-cased string leaves from a nested JSON-like value."""
    strings: list[str] = []
    if isinstance(value, str):
        strings.append(value.strip().lower())
    elif isinstance(value, dict):
        for item in value.values():
            strings.extend(_nested_strings(item))
    elif isinstance(value, list | tuple):
        for item in value:
            strings.extend(_nested_strings(item))
    return strings


def classify_episode_status(row: dict[str, Any] | None) -> str:
    """Classify whether an episode row can contribute to paired evidence."""
    if row is None:
        return "missing"
    if isinstance(row.get("scenario_exclusion"), dict):
        return "invalid"
    metadata_strings = _nested_strings(row.get("algorithm_metadata"))
    if any("fallback" in item for item in metadata_strings):
        return "fallback"
    if any("degraded" in item for item in metadata_strings):
        return "degraded"
    reason = str(row.get("termination_reason") or "").strip().lower()
    if reason == "error":
        return "failed"
    return "completed"


def _metric(row: dict[str, Any] | None, name: str) -> float | None:
    """Read a numeric metric from an episode row."""
    if row is None:
        return None
    metrics = row.get("metrics")
    if not isinstance(metrics, dict):
        return None
    value = metrics.get(name)
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _episode_values(row: dict[str, Any] | None) -> dict[str, Any]:
    """Return compact outcome values for one episode row."""
    reason = str(row.get("termination_reason") or "") if row is not None else ""
    return {
        "success": 1 if reason.lower() == "success" else 0,
        "collision": 1 if reason.lower() == "collision" else 0,
        "timeout": 1 if reason.lower() in {"max_steps", "terminated", "truncated"} else 0,
        "min_distance": _metric(row, "min_distance"),
        "termination_reason": reason or None,
    }


def _index_records_by_planner_seed(
    records_by_planner: dict[str, list[dict[str, Any]]],
) -> dict[tuple[str, str, int], dict[str, Any]]:
    """Index episode records by planner, materialized scenario id, and seed."""
    rows_by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
    for planner, records in records_by_planner.items():
        for row in records:
            scenario_id = str(row.get("scenario_id") or row.get("scenario") or "")
            seed = int(row.get("seed", 0))
            rows_by_key[(planner, scenario_id, seed)] = row
    return rows_by_key


def _scenarios_by_source(
    scenario_metadata: dict[str, dict[str, Any]],
) -> dict[str, dict[str, list[str]]]:
    """Group materialized scenario ids by source scenario and perturbation family."""
    grouped: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for scenario_id, metadata in scenario_metadata.items():
        source = metadata["source_scenario_id"]
        family = metadata["family"]
        if source and family:
            grouped[source][family].append(scenario_id)
    return grouped


def _seeds_for_pair(
    rows_by_key: dict[tuple[str, str, int], dict[str, Any]],
    *,
    planner: str,
    noop_id: str,
    variant_id: str,
) -> list[int]:
    """Return seeds observed for either side of one no-op/perturbed pair."""
    return sorted(
        {
            seed
            for key_planner, scenario_id, seed in rows_by_key
            if key_planner == planner and scenario_id in {noop_id, variant_id}
        }
    )


def _delta(after: int | float | None, before: int | float | None) -> float | None:
    """Return numeric delta when both sides are available."""
    if after is None or before is None:
        return None
    return float(after) - float(before)


def build_pair_table(
    records_by_planner: dict[str, list[dict[str, Any]]],
    scenario_metadata: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build paired no-op versus route-offset deltas for each planner and seed."""
    rows_by_key = _index_records_by_planner_seed(records_by_planner)
    grouped_scenarios = _scenarios_by_source(scenario_metadata)

    pair_rows: list[dict[str, Any]] = []
    for planner in sorted(records_by_planner):
        for source_scenario_id, families in sorted(grouped_scenarios.items()):
            noop_ids = sorted(families.get("noop", []))
            perturbed_ids = sorted(
                variant_id
                for family, variant_ids in families.items()
                if family != "noop"
                for variant_id in variant_ids
            )
            if not noop_ids:
                for variant_id in perturbed_ids:
                    pair_rows.append(
                        {
                            "planner": planner,
                            "source_scenario_id": source_scenario_id,
                            "noop_variant_id": None,
                            "perturbed_variant_id": variant_id,
                            "seed": None,
                            "pair_status": "missing_noop",
                        }
                    )
                continue
            noop_id = noop_ids[0]
            for variant_id in perturbed_ids:
                for seed in _seeds_for_pair(
                    rows_by_key,
                    planner=planner,
                    noop_id=noop_id,
                    variant_id=variant_id,
                ):
                    noop_row = rows_by_key.get((planner, noop_id, seed))
                    perturbed_row = rows_by_key.get((planner, variant_id, seed))
                    noop_status = classify_episode_status(noop_row)
                    perturbed_status = classify_episode_status(perturbed_row)
                    pair_status = (
                        "completed"
                        if noop_status == "completed" and perturbed_status == "completed"
                        else "excluded"
                    )
                    noop_values = _episode_values(noop_row)
                    perturbed_values = _episode_values(perturbed_row)
                    perturbed_metadata = scenario_metadata.get(variant_id, {})
                    pair_rows.append(
                        {
                            "planner": planner,
                            "source_scenario_id": source_scenario_id,
                            "noop_variant_id": noop_id,
                            "perturbed_variant_id": variant_id,
                            "perturbed_family": str(perturbed_metadata.get("family") or "unknown"),
                            "seed": seed,
                            "pair_status": pair_status,
                            "noop_status": noop_status,
                            "perturbed_status": perturbed_status,
                            "success_delta": _delta(
                                perturbed_values["success"], noop_values["success"]
                            ),
                            "collision_delta": _delta(
                                perturbed_values["collision"], noop_values["collision"]
                            ),
                            "timeout_delta": _delta(
                                perturbed_values["timeout"], noop_values["timeout"]
                            ),
                            "min_distance_delta": _delta(
                                perturbed_values["min_distance"], noop_values["min_distance"]
                            ),
                            "noop": noop_values,
                            "perturbed": perturbed_values,
                        }
                    )
    return pair_rows


def _summarize_pair_subset(pair_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate one pair-table subset."""
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


def _grouped_pair_summaries(
    pair_rows: list[dict[str, Any]],
    *,
    field: str,
) -> dict[str, dict[str, Any]]:
    """Aggregate pair rows by one categorical field."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pair_rows:
        key = str(row.get(field) or "unknown")
        grouped[key].append(row)
    return {key: _summarize_pair_subset(rows) for key, rows in sorted(grouped.items())}


def summarize_pairs(pair_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate pair-table status counts and mean deltas."""
    summary = _summarize_pair_subset(pair_rows)
    summary["by_planner"] = _grouped_pair_summaries(pair_rows, field="planner")
    summary["by_source_scenario"] = _grouped_pair_summaries(
        pair_rows,
        field="source_scenario_id",
    )
    summary["by_perturbation_family"] = _grouped_pair_summaries(
        pair_rows,
        field="perturbed_family",
    )
    return summary


def _write_markdown(summary: dict[str, Any], path: Path) -> None:
    """Write a compact Markdown summary next to the local pilot artifacts."""
    lines = [
        "# Scenario Perturbation Criticality Pilot",
        "",
        "## Boundary",
        "",
        "Diagnostic local pilot only. Raw JSONL remains ignored local output; tracked evidence should use the compact summary.",
        "",
        "## Aggregate",
        "",
        f"- Planners: {', '.join(summary['planners'])}",
        f"- Materialized variants: {summary['materialization']['variant_count']}",
        f"- Pair rows: {summary['pair_summary']['pairs']}",
        f"- Pair statuses: `{json.dumps(summary['pair_summary']['status_counts'], sort_keys=True)}`",
        "",
        "## Mean Deltas For Completed Pairs",
        "",
    ]
    deltas = summary["pair_summary"]["mean_deltas_completed_pairs"]
    if deltas:
        for field, value in deltas.items():
            lines.append(f"- `{field}`: `{value:.4f}`")
    else:
        lines.append("- none")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _compact_tracked_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Return the small reviewable subset suitable for docs/context/evidence."""
    materialization = dict(summary["materialization"])
    materialization.pop("scenario_matrix_path", None)
    materialization.pop("summary_path", None)
    materialization["local_artifact_boundary"] = (
        "materialized scenario matrix, route overrides, and raw episode JSONL remain ignored "
        "local outputs reproducible from the tracked manifest and command"
    )
    return {
        "schema_version": summary["schema_version"],
        "manifest": summary["manifest"],
        "planners": summary["planners"],
        "horizon": summary["horizon"],
        "dt": summary["dt"],
        "seed_limit": summary["seed_limit"],
        "materialization": materialization,
        "planner_runs": {
            planner: {
                "algo": run["algo"],
                "algo_config_path": run["algo_config_path"],
                "source": run["source"],
                "episodes": run["episodes"],
            }
            for planner, run in summary["planner_runs"].items()
        },
        "pair_summary": summary["pair_summary"],
        "pair_rows": summary["pair_rows"],
        "claim_boundary": summary["claim_boundary"],
    }


def main() -> int:
    """Run the scenario perturbation pilot."""
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

    pair_rows = build_pair_table(records_by_planner, metadata)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "manifest": args.manifest.as_posix(),
        "planners": [planner_spec.label for planner_spec in planner_specs],
        "horizon": args.horizon,
        "dt": args.dt,
        "seed_limit": args.seed_limit,
        "materialization": {
            "schema_version": materialized.schema_version,
            "manifest_id": materialized.manifest_id,
            "scenario_matrix_path": materialized.scenario_matrix_path,
            "summary_path": materialized.summary_path,
            "included_variants": list(materialized.included_variants),
            "excluded_variants": list(materialized.excluded_variants),
            "variant_count": len(materialized.included_variants),
        },
        "planner_runs": planner_runs,
        "pair_summary": summarize_pairs(pair_rows),
        "pair_rows": pair_rows,
        "claim_boundary": (
            "diagnostic local pilot only; not benchmark-strength or paper-facing evidence"
        ),
    }
    summary_path = args.pilot_output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(summary, args.pilot_output_dir / "summary.md")
    if args.evidence_summary is not None:
        args.evidence_summary.parent.mkdir(parents=True, exist_ok=True)
        args.evidence_summary.write_text(
            json.dumps(_compact_tracked_summary(summary), indent=2, sort_keys=True) + "\n",
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
