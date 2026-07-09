"""Run a bounded one-factor hybrid component ablation pilot."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from robot_sf.benchmark.identity.hash_utils import load_json as _load_json

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = Path(
    "configs/policy_search/ablation_manifests/issue_2170_one_factor_hybrid_component_manifest.yaml"
)
DEFAULT_REGISTRY = Path("docs/context/policy_search/candidate_registry.yaml")
METRICS = (
    "success_rate",
    "collision_rate",
    "near_miss_rate",
    "mean_avg_speed",
    "runtime_sec",
)


def _num(value: Any) -> float | None:
    """Return value as float when it is numeric."""
    return float(value) if isinstance(value, int | float) else None


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"Expected YAML mapping: {path}")
    return payload


def _repo_path(raw: str | Path) -> Path:
    """Resolve a path relative to the repository root."""
    path = Path(raw)
    return path if path.is_absolute() else REPO_ROOT / path


def _scenario_seeds(manifest: dict[str, Any]) -> dict[str, list[int]]:
    """Return the explicit scenario seed map from the manifest."""
    scenario_slice = manifest.get("scenario_slice")
    if not isinstance(scenario_slice, dict):
        raise ValueError("manifest missing scenario_slice mapping")
    seed_policy = scenario_slice.get("seed_policy")
    if not isinstance(seed_policy, dict):
        raise ValueError("manifest missing scenario_slice.seed_policy mapping")
    seeds = seed_policy.get("seeds")
    if not isinstance(seeds, list):
        raise ValueError("manifest seed policy must define a seed list")
    scenario_rows = scenario_slice.get("scenarios")
    if not isinstance(scenario_rows, list):
        raise ValueError("manifest scenario_slice.scenarios must be a list")
    seed_values = [int(seed) for seed in seeds]
    out: dict[str, list[int]] = {}
    for row in scenario_rows:
        if not isinstance(row, dict) or not row.get("id"):
            raise ValueError("each scenario row must define id")
        out[str(row["id"])] = list(seed_values)
    return out


def _candidate_from_config_path(path: Path) -> str:
    """Load the candidate name from a config path."""
    payload = _load_yaml(path)
    name = payload.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"Candidate config missing name: {path}")
    return name


def _row_candidates(manifest: dict[str, Any]) -> dict[str, str]:
    """Map manifest row keys to registered candidate names."""
    rows: dict[str, str] = {}
    for section in ("existing_reference_rows", "planned_one_factor_rows"):
        payload = manifest.get(section)
        if not isinstance(payload, list):
            continue
        for row in payload:
            if not isinstance(row, dict):
                continue
            key = row.get("key")
            if not isinstance(key, str):
                continue
            candidate = row.get("candidate")
            if isinstance(candidate, str) and candidate.strip():
                rows[key] = candidate
                continue
            path = row.get("planned_config_path") or row.get("existing_or_planned_config_path")
            if isinstance(path, str) and path.strip():
                rows[key] = _candidate_from_config_path(_repo_path(path))
    return rows


def _selected_comparisons(
    manifest: dict[str, Any], comparison_ids: set[str] | None
) -> list[dict[str, Any]]:
    """Return selected comparison rows."""
    comparisons = manifest.get("comparison_plan")
    if not isinstance(comparisons, list):
        raise ValueError("manifest missing comparison_plan list")
    selected: list[dict[str, Any]] = []
    for row in comparisons:
        if not isinstance(row, dict):
            continue
        comparison_id = str(row.get("id", ""))
        if comparison_ids is not None and comparison_id not in comparison_ids:
            continue
        selected.append(row)
    if not selected:
        raise ValueError("no manifest comparisons selected")
    return selected


def _write_local_funnel(
    *, manifest: dict[str, Any], output_dir: Path, horizon: int | None, workers: int | None
) -> Path:
    """Write a custom full_matrix stage funnel for the manifest slice."""
    execution = manifest.get("execution_contract")
    if not isinstance(execution, dict):
        raise ValueError("manifest missing execution_contract mapping")
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_manifest = output_dir / "scenario_seeds.yaml"
    seed_manifest.write_text(
        yaml.safe_dump(_scenario_seeds(manifest), sort_keys=True), encoding="utf-8"
    )
    stage = {
        "scenario_matrix": execution["scenario_matrix"],
        "seed_manifest": str(seed_manifest.relative_to(REPO_ROOT)),
        "benchmark_profile": execution.get("benchmark_profile", "experimental"),
        "horizon": int(horizon if horizon is not None else execution.get("horizon", 500)),
        "dt": float(execution.get("dt", 0.1)),
        "workers": int(workers if workers is not None else execution.get("workers", 1)),
        "requires_slurm": False,
        "paper_facing": False,
    }
    funnel = {
        "stage_order": ["full_matrix"],
        "stages": {"full_matrix": stage},
    }
    funnel_path = output_dir / "funnel.yaml"
    funnel_path.write_text(yaml.safe_dump(funnel, sort_keys=False), encoding="utf-8")
    return funnel_path


def _run_candidate(
    *,
    candidate: str,
    output_dir: Path,
    funnel_path: Path,
    registry_path: Path,
    workers: int | None,
    horizon: int | None,
) -> dict[str, Any]:
    """Run one candidate through the generated funnel."""
    run_dir = output_dir / "candidates" / candidate
    command = [
        sys.executable,
        "scripts/validation/run_policy_search_candidate.py",
        "--candidate",
        candidate,
        "--stage",
        "full_matrix",
        "--candidate-registry",
        str(registry_path),
        "--funnel-config",
        str(funnel_path),
        "--output-dir",
        str(run_dir),
        "--allow-expensive-stage",
    ]
    if workers is not None:
        command.extend(["--workers", str(workers)])
    if horizon is not None:
        command.extend(["--horizon", str(horizon)])
    completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
    summary_path = run_dir / "summary.json"
    row: dict[str, Any] = {
        "candidate": candidate,
        "command": ["python", *command[1:]],
        "exit_code": int(completed.returncode),
        "summary_path": str(summary_path.relative_to(REPO_ROOT)),
    }
    if completed.returncode != 0 or not summary_path.exists():
        row["status"] = "failed"
        return row
    summary_doc = _load_json(summary_path)
    summary = summary_doc.get("summary")
    batch = summary_doc.get("batch_summary")
    if not isinstance(summary, dict):
        summary = {}
    if not isinstance(batch, dict):
        batch = {}
    family_runs = summary_doc.get("family_runs")
    if not batch and isinstance(family_runs, dict):
        family_batches = [
            run.get("batch_summary")
            for run in family_runs.values()
            if isinstance(run, dict) and isinstance(run.get("batch_summary"), dict)
        ]
        batch = {
            "total_jobs": sum(int(item.get("total_jobs", 0) or 0) for item in family_batches),
            "written": sum(int(item.get("written", 0) or 0) for item in family_batches),
            "failed_jobs": sum(int(item.get("failed_jobs", 0) or 0) for item in family_batches),
        }
    runtime_sec = _num(summary.get("runtime_sec"))
    if runtime_sec is None and isinstance(family_runs, dict):
        runtime_sec = sum(
            _num(run.get("summary", {}).get("runtime_sec")) or 0.0
            for run in family_runs.values()
            if isinstance(run, dict) and isinstance(run.get("summary"), dict)
        )
        summary["runtime_sec"] = runtime_sec
    row.update(
        {
            "status": "ok",
            "decision": summary_doc.get("decision"),
            "episodes": summary.get("episodes"),
            "total_jobs": batch.get("total_jobs"),
            "written": batch.get("written"),
            "failed_jobs": batch.get("failed_jobs"),
            "summary": {metric: summary.get(metric) for metric in METRICS},
        }
    )
    return row


def _metric_delta(a: Any, b: Any) -> float | None:
    """Return numeric a-b when both inputs are numeric."""
    if isinstance(a, int | float) and isinstance(b, int | float):
        return float(a) - float(b)
    return None


def _effect_rows(
    *,
    comparisons: list[dict[str, Any]],
    row_candidates: dict[str, str],
    results: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build compact effect rows from candidate summaries."""
    rows: list[dict[str, Any]] = []
    for comparison in comparisons:
        a_key = str(comparison["a"])
        b_key = str(comparison["b"])
        a_candidate = row_candidates[a_key]
        b_candidate = row_candidates[b_key]
        a_result = results.get(a_candidate, {})
        b_result = results.get(b_candidate, {})
        row: dict[str, Any] = {
            "comparison_id": str(comparison["id"]),
            "a": a_key,
            "b": b_key,
            "a_candidate": a_candidate,
            "b_candidate": b_candidate,
            "interpretation": str(comparison.get("interpretation", "")),
            "status": "ok"
            if a_result.get("status") == "ok" and b_result.get("status") == "ok"
            else "not_available",
        }
        a_summary = a_result.get("summary") if isinstance(a_result.get("summary"), dict) else {}
        b_summary = b_result.get("summary") if isinstance(b_result.get("summary"), dict) else {}
        for metric in METRICS:
            row[f"{metric}_a"] = a_summary.get(metric)
            row[f"{metric}_b"] = b_summary.get(metric)
            row[f"{metric}_delta"] = _metric_delta(a_summary.get(metric), b_summary.get(metric))
        rows.append(row)
    return rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--candidate-registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--comparison-id", action="append")
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run selected one-factor comparisons."""
    args = parse_args(argv)
    manifest_path = _repo_path(args.manifest)
    registry_path = _repo_path(args.candidate_registry)
    output_dir = _repo_path(args.output_dir)
    manifest = _load_yaml(manifest_path)
    row_candidates = _row_candidates(manifest)
    comparisons = _selected_comparisons(
        manifest, set(args.comparison_id) if args.comparison_id else None
    )
    needed_candidates = sorted(
        {row_candidates[str(comp["a"])] for comp in comparisons}
        | {row_candidates[str(comp["b"])] for comp in comparisons}
    )
    funnel_path = _write_local_funnel(
        manifest=manifest, output_dir=output_dir, horizon=args.horizon, workers=args.workers
    )
    results: dict[str, dict[str, Any]] = {}
    if not args.dry_run:
        for candidate in needed_candidates:
            results[candidate] = _run_candidate(
                candidate=candidate,
                output_dir=output_dir,
                funnel_path=funnel_path,
                registry_path=registry_path,
                workers=args.workers,
                horizon=args.horizon,
            )
    payload = {
        "schema_version": "robot-sf-one-factor-ablation-pilot.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "manifest": str(manifest_path.relative_to(REPO_ROOT)),
        "claim_boundary": "diagnostic-only",
        "dry_run": bool(args.dry_run),
        "selected_comparisons": [str(comp["id"]) for comp in comparisons],
        "needed_candidates": needed_candidates,
        "funnel_path": str(funnel_path.relative_to(REPO_ROOT)),
        "candidate_results": results,
        "effect_rows": (
            []
            if args.dry_run
            else _effect_rows(
                comparisons=comparisons, row_candidates=row_candidates, results=results
            )
        ),
    }
    summary_path = output_dir / "one_factor_ablation_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.dry_run:
        return 0
    return 0 if all(row.get("status") == "ok" for row in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
