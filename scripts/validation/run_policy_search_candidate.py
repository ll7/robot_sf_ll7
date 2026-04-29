#!/usr/bin/env python3
"""Run one policy-search candidate through a selected evaluation stage."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections.abc import Mapping
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from robot_sf.benchmark.map_runner import run_map_batch
from robot_sf.training.scenario_loader import load_scenarios
from scripts.validation.policy_search_common import (
    infer_scenario_family,
    summarize_policy_search_records,
)
from scripts.validation.predictive_eval_common import (
    load_seed_manifest,
    make_subset_scenarios,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

_DEFAULT_BASELINES = Path("configs/policy_search/baselines.yaml")
_DEFAULT_FUNNEL = Path("configs/policy_search/funnel.yaml")
_DEFAULT_REGISTRY = Path("docs/context/policy_search/candidate_registry.yaml")
_DEFAULT_DOCS_ROOT = Path("docs/context/policy_search")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument(
        "--stage",
        required=True,
        choices=(
            "smoke",
            "nominal_sanity",
            "stress_slice",
            "full_matrix",
            "robustness_extension",
        ),
    )
    parser.add_argument("--candidate-registry", type=Path, default=_DEFAULT_REGISTRY)
    parser.add_argument("--funnel-config", type=Path, default=_DEFAULT_FUNNEL)
    parser.add_argument("--baselines", type=Path, default=_DEFAULT_BASELINES)
    parser.add_argument("--docs-root", type=Path, default=_DEFAULT_DOCS_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--dt", type=float)
    parser.add_argument("--allow-expensive-stage", action="store_true")
    return parser.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"Expected YAML mapping: {path}")
    return payload


def _resolve_path(anchor: Path, raw: str | Path | None) -> Path | None:
    if raw is None:
        return None
    path = Path(raw)
    if path.is_absolute():
        return path
    anchor_candidate = (anchor / path).resolve()
    if anchor_candidate.exists():
        return anchor_candidate
    repo_candidate = (REPO_ROOT / path).resolve()
    if repo_candidate.exists():
        return repo_candidate
    return (
        repo_candidate if not path.parts or path.parts[0] not in {".", ".."} else anchor_candidate
    )


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in overrides.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_candidate_definition(
    registry_path: Path,
    candidate_name: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Path]:
    registry = _load_yaml(registry_path)
    candidates = registry.get("candidates")
    if not isinstance(candidates, dict) or candidate_name not in candidates:
        raise KeyError(f"Unknown candidate '{candidate_name}' in {registry_path}")
    entry = candidates[candidate_name]
    if not isinstance(entry, dict):
        raise TypeError(f"Candidate entry must be a mapping: {candidate_name}")
    config_path_raw = entry.get("candidate_config_path")
    if not isinstance(config_path_raw, str) or not config_path_raw.strip():
        raise ValueError(f"Candidate '{candidate_name}' is missing candidate_config_path")
    config_path = _resolve_path(registry_path.parent, config_path_raw)
    if config_path is None:
        raise ValueError(f"Candidate '{candidate_name}' could not resolve config path")
    candidate_payload = _load_yaml(config_path)
    base_cfg: dict[str, Any] = {}
    base_path = _resolve_path(config_path.parent, candidate_payload.get("base_config_path"))
    if base_path is not None:
        base_cfg = _load_yaml(base_path)
    params = candidate_payload.get("params")
    if params is None:
        params = {}
    if not isinstance(params, dict):
        raise TypeError(f"Candidate params must be a mapping: {config_path}")
    return entry, candidate_payload, _deep_merge(base_cfg, params), config_path


def split_scenarios_by_family(
    scenarios: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for scenario in scenarios:
        family = infer_scenario_family(scenario)
        grouped.setdefault(family, []).append(dict(scenario))
    return grouped


def _prepare_scenarios_for_inline_run(
    scenarios: list[Mapping[str, Any]],
    *,
    scenario_root: Path,
) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for scenario in scenarios:
        updated = deepcopy(dict(scenario))
        map_file = updated.get("map_file")
        if isinstance(map_file, str) and map_file.strip():
            map_path = Path(map_file)
            if not map_path.is_absolute():
                updated["map_file"] = str((scenario_root / map_path).resolve())
        route_overrides_file = updated.get("route_overrides_file")
        if isinstance(route_overrides_file, str) and route_overrides_file.strip():
            route_path = Path(route_overrides_file)
            if not route_path.is_absolute():
                updated["route_overrides_file"] = str((scenario_root / route_path).resolve())
        prepared.append(updated)
    return prepared


def decide_stage_status(stage_name: str, stage_cfg: dict[str, Any], summary: dict[str, Any]) -> str:
    if stage_name == "smoke":
        return "pass" if int(summary.get("episodes", 0)) > 0 else "revise"
    gate = stage_cfg.get("gate")
    if not isinstance(gate, dict):
        return "tracked"
    min_success = gate.get("min_success_rate")
    max_collision = gate.get("max_collision_rate")
    if min_success is not None and float(summary.get("success_rate", 0.0)) < float(min_success):
        return "revise"
    if max_collision is not None and float(summary.get("collision_rate", 0.0)) > float(
        max_collision
    ):
        return "revise"
    return "pass"


def _git_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _load_stage_scenarios(
    stage_matrix: Path, seed_manifest: Path | None
) -> Path | list[dict[str, Any]]:
    if seed_manifest is None:
        return stage_matrix
    manifest = load_seed_manifest(seed_manifest)
    return make_subset_scenarios(stage_matrix, manifest)


def _load_records(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _write_records(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def _run_stage_eval(
    *,
    scenarios_or_path: Path | list[dict[str, Any]],
    algo: str,
    algo_cfg: dict[str, Any],
    out_dir: Path,
    tag: str,
    horizon: int,
    dt: float,
    workers: int,
    benchmark_profile: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    algo_cfg_path = out_dir / f"{tag}_algo.yaml"
    jsonl_path = out_dir / f"{tag}.jsonl"
    algo_cfg_path.write_text(yaml.safe_dump(algo_cfg, sort_keys=False), encoding="utf-8")
    if jsonl_path.exists():
        jsonl_path.unlink()

    batch_summary = run_map_batch(
        scenarios_or_path,
        jsonl_path,
        schema_path=Path("robot_sf/benchmark/schemas/episode.schema.v1.json"),
        algo=algo,
        algo_config_path=str(algo_cfg_path),
        horizon=int(horizon),
        dt=float(dt),
        workers=int(workers),
        resume=False,
        benchmark_profile=str(benchmark_profile),
    )
    rows = _load_records(jsonl_path)
    summary = summarize_policy_search_records(rows)
    summary["runtime_sec"] = float(max(time.perf_counter() - started, 0.0))
    return {
        "records": rows,
        "summary": summary,
        "jsonl_path": str(jsonl_path),
        "algo_config_path": str(algo_cfg_path),
        "batch_summary": batch_summary,
    }


def _baseline_deltas(
    summary: Mapping[str, Any], baselines_path: Path
) -> dict[str, dict[str, float]]:
    payload = _load_yaml(baselines_path)
    baselines = payload.get("baselines")
    if not isinstance(baselines, dict):
        return {}
    deltas: dict[str, dict[str, float]] = {}
    for name, baseline in baselines.items():
        if not isinstance(baseline, dict):
            continue
        row: dict[str, float] = {}
        for metric in ("success_rate", "collision_rate", "near_miss_rate"):
            if metric in baseline and summary.get(metric) is not None:
                row[metric] = float(summary.get(metric, 0.0)) - float(baseline[metric])
        if row:
            deltas[str(name)] = row
    return deltas


def _display_path(path: Path | None) -> str:
    if path is None:
        return "n/a"
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _write_markdown_report(
    *,
    docs_root: Path,
    candidate_name: str,
    candidate_entry: Mapping[str, Any],
    candidate_payload: Mapping[str, Any],
    stage_name: str,
    stage_cfg: Mapping[str, Any],
    stage_matrix: Path,
    seed_manifest: Path | None,
    summary: Mapping[str, Any],
    family_runs: Mapping[str, Any],
    decision: str,
    git_hash: str | None,
    baselines_path: Path,
    summary_json_path: Path,
) -> Path:
    docs_dir = docs_root / "reports"
    docs_dir.mkdir(parents=True, exist_ok=True)
    date_prefix = datetime.now(UTC).date().isoformat()
    report_path = docs_dir / f"{date_prefix}_{candidate_name}_{stage_name}.md"
    deltas = _baseline_deltas(summary, baselines_path)
    family_rows_raw = summary.get("scenario_family")
    family_rows = family_rows_raw if isinstance(family_rows_raw, dict) else {}
    lines = [
        f"# Candidate Report: {candidate_name} ({stage_name})",
        "",
        "## Decision",
        "",
        decision,
        "",
        "## Hypothesis",
        "",
        str(candidate_entry.get("hypothesis") or candidate_payload.get("hypothesis") or "n/a"),
        "",
        "## Evaluation Scope",
        "",
        f"- Stage: `{stage_name}`",
        f"- Algorithm: `{candidate_payload.get('algo', 'unknown')}`",
        f"- Scenario matrix: `{_display_path(stage_matrix)}`",
        f"- Seed manifest: `{_display_path(seed_manifest)}`"
        if seed_manifest is not None
        else "- Seed manifest: `suite default`",
        f"- Summary JSON: `{_display_path(summary_json_path)}`",
        f"- Git commit: `{git_hash}`" if git_hash else "- Git commit: `unavailable`",
        "",
        "## Aggregate Results",
        "",
        "| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    mean_min_distance = summary.get("mean_min_distance")
    mean_avg_speed = summary.get("mean_avg_speed")
    lines.append(
        f"| {summary.get('episodes', 0)} | {float(summary.get('success_rate', 0.0)):.4f} | "
        f"{float(summary.get('collision_rate', 0.0)):.4f} | {float(summary.get('near_miss_rate', 0.0)):.4f} | "
        f"{float(mean_min_distance):.4f} | {float(mean_avg_speed):.4f} |"
        if mean_min_distance is not None and mean_avg_speed is not None
        else (
            f"| {summary.get('episodes', 0)} | {float(summary.get('success_rate', 0.0)):.4f} | "
            f"{float(summary.get('collision_rate', 0.0)):.4f} | {float(summary.get('near_miss_rate', 0.0)):.4f} | n/a | n/a |"
        )
    )
    lines.extend(
        [
            "",
            "## Scenario-Family Split",
            "",
            "| Family | Episodes | Success | Collision | Near Miss |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for family, family_summary_raw in sorted(family_rows.items()):
        family_summary = family_summary_raw if isinstance(family_summary_raw, dict) else {}
        lines.append(
            f"| {family} | {family_summary.get('episodes', 0)} | "
            f"{float(family_summary.get('success_rate', 0.0)):.4f} | "
            f"{float(family_summary.get('collision_rate', 0.0)):.4f} | "
            f"{float(family_summary.get('near_miss_rate', 0.0)):.4f} |"
        )
    lines.extend(["", "## Failure Taxonomy", ""])
    failure_counts_raw = summary.get("failure_mode_counts")
    failure_counts = failure_counts_raw if isinstance(failure_counts_raw, dict) else {}
    if failure_counts:
        for key, value in sorted(failure_counts.items()):
            lines.append(f"- `{key}`: `{value}`")
    else:
        lines.append("- No failures recorded.")
    lines.extend(["", "## Baseline Deltas", ""])
    if deltas:
        lines.append("| Baseline | Success Delta | Collision Delta | Near-Miss Delta |")
        lines.append("|---|---:|---:|---:|")
        for name, row in sorted(deltas.items()):
            lines.append(
                f"| {name} | {row.get('success_rate', 0.0):+.4f} | {row.get('collision_rate', 0.0):+.4f} | {row.get('near_miss_rate', 0.0):+.4f} |"
            )
    else:
        lines.append("- Baseline deltas unavailable.")
    if family_runs:
        lines.extend(["", "## Family Override Runs", ""])
        for family, run_raw in sorted(family_runs.items()):
            run = run_raw if isinstance(run_raw, dict) else {}
            family_summary_raw = run.get("summary")
            family_summary = family_summary_raw if isinstance(family_summary_raw, dict) else {}
            lines.append(
                f"- `{family}`: success `{float(family_summary.get('success_rate', 0.0)):.4f}`, collision `{float(family_summary.get('collision_rate', 0.0)):.4f}`"
            )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> int:
    args = parse_args()
    funnel = _load_yaml(args.funnel_config)
    stages = funnel.get("stages")
    if not isinstance(stages, dict) or args.stage not in stages:
        raise KeyError(f"Unknown stage '{args.stage}' in {args.funnel_config}")
    stage_cfg = stages[args.stage]
    if not isinstance(stage_cfg, dict):
        raise TypeError(f"Stage config must be a mapping: {args.stage}")
    if bool(stage_cfg.get("requires_slurm", False)) and not bool(args.allow_expensive_stage):
        raise SystemExit(
            f"Stage '{args.stage}' is marked as expensive. Re-run with --allow-expensive-stage on an appropriate remote/SLURM host."
        )

    candidate_entry, candidate_payload, candidate_config, candidate_config_path = (
        load_candidate_definition(
            args.candidate_registry,
            args.candidate,
        )
    )
    algo = str(candidate_payload.get("algo", "")).strip().lower()
    if not algo:
        raise ValueError(f"Candidate config is missing algo: {candidate_config_path}")

    stage_matrix = _resolve_path(args.funnel_config.parent, stage_cfg.get("scenario_matrix"))
    if stage_matrix is None:
        raise ValueError(f"Stage '{args.stage}' is missing scenario_matrix")
    seed_manifest = _resolve_path(args.funnel_config.parent, stage_cfg.get("seed_manifest"))
    horizon = int(args.horizon if args.horizon is not None else stage_cfg.get("horizon", 120))
    dt = float(args.dt if args.dt is not None else stage_cfg.get("dt", 0.1))
    workers = int(args.workers if args.workers is not None else stage_cfg.get("workers", 1))
    benchmark_profile = str(stage_cfg.get("benchmark_profile", "experimental"))

    output_dir = (
        args.output_dir or Path("output/policy_search") / args.candidate / args.stage / "latest"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    scenarios_or_path = _load_stage_scenarios(stage_matrix, seed_manifest)

    family_overrides = candidate_payload.get("family_overrides")
    family_runs: dict[str, Any] = {}
    if isinstance(family_overrides, dict) and family_overrides:
        base_scenarios = (
            scenarios_or_path
            if isinstance(scenarios_or_path, list)
            else load_scenarios(Path(scenarios_or_path))
        )
        prepared_scenarios = _prepare_scenarios_for_inline_run(
            [dict(item) for item in base_scenarios],
            scenario_root=stage_matrix.parent.resolve(),
        )
        grouped = split_scenarios_by_family(prepared_scenarios)
        combined_records: list[dict[str, Any]] = []
        for family, scenarios in sorted(grouped.items()):
            family_cfg = _deep_merge(candidate_config, family_overrides.get(family, {}))
            run = _run_stage_eval(
                scenarios_or_path=scenarios,
                algo=algo,
                algo_cfg=family_cfg,
                out_dir=output_dir,
                tag=f"{args.stage}__{args.candidate}__{family}",
                horizon=horizon,
                dt=dt,
                workers=workers,
                benchmark_profile=benchmark_profile,
            )
            family_runs[family] = {key: value for key, value in run.items() if key != "records"}
            combined_records.extend(run["records"])
        combined_jsonl = output_dir / f"{args.stage}__{args.candidate}__combined.jsonl"
        _write_records(combined_jsonl, combined_records)
        combined_summary = summarize_policy_search_records(combined_records)
        summary_payload = {
            "records": combined_records,
            "summary": combined_summary,
            "jsonl_path": str(combined_jsonl),
        }
    else:
        summary_payload = _run_stage_eval(
            scenarios_or_path=scenarios_or_path,
            algo=algo,
            algo_cfg=candidate_config,
            out_dir=output_dir,
            tag=f"{args.stage}__{args.candidate}",
            horizon=horizon,
            dt=dt,
            workers=workers,
            benchmark_profile=benchmark_profile,
        )

    stage_summary = summary_payload["summary"]
    decision = decide_stage_status(args.stage, stage_cfg, stage_summary)
    git_hash = _git_hash()
    summary_doc = {
        "generated_at": datetime.now(UTC).isoformat(),
        "candidate": args.candidate,
        "stage": args.stage,
        "algo": algo,
        "candidate_registry": str(args.candidate_registry),
        "candidate_config_path": str(candidate_config_path),
        "scenario_matrix": str(stage_matrix),
        "seed_manifest": str(seed_manifest) if seed_manifest is not None else None,
        "benchmark_profile": benchmark_profile,
        "git_hash": git_hash,
        "summary": stage_summary,
        "decision": decision,
        "jsonl_path": summary_payload["jsonl_path"],
        "family_runs": family_runs,
    }
    summary_json = output_dir / "summary.json"
    summary_json.write_text(json.dumps(summary_doc, indent=2), encoding="utf-8")
    report_path = _write_markdown_report(
        docs_root=args.docs_root,
        candidate_name=args.candidate,
        candidate_entry=candidate_entry,
        candidate_payload=candidate_payload,
        stage_name=args.stage,
        stage_cfg=stage_cfg,
        stage_matrix=stage_matrix,
        seed_manifest=seed_manifest,
        summary=stage_summary,
        family_runs=family_runs,
        decision=decision,
        git_hash=git_hash,
        baselines_path=args.baselines,
        summary_json_path=summary_json,
    )
    print(
        json.dumps(
            {
                "summary": str(summary_json),
                "report": str(report_path),
                "decision": decision,
                "jsonl": str(summary_payload["jsonl_path"]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
