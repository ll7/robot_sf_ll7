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
    """Parse command-line arguments for one candidate-stage run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument(
        "--stage",
        required=True,
        choices=(
            "smoke",
            "amv_actuation_smoke",
            "nominal_sanity",
            "stress_slice",
            "full_matrix",
            "leader_collision_slice_h500",
            "full_matrix_h500",
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
    """Load a YAML mapping from disk.

    Returns:
        Parsed YAML mapping, defaulting empty files to an empty dict.
    """
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"Expected YAML mapping: {path}")
    return payload


def _resolve_path(anchor: Path, raw: str | Path | None) -> Path | None:
    """Resolve a path relative to an anchor or the repository root.

    Returns:
        Resolved path, or ``None`` when no raw path is configured.
    """
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
    """Recursively merge override values into a copied base mapping.

    Returns:
        Deep-merged mapping without mutating the inputs.
    """
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
    """Load one registered candidate and return its merged runtime config."""
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
    """Group loaded scenario mappings by inferred policy-search family."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for scenario in scenarios:
        family = infer_scenario_family(scenario)
        grouped.setdefault(family, []).append(dict(scenario))
    return grouped


def _scenario_id(scenario: Mapping[str, Any]) -> str:
    """Return the stable scenario identifier used for narrow config overrides."""
    return str(
        scenario.get("name") or scenario.get("scenario_id") or scenario.get("id") or "unknown"
    )


def _filter_scenarios(
    scenarios: list[dict[str, Any]],
    scenario_filter: list[str] | None,
) -> list[dict[str, Any]]:
    """Return only scenarios named by a stage filter when one is configured."""
    if not scenario_filter:
        return scenarios
    allowed = {str(item) for item in scenario_filter}
    filtered = [scenario for scenario in scenarios if _scenario_id(scenario) in allowed]
    if not filtered:
        raise ValueError(f"Stage scenario_filter matched no scenarios: {sorted(allowed)}")
    return filtered


def _effective_candidate_config_for_scenario(
    candidate_payload: Mapping[str, Any],
    candidate_config: Mapping[str, Any],
    scenario: Mapping[str, Any],
) -> dict[str, Any]:
    """Return candidate config after family and scenario-specific overrides."""
    effective = deepcopy(dict(candidate_config))
    family_overrides = candidate_payload.get("family_overrides")
    if isinstance(family_overrides, dict):
        family_cfg = family_overrides.get(infer_scenario_family(dict(scenario)), {})
        if isinstance(family_cfg, dict):
            effective = _deep_merge(effective, family_cfg)

    scenario_overrides = candidate_payload.get("scenario_overrides")
    if isinstance(scenario_overrides, dict):
        scenario_cfg = scenario_overrides.get(_scenario_id(scenario), {})
        if isinstance(scenario_cfg, dict):
            effective = _deep_merge(effective, scenario_cfg)
    return effective


def _scenario_algo_override_config(
    override: Mapping[str, Any],
    *,
    config_anchor: Path,
) -> dict[str, Any]:
    """Return a merged config for one scenario-specific algorithm override."""
    base_cfg: dict[str, Any] = {}
    base_path = _resolve_path(config_anchor, override.get("base_config_path"))
    if base_path is not None:
        base_cfg = _load_yaml(base_path)
    params = override.get("params")
    if params is None:
        params = {}
    if not isinstance(params, dict):
        raise TypeError("scenario_algo_overrides params must be a mapping")
    return _deep_merge(base_cfg, params)


def _effective_candidate_runtime_for_scenario(
    candidate_payload: Mapping[str, Any],
    candidate_config: Mapping[str, Any],
    scenario: Mapping[str, Any],
    *,
    default_algo: str,
    config_anchor: Path,
) -> tuple[str, dict[str, Any]]:
    """Return `(algo, config)` after scenario-level algorithm/config overrides."""
    algo_overrides = candidate_payload.get("scenario_algo_overrides")
    if isinstance(algo_overrides, dict):
        override = algo_overrides.get(_scenario_id(scenario))
        if isinstance(override, dict):
            algo = str(override.get("algo", default_algo)).strip().lower()
            if not algo:
                raise ValueError(
                    f"Scenario algo override is missing algo: {_scenario_id(scenario)}"
                )
            return algo, _scenario_algo_override_config(override, config_anchor=config_anchor)

    return (
        default_algo,
        _effective_candidate_config_for_scenario(candidate_payload, candidate_config, scenario),
    )


def _prepare_scenarios_for_inline_run(
    scenarios: list[Mapping[str, Any]],
    *,
    scenario_root: Path,
) -> list[dict[str, Any]]:
    """Convert scenario asset references to absolute paths for inline runs.

    Returns:
        Prepared scenario mappings with map and route override files resolved.
    """
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


def _group_scenarios_by_config_overrides(
    scenarios: list[Mapping[str, Any]],
    *,
    candidate_payload: Mapping[str, Any],
    candidate_config: Mapping[str, Any],
    default_algo: str,
    config_anchor: Path,
) -> dict[str, dict[str, Any]]:
    """Group scenarios by the effective candidate config they require."""
    scenario_overrides = candidate_payload.get("scenario_overrides")
    algo_overrides = candidate_payload.get("scenario_algo_overrides")
    grouped: dict[str, dict[str, Any]] = {}
    for raw_scenario in scenarios:
        scenario = dict(raw_scenario)
        family = infer_scenario_family(scenario)
        scenario_key = _scenario_id(scenario)
        scenario_has_algo_override = (
            isinstance(algo_overrides, dict) and scenario_key in algo_overrides
        )
        scenario_has_override = scenario_has_algo_override or (
            isinstance(scenario_overrides, dict) and scenario_key in scenario_overrides
        )
        tag = f"{family}__{scenario_key}" if scenario_has_override else family
        if tag not in grouped:
            algo, effective_config = _effective_candidate_runtime_for_scenario(
                candidate_payload,
                candidate_config,
                scenario,
                default_algo=default_algo,
                config_anchor=config_anchor,
            )
            grouped[tag] = {
                "algo": algo,
                "config": effective_config,
                "scenarios": [],
            }
        grouped[tag]["scenarios"].append(scenario)
    return grouped


def decide_stage_status(stage_name: str, stage_cfg: dict[str, Any], summary: dict[str, Any]) -> str:
    """Convert a stage summary into the registry/report decision label."""
    episodes = int(summary.get("episodes") or 0)
    exclusions = summary.get("scenario_exclusions")
    if isinstance(exclusions, Mapping) and episodes > 0:
        excluded_count = int(exclusions.get("count", 0) or 0)
        if excluded_count >= episodes:
            return "excluded"
    if stage_name == "smoke" or stage_name.endswith("_smoke"):
        return "pass" if episodes > 0 else "revise"
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
    """Read the current git commit for provenance metadata.

    Returns:
        Current HEAD SHA, or ``None`` when git is unavailable.
    """
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
    stage_matrix: Path,
    seed_manifest: Path | None,
    seed_list: list[int] | None = None,
    scenario_filter: list[str] | None = None,
) -> Path | list[dict[str, Any]]:
    """Resolve the scenario surface for a stage, including explicit seed overrides."""
    if seed_manifest is None:
        if seed_list or scenario_filter:
            scenarios = _filter_scenarios(load_scenarios(stage_matrix), scenario_filter)
            base_dir = stage_matrix.parent.resolve()
            prepared = _prepare_scenarios_for_inline_run(scenarios, scenario_root=base_dir)
            if seed_list:
                for scenario in prepared:
                    scenario["seeds"] = list(seed_list)
            return prepared
        return stage_matrix
    manifest = load_seed_manifest(seed_manifest)
    return _filter_scenarios(make_subset_scenarios(stage_matrix, manifest), scenario_filter)


def _stage_scenario_filter(stage_cfg: Mapping[str, Any]) -> list[str] | None:
    """Return a normalized stage scenario filter."""
    scenario_filter_raw = stage_cfg.get("scenario_filter")
    if isinstance(scenario_filter_raw, list) and scenario_filter_raw:
        return [str(item) for item in scenario_filter_raw]
    return None


def _stage_synthetic_actuation_profile(
    stage_cfg: Mapping[str, Any],
    *,
    stage_name: str,
) -> dict[str, Any] | None:
    """Return a validated synthetic-actuation profile payload for a stage."""
    synthetic_actuation_profile = stage_cfg.get("synthetic_actuation_profile")
    if synthetic_actuation_profile is None:
        return None
    if not isinstance(synthetic_actuation_profile, dict):
        raise TypeError(f"Stage synthetic_actuation_profile must be a mapping: {stage_name}")
    return synthetic_actuation_profile


def _load_records(path: Path) -> list[dict[str, Any]]:
    """Load policy-search episode records from JSONL.

    Returns:
        List of JSON object records, skipping blank lines.
    """
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _write_records(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write episode records as sorted-key JSONL."""
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def _optional_float(value: Any) -> float | None:
    """Return a float value when one can be parsed."""
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _planner_last_decision(row: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the planner runtime last-decision mapping from an episode row."""
    metadata = row.get("algorithm_metadata")
    if not isinstance(metadata, Mapping):
        return {}
    runtime = metadata.get("planner_runtime")
    if not isinstance(runtime, Mapping):
        return {}
    last_decision = runtime.get("last_decision")
    return last_decision if isinstance(last_decision, Mapping) else {}


def _first_dynamic_collision_example(last_decision: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the first dynamic-collision rejected-candidate diagnostic."""
    examples = last_decision.get("rejected_examples")
    if not isinstance(examples, list):
        return {}
    for example in examples:
        if isinstance(example, Mapping) and example.get("reason") == "dynamic_collision":
            return example
    return {}


def _is_initial_overlap_collision(row: Mapping[str, Any]) -> bool:
    """Return whether a row proves a collision was already unavoidable at step one."""
    metrics = row.get("metrics")
    metrics = metrics if isinstance(metrics, Mapping) else {}
    last_decision = _planner_last_decision(row)
    example = _first_dynamic_collision_example(last_decision)
    min_clearance = _optional_float(metrics.get("min_clearance"))
    ped_collisions = _optional_float(metrics.get("ped_collision_count")) or 0.0
    avg_speed = _optional_float(metrics.get("avg_speed"))
    nearest_ped = _optional_float(last_decision.get("nearest_pedestrian_distance"))
    collision_radius = _optional_float(example.get("collision_radius"))
    rejection_counts = last_decision.get("rejection_counts")
    dynamic_rejections = (
        int(rejection_counts.get("dynamic_collision", 0))
        if isinstance(rejection_counts, Mapping)
        else 0
    )
    return (
        str(row.get("termination_reason", "")).strip().lower() == "collision"
        and int(row.get("steps", 0) or 0) <= 1
        and min_clearance is not None
        and min_clearance < 0.0
        and ped_collisions > 0.0
        and avg_speed == 0.0
        and str(last_decision.get("planner_mode", "")).strip() == "EMERGENCY_STOP"
        and str(last_decision.get("selected_source", "")).strip() == "all_candidates_rejected"
        and dynamic_rejections > 0
        and nearest_ped is not None
        and collision_radius is not None
        and nearest_ped < collision_radius
    )


def _annotate_initial_overlap_exclusions(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Annotate reset-time robot/pedestrian overlaps as explicit scenario exclusions.

    The raw episode remains a collision. The additive ``scenario_exclusion`` block lets
    policy-search summaries report an evidence-adjusted view when the first state is already
    geometrically impossible for any first-step planner command to repair.
    """
    annotated: list[dict[str, Any]] = []
    for row in rows:
        updated = deepcopy(row)
        if "scenario_exclusion" not in updated and _is_initial_overlap_collision(row):
            metrics = row.get("metrics")
            metrics = metrics if isinstance(metrics, Mapping) else {}
            last_decision = _planner_last_decision(row)
            example = _first_dynamic_collision_example(last_decision)
            min_clearance = _optional_float(metrics.get("min_clearance"))
            nearest_ped = _optional_float(last_decision.get("nearest_pedestrian_distance"))
            collision_radius = _optional_float(example.get("collision_radius"))
            updated["scenario_exclusion"] = {
                "status": "impossible",
                "reason": "initial_robot_pedestrian_overlap",
                "evidence": [
                    "first_step_collision_with_zero_progress",
                    f"min_clearance_m={min_clearance:.4f}",
                    f"nearest_pedestrian_distance_m={nearest_ped:.4f}",
                    f"candidate_collision_radius_m={collision_radius:.4f}",
                    "all_first_step_candidates_rejected_for_dynamic_collision",
                ],
            }
        annotated.append(updated)
    return annotated


def _apply_stage_exclusions(
    *,
    stage_cfg: Mapping[str, Any],
    summary_payload: dict[str, Any],
) -> dict[str, Any]:
    """Apply explicit stage-level exclusion annotations to records and summaries."""
    if not bool(stage_cfg.get("fail_closed_initial_overlap_exclusion", False)):
        return summary_payload
    rows = summary_payload.get("records")
    if not isinstance(rows, list):
        return summary_payload
    annotated = _annotate_initial_overlap_exclusions(rows)
    if annotated == rows:
        return summary_payload
    jsonl_path = Path(str(summary_payload["jsonl_path"]))
    _write_records(jsonl_path, annotated)
    updated = dict(summary_payload)
    updated["records"] = annotated
    updated["summary"] = summarize_policy_search_records(annotated)
    return updated


def _run_stage_eval(  # noqa: PLR0913
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
    synthetic_actuation_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one stage evaluation and collect records plus summary metadata.

    Returns:
        Mapping containing episode records, aggregate summary, artifact paths,
        and the underlying batch summary.
    """
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
        synthetic_actuation_profile=synthetic_actuation_profile,
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
    """Compare stage summary metrics against configured baselines.

    Returns:
        Mapping from baseline name to available metric deltas.
    """
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
    """Format paths for Markdown reports relative to the repository when possible.

    Returns:
        Repository-relative path, absolute/external path, or ``n/a``.
    """
    if path is None:
        return "n/a"
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _format_optional_float(value: Any) -> str:
    """Format optional numeric report fields without coupling unrelated columns."""
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "n/a"


def _format_signed_optional_float(value: Any) -> str:
    """Format optional signed delta fields while preserving unavailable metrics."""
    if value is None:
        return "n/a"
    try:
        return f"{float(value):+.4f}"
    except (TypeError, ValueError):
        return "n/a"


def _evaluation_scope_lines(
    *,
    stage_name: str,
    candidate_payload: Mapping[str, Any],
    stage_cfg: Mapping[str, Any],
    stage_matrix: Path,
    seed_manifest: Path | None,
    summary_json_path: Path,
    git_hash: str | None,
) -> list[str]:
    """Return Markdown lines that describe one candidate-stage run scope."""
    lines = [
        "## Evaluation Scope",
        "",
        f"- Stage: `{stage_name}`",
        f"- Algorithm: `{candidate_payload.get('algo', 'unknown')}`",
        f"- Scenario matrix: `{_display_path(stage_matrix)}`",
    ]
    scenario_filter = stage_cfg.get("scenario_filter")
    if isinstance(scenario_filter, list) and scenario_filter:
        lines.append(f"- Scenario filter: `{', '.join(str(item) for item in scenario_filter)}`")
    lines.extend(
        [
            f"- Seed manifest: `{_display_path(seed_manifest)}`"
            if seed_manifest is not None
            else "- Seed manifest: `suite default`",
            f"- Summary JSON: `{_display_path(summary_json_path)}`",
            f"- Git commit: `{git_hash}`" if git_hash else "- Git commit: `unavailable`",
        ]
    )
    synthetic_profile = stage_cfg.get("synthetic_actuation_profile")
    if isinstance(synthetic_profile, dict):
        profile_name = str(synthetic_profile.get("name", "unknown"))
        claim_scope = str(synthetic_profile.get("claim_scope", "unknown"))
        lines.append(
            f"- Synthetic actuation profile: `{profile_name}` (`{claim_scope}`, diagnostic-only)"
        )
    return lines


def _actuation_diagnostics_lines(summary: Mapping[str, Any]) -> list[str]:
    """Return optional Markdown rows for synthetic actuation diagnostics."""
    actuation_summary_raw = summary.get("synthetic_actuation")
    actuation_summary = actuation_summary_raw if isinstance(actuation_summary_raw, dict) else {}
    if not any(value is not None for value in actuation_summary.values()):
        return []
    return [
        "",
        "## Synthetic Actuation Diagnostics",
        "",
        "| Command Clip | Yaw Saturation | Signed Braking Peak |",
        "|---:|---:|---:|",
        f"| {_format_optional_float(actuation_summary.get('command_clip_fraction_mean'))} | "
        f"{_format_optional_float(actuation_summary.get('yaw_rate_saturation_fraction_mean'))} | "
        f"{_format_optional_float(actuation_summary.get('signed_braking_peak_m_s2_mean'))} |",
    ]


def _scenario_exclusion_lines(summary: Mapping[str, Any]) -> list[str]:
    """Return Markdown rows for explicit scenario exclusions."""
    exclusions_raw = summary.get("scenario_exclusions")
    exclusions = exclusions_raw if isinstance(exclusions_raw, Mapping) else {}
    records_raw = exclusions.get("records")
    records = records_raw if isinstance(records_raw, list) else []
    if not records:
        return []
    lines = [
        "",
        "## Scenario Exclusions",
        "",
        "| Scenario | Seed | Status | Reason | Evidence |",
        "|---|---:|---|---|---|",
    ]
    for record_raw in records:
        record = record_raw if isinstance(record_raw, Mapping) else {}
        scenario_id = record.get("scenario_id")
        scenario_id = scenario_id if scenario_id is not None else "unknown"
        seed = record.get("seed")
        seed = seed if seed is not None else "n/a"
        status = record.get("status")
        status = status if status is not None else "unknown"
        reason = record.get("reason")
        reason = reason if reason is not None else "unknown"
        evidence_raw = record.get("evidence")
        evidence = evidence_raw if isinstance(evidence_raw, list) else []
        lines.append(
            f"| {scenario_id} | {seed} | {status} | {reason} | "
            f"{'; '.join(str(item) for item in evidence)} |"
        )
    return lines


def _diagnostic_claim_boundary(
    candidate_entry: Mapping[str, Any],
    candidate_payload: Mapping[str, Any],
) -> str | None:
    """Return a conservative report caveat for diagnostic-only candidates."""
    params = candidate_payload.get("params")
    params = params if isinstance(params, dict) else {}
    if (
        candidate_entry.get("claim_scope") != "diagnostic_only"
        and params.get("diagnostic_only") is not True
        and params.get("claim_boundary") != "diagnostic_only"
    ):
        return None
    return (
        "This report is diagnostic-only wiring or stage evidence. Treat aggregate metrics and "
        "baseline deltas as arithmetic context, not benchmark-strength evidence for comfort, "
        "near-miss behavior, generalization, or planner superiority."
    )


def _write_markdown_report(  # noqa: C901, PLR0913
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
    """Write the candidate-stage Markdown report.

    Returns:
        Path to the generated report.
    """
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
        *_evaluation_scope_lines(
            stage_name=stage_name,
            candidate_payload=candidate_payload,
            stage_cfg=stage_cfg,
            stage_matrix=stage_matrix,
            seed_manifest=seed_manifest,
            summary_json_path=summary_json_path,
            git_hash=git_hash,
        ),
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
        f"{_format_optional_float(mean_min_distance)} | {_format_optional_float(mean_avg_speed)} |"
    )
    evidence_adjusted_raw = summary.get("evidence_adjusted")
    evidence_adjusted = evidence_adjusted_raw if isinstance(evidence_adjusted_raw, dict) else {}
    if int(evidence_adjusted.get("excluded_episodes", 0) or 0) > 0:
        lines.extend(
            [
                "",
                "## Evidence-Adjusted Results",
                "",
                "| Episodes | Excluded | Success | Collision | Near Miss |",
                "|---:|---:|---:|---:|---:|",
                f"| {evidence_adjusted.get('episodes', 0)} | "
                f"{evidence_adjusted.get('excluded_episodes', 0)} | "
                f"{float(evidence_adjusted.get('success_rate', 0.0)):.4f} | "
                f"{float(evidence_adjusted.get('collision_rate', 0.0)):.4f} | "
                f"{float(evidence_adjusted.get('near_miss_rate', 0.0)):.4f} |",
                "",
                "Raw aggregate results above still include excluded rows; evidence-adjusted results only remove rows with explicit exclusion metadata.",
            ]
        )
    lines.extend(_actuation_diagnostics_lines(summary))
    lines.extend(_scenario_exclusion_lines(summary))
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
    claim_boundary = _diagnostic_claim_boundary(candidate_entry, candidate_payload)
    if claim_boundary is not None:
        lines.extend(["", "## Claim Boundary", "", claim_boundary])
    lines.extend(["", "## Baseline Deltas", ""])
    if deltas:
        if claim_boundary is not None:
            lines.append("_Diagnostic-only arithmetic context; not a benchmark comparison claim._")
            lines.append("")
        lines.append("| Baseline | Success Delta | Collision Delta | Near-Miss Delta |")
        lines.append("|---|---:|---:|---:|")
        for name, row in sorted(deltas.items()):
            lines.append(
                f"| {name} | {_format_signed_optional_float(row.get('success_rate'))} | "
                f"{_format_signed_optional_float(row.get('collision_rate'))} | "
                f"{_format_signed_optional_float(row.get('near_miss_rate'))} |"
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
    """Run the requested candidate stage and emit JSON plus Markdown artifacts."""
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
    seed_list_raw = stage_cfg.get("seed_list")
    seed_list = (
        [int(seed) for seed in seed_list_raw]
        if isinstance(seed_list_raw, list) and seed_list_raw
        else None
    )
    scenario_filter = _stage_scenario_filter(stage_cfg)
    horizon = int(args.horizon if args.horizon is not None else stage_cfg.get("horizon", 120))
    dt = float(args.dt if args.dt is not None else stage_cfg.get("dt", 0.1))
    workers = int(args.workers if args.workers is not None else stage_cfg.get("workers", 1))
    benchmark_profile = str(stage_cfg.get("benchmark_profile", "experimental"))
    synthetic_actuation_profile = _stage_synthetic_actuation_profile(
        stage_cfg,
        stage_name=args.stage,
    )

    output_dir = (
        args.output_dir or Path("output/policy_search") / args.candidate / args.stage / "latest"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    scenarios_or_path = _load_stage_scenarios(
        stage_matrix,
        seed_manifest,
        seed_list,
        scenario_filter,
    )

    family_overrides = candidate_payload.get("family_overrides")
    scenario_overrides = candidate_payload.get("scenario_overrides")
    scenario_algo_overrides = candidate_payload.get("scenario_algo_overrides")
    family_runs: dict[str, Any] = {}
    has_config_overrides = (
        (isinstance(family_overrides, dict) and bool(family_overrides))
        or (isinstance(scenario_overrides, dict) and bool(scenario_overrides))
        or (isinstance(scenario_algo_overrides, dict) and bool(scenario_algo_overrides))
    )
    if has_config_overrides:
        base_scenarios = (
            scenarios_or_path
            if isinstance(scenarios_or_path, list)
            else load_scenarios(Path(scenarios_or_path))
        )
        prepared_scenarios = _prepare_scenarios_for_inline_run(
            [dict(item) for item in base_scenarios],
            scenario_root=stage_matrix.parent.resolve(),
        )
        grouped = _group_scenarios_by_config_overrides(
            prepared_scenarios,
            candidate_payload=candidate_payload,
            candidate_config=candidate_config,
            default_algo=algo,
            config_anchor=candidate_config_path.parent,
        )
        combined_records: list[dict[str, Any]] = []
        for tag, group in sorted(grouped.items()):
            scenarios = group["scenarios"]
            group_algo = group["algo"]
            family_cfg = group["config"]
            run = _run_stage_eval(
                scenarios_or_path=scenarios,
                algo=group_algo,
                algo_cfg=family_cfg,
                out_dir=output_dir,
                tag=f"{args.stage}__{args.candidate}__{tag}",
                horizon=horizon,
                dt=dt,
                workers=workers,
                benchmark_profile=benchmark_profile,
                synthetic_actuation_profile=synthetic_actuation_profile,
            )
            family_runs[tag] = {key: value for key, value in run.items() if key != "records"}
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
            synthetic_actuation_profile=synthetic_actuation_profile,
        )

    summary_payload = _apply_stage_exclusions(
        stage_cfg=stage_cfg,
        summary_payload=summary_payload,
    )
    decision = decide_stage_status(args.stage, stage_cfg, summary_payload["summary"])
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
        "scenario_filter": scenario_filter,
        "benchmark_profile": benchmark_profile,
        "synthetic_actuation_profile": synthetic_actuation_profile,
        "git_hash": git_hash,
        "summary": summary_payload["summary"],
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
        summary=summary_payload["summary"],
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
