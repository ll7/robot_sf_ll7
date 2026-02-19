"""Camera-ready benchmark campaign orchestration.

This module provides a config-driven workflow to run a planner matrix over a
scenario manifest, generate campaign-level reports, and export a publication
bundle for archival/release pipelines.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from robot_sf.benchmark.aggregate import compute_aggregates_with_ci, read_jsonl
from robot_sf.benchmark.artifact_publication import export_publication_bundle
from robot_sf.benchmark.runner import run_batch
from robot_sf.benchmark.utils import _config_hash, _git_hash_fallback, load_optional_json
from robot_sf.common.artifact_paths import (
    ensure_canonical_tree,
    get_artifact_category_path,
    get_repository_root,
)
from robot_sf.training.scenario_loader import load_scenarios

CAMPAIGN_SCHEMA_VERSION = "benchmark-camera-ready-campaign.v1"
DEFAULT_EPISODE_SCHEMA_PATH = Path("robot_sf/benchmark/schemas/episode.schema.v1.json")
DEFAULT_SEED_SETS_PATH = Path("configs/benchmarks/seed_sets_v1.yaml")

_REPORT_METRICS: tuple[str, ...] = (
    "success",
    "collisions",
    "near_misses",
    "time_to_goal_norm",
    "path_efficiency",
    "comfort_exposure",
    "jerk_mean",
    "snqi",
)


@dataclass(frozen=True)
class SeedPolicy:
    """Seed selection policy for campaign scenarios."""

    mode: str = "scenario-default"
    seed_set: str | None = None
    seeds: tuple[int, ...] = ()
    seed_sets_path: Path = DEFAULT_SEED_SETS_PATH


@dataclass(frozen=True)
class PlannerSpec:
    """One planner entry in a benchmark campaign matrix."""

    key: str
    algo: str
    benchmark_profile: str = "baseline-safe"
    algo_config_path: Path | None = None
    socnav_missing_prereq_policy: str = "fail-fast"
    adapter_impact_eval: bool = False
    workers_override: int | None = None
    horizon_override: int | None = None
    dt_override: float | None = None
    enabled: bool = True


@dataclass(frozen=True)
class CampaignConfig:
    """Top-level camera-ready benchmark campaign config."""

    name: str
    scenario_matrix_path: Path
    planners: tuple[PlannerSpec, ...]
    seed_policy: SeedPolicy = SeedPolicy()
    workers: int = 1
    horizon: int | None = None
    dt: float | None = None
    record_forces: bool = True
    resume: bool = True
    bootstrap_samples: int = 400
    bootstrap_confidence: float = 0.95
    bootstrap_seed: int = 123
    snqi_weights_path: Path | None = None
    snqi_baseline_path: Path | None = None
    stop_on_failure: bool = False
    export_publication_bundle: bool = True
    include_videos_in_publication: bool = False
    overwrite_publication_bundle: bool = True
    repository_url: str = "https://github.com/ll7/robot_sf_ll7"
    release_tag: str = "{release_tag}"
    doi: str = "10.5281/zenodo.<record-id>"
    paper_interpretation_profile: str = "baseline-ready-core"


def _repo_relative(path: Path) -> str:
    """Return a repository-relative path when possible."""
    path_resolved = path.resolve()
    repo_root = get_repository_root().resolve()
    try:
        return path_resolved.relative_to(repo_root).as_posix()
    except ValueError:
        return str(path_resolved)


def _utc_now() -> str:
    """Return an ISO-8601 UTC timestamp with trailing ``Z``."""
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _hash_payload(payload: Any) -> str:
    """Compute a deterministic SHA1 short hash for a JSON-serializable payload.

    Returns:
        Twelve-character SHA1 digest prefix.
    """
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:12]


def _jsonable(value: Any) -> Any:
    """Convert nested values into JSON-serializable primitives.

    Returns:
        JSON-serializable value with ``Path`` objects converted to strings.
    """
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _sanitize_name(name: str) -> str:
    """Normalize names for stable directory identifiers.

    Returns:
        Lowercase identifier containing only letters, digits, underscores, and hyphens.
    """
    normalized = re.sub(r"[^a-zA-Z0-9_-]+", "_", name.strip().lower()).strip("_")
    return normalized or "campaign"


def _load_seed_sets(path: Path) -> dict[str, list[int]]:
    """Load seed sets file into a normalized mapping.

    Returns:
        Mapping from seed-set name to integer seed list.
    """
    if not path.exists():
        raise FileNotFoundError(f"Seed sets file not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Seed sets file must be a mapping: {path}")
    out: dict[str, list[int]] = {}
    for key, value in payload.items():
        if isinstance(value, list) and value:
            out[str(key)] = [int(seed) for seed in value]
    return out


def _resolve_seed_override(policy: SeedPolicy) -> list[int] | None:
    """Resolve seed override list based on campaign seed policy.

    Returns:
        Seed list override, or ``None`` when scenario-defined seeds should be used.
    """
    mode = policy.mode.strip().lower()
    if mode == "scenario-default":
        return None
    if mode == "fixed-list":
        if not policy.seeds:
            raise ValueError("Seed policy mode 'fixed-list' requires a non-empty seeds list")
        return [int(seed) for seed in policy.seeds]
    if mode == "seed-set":
        if not policy.seed_set:
            raise ValueError("Seed policy mode 'seed-set' requires seed_set")
        seed_sets = _load_seed_sets(policy.seed_sets_path)
        if policy.seed_set not in seed_sets:
            known = ", ".join(sorted(seed_sets))
            raise ValueError(
                f"Unknown seed set '{policy.seed_set}'. Available: {known}",
            )
        return list(seed_sets[policy.seed_set])
    raise ValueError(f"Unsupported seed policy mode: {policy.mode}")


def _load_campaign_scenarios(cfg: CampaignConfig) -> list[dict[str, Any]]:
    """Load campaign scenarios and apply optional seed override.

    Returns:
        Scenario list consumable by benchmark runners.
    """
    scenarios = load_scenarios(
        cfg.scenario_matrix_path,
        base_dir=cfg.scenario_matrix_path.parent,
    )
    scenario_dicts = [dict(scenario) for scenario in scenarios]
    matrix_root = cfg.scenario_matrix_path.parent
    normalized: list[dict[str, Any]] = []
    repo_root = get_repository_root().resolve()
    for scenario in scenario_dicts:
        patched = dict(scenario)
        map_file = patched.get("map_file")
        if isinstance(map_file, str):
            map_path = Path(map_file)
            if map_path.is_absolute():
                try:
                    patched["map_file"] = map_path.resolve().relative_to(repo_root).as_posix()
                except ValueError:
                    patched["map_file"] = map_path.resolve().as_posix()
            else:
                candidate = (matrix_root / map_path).resolve()
                if candidate.exists():
                    try:
                        patched["map_file"] = candidate.relative_to(repo_root).as_posix()
                    except ValueError:
                        patched["map_file"] = candidate.as_posix()
        normalized.append(patched)

    scenario_dicts = normalized
    seeds_override = _resolve_seed_override(cfg.seed_policy)
    if seeds_override is None:
        return scenario_dicts

    seeded: list[dict[str, Any]] = []
    for scenario in scenario_dicts:
        patched = dict(scenario)
        patched["seeds"] = list(seeds_override)
        seeded.append(patched)
    return seeded


def _resolved_seed_inventory(scenarios: list[dict[str, Any]]) -> list[int]:
    """Return sorted unique seed values actually present in campaign scenarios.

    Returns:
        Sorted list of unique integer seeds.
    """
    seeds: set[int] = set()
    for scenario in scenarios:
        scenario_seeds = scenario.get("seeds")
        if not isinstance(scenario_seeds, list):
            continue
        for value in scenario_seeds:
            try:
                seeds.add(int(value))
            except (TypeError, ValueError):
                continue
    return sorted(seeds)


def _metric_mean(block: dict[str, Any], metric: str) -> float:
    """Extract aggregate mean value for one metric.

    Returns:
        Mean metric value, or ``nan`` when unavailable.
    """
    metric_block = block.get(metric)
    if not isinstance(metric_block, dict):
        return float("nan")
    value = metric_block.get("mean")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _metric_ci(block: dict[str, Any], metric: str) -> tuple[float, float]:
    """Extract mean confidence interval values for one metric.

    Returns:
        Tuple ``(low, high)`` with ``nan`` values when unavailable.
    """
    metric_block = block.get(metric)
    if not isinstance(metric_block, dict):
        return (float("nan"), float("nan"))
    ci = metric_block.get("mean_ci")
    if not isinstance(ci, list) or len(ci) != 2:
        return (float("nan"), float("nan"))
    try:
        return (float(ci[0]), float(ci[1]))
    except (TypeError, ValueError):
        return (float("nan"), float("nan"))


def _safe_float(value: float) -> str:
    """Format a float for report tables with NaN handling.

    Returns:
        Fixed-precision string or ``\"nan\"``.
    """
    if math.isnan(value):
        return "nan"
    return f"{value:.4f}"


def _git_context() -> dict[str, str]:
    """Collect lightweight git metadata for campaign provenance.

    Returns:
        Mapping with ``commit``, ``branch``, and ``remote`` fields.
    """

    def _run(args: list[str]) -> str:
        try:
            out = subprocess.check_output(args, stderr=subprocess.DEVNULL)
            return out.decode("utf-8", errors="replace").strip()
        except (subprocess.CalledProcessError, FileNotFoundError, OSError):
            return "unknown"

    return {
        "commit": _git_hash_fallback(),
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "remote": _run(["git", "config", "--get", "remote.origin.url"]),
    }


def _campaign_id(cfg: CampaignConfig, *, label: str | None = None) -> str:
    """Build a unique campaign identifier from config name and wall-clock timestamp.

    Returns:
        Campaign identifier used for output directories and manifests.
    """
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = _sanitize_name(cfg.name)
    if label:
        suffix = _sanitize_name(label)
        return f"{base}_{suffix}_{stamp}"
    return f"{base}_{stamp}"


def _resolve_path(raw_path: str | None, *, base_dir: Path) -> Path | None:
    """Resolve optional paths relative to ``base_dir``.

    Returns:
        Absolute resolved path, or ``None`` when no path was provided.
    """
    if not raw_path:
        return None
    path = Path(raw_path)
    if path.is_absolute():
        return path

    candidate = (base_dir / path).resolve()
    if candidate.exists():
        return candidate

    repo_candidate = (get_repository_root() / path).resolve()
    if repo_candidate.exists():
        return repo_candidate

    return candidate


def load_campaign_config(path: Path) -> CampaignConfig:
    """Load and validate a camera-ready benchmark campaign YAML config.

    Returns:
        Parsed campaign configuration dataclass.
    """
    config_path = path.resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Campaign config must be a mapping: {config_path}")

    name = str(payload.get("name") or config_path.stem)
    matrix_raw = payload.get("scenario_matrix")
    if not isinstance(matrix_raw, str) or not matrix_raw.strip():
        raise ValueError("Campaign config requires a non-empty 'scenario_matrix' string")
    scenario_matrix_path = _resolve_path(matrix_raw, base_dir=config_path.parent)
    if scenario_matrix_path is None:
        raise FileNotFoundError(
            f"Could not resolve scenario_matrix '{matrix_raw}' from config '{config_path}'.",
        )

    planners_raw = payload.get("planners")
    if not isinstance(planners_raw, list) or not planners_raw:
        raise ValueError("Campaign config requires a non-empty 'planners' list")

    planner_specs: list[PlannerSpec] = []
    for entry in planners_raw:
        if not isinstance(entry, dict):
            raise ValueError("Each planners entry must be a mapping")
        key = str(entry.get("key") or entry.get("algo") or "").strip()
        algo = str(entry.get("algo") or "").strip()
        if not key or not algo:
            raise ValueError("Planner entry requires non-empty key and algo")
        planner_specs.append(
            PlannerSpec(
                key=key,
                algo=algo,
                benchmark_profile=str(entry.get("benchmark_profile", "baseline-safe")),
                algo_config_path=_resolve_path(
                    entry.get("algo_config"), base_dir=config_path.parent
                ),
                socnav_missing_prereq_policy=str(
                    entry.get("socnav_missing_prereq_policy", "fail-fast"),
                ),
                adapter_impact_eval=bool(entry.get("adapter_impact_eval", False)),
                workers_override=(
                    int(entry["workers"]) if entry.get("workers") is not None else None
                ),
                horizon_override=(
                    int(entry["horizon"]) if entry.get("horizon") is not None else None
                ),
                dt_override=(float(entry["dt"]) if entry.get("dt") is not None else None),
                enabled=bool(entry.get("enabled", True)),
            ),
        )

    seed_policy_raw = (
        payload.get("seed_policy") if isinstance(payload.get("seed_policy"), dict) else {}
    )
    mode = str(seed_policy_raw.get("mode", "scenario-default"))
    seed_set = seed_policy_raw.get("seed_set")
    seeds = seed_policy_raw.get("seeds") if isinstance(seed_policy_raw.get("seeds"), list) else []
    seed_sets_path_raw = seed_policy_raw.get("seed_sets_path")
    seed_sets_path = (
        _resolve_path(str(seed_sets_path_raw), base_dir=config_path.parent)
        if isinstance(seed_sets_path_raw, str) and seed_sets_path_raw.strip()
        else None
    )
    if seed_sets_path is None:
        seed_sets_path = (get_repository_root() / DEFAULT_SEED_SETS_PATH).resolve()

    snqi_weights = _resolve_path(payload.get("snqi_weights"), base_dir=config_path.parent)
    snqi_baseline = _resolve_path(payload.get("snqi_baseline"), base_dir=config_path.parent)

    return CampaignConfig(
        name=name,
        scenario_matrix_path=scenario_matrix_path,
        planners=tuple(planner_specs),
        seed_policy=SeedPolicy(
            mode=mode,
            seed_set=str(seed_set) if seed_set is not None else None,
            seeds=tuple(int(seed) for seed in seeds),
            seed_sets_path=seed_sets_path,
        ),
        workers=int(payload.get("workers", 1)),
        horizon=(int(payload["horizon"]) if payload.get("horizon") is not None else None),
        dt=(float(payload["dt"]) if payload.get("dt") is not None else None),
        record_forces=bool(payload.get("record_forces", True)),
        resume=bool(payload.get("resume", True)),
        bootstrap_samples=int(payload.get("bootstrap_samples", 400)),
        bootstrap_confidence=float(payload.get("bootstrap_confidence", 0.95)),
        bootstrap_seed=int(payload.get("bootstrap_seed", 123)),
        snqi_weights_path=snqi_weights,
        snqi_baseline_path=snqi_baseline,
        stop_on_failure=bool(payload.get("stop_on_failure", False)),
        export_publication_bundle=bool(payload.get("export_publication_bundle", True)),
        include_videos_in_publication=bool(payload.get("include_videos_in_publication", False)),
        overwrite_publication_bundle=bool(payload.get("overwrite_publication_bundle", True)),
        repository_url=str(payload.get("repository_url", "https://github.com/ll7/robot_sf_ll7")),
        release_tag=str(payload.get("release_tag", "{release_tag}")),
        doi=str(payload.get("doi", "10.5281/zenodo.<record-id>")),
        paper_interpretation_profile=str(
            payload.get("paper_interpretation_profile", "baseline-ready-core")
        ),
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON with stable formatting and trailing newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def _write_markdown_table(path: Path, rows: list[dict[str, Any]], headers: tuple[str, ...]) -> None:
    """Write a table in Markdown format using explicit header order."""
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in rows:
        values = [str(row.get(col, "")) for col in headers]
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write campaign summary table in CSV format."""
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_table_artifacts(
    reports_dir: Path,
    base_name: str,
    rows: list[dict[str, Any]],
    *,
    headers: tuple[str, ...],
) -> tuple[Path, Path]:
    """Write CSV and Markdown table artifacts for one table dataset.

    Returns:
        Tuple of generated ``(csv_path, markdown_path)``.
    """
    csv_path = reports_dir / f"{base_name}.csv"
    md_path = reports_dir / f"{base_name}.md"
    _write_csv(csv_path, rows)
    _write_markdown_table(md_path, rows, headers=headers)
    return csv_path, md_path


def _planner_report_row(
    planner: PlannerSpec,
    summary: dict[str, Any],
    aggregates: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build one campaign table row for a planner run.

    Returns:
        Flattened row payload for CSV/Markdown export.
    """
    if aggregates:
        groups = [name for name in aggregates.keys() if name != "_meta"]
        metric_block = aggregates.get(groups[0], {}) if groups else {}
    else:
        metric_block = {}

    success_ci = _metric_ci(metric_block, "success")
    collision_ci = _metric_ci(metric_block, "collisions")
    snqi_ci = _metric_ci(metric_block, "snqi")

    execution_mode = str(
        (summary.get("algorithm_metadata_contract") or {}).get("execution_mode", "unknown")
        if isinstance(summary.get("algorithm_metadata_contract"), dict)
        else "unknown"
    )
    preflight_status = str((summary.get("preflight") or {}).get("status", "unknown"))
    learned_policy_contract = (summary.get("preflight") or {}).get("learned_policy_contract")
    contract_status = "not_applicable"
    contract_critical = 0
    contract_warnings = 0
    if isinstance(learned_policy_contract, dict):
        contract_status = str(learned_policy_contract.get("status", "not_applicable"))
        critical_list = learned_policy_contract.get("critical_mismatches")
        warning_list = learned_policy_contract.get("warnings")
        if isinstance(critical_list, list):
            contract_critical = len(critical_list)
        if isinstance(warning_list, list):
            contract_warnings = len(warning_list)
    status = str(summary.get("status", "unknown"))
    readiness_status = "native"
    if preflight_status == "fallback":
        readiness_status = "fallback"
    elif preflight_status == "skipped" or status == "failed":
        readiness_status = "degraded"
    elif execution_mode in {"adapter", "mixed"}:
        readiness_status = "adapter"

    row = {
        "planner_key": planner.key,
        "algo": planner.algo,
        "status": status,
        "episodes": int(summary.get("written", 0)),
        "started_at_utc": str(summary.get("started_at_utc", "unknown")),
        "finished_at_utc": str(summary.get("finished_at_utc", "unknown")),
        "runtime_sec": _safe_float(summary.get("runtime_sec")),
        "episodes_per_second": _safe_float(summary.get("episodes_per_second")),
        "failed_jobs": int(summary.get("failed_jobs", 0)),
        "success_mean": _safe_float(_metric_mean(metric_block, "success")),
        "collisions_mean": _safe_float(_metric_mean(metric_block, "collisions")),
        "near_misses_mean": _safe_float(_metric_mean(metric_block, "near_misses")),
        "time_to_goal_norm_mean": _safe_float(_metric_mean(metric_block, "time_to_goal_norm")),
        "path_efficiency_mean": _safe_float(_metric_mean(metric_block, "path_efficiency")),
        "comfort_exposure_mean": _safe_float(_metric_mean(metric_block, "comfort_exposure")),
        "jerk_mean": _safe_float(_metric_mean(metric_block, "jerk_mean")),
        "snqi_mean": _safe_float(_metric_mean(metric_block, "snqi")),
        "success_ci_low": _safe_float(success_ci[0]),
        "success_ci_high": _safe_float(success_ci[1]),
        "collision_ci_low": _safe_float(collision_ci[0]),
        "collision_ci_high": _safe_float(collision_ci[1]),
        "snqi_ci_low": _safe_float(snqi_ci[0]),
        "snqi_ci_high": _safe_float(snqi_ci[1]),
        "execution_mode": execution_mode,
        "readiness_status": readiness_status,
        "readiness_tier": str((summary.get("algorithm_readiness") or {}).get("tier", "unknown")),
        "preflight_status": preflight_status,
        "socnav_prereq_policy": planner.socnav_missing_prereq_policy,
        "learned_policy_contract_status": contract_status,
        "learned_policy_contract_critical": contract_critical,
        "learned_policy_contract_warnings": contract_warnings,
    }
    return row


def _scenario_family(record: dict[str, Any]) -> str:
    """Resolve scenario-family/archetype label from episode record metadata.

    Returns:
        Best-effort scenario family label.
    """
    scenario_params = record.get("scenario_params")
    if not isinstance(scenario_params, dict):
        scenario_params = {}
    metadata = scenario_params.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    for key in ("archetype", "scenario_family", "family"):
        value = metadata.get(key) or scenario_params.get(key) or record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    scenario_id = record.get("scenario_id")
    if isinstance(scenario_id, str) and scenario_id.strip():
        return scenario_id.split("_", 1)[0]
    return "unknown"


def _build_breakdown_rows(  # noqa: C901
    run_entries: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build per-scenario and per-family campaign diagnostic rows.

    Returns:
        Tuple of per-scenario rows and per-family rows.
    """
    per_scenario: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    per_family: dict[tuple[str, str, str], dict[str, Any]] = {}

    def _add_metric(bucket: dict[str, Any], metric: str, value: float | None) -> None:
        if value is None:
            return
        bucket.setdefault(metric, []).append(value)

    def _mean(values: list[float]) -> str:
        if not values:
            return "nan"
        return _safe_float(float(sum(values) / len(values)))

    for entry in run_entries:
        planner = entry.get("planner") if isinstance(entry.get("planner"), dict) else {}
        planner_key = str(planner.get("key", "unknown"))
        algo = str(planner.get("algo", "unknown"))
        episodes_path = entry.get("episodes_path")
        if not isinstance(episodes_path, str):
            continue
        candidate = get_repository_root() / episodes_path
        if not candidate.exists():
            continue
        for record in read_jsonl(str(candidate)):
            if not isinstance(record, dict):
                continue
            scenario_id = str(record.get("scenario_id", "unknown"))
            family = _scenario_family(record)
            metrics = record.get("metrics") if isinstance(record.get("metrics"), dict) else {}
            scenario_key = (planner_key, algo, scenario_id, family)
            family_key = (planner_key, algo, family)

            scenario_bucket = per_scenario.setdefault(
                scenario_key,
                {
                    "planner_key": planner_key,
                    "algo": algo,
                    "scenario_id": scenario_id,
                    "scenario_family": family,
                    "episodes": 0,
                },
            )
            family_bucket = per_family.setdefault(
                family_key,
                {
                    "planner_key": planner_key,
                    "algo": algo,
                    "scenario_family": family,
                    "episodes": 0,
                },
            )
            scenario_bucket["episodes"] += 1
            family_bucket["episodes"] += 1
            for metric in _REPORT_METRICS:
                raw = metrics.get(metric)
                try:
                    value = float(raw)
                except (TypeError, ValueError):
                    value = None
                if value is not None and not math.isfinite(value):
                    value = None
                _add_metric(scenario_bucket, metric, value)
                _add_metric(family_bucket, metric, value)

    def _finalize(row: dict[str, Any]) -> dict[str, Any]:
        finalized = dict(row)
        for metric in _REPORT_METRICS:
            values = finalized.pop(metric, [])
            if not isinstance(values, list):
                values = []
            finalized[f"{metric}_mean"] = _mean(values)
        return finalized

    scenario_rows = sorted(
        (_finalize(row) for row in per_scenario.values()),
        key=lambda row: (
            row.get("planner_key", ""),
            row.get("scenario_id", ""),
            row.get("scenario_family", ""),
        ),
    )
    family_rows = sorted(
        (_finalize(row) for row in per_family.values()),
        key=lambda row: (
            row.get("planner_key", ""),
            row.get("scenario_family", ""),
        ),
    )
    return scenario_rows, family_rows


def _strict_vs_fallback_comparisons(rows: list[dict[str, Any]]) -> list[str]:
    """Build strict-vs-fallback comparison summaries when both modes are present.

    Returns:
        Human-readable comparison lines.
    """
    by_algo: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        algo = str(row.get("algo", "unknown"))
        by_algo.setdefault(algo, []).append(row)

    lines: list[str] = []
    for algo, algo_rows in sorted(by_algo.items()):
        strict = [row for row in algo_rows if str(row.get("socnav_prereq_policy")) == "fail-fast"]
        fallback = [row for row in algo_rows if str(row.get("socnav_prereq_policy")) == "fallback"]
        if not strict or not fallback:
            continue
        strict_row = strict[0]
        fallback_row = fallback[0]
        lines.append(
            f"`{algo}`: strict preflight={strict_row.get('preflight_status')}, "
            f"fallback preflight={fallback_row.get('preflight_status')}, "
            f"strict success={strict_row.get('success_mean')}, "
            f"fallback success={fallback_row.get('success_mean')}"
        )
    return lines


def _write_campaign_report(path: Path, payload: dict[str, Any]) -> None:  # noqa: C901, PLR0912
    """Write a human-readable campaign report in Markdown."""
    campaign = payload.get("campaign", {})
    rows = payload.get("planner_rows", [])
    warnings = payload.get("warnings", [])

    lines = [
        "# Camera-Ready Benchmark Campaign Report",
        "",
        f"- Campaign ID: `{campaign.get('campaign_id', 'unknown')}`",
        f"- Name: `{campaign.get('name', 'unknown')}`",
        f"- Created (UTC): `{campaign.get('created_at_utc', 'unknown')}`",
        f"- Scenario matrix: `{campaign.get('scenario_matrix', 'unknown')}`",
        f"- Scenario matrix hash: `{campaign.get('scenario_matrix_hash', 'unknown')}`",
        f"- Git commit: `{campaign.get('git_hash', 'unknown')}`",
        f"- Runtime sec: `{campaign.get('runtime_sec', 0.0)}`",
        f"- Episodes/sec: `{campaign.get('episodes_per_second', 0.0)}`",
        f"- Interpretation profile: `{campaign.get('paper_interpretation_profile', 'unknown')}`",
        f"- Command: `{campaign.get('invoked_command', 'unknown')}`",
        "",
        "## Planner Summary",
        "",
    ]

    if rows:
        lines.extend(
            [
                "| planner | algo | status | started (UTC) | runtime (s) | episodes | eps/s | success | collisions | snqi |",
                "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
            ],
        )
        for row in rows:
            lines.append(
                "| "
                f"{row.get('planner_key')} | {row.get('algo')} | {row.get('status')} | "
                f"{row.get('started_at_utc')} | {row.get('runtime_sec')} | {row.get('episodes')} | "
                f"{row.get('episodes_per_second')} | {row.get('success_mean')} | "
                f"{row.get('collisions_mean')} | {row.get('snqi_mean')} |",
            )
    else:
        lines.append("No planner rows were produced.")
    fallback_rows = [
        row for row in rows if str(row.get("readiness_status", "")) in {"fallback", "degraded"}
    ]
    lines.extend(["", "## Readiness & Degraded/Fallback Status", ""])
    if rows:
        lines.append(
            "| planner | execution mode | readiness status | tier | preflight | learned contract | run status |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        for row in rows:
            lines.append(
                "| "
                f"{row.get('planner_key')} | {row.get('execution_mode')} | "
                f"{row.get('readiness_status')} | {row.get('readiness_tier')} | "
                f"{row.get('preflight_status')} | {row.get('learned_policy_contract_status')} | "
                f"{row.get('status')} |"
            )
    if fallback_rows:
        lines.append("")
        lines.append("Planners in fallback/degraded mode:")
        for row in fallback_rows:
            lines.append(
                f"- `{row.get('planner_key')}`: readiness={row.get('readiness_status')}, "
                f"preflight={row.get('preflight_status')}, tier={row.get('readiness_tier')}"
            )
    else:
        lines.append("")
        lines.append("- No fallback/degraded planners detected.")

    lines.extend(["", "## SocNav Strict-vs-Fallback Disclosure", ""])
    if rows:
        lines.append("| planner | algo | prereq policy | preflight status | readiness status |")
        lines.append("|---|---|---|---|---|")
        for row in rows:
            lines.append(
                "| "
                f"{row.get('planner_key')} | {row.get('algo')} | "
                f"{row.get('socnav_prereq_policy')} | {row.get('preflight_status')} | "
                f"{row.get('readiness_status')} |"
            )
        comparisons = _strict_vs_fallback_comparisons(rows)
        if comparisons:
            lines.append("")
            lines.append("Strict-vs-fallback comparisons (where both modes are present):")
            for line in comparisons:
                lines.append(f"- {line}")
        else:
            lines.append("")
            lines.append(
                "- No within-campaign strict-vs-fallback pair available for direct comparison."
            )

    scenario_path = (payload.get("artifacts") or {}).get("scenario_breakdown_csv")
    family_path = (payload.get("artifacts") or {}).get("scenario_family_breakdown_csv")
    if isinstance(scenario_path, str) or isinstance(family_path, str):
        lines.extend(["", "## Scenario Diagnostics", ""])
        if isinstance(scenario_path, str):
            lines.append(f"- Per-scenario breakdown: `{scenario_path}`")
        if isinstance(family_path, str):
            lines.append(f"- Per-family breakdown: `{family_path}`")

    lines.extend(["", "## Campaign Warnings", ""])
    if warnings:
        for warning in warnings:
            lines.append(f"- {warning}")
    else:
        lines.append("- No campaign-level warnings.")

    publication = payload.get("publication_bundle")
    if isinstance(publication, dict):
        lines.extend(
            [
                "",
                "## Publication Bundle",
                "",
                f"- Bundle dir: `{publication.get('bundle_dir', 'unknown')}`",
                f"- Archive: `{publication.get('archive_path', 'unknown')}`",
                f"- Manifest: `{publication.get('manifest_path', 'unknown')}`",
                f"- Checksums: `{publication.get('checksums_path', 'unknown')}`",
            ],
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_campaign(  # noqa: PLR0915
    cfg: CampaignConfig,
    *,
    output_root: Path | None = None,
    label: str | None = None,
    skip_publication_bundle: bool = False,
    invoked_command: str | None = None,
) -> dict[str, Any]:
    """Execute a camera-ready planner campaign and emit campaign artifacts.

    Returns:
        Campaign execution summary with output paths and high-level counters.
    """
    ensure_canonical_tree(categories=("benchmarks",))
    snqi_weights = load_optional_json(str(cfg.snqi_weights_path) if cfg.snqi_weights_path else None)
    snqi_baseline = load_optional_json(
        str(cfg.snqi_baseline_path) if cfg.snqi_baseline_path else None
    )

    campaign_id = _campaign_id(cfg, label=label)
    base_dir = (
        output_root.resolve()
        if output_root
        else (get_artifact_category_path("benchmarks") / "camera_ready")
    )
    campaign_root = (base_dir / campaign_id).resolve()
    runs_dir = campaign_root / "runs"
    reports_dir = campaign_root / "reports"
    preflight_dir = campaign_root / "preflight"
    runs_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    preflight_dir.mkdir(parents=True, exist_ok=True)

    campaign_started_at_utc = _utc_now()
    start = time.perf_counter()
    scenarios = _load_campaign_scenarios(cfg)
    resolved_seeds = _resolved_seed_inventory(scenarios)
    scenario_hash = _hash_payload(scenarios)
    git_meta = _git_context()
    validate_config_path = preflight_dir / "validate_config.json"
    preview_scenarios_path = preflight_dir / "preview_scenarios.json"

    validate_payload = {
        "schema_version": "benchmark-preflight-validate-config.v1",
        "campaign_id": campaign_id,
        "generated_at_utc": campaign_started_at_utc,
        "scenario_matrix": _repo_relative(cfg.scenario_matrix_path),
        "scenario_count": len(scenarios),
        "planner_count": len([planner for planner in cfg.planners if planner.enabled]),
        "workers": cfg.workers,
        "horizon": cfg.horizon,
        "dt": cfg.dt,
        "resume": cfg.resume,
        "seed_policy": {
            "mode": cfg.seed_policy.mode,
            "seed_set": cfg.seed_policy.seed_set,
            "seeds": list(cfg.seed_policy.seeds),
            "resolved_seeds": resolved_seeds,
            "seed_sets_path": _repo_relative(cfg.seed_policy.seed_sets_path),
        },
    }
    preview_payload = {
        "schema_version": "benchmark-preflight-preview-scenarios.v1",
        "campaign_id": campaign_id,
        "generated_at_utc": campaign_started_at_utc,
        "scenario_count": len(scenarios),
        "scenarios": scenarios,
    }
    _write_json(validate_config_path, validate_payload)
    _write_json(preview_scenarios_path, preview_payload)

    manifest_payload: dict[str, Any] = {
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "campaign_id": campaign_id,
        "name": cfg.name,
        "created_at_utc": campaign_started_at_utc,
        "started_at_utc": campaign_started_at_utc,
        "scenario_matrix": _repo_relative(cfg.scenario_matrix_path),
        "scenario_matrix_hash": scenario_hash,
        "seed_policy": {
            "mode": cfg.seed_policy.mode,
            "seed_set": cfg.seed_policy.seed_set,
            "seeds": list(cfg.seed_policy.seeds),
            "resolved_seeds": resolved_seeds,
            "seed_sets_path": _repo_relative(cfg.seed_policy.seed_sets_path),
        },
        "git": git_meta,
        "config_hash": _config_hash(_jsonable(asdict(cfg))),
        "invoked_command": invoked_command,
        "planners": [
            {
                "key": planner.key,
                "algo": planner.algo,
                "benchmark_profile": planner.benchmark_profile,
                "algo_config_path": (
                    _repo_relative(planner.algo_config_path)
                    if planner.algo_config_path is not None
                    else None
                ),
                "enabled": planner.enabled,
            }
            for planner in cfg.planners
        ],
    }
    _write_json(campaign_root / "campaign_manifest.json", manifest_payload)

    run_entries: list[dict[str, Any]] = []
    planner_rows: list[dict[str, Any]] = []
    warnings: list[str] = []

    for planner in cfg.planners:
        if not planner.enabled:
            continue

        planner_dir = runs_dir / planner.key
        planner_dir.mkdir(parents=True, exist_ok=True)
        episodes_path = planner_dir / "episodes.jsonl"

        effective_workers = (
            planner.workers_override if planner.workers_override is not None else cfg.workers
        )
        effective_horizon = (
            planner.horizon_override if planner.horizon_override is not None else cfg.horizon
        )
        effective_dt = planner.dt_override if planner.dt_override is not None else cfg.dt

        logger.info(
            "Running campaign planner key={} algo={} profile={} workers={}",
            planner.key,
            planner.algo,
            planner.benchmark_profile,
            effective_workers,
        )

        planner_started_at_utc = _utc_now()
        planner_start = time.perf_counter()
        status = "ok"
        summary: dict[str, Any]
        aggregates: dict[str, Any] | None = None

        try:
            summary = run_batch(
                scenarios,
                out_path=episodes_path,
                schema_path=DEFAULT_EPISODE_SCHEMA_PATH,
                horizon=effective_horizon if effective_horizon is not None else 0,
                dt=effective_dt if effective_dt is not None else 0.0,
                record_forces=cfg.record_forces,
                snqi_weights=snqi_weights,
                snqi_baseline=snqi_baseline,
                algo=planner.algo,
                algo_config_path=(
                    str(planner.algo_config_path) if planner.algo_config_path is not None else None
                ),
                benchmark_profile=planner.benchmark_profile,
                socnav_missing_prereq_policy=planner.socnav_missing_prereq_policy,
                adapter_impact_eval=planner.adapter_impact_eval,
                workers=effective_workers,
                resume=cfg.resume,
            )
            if int(summary.get("failed_jobs", 0)) > 0:
                status = "partial-failure"
        except Exception as exc:
            status = "failed"
            summary = {
                "status": "failed",
                "error": repr(exc),
                "total_jobs": 0,
                "written": 0,
                "failed_jobs": 0,
                "failures": [],
            }
            warnings.append(f"Planner '{planner.key}' failed: {exc}")

        planner_finished_at_utc = _utc_now()
        runtime_sec = float(max(1e-9, time.perf_counter() - planner_start))
        episodes_written = int(summary.get("written", 0))
        summary["status"] = status
        summary["started_at_utc"] = planner_started_at_utc
        summary["finished_at_utc"] = planner_finished_at_utc
        summary["runtime_sec"] = runtime_sec
        summary["episodes_per_second"] = (
            (episodes_written / runtime_sec) if runtime_sec > 0 else 0.0
        )
        _write_json(planner_dir / "summary.json", summary)

        if status != "failed" and episodes_path.exists() and episodes_path.stat().st_size > 0:
            records = read_jsonl(str(episodes_path))
            try:
                aggregates = compute_aggregates_with_ci(
                    records,
                    group_by="scenario_params.algo",
                    bootstrap_samples=cfg.bootstrap_samples,
                    bootstrap_confidence=cfg.bootstrap_confidence,
                    bootstrap_seed=cfg.bootstrap_seed,
                )
            except Exception as exc:
                warnings.append(
                    f"Aggregation failed for planner '{planner.key}': {exc}",
                )

        row = _planner_report_row(planner, summary, aggregates)
        planner_rows.append(row)

        run_entries.append(
            {
                "planner": {
                    "key": planner.key,
                    "algo": planner.algo,
                    "benchmark_profile": planner.benchmark_profile,
                    "algo_config_path": (
                        _repo_relative(planner.algo_config_path)
                        if planner.algo_config_path is not None
                        else None
                    ),
                    "socnav_missing_prereq_policy": planner.socnav_missing_prereq_policy,
                    "adapter_impact_eval": planner.adapter_impact_eval,
                    "workers": effective_workers,
                    "horizon": effective_horizon,
                    "dt": effective_dt,
                },
                "status": status,
                "started_at_utc": planner_started_at_utc,
                "finished_at_utc": planner_finished_at_utc,
                "runtime_sec": runtime_sec,
                "episodes_path": _repo_relative(episodes_path),
                "summary_path": _repo_relative(planner_dir / "summary.json"),
                "summary": summary,
                "aggregates": aggregates,
            },
        )

        if status == "failed" and cfg.stop_on_failure:
            break

    planner_rows.sort(
        key=lambda row: (row.get("snqi_mean", "nan") == "nan", row.get("planner_key"))
    )

    summary_json_path = reports_dir / "campaign_summary.json"
    report_md_path = reports_dir / "campaign_report.md"

    csv_path, md_table_path = _write_table_artifacts(
        reports_dir,
        "campaign_table",
        planner_rows,
        headers=(
            "planner_key",
            "algo",
            "execution_mode",
            "readiness_status",
            "readiness_tier",
            "preflight_status",
            "learned_policy_contract_status",
            "socnav_prereq_policy",
            "status",
            "episodes",
            "success_mean",
            "collisions_mean",
            "near_misses_mean",
            "time_to_goal_norm_mean",
            "path_efficiency_mean",
            "comfort_exposure_mean",
            "jerk_mean",
            "snqi_mean",
        ),
    )
    core_rows = [row for row in planner_rows if str(row.get("readiness_tier")) == "baseline-ready"]
    experimental_rows = [
        row for row in planner_rows if str(row.get("readiness_tier")) != "baseline-ready"
    ]
    core_csv_path, core_md_path = _write_table_artifacts(
        reports_dir,
        "campaign_table_core",
        core_rows,
        headers=(
            "planner_key",
            "algo",
            "readiness_tier",
            "status",
            "episodes",
            "success_mean",
            "collisions_mean",
            "snqi_mean",
        ),
    )
    experimental_csv_path, experimental_md_path = _write_table_artifacts(
        reports_dir,
        "campaign_table_experimental",
        experimental_rows,
        headers=(
            "planner_key",
            "algo",
            "readiness_tier",
            "status",
            "episodes",
            "success_mean",
            "collisions_mean",
            "snqi_mean",
        ),
    )
    scenario_rows, family_rows = _build_breakdown_rows(run_entries)
    scenario_csv_path, scenario_md_path = _write_table_artifacts(
        reports_dir,
        "scenario_breakdown",
        scenario_rows,
        headers=(
            "planner_key",
            "algo",
            "scenario_family",
            "scenario_id",
            "episodes",
            "success_mean",
            "collisions_mean",
            "near_misses_mean",
            "time_to_goal_norm_mean",
            "path_efficiency_mean",
            "comfort_exposure_mean",
            "jerk_mean",
            "snqi_mean",
        ),
    )
    family_csv_path, family_md_path = _write_table_artifacts(
        reports_dir,
        "scenario_family_breakdown",
        family_rows,
        headers=(
            "planner_key",
            "algo",
            "scenario_family",
            "episodes",
            "success_mean",
            "collisions_mean",
            "near_misses_mean",
            "time_to_goal_norm_mean",
            "path_efficiency_mean",
            "comfort_exposure_mean",
            "jerk_mean",
            "snqi_mean",
        ),
    )

    campaign_finished_at_utc = _utc_now()
    runtime_sec = float(max(1e-9, time.perf_counter() - start))
    total_episodes = sum(int(entry.get("summary", {}).get("written", 0)) for entry in run_entries)
    successful_runs = sum(
        1 for entry in run_entries if str(entry.get("status", "")).startswith(("ok", "partial"))
    )

    campaign_summary = {
        "campaign": {
            "schema_version": CAMPAIGN_SCHEMA_VERSION,
            "campaign_id": campaign_id,
            "name": cfg.name,
            "created_at_utc": campaign_started_at_utc,
            "started_at_utc": campaign_started_at_utc,
            "finished_at_utc": campaign_finished_at_utc,
            "scenario_matrix": _repo_relative(cfg.scenario_matrix_path),
            "scenario_matrix_hash": scenario_hash,
            "git_hash": git_meta.get("commit", "unknown"),
            "invoked_command": invoked_command,
            "runtime_sec": runtime_sec,
            "episodes_per_second": (total_episodes / runtime_sec) if runtime_sec > 0 else 0.0,
            "total_episodes": total_episodes,
            "successful_runs": successful_runs,
            "total_runs": len(run_entries),
            "paper_interpretation_profile": cfg.paper_interpretation_profile,
        },
        "planner_rows": planner_rows,
        "runs": run_entries,
        "warnings": warnings,
        "artifacts": {
            "campaign_manifest": _repo_relative(campaign_root / "campaign_manifest.json"),
            "campaign_summary_json": _repo_relative(summary_json_path),
            "campaign_table_csv": _repo_relative(csv_path),
            "campaign_table_md": _repo_relative(md_table_path),
            "campaign_table_core_csv": _repo_relative(core_csv_path),
            "campaign_table_core_md": _repo_relative(core_md_path),
            "campaign_table_experimental_csv": _repo_relative(experimental_csv_path),
            "campaign_table_experimental_md": _repo_relative(experimental_md_path),
            "preflight_validate_config": _repo_relative(validate_config_path),
            "preflight_preview_scenarios": _repo_relative(preview_scenarios_path),
            "scenario_breakdown_csv": _repo_relative(scenario_csv_path),
            "scenario_breakdown_md": _repo_relative(scenario_md_path),
            "scenario_family_breakdown_csv": _repo_relative(family_csv_path),
            "scenario_family_breakdown_md": _repo_relative(family_md_path),
            "campaign_report_md": _repo_relative(report_md_path),
        },
    }

    publication_payload: dict[str, Any] | None = None
    if cfg.export_publication_bundle and not skip_publication_bundle:
        publication_dir = get_artifact_category_path("benchmarks") / "publication"
        bundle_name = f"{campaign_id}_publication_bundle"
        try:
            bundle = export_publication_bundle(
                campaign_root,
                publication_dir,
                bundle_name=bundle_name,
                include_videos=cfg.include_videos_in_publication,
                repository_url=cfg.repository_url,
                release_tag=cfg.release_tag,
                doi=cfg.doi,
                overwrite=cfg.overwrite_publication_bundle,
            )
            publication_payload = {
                "bundle_dir": _repo_relative(bundle.bundle_dir),
                "archive_path": _repo_relative(bundle.archive_path),
                "manifest_path": _repo_relative(bundle.manifest_path),
                "checksums_path": _repo_relative(bundle.checksums_path),
                "file_count": bundle.file_count,
                "total_bytes": bundle.total_bytes,
            }
            campaign_summary["publication_bundle"] = publication_payload
        except Exception as exc:
            warnings.append(f"Publication bundle export failed: {exc}")

    _write_json(summary_json_path, campaign_summary)
    _write_campaign_report(report_md_path, campaign_summary)

    # Add run-level metadata files for publication provenance helpers.
    run_meta = {
        "repo": {
            "remote": git_meta.get("remote", "unknown"),
            "branch": git_meta.get("branch", "unknown"),
            "commit": git_meta.get("commit", "unknown"),
        },
        "matrix_path": _repo_relative(cfg.scenario_matrix_path),
        "scenario_matrix_hash": scenario_hash,
        "seed_policy": {
            "mode": cfg.seed_policy.mode,
            "seed_set": cfg.seed_policy.seed_set,
            "seeds": list(cfg.seed_policy.seeds),
            "resolved_seeds": resolved_seeds,
            "seed_sets_path": _repo_relative(cfg.seed_policy.seed_sets_path),
        },
        "preflight_artifacts": {
            "validate_config": _repo_relative(validate_config_path),
            "preview_scenarios": _repo_relative(preview_scenarios_path),
        },
        "campaign_id": campaign_id,
        "started_at_utc": campaign_started_at_utc,
        "finished_at_utc": campaign_finished_at_utc,
        "invoked_command": invoked_command,
        "runtime_sec": runtime_sec,
        "episodes_per_second": (total_episodes / runtime_sec) if runtime_sec > 0 else 0.0,
    }
    run_manifest = {
        "git_hash": git_meta.get("commit", "unknown"),
        "scenario_matrix_hash": scenario_hash,
        "runtime_sec": runtime_sec,
        "episodes_per_second": (total_episodes / runtime_sec) if runtime_sec > 0 else 0.0,
    }
    _write_json(campaign_root / "run_meta.json", run_meta)
    _write_json(campaign_root / "manifest.json", run_manifest)
    _write_json(
        campaign_root / "campaign_manifest.json",
        {
            **manifest_payload,
            "runtime_sec": runtime_sec,
            "finished_at_utc": campaign_finished_at_utc,
        },
    )

    logger.info(
        "Camera-ready campaign finished id={} runs={} episodes={} out={}",
        campaign_id,
        len(run_entries),
        total_episodes,
        campaign_root,
    )

    return {
        "campaign_id": campaign_id,
        "campaign_root": str(campaign_root),
        "summary_json": str(summary_json_path),
        "table_csv": str(csv_path),
        "table_md": str(md_table_path),
        "report_md": str(report_md_path),
        "total_runs": len(run_entries),
        "successful_runs": successful_runs,
        "total_episodes": total_episodes,
        "runtime_sec": runtime_sec,
        "publication_bundle": publication_payload,
        "warnings": warnings,
    }


__all__ = [
    "CAMPAIGN_SCHEMA_VERSION",
    "CampaignConfig",
    "PlannerSpec",
    "SeedPolicy",
    "load_campaign_config",
    "run_campaign",
]
