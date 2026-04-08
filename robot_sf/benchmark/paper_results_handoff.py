"""Paper-facing Results handoff export for frozen benchmark bundles."""
# ruff: noqa: DOC201

from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from robot_sf.benchmark.seed_variance import build_seed_variability_rows
from robot_sf.benchmark.utils import episode_metric_value
from robot_sf.common.artifact_paths import get_artifact_category_path, get_repository_root

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path

PAPER_RESULTS_HANDOFF_SCHEMA_VERSION = "paper-results-handoff.v1"
PAPER_RESULTS_HANDOFF_METRICS = (
    "success",
    "collisions",
    "near_misses",
    "time_to_goal_norm",
    "snqi",
)
DEFAULT_CONFIDENCE_SETTINGS = {
    "method": "bootstrap_mean_over_seed_means",
    "confidence": 0.95,
    "bootstrap_samples": 400,
    "bootstrap_seed": 123,
}
_PLANNER_SUMMARY_SCENARIO_ID = "__paper_results_planner_summary__"


@dataclass(frozen=True)
class PaperResultsHandoffResult:
    """Paths and counts produced by a paper Results handoff export."""

    output_dir: Path
    json_path: Path
    csv_path: Path
    row_count: int


@dataclass(frozen=True)
class _ResolvedSource:
    """Resolved benchmark source roots for a campaign or publication bundle."""

    source_path: Path
    payload_root: Path
    source_kind: str
    publication_manifest_path: Path | None = None


def _repo_relative(path: Path) -> str:
    """Return a repository-relative path when possible."""
    repo_root = get_repository_root().resolve()
    try:
        return path.resolve().relative_to(repo_root).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _resolve_source_path(source_path: Path) -> _ResolvedSource:
    """Resolve a campaign root or publication bundle into a payload root."""
    source = source_path.expanduser().resolve()
    payload_root = source / "payload"
    publication_manifest = source / "publication_manifest.json"
    if payload_root.is_dir() and publication_manifest.exists():
        return _ResolvedSource(
            source_path=source,
            payload_root=payload_root,
            source_kind="publication_bundle",
            publication_manifest_path=publication_manifest,
        )
    if (source / "runs").is_dir():
        return _ResolvedSource(
            source_path=source,
            payload_root=source,
            source_kind="campaign_root",
        )
    raise FileNotFoundError(
        "Expected a campaign root with runs/ or a publication bundle with payload/ and "
        f"publication_manifest.json: {source}"
    )


def _load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object, returning an empty mapping for missing files."""
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object at the top level")
    return payload


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    """Read JSONL episode records and fail on malformed object payloads."""
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            yield payload


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    """Load CSV rows from ``path`` when present."""
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _campaign_table_index(payload_root: Path) -> dict[tuple[str, str], dict[str, str]]:
    """Index campaign-table planner metadata by planner and kinematics."""
    rows = _load_csv_rows(payload_root / "reports" / "campaign_table.csv")
    if not rows:
        rows = _load_csv_rows(payload_root / "reports" / "campaign_table_core.csv")
        rows.extend(_load_csv_rows(payload_root / "reports" / "campaign_table_experimental.csv"))

    indexed: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        planner_key = str(row.get("planner_key") or "").strip()
        kinematics = str(row.get("kinematics") or "differential_drive").strip()
        if planner_key:
            indexed[(planner_key, kinematics)] = row
    return indexed


def _split_run_key(run_dir: Path) -> tuple[str, str]:
    """Infer planner key and kinematics from a campaign run directory name."""
    name = run_dir.name
    if "__" not in name:
        return name, "differential_drive"
    planner_key, kinematics = name.rsplit("__", 1)
    return planner_key, kinematics


def _episode_jsonl_paths(resolved: _ResolvedSource) -> list[Path]:
    """Return sorted episode JSONL paths for a resolved campaign source."""
    runs_dir = resolved.payload_root / "runs"
    episode_paths = sorted(runs_dir.glob("*/episodes.jsonl"))
    if not episode_paths:
        raise FileNotFoundError(f"No episode JSONL files found under {runs_dir}")
    return episode_paths


def _coerce_float(value: Any) -> float | None:
    """Convert a value to a finite float when possible."""
    if isinstance(value, str):
        value = value.strip().lstrip("'")
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _coerce_int(value: Any) -> int | None:
    """Convert a value to int when possible."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _load_run_records(
    episodes_path: Path,
    *,
    metrics: Sequence[str],
    campaign_rows: Mapping[tuple[str, str], Mapping[str, str]],
) -> tuple[list[dict[str, Any]], set[int]]:
    """Load and annotate one planner run worth of episode records."""
    planner_key, kinematics = _split_run_key(episodes_path.parent)
    row_metadata = campaign_rows.get((planner_key, kinematics), {})
    algo = str(row_metadata.get("algo") or planner_key)
    planner_group = str(row_metadata.get("planner_group") or "unknown")
    benchmark_profile = str(row_metadata.get("benchmark_profile") or "unknown")

    records: list[dict[str, Any]] = []
    discovered_seeds: set[int] = set()
    for record in _read_jsonl(episodes_path):
        annotated = dict(record)
        record_metrics = dict(record.get("metrics") or {})
        for metric in metrics:
            value = episode_metric_value(record, metric)
            if value is not None:
                record_metrics[metric] = value
        annotated["metrics"] = record_metrics
        annotated["source_scenario_id"] = record.get("scenario_id")
        annotated["scenario_id"] = _PLANNER_SUMMARY_SCENARIO_ID
        annotated["planner_key"] = planner_key
        annotated["kinematics"] = kinematics
        annotated["algo"] = algo
        annotated["planner_group"] = planner_group
        annotated["benchmark_profile"] = benchmark_profile
        seed = _coerce_int(record.get("seed"))
        if seed is not None:
            discovered_seeds.add(seed)
        records.append(annotated)

    return records, discovered_seeds


def _campaign_id(
    resolved: _ResolvedSource,
    *,
    seed_payload: Mapping[str, Any],
    publication_manifest: Mapping[str, Any],
) -> str:
    """Resolve the canonical campaign id from available metadata."""
    candidate = seed_payload.get("campaign_id")
    if isinstance(candidate, str) and candidate.strip():
        return candidate.strip()
    provenance = publication_manifest.get("provenance")
    if isinstance(provenance, Mapping):
        for key in ("campaign_id", "run_id"):
            value = provenance.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    name = resolved.source_path.name
    return name.removesuffix("_publication_bundle")


def _config_hash(
    seed_payload: Mapping[str, Any],
    publication_manifest: Mapping[str, Any],
) -> str:
    """Resolve the source config hash when available."""
    rows = seed_payload.get("rows")
    if isinstance(rows, list) and rows:
        provenance = rows[0].get("provenance") if isinstance(rows[0], Mapping) else None
        if isinstance(provenance, Mapping):
            value = provenance.get("config_hash")
            if isinstance(value, str) and value.strip():
                return value.strip()
    provenance = publication_manifest.get("provenance")
    if isinstance(provenance, Mapping):
        value = provenance.get("config_hash")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "unknown"


def _git_hash(
    seed_payload: Mapping[str, Any],
    publication_manifest: Mapping[str, Any],
) -> str:
    """Resolve the source git hash when available."""
    rows = seed_payload.get("rows")
    if isinstance(rows, list) and rows:
        provenance = rows[0].get("provenance") if isinstance(rows[0], Mapping) else None
        if isinstance(provenance, Mapping):
            value = provenance.get("git_hash")
            if isinstance(value, str) and value.strip():
                return value.strip()
    provenance = publication_manifest.get("provenance")
    if isinstance(provenance, Mapping):
        repo = provenance.get("repo")
        if isinstance(repo, Mapping):
            value = repo.get("commit")
            if isinstance(value, str) and value.strip():
                return value.strip()
    return "unknown"


def _seed_policy(
    seed_payload: Mapping[str, Any], resolved_seeds: Iterable[int] | None = None
) -> dict[str, Any]:
    """Resolve seed policy metadata for the handoff payload."""
    rows = seed_payload.get("rows")
    if isinstance(rows, list) and rows:
        provenance = rows[0].get("provenance") if isinstance(rows[0], Mapping) else None
        if isinstance(provenance, Mapping):
            seed_policy = provenance.get("seed_policy")
            if isinstance(seed_policy, Mapping):
                return dict(seed_policy)
    seeds = sorted({int(seed) for seed in (resolved_seeds or [])})
    return {"mode": "derived_from_episode_records", "seed_set": None, "resolved_seeds": seeds}


def _source_paths(resolved: _ResolvedSource) -> dict[str, Any]:
    """Build source path metadata for the exported handoff."""
    paths: dict[str, Any] = {
        "source_kind": resolved.source_kind,
        "source_path": _repo_relative(resolved.source_path),
        "payload_root": _repo_relative(resolved.payload_root),
        "episodes_glob": _repo_relative(resolved.payload_root / "runs" / "*" / "episodes.jsonl"),
        "campaign_table_csv": _repo_relative(
            resolved.payload_root / "reports" / "campaign_table.csv"
        ),
        "seed_variability_json": _repo_relative(
            resolved.payload_root / "reports" / "seed_variability_by_scenario.json"
        ),
    }
    if resolved.publication_manifest_path is not None:
        paths["publication_manifest"] = _repo_relative(resolved.publication_manifest_path)
    return paths


def _flatten_handoff_row(
    row: Mapping[str, Any],
    *,
    row_metadata: Mapping[str, str],
    metrics: Sequence[str],
    seed_policy_override: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Flatten one builder row into the paper Results handoff schema."""
    provenance = row.get("provenance") if isinstance(row.get("provenance"), Mapping) else {}
    row_seed_policy = provenance.get("seed_policy") if isinstance(provenance, Mapping) else {}
    seed_policy = seed_policy_override if seed_policy_override is not None else row_seed_policy
    confidence = provenance.get("confidence") if isinstance(provenance, Mapping) else {}
    episode_count = _coerce_int(row.get("episode_count"))
    seed_count = _coerce_int(row.get("seed_count"))
    repeat_count: int | float | None = None
    if episode_count is not None and seed_count is not None and seed_count > 0:
        repeat_count = (
            episode_count // seed_count
            if episode_count % seed_count == 0
            else episode_count / seed_count
        )
    out: dict[str, Any] = {
        "planner_key": row.get("planner_key"),
        "algo": row.get("algo"),
        "planner_group": row.get("planner_group"),
        "kinematics": row.get("kinematics"),
        "status": row_metadata.get("status"),
        "readiness_tier": row_metadata.get("readiness_tier"),
        "readiness_status": row_metadata.get("readiness_status"),
        "preflight_status": row_metadata.get("preflight_status"),
        "episode_count": episode_count,
        "seed_count": seed_count,
        "repeat_count": repeat_count,
        "seed_list": list(row.get("seed_list") or []),
        "campaign_id": provenance.get("campaign_id") if isinstance(provenance, Mapping) else None,
        "config_hash": provenance.get("config_hash") if isinstance(provenance, Mapping) else None,
        "git_hash": provenance.get("git_hash") if isinstance(provenance, Mapping) else None,
        "seed_policy_mode": seed_policy.get("mode") if isinstance(seed_policy, Mapping) else None,
        "seed_policy_seed_set": seed_policy.get("seed_set")
        if isinstance(seed_policy, Mapping)
        else None,
        "confidence_method": confidence.get("method") if isinstance(confidence, Mapping) else None,
        "confidence_level": confidence.get("confidence")
        if isinstance(confidence, Mapping)
        else None,
        "bootstrap_samples": confidence.get("bootstrap_samples")
        if isinstance(confidence, Mapping)
        else None,
        "bootstrap_seed": confidence.get("bootstrap_seed")
        if isinstance(confidence, Mapping)
        else None,
    }
    summary = row.get("summary") if isinstance(row.get("summary"), Mapping) else {}
    for metric in metrics:
        metric_summary = summary.get(metric) if isinstance(summary.get(metric), Mapping) else {}
        out[f"{metric}_mean"] = (
            metric_summary.get("mean") if isinstance(metric_summary, Mapping) else None
        )
        out[f"{metric}_std"] = (
            metric_summary.get("std") if isinstance(metric_summary, Mapping) else None
        )
        out[f"{metric}_count"] = (
            metric_summary.get("count") if isinstance(metric_summary, Mapping) else None
        )
        out[f"{metric}_ci_low"] = (
            metric_summary.get("ci_low") if isinstance(metric_summary, Mapping) else None
        )
        out[f"{metric}_ci_high"] = (
            metric_summary.get("ci_high") if isinstance(metric_summary, Mapping) else None
        )
        out[f"{metric}_ci_half_width"] = (
            metric_summary.get("ci_half_width") if isinstance(metric_summary, Mapping) else None
        )
        table_mean = _coerce_float(row_metadata.get(f"{metric}_mean"))
        if table_mean is not None:
            out[f"{metric}_source_table_mean"] = table_mean
    return out


def _sanitize_json_values(value: Any) -> Any:
    """Recursively replace non-finite floats with null-safe values for JSON output."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _sanitize_json_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_json_values(item) for item in value]
    return value


def build_paper_results_handoff_payload(
    source_path: Path,
    *,
    confidence_settings: Mapping[str, Any] | None = None,
    metrics: Sequence[str] = PAPER_RESULTS_HANDOFF_METRICS,
) -> dict[str, Any]:
    """Build an interval-inclusive paper Results handoff payload from a frozen source."""
    resolved = _resolve_source_path(source_path)
    publication_manifest = (
        _load_json(resolved.publication_manifest_path)
        if resolved.publication_manifest_path is not None
        else {}
    )
    seed_payload = _load_json(
        resolved.payload_root / "reports" / "seed_variability_by_scenario.json"
    )
    campaign_rows = _campaign_table_index(resolved.payload_root)
    episode_paths = _episode_jsonl_paths(resolved)
    campaign_id = _campaign_id(
        resolved,
        seed_payload=seed_payload,
        publication_manifest=publication_manifest,
    )
    resolved_confidence = dict(DEFAULT_CONFIDENCE_SETTINGS)
    resolved_confidence.update(dict(confidence_settings or {}))
    row_seed_policy = _seed_policy(seed_payload)
    config_hash = _config_hash(seed_payload, publication_manifest)
    git_hash = _git_hash(seed_payload, publication_manifest)

    rows: list[dict[str, Any]] = []
    discovered_seeds: set[int] = set()
    for episodes_path in episode_paths:
        run_records, run_seeds = _load_run_records(
            episodes_path,
            metrics=metrics,
            campaign_rows=campaign_rows,
        )
        discovered_seeds.update(run_seeds)
        if not run_records:
            continue
        rows.extend(
            build_seed_variability_rows(
                run_records,
                metrics=metrics,
                campaign_id=campaign_id,
                config_hash=config_hash,
                git_hash=git_hash,
                seed_policy=row_seed_policy,
                confidence_settings=resolved_confidence,
            )
        )

    if not rows:
        raise ValueError(f"No episode records found under {resolved.payload_root / 'runs'}")

    seed_policy = _seed_policy(seed_payload, discovered_seeds)

    row_index = {(str(row.get("planner_key")), str(row.get("kinematics"))): row for row in rows}
    ordered_keys = [key for key in campaign_rows if key in row_index]
    ordered_keys.extend(key for key in sorted(row_index) if key not in set(ordered_keys))

    handoff_rows = [
        _flatten_handoff_row(
            row_index[key],
            row_metadata=campaign_rows.get(key, {}),
            metrics=metrics,
            seed_policy_override=seed_policy,
        )
        for key in ordered_keys
    ]

    return {
        "schema_version": PAPER_RESULTS_HANDOFF_SCHEMA_VERSION,
        "campaign_id": campaign_id,
        "source": _source_paths(resolved),
        "metrics": list(metrics),
        "confidence": resolved_confidence,
        "seed_policy": seed_policy,
        "row_count": len(handoff_rows),
        "rows": handoff_rows,
    }


def _csv_fieldnames(metrics: Iterable[str]) -> list[str]:
    """Return stable CSV field names for handoff rows."""
    fields = [
        "planner_key",
        "algo",
        "planner_group",
        "kinematics",
        "status",
        "readiness_tier",
        "readiness_status",
        "preflight_status",
        "episode_count",
        "seed_count",
        "repeat_count",
        "seed_list",
        "campaign_id",
        "config_hash",
        "git_hash",
        "seed_policy_mode",
        "seed_policy_seed_set",
        "confidence_method",
        "confidence_level",
        "bootstrap_samples",
        "bootstrap_seed",
    ]
    for metric in metrics:
        fields.extend(
            [
                f"{metric}_mean",
                f"{metric}_std",
                f"{metric}_count",
                f"{metric}_ci_low",
                f"{metric}_ci_high",
                f"{metric}_ci_half_width",
                f"{metric}_source_table_mean",
            ]
        )
    return fields


def _csv_value(value: Any) -> Any:
    """Convert JSON row values to CSV-friendly cells."""
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    return value


def write_paper_results_handoff(
    payload: Mapping[str, Any],
    output_dir: Path,
) -> PaperResultsHandoffResult:
    """Write handoff JSON and CSV files to ``output_dir``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "paper_results_handoff.json"
    csv_path = output_dir / "paper_results_handoff.csv"
    sanitized_payload = _sanitize_json_values(payload)
    if not isinstance(sanitized_payload, Mapping):
        raise TypeError("paper results handoff payload must be a mapping")
    json_path.write_text(json.dumps(sanitized_payload, indent=2) + "\n", encoding="utf-8")

    metrics = (
        sanitized_payload.get("metrics")
        if isinstance(sanitized_payload.get("metrics"), list)
        else []
    )
    fieldnames = _csv_fieldnames(str(metric) for metric in metrics)
    rows = sanitized_payload.get("rows") if isinstance(sanitized_payload.get("rows"), list) else []
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            if isinstance(row, Mapping):
                writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})

    return PaperResultsHandoffResult(
        output_dir=output_dir,
        json_path=json_path,
        csv_path=csv_path,
        row_count=len(rows),
    )


def export_paper_results_handoff(
    source_path: Path,
    *,
    output_dir: Path | None = None,
    confidence_settings: Mapping[str, Any] | None = None,
    metrics: Sequence[str] = PAPER_RESULTS_HANDOFF_METRICS,
) -> PaperResultsHandoffResult:
    """Build and write a paper Results handoff from a campaign or publication bundle."""
    payload = build_paper_results_handoff_payload(
        source_path,
        confidence_settings=confidence_settings,
        metrics=metrics,
    )
    if output_dir is None:
        output_dir = (
            get_artifact_category_path("benchmarks")
            / "publication"
            / f"{payload['campaign_id']}_paper_results_handoff"
        )
    return write_paper_results_handoff(payload, output_dir)


__all__ = [
    "DEFAULT_CONFIDENCE_SETTINGS",
    "PAPER_RESULTS_HANDOFF_METRICS",
    "PAPER_RESULTS_HANDOFF_SCHEMA_VERSION",
    "PaperResultsHandoffResult",
    "build_paper_results_handoff_payload",
    "export_paper_results_handoff",
    "write_paper_results_handoff",
]
