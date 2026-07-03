"""Paired proxemic-layer ablation reporting helpers."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

USABLE_ROW_STATUSES = frozenset({"native", "adapter", "success", "completed", "ok"})
BLOCKED_ROW_STATUSES = frozenset({"fallback", "degraded", "failed", "not_available", "blocked"})
PAIR_KEYS = ("scenario_id", "seed")
REPORT_METRICS = (
    "intrusion_rate",
    "minimum_clearance",
    "path_efficiency",
    "runtime_seconds",
)
REPORT_CAVEATS = ("success", "collision")

_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "intrusion_rate": (
        "intrusion_rate",
        "proxemic_intrusion_rate",
        "social_space_intrusion_rate",
        "metrics.intrusion_rate",
        "metrics.proxemic_intrusion_rate",
        "metrics.social_space_intrusion_rate",
    ),
    "minimum_clearance": (
        "minimum_clearance",
        "min_clearance",
        "min_distance",
        "clearing_distance_min",
        "metrics.minimum_clearance",
        "metrics.min_clearance",
        "metrics.min_distance",
        "metrics.clearing_distance_min",
    ),
    "path_efficiency": (
        "path_efficiency",
        "metrics.path_efficiency",
    ),
    "runtime_seconds": (
        "runtime_seconds",
        "wall_time_seconds",
        "episode_runtime_seconds",
        "episode_sec",
        "metrics.runtime_seconds",
        "metrics.wall_time_seconds",
        "metrics.episode_runtime_seconds",
        "metrics.episode_sec",
    ),
    "success": (
        "success",
        "metrics.success",
    ),
    "collision": (
        "collision",
        "collisions",
        "metrics.collision",
        "metrics.collisions",
    ),
}


@dataclass(frozen=True, slots=True)
class ArmSummary:
    """Aggregate metric summary for one paired arm."""

    arm: str
    rows: int
    intrusion_rate: float
    minimum_clearance: float
    path_efficiency: float
    runtime_seconds: float
    success_rate: float
    collision_rate: float


def load_records(path: Path) -> list[dict[str, Any]]:
    """Load JSON or JSONL episode rows.

    Returns:
        Episode row dictionaries.
    """

    if not path.exists():
        raise FileNotFoundError(f"episode rows not found: {path}")

    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    else:
        loaded = json.loads(path.read_text(encoding="utf-8"))
        rows = loaded.get("episodes", loaded) if isinstance(loaded, Mapping) else loaded

    if not isinstance(rows, list):
        raise ValueError(f"episode rows must be a list or object with 'episodes': {path}")
    if not all(isinstance(row, Mapping) for row in rows):
        raise ValueError(f"episode rows must contain JSON objects: {path}")
    return [dict(row) for row in rows]


def load_yaml_file(path: Path) -> dict[str, Any]:
    """Load a YAML mapping from ``path``.

    Returns:
        Parsed YAML mapping.
    """

    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):
        raise ValueError(f"config must be a YAML mapping: {path}")
    return dict(loaded)


def build_proxemic_ablation_report(
    *,
    baseline_records: Sequence[Mapping[str, Any]],
    proxemic_records: Sequence[Mapping[str, Any]],
    smoke_config_path: Path,
    proxemic_config_path: Path,
    repo_root: Path = Path("."),
) -> dict[str, Any]:
    """Build a fail-closed paired proxemic-layer ablation report.

    Returns:
        Machine-readable report dictionary.
    """

    smoke_config = load_yaml_file(repo_root / smoke_config_path)
    proxemic_config = load_yaml_file(repo_root / proxemic_config_path)
    blockers: list[str] = []

    baseline_index = _index_rows("baseline_classical", baseline_records, blockers)
    proxemic_index = _index_rows("proxemic_costmap_on", proxemic_records, blockers)
    shared_keys = sorted(set(baseline_index) & set(proxemic_index))
    if not shared_keys:
        blockers.append("no paired rows share scenario_id and seed")

    paired_baseline = [baseline_index[key] for key in shared_keys]
    paired_proxemic = [proxemic_index[key] for key in shared_keys]
    _validate_metric_fields("baseline_classical", paired_baseline, blockers)
    _validate_metric_fields("proxemic_costmap_on", paired_proxemic, blockers)

    baseline_summary = _summarize_arm("baseline_classical", paired_baseline)
    proxemic_summary = _summarize_arm("proxemic_costmap_on", paired_proxemic)
    if baseline_summary.runtime_seconds <= 0.0:
        blockers.append("baseline_classical runtime_seconds must be positive for ratio computation")

    report_status = "blocked" if blockers else "ready"

    deltas = {
        "intrusion_rate_delta": proxemic_summary.intrusion_rate - baseline_summary.intrusion_rate,
        "minimum_clearance_delta": (
            proxemic_summary.minimum_clearance - baseline_summary.minimum_clearance
        ),
        "path_efficiency_delta": proxemic_summary.path_efficiency
        - baseline_summary.path_efficiency,
        "runtime_overhead_seconds": proxemic_summary.runtime_seconds
        - baseline_summary.runtime_seconds,
        "runtime_overhead_ratio": _safe_ratio(
            proxemic_summary.runtime_seconds,
            baseline_summary.runtime_seconds,
        ),
        "success_rate_delta": proxemic_summary.success_rate - baseline_summary.success_rate,
        "collision_rate_delta": proxemic_summary.collision_rate - baseline_summary.collision_rate,
    }

    return {
        "issue": 4165,
        "report_status": report_status,
        "claim_boundary": "paired_cpu_smoke_or_fixture_report_only",
        "paired_key_fields": list(PAIR_KEYS),
        "paired_rows": len(shared_keys),
        "blocked_reasons": blockers,
        "arms": {
            "baseline_classical": _arm_summary_dict(baseline_summary),
            "proxemic_costmap_on": _arm_summary_dict(proxemic_summary),
        },
        "deltas": deltas,
        "success_collision_caveats": {
            "baseline_success_rate": baseline_summary.success_rate,
            "proxemic_success_rate": proxemic_summary.success_rate,
            "baseline_collision_rate": baseline_summary.collision_rate,
            "proxemic_collision_rate": proxemic_summary.collision_rate,
            "interpretation": (
                "Diagnostic paired smoke deltas only; fallback/degraded rows block readiness and "
                "are not treated as successful benchmark evidence."
            ),
        },
        "parameter_provenance": {
            "smoke_config": _config_provenance(repo_root / smoke_config_path, smoke_config),
            "proxemic_config": _config_provenance(
                repo_root / proxemic_config_path, proxemic_config
            ),
        },
        "out_of_scope": [
            "full benchmark campaign",
            "Slurm or GPU submission",
            "paper-facing claim update",
        ],
    }


def write_report_artifacts(report: Mapping[str, Any], output_dir: Path) -> None:
    """Write machine-readable and Markdown report artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(format_markdown_report(report), encoding="utf-8")


def format_markdown_report(report: Mapping[str, Any]) -> str:
    """Format a compact Markdown summary for context evidence.

    Returns:
        Markdown report text.
    """

    deltas = report["deltas"]
    provenance = report["parameter_provenance"]["proxemic_config"]
    lines = [
        "# Issue #4165 proxemic-layer ablation report",
        "",
        f"Status: `{report['report_status']}`",
        "",
        "This is a paired CPU smoke or fixture report only. It is not a full benchmark campaign "
        "and does not promote paper-facing claims.",
        "",
        "| Field | Delta |",
        "| --- | ---: |",
        f"| intrusion_rate_delta | {deltas['intrusion_rate_delta']:.6g} |",
        f"| minimum_clearance_delta | {deltas['minimum_clearance_delta']:.6g} |",
        f"| path_efficiency_delta | {deltas['path_efficiency_delta']:.6g} |",
        f"| runtime_overhead_seconds | {deltas['runtime_overhead_seconds']:.6g} |",
        f"| runtime_overhead_ratio | {deltas['runtime_overhead_ratio']:.6g} |",
        f"| success_rate_delta | {deltas['success_rate_delta']:.6g} |",
        f"| collision_rate_delta | {deltas['collision_rate_delta']:.6g} |",
        "",
        "## Parameter Provenance",
        "",
        f"- proxemic config: `{provenance['path']}`",
        f"- proxemic config sha256: `{provenance['sha256']}`",
        "",
        "## Blockers",
        "",
    ]
    blockers = report["blocked_reasons"]
    lines.extend(f"- {blocker}" for blocker in blockers)
    if not blockers:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def _index_rows(
    arm: str,
    records: Sequence[Mapping[str, Any]],
    blockers: list[str],
) -> dict[tuple[str, str], Mapping[str, Any]]:
    index: dict[tuple[str, str], Mapping[str, Any]] = {}
    for idx, record in enumerate(records):
        status = str(record.get("row_status", record.get("status", "native"))).lower()
        if status in BLOCKED_ROW_STATUSES or status not in USABLE_ROW_STATUSES:
            blockers.append(f"{arm} row {idx} has non-usable row_status={status!r}")
        key = _pair_key(record)
        if key is None:
            blockers.append(f"{arm} row {idx} missing pair key fields: {', '.join(PAIR_KEYS)}")
            continue
        if key in index:
            blockers.append(f"{arm} has duplicate pair key scenario_id={key[0]!r} seed={key[1]!r}")
        index[key] = record
    return index


def _pair_key(record: Mapping[str, Any]) -> tuple[str, str] | None:
    missing = [field for field in PAIR_KEYS if field not in record]
    if missing:
        return None
    return (str(record["scenario_id"]), str(record["seed"]))


def _validate_metric_fields(
    arm: str,
    records: Sequence[Mapping[str, Any]],
    blockers: list[str],
) -> None:
    for idx, record in enumerate(records):
        missing = [metric for metric in REPORT_METRICS if _extract_float(record, metric) is None]
        missing.extend(metric for metric in REPORT_CAVEATS if _get_alias(record, metric) is None)
        if missing:
            blockers.append(f"{arm} paired row {idx} missing metrics: {', '.join(missing)}")
        invalid_caveats = [
            metric
            for metric in REPORT_CAVEATS
            if _get_alias(record, metric) is not None and _extract_bool(record, metric) is None
        ]
        if invalid_caveats:
            blockers.append(
                f"{arm} paired row {idx} invalid caveat values: {', '.join(invalid_caveats)}"
            )


def _summarize_arm(arm: str, records: Sequence[Mapping[str, Any]]) -> ArmSummary:
    return ArmSummary(
        arm=arm,
        rows=len(records),
        intrusion_rate=_mean(_required_values(records, "intrusion_rate")),
        minimum_clearance=_mean(_required_values(records, "minimum_clearance")),
        path_efficiency=_mean(_required_values(records, "path_efficiency")),
        runtime_seconds=_mean(_required_values(records, "runtime_seconds")),
        success_rate=_mean(_bool_values(records, "success")),
        collision_rate=_mean(_bool_values(records, "collision")),
    )


def _required_values(records: Sequence[Mapping[str, Any]], metric: str) -> list[float]:
    return [value for record in records if (value := _extract_float(record, metric)) is not None]


def _bool_values(records: Sequence[Mapping[str, Any]], field: str) -> list[float]:
    values: list[float] = []
    for record in records:
        parsed = _extract_bool(record, field)
        if parsed is None:
            continue
        values.append(float(parsed))
    return values


def _extract_bool(record: Mapping[str, Any], field: str) -> bool | None:
    raw = _get_alias(record, field)
    if raw is None:
        return None
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        if not math.isfinite(float(raw)):
            return None
        return float(raw) > 0.0
    if isinstance(raw, str):
        normalized = raw.strip().lower()
        if normalized in {"true", "t", "yes", "y", "1", "pass", "passed", "success"}:
            return True
        if normalized in {"false", "f", "no", "n", "0", "fail", "failed", "none"}:
            return False
    return None


def _extract_float(record: Mapping[str, Any], metric: str) -> float | None:
    raw = _get_alias(record, metric)
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def _get_alias(record: Mapping[str, Any], metric: str) -> Any | None:
    for alias in _FIELD_ALIASES[metric]:
        value = _get_nested(record, alias)
        if value is not None:
            return value
    return None


def _get_nested(record: Mapping[str, Any], dotted: str) -> Any | None:
    current: Any = record
    for part in dotted.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _mean(values: Iterable[float]) -> float:
    value_list = list(values)
    if not value_list:
        return 0.0
    return sum(value_list) / len(value_list)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


def _arm_summary_dict(summary: ArmSummary) -> dict[str, float | int | str]:
    return {
        "arm": summary.arm,
        "rows": summary.rows,
        "intrusion_rate": summary.intrusion_rate,
        "minimum_clearance": summary.minimum_clearance,
        "path_efficiency": summary.path_efficiency,
        "runtime_seconds": summary.runtime_seconds,
        "success_rate": summary.success_rate,
        "collision_rate": summary.collision_rate,
    }


def _config_provenance(path: Path, parsed: Mapping[str, Any]) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "path": str(path),
        "sha256": hashlib.sha256(data).hexdigest(),
        "parameters": dict(parsed),
    }
