"""Adversarial manifest quality metrics for bounded, conservative diagnosis.

The metrics in this module are explicitly generator quality/pre-benchmark signals.
They are not planner benchmark evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import yaml

from robot_sf.adversarial.config import CandidateSpec, Pose2D
from robot_sf.adversarial.scenario_manifest import compute_control_hash

MANIFEST_QUALITY_SCHEMA_VERSION = "adversarial_manifest_quality_summary.v1"
EVIDENCE_BOUNDARY = (
    "quality-signal-only: manifests are generator health signals; "
    "this module does not make planner benchmark claims."
)

VALID_STATUSES = ("valid", "invalid", "degenerate")
_CONTROL_FIELDS = (
    ("start", "x"),
    ("start", "y"),
    ("goal", "x"),
    ("goal", "y"),
    ("spawn_time_s",),
    ("pedestrian_speed_mps",),
    ("pedestrian_delay_s",),
)


def _safe_float(value: Any) -> float | None:
    """Return a finite float, otherwise ``None``."""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _safe_int(value: Any) -> int | None:
    """Return an int when coercion is exact, otherwise ``None``."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return int(value)
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed) or not parsed.is_integer():
        return None
    return int(parsed)


def _safe_rate(numerator: int, denominator: int) -> float:
    """Return a bounded finite rate."""
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 6)


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    """Load a YAML mapping from path."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Manifest payload must be a mapping: {path}")
    return payload


def _read_text_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSONL rows, ignoring malformed lines."""
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        try:
            payload = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _collect_paths(paths: list[Path]) -> list[Path]:
    """Expand files and directories into candidate manifest paths."""
    manifest_paths: list[Path] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Manifest input missing: {path}")
        if path.is_dir():
            manifest_paths.extend(sorted(path.rglob("*.yaml")) + sorted(path.rglob("*.yml")) + [])
            continue
        if path.is_file():
            if path.suffix.lower() in {".yaml", ".yml"}:
                manifest_paths.append(path)
                continue
            raise ValueError(f"Manifest file must use .yml or .yaml: {path}")
        raise ValueError(f"Manifest input is not a file or directory: {path}")
    if not manifest_paths:
        raise ValueError("No manifest files discovered from inputs")
    return manifest_paths


def _nested_value(payload: Any, path: tuple[str, ...]) -> Any:
    """Read a nested value from nested dict payload."""
    current = payload
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _same_path(left: Path, right: Path) -> bool:
    """Return whether two paths resolve to the same filesystem location."""
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return left == right


def _candidate_from_controls(controls: Any) -> CandidateSpec:
    """Build a deterministic candidate from controls mapping."""
    if not isinstance(controls, dict):
        raise ValueError("candidate_controls must be a mapping")
    start = _nested_value(controls, ("start",))
    goal = _nested_value(controls, ("goal",))
    if not isinstance(start, dict) or not isinstance(goal, dict):
        raise ValueError("candidate_controls.start and candidate_controls.goal must be mappings")
    start_x = _safe_float(start.get("x"))
    start_y = _safe_float(start.get("y"))
    goal_x = _safe_float(goal.get("x"))
    goal_y = _safe_float(goal.get("y"))
    if start_x is None or start_y is None or goal_x is None or goal_y is None:
        raise ValueError("candidate_controls.start/goal must include finite x and y")

    spawn_time = _safe_float(controls.get("spawn_time_s"))
    speed = _safe_float(controls.get("pedestrian_speed_mps"))
    delay = _safe_float(controls.get("pedestrian_delay_s"))
    seed = _safe_int(controls.get("scenario_seed"))
    if spawn_time is None or speed is None or delay is None or seed is None:
        raise ValueError(
            "candidate_controls must include finite spawn_time_s/pedestrian_speed_mps/"
            "pedestrian_delay_s and integer scenario_seed"
        )
    return CandidateSpec(
        start=Pose2D(start_x, start_y),
        goal=Pose2D(goal_x, goal_y),
        spawn_time_s=spawn_time,
        pedestrian_speed_mps=speed,
        pedestrian_delay_s=delay,
        scenario_seed=seed,
    )


def _control_vector(controls: Any) -> tuple[float, ...] | None:
    """Return ordered control vector for perturbation computation."""
    values: list[float] = []
    if not isinstance(controls, dict):
        return None
    for key_path in _CONTROL_FIELDS:
        raw = _nested_value(controls, key_path)
        value = _safe_float(raw)
        if value is None:
            return None
        values.append(value)
    return tuple(values)


@dataclass(frozen=True)
class ManifestQualityRecord:
    """One parsed manifest summary row."""

    path: str
    status: str
    schema_version: str | None
    normalized_control_hash: str | None = None
    parse_error: str | None = None
    perturbation_distance: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe representation."""
        payload: dict[str, Any] = {
            "path": self.path,
            "status": self.status,
            "schema_version": self.schema_version,
        }
        if self.normalized_control_hash is not None:
            payload["normalized_control_hash"] = self.normalized_control_hash
        if self.parse_error is not None:
            payload["parse_error"] = self.parse_error
        if self.perturbation_distance is not None:
            payload["perturbation_distance"] = self.perturbation_distance
        return payload


@dataclass(frozen=True)
class PlannerOutcome:
    """Per-planner diagnostic outcome yields."""

    planner: str
    episodes: int | None
    failure_count: int | None
    near_miss_count: int | None
    failure_yield: float | None
    near_miss_yield: float | None
    source: str
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe representation."""
        payload = {
            "planner": self.planner,
            "episodes": self.episodes,
            "failure_count": self.failure_count,
            "near_miss_count": self.near_miss_count,
            "failure_yield": self.failure_yield,
            "near_miss_yield": self.near_miss_yield,
            "source": self.source,
        }
        if self.note is not None:
            payload["note"] = self.note
        return payload


@dataclass(frozen=True)
class PlannerOutcomeSummary:
    """Compact planner-yield payload parsed from optional smoke summary."""

    source_summary: str
    available: bool
    planners: list[PlannerOutcome]

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe representation."""
        return {
            "available": self.available,
            "source_summary": self.source_summary,
            "planners": [planner.to_dict() for planner in self.planners],
        }


@dataclass(frozen=True)
class ManifestsQualitySummary:
    """Top-level manifest quality summary."""

    input_paths: list[str]
    manifest_count: int
    parse_failures: int
    status_counts: dict[str, int]
    validity_rate: float
    invalid_rate: float
    degenerate_rate: float
    hashable_count: int
    unique_hash_count: int
    duplicate_hash_count: int
    novelty_rate: float
    duplicate_rate: float
    perturbation_reference: str | None
    perturbation_count: int
    perturbation_mean: float | None
    perturbation_min: float | None
    perturbation_max: float | None
    planner_outcomes: PlannerOutcomeSummary | None

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe representation."""
        payload: dict[str, Any] = {
            "schema_version": MANIFEST_QUALITY_SCHEMA_VERSION,
            "evidence_boundary": EVIDENCE_BOUNDARY,
            "manifest_count": self.manifest_count,
            "parse_failures": self.parse_failures,
            "status_counts": dict(self.status_counts),
            "rates": {
                "validity_rate": self.validity_rate,
                "invalid_rate": self.invalid_rate,
                "degenerate_rate": self.degenerate_rate,
            },
            "novelty": {
                "hashable_count": self.hashable_count,
                "unique_hash_count": self.unique_hash_count,
                "duplicate_hash_count": self.duplicate_hash_count,
                "novelty_rate": self.novelty_rate,
                "duplicate_rate": self.duplicate_rate,
            },
            "perturbation_from_reference": {
                "reference_manifest": self.perturbation_reference,
                "count": self.perturbation_count,
                "mean_distance": self.perturbation_mean,
                "min_distance": self.perturbation_min,
                "max_distance": self.perturbation_max,
            },
            "manifest_inputs": [str(path) for path in self.input_paths],
        }
        if self.planner_outcomes is not None:
            payload["planner_outcomes"] = self.planner_outcomes.to_dict()
        return payload


def _manifest_status(payload: dict[str, Any]) -> str:
    """Normalize validation status to one of known buckets."""
    validation = payload.get("validation")
    if not isinstance(validation, dict):
        return "invalid"
    status = str(validation.get("status", "invalid")).strip().lower()
    return status if status in VALID_STATUSES else "invalid"


def _manifest_hash(payload: dict[str, Any]) -> str | None:
    """Read or compute normalized control hash."""
    controls = payload.get("candidate_controls")
    if not isinstance(controls, dict):
        return None
    validation = payload.get("validation")
    if isinstance(validation, dict):
        raw_hash = validation.get("normalized_control_hash")
        if isinstance(raw_hash, str) and raw_hash:
            return raw_hash
    try:
        candidate = _candidate_from_controls(controls)
    except ValueError:
        return None
    return compute_control_hash(candidate)


def _load_records(
    manifest_paths: list[Path],
    reference_vector: tuple[float, ...] | None,
    reference_path: Path | None = None,
) -> tuple[list[ManifestQualityRecord], list[str]]:
    """Parse manifests and return records plus collected parse problems."""
    records: list[ManifestQualityRecord] = []
    parse_errors: list[str] = []
    for manifest_path in manifest_paths:
        if reference_path is not None and _same_path(manifest_path, reference_path):
            continue
        try:
            payload = _load_yaml_mapping(manifest_path)
            controls = payload.get("candidate_controls")
            if not isinstance(controls, dict):
                raise ValueError("candidate_controls missing or not a mapping")
            status = _manifest_status(payload)
            status = status if status in VALID_STATUSES else "invalid"
            manifest_hash = _manifest_hash(payload)
            perturbation_distance = None
            if reference_vector is not None:
                candidate_vector = _control_vector(controls)
                if candidate_vector is not None:
                    if len(candidate_vector) == len(reference_vector):
                        delta_sq = sum(
                            (candidate - reference) ** 2
                            for candidate, reference in zip(
                                candidate_vector,
                                reference_vector,
                                strict=True,
                            )
                        )
                        perturbation_distance = round(math.sqrt(delta_sq), 6)
            records.append(
                ManifestQualityRecord(
                    path=manifest_path.as_posix(),
                    status=status,
                    schema_version=str(payload.get("schema_version")),
                    normalized_control_hash=manifest_hash,
                    perturbation_distance=perturbation_distance,
                )
            )
        except (OSError, ValueError) as exc:
            records.append(
                ManifestQualityRecord(
                    path=manifest_path.as_posix(),
                    status="invalid",
                    schema_version=None,
                    parse_error=str(exc),
                )
            )
            parse_errors.append(f"{manifest_path.as_posix()}: {exc}")
    return records, parse_errors


def _summarize_hashes(records: list[ManifestQualityRecord]) -> tuple[int, int, int, float, float]:
    """Return hashability and novelty statistics."""
    hashes = [
        record.normalized_control_hash for record in records if record.normalized_control_hash
    ]
    hash_counts = Counter(hashes)
    hashable_count = len(hashes)
    unique_hash_count = len(hash_counts)
    duplicate_hash_count = hashable_count - unique_hash_count
    if hashable_count == 0:
        return 0, 0, 0, 0.0, 0.0
    novelty_rate = round(unique_hash_count / hashable_count, 6)
    duplicate_rate = round(duplicate_hash_count / hashable_count, 6)
    return (
        hashable_count,
        unique_hash_count,
        duplicate_hash_count,
        novelty_rate,
        duplicate_rate,
    )


def _summarize_perturbations(
    records: list[ManifestQualityRecord],
) -> tuple[int, float | None, float | None, float | None]:
    """Summarize perturbation distance values."""
    distances = [
        record.perturbation_distance
        for record in records
        if record.perturbation_distance is not None
    ]
    if not distances:
        return 0, None, None, None
    return (
        len(distances),
        round(mean(distances), 6),
        round(min(distances), 6),
        round(max(distances), 6),
    )


def _row_is_failure(row: dict[str, Any]) -> bool:
    """Return whether one episode row should count as failure."""
    status = str(row.get("status", "")).strip().lower()
    if status:
        return status != "success"
    termination = str(row.get("termination_reason", "")).strip().lower()
    return bool(termination) and termination != "success"


def _has_near_miss(row: dict[str, Any]) -> bool:
    """Return whether one episode row reports near-miss signal."""
    metrics = row.get("metrics")
    if not isinstance(metrics, dict):
        return False
    near_miss_value = _safe_float(metrics.get("near_misses"))
    return near_miss_value is not None and near_miss_value > 0.0


def _metric_stat(metrics: dict[str, Any], metric_name: str, stat_name: str) -> float | None:
    """Read a finite metric statistic from an aggregate metric payload."""
    raw_metric = metrics.get(metric_name)
    if not isinstance(raw_metric, dict):
        return None
    return _safe_float(raw_metric.get(stat_name))


def _count_like_sum(value: float | None, episodes: int | None) -> int | None:
    """Return an integer count when an aggregate metric sum has count semantics."""
    if value is None or episodes is None:
        return None
    rounded = round(value)
    if abs(value - rounded) > 1e-6 or rounded < 0 or rounded > episodes:
        return None
    return rounded


def _summarize_planner_outcomes(
    smoke_summary_path: Path | None,
) -> PlannerOutcomeSummary | None:
    """Summarize failure and near-miss yields from planner smoke episodes."""
    if smoke_summary_path is None:
        return None
    summary_payload = json.loads(smoke_summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary_payload, dict):
        raise ValueError(f"Smoke summary must be a JSON object: {smoke_summary_path}")
    raw_planner_runs = summary_payload.get("planner_runs")
    if not isinstance(raw_planner_runs, list):
        return PlannerOutcomeSummary(
            source_summary=smoke_summary_path.as_posix(),
            available=False,
            planners=[],
        )

    planners: list[PlannerOutcome] = []
    for run in raw_planner_runs:
        if not isinstance(run, dict):
            continue
        planner_outcome = _summarize_single_planner_run(run, smoke_summary_path=smoke_summary_path)
        if planner_outcome is not None:
            planners.append(planner_outcome)

    return PlannerOutcomeSummary(
        source_summary=smoke_summary_path.as_posix(),
        available=any(
            planner.source in {"episode_rows", "aggregate_metrics"} for planner in planners
        ),
        planners=planners,
    )


def _summarize_single_planner_run(
    run: dict[str, Any],
    *,
    smoke_summary_path: Path,
) -> PlannerOutcome | None:
    """Build one planner outcome item."""
    planner = str(run.get("planner", "unknown"))
    out_path = run.get("out_path")
    out_rows_path = Path(str(out_path)) if isinstance(out_path, str) else None
    if out_rows_path is not None and not out_rows_path.is_absolute():
        out_rows_path = smoke_summary_path.parent / out_rows_path
    if out_rows_path is None or not out_rows_path.exists():
        return _summarize_planner_run_from_aggregates(run, smoke_summary_path=smoke_summary_path)

    rows = _read_text_jsonl(out_rows_path)
    if not rows:
        return PlannerOutcome(
            planner=planner,
            episodes=_safe_int(run.get("total_jobs")),
            failure_count=0,
            near_miss_count=0,
            failure_yield=0.0,
            near_miss_yield=0.0,
            source="no_rows",
            note="episodes.jsonl exists but is empty",
        )

    failures = 0
    near_misses = 0
    for row in rows:
        if _row_is_failure(row):
            failures += 1
        if _has_near_miss(row):
            near_misses += 1
    total = len(rows)
    return PlannerOutcome(
        planner=planner,
        episodes=total,
        failure_count=failures,
        near_miss_count=near_misses,
        failure_yield=_safe_rate(failures, total),
        near_miss_yield=_safe_rate(near_misses, total),
        source="episode_rows",
    )


def _summarize_planner_run_from_aggregates(
    run: dict[str, Any],
    *,
    smoke_summary_path: Path,
) -> PlannerOutcome:
    """Build one planner outcome item from aggregate smoke-summary metrics."""
    planner = str(run.get("planner", "unknown"))
    metrics = run.get("metrics")
    if not isinstance(metrics, dict):
        return PlannerOutcome(
            planner=planner,
            episodes=_safe_int(run.get("total_jobs")),
            failure_count=None,
            near_miss_count=None,
            failure_yield=None,
            near_miss_yield=None,
            source="planner_summary_missing_rows",
            note=f"{smoke_summary_path.as_posix()} has no readable rows or aggregate metrics",
        )

    episodes = _safe_int(metrics.get("episodes"))
    if episodes is None:
        episodes = _safe_int(run.get("written")) or _safe_int(run.get("total_jobs"))
    success_count = _count_like_sum(_metric_stat(metrics, "success", "sum"), episodes)
    failure_count = None
    failure_yield = None
    if episodes is not None and success_count is not None:
        failure_count = episodes - success_count
        failure_yield = _safe_rate(failure_count, episodes)

    near_miss_count_value = _count_like_sum(_metric_stat(metrics, "near_misses", "sum"), episodes)
    near_miss_count = None
    near_miss_yield = None
    if episodes is not None and near_miss_count_value is not None:
        near_miss_count = near_miss_count_value
        near_miss_yield = _safe_rate(near_miss_count, episodes)

    return PlannerOutcome(
        planner=planner,
        episodes=episodes,
        failure_count=failure_count,
        near_miss_count=near_miss_count,
        failure_yield=failure_yield,
        near_miss_yield=near_miss_yield,
        source="aggregate_metrics",
        note=(
            "failure_yield derived from count-like aggregate success sums; near_miss_yield "
            "is available only when count-like aggregate near_misses is present"
        ),
    )


def summarize_adversarial_manifest_quality(
    manifest_inputs: list[str | Path],
    *,
    reference_manifest: str | Path | None = None,
    smoke_summary_json: str | Path | None = None,
) -> ManifestsQualitySummary:
    """Return a compact quality summary for a batch of manifests."""
    input_paths = [Path(path) for path in manifest_inputs]
    manifest_paths = _collect_paths(input_paths)
    reference_vector: tuple[float, ...] | None = None
    reference_manifest_path = None
    if reference_manifest is not None:
        reference_manifest_path = Path(reference_manifest)
        reference_payload = _load_yaml_mapping(reference_manifest_path)
        reference_vector = _control_vector(reference_payload.get("candidate_controls"))
        if reference_vector is None:
            raise ValueError(
                "reference manifest missing comparable candidate_controls: "
                f"{reference_manifest_path}"
            )

    records, parse_errors = _load_records(manifest_paths, reference_vector, reference_manifest_path)
    status_counts = Counter(record.status for record in records)
    valid = status_counts.get("valid", 0)
    invalid = status_counts.get("invalid", 0)
    degenerate = status_counts.get("degenerate", 0)
    total = len(records)

    hashable_count, unique_hash_count, duplicate_hash_count, novelty_rate, duplicate_rate = (
        _summarize_hashes(records)
    )
    perturbation_count, perturbation_mean, perturbation_min, perturbation_max = (
        _summarize_perturbations(records)
    )

    planner_outcomes = _summarize_planner_outcomes(
        Path(smoke_summary_json) if smoke_summary_json is not None else None
    )

    return ManifestsQualitySummary(
        input_paths=sorted({record.path for record in records}),
        manifest_count=total,
        parse_failures=len(parse_errors),
        status_counts=dict(status_counts),
        validity_rate=_safe_rate(valid, total),
        invalid_rate=_safe_rate(invalid, total),
        degenerate_rate=_safe_rate(degenerate, total),
        hashable_count=hashable_count,
        unique_hash_count=unique_hash_count,
        duplicate_hash_count=duplicate_hash_count,
        novelty_rate=novelty_rate,
        duplicate_rate=duplicate_rate,
        perturbation_reference=(
            reference_manifest_path.as_posix() if reference_manifest_path else None
        ),
        perturbation_count=perturbation_count,
        perturbation_mean=perturbation_mean,
        perturbation_min=perturbation_min,
        perturbation_max=perturbation_max,
        planner_outcomes=planner_outcomes,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(
        description="Summarize quality signals from adversarial manifest batches."
    )
    parser.add_argument(
        "manifests",
        nargs="+",
        type=Path,
        help="Manifest YAML files and/or directories.",
    )
    parser.add_argument(
        "--reference-manifest",
        type=Path,
        default=None,
        help="Optional reference manifest for perturbation distance.",
    )
    parser.add_argument(
        "--smoke-summary-json",
        type=Path,
        default=None,
        help="Optional planner smoke summary JSON for failure/near-miss yields.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Write JSON output to this path; if omitted, print to stdout.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = summarize_adversarial_manifest_quality(
            manifest_inputs=[*args.manifests],
            reference_manifest=args.reference_manifest,
            smoke_summary_json=args.smoke_summary_json,
        )
        payload = summary.to_dict()
        text = json.dumps(payload, indent=2, sort_keys=True)
        output = ""
        if args.output_json is not None:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(text + "\n", encoding="utf-8")
            output = args.output_json.as_posix()
        else:
            output = text
        sys.stdout.write(f"{output}\n")
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:  # pragma: no cover
        sys.stderr.write(f"Error: {exc}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
