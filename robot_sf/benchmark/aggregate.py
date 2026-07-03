"""Aggregation utilities for Social Navigation Benchmark JSONL outputs.

Functions:
- read_jsonl(paths): load episode records from one or more JSONL files
- flatten_metrics(record): flatten nested metrics dict for CSV writing
- write_episode_csv(records, out_csv, ...): write per-episode metrics CSV
- compute_aggregates(records, group_by=..., ...): grouped mean/median/p95 (+SNQI)

Notes:
- group_by supports dotted paths (e.g., "scenario_params.algo"). If the path
  is missing for a record, it falls back to the record's "scenario_id".
- If SNQI is not present and weights/baseline are provided, it's recomputed.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from loguru import logger

from robot_sf.benchmark.errors import AggregationMetadataError, EpisodeRecordInputError
from robot_sf.benchmark.grouping import EFFECTIVE_REPORT_GROUP_KEY, resolve_report_group_key
from robot_sf.benchmark.metrics import snqi as snqi_fn
from robot_sf.benchmark.thresholds import validate_threshold_parameter_consistency

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


def _format_jsonl_input_error(
    *,
    missing_paths: list[Path],
    malformed_lines: list[tuple[Path, int, str]],
) -> str:
    """Build an actionable JSONL input error message.

    Returns:
        Compact error message with counts and representative locations.
    """
    details = [
        "Episode JSONL input is not suitable for benchmark aggregation",
        f"missing_paths={len(missing_paths)}",
        f"malformed_lines={len(malformed_lines)}",
    ]
    if missing_paths:
        preview = ", ".join(str(path) for path in missing_paths[:5])
        details.append(f"missing=[{preview}]")
    if malformed_lines:
        preview = "; ".join(
            f"{path}:{line_number}: {message}" for path, line_number, message in malformed_lines[:5]
        )
        details.append(f"malformed=[{preview}]")
    return "; ".join(details)


def read_jsonl(
    paths: Sequence[str | Path] | str | Path,
    *,
    strict: bool = True,
) -> list[dict[str, Any]]:
    """Read one or more JSONL files into a list of records.

    Args:
        paths: One or more JSONL files.
        strict: When true, fail closed on missing paths and malformed records. When false, skip
            missing/malformed records for explicitly exploratory/advisory use.

    Returns:
        List of parsed episode records.
    """
    if isinstance(paths, str | Path):
        path_list = [paths]
    else:
        path_list = list(paths)  # type: ignore[arg-type]
    records: list[dict[str, Any]] = []
    missing_paths: list[Path] = []
    malformed_lines: list[tuple[Path, int, str]] = []
    for p in path_list:
        p = Path(p)
        if not p.exists():
            if strict:
                missing_paths.append(p)
            continue
        with p.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as exc:
                    if strict:
                        malformed_lines.append((p, line_number, exc.msg))
                    continue
                records.append(rec)
    if missing_paths or malformed_lines:
        raise EpisodeRecordInputError(
            _format_jsonl_input_error(
                missing_paths=missing_paths,
                malformed_lines=malformed_lines,
            )
        )
    return records


def _get_nested(d: dict[str, Any], path: str, default: Any = None) -> Any:
    """Resolve a dotted-path value from a dict.

    Returns:
        Value at the path or default when missing.
    """
    cur: Any = d
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


_EFFECTIVE_GROUP_KEY = EFFECTIVE_REPORT_GROUP_KEY
_UNKNOWN_BENCHMARK_TRACK = "unspecified"
_CAVEAT_STATUSES = {
    "degraded",
    "diagnostic-stub",
    "diagnostic_stub",
    "failed",
    "fallback",
    "not-available",
    "not_available",
    "partial-failure",
    "partial_failure",
}


def _normalize_algo(value: Any) -> str | None:
    """Normalize algorithm identifiers to a non-empty string.

    Returns:
        Normalized string or None if empty/invalid.
    """
    if isinstance(value, str):
        trimmed = value.strip()
        if trimmed:
            return trimmed
    return None


def _normalize_observation_track_mode(value: str) -> str:
    """Normalize observation-track aggregation mode strings.

    Returns:
        Canonical mode string used by aggregation internals.
    """
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized not in {"strict", "diagnostic_cross_track"}:
        raise ValueError("observation_track_mode must be 'strict' or 'diagnostic-cross-track'.")
    return normalized


def _resolve_benchmark_track(record: dict[str, Any]) -> str:
    """Return the benchmark observation track declared by an episode record."""
    candidates = [
        record.get("benchmark_track"),
        _get_nested(record, "scenario_params.benchmark_track"),
        _get_nested(record, "algorithm_metadata.benchmark_track.benchmark_track"),
        _get_nested(record, "algorithm_metadata.benchmark_track.track"),
    ]
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return _UNKNOWN_BENCHMARK_TRACK


def normalize_observation_track_mode(value: str) -> str:
    """Return the canonical observation-track aggregation mode."""
    return _normalize_observation_track_mode(value)


def resolve_benchmark_track(record: dict[str, Any]) -> str:
    """Return the public benchmark observation track for an episode record."""
    return _resolve_benchmark_track(record)


def _record_has_benchmark_caveat(record: dict[str, Any]) -> bool:
    """Return true when a record is fallback, degraded, failed, or diagnostic-only."""
    candidates = [
        record.get("availability_status"),
        record.get("status"),
        _get_nested(record, "benchmark_availability.availability_status"),
        _get_nested(record, "algorithm_metadata.status"),
        _get_nested(record, "algorithm_metadata.readiness_status"),
        _get_nested(record, "algorithm_metadata.preflight_status"),
        _get_nested(record, "algorithm_metadata.execution_mode"),
    ]
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip().lower() in _CAVEAT_STATUSES:
            return True
    return False


def build_observation_track_meta(
    records: list[dict[str, Any]],
    *,
    observation_track_mode: str = "strict",
) -> dict[str, Any]:
    """Summarize observation-track metadata for aggregate/report outputs.

    Returns:
        Metadata block describing observed tracks, mode, and caveat counts.
    """
    mode = _normalize_observation_track_mode(observation_track_mode)
    track_counts: dict[str, int] = defaultdict(int)
    caveat_count = 0
    for record in records:
        track_counts[_resolve_benchmark_track(record)] += 1
        if _record_has_benchmark_caveat(record):
            caveat_count += 1

    tracks = dict(sorted(track_counts.items()))
    mixed_tracks = len(tracks) > 1
    selected_track = next(iter(tracks), None) if len(tracks) == 1 else None
    meta: dict[str, Any] = {
        "mode": mode,
        "tracks": tracks,
        "mixed_tracks": mixed_tracks,
        "selected_track": selected_track,
        "caveat_record_count": caveat_count,
        "caveat_policy": (
            "Fallback, degraded, failed, and diagnostic-stub rows are caveats, "
            "not benchmark success evidence."
        ),
    }
    if mixed_tracks:
        meta["cross_track_caveat"] = (
            "Rows use different observation contracts; compare tracks diagnostically "
            "rather than pooling them as one benchmark."
        )
    return meta


def ensure_observation_track_policy(
    records: list[dict[str, Any]],
    *,
    observation_track_mode: str = "strict",
) -> dict[str, Any]:
    """Validate observation-track pooling policy and return report metadata.

    Returns:
        Observation-track metadata block when the input satisfies the selected policy.
    """
    meta = build_observation_track_meta(
        records,
        observation_track_mode=observation_track_mode,
    )
    if meta["mixed_tracks"] and meta["mode"] == "strict":
        tracks = ", ".join(meta["tracks"])
        raise AggregationMetadataError(
            f"Mixed benchmark_track values cannot be pooled by default: {tracks}.",
            missing_fields=(
                "benchmark_track",
                "scenario_params.benchmark_track",
                "algorithm_metadata.benchmark_track",
            ),
            advice=(
                "Aggregate one benchmark_track at a time, or rerun with "
                "--observation-track-mode diagnostic-cross-track for an explicitly caveated "
                "cross-track comparison."
            ),
        )
    return meta


def observation_track_group_label(record: dict[str, Any], group_key: str, *, mode: str) -> str:
    """Return a group label that keeps cross-track diagnostics visibly separated."""
    normalized_mode = _normalize_observation_track_mode(mode)
    if normalized_mode == "diagnostic_cross_track":
        return f"{_resolve_benchmark_track(record)} :: {group_key}"
    return group_key


def _ensure_mapping(record: dict[str, Any], key: str, episode_id: str | None) -> None:
    """Ensure a record field is a mapping when nested access is required."""
    value = record.get(key)
    if value is not None and not isinstance(value, dict):
        raise AggregationMetadataError(
            f"{key} must be a mapping to access nested algorithm metadata.",
            episode_id=episode_id,
            missing_fields=(f"{key}", f"{key}.algo"),
            advice="Regenerate the benchmark data to include structured scenario parameters.",
        )


def _resolve_group_key(
    record: dict[str, Any],
    *,
    group_by: str,
    fallback_group_by: str,
) -> str:
    """Resolve the aggregation group key with metadata fallbacks.

    Returns:
        Group key string.
    """
    episode_id = record.get("episode_id")
    episode_ref = str(episode_id) if episode_id is not None else None

    if group_by.startswith("scenario_params"):
        _ensure_mapping(record, "scenario_params", episode_ref)

    nested_algo = _normalize_algo(_get_nested(record, group_by))
    if nested_algo is not None:
        return nested_algo

    top_level_algo = _normalize_algo(record.get("algo"))
    if top_level_algo is not None:
        return top_level_algo

    # Check algorithm_metadata.algorithm as additional fallback
    metadata_algo = _normalize_algo(_get_nested(record, "algorithm_metadata.algorithm"))
    if metadata_algo is not None:
        return metadata_algo

    fallback_value = _get_nested(record, fallback_group_by)
    if fallback_value is not None:
        if group_by == "scenario_params.algo":
            raise AggregationMetadataError(
                "Episode lacks algorithm metadata required for aggregation.",
                episode_id=episode_ref,
                missing_fields=("scenario_params.algo", "algo", "algorithm_metadata.algorithm"),
                advice="Ensure the orchestrator mirrors algorithm identifiers before aggregation.",
            )
        return str(fallback_value)

    raise AggregationMetadataError(
        "Unable to determine aggregation group key for episode.",
        episode_id=episode_ref,
        missing_fields=(group_by, "algo"),
        advice="Verify that episode records include algorithm metadata.",
    )


def _flatten_pedestrian_impact_block(base: dict[str, Any], ped_impact: Any) -> None:
    """Flatten schema-backed pedestrian-impact reductions into ``base``."""
    if not isinstance(ped_impact, dict):
        return
    reductions = ped_impact.get("canonical_reductions") or {}
    if isinstance(reductions, dict):
        for source_key, target_key in (
            ("accel_delta_mean", "ped_impact_accel_delta_mean"),
            ("accel_delta_median", "ped_impact_accel_delta_median"),
            ("accel_delta_valid_pedestrians", "ped_impact_accel_delta_valid"),
            ("turn_rate_delta_mean", "ped_impact_turn_rate_delta_mean"),
            ("turn_rate_delta_median", "ped_impact_turn_rate_delta_median"),
            ("turn_rate_delta_valid_pedestrians", "ped_impact_turn_rate_delta_valid"),
        ):
            base[target_key] = reductions.get(source_key)
    sample_counts = ped_impact.get("sample_counts") or {}
    if isinstance(sample_counts, dict):
        for source_key, target_key in (
            ("pedestrians", "ped_impact_ped_count"),
            ("near_samples", "ped_impact_near_samples"),
            ("far_samples", "ped_impact_far_samples"),
            ("near_sample_frac", "ped_impact_near_sample_frac"),
        ):
            base[target_key] = sample_counts.get(source_key)


def _flatten_social_acceptability_block(base: dict[str, Any], social_acceptability: Any) -> None:
    """Flatten schema-backed social-acceptability pilot reductions into ``base``."""
    if not isinstance(social_acceptability, dict):
        return
    base["social_proxemic_available"] = social_acceptability.get("available")
    parameters = social_acceptability.get("parameters") or {}
    if isinstance(parameters, dict):
        base["social_proxemic_radius_m"] = parameters.get("proxemic_radius_m")
    sample_counts = social_acceptability.get("sample_counts") or {}
    if isinstance(sample_counts, dict):
        base["social_proxemic_ped_count"] = sample_counts.get("pedestrians")
        base["social_proxemic_intrusion_steps"] = sample_counts.get("timesteps")
    proxemic = social_acceptability.get("proxemic") or {}
    if isinstance(proxemic, dict):
        for source_key, target_key in (
            ("intrusion_frac", "social_proxemic_intrusion_frac"),
            ("intrusion_area_m_s", "social_proxemic_intrusion_area_m_s"),
            ("min_clearance_m", "social_proxemic_min_clearance_m"),
        ):
            base[target_key] = proxemic.get(source_key)


def _flatten_human_interaction_proxy_block(
    base: dict[str, Any],
    human_interaction_proxy: Any,
) -> None:
    """Flatten schema-backed human-interaction proxy reductions into ``base``."""
    if not isinstance(human_interaction_proxy, dict):
        return
    base["human_proxy_available"] = human_interaction_proxy.get("available")
    parameters = human_interaction_proxy.get("parameters") or {}
    if isinstance(parameters, dict):
        base["human_proxy_proxemic_radius_m"] = parameters.get("proxemic_radius_m")
        base["human_proxy_yield_speed_mps"] = parameters.get("yield_speed_mps")
    sample_counts = human_interaction_proxy.get("sample_counts") or {}
    if isinstance(sample_counts, dict):
        base["human_proxy_ped_count"] = sample_counts.get("pedestrians")
        base["human_proxy_timestep_count"] = sample_counts.get("timesteps")
    reductions = human_interaction_proxy.get("canonical_reductions") or {}
    if isinstance(reductions, dict):
        for source_key in (
            "human_discomfort_exposure_m_s",
            "intrusion_duration_s",
            "time_to_yield_s",
            "robot_yield_distance_m",
            "pedestrian_path_deviation_proxy_m",
            "group_split_intrusion_available",
        ):
            base[source_key] = reductions.get(source_key)


def _flatten_social_mini_game_block(base: dict[str, Any], social_mini_game: Any) -> None:
    """Flatten available Social Mini-Game row values with a stable prefix."""
    if not isinstance(social_mini_game, dict) or not social_mini_game:
        return
    base["social_mini_game_status"] = social_mini_game.get("status")
    base["social_mini_game_mechanism_family"] = social_mini_game.get("mechanism_family")
    rows = social_mini_game.get("rows") or []
    if not isinstance(rows, list):
        return
    for row in rows:
        if not isinstance(row, dict):
            continue
        metric = row.get("metric")
        if not isinstance(metric, str) or not metric:
            continue
        prefix = f"social_mini_game_{metric}"
        base[f"{prefix}_status"] = row.get("status")
        base[f"{prefix}_support_count"] = row.get("support_count")
        if row.get("status") == "available" and "value" in row:
            base[prefix] = row.get("value")


def _flatten_clear_tracking_block(base: dict[str, Any], clear_tracking: Any) -> None:
    """Flatten schema-backed CLEAR tracking diagnostics into aggregate/report columns."""
    if not isinstance(clear_tracking, dict):
        return
    if "enabled" in clear_tracking:
        base["clear_tracking_enabled"] = clear_tracking.get("enabled")
    if "mota" in clear_tracking:
        base["clear_mota"] = clear_tracking.get("mota")
    if "motp_m" in clear_tracking:
        base["clear_motp_m"] = clear_tracking.get("motp_m")
    counts = clear_tracking.get("counts") or {}
    if not isinstance(counts, dict):
        return
    count_fields = {
        "ground_truth": "clear_ground_truth_count",
        "detections": "clear_detection_count",
        "missed_detections": "clear_missed_detection_count",
        "false_positives": "clear_false_positive_count",
        "id_switches": "clear_id_switch_count",
        "motp_matches": "clear_motp_match_count",
    }
    for source_key, target_key in count_fields.items():
        if source_key in counts:
            base[target_key] = counts.get(source_key)


def flatten_metrics(rec: dict[str, Any]) -> dict[str, Any]:
    """Flatten metrics dict into a flat per-episode row.

    Returns:
        Flattened metrics row for CSV or aggregation.
    """
    base = {
        "episode_id": rec.get("episode_id"),
        "scenario_id": rec.get("scenario_id"),
        "seed": rec.get("seed"),
    }
    metrics = dict(rec.get("metrics") or {})
    fq = metrics.pop("force_quantiles", {}) or {}
    metrics.pop("distributional_disruption", None)
    ped_impact = metrics.pop("pedestrian_impact", {}) or {}
    social_acceptability = metrics.pop("social_acceptability", {}) or {}
    human_interaction_proxy = metrics.pop("human_interaction_proxy", {}) or {}
    social_mini_game = metrics.pop("social_mini_game", {}) or {}
    clear_tracking = metrics.pop("clear_tracking_uncertainty", {}) or {}
    inter_robot = metrics.pop("inter_robot", {}) or {}
    # Flatten known force quantiles
    for qk in ("q50", "q90", "q95"):
        key = f"force_{qk}"
        base[key] = fq.get(qk)
    # Flatten the schema-backed pedestrian-impact block for records that do not also carry
    # legacy flat ped_impact_* keys.
    _flatten_pedestrian_impact_block(base, ped_impact)
    _flatten_social_acceptability_block(base, social_acceptability)
    _flatten_human_interaction_proxy_block(base, human_interaction_proxy)
    _flatten_social_mini_game_block(base, social_mini_game)
    _flatten_clear_tracking_block(base, clear_tracking)
    if isinstance(inter_robot, dict):
        for key, value in inter_robot.items():
            base[str(key)] = value
    # Remainder metrics (flat numbers)
    base.update(metrics)
    return base


def _ensure_snqi(
    rec: dict[str, Any],
    weights: dict[str, float] | None,
    baseline: dict[str, dict[str, float]] | None,
    *,
    recompute: bool = False,
    strict: bool = False,
) -> None:
    """Compute and attach SNQI when missing, or recompute when ``recompute`` is set."""
    if rec.get("metrics") is None:
        return
    if "snqi" in rec["metrics"] and not recompute:
        return
    if weights is None:
        return
    try:
        rec["metrics"]["snqi"] = float(snqi_fn(rec["metrics"], weights, baseline_stats=baseline))
    except Exception:
        logger.bind(
            event="aggregation_snqi_compute_failed",
            episode_id=rec.get("episode_id"),
            scenario_id=rec.get("scenario_id"),
            seed=rec.get("seed"),
            algo=rec.get("algo")
            if rec.get("algo") is not None
            else _get_nested(rec, "scenario_params.algo"),
            recompute_snqi=recompute,
            strict=strict,
        ).exception("Failed to compute SNQI for episode record.")
        if strict:
            raise


def write_episode_csv(
    records: list[dict[str, Any]],
    out_csv: str | Path,
    *,
    snqi_weights: dict[str, float] | None = None,
    snqi_baseline: dict[str, dict[str, float]] | None = None,
) -> str:
    # Optionally compute SNQI per record if missing
    """Write per-episode metrics to CSV.

    Returns:
        Path string to the written CSV file.
    """
    for rec in records:
        _ensure_snqi(rec, snqi_weights, snqi_baseline)

    # Determine all metric keys across records for CSV header
    flat_rows = [flatten_metrics(r) for r in records]
    keys = set()
    for row in flat_rows:
        keys.update(row.keys())
    # Ensure stable ordering: id fields first
    id_keys = ["episode_id", "scenario_id", "seed"]
    metric_keys = sorted(k for k in keys if k not in id_keys)
    header = id_keys + metric_keys

    out_csv = str(out_csv)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in flat_rows:
            writer.writerow({k: row.get(k) for k in header})
    return out_csv


def _numeric_items(d: dict[str, Any]) -> dict[str, float]:
    """Extract numeric values from a flattened metrics row.

    Returns:
        Mapping of numeric metric values.
    """
    out: dict[str, float] = {}
    for k, v in d.items():
        if k in ("episode_id", "scenario_id", "seed"):
            continue
        if v is None:
            continue
        # NOTE: benchmark metrics store ``success`` as a Python bool
        # (see ``post_process_metrics``), so booleans MUST be coerced to
        # 0.0/1.0 here rather than dropped — excluding them nulls
        # ``success.mean`` across the entire aggregate surface. Only the
        # non-metric report scripts filter stray bools out.
        if isinstance(v, bool):
            out[k] = float(v)
            continue
        if isinstance(v, int | float) and math.isfinite(float(v)):
            out[k] = float(v)
    return out


def compute_aggregates(  # noqa: PLR0913
    records: list[dict[str, Any]],
    *,
    group_by: str = "scenario_params.algo",
    fallback_group_by: str = "scenario_id",
    snqi_weights: dict[str, float] | None = None,
    snqi_baseline: dict[str, dict[str, float]] | None = None,
    recompute_snqi: bool = False,
    expected_algorithms: set[str] | None = None,
    observation_track_mode: str = "strict",
    logger_ctx=None,
) -> dict[str, dict[str, dict[str, float]]]:
    """Aggregate metrics by group and compute summary statistics.

    Returns:
        Nested dict of group -> metric -> summary statistics.
    """
    for rec in records:
        _ensure_snqi(
            rec,
            snqi_weights,
            snqi_baseline,
            recompute=recompute_snqi,
            strict=_normalize_observation_track_mode(observation_track_mode) == "strict",
        )
    observation_track_meta = ensure_observation_track_policy(
        records,
        observation_track_mode=observation_track_mode,
    )
    threshold_meta = validate_threshold_parameter_consistency(records)

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    present_algorithms: set[str] = set()
    for rec in records:
        key = _resolve_group_key(rec, group_by=group_by, fallback_group_by=fallback_group_by)
        present_algorithms.add(str(key))
        key_str = observation_track_group_label(
            rec,
            str(key),
            mode=observation_track_meta["mode"],
        )
        groups[key_str].append(flatten_metrics(rec))

    summary: dict[str, dict[str, dict[str, float]]] = {}
    for g, rows in groups.items():
        # collect numeric columns
        cols: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            num = _numeric_items(row)
            for k, v in num.items():
                cols[k].append(v)
        agg: dict[str, dict[str, float]] = {}
        for k, vals in cols.items():
            arr = np.asarray(vals, dtype=float)
            agg[k] = {
                "mean": float(np.nanmean(arr)),
                "median": float(np.nanmedian(arr)),
                "p95": float(np.nanpercentile(arr, 95)),
            }
        summary[g] = agg

    meta: dict[str, Any] = {
        "group_by": group_by,
        "effective_group_key": _EFFECTIVE_GROUP_KEY,
        "missing_algorithms": [],
        "warnings": [],
        "metric_parameters": {
            "threshold_profile": threshold_meta["threshold_profile"],
            "threshold_signature": threshold_meta["threshold_signature"],
            "missing_profile_records": threshold_meta["missing_profile_records"],
            "explicit_profile_records": threshold_meta["explicit_profile_records"],
        },
        "observation_tracks": observation_track_meta,
    }

    if expected_algorithms:
        expected_set = {str(v) for v in expected_algorithms}
        missing = sorted(expected_set - present_algorithms)
        meta["missing_algorithms"] = missing
        if missing:
            warning_text = f"Missing algorithms detected: {', '.join(missing)}"
            meta["warnings"] = [warning_text]
            (logger_ctx or logger).bind(
                event="aggregation_missing_algorithms",
                expected=sorted(expected_set),
                present=sorted(present_algorithms),
                missing=missing,
            ).warning(warning_text)

    summary["_meta"] = meta
    return summary


# --- Optional bootstrap confidence intervals ---


def _bootstrap_ci(
    data: np.ndarray,
    stat_fn: Callable[[np.ndarray], float],
    *,
    samples: int = 1000,
    confidence: float = 0.95,
    seed: int | None = None,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for a statistic.

    Args:
        data: One-dimensional array of samples; NaNs are ignored.
        stat_fn: Callable that computes a scalar statistic over a 1D array.
        samples: Number of bootstrap resamples to draw.
        confidence: Confidence level for the returned interval (e.g., 0.95).
        seed: Optional RNG seed for reproducibility.

    Returns:
        tuple[float, float]: ``(low, high)`` bounds, ``(nan, nan)`` when insufficient data.
    """
    if samples <= 0:
        return (float("nan"), float("nan"))
    x = np.asarray(data, dtype=float)
    x = x[~np.isnan(x)]
    n = x.size
    if n == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    stats = np.empty(samples, dtype=float)
    for i in range(samples):
        idx = rng.integers(0, n, size=n)
        xb = x[idx]
        try:
            stats[i] = float(stat_fn(xb))
        except Exception:
            stats[i] = float("nan")
    stats = stats[~np.isnan(stats)]
    if stats.size == 0:
        return (float("nan"), float("nan"))
    alpha = (1.0 - confidence) / 2.0
    lo = float(np.percentile(stats, 100.0 * alpha))
    hi = float(np.percentile(stats, 100.0 * (1.0 - alpha)))
    return (lo, hi)


def _group_flattened(
    records: list[dict[str, Any]],
    *,
    group_by: str,
    fallback_group_by: str,
    observation_track_mode: str = "strict",
) -> dict[str, list[dict[str, Any]]]:
    """Group flattened episode rows by aggregation key.

    Returns:
        Mapping of group key to flattened rows.
    """
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    mode = _normalize_observation_track_mode(observation_track_mode)
    for rec in records:
        key = _resolve_group_key(rec, group_by=group_by, fallback_group_by=fallback_group_by)
        key_str = observation_track_group_label(rec, str(key), mode=mode)
        groups[key_str].append(flatten_metrics(rec))
    return groups


def _attach_ci_for_group(
    dst_group: dict[str, dict[str, Any]],
    values_by_metric: dict[str, list[float]],
    *,
    bootstrap_samples: int,
    bootstrap_confidence: float,
    bootstrap_seed: int | None,
) -> None:
    """Attach bootstrap confidence intervals for each metric in a group."""
    for metric_name, values in values_by_metric.items():
        arr = np.asarray(values, dtype=float)

        def mean_fn(a: np.ndarray) -> float:
            """Return mean for bootstrap sample."""
            return float(np.mean(a))

        def median_fn(a: np.ndarray) -> float:
            """Return median for bootstrap sample."""
            return float(np.median(a))

        def p95_fn(a: np.ndarray) -> float:
            """Return p95 for bootstrap sample."""
            return float(np.percentile(a, 95))

        lo_hi_mean = _bootstrap_ci(
            arr,
            mean_fn,
            samples=bootstrap_samples,
            confidence=bootstrap_confidence,
            seed=bootstrap_seed,
        )
        lo_hi_median = _bootstrap_ci(
            arr,
            median_fn,
            samples=bootstrap_samples,
            confidence=bootstrap_confidence,
            seed=bootstrap_seed,
        )
        lo_hi_p95 = _bootstrap_ci(
            arr,
            p95_fn,
            samples=bootstrap_samples,
            confidence=bootstrap_confidence,
            seed=bootstrap_seed,
        )
        dst_group.setdefault(metric_name, {})
        dst_group[metric_name]["mean_ci"] = [float(lo_hi_mean[0]), float(lo_hi_mean[1])]
        dst_group[metric_name]["median_ci"] = [float(lo_hi_median[0]), float(lo_hi_median[1])]
        dst_group[metric_name]["p95_ci"] = [float(lo_hi_p95[0]), float(lo_hi_p95[1])]


def _pair_identity(row: dict[str, Any]) -> tuple[str, str] | None:
    """Return the shared episode identity used for paired planner contrasts."""
    scenario_id = row.get("scenario_id")
    seed = row.get("seed", row.get("seed_index"))
    if scenario_id is None or seed is None:
        return None
    return (str(scenario_id), str(seed))


def _paired_metric_differences(
    left_rows: list[dict[str, Any]],
    right_rows: list[dict[str, Any]],
) -> dict[str, np.ndarray]:
    """Return paired ``right - left`` metric differences keyed by metric name."""
    left_by_id = {
        identity: _numeric_items(row)
        for row in left_rows
        if (identity := _pair_identity(row)) is not None
    }
    right_by_id = {
        identity: _numeric_items(row)
        for row in right_rows
        if (identity := _pair_identity(row)) is not None
    }
    paired_ids = sorted(set(left_by_id) & set(right_by_id))
    diffs: dict[str, list[float]] = defaultdict(list)
    for identity in paired_ids:
        left_metrics = left_by_id[identity]
        right_metrics = right_by_id[identity]
        for metric_name in sorted(set(left_metrics) & set(right_metrics)):
            diffs[metric_name].append(right_metrics[metric_name] - left_metrics[metric_name])
    return {name: np.asarray(values, dtype=float) for name, values in diffs.items()}


def _bootstrap_delta_stats(
    differences: np.ndarray,
    *,
    bootstrap_samples: int,
    bootstrap_confidence: float,
    bootstrap_seed: int | None,
) -> dict[str, Any]:
    """Summarize paired differences with percentile bootstrap and effect size metadata.

    Returns:
        Pairwise contrast statistics, or an empty mapping when no pairs are available.
    """
    clean = np.asarray(differences, dtype=float)
    clean = clean[~np.isnan(clean)]
    n_pairs = int(clean.size)
    if n_pairs == 0:
        return {}

    delta_mean = float(np.mean(clean))
    delta_median = float(np.median(clean))
    sd = float(np.std(clean, ddof=1)) if n_pairs > 1 else float("nan")
    effect_size = delta_mean / sd if sd > 0.0 and math.isfinite(sd) else None
    rng = np.random.default_rng(bootstrap_seed)
    boot_means = np.empty(int(bootstrap_samples), dtype=float)
    for i in range(int(bootstrap_samples)):
        idx = rng.integers(0, n_pairs, size=n_pairs)
        boot_means[i] = float(np.mean(clean[idx]))

    alpha = (1.0 - bootstrap_confidence) / 2.0
    ci = [
        float(np.percentile(boot_means, 100.0 * alpha)),
        float(np.percentile(boot_means, 100.0 * (1.0 - alpha))),
    ]
    p_lower = float(np.mean(boot_means <= 0.0))
    p_upper = float(np.mean(boot_means >= 0.0))
    p_value = float(min(1.0, 2.0 * min(p_lower, p_upper)))
    return {
        "n_pairs": n_pairs,
        "delta_mean": delta_mean,
        "delta_median": delta_median,
        "delta_ci": ci,
        "p_value": p_value,
        "effect_size": {
            "type": "paired_cohens_dz",
            "value": None if effect_size is None else float(effect_size),
        },
    }


def _holm_adjust(p_values: list[tuple[str, str, float]]) -> dict[tuple[str, str], float]:
    """Return Holm-adjusted p-values keyed by ``(comparison_key, metric_name)``."""
    adjusted: dict[tuple[str, str], float] = {}
    m = len(p_values)
    running_max = 0.0
    for rank, (comparison_key, metric_name, p_value) in enumerate(
        sorted(p_values, key=lambda item: item[2]),
        start=1,
    ):
        raw_adjusted = min(1.0, (m - rank + 1) * p_value)
        running_max = max(running_max, raw_adjusted)
        adjusted[(comparison_key, metric_name)] = running_max
    return adjusted


def _compute_pairwise_contrasts(
    groups: dict[str, list[dict[str, Any]]],
    *,
    bootstrap_samples: int,
    bootstrap_confidence: float,
    bootstrap_seed: int | None,
) -> dict[str, Any]:
    """Compute paired planner contrasts for every group pair and metric.

    Returns:
        Additive ``pairwise_contrasts`` payload, or an empty mapping when unavailable.
    """
    if bootstrap_samples <= 0 or len(groups) < 2:
        return {}

    contrasts: dict[str, Any] = {
        "_meta": {
            "method": "paired_bootstrap_mean_delta",
            "delta": "right_minus_left",
            "pairing_keys": ["scenario_id", "seed_or_seed_index"],
            "bootstrap_samples": int(bootstrap_samples),
            "bootstrap_confidence": float(bootstrap_confidence),
            "bootstrap_seed": bootstrap_seed,
            "p_value_method": "two_sided_bootstrap_sign",
            "p_value_correction": "holm",
            "correction_family": ["family", "metric"],
            "family": "all",
        }
    }
    p_values_by_metric: dict[str, list[tuple[str, str, float]]] = defaultdict(list)

    for left_name, right_name in combinations(sorted(groups), 2):
        comparison_key = f"{left_name}__vs__{right_name}"
        metric_diffs = _paired_metric_differences(groups[left_name], groups[right_name])
        metric_stats: dict[str, dict[str, Any]] = {}
        for metric_name, differences in metric_diffs.items():
            stats = _bootstrap_delta_stats(
                differences,
                bootstrap_samples=bootstrap_samples,
                bootstrap_confidence=bootstrap_confidence,
                bootstrap_seed=bootstrap_seed,
            )
            if not stats:
                continue
            metric_stats[metric_name] = stats
            p_values_by_metric[metric_name].append((comparison_key, metric_name, stats["p_value"]))
        if metric_stats:
            contrasts[comparison_key] = {
                "left": left_name,
                "right": right_name,
                "metrics": metric_stats,
            }

    for metric_name, p_values in p_values_by_metric.items():
        adjusted = _holm_adjust(p_values)
        for comparison_key, _, _ in p_values:
            contrasts[comparison_key]["metrics"][metric_name]["p_value_holm"] = adjusted[
                (comparison_key, metric_name)
            ]
    return contrasts if len(contrasts) > 1 else {}


def compute_aggregates_with_ci(  # noqa: PLR0913
    records: list[dict[str, Any]],
    *,
    group_by: str = "scenario_params.algo",
    fallback_group_by: str = "scenario_id",
    snqi_weights: dict[str, float] | None = None,
    snqi_baseline: dict[str, dict[str, float]] | None = None,
    recompute_snqi: bool = False,
    return_ci: bool = True,
    bootstrap_samples: int = 1000,
    bootstrap_confidence: float = 0.95,
    bootstrap_seed: int | None = None,
    expected_algorithms: set[str] | None = None,
    observation_track_mode: str = "strict",
    logger_ctx=None,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Compute grouped aggregates and optional bootstrap CIs.

    This preserves the original aggregate keys (mean, median, p95) and, when
    return_ci is True and bootstrap_samples>0, adds parallel keys 'mean_ci',
    'median_ci', 'p95_ci' with [low, high] bounds using percentile bootstrap.

    Returns:
        Nested dictionary mapping group names to metric names to aggregate statistics.
    """
    # Start from base aggregates (no CI) for consistency
    base = compute_aggregates(
        records,
        group_by=group_by,
        fallback_group_by=fallback_group_by,
        snqi_weights=snqi_weights,
        snqi_baseline=snqi_baseline,
        recompute_snqi=recompute_snqi,
        expected_algorithms=expected_algorithms,
        observation_track_mode=observation_track_mode,
        logger_ctx=logger_ctx,
    )
    if not return_ci or bootstrap_samples <= 0:
        # Upcast type to Any container for compatibility, but keep content unchanged
        return cast("dict[str, dict[str, dict[str, Any]]]", base)

    # Rebuild groups with flattened numeric values to avoid rework
    for rec in records:
        _ensure_snqi(
            rec,
            snqi_weights,
            snqi_baseline,
            recompute=recompute_snqi,
            strict=_normalize_observation_track_mode(observation_track_mode) == "strict",
        )
    groups = _group_flattened(
        records,
        group_by=group_by,
        fallback_group_by=fallback_group_by,
        observation_track_mode=observation_track_mode,
    )
    pairwise_contrasts = _compute_pairwise_contrasts(
        groups,
        bootstrap_samples=bootstrap_samples,
        bootstrap_confidence=bootstrap_confidence,
        bootstrap_seed=bootstrap_seed,
    )

    out: dict[str, dict[str, dict[str, Any]]] = {
        k: dict(v) for k, v in base.items() if k != "_meta"
    }
    if "_meta" in base:
        out["_meta"] = dict(base["_meta"])  # type: ignore[assignment]
    for g, rows in groups.items():
        # collect numeric columns per group
        cols: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            num = _numeric_items(row)
            for k, v in num.items():
                cols[k].append(v)
        _attach_ci_for_group(
            out.setdefault(g, {}),
            cols,
            bootstrap_samples=bootstrap_samples,
            bootstrap_confidence=bootstrap_confidence,
            bootstrap_seed=bootstrap_seed,
        )
    if pairwise_contrasts:
        out["pairwise_contrasts"] = pairwise_contrasts  # type: ignore[assignment]
    return out


__all__ = [
    "build_observation_track_meta",
    "compute_aggregates",
    "compute_aggregates_with_ci",
    "ensure_observation_track_policy",
    "flatten_metrics",
    "normalize_observation_track_mode",
    "observation_track_group_label",
    "read_jsonl",
    "resolve_benchmark_track",
    "resolve_report_group_key",
    "write_episode_csv",
]
