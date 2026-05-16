"""Config-only scenario coverage entropy and novelty analysis."""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "scenario_coverage_entropy.v1"

_REDUNDANT_THRESHOLD = 0.15
_NOVEL_THRESHOLD = 0.35


def _scenario_id(scenario: Mapping[str, Any], index: int) -> str:
    """Return the stable scenario identifier used by coverage reports."""
    for field in ("name", "id", "scenario_id"):
        value = scenario.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return f"scenario_{index:03d}"


def _metadata(scenario: Mapping[str, Any]) -> dict[str, Any]:
    """Return a shallow metadata mapping for a scenario."""
    value = scenario.get("metadata")
    return dict(value) if isinstance(value, Mapping) else {}


def _finite_float(value: Any) -> float | None:
    """Parse a finite float while preserving missing values.

    Returns:
        Finite float, or ``None`` for missing, invalid, NaN, or infinite values.
    """
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _density_bin(value: Any) -> str:
    """Bucket pedestrian density into coarse, reviewable labels.

    Returns:
        One of ``low``, ``medium``, ``high``, or ``unknown``.
    """
    parsed = _finite_float(value)
    if parsed is None:
        return "unknown"
    if parsed <= 0.025:
        return "low"
    if parsed <= 0.075:
        return "medium"
    return "high"


def _step_bin(value: Any) -> str:
    """Bucket max episode steps into coarse runtime-shape labels.

    Returns:
        One of ``short``, ``medium``, ``long``, or ``unknown``.
    """
    parsed = _finite_float(value)
    if parsed is None:
        return "unknown"
    if parsed <= 300:
        return "short"
    if parsed <= 700:
        return "medium"
    return "long"


def _count_bin(count: int) -> str:
    """Bucket small actor counts for categorical novelty features.

    Returns:
        One of ``none``, ``few``, ``some``, or ``many``.
    """
    if count <= 0:
        return "none"
    if count <= 2:
        return "few"
    if count <= 6:
        return "some"
    return "many"


def _has_wait_behavior(single_pedestrians: Sequence[Any]) -> bool:
    """Return whether any authored pedestrian trajectory contains wait rules."""
    for pedestrian in single_pedestrians:
        if not isinstance(pedestrian, Mapping):
            continue
        wait_at = pedestrian.get("wait_at")
        if isinstance(wait_at, Sequence) and not isinstance(wait_at, (str, bytes)) and wait_at:
            return True
    return False


def _feature_tokens(scenario: Mapping[str, Any]) -> dict[str, str]:
    """Extract the v1 config-only feature vocabulary for one scenario.

    Returns:
        Mapping from feature name to normalized categorical token.
    """
    metadata = _metadata(scenario)
    simulation_config = scenario.get("simulation_config")
    sim = dict(simulation_config) if isinstance(simulation_config, Mapping) else {}
    single_raw = scenario.get("single_pedestrians")
    single_pedestrians = (
        list(single_raw)
        if isinstance(single_raw, Sequence) and not isinstance(single_raw, (str, bytes))
        else []
    )
    seeds_raw = scenario.get("seeds")
    seeds = (
        list(seeds_raw)
        if isinstance(seeds_raw, Sequence) and not isinstance(seeds_raw, str)
        else []
    )
    map_file = str(scenario.get("map_file") or scenario.get("map") or "unknown")

    optional = bool(metadata.get("optional") or scenario.get("optional"))
    wait_behavior = _has_wait_behavior(single_pedestrians)
    static_markers = any(
        isinstance(ped, Mapping) and ped.get("goal") is None and ped.get("trajectory") is None
        for ped in single_pedestrians
    )

    tokens = {
        "archetype": str(metadata.get("archetype") or metadata.get("family") or "unknown"),
        "density_label": str(metadata.get("density") or "unknown"),
        "ped_density_bin": _density_bin(sim.get("ped_density")),
        "flow": str(metadata.get("flow") or "unknown"),
        "evaluation_scope": str(metadata.get("evaluation_scope") or "unknown"),
        "map": Path(map_file).name,
        "single_pedestrians": _count_bin(len(single_pedestrians)),
        "wait_behavior": str(wait_behavior).lower(),
        "static_markers": str(static_markers).lower(),
        "optional": str(optional).lower(),
        "seed_count": _count_bin(len(seeds)),
        "max_episode_steps": _step_bin(sim.get("max_episode_steps")),
    }
    platform_variant = metadata.get("platform_variant")
    if isinstance(platform_variant, str) and platform_variant.strip():
        tokens["platform_variant"] = platform_variant.strip()
    return tokens


def _jaccard_distance(left: Mapping[str, str], right: Mapping[str, str]) -> float:
    """Compute Jaccard distance over key/value feature tokens.

    Returns:
        Distance in the inclusive range ``[0.0, 1.0]``.
    """
    left_tokens = {f"{key}={value}" for key, value in left.items()}
    right_tokens = {f"{key}={value}" for key, value in right.items()}
    union = left_tokens | right_tokens
    if not union:
        return 0.0
    return float(1.0 - (len(left_tokens & right_tokens) / len(union)))


def _coverage_entropy(feature_rows: Sequence[Mapping[str, str]]) -> float:
    """Return normalized Shannon entropy across all feature-token values."""
    counts: Counter[str] = Counter()
    for row in feature_rows:
        counts.update(f"{key}={value}" for key, value in row.items())
    total = sum(counts.values())
    if total <= 0 or len(counts) <= 1:
        return 0.0
    entropy = -sum((count / total) * math.log2(count / total) for count in counts.values())
    return float(entropy / math.log2(len(counts)))


def _distinct_features(
    row: Mapping[str, str],
    other_rows: Sequence[Mapping[str, str]],
) -> list[str]:
    """Return feature names whose values are unique within this report."""
    distinct: list[str] = []
    for key, value in row.items():
        if all(other.get(key) != value for other in other_rows):
            distinct.append(key)
    return sorted(distinct)


def _recommendation(novelty_score: float, distinct_features: Sequence[str]) -> str:
    """Map diagnostic novelty values to a conservative recommendation bucket.

    Returns:
        Conservative curation bucket for the scenario row.
    """
    if novelty_score <= _REDUNDANT_THRESHOLD:
        return "merge_or_drop"
    if novelty_score >= _NOVEL_THRESHOLD or distinct_features:
        return "retain_or_investigate"
    return "review"


def _probe_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Return reviewable distinct-coverage probe metadata when present."""
    probe = metadata.get("distinct_coverage_probe")
    return dict(probe) if isinstance(probe, Mapping) else {}


def build_scenario_coverage_report(
    scenarios: Sequence[Mapping[str, Any]],
    *,
    source: str,
) -> dict[str, Any]:
    """Build a deterministic config-only scenario coverage report.

    The v1 report is diagnostic. It uses authored scenario metadata and static
    config fields only; it is not a benchmark-success, safety, or promotion
    metric.

    Returns:
        JSON-serializable scenario coverage report payload.
    """
    scenario_entries = [dict(scenario) for scenario in scenarios]
    if not scenario_entries:
        raise ValueError("scenario coverage report requires at least one scenario")

    ids = [_scenario_id(scenario, index) for index, scenario in enumerate(scenario_entries)]
    feature_rows = [_feature_tokens(scenario) for scenario in scenario_entries]
    duplicate_ids = [scenario_id for scenario_id, count in Counter(ids).items() if count > 1]
    if duplicate_ids:
        raise ValueError(f"duplicate scenario ids in coverage input: {', '.join(duplicate_ids)}")

    rows: list[dict[str, Any]] = []
    for index, (scenario_id, scenario, features) in enumerate(
        zip(ids, scenario_entries, feature_rows, strict=True)
    ):
        neighbor_id = None
        nearest_distance = 0.0
        if len(feature_rows) > 1:
            distances = [
                (_jaccard_distance(features, other_features), ids[other_index])
                for other_index, other_features in enumerate(feature_rows)
                if other_index != index
            ]
            nearest_distance, neighbor_id = min(distances, key=lambda item: (item[0], item[1]))
        others = [row for row_index, row in enumerate(feature_rows) if row_index != index]
        distinct = _distinct_features(features, others)
        metadata = _metadata(scenario)
        row = {
            "scenario_id": scenario_id,
            "feature_tokens": dict(sorted(features.items())),
            "novelty_score": round(float(nearest_distance), 6),
            "nearest_neighbor": neighbor_id,
            "distinct_features": distinct,
            "recommendation": _recommendation(nearest_distance, distinct),
            "metadata": {},
        }
        probe = _probe_metadata(metadata)
        if probe:
            row["metadata"]["distinct_coverage_probe"] = probe
        rows.append(row)

    rows.sort(key=lambda row: (-float(row["novelty_score"]), str(row["scenario_id"])))
    feature_keys = sorted({key for features in feature_rows for key in features})
    return {
        "schema_version": SCHEMA_VERSION,
        "source": source,
        "feature_contract": {
            "mode": "config_only",
            "feature_keys": feature_keys,
            "novelty_metric": "nearest_neighbor_jaccard_distance",
            "entropy_metric": "normalized_shannon_entropy_over_feature_tokens",
            "metrics_are_benchmark_claims": False,
            "interpretation": (
                "Diagnostic scenario-set coverage only; not a benchmark-success or safety metric."
            ),
        },
        "summary": {
            "scenario_count": len(rows),
            "feature_count": len(feature_keys),
            "coverage_entropy": round(_coverage_entropy(feature_rows), 6),
            "redundant_count": sum(1 for row in rows if row["recommendation"] == "merge_or_drop"),
            "novel_count": sum(
                1 for row in rows if row["recommendation"] == "retain_or_investigate"
            ),
        },
        "scenario_rows": rows,
    }


def scenario_coverage_report_markdown(report: Mapping[str, Any]) -> str:
    """Render a scenario coverage report as Markdown.

    Returns:
        Markdown report body.
    """
    summary = report.get("summary")
    summary_map = summary if isinstance(summary, Mapping) else {}
    contract = report.get("feature_contract")
    contract_map = contract if isinstance(contract, Mapping) else {}
    lines = [
        "# Scenario Coverage Report",
        "",
        "## Scenario Coverage Entropy",
        "",
        f"- Source: `{report.get('source', '')}`",
        f"- Schema: `{report.get('schema_version', SCHEMA_VERSION)}`",
        f"- Mode: `{contract_map.get('mode', 'config_only')}`",
        "- Interpretation: diagnostic scenario-set coverage only; not a benchmark-success or safety metric.",
        "",
        "## Summary",
        "",
        f"- Scenarios: {int(summary_map.get('scenario_count', 0))}",
        f"- Feature count: {int(summary_map.get('feature_count', 0))}",
        f"- Coverage entropy: {float(summary_map.get('coverage_entropy', 0.0)):.6f}",
        f"- Redundant candidates: {int(summary_map.get('redundant_count', 0))}",
        f"- Novel candidates: {int(summary_map.get('novel_count', 0))}",
        "",
        "## Scenario Rows",
        "",
        "| Scenario | Novelty | Nearest Neighbor | Recommendation | Distinct Features |",
        "| --- | ---: | --- | --- | --- |",
    ]
    rows = report.get("scenario_rows")
    for row_raw in rows if isinstance(rows, Sequence) else []:
        if not isinstance(row_raw, Mapping):
            continue
        distinct_raw = row_raw.get("distinct_features")
        distinct = (
            ", ".join(str(item) for item in distinct_raw)
            if isinstance(distinct_raw, Sequence) and not isinstance(distinct_raw, (str, bytes))
            else ""
        )
        lines.append(
            "| "
            f"{row_raw.get('scenario_id', '')} | "
            f"{float(row_raw.get('novelty_score', 0.0)):.6f} | "
            f"{row_raw.get('nearest_neighbor') or ''} | "
            f"{row_raw.get('recommendation', '')} | "
            f"{distinct} |"
        )
    return "\n".join(lines) + "\n"


def write_scenario_coverage_report(
    report: Mapping[str, Any],
    *,
    json_path: Path | None = None,
    markdown_path: Path | None = None,
) -> None:
    """Write report artifacts when output paths are provided."""
    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if markdown_path is not None:
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(scenario_coverage_report_markdown(report), encoding="utf-8")
