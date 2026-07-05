"""Compare authored and repository-trace-derived scenario prior ranges for Issue #2919.

The comparison is deliberately analysis-only. It uses the Issue #2917 prior-card registry as the
source router, extracts compact numeric priors from machine-readable repository files, and emits
gap proposals without making dataset-realism, planner-ranking, or benchmark-superiority claims.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from robot_sf.research.scenario_prior_staging_contract import (
    CONTRACT_STATUS_READY,
    ScenarioPriorStagingContractError,
    ScenarioPriorStagingContractReport,
    check_scenario_prior_staging_contract,
    load_scenario_prior_staging_contract,
)

DEFAULT_REGISTRY_PATH = Path("configs/research/scenario_prior_cards_issue_2917.yaml")
DEFAULT_OUTPUT_DIR = Path("docs/context/evidence/issue_2919_scenario_prior_gap_2026-06-21")
DEFAULT_DATASET_STAGING_CONTRACT_PATH = Path(
    "configs/research/scenario_prior_staging_contract_issue_3161.yaml"
)
REPORT_NAME = "scenario_prior_gap_report.md"
SUMMARY_NAME = "summary.json"
CSV_NAME = "parameter_comparisons.csv"

GROUP_AUTHORED = "authored"
GROUP_TRACE_DERIVED = "repository_trace_derived"
COMPARABLE_CLASSIFICATIONS = {GROUP_AUTHORED, GROUP_TRACE_DERIVED}

CLAIM_BOUNDARY = (
    "analysis_only_repository_prior_gap; no planner ranking, benchmark superiority, "
    "real-world representativeness, or dataset-backed prior claim"
)
DATASET_DEFERRAL = (
    "Dataset-backed SDD/ETH/AMV scenario-prior comparison is explicitly deferred to #3161."
)
NO_RANKING_NOTE = (
    "No planner ranking, benchmark superiority, or real-world representativeness is inferred."
)

PARAMETER_GROUPS: dict[str, tuple[str, ...]] = {
    "pedestrian_density": (
        "ped_density",
        "density_delta",
        "max_abs_density_delta",
        "max_ped_density",
        "max_pedestrian_density",
        "max_pedestrian_density_delta",
    ),
    "pedestrian_speed": (
        "pedestrian_speed",
        "pedestrian_speed_mps",
        "speed_delta",
        "speed_delta_m_s",
        "max_abs_speed_delta",
        "max_abs_speed_delta_m_s",
        "max_speed",
        "max_speed_m_s",
        "max_single_pedestrian_speed_delta",
        "max_single_pedestrian_speed_delta_m_s",
        "max_single_pedestrian_speed",
        "max_single_pedestrian_speed_m_s",
    ),
    "timing_offset_s": (
        "dt",
        "dt_s",
        "pedestrian_delay",
        "pedestrian_delay_s",
        "spawn_time",
        "spawn_time_s",
        "wait_delta",
        "wait_delta_s",
        "max_abs_dt",
        "max_abs_dt_s",
        "max_abs_wait_delta",
        "max_abs_wait_delta_s",
        "max_start_delay_offset",
        "max_start_delay_offset_s",
        "max_wait_duration_offset",
        "max_wait_duration_offset_s",
    ),
    "spatial_offset_m": (
        "dx",
        "dx_m",
        "dy",
        "dy_m",
        "max_magnitude",
        "max_magnitude_m",
        "max_route_offset",
        "max_route_offset_m",
        "max_single_pedestrian_trajectory_waypoint_offset",
        "max_single_pedestrian_trajectory_waypoint_offset_m",
    ),
    "route_coordinate_m": (
        "start_x",
        "start_y",
        "goal_x",
        "goal_y",
    ),
    "clearance_distance_m": (
        "min_distance",
        "mean_distance",
        "max_distance",
        "min_start_goal_distance",
        "min_start_goal_distance_m",
    ),
    "episode_horizon_steps": ("max_episode_steps",),
}
KEY_TO_GROUP = {key: group for group, keys in PARAMETER_GROUPS.items() for key in keys}

SKIP_KEY_PARTS = (
    "issue",
    "seed",
    "schema_version",
    "version",
    "hash",
    "checksum",
    "exit_code",
    "candidate_index",
    "reference_candidate_index",
    "jobs",
    "episodes",
    "failure",
    "valid",
    "written",
    "count",
    "rate",
)


class ScenarioPriorComparisonError(RuntimeError):
    """Raised when the Issue #2919 comparison cannot be completed."""


@dataclass(frozen=True)
class ParameterSample:
    """One extracted numeric prior sample."""

    parameter: str
    source_key: str
    value: float
    card_id: str
    classification: str
    source_path: str
    yaml_path: str


@dataclass
class GroupedParameterSamples:
    """Grouped samples and provenance for one canonical parameter."""

    values: list[float] = field(default_factory=list)
    source_keys: set[str] = field(default_factory=set)
    source_paths: set[str] = field(default_factory=set)


def repository_root() -> Path:
    """Return the current Git repository root."""

    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(result.stdout.strip())


def load_yaml_or_json(path: Path) -> Any:
    """Load a YAML or JSON document from a machine-readable source path."""

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ScenarioPriorComparisonError(f"could not read {path}: {exc}") from exc
    if path.suffix == ".json":
        return json.loads(text)
    return yaml.safe_load(text)


def iter_machine_readable_files(path: Path):
    """Yield YAML/JSON files for a card source trace."""

    if path.is_dir():
        yield from sorted(
            child
            for child in path.rglob("*")
            if child.is_file() and child.suffix.lower() in {".yaml", ".yml", ".json"}
        )
    elif path.is_file() and path.suffix.lower() in {".yaml", ".yml", ".json"}:
        yield path


def _is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _path_label(path: tuple[str, ...]) -> str:
    return ".".join(path)


def _should_skip_key(key: str) -> bool:
    normalized = key.lower()
    return any(part in normalized for part in SKIP_KEY_PARTS)


def _canonical_parameter(key: str) -> str | None:
    normalized = key.lower()
    if _should_skip_key(normalized):
        return None
    return KEY_TO_GROUP.get(normalized)


def _append_sample(
    samples: list[ParameterSample],
    *,
    card_id: str,
    classification: str,
    source_path: Path,
    repo_root: Path,
    yaml_path: tuple[str, ...],
    source_key: str,
    value: float,
) -> None:
    parameter = _canonical_parameter(source_key)
    if parameter is None:
        return
    samples.append(
        ParameterSample(
            parameter=parameter,
            source_key=source_key,
            value=float(value),
            card_id=card_id,
            classification=classification,
            source_path=source_path.relative_to(repo_root).as_posix(),
            yaml_path=_path_label(yaml_path),
        )
    )


def _range_values(node: dict[Any, Any]) -> list[float]:
    """Return min/max range endpoints from a mapping when present."""

    if _is_numeric(node.get("min")) and _is_numeric(node.get("max")):
        return [float(node["min"]), float(node["max"])]
    if _is_numeric(node.get("minimum")) and _is_numeric(node.get("maximum")):
        return [float(node["minimum"]), float(node["maximum"])]
    return []


def _append_range_samples(
    samples: list[ParameterSample],
    *,
    node: dict[Any, Any],
    path: tuple[str, ...],
    card_id: str,
    classification: str,
    source_path: Path,
    repo_root: Path,
) -> None:
    """Append range endpoints for a recognized range mapping."""

    parent_key = path[-1] if path else ""
    for value in _range_values(node):
        _append_sample(
            samples,
            card_id=card_id,
            classification=classification,
            source_path=source_path,
            repo_root=repo_root,
            yaml_path=path,
            source_key=parent_key,
            value=value,
        )


def _append_scalar_sample(
    samples: list[ParameterSample],
    *,
    node: Any,
    path: tuple[str, ...],
    card_id: str,
    classification: str,
    source_path: Path,
    repo_root: Path,
) -> None:
    """Append a scalar numeric sample when its key maps to a parameter group."""

    if not (_is_numeric(node) and path):
        return
    _append_sample(
        samples,
        card_id=card_id,
        classification=classification,
        source_path=source_path,
        repo_root=repo_root,
        yaml_path=path,
        source_key=path[-1],
        value=float(node),
    )


def extract_parameter_samples(
    payload: Any,
    *,
    card_id: str,
    classification: str,
    source_path: Path,
    repo_root: Path,
) -> list[ParameterSample]:
    """Extract compact numeric prior samples from a YAML/JSON payload."""

    samples: list[ParameterSample] = []

    def walk(node: Any, path: tuple[str, ...]) -> None:
        if isinstance(node, dict):
            _append_range_samples(
                samples,
                node=node,
                path=path,
                card_id=card_id,
                classification=classification,
                source_path=source_path,
                repo_root=repo_root,
            )
            for key, value in node.items():
                walk(value, (*path, str(key)))
            return
        if isinstance(node, list):
            for value in node:
                walk(value, path)
            return
        _append_scalar_sample(
            samples,
            node=node,
            path=path,
            card_id=card_id,
            classification=classification,
            source_path=source_path,
            repo_root=repo_root,
        )

    walk(payload, ())
    return samples


def load_registry_cards(registry_path: Path) -> list[dict[str, Any]]:
    """Load the Issue #2917 registry and return selected authored/trace-derived cards."""

    registry = load_yaml_or_json(registry_path)
    if not isinstance(registry, dict):
        raise ScenarioPriorComparisonError(f"registry is not a mapping: {registry_path}")
    cards = registry.get("cards")
    if not isinstance(cards, list):
        raise ScenarioPriorComparisonError(f"registry has no cards list: {registry_path}")
    selected = [
        card
        for card in cards
        if isinstance(card, dict) and card.get("classification") in COMPARABLE_CLASSIFICATIONS
    ]
    if not selected:
        raise ScenarioPriorComparisonError("registry contains no authored or trace-derived cards")
    return selected


def collect_samples(
    registry_path: Path, repo_root: Path
) -> tuple[list[ParameterSample], list[str]]:
    """Collect samples from machine-readable source traces in the prior-card registry."""

    samples: list[ParameterSample] = []
    skipped_sources: list[str] = []
    for card in load_registry_cards(registry_path):
        card_id = str(card.get("card_id", "unknown_card"))
        classification = str(card["classification"])
        source_traces = card.get("source_traces") or []
        if not isinstance(source_traces, list):
            skipped_sources.append(f"{card_id}: source_traces is not a list")
            continue
        for source in source_traces:
            source_path = (repo_root / str(source)).resolve()
            readable_files = list(iter_machine_readable_files(source_path))
            if not readable_files:
                skipped_sources.append(f"{card_id}: {source}")
                continue
            for readable_file in readable_files:
                payload = load_yaml_or_json(readable_file)
                samples.extend(
                    extract_parameter_samples(
                        payload,
                        card_id=card_id,
                        classification=classification,
                        source_path=readable_file,
                        repo_root=repo_root,
                    )
                )
    return samples, skipped_sources


def quantile(values: list[float], probability: float) -> float | None:
    """Return a linearly interpolated quantile for non-empty values."""

    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def distribution_summary(values: list[float]) -> dict[str, float | int | None]:
    """Return range and quantile statistics for one distribution."""

    ordered = sorted(values)
    return {
        "n": len(ordered),
        "min": ordered[0] if ordered else None,
        "q05": quantile(ordered, 0.05),
        "q25": quantile(ordered, 0.25),
        "median": quantile(ordered, 0.50),
        "q75": quantile(ordered, 0.75),
        "q95": quantile(ordered, 0.95),
        "max": ordered[-1] if ordered else None,
        "range": (ordered[-1] - ordered[0]) if len(ordered) > 1 else 0.0 if ordered else None,
    }


def empirical_cdf(values: list[float], threshold: float) -> float:
    """Return the empirical CDF value at threshold."""

    return sum(1 for value in values if value <= threshold) / len(values)


def ks_distance(left: list[float], right: list[float]) -> float | None:
    """Return a two-sample KS distance for non-empty samples."""

    if not left or not right:
        return None
    points = sorted(set(left + right))
    return max(abs(empirical_cdf(left, point) - empirical_cdf(right, point)) for point in points)


def wasserstein_like_distance(left: list[float], right: list[float]) -> float | None:
    """Return a simple 1D Wasserstein-like distance over matched quantile grid points."""

    if not left or not right:
        return None
    probabilities = [index / 100 for index in range(101)]
    distances = [
        abs(quantile(left, probability) - quantile(right, probability))
        for probability in probabilities
    ]
    return sum(distances) / len(distances)


def classify_prior_gap(authored: list[float], trace: list[float]) -> tuple[str, str]:
    """Classify authored samples relative to trace-derived samples."""

    authored_stats = distribution_summary(authored)
    trace_stats = distribution_summary(trace)
    authored_min = float(authored_stats["min"])
    authored_max = float(authored_stats["max"])
    authored_median = float(authored_stats["median"])
    trace_min = float(trace_stats["min"])
    trace_max = float(trace_stats["max"])
    trace_q25 = float(trace_stats["q25"])
    trace_q75 = float(trace_stats["q75"])
    trace_range = float(trace_stats["range"])
    authored_range = float(authored_stats["range"])
    tolerance = max(abs(trace_range) * 0.05, 1e-9)

    if trace_range <= tolerance:
        if authored_range <= tolerance and abs(authored_median - trace_min) <= tolerance:
            return "representative", "authored samples collapse near the trace-derived point mass"
        if authored_min <= trace_min <= authored_max:
            return (
                "too_broad",
                "trace-derived samples are point-like while authored range spans wider",
            )
        return "too_extreme", "authored samples miss the trace-derived point-like value"

    if authored_max < trace_q25 - tolerance or authored_min > trace_q75 + tolerance:
        return "too_extreme", "authored range is shifted outside the central trace-derived mass"
    if authored_min > trace_min + tolerance and authored_max < trace_max - tolerance:
        return "too_narrow", "authored range sits inside the trace-derived support"
    if authored_range < trace_range * 0.60:
        return "too_narrow", "authored range is substantially narrower than trace-derived support"
    if authored_range > trace_range * 1.60:
        return "too_broad", "authored range is substantially wider than trace-derived support"
    if authored_min < trace_min - tolerance or authored_max > trace_max + tolerance:
        return (
            "too_extreme",
            "authored range extends beyond trace-derived support without broad mismatch",
        )
    return "representative", "authored and trace-derived ranges broadly overlap"


def proposal_for(parameter: str, classification: str, trace_stats: dict[str, Any]) -> str:
    """Return a concrete scenario-family proposal for a classified parameter gap."""

    trace_low = trace_stats.get("q05")
    trace_high = trace_stats.get("q95")
    trace_median = trace_stats.get("median")
    if classification == "too_narrow":
        return (
            f"Add a `{parameter}_trace_span` family that samples the trace-derived q05-q95 "
            f"interval [{trace_low}, {trace_high}] while preserving certification gates."
        )
    if classification == "too_broad":
        return (
            f"Split authored `{parameter}` stress variants into a centered trace-aligned family "
            f"near median {trace_median} plus a separately labeled stress/extreme family."
        )
    if classification == "too_extreme":
        return (
            f"Add a `{parameter}_centered_trace_probe` family around trace median {trace_median} "
            "and keep out-of-support authored variants labeled diagnostic stress only."
        )
    return (
        f"Keep `{parameter}` as an audit baseline and add no planner-ranking interpretation; "
        "future dataset-backed calibration belongs in #3161."
    )


def group_samples(samples: list[ParameterSample]) -> dict[str, dict[str, GroupedParameterSamples]]:
    """Group samples by parameter and classification."""

    grouped: dict[str, dict[str, GroupedParameterSamples]] = defaultdict(
        lambda: defaultdict(GroupedParameterSamples)
    )
    for sample in samples:
        bucket = grouped[sample.parameter][sample.classification]
        bucket.values.append(sample.value)
        bucket.source_keys.add(sample.source_key)
        bucket.source_paths.add(sample.source_path)
    return grouped


def build_comparison_rows(samples: list[ParameterSample]) -> list[dict[str, Any]]:
    """Build per-parameter distribution comparison rows."""

    rows: list[dict[str, Any]] = []
    for parameter, by_classification in sorted(group_samples(samples).items()):
        authored = by_classification.get(GROUP_AUTHORED)
        trace = by_classification.get(GROUP_TRACE_DERIVED)
        if authored is None or trace is None or not authored.values or not trace.values:
            continue
        authored_stats = distribution_summary(authored.values)
        trace_stats = distribution_summary(trace.values)
        classification, rationale = classify_prior_gap(authored.values, trace.values)
        rows.append(
            {
                "parameter": parameter,
                "classification": classification,
                "classification_rationale": rationale,
                "proposal": proposal_for(parameter, classification, trace_stats),
                "authored": authored_stats,
                "trace_derived": trace_stats,
                "ks_distance": ks_distance(authored.values, trace.values),
                "wasserstein_like_distance": wasserstein_like_distance(
                    authored.values, trace.values
                ),
                "authored_source_keys": sorted(authored.source_keys),
                "trace_source_keys": sorted(trace.source_keys),
                "authored_source_paths": sorted(authored.source_paths),
                "trace_source_paths": sorted(trace.source_paths),
                "comparison_caveat": (
                    "Repository-authored config values and repository-trace-derived config/summary "
                    "values may mix offsets, caps, and realized values within a canonical parameter "
                    "family; use classifications as gap-finding proposals only."
                ),
            }
        )
    return rows


def _format_number(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Write a compact CSV table for review."""

    fieldnames = [
        "parameter",
        "classification",
        "authored_n",
        "authored_min",
        "authored_median",
        "authored_max",
        "trace_n",
        "trace_min",
        "trace_median",
        "trace_max",
        "ks_distance",
        "wasserstein_like_distance",
        "proposal",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "parameter": row["parameter"],
                    "classification": row["classification"],
                    "authored_n": row["authored"]["n"],
                    "authored_min": _format_number(row["authored"]["min"]),
                    "authored_median": _format_number(row["authored"]["median"]),
                    "authored_max": _format_number(row["authored"]["max"]),
                    "trace_n": row["trace_derived"]["n"],
                    "trace_min": _format_number(row["trace_derived"]["min"]),
                    "trace_median": _format_number(row["trace_derived"]["median"]),
                    "trace_max": _format_number(row["trace_derived"]["max"]),
                    "ks_distance": _format_number(row["ks_distance"]),
                    "wasserstein_like_distance": _format_number(row["wasserstein_like_distance"]),
                    "proposal": row["proposal"],
                }
            )


def analysis_base_commit(repo_root: Path) -> str:
    """Return the stable base commit used for repository-source comparison.

    The generated summary is tracked in the same commit as this script. Recording ``HEAD``
    would make the artifact self-referential and dirty on every post-commit reproduction.
    The merge base with ``origin/main`` is stable for the branch and identifies the repository
    inputs this analysis compared.
    """

    try:
        result = subprocess.run(
            ["git", "merge-base", "HEAD", "origin/main"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def load_dataset_staging_report(
    contract_path: Path | None,
    *,
    repo_root: Path,
) -> ScenarioPriorStagingContractReport | None:
    """Load optional Issue #3161 dataset-backed staging readiness report."""

    if contract_path is None:
        return None
    resolved_contract_path = (repo_root / contract_path).resolve()
    contract = load_scenario_prior_staging_contract(resolved_contract_path)
    return check_scenario_prior_staging_contract(
        contract,
        allowed_distribution_groups=set(PARAMETER_GROUPS),
        source=resolved_contract_path,
    )


def _dataset_staging_summary(
    staging_report: ScenarioPriorStagingContractReport | None,
) -> dict[str, Any] | None:
    """Return compact JSON-safe dataset-backed comparison readiness summary."""

    if staging_report is None:
        return None
    return {
        "schema_version": staging_report.schema_version,
        "contract_id": staging_report.contract_id,
        "issue": staging_report.issue,
        "evidence_boundary": staging_report.evidence_boundary,
        "contract_status": staging_report.contract_status,
        "dataset_backed_comparison_allowed": staging_report.dataset_backed_comparison_allowed,
        "comparison_ready_datasets": staging_report.comparison_ready_datasets,
        "blockers": staging_report.blockers,
        "datasets": [dataset.to_dict() for dataset in staging_report.datasets],
    }


def _dataset_staging_markdown_lines(
    staging_report: ScenarioPriorStagingContractReport | None,
) -> list[str]:
    """Return report lines for Issue #3161 dataset-backed staging readiness."""

    if staging_report is None:
        return [
            "- Dataset-backed staging contract was not checked in this run.",
            "- Dataset-backed comparison remains out of scope for this report.",
        ]
    lines = [
        f"- Contract status: `{staging_report.contract_status}`.",
        f"- Dataset-backed comparison allowed: `{staging_report.dataset_backed_comparison_allowed}`.",
        f"- Comparison-ready datasets: `{', '.join(staging_report.comparison_ready_datasets) or 'none'}`.",
        f"- Evidence boundary: {staging_report.evidence_boundary}.",
    ]
    if staging_report.blockers:
        lines.append("- Blockers:")
        lines.extend(f"  - {blocker}" for blocker in staging_report.blockers)
    lines.append("- Dataset statuses:")
    lines.extend(
        (
            f"  - `{dataset.dataset_id}`: declared `{dataset.declared_staging_status}`, "
            f"live `{dataset.live_staging_status or 'not-probed'}`, "
            f"ready `{dataset.comparison_ready}`."
        )
        for dataset in staging_report.datasets
    )
    return lines


def _dataset_not_ready_reasons(staging_report: ScenarioPriorStagingContractReport) -> list[str]:
    """Return actionable reasons dataset-backed comparison is not ready."""

    reasons = list(staging_report.blockers)
    reasons.extend(
        f"{dataset.dataset_id}: declared {dataset.declared_staging_status}"
        for dataset in staging_report.datasets
        if not dataset.comparison_ready and not dataset.blockers
    )
    return reasons


def write_summary(
    *,
    rows: list[dict[str, Any]],
    samples: list[ParameterSample],
    skipped_sources: list[str],
    output_dir: Path,
    repo_root: Path,
    registry_path: Path,
    dataset_staging_report: ScenarioPriorStagingContractReport | None = None,
) -> dict[str, Any]:
    """Write the machine-readable summary and return its payload."""

    payload: dict[str, Any] = {
        "schema_version": "scenario_prior_gap_issue_2919.v1",
        "issue": 2919,
        "registry_path": registry_path.relative_to(repo_root).as_posix(),
        "analysis_base_commit": analysis_base_commit(repo_root),
        "evidence_tier": "analysis_only",
        "claim_boundary": CLAIM_BOUNDARY,
        "dataset_comparison_deferral": DATASET_DEFERRAL,
        "dataset_backed_staging": _dataset_staging_summary(dataset_staging_report),
        "no_planner_ranking": NO_RANKING_NOTE,
        "sample_count": len(samples),
        "skipped_non_machine_readable_sources": skipped_sources,
        "comparison_count": len(rows),
        "classification_counts": {
            classification: sum(1 for row in rows if row["classification"] == classification)
            for classification in ("too_narrow", "too_broad", "too_extreme", "representative")
        },
        "comparisons": rows,
        "files": {
            "markdown_report": (output_dir / REPORT_NAME).relative_to(repo_root).as_posix(),
            "comparison_csv": (output_dir / CSV_NAME).relative_to(repo_root).as_posix(),
            "summary_json": (output_dir / SUMMARY_NAME).relative_to(repo_root).as_posix(),
        },
    }
    (output_dir / SUMMARY_NAME).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def write_markdown_report(
    *,
    rows: list[dict[str, Any]],
    output_dir: Path,
    repo_root: Path,
    registry_path: Path,
    dataset_staging_report: ScenarioPriorStagingContractReport | None = None,
) -> None:
    """Write the human-readable gap report."""

    lines = [
        "# Issue #2919 Scenario Prior Gap Report",
        "",
        "- Evidence status: `analysis_only`.",
        f"- Claim boundary: {CLAIM_BOUNDARY}.",
        f"- Registry input: `{registry_path.relative_to(repo_root).as_posix()}`.",
        f"- {DATASET_DEFERRAL}",
        f"- {NO_RANKING_NOTE}",
        "",
        "## Dataset-Backed Readiness",
        "",
        *_dataset_staging_markdown_lines(dataset_staging_report),
        "",
        "## Method",
        "",
        "The script loads the Issue #2917 prior-card registry, selects cards classified as "
        "`authored` or `repository_trace_derived`, and extracts numeric samples only from "
        "machine-readable YAML/JSON source traces. Text docs, Python scripts, raw datasets, and "
        "external-dataset candidate cards are excluded from this run.",
        "",
        "Distances are diagnostic: KS distance uses empirical CDF separation and the "
        "Wasserstein-like value averages absolute matched-quantile gaps over a 0-100% grid.",
        "",
        "## Parameter Comparisons",
        "",
        "| Parameter | Class | Authored range | Trace-derived range | KS | Wasserstein-like | Proposal |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        authored = row["authored"]
        trace = row["trace_derived"]
        authored_range = (
            f"{_format_number(authored['min'])} to {_format_number(authored['max'])} "
            f"(n={authored['n']})"
        )
        trace_range = (
            f"{_format_number(trace['min'])} to {_format_number(trace['max'])} (n={trace['n']})"
        )
        lines.append(
            "| {parameter} | `{classification}` | {authored_range} | {trace_range} | "
            "{ks} | {wasserstein} | {proposal} |".format(
                parameter=row["parameter"],
                classification=row["classification"],
                authored_range=authored_range,
                trace_range=trace_range,
                ks=_format_number(row["ks_distance"]),
                wasserstein=_format_number(row["wasserstein_like_distance"]),
                proposal=row["proposal"],
            )
        )
    lines.extend(
        [
            "",
            "## Classification Notes",
            "",
            "- `too_narrow`: authored support is materially inside the trace-derived support.",
            "- `too_broad`: authored support is materially wider than trace-derived support.",
            "- `too_extreme`: authored support is shifted outside trace-derived central mass or support.",
            "- `representative`: authored and trace-derived ranges broadly overlap for this repository-only comparison.",
            "",
            "## Limitations",
            "",
            "- This is not dataset-backed prior calibration; SDD/ETH/AMV comparison is deferred to #3161.",
            "- Repository trace-derived values come from existing repo configs and compact summaries, not raw real-world traces.",
            "- Some canonical groups mix offsets, caps, and realized values; proposals are scenario-family design prompts, not statistical claims.",
            "- No planner ranking, benchmark superiority, or real-world representativeness is inferred.",
            "",
        ]
    )
    (output_dir / REPORT_NAME).write_text("\n".join(lines), encoding="utf-8")


def run_comparison(
    *,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    repo_root: Path | None = None,
    dataset_staging_contract: Path | None = DEFAULT_DATASET_STAGING_CONTRACT_PATH,
    require_dataset_backed_ready: bool = False,
) -> dict[str, Any]:
    """Run the Issue #2919 comparison and write evidence files."""

    repo_root = repo_root or repository_root()
    registry_path = (repo_root / registry_path).resolve()
    output_dir = (repo_root / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_staging_report = load_dataset_staging_report(
        dataset_staging_contract,
        repo_root=repo_root,
    )
    if (
        require_dataset_backed_ready
        and dataset_staging_report is not None
        and dataset_staging_report.contract_status != CONTRACT_STATUS_READY
    ):
        raise ScenarioPriorComparisonError(
            "dataset-backed comparison requested but Issue #3161 staging contract is "
            f"{dataset_staging_report.contract_status!r}; "
            f"blockers: {'; '.join(_dataset_not_ready_reasons(dataset_staging_report)) or 'none'}"
        )
    samples, skipped_sources = collect_samples(registry_path, repo_root)
    rows = build_comparison_rows(samples)
    if not rows:
        raise ScenarioPriorComparisonError(
            "no comparable authored and trace-derived parameters found"
        )
    write_csv(rows, output_dir / CSV_NAME)
    write_markdown_report(
        rows=rows,
        output_dir=output_dir,
        repo_root=repo_root,
        registry_path=registry_path,
        dataset_staging_report=dataset_staging_report,
    )
    return write_summary(
        rows=rows,
        samples=samples,
        skipped_sources=skipped_sources,
        output_dir=output_dir,
        repo_root=repo_root,
        registry_path=registry_path,
        dataset_staging_report=dataset_staging_report,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Compare authored and repository-trace-derived scenario priors from the Issue #2917 "
            "prior-card registry. Dataset-backed SDD/ETH/AMV comparison is deferred to #3161."
        )
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=DEFAULT_REGISTRY_PATH,
        help=f"Prior-card registry path (default: {DEFAULT_REGISTRY_PATH})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Evidence output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--dataset-staging-contract",
        type=Path,
        default=DEFAULT_DATASET_STAGING_CONTRACT_PATH,
        help=(
            "Issue #3161 dataset-backed staging contract path "
            f"(default: {DEFAULT_DATASET_STAGING_CONTRACT_PATH})"
        ),
    )
    parser.add_argument(
        "--skip-dataset-staging-contract",
        action="store_true",
        help="Do not load the Issue #3161 dataset-backed staging contract.",
    )
    parser.add_argument(
        "--require-dataset-backed-ready",
        action="store_true",
        help="Fail unless the Issue #3161 staging contract permits dataset-backed comparison.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    dataset_staging_contract = (
        None if args.skip_dataset_staging_contract else args.dataset_staging_contract
    )
    try:
        summary = run_comparison(
            registry_path=args.registry,
            output_dir=args.output_dir,
            dataset_staging_contract=dataset_staging_contract,
            require_dataset_backed_ready=args.require_dataset_backed_ready,
        )
    except (ScenarioPriorComparisonError, ScenarioPriorStagingContractError) as exc:
        raise SystemExit(f"error: {exc}") from exc
    print(f"wrote {summary['files']['markdown_report']}")
    print(f"wrote {summary['files']['comparison_csv']}")
    print(f"wrote {summary['files']['summary_json']}")
    print(DATASET_DEFERRAL)
    print(NO_RANKING_NOTE)


if __name__ == "__main__":
    main()
