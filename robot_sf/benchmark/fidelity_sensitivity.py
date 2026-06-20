"""Utilities for simulation-fidelity sensitivity launch packets.

The helpers in this module are deliberately pure. They validate the issue #3207
study contract and compute rank-stability summaries for result rows, but they do
not run benchmark episodes or promote any planner-ranking claim by themselves.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

SCHEMA_VERSION = "fidelity-sensitivity.v1"
CLAIM_BOUNDARY = (
    "launch_packet_only_not_benchmark_evidence: defines deliberate simulation-fidelity "
    "sensitivity probes and rank-stability/drift metrics; does not establish planner "
    "ranking, sensor realism, sim-to-real validity, or paper-facing benchmark evidence."
)
DIAGNOSTIC_SMOKE_CLAIM_BOUNDARY = (
    "diagnostic_smoke_not_benchmark_evidence: summarizes a small local same-scenario "
    "fidelity sensitivity smoke. It can show wiring, metric drift, and rank-stability "
    "calculation behavior, but it does not establish planner ranking, simulator realism, "
    "sim-to-real validity, or paper-facing benchmark evidence."
)


def load_fidelity_sensitivity_config(path: str | Path) -> dict[str, Any]:
    """Load and validate a fidelity-sensitivity YAML config.

    Returns:
        Validated config mapping.
    """
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"fidelity-sensitivity config must be a mapping: {config_path}")
    return validate_fidelity_sensitivity_config(payload)


def validate_fidelity_sensitivity_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the issue #3207 launch-packet config contract.

    Returns:
        Shallow-normalized dictionary with the original config values.
    """
    normalized = dict(config)
    if normalized.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION!r}")
    _validate_axes(normalized.get("axes"))
    _validate_fixed_scope(normalized.get("fixed_scope"))
    _validate_metrics(normalized.get("ranking"), normalized.get("metrics"))
    return normalized


def _validate_axes(axes: Any) -> None:
    """Validate the fidelity-axis list."""
    if not isinstance(axes, list) or len(axes) < 3:
        raise ValueError("fidelity-sensitivity config must define at least three axes")

    for axis in axes:
        if not isinstance(axis, dict):
            raise ValueError("each fidelity axis must be a mapping")
        axis_key = axis.get("key")
        if not axis_key:
            raise ValueError("each fidelity axis must define a key")
        variants = axis.get("variants")
        if not isinstance(variants, list) or len(variants) < 2:
            raise ValueError(f"axis {axis_key!r} must define at least two variants")
        baseline_count = _validate_axis_variants(axis_key=str(axis_key), variants=variants)
        if baseline_count != 1:
            raise ValueError(f"axis {axis_key!r} must mark exactly one baseline variant")


def _validate_axis_variants(*, axis_key: str, variants: list[Any]) -> int:
    """Validate one axis variant list.

    Returns:
        Number of variants marked as baseline.
    """
    baseline_count = 0
    seen_variant_keys: set[str] = set()
    for variant in variants:
        if not isinstance(variant, dict):
            raise ValueError(f"axis {axis_key!r} variants must be mappings")
        variant_key = str(variant.get("key") or "")
        if not variant_key:
            raise ValueError(f"axis {axis_key!r} variant must define a key")
        if variant_key in seen_variant_keys:
            raise ValueError(f"axis {axis_key!r} repeats variant key {variant_key!r}")
        seen_variant_keys.add(variant_key)
        baseline_count += int(bool(variant.get("baseline", False)))
    return baseline_count


def _validate_fixed_scope(fixed_scope: Any) -> None:
    """Validate fixed scenario/seed/planner scope."""
    if not isinstance(fixed_scope, dict):
        raise ValueError("fixed_scope must be a mapping")
    _require_non_empty_list(fixed_scope, "seeds")
    _require_non_empty_list(fixed_scope, "planner_groups")
    if not fixed_scope.get("scenario_set"):
        raise ValueError("fixed_scope.scenario_set must be set")


def _validate_metrics(ranking: Any, metrics: Any) -> None:
    """Validate ranking metric membership in the metric list."""
    if not isinstance(ranking, dict) or not ranking.get("metric"):
        raise ValueError("ranking.metric must be set")
    if not isinstance(metrics, list) or not metrics:
        raise ValueError("metrics must be a non-empty list")
    metric_names = {str(item.get("name")) for item in metrics if isinstance(item, dict)}
    if str(ranking["metric"]) not in metric_names:
        raise ValueError("ranking.metric must appear in metrics")


def rank_order(scores: Mapping[str, float], *, higher_is_better: bool) -> list[str]:
    """Return a stable ranking order from a group-to-score mapping.

    Returns:
        Ordered group names, best first under the requested metric direction.
    """
    clean_scores = {str(key): _finite_float(value, key=str(key)) for key, value in scores.items()}
    return sorted(
        clean_scores,
        key=lambda key: (
            -clean_scores[key] if higher_is_better else clean_scores[key],
            key,
        ),
    )


def kendall_tau_from_orders(baseline_order: Sequence[str], variant_order: Sequence[str]) -> float:
    """Compute Kendall tau for two complete rankings over the same items.

    Returns:
        Kendall tau in [-1, 1].
    """
    left = list(baseline_order)
    right = list(variant_order)
    if set(left) != set(right):
        raise ValueError("rank orders must contain the same items")
    n = len(left)
    if n < 2:
        return 1.0
    right_rank = {item: rank for rank, item in enumerate(right)}
    concordant = 0
    discordant = 0
    for i, item_i in enumerate(left):
        for j in range(i + 1, n):
            item_j = left[j]
            baseline_delta = i - j
            variant_delta = right_rank[item_i] - right_rank[item_j]
            if baseline_delta * variant_delta > 0:
                concordant += 1
            else:
                discordant += 1
    pairs = n * (n - 1) / 2
    return float((concordant - discordant) / pairs)


def rank_flip_count(baseline_order: Sequence[str], variant_order: Sequence[str]) -> int:
    """Count groups whose rank position changes between two complete orders.

    Returns:
        Number of groups with changed ordinal rank.
    """
    if set(baseline_order) != set(variant_order):
        raise ValueError("rank orders must contain the same items")
    variant_rank = {item: rank for rank, item in enumerate(variant_order)}
    return sum(1 for rank, item in enumerate(baseline_order) if variant_rank[item] != rank)


def build_rank_stability_summary(
    baseline_scores: Mapping[str, float],
    variant_scores: Mapping[str, float],
    *,
    higher_is_better: bool,
    min_tau: float = 0.8,
) -> dict[str, Any]:
    """Summarize rank stability for a variant against the nominal baseline.

    Returns:
        Ranking orders, Kendall tau, flip count, and stability flags.
    """
    baseline_order = rank_order(baseline_scores, higher_is_better=higher_is_better)
    variant_order = rank_order(variant_scores, higher_is_better=higher_is_better)
    tau = kendall_tau_from_orders(baseline_order, variant_order)
    flips = rank_flip_count(baseline_order, variant_order)
    return {
        "baseline_order": baseline_order,
        "variant_order": variant_order,
        "kendall_tau_vs_baseline": tau,
        "rank_flip_count": flips,
        "ranking_flipped": flips > 0,
        "stable_by_tau_threshold": tau >= float(min_tau),
    }


def metric_drift(
    baseline_metrics: Mapping[str, float],
    variant_metrics: Mapping[str, float],
) -> dict[str, dict[str, float | None]]:
    """Compute absolute and relative metric drift for common metric keys.

    Returns:
        Per-metric baseline, variant, absolute delta, and relative delta.
    """
    common_metrics = sorted(set(baseline_metrics) & set(variant_metrics))
    drift: dict[str, dict[str, float | None]] = {}
    for metric in common_metrics:
        baseline = _finite_float(baseline_metrics[metric], key=metric)
        variant = _finite_float(variant_metrics[metric], key=metric)
        delta = variant - baseline
        relative = None if baseline == 0.0 else delta / baseline
        drift[metric] = {
            "baseline": baseline,
            "variant": variant,
            "absolute_delta": delta,
            "relative_delta": relative,
        }
    return drift


def build_launch_packet(
    config: Mapping[str, Any],
    *,
    config_path: str,
    git_head: str,
    date: str | None = None,
) -> dict[str, Any]:
    """Build a compact launch-packet summary for issue #3207.

    Returns:
        JSON-serializable launch-packet payload.
    """
    validated = validate_fidelity_sensitivity_config(config)
    axes = validated["axes"]
    claim_boundary = validated.get("claim_boundary")
    return {
        "schema_version": "fidelity-sensitivity-launch-packet.v1",
        "issue": int(validated.get("issue", 3207)),
        "study_id": str(validated["study_id"]),
        "status": "launch_packet_only",
        "config_path": config_path,
        "git_head": git_head,
        "date": date,
        "claim_boundary": str(claim_boundary) if claim_boundary is not None else CLAIM_BOUNDARY,
        "axis_count": len(axes),
        "axes": [
            {
                "key": str(axis["key"]),
                "variant_count": len(axis["variants"]),
                "baseline_variant": _baseline_variant_key(axis),
                "rationale": str(axis.get("rationale", "")),
            }
            for axis in axes
        ],
        "fixed_scope": validated["fixed_scope"],
        "ranking": validated["ranking"],
        "metrics": validated["metrics"],
        "result_contract": validated["result_contract"],
        "next_command_template": (
            "uv run python scripts/<fidelity_sweep>.py --config "
            f"{config_path} --out output/fidelity_sensitivity/"
        ),
    }


def format_launch_packet_markdown(packet: Mapping[str, Any]) -> str:
    """Format a fidelity-sensitivity launch packet as Markdown.

    Returns:
        Markdown report text.
    """
    axes = packet["axes"]
    issue = packet.get("issue", 3207)
    date_suffix = f" {packet['date']}" if packet.get("date") else ""
    lines = [
        f"# Issue #{issue} Fidelity Sensitivity Launch Packet{date_suffix}",
        "",
        f"- Status: `{packet['status']}`",
        f"- Study: `{packet['study_id']}`",
        f"- Config: `{packet['config_path']}`",
        f"- Git head: `{packet['git_head']}`",
        f"- Claim boundary: {packet['claim_boundary']}",
        "",
        "## Scope",
        "",
        f"- Scenario set: `{packet['fixed_scope']['scenario_set']}`",
        f"- Seeds: `{', '.join(str(seed) for seed in packet['fixed_scope']['seeds'])}`",
        f"- Planner groups: `{', '.join(packet['fixed_scope']['planner_groups'])}`",
        f"- Ranking metric: `{packet['ranking']['metric']}`",
        "",
        "## Fidelity Axes",
        "",
        "| Axis | Variants | Baseline Variant | Rationale |",
        "|---|---:|---|---|",
    ]
    for axis in axes:
        lines.append(
            f"| `{axis['key']}` | {axis['variant_count']} | "
            f"`{axis['baseline_variant']}` | {axis['rationale']} |"
        )
    lines.extend(
        [
            "",
            "## Result Contract",
            "",
            "Before this packet can support a validity-boundary claim, the sweep output must report:",
        ]
    )
    for field in packet["result_contract"]["required_outputs"]:
        lines.append(f"- `{field}`")
    lines.extend(
        [
            "",
            "Any axis with `rank_flip_count > 0` is a caveat/calibration candidate.",
            "",
            "## Next Command Template",
            "",
            "```bash",
            str(packet["next_command_template"]),
            "```",
            "",
            "This packet is not benchmark evidence until a sweep is run and promoted with provenance.",
            "",
        ]
    )
    return "\n".join(lines)


def write_launch_packet(packet: Mapping[str, Any], output_dir: str | Path) -> None:
    """Write JSON and Markdown launch-packet files."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "launch_packet.json").write_text(
        json.dumps(packet, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out / "README.md").write_text(format_launch_packet_markdown(packet), encoding="utf-8")


def _baseline_variant_key(axis: Mapping[str, Any]) -> str:
    for variant in axis["variants"]:
        if variant.get("baseline", False):
            return str(variant["key"])
    raise ValueError(f"axis {axis.get('key')!r} has no baseline variant")


def _finite_float(value: Any, *, key: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be numeric") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{key} must be finite")
    return numeric


def _require_non_empty_list(mapping: Mapping[str, Any], key: str) -> None:
    value = mapping.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"fixed_scope.{key} must be a non-empty list")


__all__ = [
    "CLAIM_BOUNDARY",
    "DIAGNOSTIC_SMOKE_CLAIM_BOUNDARY",
    "SCHEMA_VERSION",
    "build_launch_packet",
    "build_rank_stability_summary",
    "format_launch_packet_markdown",
    "kendall_tau_from_orders",
    "load_fidelity_sensitivity_config",
    "metric_drift",
    "rank_flip_count",
    "rank_order",
    "validate_fidelity_sensitivity_config",
    "write_launch_packet",
]
