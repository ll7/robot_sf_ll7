#!/usr/bin/env python3
"""Bounded same-seed robot-influence-on-pedestrian-flow slice for issue #3066.

This script drives a tiny, fail-closed, same-seed benchmark campaign that compares
>=2 distinct *robot* policies (default: ``social_force`` and ``orca``) on a
corridor/crossing scenario subset, and asks a single bounded question:

    Does the robot's policy measurably change pedestrian-flow dynamics, beyond
    ordinary seed-to-seed variance?

It reuses the repository benchmark runner (``robot_sf.benchmark.runner.run_batch``)
to execute episodes; it does NOT reimplement the simulator. Each emitted episode
row carries success/safety metrics PLUS pedestrian flow/density/speed/deviation/
comfort proxies and uncertainty/denominator fields. The script then:

  * classifies each row's execution status (fail-closed: degraded / fallback /
    unavailable / failed rows are NEVER counted as usable evidence),
  * separates *robot-influence* signals (how the robot perturbs nearby
    pedestrians) from ordinary *nav-performance* signals (success/collision),
  * computes policy-vs-policy flow deltas and compares them against same-policy
    seed variance,
  * emits an overall classification in {benchmark, diagnostic, blocked, non_claim}.

HONESTY / CLAIM BOUNDARY (issue #3066, evidence_tier=smoke):
  * This is a LOCAL bounded v0 diagnostic. It is NOT a real-world pedestrian-flow
    claim, NOT sim-to-real, NOT paper-grade.
  * A null result (flow deltas within seed variance) is an honest, acceptable
    outcome and is reported as ``diagnostic``.
  * If the campaign cannot execute (no usable rows for a policy/scenario), the
    overall classification is ``blocked`` with the concrete reason -- numbers are
    never fabricated to force a success.

The robot-influence signal of record is the experimental near-vs-far pedestrian
reduction (``ped_impact_accel_delta_mean`` / ``ped_impact_turn_rate_delta_mean``):
the difference between pedestrian acceleration / turn-rate when *near* the robot
versus *far* from it. Larger magnitude => the robot perturbs pedestrian motion
more. These are diagnostic simulation proxies, not validated human measures.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- #
# Constants / vocabulary (kept stable -- tests assert on these literals).
# --------------------------------------------------------------------------- #

CLASSIFICATIONS = ("benchmark", "diagnostic", "blocked", "non_claim")
"""Overall classification vocabulary. Stable: tests assert on these strings."""

CLAIM_BOUNDARY = "diagnostic_only"
EVIDENCE_TIER = "smoke"
PAPER_GRADE = False

# Per-row execution status: which statuses are USABLE as evidence vs fail-closed.
# A benchmark nav-outcome of "failure" (timeout / did-not-reach-goal) is still a
# *usable* row for flow analysis -- the robot ran and perturbed pedestrians. What
# is fail-closed is *degraded execution* (fallback/degraded/unavailable/error):
# the policy did not genuinely run.
USABLE_ROW_STATUSES = ("success", "failure")
"""Episode-level nav outcomes that still yield a usable physics row."""

DEGRADED_ROW_STATUSES = ("degraded", "fallback", "unavailable", "error", "failed")
"""Execution statuses that are fail-closed: never counted as usable evidence."""

# Flow / influence metric keys we extract from each episode's ``metrics`` block.
INFLUENCE_METRIC_KEYS = (
    "ped_impact_accel_delta_mean",  # robot-influence: near-vs-far accel reduction
    "ped_impact_turn_rate_delta_mean",  # robot-influence: near-vs-far turn-rate
)
DENOMINATOR_KEYS = (
    "ped_impact_near_samples",
    "ped_impact_far_samples",
    "ped_impact_ped_count",
    "ped_impact_accel_delta_valid",
    "ped_impact_turn_rate_delta_valid",
)
# Additional flow/density/speed/deviation/comfort context (reported, not the
# primary delta-vs-variance test).
CONTEXT_METRIC_KEYS = (
    "avg_speed",  # robot speed proxy
    "robot_ped_within_5m_frac",  # density / exposure proxy
    "social_proxemic_intrusion_frac",  # comfort proxy
    "min_distance",  # deviation / closest-approach proxy
    "mean_clearance",  # deviation / clearance proxy
    "near_misses",  # safety
    "near_miss_rate",
)
NAV_METRIC_KEYS = (
    "success",  # nav performance (kept separate from influence)
    "total_collision_count",
    "ped_collision_count",
)

DEFAULT_SCHEMA = "robot_sf/benchmark/schemas/episode.schema.v1.json"
DEFAULT_POLICIES = ("social_force", "orca")
DEFAULT_SEEDS = (111, 112, 113)
# Corridor + crossing subset of the issue_3059 suite (map-based scenarios).
DEFAULT_SCENARIOS: tuple[dict[str, Any], ...] = (
    {
        "name": "classic_head_on_corridor_low",
        "map_rel": "maps/svg_maps/classic_head_on_corridor.svg",
        "archetype": "head_on_corridor",
        "ped_density": 0.02,
        "max_episode_steps": 240,
    },
    {
        "name": "classic_crossing_low",
        "map_rel": "maps/svg_maps/classic_crossing.svg",
        "archetype": "crossing",
        "ped_density": 0.02,
        "max_episode_steps": 240,
    },
)


# --------------------------------------------------------------------------- #
# Provenance.
# --------------------------------------------------------------------------- #


def git_head(repo_root: Path) -> str:
    """Return the current git HEAD short hash, or ``"unknown"`` on failure.

    Args:
        repo_root: Repository root used as the git working directory.

    Returns:
        The abbreviated commit hash string, or ``"unknown"`` if git is unavailable.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"


# --------------------------------------------------------------------------- #
# Row classification (pure -- unit tested on synthetic rows).
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FlowRow:
    """One classified episode row for the robot-influence flow slice."""

    policy: str
    scenario: str
    seed: int
    row_status: str  # raw episode status as emitted by the runner
    usable: bool  # True only when execution was not degraded/fallback/etc.
    degraded_reason: str | None
    influence: dict[str, float]  # robot-influence flow deltas (may hold NaN)
    context: dict[str, float]  # density/speed/deviation/comfort proxies
    nav: dict[str, float]  # nav-performance metrics, kept separate
    denominators: dict[str, float]


def _coerce_float(value: Any) -> float:
    """Coerce a metric value to float, mapping missing/bad inputs to NaN.

    Returns:
        The float value, or ``float('nan')`` when the value is missing or
        non-numeric (e.g. ``None`` or a bool used as a sentinel).
    """
    if value is None or isinstance(value, bool):
        # bools are intentionally excluded here: success/flags handled elsewhere.
        return float(value) if isinstance(value, bool) else float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def classify_row(record: dict[str, Any]) -> FlowRow:
    """Classify a single runner episode record into a :class:`FlowRow`.

    The episode ``status`` distinguishes a usable physics row (``success`` /
    ``failure`` = a real run, possibly a nav timeout) from a degraded/fallback
    execution that must be fail-closed.

    Args:
        record: One parsed JSONL episode record from the benchmark runner.

    Returns:
        A classified :class:`FlowRow`.
    """
    metrics = record.get("metrics", {}) or {}
    raw_status = str(record.get("status", "")).strip().lower()
    algo_meta = record.get("algorithm_metadata", {}) or {}

    # Detect degraded/fallback execution from status or algorithm metadata.
    degraded_reason: str | None = None
    usable = True
    if raw_status in DEGRADED_ROW_STATUSES:
        usable = False
        degraded_reason = f"episode status '{raw_status}'"
    elif raw_status not in USABLE_ROW_STATUSES:
        usable = False
        degraded_reason = f"unrecognized episode status '{raw_status}'"
    # Algorithm-level fallback signal (e.g. adapter fallback) is fail-closed too.
    fallback_flag = algo_meta.get("fallback") or algo_meta.get("availability_fallback")
    if fallback_flag:
        usable = False
        degraded_reason = (degraded_reason or "") + " algorithm fallback active"

    influence = {k: _coerce_float(metrics.get(k)) for k in INFLUENCE_METRIC_KEYS}
    context = {k: _coerce_float(metrics.get(k)) for k in CONTEXT_METRIC_KEYS}
    nav = {k: _coerce_float(metrics.get(k)) for k in NAV_METRIC_KEYS}
    denominators = {k: _coerce_float(metrics.get(k)) for k in DENOMINATOR_KEYS}

    return FlowRow(
        policy=str(record.get("algo", "")),
        scenario=str(record.get("scenario_id", "")),
        seed=int(record.get("seed", -1)),
        row_status=raw_status or "missing",
        usable=usable,
        degraded_reason=degraded_reason.strip() if degraded_reason else None,
        influence=influence,
        context=context,
        nav=nav,
        denominators=denominators,
    )


# --------------------------------------------------------------------------- #
# Aggregation + delta-vs-variance logic (pure -- unit tested).
# --------------------------------------------------------------------------- #


def _finite(values: list[float]) -> list[float]:
    """Return only the finite (non-NaN, non-inf) values of a list.

    Returns:
        A list with NaN and infinite entries removed.
    """
    return [v for v in values if isinstance(v, float) and math.isfinite(v)]


def _mean(values: list[float]) -> float:
    """Return the mean of finite values, or NaN when none are finite.

    Returns:
        Arithmetic mean of finite entries, or ``float('nan')``.
    """
    finite = _finite(values)
    return statistics.fmean(finite) if finite else float("nan")


def _stdev(values: list[float]) -> float:
    """Return the sample stdev of finite values; 0.0 for a single value, NaN for none.

    Returns:
        Sample standard deviation, ``0.0`` for one value, or ``float('nan')``.
    """
    finite = _finite(values)
    if len(finite) >= 2:
        return statistics.stdev(finite)
    if len(finite) == 1:
        return 0.0
    return float("nan")


@dataclass
class PolicyScenarioAggregate:
    """Aggregate of usable rows for one (policy, scenario) cell."""

    policy: str
    scenario: str
    n_rows: int = 0
    n_usable: int = 0
    n_degraded: int = 0
    influence_mean: dict[str, float] = field(default_factory=dict)
    influence_std: dict[str, float] = field(default_factory=dict)
    context_mean: dict[str, float] = field(default_factory=dict)
    nav_mean: dict[str, float] = field(default_factory=dict)
    denom_sum: dict[str, float] = field(default_factory=dict)


def aggregate_cell(rows: list[FlowRow]) -> PolicyScenarioAggregate:
    """Aggregate usable rows for a single (policy, scenario) cell.

    Args:
        rows: All rows (usable and degraded) for one policy+scenario cell.

    Returns:
        A :class:`PolicyScenarioAggregate` summarizing usable rows. Degraded
        rows are counted but excluded from all metric means (fail-closed).
    """
    if not rows:
        msg = "aggregate_cell requires at least one row"
        raise ValueError(msg)
    policy = rows[0].policy
    scenario = rows[0].scenario
    agg = PolicyScenarioAggregate(policy=policy, scenario=scenario, n_rows=len(rows))
    usable = [r for r in rows if r.usable]
    agg.n_usable = len(usable)
    agg.n_degraded = len(rows) - len(usable)

    for key in INFLUENCE_METRIC_KEYS:
        vals = [r.influence.get(key, float("nan")) for r in usable]
        agg.influence_mean[key] = _mean(vals)
        agg.influence_std[key] = _stdev(vals)
    for key in CONTEXT_METRIC_KEYS:
        agg.context_mean[key] = _mean([r.context.get(key, float("nan")) for r in usable])
    for key in NAV_METRIC_KEYS:
        agg.nav_mean[key] = _mean([r.nav.get(key, float("nan")) for r in usable])
    for key in DENOMINATOR_KEYS:
        finite = _finite([r.denominators.get(key, float("nan")) for r in usable])
        agg.denom_sum[key] = float(sum(finite)) if finite else 0.0
    return agg


@dataclass
class FlowDelta:
    """Policy-vs-policy flow delta for one scenario + influence metric."""

    scenario: str
    metric: str
    policy_a: str
    policy_b: str
    mean_a: float
    mean_b: float
    delta: float  # mean_b - mean_a
    seed_variance: float  # pooled per-policy seed stdev
    exceeds_variance: bool
    powered: bool  # enough usable, finite samples to interpret
    note: str


def _pooled_seed_variance(std_a: float, std_b: float) -> float:
    """Pool two per-policy seed stdevs into a single variance scale.

    Returns:
        The root-mean-square of the two finite stdevs, or ``float('nan')`` when
        neither is finite.
    """
    finite = _finite([std_a, std_b])
    if not finite:
        return float("nan")
    return math.sqrt(sum(s * s for s in finite) / len(finite))


def compute_flow_deltas(
    aggregates: dict[tuple[str, str], PolicyScenarioAggregate],
    policy_a: str,
    policy_b: str,
    *,
    min_usable_per_cell: int = 2,
) -> list[FlowDelta]:
    """Compute policy-vs-policy flow deltas per scenario and influence metric.

    A delta is flagged ``exceeds_variance`` only when its magnitude is strictly
    greater than the pooled per-policy seed variance scale AND both cells are
    ``powered`` (>= ``min_usable_per_cell`` usable rows with a finite mean).
    Underpowered comparisons never assert influence.

    Args:
        aggregates: Mapping of (policy, scenario) -> aggregate.
        policy_a: Baseline policy name (e.g. ``"social_force"``).
        policy_b: Comparison policy name (e.g. ``"orca"``).
        min_usable_per_cell: Minimum usable rows per cell to call it powered.

    Returns:
        A list of :class:`FlowDelta`, one per (scenario, influence metric).
    """
    scenarios = sorted({scen for (_pol, scen) in aggregates})
    deltas: list[FlowDelta] = []
    for scenario in scenarios:
        agg_a = aggregates.get((policy_a, scenario))
        agg_b = aggregates.get((policy_b, scenario))
        for metric in INFLUENCE_METRIC_KEYS:
            if agg_a is None or agg_b is None:
                deltas.append(
                    FlowDelta(
                        scenario=scenario,
                        metric=metric,
                        policy_a=policy_a,
                        policy_b=policy_b,
                        mean_a=float("nan"),
                        mean_b=float("nan"),
                        delta=float("nan"),
                        seed_variance=float("nan"),
                        exceeds_variance=False,
                        powered=False,
                        note="missing policy cell",
                    )
                )
                continue
            mean_a = agg_a.influence_mean.get(metric, float("nan"))
            mean_b = agg_b.influence_mean.get(metric, float("nan"))
            std_a = agg_a.influence_std.get(metric, float("nan"))
            std_b = agg_b.influence_std.get(metric, float("nan"))
            seed_var = _pooled_seed_variance(std_a, std_b)
            powered = (
                agg_a.n_usable >= min_usable_per_cell
                and agg_b.n_usable >= min_usable_per_cell
                and math.isfinite(mean_a)
                and math.isfinite(mean_b)
            )
            delta = (
                mean_b - mean_a
                if (math.isfinite(mean_a) and math.isfinite(mean_b))
                else float("nan")
            )
            exceeds = bool(
                powered
                and math.isfinite(delta)
                and math.isfinite(seed_var)
                and abs(delta) > seed_var
            )
            if not powered:
                note = "underpowered: insufficient usable/finite rows"
            elif not math.isfinite(seed_var):
                note = "seed variance undefined"
            elif exceeds:
                note = "delta magnitude exceeds pooled seed variance"
            else:
                note = "delta within pooled seed variance"
            deltas.append(
                FlowDelta(
                    scenario=scenario,
                    metric=metric,
                    policy_a=policy_a,
                    policy_b=policy_b,
                    mean_a=mean_a,
                    mean_b=mean_b,
                    delta=delta,
                    seed_variance=seed_var,
                    exceeds_variance=exceeds,
                    powered=powered,
                    note=note,
                )
            )
    return deltas


def classify_overall(
    aggregates: dict[tuple[str, str], PolicyScenarioAggregate],
    deltas: list[FlowDelta],
    policies: tuple[str, ...],
) -> tuple[str, str]:
    """Decide the overall campaign classification and a one-line rationale.

    Decision order (fail-closed):
      1. ``blocked`` -- any policy produced zero usable rows anywhere, so a
         same-policy comparison is impossible.
      2. ``diagnostic`` -- usable rows exist but no powered delta exceeds seed
         variance (an honest null / underpowered outcome).
      3. ``diagnostic`` -- at least one powered delta exceeds seed variance.
         Even a positive signal stays ``diagnostic`` at v0 smoke tier: this is
         not a benchmark-strength or real-world claim. ``benchmark`` is reserved
         and never returned by this v0 driver.

    Args:
        aggregates: (policy, scenario) -> aggregate.
        deltas: Computed flow deltas.
        policies: The policies that were requested.

    Returns:
        A tuple ``(classification, rationale)`` with classification in
        :data:`CLASSIFICATIONS`.
    """
    # Fail-closed: each requested policy must have at least one usable row.
    for policy in policies:
        usable_total = sum(
            agg.n_usable for (pol, _scen), agg in aggregates.items() if pol == policy
        )
        if usable_total == 0:
            return (
                "blocked",
                f"policy '{policy}' produced zero usable rows; "
                "same-seed comparison impossible (fail-closed)",
            )

    powered_deltas = [d for d in deltas if d.powered]
    if not powered_deltas:
        return (
            "diagnostic",
            "no powered policy-vs-policy comparison available; "
            "result is diagnostic-only (underpowered)",
        )
    exceed = [d for d in powered_deltas if d.exceeds_variance]
    if exceed:
        return (
            "diagnostic",
            f"{len(exceed)}/{len(powered_deltas)} powered flow deltas exceed seed "
            "variance; robot-policy influence is suggested but remains diagnostic-only "
            "at v0 smoke tier (not a real-world or benchmark-strength claim)",
        )
    return (
        "diagnostic",
        "all powered flow deltas fall within seed variance; "
        "no robot-policy influence detected (honest null, diagnostic-only)",
    )


# --------------------------------------------------------------------------- #
# Campaign execution (reuses runner; not unit-tested live).
# --------------------------------------------------------------------------- #


def _write_matrix(
    scenario: dict[str, Any],
    seeds: tuple[int, ...],
    repo_root: Path,
    out_dir: Path,
) -> Path:
    """Write a single-scenario benchmark matrix YAML and return its path.

    Returns:
        Path to the written matrix YAML file.
    """
    import yaml

    map_abs = (repo_root / scenario["map_rel"]).resolve()
    matrix = {
        "scenarios": [
            {
                "name": scenario["name"],
                "map_file": str(map_abs),
                "simulation_config": {
                    "max_episode_steps": int(scenario["max_episode_steps"]),
                    "ped_density": float(scenario["ped_density"]),
                },
                "robot_config": {},
                "metadata": {
                    "archetype": scenario["archetype"],
                    "density": "low",
                    "flow": "bi",
                },
                "seeds": list(seeds),
            }
        ]
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"matrix_{scenario['name']}.yaml"
    path.write_text(yaml.safe_dump(matrix, sort_keys=False), encoding="utf-8")
    return path


def run_campaign(
    *,
    policies: tuple[str, ...],
    scenarios: tuple[dict[str, Any], ...],
    seeds: tuple[int, ...],
    repo_root: Path,
    out_dir: Path,
    horizon: int,
    schema_path: str,
) -> list[dict[str, Any]]:
    """Execute the bounded same-seed campaign and return all episode records.

    Reuses ``robot_sf.benchmark.runner.run_batch`` for every (policy, scenario)
    cell; the same ``seeds`` are reused across policies (same-seed comparison).

    Returns:
        A flat list of parsed JSONL episode records across all cells.
    """
    from robot_sf.benchmark.runner import run_batch

    records: list[dict[str, Any]] = []
    episodes_dir = out_dir / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    for scenario in scenarios:
        matrix_path = _write_matrix(scenario, seeds, repo_root, out_dir / "matrices")
        for policy in policies:
            jsonl = episodes_dir / f"{policy}__{scenario['name']}.jsonl"
            if jsonl.exists():
                jsonl.unlink()
            run_batch(
                str(matrix_path),
                str(jsonl),
                schema_path,
                algo=policy,
                horizon=horizon,
                experimental_ped_impact=True,
                append=False,
                resume=False,
                fail_fast=False,
            )
            if jsonl.exists():
                for line in jsonl.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
    return records


# --------------------------------------------------------------------------- #
# Report assembly.
# --------------------------------------------------------------------------- #


def build_report(
    *,
    records: list[dict[str, Any]],
    policies: tuple[str, ...],
    scenarios: tuple[str, ...],
    seeds: tuple[int, ...],
    git_hash: str,
    horizon: int,
) -> dict[str, Any]:
    """Assemble the JSON-serializable report from raw episode records.

    Returns:
        A report dict with rows, aggregates, flow deltas, and classification.
    """
    rows = [classify_row(rec) for rec in records]
    cells: dict[tuple[str, str], list[FlowRow]] = {}
    for row in rows:
        cells.setdefault((row.policy, row.scenario), []).append(row)
    aggregates = {key: aggregate_cell(cell_rows) for key, cell_rows in cells.items()}

    policy_a = policies[0]
    policy_b = policies[1] if len(policies) > 1 else policies[0]
    deltas = compute_flow_deltas(aggregates, policy_a, policy_b)
    classification, rationale = classify_overall(aggregates, deltas, policies)

    return {
        "schema": "robot-influence-flow-slice.issue-3066.v1",
        "issue": 3066,
        "generated_at": datetime.now(UTC).isoformat(),
        "git_head": git_hash,
        "claim_boundary": CLAIM_BOUNDARY,
        "evidence_tier": EVIDENCE_TIER,
        "paper_grade": PAPER_GRADE,
        "non_claims": (
            "No real-world pedestrian-flow, crowd-comfort, fairness, or sim-to-real "
            "claim. Robot-influence signals are diagnostic simulation proxies and are "
            "interpreted separately from nav performance."
        ),
        "config": {
            "policies": list(policies),
            "scenarios": list(scenarios),
            "seeds": list(seeds),
            "horizon": horizon,
            "policy_baseline": policy_a,
            "policy_comparison": policy_b,
            "influence_metrics": list(INFLUENCE_METRIC_KEYS),
        },
        "row_counts": {
            "total": len(rows),
            "usable": sum(1 for r in rows if r.usable),
            "degraded_failclosed": sum(1 for r in rows if not r.usable),
        },
        "rows": [
            {
                "policy": r.policy,
                "scenario": r.scenario,
                "seed": r.seed,
                "row_status": r.row_status,
                "usable": r.usable,
                "degraded_reason": r.degraded_reason,
                "influence": r.influence,
                "context": r.context,
                "nav": r.nav,
                "denominators": r.denominators,
            }
            for r in rows
        ],
        "aggregates": [
            {
                "policy": agg.policy,
                "scenario": agg.scenario,
                "n_rows": agg.n_rows,
                "n_usable": agg.n_usable,
                "n_degraded": agg.n_degraded,
                "influence_mean": agg.influence_mean,
                "influence_std": agg.influence_std,
                "context_mean": agg.context_mean,
                "nav_mean": agg.nav_mean,
                "denom_sum": agg.denom_sum,
            }
            for agg in aggregates.values()
        ],
        "flow_deltas": [
            {
                "scenario": d.scenario,
                "metric": d.metric,
                "policy_a": d.policy_a,
                "policy_b": d.policy_b,
                "mean_a": d.mean_a,
                "mean_b": d.mean_b,
                "delta": d.delta,
                "seed_variance": d.seed_variance,
                "exceeds_variance": d.exceeds_variance,
                "powered": d.powered,
                "note": d.note,
            }
            for d in deltas
        ],
        "classification": classification,
        "rationale": rationale,
    }


def _fmt(value: float) -> str:
    """Format a float for markdown, mapping NaN to ``n/a``.

    Returns:
        A short string representation, ``"n/a"`` for non-finite values.
    """
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        return "n/a"
    return f"{value:.4f}"


def render_markdown(report: dict[str, Any]) -> str:
    """Render a compact human-readable markdown report.

    Returns:
        A markdown string summarizing config, deltas, and classification.
    """
    cfg = report["config"]
    lines: list[str] = []
    lines.append("# Issue #3066 - Robot Influence on Pedestrian Flow (v0 slice)")
    lines.append("")
    lines.append(f"- **Classification**: `{report['classification']}`")
    lines.append(f"- **Rationale**: {report['rationale']}")
    lines.append(
        f"- **Claim boundary**: {report['claim_boundary']} | "
        f"evidence tier: {report['evidence_tier']} | "
        f"paper_grade: {report['paper_grade']}"
    )
    lines.append(f"- **git HEAD**: `{report['git_head']}`")
    lines.append(
        f"- **Policies**: {', '.join(cfg['policies'])} "
        f"(baseline=`{cfg['policy_baseline']}`, compare=`{cfg['policy_comparison']}`)"
    )
    lines.append(f"- **Scenarios**: {', '.join(cfg['scenarios'])}")
    lines.append(f"- **Seeds (same-seed across policies)**: {cfg['seeds']}")
    rc = report["row_counts"]
    lines.append(
        f"- **Rows**: {rc['total']} total, {rc['usable']} usable, "
        f"{rc['degraded_failclosed']} degraded/fail-closed"
    )
    lines.append("")
    lines.append("## Robot-influence flow deltas (policy vs policy)")
    lines.append("")
    lines.append(
        "Signal: near-vs-far pedestrian reduction; larger magnitude = robot perturbs "
        "pedestrians more. Diagnostic proxy only."
    )
    lines.append("")
    lines.append(
        "| scenario | metric | mean_a | mean_b | delta | seed_var | powered | exceeds_var |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for d in report["flow_deltas"]:
        lines.append(
            f"| {d['scenario']} | {d['metric']} | {_fmt(d['mean_a'])} | {_fmt(d['mean_b'])} | "
            f"{_fmt(d['delta'])} | {_fmt(d['seed_variance'])} | {d['powered']} | "
            f"{d['exceeds_variance']} |"
        )
    lines.append("")
    lines.append("## Nav performance (kept SEPARATE from robot influence)")
    lines.append("")
    lines.append("| policy | scenario | n_usable | success | collisions |")
    lines.append("|---|---|---|---|---|")
    for agg in report["aggregates"]:
        lines.append(
            f"| {agg['policy']} | {agg['scenario']} | {agg['n_usable']} | "
            f"{_fmt(agg['nav_mean'].get('success', float('nan')))} | "
            f"{_fmt(agg['nav_mean'].get('total_collision_count', float('nan')))} |"
        )
    lines.append("")
    lines.append("## Non-claims")
    lines.append("")
    lines.append(report["non_claims"])
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for the slice driver.

    Returns:
        A configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Bounded same-seed robot-influence-on-pedestrian-flow slice for issue #3066. "
            "Diagnostic-only / smoke tier; never a real-world claim."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        default=list(DEFAULT_POLICIES),
        help="Robot policies to compare (>=2; first is the baseline). Must be "
        "map-runner-compatible rule-based baselines (no checkpoints/GPU).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help="Seeds reused identically across every policy (same-seed campaign).",
    )
    parser.add_argument("--horizon", type=int, default=240, help="Episode horizon (steps).")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("output/issue_3066_robot_influence"),
        help="Output directory for episodes + report (git-ignored).",
    )
    parser.add_argument(
        "--schema",
        default=DEFAULT_SCHEMA,
        help="Episode schema path for runner validation.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip live execution; emit a 'blocked' report (no campaign run).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Returns:
        Process exit code: 0 on a produced report (any classification), 1 on a
        hard execution error.
    """
    args = build_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    git_hash = git_head(repo_root)
    policies = tuple(args.policies)
    seeds = tuple(args.seeds)
    scenario_names = tuple(s["name"] for s in DEFAULT_SCENARIOS)

    if len(policies) < 2:
        print("ERROR: need >=2 distinct robot policies for a comparison", file=sys.stderr)
        return 1

    if args.dry_run:
        report = {
            "schema": "robot-influence-flow-slice.issue-3066.v1",
            "issue": 3066,
            "generated_at": datetime.now(UTC).isoformat(),
            "git_head": git_hash,
            "claim_boundary": CLAIM_BOUNDARY,
            "evidence_tier": EVIDENCE_TIER,
            "paper_grade": PAPER_GRADE,
            "config": {
                "policies": list(policies),
                "scenarios": list(scenario_names),
                "seeds": list(seeds),
                "horizon": args.horizon,
            },
            "row_counts": {"total": 0, "usable": 0, "degraded_failclosed": 0},
            "rows": [],
            "aggregates": [],
            "flow_deltas": [],
            "classification": "blocked",
            "rationale": "dry-run: campaign not executed",
            "non_claims": "Dry run; no measurements taken.",
        }
    else:
        records = run_campaign(
            policies=policies,
            scenarios=DEFAULT_SCENARIOS,
            seeds=seeds,
            repo_root=repo_root,
            out_dir=out_dir,
            horizon=args.horizon,
            schema_path=args.schema,
        )
        report = build_report(
            records=records,
            policies=policies,
            scenarios=scenario_names,
            seeds=seeds,
            git_hash=git_hash,
            horizon=args.horizon,
        )

    json_path = out_dir / "robot_influence_flow_report.json"
    md_path = out_dir / "robot_influence_flow_report.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(
        render_markdown(report)
        if report["rows"] or report["flow_deltas"]
        else f"# Issue #3066 (blocked)\n\n{report['rationale']}\n",
        encoding="utf-8",
    )

    print(f"classification={report['classification']}")
    print(f"rationale={report['rationale']}")
    print(f"json={json_path}")
    print(f"markdown={md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
