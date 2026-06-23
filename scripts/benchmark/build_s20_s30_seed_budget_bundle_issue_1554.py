#!/usr/bin/env python3
"""Build the issue #1554 S20/S30 seed-budget uncertainty bundle (or block it).

The repository has durable S10 h500 comparison evidence but no S20/S30 (20/30
seed) comparison rows. This tool reads S20/S30 comparison rows from a canonical
campaign result store, classifies them fail-closed, and -- only when real rows
exist -- computes per-planner-by-seed summaries plus bootstrap/per-seed
uncertainty and a seed-resampling rank-flip analysis.

It REUSES canonical statistics helpers and never reinvents them:

- ``robot_sf.benchmark.seed_variance.build_seed_variability_rows`` /
  ``compute_seed_variance`` for per-seed grouping and per-metric variance.
- ``robot_sf.benchmark.snqi.bootstrap.bootstrap_stability`` for deterministic
  stratified bootstrap ranking stability (SNQI surface).
- ``robot_sf.benchmark.rank_metrics`` (``kendall_tau``, ``rank_order``) for the
  seed-resampling rank-flip analysis on direct outcome metrics.

Honesty contract: if the S20/S30 rows are absent or insufficient, this tool
emits a ``blocked_until_run`` status naming the missing seed tier instead of a
fabricated bundle. Fallback / degraded / failed / unavailable / partial /
``not_available`` rows are never counted as success.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from robot_sf.benchmark.rank_metrics import kendall_tau, rank_order
from robot_sf.benchmark.seed_variance import (
    build_seed_variability_rows,
    compute_seed_variance,
)
from robot_sf.benchmark.snqi.bootstrap import bootstrap_stability

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]

# Default canonical store location. The repository ships no durable S20/S30
# store, so this default normally resolves to a missing path and the tool blocks.
DEFAULT_STORE = "output/campaign_result_store/s20_s30_h500_social_navigation"

# Seed tiers this bundle expects. A store must reach at least the S20 tier on the
# primary metric surface to produce a real (non-blocked) bundle.
S20_MIN_SEEDS = 20
S30_MIN_SEEDS = 30

# Primary direct-outcome metric surface (artifact-equivalent names accepted).
PRIMARY_METRICS = (
    "success",
    "collisions",
    "near_misses",
    "time_to_goal_norm",
)
# Secondary descriptive-only metrics (no per-seed uncertainty claim by default).
DESCRIPTIVE_METRICS = (
    "min_clearance",
    "min_distance",
    "mean_clearance",
    "raw_timeout_or_unfinished_rate",
    "low_progress_rate",
)

# Fail-closed row statuses: never counted as benchmark-success evidence.
# Mirrors campaign_result_store / issue #691 fallback policy.
VALID_ROW_STATUSES = frozenset({"native", "adapter"})
FAIL_CLOSED_STATUSES = frozenset(
    {
        "fallback",
        "degraded",
        "unavailable",
        "failed",
        "partial",
        "not_available",
        "diagnostic_only",
    }
)


@dataclass(frozen=True, slots=True)
class StoreRows:
    """Loaded episode rows plus the source description."""

    rows: list[dict[str, Any]]
    source: str
    source_kind: str


def _git_head() -> str:
    """Return the short git HEAD for provenance, or ``unknown``."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _load_rows(rows_path: Path) -> StoreRows:
    """Load episode rows from a JSON list, a result-store dir, or a parquet file.

    Returns:
        Loaded rows plus their source description. An empty row list is a valid
        outcome and drives the ``blocked_until_run`` path downstream.
    """
    if not rows_path.exists():
        return StoreRows(rows=[], source=str(rows_path), source_kind="missing")

    if rows_path.is_dir():
        parquet = rows_path / "episodes.parquet"
        if parquet.is_file():
            return _load_parquet(parquet)
        # Allow a plain JSON rows file inside the store directory for fixtures.
        json_rows = rows_path / "episode_rows.json"
        if json_rows.is_file():
            return _load_json(json_rows)
        return StoreRows(rows=[], source=str(rows_path), source_kind="empty_store")

    if rows_path.suffix == ".parquet":
        return _load_parquet(rows_path)
    return _load_json(rows_path)


def _load_json(path: Path) -> StoreRows:
    """Load episode rows from a JSON file (list or ``{"rows": [...]}``)."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows", payload) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError(f"{path} must contain a JSON list of episode rows")
    return StoreRows(rows=list(rows), source=str(path), source_kind="json")


def _load_parquet(path: Path) -> StoreRows:
    """Load episode rows from a canonical result-store parquet file."""
    from scripts.tools.campaign_result_store import read_parquet_frame

    frame = read_parquet_frame(path)
    rows = frame.to_dict(orient="records")
    return StoreRows(rows=rows, source=str(path), source_kind="parquet")


def _row_status(row: dict[str, Any]) -> str:
    """Return the row's fail-closed status label (defaults to ``unavailable``)."""
    status = row.get("row_status")
    if status is None or (isinstance(status, str) and not status.strip()):
        return "unavailable"
    return str(status).strip()


def _planner_key(row: dict[str, Any]) -> str:
    """Return a planner identifier for grouping."""
    return str(row.get("planner") or row.get("planner_key") or row.get("algo") or "unknown")


def _metric_value(row: dict[str, Any], metric: str) -> float | None:
    """Read a metric from a flat row or a ``metrics``-nested record.

    The canonical seed-variance helpers flatten episode records whose outcome
    metrics live under ``metrics.*``; thin result-store rows may carry the same
    metric at the top level. Accept both.

    Returns:
        The metric value as-is (caller coerces), or ``None`` when absent.
    """
    if metric in row and row[metric] is not None:
        return row[metric]
    metrics = row.get("metrics")
    if isinstance(metrics, dict) and metrics.get(metric) is not None:
        return metrics.get(metric)
    return None


def classify_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Partition rows into valid vs fail-closed, with exact per-planner reasons.

    Returns:
        A classification payload listing valid rows, fail-closed rows, the seeds
        observed per planner, and explicit fail-closed reasons.
    """
    valid: list[dict[str, Any]] = []
    fail_closed: list[dict[str, Any]] = []
    reasons: dict[str, dict[str, Any]] = {}
    seeds_by_planner: dict[str, set[int]] = {}
    status_counts: dict[str, int] = {}

    for row in rows:
        status = _row_status(row)
        status_counts[status] = status_counts.get(status, 0) + 1
        planner = _planner_key(row)
        if status in VALID_ROW_STATUSES:
            valid.append(row)
            try:
                seeds_by_planner.setdefault(planner, set()).add(int(row.get("seed", -1)))
            except (TypeError, ValueError):
                pass
        else:
            fail_closed.append(row)
            entry = reasons.setdefault(
                planner,
                {"planner": planner, "statuses": {}, "count": 0},
            )
            entry["statuses"][status] = entry["statuses"].get(status, 0) + 1
            entry["count"] += 1

    return {
        "valid_rows": valid,
        "fail_closed_rows": fail_closed,
        "fail_closed_reasons": sorted(reasons.values(), key=lambda item: item["planner"]),
        "seeds_by_planner": {k: sorted(v) for k, v in sorted(seeds_by_planner.items())},
        "row_status_counts": dict(sorted(status_counts.items())),
    }


def _seed_tier(seed_count: int) -> str:
    """Classify a per-planner seed count into a budget tier."""
    if seed_count >= S30_MIN_SEEDS:
        return "s30"
    if seed_count >= S20_MIN_SEEDS:
        return "s20"
    if seed_count >= 10:
        return "s10"
    return "below_s10"


def _achieved_tier(seeds_by_planner: dict[str, list[int]]) -> tuple[str, int]:
    """Return the minimum seed tier achieved across all valid planner rows.

    The bundle can only be as strong as its weakest planner row, so the
    governing tier is the minimum per-planner seed count.

    Returns:
        The governing tier label and the minimum seed count observed.
    """
    if not seeds_by_planner:
        return "no_valid_rows", 0
    min_seeds = min(len(seeds) for seeds in seeds_by_planner.values())
    return _seed_tier(min_seeds), min_seeds


def _bootstrap_settings() -> dict[str, Any]:
    """Return the deterministic bootstrap settings used for seed-mean CIs."""
    return {
        "method": "bootstrap_mean_over_seed_means",
        "confidence": 0.95,
        "bootstrap_samples": 1000,
        "bootstrap_seed": 123,
    }


def _planner_seed_means(
    valid_rows: Sequence[dict[str, Any]], *, metric: str
) -> dict[str, dict[int, float]]:
    """Build planner -> {seed: mean metric value} from valid rows.

    Returns:
        Per-planner mapping of seed to the mean finite metric value.
    """
    per_planner_seed_vals: dict[str, dict[int, list[float]]] = {}
    for row in valid_rows:
        value = _metric_value(row, metric)
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(numeric):
            continue
        planner = _planner_key(row)
        try:
            seed_val = int(row.get("seed", -1))
        except (TypeError, ValueError):
            continue
        per_planner_seed_vals.setdefault(planner, {}).setdefault(seed_val, []).append(numeric)

    return {
        planner: {s: float(np.mean(vals)) for s, vals in seed_vals.items()}
        for planner, seed_vals in per_planner_seed_vals.items()
    }


def _seed_resampling_rank_flip(
    valid_rows: Sequence[dict[str, Any]],
    *,
    metric: str,
    higher_is_better: bool,
    samples: int = 200,
    seed: int = 1554,
) -> dict[str, Any]:
    """Detect whether planner rankings flip under seed resampling for a metric.

    Reuses ``rank_metrics.rank_order`` / ``kendall_tau``: the baseline ordering
    uses all observed seeds; each resample draws seeds with replacement, ranks
    planners by their resampled per-seed mean, and measures Kendall tau against
    baseline. Any tau below 1.0 indicates at least one rank flip.

    Returns:
        A payload with the baseline ordering, flip fraction, and tau summary.
    """
    planner_seed_means = _planner_seed_means(valid_rows, metric=metric)
    planners = sorted(p for p, sm in planner_seed_means.items() if sm)
    if len(planners) < 2:
        return {
            "metric": metric,
            "status": "insufficient_planners",
            "planner_count": len(planners),
        }

    # Common seed universe used for resampling.
    seed_universe = sorted({s for sm in planner_seed_means.values() for s in sm})
    if len(seed_universe) < 2:
        return {"metric": metric, "status": "insufficient_seeds", "seed_count": len(seed_universe)}

    baseline_means = {
        planner: float(np.mean(list(sm.values()))) for planner, sm in planner_seed_means.items()
    }
    baseline_order = rank_order(baseline_means, higher_is_better=higher_is_better)

    rng = np.random.default_rng(seed)
    taus: list[float] = []
    flips = 0
    for _ in range(samples):
        drawn = rng.integers(0, len(seed_universe), size=len(seed_universe))
        sampled_seeds = [seed_universe[int(i)] for i in drawn]
        sample_means: dict[str, float] = {}
        for planner, sm in planner_seed_means.items():
            vals = [sm[s] for s in sampled_seeds if s in sm]
            if vals:
                sample_means[planner] = float(np.mean(vals))
        if len(sample_means) < 2:
            continue
        sample_order = rank_order(sample_means, higher_is_better=higher_is_better)
        tau = kendall_tau(baseline_order, sample_order, degenerate=None)
        if tau is None:
            continue
        taus.append(float(tau))
        if float(tau) < 1.0:
            flips += 1

    if not taus:
        return {"metric": metric, "status": "no_valid_resamples"}

    return {
        "metric": metric,
        "status": "ok",
        "higher_is_better": higher_is_better,
        "baseline_order": baseline_order,
        "samples": len(taus),
        "rank_flip_observed": flips > 0,
        "rank_flip_fraction": flips / len(taus),
        "kendall_tau_mean": float(np.mean(taus)),
        "kendall_tau_min": float(np.min(taus)),
        "method": "seed_resampling_kendall_tau",
        "seed_resample_seed": seed,
    }


_METRIC_DIRECTION = {
    "success": True,
    "collisions": False,
    "near_misses": False,
    "time_to_goal_norm": False,
}


def build_bundle(store: StoreRows, *, git_head: str) -> dict[str, Any]:
    """Build the S20/S30 bundle payload, or a ``blocked_until_run`` status.

    Returns:
        A JSON-compatible bundle. ``status`` is ``ok`` only when real S20+ valid
        rows exist for at least two planners; otherwise it is ``blocked_until_run``.
    """
    classification = classify_rows(store.rows)
    valid_rows = classification["valid_rows"]
    seeds_by_planner = classification["seeds_by_planner"]
    tier, min_seeds = _achieved_tier(seeds_by_planner)

    base: dict[str, Any] = {
        "schema_version": "s20-s30-seed-budget-bundle.v1",
        "issue": 1554,
        "surface": "h500_social_navigation",
        "git_head": git_head,
        "source": store.source,
        "source_kind": store.source_kind,
        "fail_closed_policy": {
            "valid_row_statuses": sorted(VALID_ROW_STATUSES),
            "fail_closed_statuses": sorted(FAIL_CLOSED_STATUSES),
        },
        "row_status_counts": classification["row_status_counts"],
        "fail_closed_reasons": classification["fail_closed_reasons"],
        "seeds_by_planner": seeds_by_planner,
        "achieved_seed_tier": tier,
        "min_seeds_per_planner": min_seeds,
        "primary_metrics": list(PRIMARY_METRICS),
        "descriptive_only_metrics": list(DESCRIPTIVE_METRICS),
        "methodology_reference": "docs/context/issue_1545_power_aware_seed_budget_planning.md",
        "reused_canonical_functions": [
            "robot_sf.benchmark.seed_variance.build_seed_variability_rows",
            "robot_sf.benchmark.seed_variance.compute_seed_variance",
            "robot_sf.benchmark.snqi.bootstrap.bootstrap_stability",
            "robot_sf.benchmark.rank_metrics.rank_order",
            "robot_sf.benchmark.rank_metrics.kendall_tau",
        ],
    }

    # Fail-closed / blocked path: no rows, no valid rows, fewer than two planners,
    # or the governing per-planner seed budget is below the S20 paper tier.
    distinct_planners = len(seeds_by_planner)
    if not store.rows or not valid_rows or distinct_planners < 2 or min_seeds < S20_MIN_SEEDS:
        missing_tier = "s20_and_s30" if min_seeds < S20_MIN_SEEDS else "s30"
        if not store.rows:
            reason = (
                f"no S20/S30 comparison rows found at source {store.source!r}; "
                "the durable repository evidence is S10, not S20/S30"
            )
        elif not valid_rows:
            reason = "all rows classified fail-closed; no native/adapter rows to summarize"
        elif distinct_planners < 2:
            reason = f"only {distinct_planners} valid planner row(s); need >=2 to compare"
        else:
            reason = (
                f"governing per-planner seed budget is {min_seeds} (tier {tier}), "
                f"below the S20 paper-facing tier of {S20_MIN_SEEDS}"
            )
        base.update(
            {
                "status": "blocked_until_run",
                "missing_seed_tier": missing_tier,
                "blocked_reason": reason,
                "claim_boundary": (
                    "No S20/S30 comparison claim exists. This bundle is blocked until the "
                    "SLURM S20/S30 h500 social-navigation campaign produces valid rows."
                ),
            }
        )
        return base

    # Real-bundle path. Compute per-planner-by-seed summaries with bootstrap CIs.
    confidence_settings = _bootstrap_settings()
    seed_variability_rows = build_seed_variability_rows(
        valid_rows,
        metrics=PRIMARY_METRICS,
        campaign_id="issue_1554_s20_s30_h500_social_navigation",
        config_hash="from_result_store",
        git_hash=git_head,
        seed_policy={"mode": "seed-set", "seed_set": f"paper_eval_{tier}"},
        confidence_settings=confidence_settings,
    )
    scenario_variance = compute_seed_variance(
        valid_rows,
        group_by="planner",
        fallback_group_by="planner",
        metrics=PRIMARY_METRICS,
    )

    # Seed-resampling rank-flip analysis on each primary metric.
    rank_flip = {
        metric: _seed_resampling_rank_flip(
            valid_rows,
            metric=metric,
            higher_is_better=_METRIC_DIRECTION.get(metric, False),
        )
        for metric in PRIMARY_METRICS
        if metric in _METRIC_DIRECTION
    }

    # SNQI ranking stability via the canonical bootstrap helper (only when the
    # rows carry per-episode metrics.snqi; otherwise mark not-available).
    snqi_stability = _maybe_bootstrap_stability(valid_rows, git_head=git_head)

    any_flip = any(
        isinstance(entry, dict) and entry.get("rank_flip_observed") for entry in rank_flip.values()
    )
    base.update(
        {
            "status": "ok",
            "claim_boundary": (
                "Descriptive S20/S30 comparison summary with per-seed bootstrap uncertainty. "
                "Treat as effect-size-planned per issue #1545; not a significance claim."
            ),
            "confidence_settings": confidence_settings,
            "per_planner_seed_variability": seed_variability_rows,
            "per_planner_metric_variance": scenario_variance,
            "seed_resampling_rank_flip": rank_flip,
            "snqi_ranking_stability": snqi_stability,
            "rank_conclusion_flips_under_resampling": any_flip,
        }
    )
    return base


def _maybe_bootstrap_stability(
    valid_rows: Sequence[dict[str, Any]], *, git_head: str
) -> dict[str, Any]:
    """Run canonical SNQI bootstrap stability when per-episode SNQI is present.

    Returns:
        The canonical ``bootstrap_stability`` payload, or a not-available marker.
    """
    episodes: list[dict[str, Any]] = []
    for row in valid_rows:
        snqi = _nested_snqi(row)
        if snqi is None:
            continue
        episodes.append({"algo": _planner_key(row), "metrics": {"snqi": snqi}})
    distinct = {ep["algo"] for ep in episodes}
    if len(distinct) < 2:
        return {
            "status": "not_available",
            "reason": "fewer than two planner groups carry finite metrics.snqi",
        }
    try:
        return bootstrap_stability(
            episodes,
            weights={"snqi": 1.0},
            rng=np.random.default_rng(123),
            samples=200,
            group_key="algo",
        )
    except ValueError as exc:
        return {"status": "not_available", "reason": str(exc)}


def _nested_snqi(row: dict[str, Any]) -> float | None:
    """Read a finite SNQI value from a flat or ``metrics``-nested row."""
    candidates: Iterable[Any] = (
        row.get("snqi"),
        (row.get("metrics") or {}).get("snqi") if isinstance(row.get("metrics"), dict) else None,
    )
    for value in candidates:
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric):
            return numeric
    return None


def render_markdown(bundle: dict[str, Any]) -> str:
    """Render a compact human-readable bundle summary.

    Returns:
        Markdown text describing the bundle status, tier, and key conclusions.
    """
    lines: list[str] = []
    lines.append("# Issue #1554 S20/S30 Seed-Budget Bundle")
    lines.append("")
    lines.append(f"- Status: `{bundle.get('status')}`")
    lines.append(f"- Surface: `{bundle.get('surface')}`")
    lines.append(f"- Git HEAD: `{bundle.get('git_head')}`")
    lines.append(f"- Source: `{bundle.get('source')}` (`{bundle.get('source_kind')}`)")
    lines.append(f"- Achieved seed tier: `{bundle.get('achieved_seed_tier')}`")
    lines.append(f"- Min seeds per planner: `{bundle.get('min_seeds_per_planner')}`")
    lines.append(f"- Methodology reference: `{bundle.get('methodology_reference')}`")
    lines.append("")
    lines.append("## Claim boundary")
    lines.append("")
    lines.append(str(bundle.get("claim_boundary", "")))
    lines.append("")
    if bundle.get("status") == "blocked_until_run":
        lines.append("## Blocked")
        lines.append("")
        lines.append(f"- Missing seed tier: `{bundle.get('missing_seed_tier')}`")
        lines.append(f"- Reason: {bundle.get('blocked_reason')}")
        lines.append("")
        lines.append(
            "Run the SLURM S20/S30 campaign from "
            "`configs/benchmarks/s20_s30_seed_budget_issue_1554_launch_packet.yaml` "
            "and re-run this tool against the resulting result store."
        )
    else:
        lines.append("## Conclusions")
        lines.append("")
        lines.append(
            f"- Rank conclusion flips under seed resampling: "
            f"`{bundle.get('rank_conclusion_flips_under_resampling')}`"
        )
        flip = bundle.get("seed_resampling_rank_flip", {})
        for metric, entry in flip.items():
            if isinstance(entry, dict) and entry.get("status") == "ok":
                lines.append(
                    f"  - `{metric}`: flip_fraction="
                    f"`{entry.get('rank_flip_fraction'):.3f}`, "
                    f"tau_min=`{entry.get('kendall_tau_min'):.3f}`"
                )
    lines.append("")
    lines.append("## Reused canonical functions")
    lines.append("")
    for fn in bundle.get("reused_canonical_functions", []):
        lines.append(f"- `{fn}`")
    lines.append("")
    if bundle.get("fail_closed_reasons"):
        lines.append("## Fail-closed rows")
        lines.append("")
        for reason in bundle["fail_closed_reasons"]:
            lines.append(
                f"- `{reason['planner']}`: {reason['count']} row(s) statuses={reason['statuses']}"
            )
        lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rows",
        type=Path,
        default=None,
        help="Path to a result-store dir, episodes.parquet, or a JSON rows file.",
    )
    parser.add_argument(
        "--campaign",
        type=Path,
        default=None,
        help="Alias for --rows (canonical campaign result-store path).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/issue_1554_s20_s30"),
        help="Directory for bundle.json and BUNDLE.md.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Build the bundle (or block it) and write JSON + markdown."""
    args = parse_args(argv)
    rows_path = args.rows or args.campaign or (REPO_ROOT / DEFAULT_STORE)
    store = _load_rows(Path(rows_path))
    bundle = build_bundle(store, git_head=_git_head())

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "bundle.json").write_text(
        json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "BUNDLE.md").write_text(render_markdown(bundle), encoding="utf-8")

    print(f"status={bundle['status']} tier={bundle.get('achieved_seed_tier')}")
    print(f"wrote {output_dir / 'bundle.json'}")
    # blocked_until_run is an honest, expected outcome -> exit 0.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
