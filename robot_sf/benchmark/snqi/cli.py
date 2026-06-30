"""
SNQI CLI Tools

This module provides command-line interfaces for SNQI weight management:
- Weight recomputation from baseline statistics
- SNQI component ablation analysis
"""

import argparse
import json
import statistics
import sys
from collections.abc import Iterable
from pathlib import Path

from loguru import logger

from robot_sf.benchmark.snqi.compute import (
    WEIGHT_NAMES,
    compute_snqi_ablation,
    recompute_snqi_weights,
)
from robot_sf.benchmark.snqi.types import SNQIWeights


def _log_cli_failure(stage: str, message: str, **context: object) -> None:
    """Log a structured SNQI CLI failure before returning a non-zero status."""
    logger.bind(event="snqi_cli_failed", stage=stage, **context).error(message)


def cmd_recompute_weights(args: argparse.Namespace) -> int:
    """Implement weight recompute CLI command.

    Returns:
        Exit code: 0 on success, 1 on failure.
    """
    try:
        # Load baseline statistics
        baseline_stats_path = Path(args.baseline_stats)
        if not baseline_stats_path.exists():
            _log_cli_failure(
                "load_baseline_stats",
                "SNQI CLI baseline statistics file is missing.",
                path=str(baseline_stats_path),
            )
            return 1

        with baseline_stats_path.open("r", encoding="utf-8") as f:
            baseline_stats = json.load(f)

        method = args.method

        if method not in {"canonical", "balanced", "optimized"}:
            _log_cli_failure("validate_method", "SNQI CLI recompute method is unsupported.")
            return 2

        # Recompute SNQI weights using baseline statistics
        weights = recompute_snqi_weights(
            baseline_stats=baseline_stats,
            method=method,
            seed=args.seed,
        )

        # Save the recomputed weights
        output_path = Path(args.out)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        weights.save(output_path)

        # Print summary of computed weights
        for _component, _weight in weights.weights.items():
            pass

        return 0

    except Exception:
        logger.bind(event="snqi_cli_failed", stage="recompute_weights").exception(
            "SNQI CLI failed while recomputing weights."
        )
        return 1


def _load_episodes_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file of episode records, skipping malformed lines.

    Returns:
        List of parsed episode dicts.
    """
    episodes: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                episodes.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # skip malformed
    return episodes


def _extract_metric_values(episodes: Iterable[dict], metric: str) -> list[float]:
    """Extract a numeric metric across episode records.

    Returns:
        List of numeric metric values.
    """
    vals = []
    for ep in episodes:
        metrics = ep.get("metrics", {})
        v = metrics.get(metric)
        if v is not None:
            try:
                vals.append(float(v))
            except (TypeError, ValueError):  # pragma: no cover - defensive
                continue
    return vals


def _compute_baseline_stats(episodes: list[dict]) -> dict[str, dict[str, float]]:
    """Compute median & p95 for metrics required in SNQI normalization.

    Metrics chosen align with those referenced in compute_snqi(): collisions, near_misses,
    force_exceed_events, jerk_mean. Missing metrics default to med=0, p95=1 for neutral scaling.

    Returns:
        Dictionary mapping metric names to {'med': float, 'p95': float} statistics.
    """
    metric_names = [
        "collisions",
        "near_misses",
        "force_exceed_events",
        "jerk_mean",
    ]
    stats: dict[str, dict[str, float]] = {}
    for name in metric_names:
        values = _extract_metric_values(episodes, name)
        if not values:
            stats[name] = {"med": 0.0, "p95": 1.0}
            continue
        values_sorted = sorted(values)
        med = statistics.median(values_sorted)
        # p95 index calculation
        idx = min(len(values_sorted) - 1, max(0, round(0.95 * (len(values_sorted) - 1))))
        p95 = float(values_sorted[idx])
        if abs(p95 - med) < 1e-12:  # ensure non-zero span for normalization downstream
            p95 = med + 1.0
        stats[name] = {"med": float(med), "p95": float(p95)}
    return stats


def cmd_ablation_analysis(args: argparse.Namespace) -> int:
    """Implement SNQI ablation CLI command.

    This implementation loads episodes from JSONL, derives baseline statistics on-the-fly
    (unless future extension supplies an external stats file), and computes the impact on
    mean SNQI when zeroing each component weight individually.

    Returns:
        Exit code: 0 on success, 1 on failure.
    """
    try:
        episodes_path = Path(args.episodes)
        if not episodes_path.exists():
            _log_cli_failure(
                "load_episodes",
                "SNQI CLI episodes file is missing.",
                path=str(episodes_path),
            )
            return 1

        episodes = _load_episodes_jsonl(episodes_path)
        if not episodes:
            _log_cli_failure(
                "load_episodes",
                "SNQI CLI episodes file produced no valid records.",
                path=str(episodes_path),
            )
            return 1

        # Determine weights
        weight_map: dict[str, float]
        if args.weights:
            weights_path = Path(args.weights)
            if not weights_path.exists():
                _log_cli_failure(
                    "load_weights",
                    "SNQI CLI weights file is missing.",
                    path=str(weights_path),
                )
                return 1
            weight_obj = SNQIWeights.load(weights_path)
            weight_map = dict(weight_obj.weights)
        else:
            # Fallback canonical defaults (mirrors compute.recompute_snqi_weights canonical branch)
            weight_map = {
                "w_success": 1.0,
                "w_time": 0.8,
                "w_collisions": 2.0,
                "w_near": 1.0,
                "w_comfort": 0.5,
                "w_force_exceed": 1.5,
                "w_jerk": 0.3,
            }

        # Filter components if user provided subset
        target_components = args.components if args.components else list(WEIGHT_NAMES)

        baseline_stats = _compute_baseline_stats(episodes)

        # Run ablation
        impacts = compute_snqi_ablation(
            episodes_data=episodes,
            weights=weight_map,
            baseline_stats=baseline_stats,
            components=target_components,
        )

        # Compose output structure (extensible for future ranking deltas)
        result_payload = {
            "impacts": impacts,
            "baseline_stats": baseline_stats,
            "weights_used": weight_map,
            "components": target_components,
            "episode_count": len(episodes),
            "seed": args.seed,
        }

        output_path = Path(args.summary_out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(result_payload, f, indent=2, sort_keys=True)

        for comp in target_components:
            impacts.get(comp, 0.0)

        return 0
    except Exception:
        logger.bind(event="snqi_cli_failed", stage="ablation_analysis").exception(
            "SNQI CLI failed while computing ablation analysis."
        )
        return 1


def cmd_inventory_weights(args: argparse.Namespace) -> int:
    """Implement the SNQI weight-set provenance inventory CLI command.

    Read-only diagnostic: discovers all known SNQI weight sets (code default +
    shipped JSON), reports their dominant term / scale, and lists provenance
    conflicts. With ``--fail-on-conflict`` (default), a blocking conflict makes
    the command exit non-zero (fail-closed) without altering any scoring.

    Returns:
        Exit code: 0 when no blocking conflict (or ``--no-fail-on-conflict``);
        ``EXIT_VALIDATION_ERROR`` (2) when a blocking conflict is detected;
        ``EXIT_RUNTIME_ERROR`` (3) on unexpected failure.
    """
    # Imported lazily so the rest of the SNQI CLI does not depend on the
    # inventory module (and its repo-root resolution) at import time.
    from robot_sf.benchmark.snqi.exit_codes import (  # noqa: PLC0415
        EXIT_RUNTIME_ERROR,
        EXIT_SUCCESS,
        EXIT_VALIDATION_ERROR,
    )
    from robot_sf.benchmark.snqi.weights_inventory import (  # noqa: PLC0415
        build_inventory_report,
    )

    try:
        report = build_inventory_report()
        payload = report.to_dict()

        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print("SNQI weight-set inventory")
            print("=" * 60)
            for rec in report.records:
                if rec.available:
                    print(
                        f"- {rec.name:18s} dominant={rec.dominant_term:14s} "
                        f"scale={rec.scale_class:18s} "
                        f"canonical={'yes' if rec.declares_canonical else 'no'} "
                        f"sha256={(rec.content_sha256 or 'unknown')[:12]} "
                        f"({rec.relpath or 'code default'})"
                    )
                else:
                    print(f"- {rec.name:18s} UNAVAILABLE ({rec.relpath}): {rec.load_error}")
            print("-" * 60)
            if report.conflicts:
                print(f"Conflicts ({len(report.conflicts)}):")
                for c in report.conflicts:
                    print(f"  [{c.severity}] {c.kind}: {c.detail}")
            else:
                print("No provenance conflicts detected.")

        if args.fail_on_conflict and report.has_blocking_conflict:
            _log_cli_failure(
                "weights_inventory",
                "SNQI weight-set provenance preflight failed (fail-closed).",
                blocking=[c.kind for c in report.conflicts if c.severity == "error"],
            )
            return EXIT_VALIDATION_ERROR
        return EXIT_SUCCESS
    except Exception:
        logger.bind(event="snqi_cli_failed", stage="weights_inventory").exception(
            "SNQI CLI failed while building the weight-set inventory."
        )
        return EXIT_RUNTIME_ERROR


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for SNQI CLI tools.

    Returns:
        Configured ArgumentParser with subcommands for SNQI operations.
    """
    parser = argparse.ArgumentParser(
        prog="robot_sf_snqi",
        description="SNQI weight management and analysis tools",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Recompute weights subcommand
    recompute_parser = subparsers.add_parser(
        "recompute",
        help="Recompute SNQI weights from baseline statistics",
    )
    recompute_parser.add_argument(
        "--baseline-stats",
        required=True,
        help="Path to baseline statistics JSON file",
    )
    recompute_parser.add_argument(
        "--out",
        required=True,
        help="Output path for recomputed weights JSON file",
    )
    recompute_parser.add_argument(
        "--method",
        default="canonical",
        choices=[
            "canonical",
            "balanced",
            "optimized",
        ],
        help="Weight computation method: canonical|balanced|optimized. Default: canonical",
    )
    recompute_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible weight computation (default: 42)",
    )

    # Ablation analysis subcommand
    ablation_parser = subparsers.add_parser(
        "ablation",
        help="Perform SNQI component ablation analysis",
    )
    ablation_parser.add_argument("--episodes", required=True, help="Path to episodes JSONL file")
    ablation_parser.add_argument(
        "--summary-out",
        required=True,
        help="Output path for ablation results JSON",
    )
    ablation_parser.add_argument(
        "--weights",
        help="Path to SNQI weights JSON file (optional, uses defaults if not provided)",
    )
    ablation_parser.add_argument(
        "--components",
        nargs="*",
        help="Specific SNQI components to analyze (default: all components)",
    )
    ablation_parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for reproducible analysis (default: 123)",
    )

    # Weight-set provenance inventory subcommand (read-only diagnostic)
    inventory_parser = subparsers.add_parser(
        "inventory",
        aliases=["weights-inventory"],
        help="Inventory SNQI weight sets and report provenance conflicts (read-only)",
    )
    inventory_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the inventory report as JSON instead of a text summary",
    )
    inventory_parser.add_argument(
        "--fail-on-conflict",
        dest="fail_on_conflict",
        action="store_true",
        default=True,
        help="Exit non-zero when a blocking provenance conflict is detected (default)",
    )
    inventory_parser.add_argument(
        "--no-fail-on-conflict",
        dest="fail_on_conflict",
        action="store_false",
        help="Report conflicts but always exit 0 (inspection mode)",
    )

    return parser


def main() -> int:
    """Main entry point for SNQI CLI tools.

    Returns:
        Exit code: 0 on success, 1 on failure or invalid command.
    """
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    if args.command == "recompute":
        return cmd_recompute_weights(args)
    elif args.command == "ablation":
        return cmd_ablation_analysis(args)
    elif args.command in {"inventory", "weights-inventory"}:
        return cmd_inventory_weights(args)
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
