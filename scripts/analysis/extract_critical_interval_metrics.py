#!/usr/bin/env python3
"""Offline CLI: extract critical-interval metrics from episode JSONL traces.

Usage::

    uv run python scripts/analysis/extract_critical_interval_metrics.py \
        --episodes-jsonl <path> \
        [--config configs/benchmarks/critical_intervals_default.yaml] \
        [--output-json output/critical_intervals/report.json]

This is an experimental, opt-in diagnostic tool.  It reads a JSONL file of
episode records, extracts critical intervals around safety-relevant events,
and writes a structured JSON report with whole-run and per-interval metrics.

Exit codes
----------
0 : success
1 : fatal error (bad arguments, corrupt trace, etc.)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from loguru import logger

from robot_sf.benchmark.critical_intervals import (
    extract_critical_intervals,
    load_config,
    report_to_dict,
    summarize_interval_metrics,
)


def _trace_from_record(record: dict) -> dict:
    """Convert an episode JSONL record to the trace format expected by the API.

    Handles both native EpisodeData-style keys and common JSONL exports.
    """

    trace = dict(record)

    # Ensure numeric arrays are lists (not numpy arrays)
    for key in ("robot_pos", "peds_pos", "robot_vel", "ped_vel"):
        if key in trace:
            arr = trace[key]
            if hasattr(arr, "tolist"):
                trace[key] = arr.tolist()

    return trace


def main(argv: list[str] | None = None) -> int:  # noqa: C901
    """Run the CLI entry point.

    Returns
    -------
    Exit code (0 for success).
    """
    parser = argparse.ArgumentParser(
        description="Extract critical-interval metrics from episode traces.",
    )
    parser.add_argument(
        "--episodes-jsonl",
        required=True,
        help="Path to the episode JSONL file.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to the critical-intervals YAML config (default: built-in minimal).",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Path for the output JSON report (default: stdout).",
    )
    args = parser.parse_args(argv)

    # Load config
    try:
        cfg = load_config(args.config)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Config error: {}", exc)
        return 1

    # Build default config if none provided
    if not cfg.get("critical_intervals"):
        cfg = load_config(
            config_dict={
                "schema_version": "critical-intervals.v1",
                "critical_intervals": {
                    "closest_approach": {
                        "enabled": True,
                        "before_s": 2.0,
                        "after_s": 1.0,
                    },
                    "ttc_threshold_crossing": {
                        "enabled": True,
                        "threshold_s": 1.5,
                        "before_s": 1.0,
                        "after_s": 2.0,
                    },
                },
            }
        )

    # Read episodes
    jsonl_path = Path(args.episodes_jsonl)
    if not jsonl_path.exists():
        logger.error("Episode file not found: {}", jsonl_path)
        return 1

    episodes = []
    with open(jsonl_path) as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                episodes.append(record)
            except json.JSONDecodeError as exc:
                logger.warning(
                    "Skipping malformed line {} in {}: {}",
                    line_no,
                    jsonl_path,
                    exc,
                )

    if not episodes:
        logger.error("No valid episodes found in {}", jsonl_path)
        return 1

    # Process each episode
    all_results = []
    for ep in episodes:
        try:
            trace = _trace_from_record(ep)
            intervals = extract_critical_intervals(trace, cfg)
            report = summarize_interval_metrics(trace, intervals)
            result = report_to_dict(report)
            if "episode_id" in ep:
                result["episode_id"] = ep["episode_id"]
            if "scenario_id" in ep:
                result["scenario_id"] = ep["scenario_id"]
            all_results.append(result)
        except (KeyError, ValueError, TypeError) as exc:
            ep_id = ep.get("episode_id", "unknown")
            logger.warning("Failed to process episode {}: {}", ep_id, exc)

    output = {
        "schema_version": "critical-intervals.v1",
        "episodes": all_results,
        "n_episodes_processed": len(all_results),
    }

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(output, f, indent=2, default=str)
        logger.info("Wrote report to {}", out)
    else:
        print(json.dumps(output, indent=2, default=str))

    return 0


if __name__ == "__main__":
    sys.exit(main())
