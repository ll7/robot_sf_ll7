"""Demonstrate the stable public API facade with EpisodeRecord persistence.

Usage:
    uv run python examples/quickstart/05_public_api.py
    uv run python examples/quickstart/05_public_api.py --format json --output-dir output/example-public-api

Prerequisites:
    - None

Expected Output:
    - Resolves the canonical quickstart scenario via `robot_sf.load_scenario`.
    - Creates a headless simulation environment via `robot_sf.make_env`.
    - Steps a deterministic short episode with a model-free protocol planner.
    - Saves and reloads the resulting `EpisodeRecord` to verify round-trip integrity.
    - Reports execution metrics, identity, and deterministic digest in friendly or JSON format.

Limitations:
    - Demonstration output is for tutorial and validation purposes only;
      it is not benchmark evidence.
    - Uses a simple constant-velocity protocol planner for reproducibility.

References:
    - docs/public_api.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import robot_sf

DEFAULT_SEED = 42
DEFAULT_HORIZON = 5
DEFAULT_OUTPUT_DIR = Path("output/examples/quickstart_public_api")
BENCHMARK_CAVEAT_NOTE = "Demonstration output only; not benchmark evidence."


class GentleForwardPlanner:
    """A minimal local planner conforming to PlannerProtocol.

    Issues a gentle forward velocity command without requiring machine
    learning models, external weights, or heavy optional dependencies.
    """

    def __init__(self, forward_speed: float = 0.5) -> None:
        """Initialize gentle forward planner with desired forward speed."""
        self.forward_speed = forward_speed

    def step(self, obs: Any) -> dict[str, float]:
        """Return a protocol-compatible velocity command."""
        return {"v": self.forward_speed, "omega": 0.0}

    def reset(self, seed: int | None = None) -> None:
        """Reset planner state for a new episode."""
        pass


def _step_budget(default: int) -> int:
    """Return a smaller rollout budget when running under smoke test harnesses."""
    override = os.environ.get("ROBOT_SF_EXAMPLES_MAX_STEPS")
    if override:
        try:
            return max(1, int(override))
        except ValueError:
            pass
    if os.environ.get("ROBOT_SF_FAST_DEMO", "0") == "1":
        return min(default, 3)
    return default


def compute_record_digest(record: robot_sf.EpisodeRecord) -> str:
    """Compute a deterministic compact digest from an EpisodeRecord payload."""
    payload = json.dumps(record.to_dict(), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def run_quickstart(
    *,
    output_dir: Path,
    seed: int = DEFAULT_SEED,
    horizon: int = DEFAULT_HORIZON,
    output_format: str = "friendly",
) -> dict[str, Any]:
    """Execute the public facade quickstart rollout, save record, and report findings."""
    step_limit = _step_budget(horizon)

    # 1. Load the canonical scenario definition using the top-level facade
    scenario = robot_sf.load_scenario("quickstart_demo")

    # 2. Construct the simulation environment
    env = robot_sf.make_env(scenario=scenario, seed=seed)

    planner = GentleForwardPlanner(forward_speed=0.5)

    try:
        # 3. Execute the episode with lifecycle and seed guarantees
        record = robot_sf.run_episode(
            env,
            planner=planner,
            max_steps=step_limit,
            seed=seed,
        )
    finally:
        # 4. Guarantee environment cleanup on success and failure
        env.close()

    # 5. Persist the EpisodeRecord to the caller-owned output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    save_target = output_dir / f"{record.episode_id}.json"
    written_path = record.save(save_target)

    # 6. Reload and verify record identity, seed, horizon, and metrics
    reloaded_record = robot_sf.EpisodeRecord.load(written_path)
    assert reloaded_record.episode_id == record.episode_id, "Episode ID mismatch on reload"
    assert reloaded_record.seed == record.seed, "Seed mismatch on reload"
    assert reloaded_record.horizon == record.horizon, "Horizon mismatch on reload"
    assert reloaded_record.metrics.values == record.metrics.values, (
        "Metrics values mismatch on reload"
    )

    digest = compute_record_digest(reloaded_record)
    metrics = reloaded_record.metrics.values

    summary: dict[str, Any] = {
        "episode_id": reloaded_record.episode_id,
        "scenario_id": reloaded_record.scenario_id,
        "seed": reloaded_record.seed,
        "horizon": reloaded_record.horizon,
        "steps": metrics.get("steps", float(reloaded_record.horizon or 0)),
        "success": metrics.get("success", 0.0),
        "collision": metrics.get("collision", 0.0),
        "total_reward": round(metrics.get("total_reward", 0.0), 4),
        "duration_s": round(metrics.get("duration_s", 0.0), 4),
        "record_path": str(written_path),
        "record_digest": digest,
        "reloaded_verified": True,
        "note": BENCHMARK_CAVEAT_NOTE,
    }

    if output_format == "json":
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print("=" * 60)
        print("Robot SF Public API Quickstart")
        print("=" * 60)
        print(f"Scenario ID:        {summary['scenario_id']}")
        print(f"Episode ID:         {summary['episode_id']}")
        print(f"Seed:               {summary['seed']}")
        print(f"Horizon:            {summary['horizon']}")
        print(f"Steps Executed:     {summary['steps']}")
        print(f"Success:            {summary['success']}")
        print(f"Collision:          {summary['collision']}")
        print(f"Total Reward:       {summary['total_reward']}")
        print(f"Duration (s):       {summary['duration_s']}")
        print(f"Record Path:        {summary['record_path']}")
        print(f"Record Digest:      {summary['record_digest']}")
        print(f"Reload Verified:    {summary['reloaded_verified']}")
        print("-" * 60)
        print(f"Note: {summary['note']}")
        print("=" * 60)

    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line options for the public API quickstart."""
    parser = argparse.ArgumentParser(
        description="Demonstrate the Robot SF top-level public API facade."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where EpisodeRecord JSON is saved (default: output/examples/quickstart_public_api).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Deterministic random seed (default: {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=DEFAULT_HORIZON,
        help=f"Maximum step budget for the rollout (default: {DEFAULT_HORIZON}).",
    )
    parser.add_argument(
        "--format",
        choices=["friendly", "json"],
        default="friendly",
        help="Output format: 'friendly' for human-readable report or 'json' for structured data.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = parse_args(argv)
    try:
        run_quickstart(
            output_dir=args.output_dir,
            seed=args.seed,
            horizon=args.horizon,
            output_format=args.format,
        )
    except Exception as exc:
        print(f"ERROR: Quickstart failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
