"""CPU smoke comparison for issue #4164 goal-posterior planner input.

This script exercises the opt-in planner metadata channel with and without the
Bayesian goal posterior enabled. It is diagnostic wiring evidence only, not a
full benchmark campaign or calibrated intention-prediction claim.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import yaml

from robot_sf.prediction.goal_intention import (
    GoalPosteriorConfig,
    planner_goal_posterior_channel_from_state,
)

CLAIM_BOUNDARY = (
    "diagnostic CPU smoke for opt-in planner input wiring; no full benchmark campaign, "
    "no calibrated human-intention claim, no planner-performance claim"
)


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("smoke config must be a YAML mapping")
    return payload


def _posterior_config(payload: dict[str, Any]) -> GoalPosteriorConfig:
    raw = payload.get("posterior_config", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("posterior_config must be a mapping when present")
    return GoalPosteriorConfig(**raw)


def _scenario_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    scenarios = config.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("config.scenarios must be a non-empty list")
    for index, scenario in enumerate(scenarios):
        if not isinstance(scenario, dict):
            raise ValueError(f"scenarios[{index}] must be a mapping")
        for key in ("id", "positions", "velocities", "goals"):
            if key not in scenario:
                raise ValueError(f"scenarios[{index}] missing required key {key!r}")
    return scenarios


def _run_arm(
    *,
    enabled: bool,
    scenario: dict[str, Any],
    posterior_config: GoalPosteriorConfig,
) -> dict[str, Any]:
    start = time.perf_counter()
    channel = planner_goal_posterior_channel_from_state(
        enabled=enabled,
        positions=scenario["positions"],
        velocities=scenario["velocities"],
        goals=scenario["goals"],
        pedestrian_ids=scenario.get("pedestrian_ids"),
        config=posterior_config,
    )
    runtime_s = time.perf_counter() - start
    posteriors = channel["pedestrian_goal_posteriors"]
    return {
        "enabled": enabled,
        "runtime_s": runtime_s,
        "channel_present": bool(channel["enabled"]) and bool(posteriors),
        "pedestrian_count": len(scenario["positions"]),
        "top_goal_ids": {
            pedestrian_id: summary["top_goal_id"] for pedestrian_id, summary in posteriors.items()
        },
        "top_goal_confidences": {
            pedestrian_id: summary["top_goal_confidence"]
            for pedestrian_id, summary in posteriors.items()
        },
        "blockers": {
            pedestrian_id: summary["blocker"]
            for pedestrian_id, summary in posteriors.items()
            if summary["blocker"] is not None
        },
        "channel": channel,
    }


def build_report(config_path: Path) -> dict[str, Any]:
    """Return diagnostic with/without planner-input channel comparison."""

    config = _load_config(config_path)
    posterior_config = _posterior_config(config)
    scenario_reports = []
    for scenario in _scenario_rows(config):
        without_channel = _run_arm(
            enabled=False,
            scenario=scenario,
            posterior_config=posterior_config,
        )
        with_channel = _run_arm(
            enabled=True,
            scenario=scenario,
            posterior_config=posterior_config,
        )
        scenario_reports.append(
            {
                "scenario_id": scenario["id"],
                "planner_path": "metadata_channel",
                "without_goal_posterior": without_channel,
                "with_goal_posterior": with_channel,
                "runtime_overhead_s": (with_channel["runtime_s"] - without_channel["runtime_s"]),
            }
        )
    return {
        "schema_version": "issue_4164_goal_posterior_planner_input_smoke.v1",
        "config_path": str(config_path),
        "claim_boundary": CLAIM_BOUNDARY,
        "posterior_config": {
            "heading_kappa": posterior_config.heading_kappa,
            "velocity_min_mps": posterior_config.velocity_min_mps,
            "prior_floor": posterior_config.prior_floor,
            "config_hash": posterior_config.config_hash,
        },
        "scenarios": scenario_reports,
    }


def main(argv: list[str] | None = None) -> int:
    """Run the issue #4164 CPU smoke report CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/benchmarks/issue_4164_goal_intention_smoke.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/benchmarks/issue_4164_goal_intention_smoke.json"),
    )
    args = parser.parse_args(argv)

    report = build_report(args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
