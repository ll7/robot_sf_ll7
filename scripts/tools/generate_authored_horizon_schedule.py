"""Generate an explicit campaign schedule from the resolved authored scenario limits."""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import replace
from pathlib import Path

import yaml

from robot_sf.benchmark.camera_ready._config import _load_campaign_scenarios, load_campaign_config


def authored_schedule_bytes(template: Path, *, repository_root: Path) -> bytes:
    """Return deterministic schedule bytes from the template's authored scenario closure."""
    cfg = load_campaign_config(template, repository_root=repository_root)
    cfg = replace(
        cfg,
        horizon=None,
        scenario_horizons_path=None,
        scenario_horizons_sha256=None,
        planners=tuple(replace(p, horizon_override=None) for p in cfg.planners),
    )
    scenarios = _load_campaign_scenarios(cfg, repository_root=repository_root)
    budgets = {}
    for scenario in sorted(scenarios, key=lambda s: s["name"]):
        budget = scenario.get("simulation_config", {}).get("max_episode_steps")
        if type(budget) is not int or budget <= 0:
            raise ValueError(f"Scenario {scenario['name']} lacks a positive authored step budget")
        budgets[scenario["name"]] = {
            "recommended_horizon_steps": budget,
            "status": "authored",
        }
    return (
        "# Generated from authored limits; do not infer budgets from episode outcomes.\n"
        + yaml.safe_dump({"schema_version": 1, "scenarios": budgets}, sort_keys=False)
    ).encode("utf-8")


def main() -> None:
    """Write or verify a schedule and print its SHA-256 for the campaign declaration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    payload = authored_schedule_bytes(args.template.resolve(), repository_root=root)
    if args.check:
        if args.output.read_bytes() != payload:
            raise SystemExit("Schedule differs from resolved authored limits")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(payload)
    print(hashlib.sha256(payload).hexdigest())


if __name__ == "__main__":
    main()
