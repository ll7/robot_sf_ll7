#!/usr/bin/env python3
"""Cheap-lane CPU execution of the control-action-latency fidelity sweep (issue #5034).

This is the EXECUTION step that prior PRs (#5061, #5085, #5536, #5620, #5629) left
un-done: they built the fail-closed preflight, the durable evidence promoter, the
fixed-scope launch plan, and the coverage/reconciliation gate, but none of them ran
a single real latency episode. The full 7,344-episode fixed-scope campaign requires
the native ORCA/hybrid planner set and Slurm capacity, which the cheap lane must not
use. This script instead executes REAL Robot SF simulator episodes on CPU for the two
dependency-free native planners (``goal_seek``, ``baseline_social_force``) on the
``control_action_latency`` axis only (0/1/3 steps, the 0/100/300 ms-equivalent
delays), and writes raw episode rows the durable promoter consumes.

It calls the existing :func:`run_episode` in
``scripts/benchmark/run_fidelity_sensitivity_campaign.py`` so the emitted row shape is
byte-for-byte what ``promote_control_action_latency_evidence.py`` expects
(``axis == "control_action_latency"`` with structured ``action_latency`` metadata
from the env ``reset``). No new row schema, no new planner path.

This is a conservative, fail-closed CPU slice. It:
- runs only the latency axis (never silently sweeps other fidelity axes);
- runs only planners with a native, dependency-free runner binding;
- writes raw rows under ``output/`` (gitignored) for durable promotion downstream;
- makes NO benchmark / simulator-realism / sim-to-real / paper-facing claim.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = "configs/research/fidelity_sensitivity_v1.yaml"
DEFAULT_SCENARIO_SET = "configs/scenarios/sets/issue_5034_latency_sweep_cpu_v1.yaml"
DEFAULT_RAW_ROOT = "output/fidelity_latency_raw"
LATENCY_AXIS_KEY = "control_action_latency"
#: Dependency-free native planners the cheap-lane CPU slice may execute.
NATIVE_CPU_PLANNERS = ("goal_seek", "baseline_social_force")
CLAIM_BOUNDARY = (
    "cheap-lane cpu latency-sweep slice only: executes real Robot SF episodes for the "
    "control_action_latency axis (0/1/3 steps) on the two dependency-free native planners. "
    "It is a bounded CPU slice of issue #5034, not the full fixed-scope campaign (native "
    "ORCA/hybrid planners + Slurm), not simulator-realism evidence, not sim-to-real evidence, "
    "and not paper-facing evidence."
)


def _load_campaign_runner() -> ModuleType:
    """Import the existing fidelity-campaign runner to reuse its ``run_episode``."""
    module_path = REPO_ROOT / "scripts" / "benchmark" / "run_fidelity_sensitivity_campaign.py"
    spec = importlib.util.spec_from_file_location(
        "fidelity_sensitivity_campaign_runner", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load campaign runner module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["fidelity_sensitivity_campaign_runner"] = module
    spec.loader.exec_module(module)
    return module


def _latency_variants(config: Mapping[str, Any]) -> list[Any]:
    """Return the materialized ``control_action_latency`` variant specs only."""
    runner = _load_campaign_runner()
    variant_index = runner.build_fixed_scope_variant_index(config)
    variants = [spec for (axis, _source), spec in variant_index.items() if axis == LATENCY_AXIS_KEY]
    if not variants:
        raise ValueError(
            f"config {DEFAULT_CONFIG} has no '{LATENCY_AXIS_KEY}' axis; "
            "the latency sweep requires PR #5026's axis schema."
        )
    return variants


def run_sweep(
    *,
    config: Mapping[str, Any],
    scenario_path: Path,
    planner_names: Sequence[str],
    horizon: int,
    seeds: Sequence[int],
) -> list[dict[str, Any]]:
    """Execute real latency-axis episodes for each native planner and seed.

    Only dependency-free native planners are accepted. Any planner name outside
    ``NATIVE_CPU_PLANNERS`` is rejected so the cheap-lane slice fails closed rather
    than silently narrowing a requested campaign that needs the native ORCA/hybrid
    runtime or Slurm capacity.
    """
    runner = _load_campaign_runner()
    variants = _latency_variants(config)
    scenarios = list(runner.load_scenarios(scenario_path))
    if not scenarios:
        raise ValueError(f"scenario set produced no scenarios: {scenario_path}")
    planner_names = list(planner_names)
    unsupported = sorted({planner for planner in planner_names if planner not in NATIVE_CPU_PLANNERS})
    if unsupported:
        raise ValueError(
            "unsupported planner request for cheap-lane CPU slice: "
            f"{unsupported}; allowed={list(NATIVE_CPU_PLANNERS)}"
        )
    if not planner_names:
        raise ValueError(
            "no dependency-free native planner requested; cheap-lane slice runs only "
            f"{NATIVE_CPU_PLANNERS}"
        )
    rows: list[dict[str, Any]] = []
    for planner in planner_names:
        for scenario in scenarios:
            for variant in variants:
                for seed in seeds:
                    rows.append(
                        runner.run_episode(
                            scenario,
                            scenario_path=scenario_path,
                            variant=variant,
                            planner_name=planner,
                            seed=int(seed),
                            horizon=int(horizon),
                        )
                    )
    return rows


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--scenario-set", default=DEFAULT_SCENARIO_SET)
    parser.add_argument("--raw-root", default=DEFAULT_RAW_ROOT)
    parser.add_argument("--horizon", type=int, default=80)
    parser.add_argument(
        "--planners",
        nargs="+",
        default=list(NATIVE_CPU_PLANNERS),
        help="Native dependency-free planners to execute (default: all CPU-native planners).",
    )
    parser.add_argument(
        "--seed",
        action="append",
        type=int,
        dest="seeds",
        help="Episode seeds; repeatable. Defaults to the scenario-set seeds.",
    )
    parser.add_argument("--date", default=dt.datetime.now(tz=dt.UTC).date().isoformat())
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the cheap-lane CPU latency sweep and write raw episode rows."""
    args = _parse_args(argv)
    config_path = REPO_ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    scenario_path = REPO_ROOT / args.scenario_set

    planner_names = list(args.planners)

    # Scenario-set seeds unless overridden on the CLI.
    scenario_seeds = []
    scenario_data = yaml.safe_load(scenario_path.read_text(encoding="utf-8")) or {}
    for scenario in scenario_data.get("scenarios", []):
        scenario_seeds.extend(int(s) for s in scenario.get("seeds", []))
    seeds = args.seeds if args.seeds else sorted(set(scenario_seeds))
    if not seeds:
        seeds = [111, 112, 113]

    rows = run_sweep(
        config=config,
        scenario_path=scenario_path,
        planner_names=planner_names,
        horizon=int(args.horizon),
        seeds=seeds,
    )

    raw_root = REPO_ROOT / args.raw_root
    raw_root.mkdir(parents=True, exist_ok=True)
    raw_rows_path = raw_root / "episode_rows.jsonl"
    raw_rows_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    latency_rows = [row for row in rows if str(row.get("axis")) == LATENCY_AXIS_KEY]
    observed_steps = sorted({int(row["action_latency"]["effective_steps"]) for row in latency_rows})
    print(f"cheap-lane latency sweep complete: {len(rows)} raw rows written")
    print(f"  raw_rows_path={raw_root / 'episode_rows.jsonl'}")
    print(f"  planners={planner_names}")
    print(f"  latency axis rows={len(latency_rows)}")
    print(f"  observed action-latency steps={observed_steps}")
    print(f"  claim_boundary={CLAIM_BOUNDARY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
