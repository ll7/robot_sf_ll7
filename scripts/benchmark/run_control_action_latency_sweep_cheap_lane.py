#!/usr/bin/env python3
"""Cheap-lane CPU execution of the control-action-latency fidelity sweep (issue #5034).

This is the EXECUTION step that prior PRs (#5061, #5085, #5536, #5620, #5629, #5648)
progressively built up: the fail-closed preflight, the durable evidence promoter, the
fixed-scope launch plan, the coverage/reconciliation gate, and a first 36-row CPU
slice over the two always-dependency-free planners. The full 7,344-episode
fixed-scope campaign (all 48 scenarios x all planner groups x all seeds) is a
long-running campaign the cheap lane does not launch.

This script executes REAL Robot SF simulator episodes on CPU for *all* native,
CPU-runnable planner groups on the ``control_action_latency`` axis only (0/1/3 steps,
the 0/100/300 ms-equivalent delays), and writes raw episode rows the durable
promoter consumes. The native planner groups are:

- ``goal_seek`` and ``baseline_social_force``: always dependency-free (pure Python).
- ``orca``: native on CPU through the ``rvo2`` C extension; fail-closed when ``rvo2``
  is not importable (never silently falls back to a heuristic ORCA).
- ``hybrid_rule_v0_minimal``: pure-Python rule planner; gated by the existing
  ``allow_testing_algorithms`` opt-in flag in ``configs/algos/hybrid_rule_v0_minimal.yaml``.

The ORCA and hybrid groups were previously treated as out of cheap-lane reach, but
on a CPU-capable host with ``rvo2`` installed both run natively in well under a second
per episode; neither requires Slurm, GPU, or any degraded/fallback path. Adding their
native latency rows is genuine new evidence toward issue #5034's fixed scope, not a
readiness/decision packet.

It calls the existing :func:`run_episode` in
``scripts/benchmark/run_fidelity_sensitivity_campaign.py`` so the emitted row shape is
byte-for-byte what ``promote_control_action_latency_evidence.py`` expects
(``axis == "control_action_latency"`` with structured ``action_latency`` metadata
from the env ``reset``). No new row schema, no new planner path.

This is a conservative, fail-closed CPU slice. It:
- runs only the latency axis (never silently sweeps other fidelity axes);
- runs only native planner groups whose runtime is actually importable on this host
  (capability-guarded, not just name-allowlisted), and rejects everything else;
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
#: Native CPU-runnable planner groups the cheap-lane slice may execute. ``orca`` is
#: native through the optional ``rvo2`` C extension; ``hybrid_rule_v0_minimal`` is
#: pure Python but gated by an opt-in flag. Neither requires Slurm or GPU.
NATIVE_CPU_PLANNERS = (
    "goal_seek",
    "baseline_social_force",
    "orca",
    "hybrid_rule_v0_minimal",
)
#: Native planners whose runtime is an importable optional dependency. Each maps to a
#: cheap boolean probe so the slice fails closed (rather than silently dropping) when a
#: planner's native runtime is absent on the host.
_RUNTIME_OPTIONAL_PLANNERS = {"orca": "rvo2"}
CLAIM_BOUNDARY = (
    "cheap-lane cpu latency-sweep slice only: executes real Robot SF episodes for the "
    "control_action_latency axis (0/1/3 steps) on the native CPU-runnable planner groups "
    "(goal_seek, baseline_social_force, orca via rvo2, hybrid_rule_v0_minimal). It is a "
    "bounded CPU slice of issue #5034, not the full 7,344-episode fixed-scope campaign, "
    "not simulator-realism evidence, not sim-to-real evidence, and not paper-facing evidence."
)


def _runtime_module_importable(module_name: str) -> bool:
    """Return ``True`` when an optional runtime module imports cleanly.

    Used as a capability probe so the cheap-lane slice can include native planners
    backed by optional C extensions (e.g. ``rvo2`` for ORCA) exactly when they are
    present, and fail closed with a precise reason when they are not.
    """
    try:
        __import__(module_name)
    except (ImportError, OSError):
        return False
    return True


def _native_runtime_available(planner: str) -> bool:
    """Return ``True`` when a native planner's runtime is usable on this host.

    Planners with no optional runtime dependency (``goal_seek``,
    ``baseline_social_force``, ``hybrid_rule_v0_minimal``) are always available on a
    CPU-capable host. Planners backed by an optional module are available only when
    that module imports.
    """
    required = _RUNTIME_OPTIONAL_PLANNERS.get(planner)
    if required is None:
        return True
    return _runtime_module_importable(required)


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

    Only native CPU-runnable planner groups are accepted, and only when their runtime
    is actually importable on this host. Any planner name outside
    ``NATIVE_CPU_PLANNERS`` is rejected, as is any native planner whose optional
    runtime is missing (e.g. ``orca`` without ``rvo2``), so the cheap-lane slice fails
    closed rather than silently narrowing or falling back to a degraded path.
    """
    runner = _load_campaign_runner()
    variants = _latency_variants(config)
    scenarios = list(runner.load_scenarios(scenario_path))
    if not scenarios:
        raise ValueError(f"scenario set produced no scenarios: {scenario_path}")
    planner_names = list(planner_names)
    if not planner_names:
        raise ValueError(
            f"no native planner requested; cheap-lane slice runs only {NATIVE_CPU_PLANNERS}"
        )
    unsupported = sorted(
        {planner for planner in planner_names if planner not in NATIVE_CPU_PLANNERS}
    )
    if unsupported:
        raise ValueError(
            "unsupported planner request for cheap-lane CPU slice: "
            f"{unsupported}; allowed={list(NATIVE_CPU_PLANNERS)}"
        )
    runtime_missing = sorted(
        planner for planner in planner_names if not _native_runtime_available(planner)
    )
    if runtime_missing:
        missing_modules = [
            f"{planner} requires {_RUNTIME_OPTIONAL_PLANNERS[planner]}"
            for planner in runtime_missing
            if planner in _RUNTIME_OPTIONAL_PLANNERS
        ]
        raise ValueError(
            "native runtime not importable on this host for cheap-lane CPU slice: "
            f"{missing_modules}; install the missing dependency or drop the planner"
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
        help="Native CPU-runnable planners to execute (default: all native groups).",
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
