#!/usr/bin/env python3
"""CPU VecEnv worker-mode throughput comparator.

Loads a standard training config, constructs ``dummy``, ``subproc``,
``threaded``, and ``threaded_lidar_batch`` VecEnv modes, runs bounded warmup
and step loops, and writes machine-readable JSON with throughput and
host/config provenance.

Usage
-----
::

    # default smoke config, 4 envs, 20 warmup + 100 measured steps
    uv run python scripts/validation/run_vecenv_worker_mode_throughput.py

    # explicit config and env count
    uv run python scripts/validation/run_vecenv_worker_mode_throughput.py \\
        --config configs/training/lidar/lidar_ppo_mlp_smoke_issue_1662.yaml \\
        --num-envs 4 --warmup-steps 10 --measure-steps 50 \\
        --output output/vecenv_throughput.json

Output schema (``vecenv_throughput_comparator.v1``)
----------------------------------------------------
::

    {
      "schema": "vecenv_throughput_comparator.v1",
      "config_path": "...",
      "config_sha256": "...",
      "commit": "...",
      "host": "...",
      "num_envs": 4,
      "warmup_steps": 20,
      "measure_steps": 100,
      "baseline_mode": "dummy",
      "results": [
        {
          "mode": "dummy",
          "transitions_per_second": 42.3,
          "speedup_vs_baseline": 1.0,
          "status": "ok",
          "error": null
        },
        ...
      ]
    }

Notes
-----
- ``subproc`` mode spawns sub-processes; the script uses ``spawn`` start
  method and must be invoked as an executable file (not via ``python -c``).
- ``threaded_lidar_batch`` uses the same in-process workers as ``threaded``
  while enabling its cross-environment LiDAR batch coordinator.

Found while implementing #4981; tracked as issue #5118.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import json
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).parent.parent.parent.resolve()
_DEFAULT_CONFIG = _REPO_ROOT / "configs/training/lidar/lidar_ppo_mlp_smoke_issue_1662.yaml"
_SCHEMA = "vecenv_throughput_comparator.v1"
_SUPPORTED_MODES = ("dummy", "subproc", "threaded", "threaded_lidar_batch")
_RECOVERABLE_MODE_ERRORS = (
    EOFError,
    BrokenPipeError,
    ImportError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


# ---------------------------------------------------------------------------
# Picklable env factory (needed for SubprocVecEnv spawn workers)
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _EnvFactory:
    """Picklable callable that constructs one training environment."""

    scenario_path: Path
    env_factory_kwargs: dict[str, Any]
    seed: int | None

    def __call__(self):
        from robot_sf.gym_env.environment_factory import make_robot_env
        from robot_sf.training.scenario_loader import (
            build_robot_config_from_scenario,
            load_scenarios,
        )

        scenarios = load_scenarios(self.scenario_path)
        if not scenarios:
            raise RuntimeError(f"No scenarios found in {self.scenario_path}")
        config = build_robot_config_from_scenario(
            scenarios[0],
            scenario_path=self.scenario_path,
        )
        return make_robot_env(
            config=config,
            seed=self.seed,
            **self.env_factory_kwargs,
        )


# ---------------------------------------------------------------------------
# Config loading helpers
# ---------------------------------------------------------------------------


def _load_yaml_raw(path: Path) -> dict[str, Any]:
    import yaml

    try:
        with path.open(encoding="utf-8") as fh:
            payload = yaml.safe_load(fh)
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"could not load config {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"config {path} must contain a YAML mapping")
    return dict(payload)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def _git_commit(repo_root: Path) -> str:
    try:
        result = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return result.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _resolve_scenario_path(config: dict[str, Any], config_path: Path) -> Path:
    """Resolve the scenario_config path relative to the config file."""
    raw = config.get("scenario_config")
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("config scenario_config must be a non-empty path string")
    scenario = Path(raw)
    if scenario.is_absolute():
        return scenario
    return (config_path.parent / scenario).resolve()


def _resolve_num_envs(config: dict[str, Any], override: int | None) -> int:
    raw = override if override is not None else config.get("num_envs")
    if raw is None or str(raw).strip().lower().startswith("auto"):
        return 4
    try:
        num_envs = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"num_envs must be a positive integer, got {raw!r}") from exc
    if num_envs < 1:
        raise ValueError(f"num_envs must be a positive integer, got {num_envs}")
    return num_envs


def _env_factory_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    raw = config.get("env_factory_kwargs") or {}
    if not isinstance(raw, dict):
        raise ValueError("config env_factory_kwargs must be a mapping when provided")
    return dict(raw)


# ---------------------------------------------------------------------------
# VecEnv construction
# ---------------------------------------------------------------------------


def _build_env_fns(
    scenario_path: Path,
    env_factory_kwargs: dict[str, Any],
    num_envs: int,
    base_seed: int,
) -> list[_EnvFactory]:
    return [
        _EnvFactory(
            scenario_path=scenario_path,
            env_factory_kwargs=env_factory_kwargs,
            seed=base_seed + i,
        )
        for i in range(num_envs)
    ]


def _build_vec_env(mode: str, env_fns: list[_EnvFactory]):
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    from robot_sf.training.threaded_vec_env import ThreadedVecEnv

    if mode == "dummy":
        return DummyVecEnv(env_fns)
    if mode == "subproc":
        return SubprocVecEnv(env_fns, start_method="spawn")
    if mode == "threaded":
        return ThreadedVecEnv(env_fns)
    if mode == "threaded_lidar_batch":
        return ThreadedVecEnv(env_fns, batch_lidar=True)
    raise ValueError(f"Unsupported mode: {mode!r}; expected one of {_SUPPORTED_MODES}")


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def _measure_mode(
    mode: str,
    env_fns: list[_EnvFactory],
    warmup_steps: int,
    measure_steps: int,
) -> dict[str, Any]:
    """Run warmup then measured steps; return throughput record."""
    import numpy as np

    try:
        vec_env = _build_vec_env(mode, env_fns)
    except _RECOVERABLE_MODE_ERRORS as exc:
        return {
            "mode": mode,
            "transitions_per_second": None,
            "speedup_vs_baseline": None,
            "status": "construction_failed",
            "error": str(exc),
        }

    num_envs = vec_env.num_envs
    try:
        vec_env.reset()
        action_space = vec_env.action_space
        for _ in range(warmup_steps):
            actions = np.array([action_space.sample() for _ in range(num_envs)])
            vec_env.step(actions)

        vec_env.reset()
        t0 = time.perf_counter()
        for _ in range(measure_steps):
            actions = np.array([action_space.sample() for _ in range(num_envs)])
            vec_env.step(actions)
        elapsed = time.perf_counter() - t0

        transitions = num_envs * measure_steps
        tps = transitions / elapsed if elapsed > 0 else float("inf")
        return {
            "mode": mode,
            "transitions_per_second": round(tps, 2),
            "speedup_vs_baseline": None,  # filled in by caller
            "status": "ok",
            "error": None,
        }
    except _RECOVERABLE_MODE_ERRORS as exc:
        return {
            "mode": mode,
            "transitions_per_second": None,
            "speedup_vs_baseline": None,
            "status": "step_failed",
            "error": str(exc),
        }
    finally:
        with contextlib.suppress(OSError, RuntimeError):
            vec_env.close()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for the VecEnv throughput comparator."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare CPU VecEnv worker-mode throughput "
            "(dummy / subproc / threaded / threaded_lidar_batch). "
            "Writes machine-readable JSON with provenance for acceptance audits."
        )
    )
    parser.add_argument(
        "--config",
        default=str(_DEFAULT_CONFIG),
        metavar="YAML",
        help="Training config YAML (default: lidar_ppo_mlp_smoke_issue_1662.yaml).",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=None,
        metavar="N",
        help="Number of parallel environments (overrides config; default: config value or 4).",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=20,
        metavar="N",
        help="Steps to discard before measuring (default: 20).",
    )
    parser.add_argument(
        "--measure-steps",
        type=int,
        default=100,
        metavar="N",
        help="Steps to time per mode (default: 100).",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=list(_SUPPORTED_MODES),
        choices=list(_SUPPORTED_MODES),
        metavar="MODE",
        help=f"Worker modes to test (default: all; choices: {_SUPPORTED_MODES}).",
    )
    parser.add_argument(
        "--skip-subproc",
        action="store_true",
        help="Skip subproc mode (useful when calling from python -c or import context).",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="PATH",
        help="JSON output path (default: output/vecenv_throughput_<host>.json).",
    )
    return parser


def _run_modes(
    modes: list[str],
    env_fns: list[_EnvFactory],
    warmup_steps: int,
    measure_steps: int,
) -> tuple[list[dict[str, Any]], float | None]:
    """Measure throughput for each mode; return results and baseline TPS."""
    results: list[dict[str, Any]] = []
    baseline_tps: float | None = None
    baseline_mode = modes[0] if modes else "dummy"

    for mode in modes:
        print(f"  measuring mode={mode} ...", file=sys.stderr)
        record = _measure_mode(mode, env_fns, warmup_steps, measure_steps)
        results.append(record)
        if mode == baseline_mode and record["transitions_per_second"] is not None:
            baseline_tps = record["transitions_per_second"]

    for record in results:
        tps = record["transitions_per_second"]
        if tps is not None and baseline_tps is not None and baseline_tps > 0:
            record["speedup_vs_baseline"] = round(tps / baseline_tps, 3)

    return results, baseline_tps


def _write_results(
    output_data: dict[str, Any],
    args_output: str | None,
    host: str,
) -> Path:
    """Write JSON output; return the path written."""
    if args_output:
        output_path = Path(args_output)
    else:
        out_dir = _REPO_ROOT / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / f"vecenv_throughput_{host}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_data, indent=2))
    return output_path


def _print_table(results: list[dict[str, Any]]) -> None:
    """Print a human-readable summary table to stdout."""
    print(f"\n{'mode':<20} {'tps':>12} {'speedup':>10} {'status':<20}")
    print("-" * 66)
    for rec in results:
        tps = rec["transitions_per_second"]
        speedup = rec["speedup_vs_baseline"]
        tps_str = f"{tps:.1f}" if tps is not None else "N/A"
        speedup_str = f"{speedup:.3f}x" if speedup is not None else "N/A"
        print(f"{rec['mode']:<20} {tps_str:>12} {speedup_str:>10} {rec['status']:<20}")


def _validate_run_parameters(args: argparse.Namespace) -> None:
    """Reject invalid timing inputs instead of silently emitting misleading measurements."""
    if not args.modes or args.modes[0] != "dummy":
        raise ValueError("--modes must begin with dummy so speedup has a stable baseline")
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps must be >= 0")
    if args.measure_steps < 1:
        raise ValueError("--measure-steps must be >= 1")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: measure and report VecEnv worker-mode throughput."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        print(f"ERROR: config not found: {config_path}", file=sys.stderr)
        return 1

    try:
        _validate_run_parameters(args)
        raw_config = _load_yaml_raw(config_path)
        scenario_path = _resolve_scenario_path(raw_config, config_path)
        num_envs = _resolve_num_envs(raw_config, args.num_envs)
        env_factory_kwargs = _env_factory_kwargs(raw_config)
    except ValueError as exc:
        print(f"ERROR: invalid comparator configuration: {exc}", file=sys.stderr)
        return 1
    if not scenario_path.exists():
        print(f"ERROR: scenario not found: {scenario_path}", file=sys.stderr)
        return 1

    modes = list(args.modes)
    if args.skip_subproc and "subproc" in modes:
        modes.remove("subproc")
        print("INFO: subproc mode skipped via --skip-subproc", file=sys.stderr)

    host = socket.gethostname()
    commit = _git_commit(_REPO_ROOT)

    print(
        f"VecEnv throughput comparator | "
        f"host={host} commit={commit[:12]} "
        f"num_envs={num_envs} warmup={args.warmup_steps} measure={args.measure_steps}",
        file=sys.stderr,
    )

    env_fns = _build_env_fns(scenario_path, env_factory_kwargs, num_envs, base_seed=42)
    results, _ = _run_modes(modes, env_fns, args.warmup_steps, args.measure_steps)

    output_data: dict[str, Any] = {
        "schema": _SCHEMA,
        "config_path": str(config_path),
        "config_sha256": _sha256_file(config_path),
        "commit": commit,
        "host": host,
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "platform": platform.platform(),
        "num_envs": num_envs,
        "warmup_steps": args.warmup_steps,
        "measure_steps": args.measure_steps,
        "baseline_mode": modes[0] if modes else "dummy",
        "results": results,
    }

    output_path = _write_results(output_data, args.output, host)
    print(f"  wrote: {output_path}", file=sys.stderr)
    _print_table(results)

    failed = [r for r in results if r["status"] != "ok"]
    if failed:
        modes_str = ", ".join(r["mode"] for r in failed)
        print(f"\nWARN: failed modes: {modes_str}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
