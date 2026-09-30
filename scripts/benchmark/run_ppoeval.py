#!/usr/bin/env python3
"""Diagnostic-only PPO semantics campaign. Never submits or publishes."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

# Eager CPU inference: optional Triton is unused and crashes on some workstations.
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
sys.modules["triton"] = None

import torch
import yaml

from robot_sf.benchmark.camera_ready._config import _load_campaign_scenarios, _resolve_seed_override
from robot_sf.benchmark.camera_ready_campaign import load_campaign_config, run_campaign
from robot_sf.benchmark.fallback_policy import campaign_exit_code
from robot_sf.models import get_registry_entry, resolve_model_path, sha256_of_file


def main():  # noqa: C901
    """Validate the frozen diagnostic contract and execute the canonical campaign."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/benchmarks/diagnostics/ppoeval.yaml")
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--head-sha", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-arm", choices=["ppo", "guarded_ppo"], default="ppo")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if head != args.head_sha:
        raise ValueError("HEAD differs from frozen campaign SHA")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("Campaign source checkout must be clean")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cfg = load_campaign_config(args.config)
    if {p.key for p in cfg.planners} != {"ppo", "guarded_ppo"}:
        raise ValueError("Only the two requested PPO arms are admitted")
    seeds = list(_resolve_seed_override(cfg.seed_policy))
    if seeds != list(range(1001, 1011)):
        raise ValueError("Campaign must use precisely dev seeds 1001-1010")
    if not 1 <= cfg.workers <= 3 or cfg.arm_isolation != "in_process":
        raise ValueError("At most parent + three workers; arms must run in process")
    if not cfg.record_simulation_step_trace or cfg.horizon != 600 or cfg.dt != 0.1:
        raise ValueError("Trace/H600/dt0.1 contract violated")
    if args.smoke:
        cfg = replace(
            cfg,
            workers=1,
            planners=tuple(p for p in cfg.planners if p.key == args.smoke_arm),
            seed_policy=replace(cfg.seed_policy, seeds=(1001,)),
            scenario_candidates=replace(
                cfg.scenario_candidates, names=("classic_head_on_corridor_medium",)
            ),
        )
    scenarios = _load_campaign_scenarios(cfg)
    expected = len(scenarios) * len(cfg.planners) * len(_resolve_seed_override(cfg.seed_policy))
    if expected != (1 if args.smoke else 320):
        raise ValueError(f"Unexpected matrix size {expected}")
    checkpoint_identity = {}
    for arm in cfg.planners:
        arm_cfg = yaml.safe_load(arm.algo_config_path.read_text())
        for model_id in [arm_cfg["model_id"]] + (
            [arm_cfg["predictive_foresight_model_id"]]
            if arm_cfg.get("predictive_foresight_enabled")
            else []
        ):
            path = resolve_model_path(model_id)
            digest = sha256_of_file(path)
            if digest != get_registry_entry(model_id)["github_release"]["sha256"]:
                raise ValueError(f"Checkpoint digest mismatch: {model_id}")
            checkpoint_identity[model_id] = digest
    if args.check_only:
        print(
            json.dumps(
                {
                    "status": "ready",
                    "expected_episodes": expected,
                    "head_sha": head,
                    "checkpoints": checkpoint_identity,
                }
            )
        )
        return 0
    output = args.output_root.resolve() / args.campaign_id
    if output.exists():
        raise ValueError("Fresh campaign ID required; preserve failed or partial runs")
    output.mkdir(parents=True)
    receipt = {
        "evidence_status": "diagnostic-only",
        "head_sha": head,
        "python": platform.python_version(),
        "expected_episodes": expected,
        "checkpoint_sha256": checkpoint_identity,
        "config_sha256": sha256_of_file(args.config),
        "command": shlex.join(sys.argv),
        "torch": torch.__version__,
        "process_limit": 4,
        "smoke": args.smoke,
    }
    (output / "ppoeval_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    try:
        result = run_campaign(
            cfg,
            output_root=args.output_root.resolve(),
            campaign_id=args.campaign_id,
            skip_publication_bundle=True,
            invoked_command=shlex.join(sys.argv),
            arm_isolation="in_process",
        )
        (output / "ppoeval_result.json").write_text(
            json.dumps(result, indent=2, default=str) + "\n"
        )
        # Acquisition completion is distinct from release/benchmark admission.
        # The canonical result and non-success exit status remain preserved.
        rows = [
            json.loads(line)
            for path in (output / "runs").rglob("episodes.jsonl")
            for line in path.read_text().splitlines()
        ]
        if len(rows) != expected or any(row.get("status") == "error" for row in rows):
            raise ValueError(f"Incomplete diagnostic acquisition: {len(rows)}/{expected}")
        for row in rows:
            steps = (
                row.get("algorithm_metadata", {}).get("simulation_step_trace", {}).get("steps", [])
            )
            if len(steps) != row["steps"] or any(
                not step["planner"].get("ppoeval_proposal", {}).get("raw_policy_output")
                for step in steps
            ):
                raise ValueError("Missing native PPO proposal or complete step trace")
        completion = {
            "acquisition_status": "complete",
            "evidence_status": "diagnostic-only",
            "episodes": len(rows),
            "benchmark_success": result.get("benchmark_success", False),
            "canonical_campaign_exit_code": campaign_exit_code(result),
        }
        (output / "ppoeval_completion.json").write_text(json.dumps(completion, indent=2) + "\n")
        print(json.dumps(completion))
        return 0
    finally:
        checksums = {
            p.relative_to(output).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(output.rglob("*"))
            if p.is_file() and p.name != "ppoeval_checksums.json"
        }
        (output / "ppoeval_checksums.json").write_text(json.dumps(checksums, indent=2) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
