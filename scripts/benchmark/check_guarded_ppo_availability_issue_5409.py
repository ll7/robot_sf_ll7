#!/usr/bin/env python3
"""CPU-only guarded-PPO availability preflight for issue #5409 horizon ablation.

Plain-language summary: the issue #5409 roster-matched horizon ablation configs
(`configs/benchmarks/issue_5409_horizon_ablation_h500.yaml` and the h600 twin)
declare the `guarded_ppo` arm with `availability_gate: dependency_gated` and
`fail_closed_reason: guarded_ppo_checkpoint_observation_contract_missing`. That
flag is a conservative *claim of unavailability*, not proof. This preflight
resolves the two dependencies that flag actually refers to, using the
repository's existing CPU-only infrastructure:

1. the arm checkpoint (`model_id` in the algo_config), resolved through
   `robot_sf.benchmark.campaign_checkpoint_preflight` (network-free cheap mode);
2. the learned-checkpoint observation contract, resolved through
   `robot_sf.benchmark.algorithm_metadata.resolve_learned_checkpoint_observation_contract`.

When both resolve, the arm is AVAILABLE and the `fail_closed_reason` no longer
applies; when either is missing this reports exactly which dependency is absent
and keeps the arm classified `not_available`. The preflight stages nothing,
downloads nothing, executes no episode, and submits no campaign, so it is safe
to run on any CPU node without the #4826 GPU-lifecycle gate.

Exit codes are branchable for a submit wrapper:
    0 -- the guarded_ppo arm is available (both dependencies resolve).
    1 -- the guarded_ppo arm is not available (a dependency is missing).
    2 -- the config could not be loaded or has no guarded_ppo arm.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from loguru import logger

from robot_sf.benchmark.algorithm_metadata import (
    canonical_algorithm_name,
    resolve_learned_checkpoint_observation_contract,
)
from robot_sf.benchmark.camera_ready_campaign import load_campaign_config
from robot_sf.benchmark.campaign_checkpoint_preflight import (
    CampaignCheckpointPreflightError,
    iter_campaign_arm_checkpoint_references,
    resolve_arm_checkpoint,
)

EXIT_AVAILABLE = 0
EXIT_NOT_AVAILABLE = 1
EXIT_CONFIG_ERROR = 2

GUARDED_PPO_ALGO = "guarded_ppo"


def _find_guarded_ppo_planner(cfg: Any) -> Any | None:
    """Return the first enabled guarded_ppo planner spec in a loaded campaign config."""
    for planner in cfg.planners:
        if planner.enabled and canonical_algorithm_name(planner.algo) == GUARDED_PPO_ALGO:
            return planner
    return None


def _checkpoint_resolution(
    cfg: Any, planner_key: str, *, registry_path: Path | None
) -> dict[str, Any]:
    """Resolve the guarded_ppo arm checkpoints in network-free cheap mode."""
    references = [
        ref
        for ref in iter_campaign_arm_checkpoint_references(cfg)
        if ref.planner_key == planner_key
    ]
    if not references:
        return {
            "checked": 0,
            "resolvable": True,
            "status": "no_checkpoint_declared",
            "references": [],
        }
    resolutions = [
        resolve_arm_checkpoint(ref, stage=False, registry_path=registry_path) for ref in references
    ]
    return {
        "checked": len(resolutions),
        "resolvable": all(r.resolvable for r in resolutions),
        "status": "ok" if all(r.resolvable for r in resolutions) else "unresolved",
        "references": [
            {
                "kind": r.reference.kind,
                "value": r.reference.value,
                "status": r.status,
                "detail": r.detail,
            }
            for r in resolutions
        ],
    }


def _observation_contract_resolution(planner: Any) -> dict[str, Any]:
    """Resolve the guarded_ppo learned-checkpoint observation contract."""
    from robot_sf.benchmark.camera_ready._config_types import PlannerSpec

    algo_config: dict[str, Any] = {}
    if isinstance(planner, PlannerSpec) and planner.algo_config_path is not None:
        import yaml

        path = Path(planner.algo_config_path)
        if path.is_file():
            algo_config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    try:
        contract = resolve_learned_checkpoint_observation_contract(planner.algo, algo_config)
    except (ValueError, KeyError, TypeError, FileNotFoundError) as exc:
        return {
            "resolvable": False,
            "status": "error",
            "active_observation_mode": None,
            "metadata_source": None,
            "detail": f"{type(exc).__name__}: {exc}",
        }
    return {
        "resolvable": contract.get("status") not in ("error", "missing"),
        "status": contract.get("status"),
        "active_observation_mode": contract.get("active_observation_mode"),
        "metadata_source": contract.get("metadata_source"),
        "detail": "observation contract resolved",
    }


def _readiness_tier(planner: Any) -> dict[str, Any]:
    """Return the guarded_ppo algorithm readiness tier without importing heavy deps."""
    from robot_sf.benchmark.algorithm_readiness import get_algorithm_readiness

    readiness = get_algorithm_readiness(planner.algo)
    if readiness is None:
        return {"canonical_name": planner.algo, "tier": "unknown"}
    return {"canonical_name": readiness.canonical_name, "tier": readiness.tier}


def check_guarded_ppo_availability(
    config_path: str | Path,
    *,
    registry_path: str | Path | None = None,
) -> dict[str, Any]:
    """Compute the guarded_ppo availability verdict for a campaign config.

    Args:
        config_path: Path to a camera-ready campaign config YAML that may contain
            a guarded_ppo arm (declared via `availability_gate: dependency_gated`).
        registry_path: Optional model-registry path override (useful for tests).

    Returns:
        dict[str, Any]: Structured verdict with ``available`` boolean, the resolved
        checkpoint and observation-contract status, the declared availability gate,
        and the actionable ``fail_closed_reason`` (carried from the config, retained
        when the arm is not available).

    Raises:
        FileNotFoundError / TypeError / ValueError: When the config cannot be loaded.
    """
    cfg = load_campaign_config(Path(config_path))
    planner = _find_guarded_ppo_planner(cfg)
    if planner is None:
        return {
            "config": str(config_path),
            "present": False,
            "available": False,
            "reason": "no enabled guarded_ppo planner arm found in config",
        }

    checkpoint = _checkpoint_resolution(
        cfg,
        planner.key,
        registry_path=Path(registry_path) if registry_path else None,
    )
    observation = _observation_contract_resolution(planner)
    readiness = _readiness_tier(planner)

    available = bool(checkpoint["resolvable"] and observation["resolvable"])

    missing: list[str] = []
    if not checkpoint["resolvable"]:
        missing.append("checkpoint")
    if not observation["resolvable"]:
        missing.append("observation_contract")

    fail_closed_reason = str(getattr(planner, "fail_closed_reason", "") or "").strip()

    return {
        "config": str(config_path),
        "present": True,
        "available": available,
        "planner_key": planner.key,
        "algo": planner.algo,
        "declared_availability_gate": getattr(planner, "availability_gate", None),
        "declared_fail_closed_reason": fail_closed_reason or None,
        "readiness": readiness,
        "checkpoint": checkpoint,
        "observation_contract": observation,
        "remaining_missing_dependencies": missing,
        "verdict": (
            "available"
            if available
            else "not_available" + (f": missing {missing}" if missing else "")
        ),
    }


def main(argv: list[str] | None = None) -> int:
    """Run the guarded-PPO availability preflight CLI."""
    parser = argparse.ArgumentParser(
        description="CPU-only guarded-PPO availability preflight for issue #5409.",
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to a camera-ready campaign config YAML (e.g. the issue #5409 h500/h600 config).",
    )
    parser.add_argument(
        "--registry-path",
        type=Path,
        default=None,
        help="Optional model-registry path override.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the availability verdict as JSON on stdout.",
    )
    args = parser.parse_args(argv)

    if not args.config.is_file():
        print(f"error: campaign config not found: {args.config}", file=sys.stderr)
        return EXIT_CONFIG_ERROR

    try:
        verdict = check_guarded_ppo_availability(
            args.config,
            registry_path=args.registry_path,
        )
    except (FileNotFoundError, TypeError, ValueError, CampaignCheckpointPreflightError) as exc:
        print(f"error: could not evaluate {args.config}: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR

    if not verdict["present"]:
        print(f"NOT AVAILABLE: {verdict['reason']}", file=sys.stderr)
        if args.json:
            print(json.dumps(verdict, indent=2))
        return EXIT_CONFIG_ERROR

    if verdict["available"]:
        if args.json:
            print(json.dumps(verdict, indent=2))
        else:
            logger.info(
                "guarded_ppo arm AVAILABLE: checkpoint + observation contract both resolve "
                f"({verdict['checkpoint']['status']}, {verdict['observation_contract']['status']})"
            )
            print(
                f"AVAILABLE: guarded_ppo arm in {args.config} is runnable "
                "(checkpoint and observation contract resolve; "
                f"readiness tier={verdict['readiness']['tier']})."
            )
        return EXIT_AVAILABLE

    if args.json:
        print(json.dumps(verdict, indent=2))
    else:
        missing = ", ".join(verdict["remaining_missing_dependencies"])
        declared = verdict["declared_fail_closed_reason"] or "unspecified"
        print(
            f"NOT AVAILABLE: guarded_ppo arm in {args.config} missing {missing}. "
            f"Declared fail_closed_reason='{declared}' still applies.",
            file=sys.stderr,
        )
    return EXIT_NOT_AVAILABLE


if __name__ == "__main__":
    raise SystemExit(main())
