#!/usr/bin/env python3
"""Build and check the issue #6642 collision-envelope radius-sweep preparation manifest.

This script is a dry-run preparation tool. It reads the tracked manifest config
(``configs/benchmarks/issue_6642_radius_sweep_manifest_v1.yaml``), resolves the
fixed factors and campaign identity of every radius arm (0.5/0.8/1.0 m) from its
own tracked arm campaign config plus the scenario matrix, fails closed on any
non-radius drift across arms, and writes a preparation-only manifest plus a
checker summary. It does NOT submit, run, or authorize any production SLURM
compute.

Exit codes preserve fail-closed semantics:
- 0: manifest built and checker passes
- 2: contract violation (checker ``passes`` is false) or unexpected failure
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from collections.abc import Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST_CONFIG = "configs/benchmarks/issue_6642_radius_sweep_manifest_v1.yaml"
DEFAULT_OUTPUT_ROOT = "output/issue_6642_radius_sweep/manifest"


def _git_head(repo_root: Path) -> str:
    """Return the HEAD sha, or 'pending_launch' when git is unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return "pending_launch"
    return result.stdout.strip() if result.returncode == 0 else "pending_launch"


def _repo_relative(path: Path) -> str:
    """Return a repo-relative posix path string when resolvable."""
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping, failing closed on a non-mapping payload."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return payload


def _resolve_arm(
    manifest_config: dict[str, Any],
    repo_root: Path,
    *,
    arm_key: str,
    radius_m: float,
    baseline: bool,
) -> tuple[Any, Any]:
    """Resolve one radius arm's fixed factors and campaign identity.

    Imports are lazy so this script's argparse/help path stays import-light.

    Returns:
        Tuple of ``(FixedFactors, ArmCampaignIdentity)`` for the arm.
    """
    from robot_sf.benchmark.camera_ready import load_campaign_config
    from robot_sf.benchmark.radius_sweep_manifest import (
        ARM_CONFIG_KEYS,
        EXPECTED_DT,
        EXPECTED_HORIZON,
        EXPECTED_KINEMATICS,
        ArmCampaignIdentity,
        FixedFactors,
        validate_arm_campaign_payload,
        validate_arm_fixed_factors,
    )
    from robot_sf.training.scenario_loader import load_scenarios

    config_key = ARM_CONFIG_KEYS[arm_key]
    declared = manifest_config.get(config_key)
    if not isinstance(declared, str) or not declared.strip():
        raise ValueError(f"Manifest config requires a non-empty {config_key!r}")
    arm_config_path = repo_root / declared
    if not arm_config_path.is_file():
        raise FileNotFoundError(f"{arm_key} arm campaign config not found: {arm_config_path}")
    cfg = load_campaign_config(arm_config_path)
    validate_arm_campaign_payload(
        _load_yaml(arm_config_path), arm_key=arm_key, radius_m=radius_m, baseline=baseline
    )
    if cfg.radius_sweep is None:
        raise ValueError(f"{arm_key} arm campaign config must declare radius_sweep metadata")

    if cfg.horizon is None or cfg.dt is None:
        raise ValueError(f"{arm_key} arm campaign config must declare both horizon and dt")
    if tuple(cfg.kinematics_matrix) != (EXPECTED_KINEMATICS,):
        raise ValueError(
            f"{arm_key} arm kinematics_matrix must be [{EXPECTED_KINEMATICS!r}], "
            f"got {list(cfg.kinematics_matrix)!r}"
        )
    if cfg.horizon != EXPECTED_HORIZON:
        raise ValueError(f"{arm_key} arm horizon must be {EXPECTED_HORIZON}, got {cfg.horizon}")
    if cfg.dt != EXPECTED_DT:
        raise ValueError(f"{arm_key} arm dt must be {EXPECTED_DT}, got {cfg.dt}")

    planner_keys = tuple(planner.key for planner in cfg.planners if planner.enabled)

    # Resolve the scenario roster and seed list directly from the declared sources
    # so the manifest's row-identity ledger stays grounded in tracked configs.
    scenario_matrix_path = cfg.scenario_matrix_path
    scenarios = load_scenarios(scenario_matrix_path, base_dir=scenario_matrix_path.parent)
    scenario_names = tuple(
        sorted(
            str(scenario.get("name") or scenario.get("scenario_id") or scenario.get("id") or "")
            for scenario in scenarios
        )
    )

    seed_sets_path = cfg.seed_policy.seed_sets_path
    if seed_sets_path is None or not seed_sets_path.is_file():
        raise FileNotFoundError(f"Seed sets file not found: {seed_sets_path}")
    seed_sets = _load_yaml(seed_sets_path)
    seed_set_name = cfg.seed_policy.seed_set
    if not seed_set_name or seed_set_name not in seed_sets:
        raise ValueError(f"seed_set {seed_set_name!r} not found in {seed_sets_path}")
    seeds = tuple(int(seed) for seed in seed_sets[seed_set_name])

    factors = FixedFactors(
        scenario_matrix=_repo_relative(scenario_matrix_path),
        scenario_count=len(scenario_names),
        scenario_names=scenario_names,
        planner_keys=planner_keys,
        seed_set=str(seed_set_name),
        seeds=seeds,
        horizon=int(cfg.horizon),
        dt=float(cfg.dt),
        kinematics=str(cfg.kinematics_matrix[0]),
        release_tag=str(cfg.release_tag),
    )
    validate_arm_fixed_factors(factors, arm_key=arm_key)
    identity = ArmCampaignIdentity(
        arm_key=arm_key,
        campaign_config=_repo_relative(arm_config_path),
        release_tag=str(cfg.release_tag),
        runtime_binding_status=cfg.radius_sweep.runtime_binding_status,
        binding_contract_version=cfg.radius_sweep.binding_contract_version,
        gate1_canary_issue=cfg.radius_sweep.gate1_canary_issue,
        gate1_receipt_sha256=cfg.radius_sweep.gate1_receipt_sha256,
        gate1_source_commit=cfg.radius_sweep.gate1_source_commit,
        runtime_binding_note=cfg.radius_sweep.runtime_binding_note,
    )
    return factors, identity


def _resolve_all_arms(manifest_config: dict[str, Any], repo_root: Path) -> tuple[Any, list[Any]]:
    """Resolve and cross-validate all three radius arms.

    The #6600/#6642 stop rule fixes every non-radius factor across arms, so any
    drift between one arm's resolved factors and the baseline arm's factors fails
    closed before any manifest is built.

    Returns:
        Tuple of the baseline (1.0 m) arm's fixed factors and the arm campaign
        identities in radius order (0.5/0.8/1.0 m).
    """
    from dataclasses import replace

    from robot_sf.benchmark.radius_sweep_manifest import (
        BASELINE_RADIUS,
        PRODUCTION_RADII,
        PRODUCTION_RADIUS_KEYS,
    )

    resolved = [
        _resolve_arm(
            manifest_config,
            repo_root,
            arm_key=arm_key,
            radius_m=radius_m,
            baseline=radius_m == BASELINE_RADIUS,
        )
        for arm_key, radius_m in zip(PRODUCTION_RADIUS_KEYS, PRODUCTION_RADII, strict=True)
    ]
    baseline_factors = resolved[-1][0]
    baseline_shared = replace(baseline_factors, release_tag="")
    for arm_key, (factors, _identity) in zip(PRODUCTION_RADIUS_KEYS, resolved, strict=True):
        if replace(factors, release_tag="") != baseline_shared:
            raise ValueError(
                f"non-radius fixed factors differ between arm {arm_key!r} and the "
                "baseline arm; all non-radius factors must stay fixed across arms"
            )
    return baseline_factors, [identity for _factors, identity in resolved]


def _resolve_arm_fixed_factors(manifest_config: dict[str, Any], repo_root: Path):
    """Resolve fixed factors from the baseline 1.0 m arm campaign config.

    Returns:
        The baseline arm's ``FixedFactors``.
    """
    from robot_sf.benchmark.radius_sweep_manifest import BASELINE_RADIUS

    factors, _identity = _resolve_arm(
        manifest_config,
        repo_root,
        arm_key="r1p0",
        radius_m=BASELINE_RADIUS,
        baseline=True,
    )
    return factors


def build_and_check(
    manifest_config_path: Path,
    output_root: Path,
    repo_root: Path,
    *,
    check_only: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    """Build the manifest, run the checker, and write both artifacts.

    With ``check_only`` the checker summary is produced without writing any
    manifest artifacts.
    """
    from robot_sf.benchmark.radius_sweep_manifest import (
        ManifestOptions,
        RadiusSweepManifestError,
        build_radius_sweep_manifest,
        check_radius_sweep_manifest,
        write_radius_sweep_manifest,
        write_radius_sweep_manifest_check,
    )

    manifest_config = _load_yaml(manifest_config_path)
    fixed_factors, arm_identities = _resolve_all_arms(manifest_config, repo_root)
    options = ManifestOptions(
        config_path=_repo_relative(manifest_config_path),
        git_head=_git_head(repo_root),
    )
    try:
        manifest = build_radius_sweep_manifest(
            manifest_config,
            fixed_factors=fixed_factors,
            arm_identities=arm_identities,
            options=options,
        )
        check_summary = check_radius_sweep_manifest(manifest)
    except RadiusSweepManifestError as exc:
        check_summary = {
            "status": "manifest_check_only",
            "evidence_status": "not_benchmark_evidence",
            "violations": [str(exc)],
            "passes": False,
        }
        return {}, check_summary, Path(), Path()

    if check_only:
        return manifest, check_summary, Path(), Path()

    manifest_path = write_radius_sweep_manifest(manifest, output_root)
    check_path = write_radius_sweep_manifest_check(check_summary, output_root)
    return manifest, check_summary, manifest_path, check_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-config",
        type=Path,
        default=Path(DEFAULT_MANIFEST_CONFIG),
        help=f"Tracked radius-sweep manifest config path (default: {DEFAULT_MANIFEST_CONFIG}).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(DEFAULT_OUTPUT_ROOT),
        help=f"Output directory for manifest artifacts (default: {DEFAULT_OUTPUT_ROOT}).",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only print the checker summary JSON without writing manifest artifacts.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Build and check the radius-sweep preparation manifest."""
    args = parse_args(argv)
    repo_root = REPO_ROOT
    manifest_config_path = (
        args.manifest_config
        if args.manifest_config.is_absolute()
        else repo_root / args.manifest_config
    )
    if not manifest_config_path.is_file():
        print(f"Manifest config not found: {manifest_config_path}", file=sys.stderr)
        return 2
    output_root = args.out if args.out.is_absolute() else repo_root / args.out

    _manifest, check_summary, manifest_path, check_path = build_and_check(
        manifest_config_path, output_root, repo_root, check_only=args.check_only
    )

    print(json.dumps(check_summary, indent=2, sort_keys=True))
    if not check_summary.get("passes", False):
        return 2
    if not args.check_only and manifest_path:
        print(f"manifest: {manifest_path}", file=sys.stderr)
        print(f"check:    {check_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
