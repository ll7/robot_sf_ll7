#!/usr/bin/env python3
"""Compile the issue #5578 robot speed-tier campaign manifest and run a disjoint-seed
activation preflight.

Plain-language summary
----------------------
This module turns the amended issue #5578 preregistration (frozen by #6100 in
``configs/benchmarks/issue_5578_robot_speed_tier_preregistration.yaml``) into one
exact, auditable, campaign-lane execution manifest, and proves -- using a small,
explicitly non-evidence preflight -- that the robot speed-cap intervention reaches
the real Robot SF runtime and measurably activates across all three speed tiers
(2.0 / 3.0 / 4.0 m/s) before the 2,160 registered episodes are committed.

What this module owns
---------------------
1. ``compile_campaign_manifest``: materialize exactly 2,160 registered identities
   (6 frozen scenarios x 3 frozen speed tiers x 4 frozen planners x 30 frozen seeds)
   plus every frozen runtime value (drive model, acceleration, deceleration,
   stopping-distance envelope, command bounds, action contract), with duplicate and
   missing-cell rejection. It reads the validated preregistration through the #6100
   checker so the manifest cannot drift from the reviewed contract.
2. A read-only / check-only CLI (``--check-only``) that validates the manifest and
   prints the complete run plan with no episode launch, scheduler, remote, tmux, or
   process side effect (the only intentional output is the declared ``--manifest-out``
   file).
3. A bounded ``--preflight`` that exercises the real bicycle-drive / action binding
   end to end on disjoint seeds outside the registered 111-140 block, and reports the
   binary activation gate frozen by #6100.
4. A ``--synthesize`` adapter that feeds file-backed per-cell summaries directly into
   the reviewed #5578 synthesizer, so campaign rows connect to the reviewed
   synthesis path without modifying the frozen synthesizer contract.
5. A ``--full-run`` surface that is documented but **fails closed** here: registered
   execution belongs to the downstream campaign lane (#6102) and is not authorized
   in this issue.

Evidence boundary
-----------------
Completion of this module proves run readiness and intervention activation only.
It is NOT evidence of planner robustness, harm, safety, generalization, or ranking
stability. The preflight artifact states prominently:
``NOT BENCHMARK EVIDENCE -- DISJOINT-SEED ACTIVATION CHECK ONLY``.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from robot_sf.benchmark.issue_5578_speed_tier_synthesis import (
    DECLARED_PLANNERS,
    DECLARED_SCENARIOS,
    DECLARED_SEEDS,
    MIN_ACTIVATION_FRACTION_ABOVE_2_0,
    MIN_ACTIVATION_PEAK_SPEED,
    NOMINAL_TIER_ID,
    NON_NOMINAL_TIERS,
    TIER_ACTUATION_ENVELOPES,
    synthesize_speed_tier_sweep,
)
from robot_sf.robot.actuation_envelope import actuation_envelope_from_drive_config
from robot_sf.robot.bicycle_drive import BicycleDriveSettings
from scripts.benchmark.run_fidelity_sensitivity_campaign import (
    GoalSeekPlanner,
    _env_action,
    _robot_speed_cap,
)
from scripts.validation.check_issue_5578_robot_speed_tier_preregistration import (
    DEFAULT_CONFIG,
    load_preregistration,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
MANIFEST_SCHEMA_VERSION = "robot_sf.issue_5578_speed_tier_campaign_manifest.v1"
PREFLIGHT_SCHEMA_VERSION = "robot_sf.issue_5578_speed_tier_activation_preflight.v1"
ISSUE = 5578
PARENT_ISSUE = 5578
AMENDMENT_ISSUE = 6100
THIS_ISSUE = 6101
EXPECTED_CELL_COUNT = 2160
EXPECTED_SCENARIO_COUNT = 6
EXPECTED_TIER_COUNT = 3
EXPECTED_PLANNER_COUNT = 4
EXPECTED_SEED_COUNT = 30
HORIZON_STEPS = 600
DT_SECONDS = 0.1
# Registered seeds are frozen at 111-140 by the preregistration. The activation
# preflight MUST use only disjoint seeds outside that block so it never executes or
# modifies a registered row. These disjoint seeds are far from 111-140 and are
# documented in every preflight artifact.
PREFLIGHT_SEEDS = (211, 212, 213, 214)
PREFLIGHT_SCENARIO = "classic_merging_medium"
# A goal-saturating command is the canonical probe for a speed-cap intervention: it
# drives the robot toward its goal at the tier command cap, which is exactly what is
# required to prove the cap binds to the real drive model and is reachable. It is an
# intervention-mechanism probe, NOT planner-behaviour evidence.
PREFLIGHT_PLANNER = "goal_seek"
PREFLIGHT_STEPS = 120
CAMPAIGN_RUNTIME_PATH = REPO_ROOT / "scripts/benchmark/run_fidelity_sensitivity_campaign.py"
SYNTHESIZER_PATH = REPO_ROOT / "robot_sf/benchmark/issue_5578_speed_tier_synthesis.py"
EVIDENCE_BASE_DIR = REPO_ROOT / "docs/context/evidence/issue_5578_robot_speed_tier_sweep"
PREFLIGHT_EVIDENCE_DIR = EVIDENCE_BASE_DIR / "preflight"
NOT_EVIDENCE_BANNER = "NOT BENCHMARK EVIDENCE -- DISJOINT-SEED ACTIVATION CHECK ONLY"
CLAIM_BOUNDARY = (
    "Campaign-lane manifest compilation and disjoint-seed activation preflight only. "
    "This does not execute registered episodes, establish planner robustness, promote "
    "speed-effect claims, or edit paper/dissertation claims. The preflight proves the "
    "speed-cap intervention binds to the real runtime and measurably activates; it is "
    "explicitly not benchmark evidence."
)
FULL_RUN_BLOCKED_REASON = (
    "registered campaign execution belongs to the downstream campaign lane (#6102) "
    "and is not authorized in issue #6101; this surface is documented only"
)
# Default output locations documented for the campaign lane (NOT created here).
DEFAULT_RAW_ROOT = "output/issue_5578_robot_speed_tier_sweep/raw"
DEFAULT_CELL_SUMMARY_PATH = "output/issue_5578_robot_speed_tier_sweep/cell_summaries.jsonl"
DEFAULT_SYNTHESIS_PATH = "docs/context/evidence/issue_5578_robot_speed_tier_sweep/synthesis.json"


class CampaignManifestError(ValueError):
    """Raised when the compiled manifest drifts from the frozen #6100 contract."""


class PreflightActivationError(RuntimeError):
    """Raised when the activation preflight cannot run natively."""


class FullRunBlockedError(RuntimeError):
    """Raised when the documented full-run surface is invoked in this issue."""


@dataclass(frozen=True)
class SpeedTierRuntime:
    """One frozen speed tier with its full resolved runtime binding."""

    tier_id: str
    runtime_variant_key: str
    cap_m_s: float
    drive_model: str
    max_accel_m_s2: float
    max_decel_m_s2: float
    stopping_distance_envelope_m: float
    role: str
    planner_command_contract: Mapping[str, Any]
    environment_action_contract: Mapping[str, Any]
    resolved_actuation_envelope: Mapping[str, Any]


@dataclass(frozen=True)
class PlannerIdentity:
    """One frozen planner identity from the four-arm roster."""

    planner_id: str
    algorithm: str
    role: str
    config_path: str | None
    command_adapter_contract: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class ScenarioIdentity:
    """One frozen scenario from the six-row middle-band subset."""

    scenario_id: str
    source_path: str
    mechanism: str


@dataclass
class CampaignManifest:
    """The exact, auditable set of 2,160 registered campaign identities."""

    schema_version: str
    issue: int
    parent_issue: int
    amendment_issue: int
    this_issue: int
    study_id: str
    claim_boundary: str
    scenarios: list[ScenarioIdentity]
    speed_tiers: list[SpeedTierRuntime]
    planners: list[PlannerIdentity]
    seeds: list[int]
    horizon_steps: int
    dt_seconds: float
    expected_cell_count: int
    identities: list[dict[str, Any]]
    manifest_hash: str
    runtime_resolution: dict[str, Any]


def _git_head() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _git_status_short() -> list[str]:
    try:
        result = subprocess.run(
            ["git", "status", "--short"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return ["unknown"]
    if result.returncode != 0:
        return ["unknown"]
    return result.stdout.splitlines()


def _git_provenance() -> dict[str, Any]:
    status_short = _git_status_short()
    return {
        "git_head": _git_head(),
        "git_worktree_dirty": bool(status_short),
        "git_status_short": status_short,
    }


def _repo_rel(path: pathlib.Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CampaignManifestError(message)


def _hash_payload(payload: Any) -> str:
    import hashlib

    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _resolved_envelope(tier: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve the #4976 actuation envelope for a tier from its frozen values."""
    cap = float(tier["cap_m_s"])
    accel = float(tier["max_accel_m_s2"])
    decel = float(tier["max_decel_m_s2"])
    settings = BicycleDriveSettings(
        max_velocity=cap,
        max_accel=accel,
        max_decel=decel,
    )
    envelope = dict(actuation_envelope_from_drive_config(settings))
    expected_stopping = cap**2 / (2.0 * decel)
    _require(
        math.isclose(
            float(envelope["stopping_distance_envelope_m"]), expected_stopping, abs_tol=1e-9
        ),
        f"resolved stopping-distance envelope drift for {tier['tier_id']}",
    )
    _require(
        math.isclose(float(envelope["peak_forward_speed_m_s"]), cap, abs_tol=1e-9),
        f"resolved peak forward speed drift for {tier['tier_id']}",
    )
    _require(
        math.isclose(float(envelope["max_forward_accel_m_s2"]), accel, abs_tol=1e-9),
        f"resolved forward accel drift for {tier['tier_id']}",
    )
    _require(
        math.isclose(float(envelope["max_braking_decel_m_s2"]), decel, abs_tol=1e-9),
        f"resolved braking decel drift for {tier['tier_id']}",
    )
    return envelope


def _build_tier_runtime(tier: Mapping[str, Any]) -> SpeedTierRuntime:
    """Build a fully resolved runtime binding for one frozen tier."""
    resolved_envelope = _resolved_envelope(tier)
    return SpeedTierRuntime(
        tier_id=str(tier["tier_id"]),
        runtime_variant_key=str(tier["runtime_variant_key"]),
        cap_m_s=float(tier["cap_m_s"]),
        drive_model=str(tier["drive_model"]),
        max_accel_m_s2=float(tier["max_accel_m_s2"]),
        max_decel_m_s2=float(tier["max_decel_m_s2"]),
        stopping_distance_envelope_m=float(tier["stopping_distance_envelope_m"]),
        role=str(tier["role"]),
        planner_command_contract=dict(tier["planner_command_contract"]),
        environment_action_contract=dict(tier["environment_action_contract"]),
        resolved_actuation_envelope=resolved_envelope,
    )


def _build_planner_identity(arm: Mapping[str, Any]) -> PlannerIdentity:
    adapter = arm.get("command_adapter_contract")
    return PlannerIdentity(
        planner_id=str(arm["planner_id"]),
        algorithm=str(arm["algorithm"]),
        role=str(arm["role"]),
        config_path=arm.get("config_path"),
        command_adapter_contract=dict(adapter) if isinstance(adapter, Mapping) else None,
    )


def _build_scenario_identity(row: Mapping[str, Any]) -> ScenarioIdentity:
    return ScenarioIdentity(
        scenario_id=str(row["scenario_id"]),
        source_path=str(row["source_path"]),
        mechanism=str(row["mechanism"]),
    )


def compile_campaign_manifest(
    preregistration: Mapping[str, Any] | None = None,
    *,
    config_path: str | pathlib.Path = DEFAULT_CONFIG,
) -> CampaignManifest:
    """Compile and validate exactly 2,160 registered campaign identities.

    Reads the validated preregistration through the #6100 checker (so the manifest
    cannot drift from the reviewed contract) and materializes the full
    scenario x tier x planner x seed cross with every frozen runtime value. It
    rejects duplicate and missing cells and cross-checks each tier's resolved
    actuation envelope against the frozen #6100 values.

    Args:
        preregistration: An already-validated preregistration payload. When
            ``None`` the tracked config is loaded and validated via the checker.
        config_path: Repo-relative path recorded for provenance.

    Returns:
        The compiled ``CampaignManifest``.

    Raises:
        CampaignManifestError: If any frozen value, count, or identity drifts.
    """
    if preregistration is None:
        preregistration = load_preregistration(config_path)
    scenario_contract = preregistration["scenario_contract"]
    speed_axis = preregistration["robot_speed_axis"]
    roster = preregistration["planner_roster"]
    seed_policy = preregistration["seed_policy"]
    baseline = preregistration["baseline_protocol"]

    scenarios = [_build_scenario_identity(row) for row in scenario_contract["selected_scenarios"]]
    tiers = [_build_tier_runtime(tier) for tier in speed_axis["tiers"]]
    planners = [_build_planner_identity(arm) for arm in roster["arms"]]
    seeds = [int(seed) for seed in seed_policy["seeds"]]

    # Frozen-count contract: 6 x 3 x 4 x 30 = 2160.
    _require(len(scenarios) == EXPECTED_SCENARIO_COUNT, "scenario count drift")
    _require(len(tiers) == EXPECTED_TIER_COUNT, "speed tier count drift")
    _require(len(planners) == EXPECTED_PLANNER_COUNT, "planner count drift")
    _require(len(seeds) == EXPECTED_SEED_COUNT, "seed count drift")
    _require(
        {s.scenario_id for s in scenarios} == set(DECLARED_SCENARIOS),
        "scenario identities drifted from the frozen six-row subset",
    )
    _require(
        {p.planner_id for p in planners} == set(DECLARED_PLANNERS),
        "planner identities drifted from the frozen four-arm roster",
    )
    _require(set(seeds) == set(DECLARED_SEEDS), "seeds drifted from the frozen 111-140 block")
    _require(
        tuple(t.tier_id for t in tiers) == (NOMINAL_TIER_ID, *NON_NOMINAL_TIERS),
        "speed tier order drifted",
    )
    _require(int(baseline["horizon_steps"]) == HORIZON_STEPS, "horizon_steps drift")
    _require(float(baseline["dt_seconds"]) == DT_SECONDS, "dt_seconds drift")

    # Cross-check resolved envelopes against the frozen synthesizer contract.
    for tier in tiers:
        frozen = TIER_ACTUATION_ENVELOPES[tier.tier_id]
        resolved = tier.resolved_actuation_envelope
        for key, expected in frozen.items():
            actual = resolved.get(key)
            if isinstance(expected, str):
                _require(actual == expected, f"{tier.tier_id}.envelope.{key} drift")
            else:
                _require(
                    isinstance(actual, (int, float))
                    and not isinstance(actual, bool)
                    and math.isclose(float(actual), expected, abs_tol=1e-9),
                    f"{tier.tier_id}.envelope.{key} drift: expected {expected}, got {actual}",
                )

    identities: list[dict[str, Any]] = []
    for scenario in scenarios:
        for tier in tiers:
            for planner in planners:
                for seed in seeds:
                    identities.append(_build_identity(scenario, tier, planner, seed))

    _validate_identity_grid(identities, scenarios, tiers, planners, seeds)

    runtime_resolution = {
        "drive_models_by_tier": {t.tier_id: t.drive_model for t in tiers},
        "resolved_actuation_envelopes_by_tier": {
            t.tier_id: dict(t.resolved_actuation_envelope) for t in tiers
        },
        "command_bounds_by_tier": {t.tier_id: dict(t.planner_command_contract) for t in tiers},
        "action_contract_by_tier": {t.tier_id: dict(t.environment_action_contract) for t in tiers},
        "runtime_converter": "scripts/benchmark/run_fidelity_sensitivity_campaign.py::_env_action",
        "speed_cap_reader": "scripts/benchmark/run_fidelity_sensitivity_campaign.py::_robot_speed_cap",
        "angular_cap_reader": "scripts/benchmark/run_fidelity_sensitivity_campaign.py::_robot_angular_cap",
        "native_action_space": "robot_sf.robot.bicycle_drive.BicycleDriveRobot.action_space",
    }

    manifest = CampaignManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        issue=ISSUE,
        parent_issue=PARENT_ISSUE,
        amendment_issue=AMENDMENT_ISSUE,
        this_issue=THIS_ISSUE,
        study_id=str(preregistration["study_id"]),
        claim_boundary=CLAIM_BOUNDARY,
        scenarios=scenarios,
        speed_tiers=tiers,
        planners=planners,
        seeds=seeds,
        horizon_steps=HORIZON_STEPS,
        dt_seconds=DT_SECONDS,
        expected_cell_count=EXPECTED_CELL_COUNT,
        identities=identities,
        manifest_hash="",
        runtime_resolution=runtime_resolution,
    )
    # Hash is computed over the serializable identity set so it is stable and auditable.
    object.__setattr__(
        manifest,
        "manifest_hash",
        _hash_payload(
            {
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "study_id": manifest.study_id,
                "identities": manifest.identities,
            }
        ),
    )
    return manifest


def _build_identity(
    scenario: ScenarioIdentity,
    tier: SpeedTierRuntime,
    planner: PlannerIdentity,
    seed: int,
) -> dict[str, Any]:
    """Build one registered identity row with its full frozen runtime values."""
    identity_key = f"{scenario.scenario_id}__{tier.tier_id}__{planner.planner_id}__{seed}"
    return {
        "identity_key": identity_key,
        "scenario_id": scenario.scenario_id,
        "scenario_source_path": scenario.source_path,
        "scenario_mechanism": scenario.mechanism,
        "speed_tier_id": tier.tier_id,
        "speed_cap_m_s": tier.cap_m_s,
        "runtime_variant_key": tier.runtime_variant_key,
        "drive_model": tier.drive_model,
        "max_accel_m_s2": tier.max_accel_m_s2,
        "max_decel_m_s2": tier.max_decel_m_s2,
        "stopping_distance_envelope_m": tier.stopping_distance_envelope_m,
        "planner_command_contract": dict(tier.planner_command_contract),
        "environment_action_contract": dict(tier.environment_action_contract),
        "resolved_actuation_envelope": dict(tier.resolved_actuation_envelope),
        "planner_id": planner.planner_id,
        "planner_algorithm": planner.algorithm,
        "planner_role": planner.role,
        "planner_config_path": planner.config_path,
        "planner_command_adapter_contract": (
            dict(planner.command_adapter_contract)
            if planner.command_adapter_contract is not None
            else None
        ),
        "seed": seed,
        "horizon_steps": HORIZON_STEPS,
        "dt_seconds": DT_SECONDS,
        "execution_mode": "native",
        "resampling_unit": "paired_seed_block",
        "registered": True,
    }


def _validate_identity_grid(
    identities: Sequence[Mapping[str, Any]],
    scenarios: Sequence[ScenarioIdentity],
    tiers: Sequence[SpeedTierRuntime],
    planners: Sequence[PlannerIdentity],
    seeds: Sequence[int],
) -> None:
    """Reject duplicate and missing cells and assert the exact 2,160 count."""
    _require(len(identities) == EXPECTED_CELL_COUNT, "identity count is not 2160")
    seen: set[str] = set()
    for row in identities:
        key = str(row["identity_key"])
        _require(key not in seen, f"duplicate registered identity: {key}")
        seen.add(key)
    expected_keys = {
        f"{s.scenario_id}__{t.tier_id}__{p.planner_id}__{seed}"
        for s in scenarios
        for t in tiers
        for p in planners
        for seed in seeds
    }
    missing = expected_keys - seen
    extra = seen - expected_keys
    _require(not missing, f"missing registered identities: {sorted(missing)[:5]}")
    _require(not extra, f"unexpected identities: {sorted(extra)[:5]}")
    _require(seen == expected_keys, "identity grid does not match the frozen cross")
    # Every row must be native and registered (no fallback/degraded in the manifest).
    for row in identities:
        _require(row["execution_mode"] == "native", "manifest row execution_mode must be native")
        _require(row["registered"] is True, "manifest row must be registered")


def manifest_to_dict(manifest: CampaignManifest) -> dict[str, Any]:
    """Serialize a compiled manifest to a JSON-serializable mapping."""
    return {
        "schema_version": manifest.schema_version,
        "issue": manifest.issue,
        "parent_issue": manifest.parent_issue,
        "amendment_issue": manifest.amendment_issue,
        "this_issue": manifest.this_issue,
        "study_id": manifest.study_id,
        "claim_boundary": manifest.claim_boundary,
        "frozen_contract": {
            "scenarios": [asdict(s) for s in manifest.scenarios],
            "speed_tiers": [asdict(t) for t in manifest.speed_tiers],
            "planners": [asdict(p) for p in manifest.planners],
            "seeds": manifest.seeds,
            "horizon_steps": manifest.horizon_steps,
            "dt_seconds": manifest.dt_seconds,
            "expected_cell_count": manifest.expected_cell_count,
            "activation_rule": {
                "min_fraction_above_2_0_mps": MIN_ACTIVATION_FRACTION_ABOVE_2_0,
                "min_peak_speed_m_s": MIN_ACTIVATION_PEAK_SPEED,
                "rule": (
                    "For non-nominal tiers (3.0 and 4.0 m/s), an intervention is activated "
                    "if fraction_above_2_0_mps >= 0.05 OR realized_speed_peak_m_s > 2.2."
                ),
            },
        },
        "runtime_resolution": manifest.runtime_resolution,
        "identities": manifest.identities,
        "manifest_hash": manifest.manifest_hash,
    }


def write_manifest(manifest: CampaignManifest, path: str | pathlib.Path) -> pathlib.Path:
    """Write the compiled manifest as deterministic JSON to ``path``."""
    out = pathlib.Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = manifest_to_dict(manifest)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def evaluate_activation_rule(
    fraction_above_2_0_mps: float,
    realized_speed_peak_m_s: float,
    *,
    tier_id: str,
) -> dict[str, Any]:
    """Evaluate the binary activation gate frozen by #6100 for one tier.

    The nominal 2.0 m/s tier is the reference axis level, not a treated
    intervention, so it is always reported as ``not_applicable``. Non-nominal tiers
    are activated when the realized speed measurably exceeds the 2.0 m/s boundary.

    Returns:
        A JSON-serializable activation-gate record.
    """
    if tier_id == NOMINAL_TIER_ID:
        return {
            "tier_id": tier_id,
            "activated": True,
            "applicability": "nominal_reference_not_a_treated_intervention",
            "fraction_above_2_0_mps": float(fraction_above_2_0_mps),
            "realized_speed_peak_m_s": float(realized_speed_peak_m_s),
            "min_fraction_above_2_0_mps": MIN_ACTIVATION_FRACTION_ABOVE_2_0,
            "min_peak_speed_m_s": MIN_ACTIVATION_PEAK_SPEED,
        }
    activated = (
        float(fraction_above_2_0_mps) >= MIN_ACTIVATION_FRACTION_ABOVE_2_0
        or float(realized_speed_peak_m_s) > MIN_ACTIVATION_PEAK_SPEED
    )
    return {
        "tier_id": tier_id,
        "activated": bool(activated),
        "applicability": "treated_intervention",
        "fraction_above_2_0_mps": float(fraction_above_2_0_mps),
        "realized_speed_peak_m_s": float(realized_speed_peak_m_s),
        "fraction_above_2_0_mps_threshold": MIN_ACTIVATION_FRACTION_ABOVE_2_0,
        "realized_speed_peak_m_s_threshold": MIN_ACTIVATION_PEAK_SPEED,
        "fraction_above_2_0_mps_passes": float(fraction_above_2_0_mps)
        >= MIN_ACTIVATION_FRACTION_ABOVE_2_0,
        "peak_speed_passes": float(realized_speed_peak_m_s) > MIN_ACTIVATION_PEAK_SPEED,
        "rule": "fraction_above_2_0_mps >= 0.05 OR realized_speed_peak_m_s > 2.2",
    }


def _tier_variant_patch(tier: SpeedTierRuntime) -> dict[str, Any]:
    """Build the runtime variant patch that the canonical runner applies for a tier."""
    return {
        "type": tier.drive_model,
        "max_velocity": tier.cap_m_s,
        "max_accel": tier.max_accel_m_s2,
        "max_decel": tier.max_decel_m_s2,
    }


def _load_frozen_scenario(scenario_id: str) -> tuple[Mapping[str, Any], pathlib.Path]:
    """Load one frozen scenario definition from the tracked scenario manifest."""
    from robot_sf.training.scenario_loader import load_scenarios

    matrix_path = REPO_ROOT / "configs/scenarios/classic_interactions.yaml"
    scenarios = list(load_scenarios(matrix_path))
    for scenario in scenarios:
        if str(scenario.get("name")) == scenario_id:
            return scenario, matrix_path
    raise PreflightActivationError(f"frozen scenario not found: {scenario_id}")


def _build_env_for_tier(
    scenario: Mapping[str, Any],
    scenario_path: pathlib.Path,
    tier: SpeedTierRuntime,
    *,
    seed: int,
) -> tuple[Any, Any, float]:
    """Build a real Robot SF env with the tier's speed cap bound end to end.

    This is the real drive/action binding the preflight exercises: it applies the
    frozen tier variant through the canonical runner's ``apply_variant`` (which sets
    ``robot_config.drive_speed_cap``), constructs the real ``BicycleDriveRobot``, and
    reads back the resolved cap through the canonical ``_robot_speed_cap`` reader so
    the binding is proven, not asserted.
    """
    from robot_sf.gym_env.environment_factory import make_robot_env
    from robot_sf.training.scenario_loader import build_robot_config_from_scenario
    from scripts.benchmark.run_fidelity_sensitivity_campaign import (
        VariantSpec,
        apply_variant,
    )

    config = build_robot_config_from_scenario(scenario, scenario_path=scenario_path)
    variant = VariantSpec(
        axis="robot_speed_band",
        key=f"issue_5578_preflight_{tier.tier_id}",
        source_key=tier.runtime_variant_key,
        baseline=tier.tier_id == NOMINAL_TIER_ID,
        patch={"robot_config": _tier_variant_patch(tier)},
        observation_noise={},
        runtime_binding="robot_config.drive_speed_cap",
    )
    apply_variant(config, variant, seed=seed)
    resolved_cap = _robot_speed_cap(config.robot_config)
    if not math.isclose(resolved_cap, tier.cap_m_s, abs_tol=1e-9):
        raise PreflightActivationError(
            f"resolved speed cap {resolved_cap} does not match tier {tier.tier_id} cap "
            f"{tier.cap_m_s}; the drive/action binding did not reach the runtime"
        )
    env = make_robot_env(config=config, seed=seed, debug=False)
    planner = GoalSeekPlanner(
        max_linear_speed=resolved_cap,
        max_angular_speed=_angular_cap_for_tier(tier),
    )
    return env, planner, resolved_cap


def _angular_cap_for_tier(tier: SpeedTierRuntime) -> float:
    """Read the planner angular command bound for a tier from its frozen contract."""
    bounds = tier.planner_command_contract["angular_velocity_bounds_rad_s"]
    return float(bounds[1])


def _goal_saturating_command(planner: Any, env: Any) -> dict[str, float]:
    """Build a goal-seeking observation and return the planner's cap-bound command.

    The goal-saturating command drives the robot toward its goal at the tier command
    cap; it is the canonical probe for a speed-cap intervention and is not a claim
    about planner behaviour. It exercises the real bicycle-drive acceleration limits
    and the real ``_env_action`` converter.
    """
    from robot_sf.baselines.interface import Observation

    robot = env.simulator.robots[0]
    robot_pos = np.asarray(env.simulator.robot_pos[0], dtype=float)
    goal = np.asarray(env.simulator.goal_pos[0], dtype=float)
    heading = float(robot.pose[1])
    linear, _angular = robot.current_speed
    obs = Observation(
        dt=float(env.env_config.sim_config.time_per_step_in_secs),
        robot={
            "position": robot_pos.tolist(),
            "velocity": [float(linear) * math.cos(heading), float(linear) * math.sin(heading)],
            "goal": goal.tolist(),
            "heading": heading,
            "radius": float(robot.config.radius),
        },
        agents=[],
        obstacles=[],
    )
    return planner.step(obs)


@dataclass
class TierPreflightResult:
    """Activation diagnostics for one tier across the disjoint preflight seeds."""

    tier_id: str
    cap_m_s: float
    resolved_cap_m_s: float
    drive_model: str
    seeds: list[int]
    commanded_speed_mean_m_s: float
    realized_speed_mean_m_s: float
    realized_speed_peak_m_s: float
    fraction_above_2_0_mps: float
    cap_saturation_fraction: float
    resolved_actuation_envelope: dict[str, Any]
    steps_observed: int
    activation_gate: dict[str, Any]
    per_seed: list[dict[str, Any]]


def _run_tier_preflight(
    tier: SpeedTierRuntime,
    *,
    scenario_id: str,
    seeds: Sequence[int],
    steps: int,
) -> TierPreflightResult:
    """Run the activation preflight for one tier across disjoint seeds.

    Drives the real robot toward its goal at the tier command cap and records the
    realized speed, cap saturation, and fraction of steps above 2.0 m/s so the
    activation gate can be evaluated against the #6100 threshold.
    """
    scenario, scenario_path = _load_frozen_scenario(scenario_id)
    per_seed: list[dict[str, Any]] = []
    commanded_speeds: list[float] = []
    realized_speeds: list[float] = []
    realized_peaks: list[float] = []
    fractions_above: list[float] = []
    cap_saturations: list[float] = []
    total_steps = 0
    resolved_cap_ref = tier.cap_m_s
    for seed in seeds:
        env, planner, resolved_cap = _build_env_for_tier(scenario, scenario_path, tier, seed=seed)
        resolved_cap_ref = resolved_cap
        try:
            env.reset(seed=seed)
            robot = env.simulator.robots[0]
            seed_commanded: list[float] = []
            seed_realized: list[float] = []
            for _ in range(steps):
                command = _goal_saturating_command(planner, env)
                env.step(_env_action(env, command))
                linear, _angular = robot.current_speed
                seed_commanded.append(float(command.get("v", 0.0)))
                seed_realized.append(float(linear))
            seed_steps = len(seed_realized)
            total_steps += seed_steps
            seed_peak = max(seed_realized) if seed_realized else 0.0
            seed_frac_above = (
                sum(1 for s in seed_realized if s > 2.0) / seed_steps if seed_steps else 0.0
            )
            seed_cap_sat = (
                sum(1 for s in seed_realized if s >= resolved_cap - 1e-6) / seed_steps
                if seed_steps
                else 0.0
            )
            commanded_speeds.extend(seed_commanded)
            realized_speeds.extend(seed_realized)
            realized_peaks.append(seed_peak)
            fractions_above.append(seed_frac_above)
            cap_saturations.append(seed_cap_sat)
            per_seed.append(
                {
                    "seed": int(seed),
                    "commanded_speed_mean_m_s": float(np.mean(seed_commanded))
                    if seed_commanded
                    else 0.0,
                    "realized_speed_mean_m_s": float(np.mean(seed_realized))
                    if seed_realized
                    else 0.0,
                    "realized_speed_peak_m_s": seed_peak,
                    "fraction_above_2_0_mps": seed_frac_above,
                    "cap_saturation_fraction": seed_cap_sat,
                    "steps": seed_steps,
                    "registered_seed": False,
                }
            )
        finally:
            env.close()
    commanded_mean = float(np.mean(commanded_speeds)) if commanded_speeds else 0.0
    realized_mean = float(np.mean(realized_speeds)) if realized_speeds else 0.0
    realized_peak = max(realized_peaks) if realized_peaks else 0.0
    fraction_above = float(np.mean(fractions_above)) if fractions_above else 0.0
    cap_saturation = float(np.mean(cap_saturations)) if cap_saturations else 0.0
    activation_gate = evaluate_activation_rule(
        fraction_above,
        realized_peak,
        tier_id=tier.tier_id,
    )
    envelope = dict(tier.resolved_actuation_envelope)
    return TierPreflightResult(
        tier_id=tier.tier_id,
        cap_m_s=tier.cap_m_s,
        resolved_cap_m_s=resolved_cap_ref,
        drive_model=tier.drive_model,
        seeds=[int(seed) for seed in seeds],
        commanded_speed_mean_m_s=commanded_mean,
        realized_speed_mean_m_s=realized_mean,
        realized_speed_peak_m_s=realized_peak,
        fraction_above_2_0_mps=fraction_above,
        cap_saturation_fraction=cap_saturation,
        resolved_actuation_envelope=envelope,
        steps_observed=total_steps,
        activation_gate=activation_gate,
        per_seed=per_seed,
    )


def run_activation_preflight(
    manifest: CampaignManifest,
    *,
    seeds: Sequence[int] = PREFLIGHT_SEEDS,
    scenario_id: str = PREFLIGHT_SCENARIO,
    steps: int = PREFLIGHT_STEPS,
    git_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the bounded disjoint-seed activation preflight across all three tiers.

    The preflight uses only disjoint seeds outside the registered 111-140 block and
    exercises the real bicycle-drive / action binding end to end for all three speed
    tiers. It reports planned versus resolved cap/acceleration/deceleration values,
    commanded and realized speed summaries, cap-saturation and fraction-above-2.0
    summaries, native execution status, and the binary activation gate frozen by
    #6100.

    Raises:
        PreflightActivationError: If a disjoint seed overlaps the registered block,
            the chosen scenario is not a frozen scenario, or any tier cannot run
            natively.
    """
    registered = set(DECLARED_SEEDS)
    overlap = sorted(set(seeds) & registered)
    if overlap:
        raise PreflightActivationError(
            f"preflight seeds overlap the registered 111-140 block: {overlap}"
        )
    frozen_scenario_ids = {s.scenario_id for s in manifest.scenarios}
    if scenario_id not in frozen_scenario_ids:
        raise PreflightActivationError(
            f"preflight scenario {scenario_id!r} is not one of the frozen scenarios"
        )
    provenance = dict(git_provenance) if git_provenance is not None else _git_provenance()

    tier_results: list[TierPreflightResult] = []
    for tier in manifest.speed_tiers:
        tier_results.append(
            _run_tier_preflight(tier, scenario_id=scenario_id, seeds=seeds, steps=steps)
        )

    all_native = all(
        math.isclose(t.resolved_cap_m_s, t.cap_m_s, abs_tol=1e-9) for t in tier_results
    )
    planned_vs_resolved = [
        {
            "tier_id": t.tier_id,
            "planned_cap_m_s": t.cap_m_s,
            "resolved_cap_m_s": t.resolved_cap_m_s,
            "cap_matches": math.isclose(t.resolved_cap_m_s, t.cap_m_s, abs_tol=1e-9),
            "planned_max_accel_m_s2": next(
                ti.max_accel_m_s2 for ti in manifest.speed_tiers if ti.tier_id == t.tier_id
            ),
            "planned_max_decel_m_s2": next(
                ti.max_decel_m_s2 for ti in manifest.speed_tiers if ti.tier_id == t.tier_id
            ),
            "resolved_actuation_envelope": t.resolved_actuation_envelope,
        }
        for t in tier_results
    ]
    non_nominal_gates = [t.activation_gate for t in tier_results if t.tier_id in NON_NOMINAL_TIERS]
    all_non_nominal_activated = all(g["activated"] for g in non_nominal_gates)

    return {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "issue": ISSUE,
        "this_issue": THIS_ISSUE,
        "amendment_issue": AMENDMENT_ISSUE,
        "study_id": manifest.study_id,
        "not_evidence_banner": NOT_EVIDENCE_BANNER,
        "claim_boundary": (
            "Disjoint-seed activation preflight only. It proves the robot speed-cap "
            "intervention binds to the real runtime and measurably activates across all "
            "three tiers; it is explicitly NOT benchmark evidence and must not be used to "
            "tune harm thresholds, choose favourable scenarios/planners, or preview the "
            "registered primary-outcome verdict."
        ),
        "activation_probe": {
            "command_source": PREFLIGHT_PLANNER,
            "command_source_description": (
                "goal-saturating command toward the robot goal at the tier command cap; "
                "the canonical probe for a speed-cap intervention (mechanism check, not "
                "planner-behaviour evidence)"
            ),
            "scenario_id": scenario_id,
            "seeds": [int(seed) for seed in seeds],
            "seeds_disjoint_from_registered_111_140": True,
            "steps_per_seed": steps,
            "registered_seed_overlap": [],
        },
        "git_provenance": provenance,
        "command_environment_manifest": {
            "runtime_converter": "scripts/benchmark/run_fidelity_sensitivity_campaign.py::_env_action",
            "speed_cap_reader": "scripts/benchmark/run_fidelity_sensitivity_campaign.py::_robot_speed_cap",
            "angular_cap_reader": "scripts/benchmark/run_fidelity_sensitivity_campaign.py::_robot_angular_cap",
            "variant_applier": "scripts/benchmark/run_fidelity_sensitivity_campaign.py::apply_variant",
            "env_factory": "robot_sf.gym_env.environment_factory.make_robot_env",
            "scenario_loader": "robot_sf.training.scenario_loader.build_robot_config_from_scenario",
            "actuation_envelope": "robot_sf.robot.actuation_envelope.actuation_envelope_from_drive_config",
            "native_action_space": "robot_sf.robot.bicycle_drive.BicycleDriveRobot.action_space",
        },
        "activation_rule": {
            "min_fraction_above_2_0_mps": MIN_ACTIVATION_FRACTION_ABOVE_2_0,
            "min_peak_speed_m_s": MIN_ACTIVATION_PEAK_SPEED,
            "rule": "fraction_above_2_0_mps >= 0.05 OR realized_speed_peak_m_s > 2.2",
        },
        "planned_vs_resolved": planned_vs_resolved,
        "tier_results": [asdict(t) for t in tier_results],
        "execution_status": {
            "native": all_native,
            "fallback": False,
            "degraded": False,
            "all_tiers_native": all_native,
        },
        "activation_gate_summary": {
            "all_non_nominal_tiers_activated": all_non_nominal_activated,
            "per_tier": {t.tier_id: t.activation_gate for t in tier_results},
        },
        "preflight_passed": bool(all_native and all_non_nominal_activated),
    }


def write_preflight_artifact(
    preflight: Mapping[str, Any], output_dir: str | pathlib.Path
) -> pathlib.Path:
    """Write the preflight artifact as deterministic JSON.

    The default location is the tracked, explicitly non-evidence preflight evidence
    directory so the activation record is durable and discoverable for review.
    """
    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    artifact_path = out / "issue_5578_activation_preflight.json"
    artifact_path.write_text(
        json.dumps(preflight, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return artifact_path


def synthesize_from_cell_summaries(
    cell_summaries_path: str | pathlib.Path,
    *,
    output_path: str | pathlib.Path | None = None,
    declared_scenarios: set[str] | None = None,
    declared_planners: set[str] | None = None,
    declared_seeds: set[int] | None = None,
) -> dict[str, Any]:
    """Feed file-backed per-cell summaries into the reviewed #5578 synthesizer.

    This is the deterministic checked adapter that connects campaign rows to the
    reviewed synthesis path without modifying the frozen synthesizer contract. It
    reads one JSON object per line (JSONL) or a single JSON array, validates each row
    through the synthesizer's fail-closed parser, and writes the synthesis result when
    an ``output_path`` is given.

    By default the adapter asserts the frozen full dimensions (6 scenarios x 4 planners
    x 30 seeds), so a real campaign row file is checked against the exact registered
    grid. The optional declared-dimension overrides exist only for adapter smoke
    checks and reduce the result to ``smoke_or_incomplete_not_benchmark_evidence``.

    Returns:
        The synthesizer report mapping.

    Raises:
        ValueError: If the file is empty or the synthesis fails closed.
    """
    path = pathlib.Path(cell_summaries_path)
    if not path.is_file():
        raise ValueError(f"cell summaries file not found: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"cell summaries file is empty: {path}")
    rows: list[dict[str, Any]]
    if text[0] == "[":
        rows = json.loads(text)
    else:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"no cell summaries parsed from {path}")
    result = synthesize_speed_tier_sweep(
        rows,
        declared_scenarios=declared_scenarios,
        declared_planners=declared_planners,
        declared_seeds=declared_seeds,
    )
    report = {
        "schema_version": "robot_sf.issue_5578_speed_tier_synthesis_adapter.v1",
        "issue": ISSUE,
        "claim_boundary": result.claim_boundary,
        "per_cell_count": result.per_cell_count,
        "native_cell_count": result.native_cell_count,
        "excluded_cell_count": result.excluded_cell_count,
        "all_native": result.all_native,
        "grid_complete": result.grid_complete,
        "evidence_status": result.evidence_status,
        "decision_table": result.decision_table,
        "descriptive_ranking_stability": result.descriptive_ranking_stability,
        "exclusions": result.exclusions,
        "source_path": _repo_rel(path),
    }
    if output_path is not None:
        out = pathlib.Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _full_run_documentation(
    cell_summaries_path: str | pathlib.Path | None = None,
) -> dict[str, Any]:
    """Document the full-run command and expected output/provenance locations.

    The full-run command is intentionally NOT executable here: registered execution
    belongs to the downstream campaign lane (#6102) and is not authorized in this
    issue. This documentation exists so the campaign lane has the exact surfaces.
    """
    cell_summaries = pathlib.Path(cell_summaries_path or DEFAULT_CELL_SUMMARY_PATH)
    return {
        "full_run_status": "documented_not_authorized_in_this_issue",
        "blocked_reason": FULL_RUN_BLOCKED_REASON,
        "documented_command": (
            "uv run python scripts/benchmark/run_issue_5578_speed_tier_campaign.py --full-run "
            f"--cell-summaries-out {cell_summaries}"
        ),
        "expected_output_locations": {
            "raw_episode_jsonl": DEFAULT_RAW_ROOT,
            "cell_summaries": str(cell_summaries),
            "synthesis": DEFAULT_SYNTHESIS_PATH,
        },
        "expected_output_contract": (
            "each per-cell summary row MUST conform to the synthesizer's required_cell_keys "
            "(scenario_id, speed_tier_id, speed_cap_m_s, planner_id, seed, horizon_steps, "
            "dt_seconds, execution_mode, primary metrics, typed collisions, activation "
            "diagnostics, exposure diagnostics); feed via --synthesize."
        ),
        "provenance_requirements": [
            "public_git_sha_and_clean_dirty_state",
            "exact_command_and_environment_manifest",
            "per_episode_jsonl_and_aggregate_summary",
            "typed_collision_breakdown_and_denominator_table",
            "native_execution_mode_for_every_claimed_row",
            "exclusion_table_for_missing_failed_fallback_or_degraded_rows",
        ],
        "registered_seed_guard": (
            "registered seeds 111-140 may only be executed by the authorized campaign lane; "
            "they must not run in this issue."
        ),
    }


def _print_check_only_summary(manifest: CampaignManifest, manifest_path: pathlib.Path) -> None:
    """Print a compact human-readable run-plan summary for check-only mode."""
    top = manifest.speed_tiers[-1]
    print("PASS: issue #5578 campaign manifest compiled (check-only, no side effects).")
    print(
        f"  identities: {len(manifest.identities)} "
        f"({len(manifest.scenarios)} scenarios x {len(manifest.speed_tiers)} tiers x "
        f"{len(manifest.planners)} planners x {len(manifest.seeds)} seeds)"
    )
    print(f"  manifest_hash: {manifest.manifest_hash}")
    print(
        f"  manifest_out: {_repo_rel(manifest_path) if manifest_path.is_absolute() else manifest_path}"
    )
    print(
        f"  top tier: {top.tier_id} cap={top.cap_m_s} m/s variant={top.runtime_variant_key} "
        f"(4.2 m/s amended to supported 4.0 m/s by #6100)"
    )
    print(f"  drive models: { {t.tier_id: t.drive_model for t in manifest.speed_tiers} }")
    print("  side_effects: none (no episode launch, scheduler, remote, tmux, or process spawn)")


def _run_check_only(args: argparse.Namespace) -> int:
    """Compile and validate the manifest with no execution side effects."""
    manifest = compile_campaign_manifest(config_path=args.config)
    manifest_path: pathlib.Path | None = None
    if args.manifest_out is not None:
        manifest_path = write_manifest(manifest, args.manifest_out)
    if args.json:
        payload = manifest_to_dict(manifest)
        payload["manifest_out"] = str(manifest_path) if manifest_path is not None else None
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif manifest_path is None:
        print(
            "PASS: issue #5578 campaign manifest compiled "
            f"({len(manifest.identities)} identities, hash={manifest.manifest_hash})"
        )
    else:
        _print_check_only_summary(manifest, manifest_path)
    return 0


def _run_preflight(args: argparse.Namespace) -> int:
    """Run the bounded disjoint-seed activation preflight."""
    manifest = compile_campaign_manifest(config_path=args.config)
    preflight = run_activation_preflight(
        manifest,
        seeds=tuple(args.preflight_seeds),
        scenario_id=args.preflight_scenario,
        steps=args.preflight_steps,
    )
    artifact_path = write_preflight_artifact(preflight, args.preflight_out)
    if args.json:
        payload = dict(preflight)
        payload["artifact_path"] = _repo_rel(artifact_path)
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        status = "PASS" if preflight["preflight_passed"] else "FAIL"
        print(f"{status}: issue #5578 activation preflight ({NOT_EVIDENCE_BANNER}).")
        print(f"  artifact: {_repo_rel(artifact_path)}")
        for tier in preflight["tier_results"]:
            gate = tier["activation_gate"]
            print(
                f"  tier {tier['tier_id']}: cap={tier['cap_m_s']} "
                f"resolved={tier['resolved_cap_m_s']} "
                f"peak={tier['realized_speed_peak_m_s']:.3f} "
                f"frac>2.0={tier['fraction_above_2_0_mps']:.3f} "
                f"activated={gate['activated']}"
            )
    return 0 if preflight["preflight_passed"] else 1


def _run_full_run(args: argparse.Namespace) -> int:
    """Document the full-run surface and fail closed (not authorized here)."""
    doc = _full_run_documentation(args.cell_summaries_out)
    if args.json:
        print(json.dumps(doc, indent=2, sort_keys=True))
    else:
        print("BLOCKED: issue #5578 full-run is documented but not authorized in this issue.")
        print(f"  reason: {FULL_RUN_BLOCKED_REASON}")
        print(f"  documented_command: {doc['documented_command']}")
        print(f"  expected_outputs: {json.dumps(doc['expected_output_locations'])}")
    raise FullRunBlockedError(FULL_RUN_BLOCKED_REASON)


def _run_synthesize(args: argparse.Namespace) -> int:
    """Feed file-backed per-cell summaries through the reviewed adapter."""
    smoke_scenarios = {args.smoke_declared_scenario} if args.smoke_declared_scenario else None
    smoke_planners = {args.smoke_declared_planner} if args.smoke_declared_planner else None
    smoke_seeds = {args.smoke_declared_seed} if args.smoke_declared_seed is not None else None
    report = synthesize_from_cell_summaries(
        args.synthesize,
        output_path=args.synthesis_out,
        declared_scenarios=smoke_scenarios,
        declared_planners=smoke_planners,
        declared_seeds=smoke_seeds,
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(
            f"PASS: issue #5578 synthesis adapter "
            f"({report['native_cell_count']} native cells, "
            f"evidence_status={report['evidence_status']})"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the issue #5578 campaign manifest / preflight CLI."""
    args = _parse_args(argv)
    if args.check_only:
        return _run_check_only(args)
    if args.preflight:
        return _run_preflight(args)
    if args.full_run:
        return _run_full_run(args)
    if args.synthesize:
        return _run_synthesize(args)
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse the mutually-exclusive mode CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        default=DEFAULT_CONFIG,
        help="Path to the issue #5578 preregistration YAML.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--check-only",
        action="store_true",
        help="Compile and validate the manifest with no execution side effects.",
    )
    mode.add_argument(
        "--preflight",
        action="store_true",
        help="Run the bounded disjoint-seed activation preflight (not benchmark evidence).",
    )
    mode.add_argument(
        "--full-run",
        action="store_true",
        help="Documented only; registered execution is not authorized in this issue.",
    )
    parser.add_argument(
        "--cell-summaries-out",
        type=pathlib.Path,
        help="Cell-summary output path recorded by the documented full-run handoff.",
    )
    mode.add_argument(
        "--synthesize",
        type=pathlib.Path,
        help="Synthesize file-backed per-cell summaries through the reviewed adapter.",
    )
    parser.add_argument(
        "--manifest-out",
        type=pathlib.Path,
        help="Write the compiled manifest JSON to this path (check-only).",
    )
    parser.add_argument(
        "--preflight-out",
        type=pathlib.Path,
        default=PREFLIGHT_EVIDENCE_DIR,
        help="Directory for the preflight artifact.",
    )
    parser.add_argument(
        "--preflight-scenario",
        type=str,
        default=PREFLIGHT_SCENARIO,
        help="Frozen scenario to exercise in the preflight.",
    )
    parser.add_argument(
        "--preflight-seeds",
        type=int,
        nargs="+",
        default=list(PREFLIGHT_SEEDS),
        help="Disjoint preflight seeds (must be outside 111-140).",
    )
    parser.add_argument(
        "--preflight-steps",
        type=int,
        default=PREFLIGHT_STEPS,
        help="Steps per preflight seed.",
    )
    parser.add_argument(
        "--synthesis-out",
        type=pathlib.Path,
        help="Write the synthesis report to this path (--synthesize).",
    )
    parser.add_argument(
        "--smoke-declared-scenario",
        type=str,
        help=(
            "Adapter smoke check only: restrict the declared scenario dimension "
            "(reduces result to smoke, not benchmark evidence)."
        ),
    )
    parser.add_argument(
        "--smoke-declared-planner",
        type=str,
        help="Adapter smoke check only: restrict the declared planner dimension.",
    )
    parser.add_argument(
        "--smoke-declared-seed",
        type=int,
        help="Adapter smoke check only: restrict the declared seed dimension.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
