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
6. A narrow ``--authorized-full-run`` surface that requires the explicit
   ``--authorization-issue 6102`` flag, invokes the canonical native map runner for
   the exact manifest, rejects fallback/degraded rows, and feeds complete native
   output into the reviewed synthesis adapter.

Evidence boundary
-----------------
Completion of this module proves run readiness and intervention activation only.
It is NOT evidence of planner robustness, harm, safety, generalization, or ranking
stability. The preflight artifact states prominently:
``NOT BENCHMARK EVIDENCE -- DISJOINT-SEED ACTIVATION CHECK ONLY``.
"""

# evidence-writer-exempt: Existing writers unchanged; separate migration preserves output contracts.
from __future__ import annotations

import argparse
import copy
import datetime
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
AUTHORIZED_EXECUTION_ISSUE = 6102
AUTHORIZED_EXECUTION_SCHEMA_VERSION = "robot_sf.issue_5578_authorized_campaign_execution.v1"
ROW_PROVENANCE_SCHEMA_VERSION = "benchmark_row_provenance.v1"
VALID_PLANNER_EXECUTION_MODES = frozenset({"native", "native_command", "adapter", "mixed"})
CAMPAIGN_SCENARIO_MATRIX = REPO_ROOT / "configs/scenarios/classic_interactions.yaml"
EPISODE_SCHEMA_PATH = REPO_ROOT / "robot_sf/benchmark/schemas/episode.schema.v1.json"
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


class CampaignAuthorizationError(RuntimeError):
    """Raised when the authorized campaign mode is not explicitly authorized."""


class AuthorizedCampaignError(RuntimeError):
    """Raised when an authorized campaign cannot produce a complete native grid."""


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


def _require_clean_execution_provenance(provenance: Mapping[str, Any]) -> str:
    """Return a usable execution commit or reject an unreproducible campaign launch."""
    git_head = provenance.get("git_head")
    if not _is_hex(git_head, length=40):
        raise AuthorizedCampaignError("authorized campaign requires a known 40-character git HEAD")
    if provenance.get("git_worktree_dirty") is not False:
        raise AuthorizedCampaignError(
            "authorized campaign requires a clean git worktree before registered execution"
        )
    return str(git_head)


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


def _utc_timestamp() -> str:
    """Return a stable UTC timestamp for operational provenance."""
    return datetime.datetime.now(datetime.UTC).isoformat().replace("+00:00", "Z")


def _read_jsonl_records(path: pathlib.Path) -> list[dict[str, Any]]:
    """Read a JSONL episode artifact and reject malformed rows."""
    if not path.is_file():
        raise AuthorizedCampaignError(f"raw episode artifact not found: {path}")
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AuthorizedCampaignError(
                    f"invalid JSONL in {path} at line {line_number}: {exc}"
                ) from exc
            if not isinstance(payload, Mapping):
                raise AuthorizedCampaignError(
                    f"episode row in {path} at line {line_number} is not an object"
                )
            records.append(dict(payload))
    return records


def _write_json_object(path: pathlib.Path, payload: Mapping[str, Any]) -> None:
    """Write one deterministic JSON object artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _finite_episode_number(value: Any, field_name: str) -> float:
    """Coerce one JSON episode value to a finite float."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AuthorizedCampaignError(f"episode field {field_name} must be numeric, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise AuthorizedCampaignError(f"episode field {field_name} must be finite, got {value!r}")
    return number


def _episode_metric(metrics: Mapping[str, Any], *names: str) -> float:
    """Read the first present finite metric from a map-runner metrics payload."""
    for name in names:
        if name in metrics:
            return _finite_episode_number(metrics[name], f"metrics.{name}")
    raise AuthorizedCampaignError(
        "episode metrics missing required field; tried " + ", ".join(names)
    )


def _binary_metric(value: Any, field_name: str) -> float:
    """Convert an event count/flag to the per-episode rate contract."""
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    number = _finite_episode_number(value, field_name)
    return 1.0 if number > 0.0 else 0.0


def _is_hex(value: Any, *, length: int) -> bool:
    """Return whether ``value`` is a lower/upper-case hexadecimal identifier."""
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


def _episode_integer(value: Any, field_name: str) -> int:
    """Read one integer-valued episode field without silently truncating it."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise AuthorizedCampaignError(
            f"episode field {field_name} must be an integer, got {value!r}"
        )
    return int(value)


def _validate_episode_provenance(  # noqa: C901
    record: Mapping[str, Any],
    *,
    scenario_id: str,
    seed: int,
    horizon_steps: int,
    dt_seconds: float,
    expected_repo_commit: str,
) -> str:
    """Validate the row provenance required before a cell can become campaign input."""
    provenance = record.get("result_provenance")
    if not isinstance(provenance, Mapping):
        raise AuthorizedCampaignError("raw episode missing result_provenance")
    if provenance.get("schema_version") != ROW_PROVENANCE_SCHEMA_VERSION:
        raise AuthorizedCampaignError(
            "raw episode result_provenance.schema_version is missing or unsupported"
        )
    if str(provenance.get("scenario_id", "")).strip() != scenario_id:
        raise AuthorizedCampaignError("raw episode result_provenance scenario mismatch")
    if _episode_integer(provenance.get("seed"), "result_provenance.seed") != int(seed):
        raise AuthorizedCampaignError("raw episode result_provenance seed mismatch")

    config_hash = provenance.get("config_hash")
    if not _is_hex(config_hash, length=16):
        raise AuthorizedCampaignError(
            "raw episode result_provenance.config_hash must be a 16-character hexadecimal hash"
        )
    record_config_hash = record.get("config_hash")
    if record_config_hash is not None and record_config_hash != config_hash:
        raise AuthorizedCampaignError("raw episode config_hash disagrees with result_provenance")

    repo_commit = provenance.get("repo_commit")
    if not _is_hex(repo_commit, length=40):
        raise AuthorizedCampaignError(
            "raw episode result_provenance.repo_commit must be a 40-character git SHA"
        )
    if not _is_hex(expected_repo_commit, length=40):
        raise AuthorizedCampaignError("authorized campaign has no valid execution commit")
    if str(repo_commit).lower() != expected_repo_commit.lower():
        raise AuthorizedCampaignError(
            "raw episode result_provenance.repo_commit does not match the authorized execution "
            "commit"
        )
    record_git_hash = record.get("git_hash")
    if record_git_hash is not None and record_git_hash != repo_commit:
        raise AuthorizedCampaignError("raw episode git_hash disagrees with result_provenance")

    simulator_settings = provenance.get("simulator_settings")
    if not isinstance(simulator_settings, Mapping):
        raise AuthorizedCampaignError("raw episode result_provenance.simulator_settings is missing")
    if _episode_integer(
        simulator_settings.get("horizon"),
        "result_provenance.simulator_settings.horizon",
    ) != int(horizon_steps):
        raise AuthorizedCampaignError("raw episode result_provenance horizon mismatch")
    provenance_dt = _finite_episode_number(
        simulator_settings.get("dt"),
        "result_provenance.simulator_settings.dt",
    )
    if not math.isclose(provenance_dt, dt_seconds, abs_tol=1e-9):
        raise AuthorizedCampaignError(
            f"raw episode result_provenance dt drift: expected {dt_seconds}, got {provenance_dt}"
        )

    postprocessing = provenance.get("postprocessing")
    if not isinstance(postprocessing, list):
        raise AuthorizedCampaignError("raw episode result_provenance.postprocessing is missing")
    completed_steps = {
        str(step.get("step"))
        for step in postprocessing
        if isinstance(step, Mapping) and step.get("status") == "completed"
    }
    if {"compute_all_metrics", "post_process_metrics"} - completed_steps:
        raise AuthorizedCampaignError("raw episode result_provenance.postprocessing is incomplete")
    return str(repo_commit)


def _campaign_execution_disposition(  # noqa: C901, PLR0912
    record: Mapping[str, Any],
) -> tuple[str, str | None]:
    """Classify planner runtime availability without confusing adapters with fallback.

    The preregistered roster explicitly allows planner command adapters (for example
    ORCA's world-velocity-to-unicycle adapter). Those are still native benchmark rows.
    Only an explicit fallback/degraded status, fallback counter, or ineligible predictive
    foresight state excludes a row from native campaign evidence.
    """
    metadata = record.get("algorithm_metadata")
    if not isinstance(metadata, Mapping):
        return "failed", "missing algorithm_metadata"

    status = str(metadata.get("status", "")).strip().lower()
    degraded_prefixes = (
        "fallback",
        "degraded",
        "policy_step_error_fallback",
        "policy_step_timeout_fallback",
        "predictive_foresight_model_fallback",
    )
    if status.startswith(degraded_prefixes):
        return "degraded", f"algorithm_metadata.status={status}"

    timeout_metadata = metadata.get("policy_step_timeout")
    if timeout_metadata is not None and not isinstance(timeout_metadata, Mapping):
        return "failed", "malformed algorithm_metadata.policy_step_timeout"
    if isinstance(timeout_metadata, Mapping):
        fallback_actions = timeout_metadata.get("fallback_actions")
        if fallback_actions is not None:
            try:
                fallback_count = float(fallback_actions)
                if not math.isfinite(fallback_count) or fallback_count < 0.0:
                    return "failed", "malformed policy_step_timeout.fallback_actions"
                if fallback_count > 0.0:
                    return "degraded", "policy_step_timeout.fallback_actions>0"
            except (TypeError, ValueError):
                return "failed", "malformed policy_step_timeout.fallback_actions"

    fallback_marker = metadata.get("fallback_or_degraded")
    if fallback_marker is True:
        return "degraded", "algorithm_metadata.fallback_or_degraded=true"
    if fallback_marker not in (None, False):
        return "failed", "malformed algorithm_metadata.fallback_or_degraded"

    foresight = metadata.get("foresight_prediction")
    if foresight is not None and not isinstance(foresight, Mapping):
        return "failed", "malformed algorithm_metadata.foresight_prediction"
    if isinstance(foresight, Mapping):
        fallback_used = foresight.get("fallback_used")
        if fallback_used is True:
            return "degraded", "foresight_prediction.fallback_used=true"
        if fallback_used not in (None, False):
            return "failed", "malformed foresight_prediction.fallback_used"
        evidence_eligible = foresight.get("evidence_eligible")
        if evidence_eligible is False:
            return "degraded", "foresight_prediction.evidence_eligible=false"
        if evidence_eligible not in (None, True):
            return "failed", "malformed foresight_prediction.evidence_eligible"

    planner_kinematics = metadata.get("planner_kinematics")
    if not isinstance(planner_kinematics, Mapping):
        return "failed", "missing algorithm_metadata.planner_kinematics"
    planner_mode = str(planner_kinematics.get("execution_mode", "")).strip().lower()
    if planner_mode in {"fallback", "degraded"}:
        return "degraded", f"planner_kinematics.execution_mode={planner_mode}"
    if planner_mode not in VALID_PLANNER_EXECUTION_MODES:
        return "failed", f"unsupported planner_kinematics.execution_mode={planner_mode}"

    if status == "ok":
        return "native", None
    if not status:
        return "failed", "algorithm_metadata.status is missing"
    return "failed", f"unsupported algorithm_metadata.status={status}"


def _trace_speed_diagnostics(  # noqa: C901
    record: Mapping[str, Any],
    *,
    cap_m_s: float,
    expected_dt: float,
) -> dict[str, float]:
    """Derive the preregistered speed diagnostics from the retained native trace."""
    metadata = record.get("algorithm_metadata")
    trace = metadata.get("simulation_step_trace") if isinstance(metadata, Mapping) else None
    if not isinstance(trace, Mapping) or not isinstance(trace.get("steps"), list):
        raise AuthorizedCampaignError("episode is missing simulation_step_trace.steps")
    steps = trace["steps"]
    if not steps:
        raise AuthorizedCampaignError("episode simulation_step_trace.steps is empty")
    trace_dt = _finite_episode_number(trace.get("dt"), "simulation_step_trace.dt")
    if not math.isclose(trace_dt, expected_dt, abs_tol=1e-9):
        raise AuthorizedCampaignError(
            f"episode trace dt drift: expected {expected_dt}, got {trace_dt}"
        )

    commanded: list[float] = []
    realized: list[float] = []
    for index, step in enumerate(steps):
        if not isinstance(step, Mapping):
            raise AuthorizedCampaignError(f"simulation trace step {index} is not an object")
        planner = step.get("planner")
        planner = planner if isinstance(planner, Mapping) else {}
        action = planner.get("selected_action")
        action = action if isinstance(action, Mapping) else {}
        command = action.get("linear_velocity", action.get("v"))
        if command is None:
            amv = planner.get("amv")
            if isinstance(amv, Mapping):
                command = amv.get("requested_linear_m_s")
        commanded.append(_finite_episode_number(command, f"trace[{index}].planner.linear_velocity"))

        robot = step.get("robot")
        robot = robot if isinstance(robot, Mapping) else {}
        velocity = robot.get("velocity")
        if isinstance(velocity, (list, tuple)) and len(velocity) >= 2:
            vx = _finite_episode_number(velocity[0], f"trace[{index}].robot.velocity[0]")
            vy = _finite_episode_number(velocity[1], f"trace[{index}].robot.velocity[1]")
            realized.append(math.hypot(vx, vy))
        else:
            realized.append(_finite_episode_number(velocity, f"trace[{index}].robot.velocity"))

    count = len(realized)
    commanded_mean = sum(commanded) / len(commanded)
    realized_peak = max(realized)
    if commanded_mean > cap_m_s + 1e-9:
        raise AuthorizedCampaignError(
            f"commanded speed mean exceeds tier cap: cap={cap_m_s}, mean={commanded_mean}"
        )
    if realized_peak > cap_m_s + 1e-9:
        raise AuthorizedCampaignError(
            f"realized speed peak exceeds tier cap: cap={cap_m_s}, peak={realized_peak}"
        )
    return {
        "commanded_speed_mean_m_s": commanded_mean,
        "realized_speed_mean_m_s": sum(realized) / count,
        "realized_speed_peak_m_s": realized_peak,
        "fraction_above_2_0_mps": sum(speed > 2.0 for speed in realized) / count,
        "cap_saturation_fraction": sum(speed >= cap_m_s - 1e-6 for speed in realized) / count,
    }


def _cell_summary_from_episode(
    record: Mapping[str, Any],
    *,
    scenario_id: str,
    tier: SpeedTierRuntime,
    planner: PlannerIdentity,
    seed: int,
    horizon_steps: int,
    dt_seconds: float,
    expected_repo_commit: str,
) -> dict[str, Any]:
    """Convert one canonical map-runner episode into the reviewed cell contract."""
    if str(record.get("scenario_id")) != scenario_id:
        raise AuthorizedCampaignError(
            f"raw episode scenario mismatch: expected {scenario_id}, got {record.get('scenario_id')!r}"
        )
    if _episode_integer(record.get("seed"), "seed") != int(seed):
        raise AuthorizedCampaignError(
            f"raw episode seed mismatch: expected {seed}, got {record.get('seed')!r}"
        )
    if _episode_integer(record.get("horizon"), "horizon") != horizon_steps:
        raise AuthorizedCampaignError(
            f"raw episode horizon mismatch: expected {horizon_steps}, got {record.get('horizon')!r}"
        )
    episode_id = record.get("episode_id")
    if not isinstance(episode_id, str) or not episode_id.strip():
        raise AuthorizedCampaignError("raw episode is missing a non-empty episode_id")
    raw_repo_commit = _validate_episode_provenance(
        record,
        scenario_id=scenario_id,
        seed=seed,
        horizon_steps=horizon_steps,
        dt_seconds=dt_seconds,
        expected_repo_commit=expected_repo_commit,
    )

    metadata = record.get("algorithm_metadata")
    if not isinstance(metadata, Mapping):
        raise AuthorizedCampaignError("raw episode missing algorithm_metadata")
    canonical_algorithm = str(metadata.get("canonical_algorithm", "")).strip()
    if canonical_algorithm != planner.algorithm:
        raise AuthorizedCampaignError(
            f"raw episode planner mismatch: expected {planner.algorithm}, got {canonical_algorithm}"
        )
    disposition, disposition_reason = _campaign_execution_disposition(record)
    speed = _trace_speed_diagnostics(
        record,
        cap_m_s=tier.cap_m_s,
        expected_dt=dt_seconds,
    )
    metrics = record.get("metrics")
    if not isinstance(metrics, Mapping):
        raise AuthorizedCampaignError("raw episode missing metrics")

    total_collision_count = max(
        _episode_metric(metrics, "total_collision_count", "collisions"),
        _episode_metric(metrics, "collisions", "total_collision_count"),
    )
    ped_collision_count = _episode_metric(metrics, "ped_collision_count")
    obstacle_collision_count = _episode_metric(metrics, "obstacle_collision_count")
    agent_collision_count = _episode_metric(metrics, "agent_collision_count")
    typed_count = ped_collision_count + obstacle_collision_count + agent_collision_count
    exposure = record.get("interaction_exposure")
    if not isinstance(exposure, Mapping):
        raise AuthorizedCampaignError("raw episode missing interaction_exposure")
    exposure_share = _finite_episode_number(
        exposure.get("interaction_exposure_share"),
        "interaction_exposure.interaction_exposure_share",
    )
    exposure_steps = _finite_episode_number(
        exposure.get("interaction_exposure_denominator_steps"),
        "interaction_exposure.interaction_exposure_denominator_steps",
    )

    planner_execution_mode = "unknown"
    planner_kinematics = metadata.get("planner_kinematics")
    if isinstance(planner_kinematics, Mapping):
        planner_execution_mode = str(planner_kinematics.get("execution_mode", "unknown"))

    return {
        "scenario_id": scenario_id,
        "speed_tier_id": tier.tier_id,
        "speed_cap_m_s": tier.cap_m_s,
        "planner_id": planner.planner_id,
        "seed": int(seed),
        "horizon_steps": horizon_steps,
        "dt_seconds": dt_seconds,
        "execution_mode": disposition,
        "execution_disposition_reason": disposition_reason,
        "planner_execution_mode": planner_execution_mode,
        "algorithm_metadata_status": str(metadata.get("status", "")),
        "success_rate": _binary_metric(metrics.get("success"), "metrics.success"),
        "collision_rate": 1.0 if total_collision_count > 0.0 else 0.0,
        "near_miss_rate": _binary_metric(
            _episode_metric(metrics, "near_misses"), "metrics.near_misses"
        ),
        "ped_collision_rate": 1.0 if ped_collision_count > 0.0 else 0.0,
        "obstacle_collision_rate": 1.0 if obstacle_collision_count > 0.0 else 0.0,
        "agent_collision_rate": 1.0 if agent_collision_count > 0.0 else 0.0,
        "unclassified_collision_rate": 1.0 if total_collision_count > typed_count + 1e-9 else 0.0,
        **speed,
        "resolved_actuation_envelope": dict(tier.resolved_actuation_envelope),
        "time_to_goal_norm": _episode_metric(metrics, "time_to_goal_norm"),
        "total_exposure_seconds": exposure_share * exposure_steps * dt_seconds,
        "travel_distance_m": _episode_metric(metrics, "socnavbench_path_length"),
        "mean_clearance_m": _episode_metric(metrics, "mean_clearance"),
        "min_clearance_m": _episode_metric(metrics, "min_clearance"),
        "raw_episode_id": episode_id,
        "raw_repo_commit": raw_repo_commit,
    }


def _authorized_campaign_scenarios(manifest: CampaignManifest) -> dict[str, Mapping[str, Any]]:
    """Build one exact scenario payload per frozen campaign scenario."""
    from robot_sf.training.scenario_loader import load_scenarios

    source = {str(s["name"]): s for s in load_scenarios(CAMPAIGN_SCENARIO_MATRIX)}
    result: dict[str, Mapping[str, Any]] = {}
    for scenario_identity in manifest.scenarios:
        base = source.get(scenario_identity.scenario_id)
        if base is None:
            raise AuthorizedCampaignError(
                f"frozen scenario missing from campaign matrix: {scenario_identity.scenario_id}"
            )
        result[scenario_identity.scenario_id] = base
    return result


def _authorized_batch_scenario(
    base: Mapping[str, Any],
    *,
    tier: SpeedTierRuntime,
    seeds: Sequence[int],
    horizon_steps: int,
) -> dict[str, Any]:
    """Apply one frozen tier and registered seed block to a scenario copy."""
    scenario = copy.deepcopy(dict(base))
    robot_config = scenario.get("robot_config")
    robot_config = dict(robot_config) if isinstance(robot_config, Mapping) else {}
    robot_config.update(_tier_variant_patch(tier))
    scenario["robot_config"] = robot_config
    simulation_config = scenario.get("simulation_config")
    simulation_config = dict(simulation_config) if isinstance(simulation_config, Mapping) else {}
    simulation_config["max_episode_steps"] = horizon_steps
    scenario["simulation_config"] = simulation_config
    scenario["seeds"] = [int(seed) for seed in seeds]
    return scenario


def _run_native_batch(
    scenarios: list[dict[str, Any]],
    out_path: pathlib.Path,
    *,
    algo: str,
    algo_config_path: str | None,
    horizon_steps: int,
    dt_seconds: float,
    resume: bool,
) -> dict[str, Any]:
    """Invoke the canonical map runner; this seam is intentionally testable."""
    from robot_sf.benchmark.map_runner import run_map_batch

    return run_map_batch(
        scenarios,
        out_path,
        EPISODE_SCHEMA_PATH,
        scenario_path=CAMPAIGN_SCENARIO_MATRIX,
        horizon=horizon_steps,
        dt=dt_seconds,
        record_forces=False,
        algo=algo,
        algo_config_path=(str(REPO_ROOT / algo_config_path) if algo_config_path else None),
        benchmark_profile="experimental",
        socnav_missing_prereq_policy="fail-fast",
        record_simulation_step_trace=True,
        workers=1,
        resume=resume,
    )


def _summary_count(summary: Mapping[str, Any], field_name: str) -> int:
    """Read a non-negative integer count from a canonical runner summary."""
    value = summary.get(field_name, 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AuthorizedCampaignError(
            f"runner summary {field_name} must be a non-negative integer, got {value!r}"
        )
    return int(value)


def _validate_runner_summary(  # noqa: C901
    summary: Mapping[str, Any],
    *,
    batch_name: str,
    expected_count: int,
    resume: bool,
) -> None:
    """Reject runner-level failures before any rows enter the campaign grid."""
    if not isinstance(summary, Mapping):
        raise AuthorizedCampaignError(f"runner summary for {batch_name} is not an object")
    failed_jobs = _summary_count(summary, "failed_jobs")
    skipped_jobs = _summary_count(summary, "skipped_jobs")
    written = _summary_count(summary, "written")
    if failed_jobs:
        raise AuthorizedCampaignError(
            f"runner batch {batch_name} reported {failed_jobs} failed jobs"
        )
    if skipped_jobs:
        raise AuthorizedCampaignError(
            f"runner batch {batch_name} reported {skipped_jobs} skipped jobs"
        )
    failures = summary.get("failures", [])
    if failures is not None and not isinstance(failures, list):
        raise AuthorizedCampaignError(
            f"runner batch {batch_name} reported malformed failure details"
        )
    if failures:
        raise AuthorizedCampaignError(
            f"runner batch {batch_name} reported failure details despite zero failed_jobs"
        )
    if not resume and written != expected_count:
        raise AuthorizedCampaignError(
            f"runner batch {batch_name} wrote {written} rows; expected {expected_count}"
        )

    availability = summary.get("benchmark_availability")
    if not isinstance(availability, Mapping):
        raise AuthorizedCampaignError(
            f"runner batch {batch_name} is missing benchmark_availability"
        )
    if availability.get("availability_status") != "available":
        raise AuthorizedCampaignError(
            f"runner batch {batch_name} is not benchmark-available: "
            f"{availability.get('availability_status')!r}"
        )
    if availability.get("benchmark_success") is not True:
        raise AuthorizedCampaignError(
            f"runner batch {batch_name} did not report benchmark_success=true"
        )

    provenance = summary.get("provenance")
    if not isinstance(provenance, Mapping):
        raise AuthorizedCampaignError(f"runner batch {batch_name} is missing provenance")
    manifest_status = provenance.get("result_manifest_status")
    if manifest_status != "available":
        raise AuthorizedCampaignError(
            f"runner batch {batch_name} result provenance is not available: {manifest_status!r}"
        )


def _campaign_command_environment_manifest(manifest: CampaignManifest) -> dict[str, Any]:
    """Describe the exact deterministic execution surface recorded with the campaign."""
    return {
        "authorized_command": (
            "uv run python scripts/benchmark/run_issue_5578_speed_tier_campaign.py "
            "--authorized-full-run --authorization-issue 6102"
        ),
        "entrypoint": "scripts/benchmark/run_issue_5578_speed_tier_campaign.py",
        "runner": "robot_sf.benchmark.map_runner.run_map_batch",
        "scenario_matrix": _repo_rel(CAMPAIGN_SCENARIO_MATRIX),
        "episode_schema": _repo_rel(EPISODE_SCHEMA_PATH),
        "planner_configs": sorted(
            planner.config_path for planner in manifest.planners if planner.config_path is not None
        ),
        "benchmark_profile": "experimental",
        "socnav_missing_prereq_policy": "fail-fast",
        "record_forces": False,
        "record_simulation_step_trace": True,
        "workers": 1,
        "horizon_steps": manifest.horizon_steps,
        "dt_seconds": manifest.dt_seconds,
        "manifest_hash": manifest.manifest_hash,
    }


def run_authorized_runtime_preflight(  # noqa: C901, PLR0912, PLR0915
    manifest: CampaignManifest,
    *,
    output_root: str | pathlib.Path,
    authorization_issue: int,
    scenario_id: str = PREFLIGHT_SCENARIO,
    seeds: Sequence[int] = (PREFLIGHT_SEEDS[0],),
) -> dict[str, Any]:
    """Exercise every planner/tier binding on disjoint seeds before registration.

    This preflight is operational launch evidence only. It uses one frozen scenario
    and disjoint seeds, retains traces, and verifies that every planner/tier batch
    reaches the native benchmark-row boundary without fallback or degradation. It
    never executes a registered seed and never produces benchmark synthesis.
    """
    if int(authorization_issue) != AUTHORIZED_EXECUTION_ISSUE:
        raise CampaignAuthorizationError(
            f"runtime preflight requires exactly --authorization-issue {AUTHORIZED_EXECUTION_ISSUE}"
        )
    if scenario_id not in {scenario.scenario_id for scenario in manifest.scenarios}:
        raise AuthorizedCampaignError(
            f"runtime preflight scenario is not frozen in the manifest: {scenario_id}"
        )
    preflight_seeds = tuple(int(seed) for seed in seeds)
    if not preflight_seeds:
        raise AuthorizedCampaignError("runtime preflight requires at least one disjoint seed")
    if set(preflight_seeds) & set(manifest.seeds):
        raise AuthorizedCampaignError("runtime preflight seeds overlap registered seeds 111-140")
    execution_provenance = _git_provenance()
    execution_commit = _require_clean_execution_provenance(execution_provenance)

    root = pathlib.Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    descriptor = {
        "schema_version": AUTHORIZED_EXECUTION_SCHEMA_VERSION,
        "preflight_kind": "native_runtime_binding",
        "authorization_issue": AUTHORIZED_EXECUTION_ISSUE,
        "manifest_hash": manifest.manifest_hash,
        "scenario_id": scenario_id,
        "seeds": list(preflight_seeds),
        "horizon_steps": manifest.horizon_steps,
        "dt_seconds": manifest.dt_seconds,
        "not_benchmark_evidence": True,
    }
    descriptor_path = root / "runtime_preflight_descriptor.json"
    if descriptor_path.exists():
        existing = json.loads(descriptor_path.read_text(encoding="utf-8"))
        if existing != descriptor:
            raise AuthorizedCampaignError(
                "runtime preflight output exists for a different manifest or seed set"
            )
        if any(root.glob("*.jsonl")) or (root / "runtime_preflight_report.json").exists():
            raise AuthorizedCampaignError(
                "runtime preflight output already exists for this manifest; refusing to rerun"
            )
    elif any(root.glob("*.jsonl")):
        raise AuthorizedCampaignError(
            "runtime preflight JSONL already exists without a matching descriptor"
        )
    descriptor_path.write_text(
        json.dumps(descriptor, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    base = _authorized_campaign_scenarios(manifest)[scenario_id]
    rows: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "schema_version": AUTHORIZED_EXECUTION_SCHEMA_VERSION,
        "preflight_kind": "native_runtime_binding",
        "not_benchmark_evidence": True,
        "authorization_issue": AUTHORIZED_EXECUTION_ISSUE,
        "manifest_hash": manifest.manifest_hash,
        "scenario_id": scenario_id,
        "seeds": list(preflight_seeds),
        "batches": [],
        "status": "running",
        "git_provenance": execution_provenance,
        "command_environment_manifest": _campaign_command_environment_manifest(manifest),
    }
    try:
        for tier in manifest.speed_tiers:
            for planner in manifest.planners:
                batch_name = f"{tier.tier_id}__{planner.planner_id}"
                raw_batch_path = root / f"{batch_name}.jsonl"
                scenario = _authorized_batch_scenario(
                    base,
                    tier=tier,
                    seeds=preflight_seeds,
                    horizon_steps=manifest.horizon_steps,
                )
                runner_summary = _run_native_batch(
                    [scenario],
                    raw_batch_path,
                    algo=planner.algorithm,
                    algo_config_path=planner.config_path,
                    horizon_steps=manifest.horizon_steps,
                    dt_seconds=manifest.dt_seconds,
                    resume=False,
                )
                _validate_runner_summary(
                    runner_summary,
                    batch_name=batch_name,
                    expected_count=len(preflight_seeds),
                    resume=False,
                )
                records = _read_jsonl_records(raw_batch_path)
                if len(records) != len(preflight_seeds):
                    raise AuthorizedCampaignError(
                        f"runtime preflight batch {batch_name} produced {len(records)} rows; "
                        f"expected {len(preflight_seeds)}"
                    )
                expected_keys = {(scenario_id, seed) for seed in preflight_seeds}
                seen_keys: set[tuple[str, int]] = set()
                modes: dict[str, int] = {}
                for record in records:
                    record_seed = _episode_integer(record.get("seed"), "seed")
                    record_key = (str(record.get("scenario_id")), record_seed)
                    if record_key in seen_keys:
                        raise AuthorizedCampaignError(
                            f"runtime preflight batch {batch_name} contains duplicate row "
                            f"{record_key}"
                        )
                    seen_keys.add(record_key)
                    cell = _cell_summary_from_episode(
                        record,
                        scenario_id=scenario_id,
                        tier=tier,
                        planner=planner,
                        seed=record_seed,
                        horizon_steps=manifest.horizon_steps,
                        dt_seconds=manifest.dt_seconds,
                        expected_repo_commit=execution_commit,
                    )
                    rows.append(cell)
                    mode = str(cell["execution_mode"])
                    modes[mode] = modes.get(mode, 0) + 1
                if seen_keys != expected_keys:
                    raise AuthorizedCampaignError(
                        f"runtime preflight batch {batch_name} does not cover the exact "
                        "disjoint scenario/seed block"
                    )
                report["batches"].append(
                    {
                        "batch": batch_name,
                        "speed_tier_id": tier.tier_id,
                        "planner_id": planner.planner_id,
                        "row_count": len(records),
                        "execution_modes": modes,
                        "runner_written": runner_summary.get("written"),
                        "runner_failed_jobs": runner_summary.get("failed_jobs"),
                    }
                )
        non_native = [row for row in rows if row["execution_mode"] != "native"]
        if non_native:
            report.update(
                {
                    "status": "rejected_non_native_rows",
                    "native_cell_count": len(rows) - len(non_native),
                    "excluded_cell_count": len(non_native),
                    "exclusions": [
                        {
                            "scenario_id": row["scenario_id"],
                            "speed_tier_id": row["speed_tier_id"],
                            "planner_id": row["planner_id"],
                            "seed": row["seed"],
                            "execution_mode": row["execution_mode"],
                            "reason": row.get("execution_disposition_reason"),
                        }
                        for row in non_native
                    ],
                    "finished_at_utc": _utc_timestamp(),
                }
            )
            _write_json_object(root / "runtime_preflight_report.json", report)
            raise AuthorizedCampaignError(
                f"runtime preflight rejected {len(non_native)} non-native rows"
            )
        report.update(
            {
                "status": "complete_native",
                "native_cell_count": len(rows),
                "excluded_cell_count": 0,
                "finished_at_utc": _utc_timestamp(),
            }
        )
        _write_json_object(root / "runtime_preflight_report.json", report)
        return report
    except AuthorizedCampaignError:
        if report.get("status") == "running":
            report["status"] = "blocked_or_failed"
        report["finished_at_utc"] = _utc_timestamp()
        _write_json_object(root / "runtime_preflight_report.json", report)
        raise
    except Exception as exc:  # campaign fail-closed boundary: record failure, re-raise (#6690)
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["finished_at_utc"] = _utc_timestamp()
        _write_json_object(root / "runtime_preflight_report.json", report)
        raise AuthorizedCampaignError(str(exc)) from exc


def _campaign_descriptor(manifest: CampaignManifest) -> dict[str, Any]:
    """Return the immutable descriptor used by duplicate/resume preflight."""
    return {
        "schema_version": AUTHORIZED_EXECUTION_SCHEMA_VERSION,
        "authorization_issue": AUTHORIZED_EXECUTION_ISSUE,
        "manifest_hash": manifest.manifest_hash,
        "expected_cell_count": manifest.expected_cell_count,
        "scenario_ids": [s.scenario_id for s in manifest.scenarios],
        "speed_tier_ids": [t.tier_id for t in manifest.speed_tiers],
        "planner_ids": [p.planner_id for p in manifest.planners],
        "seeds": list(manifest.seeds),
        "horizon_steps": manifest.horizon_steps,
        "dt_seconds": manifest.dt_seconds,
        "execution_boundary": "native_campaign_rows_only; fallback_and_degraded_rejected",
    }


def _prepare_authorized_output_root(
    raw_root: pathlib.Path,
    manifest: CampaignManifest,
    *,
    resume: bool,
) -> pathlib.Path:
    """Run duplicate and descriptor checks before any episode launch."""
    raw_root.mkdir(parents=True, exist_ok=True)
    descriptor_path = raw_root / "campaign_descriptor.json"
    descriptor = _campaign_descriptor(manifest)
    if descriptor_path.exists():
        try:
            existing = json.loads(descriptor_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise AuthorizedCampaignError(
                f"campaign descriptor is invalid JSON: {descriptor_path}"
            ) from exc
        if existing != descriptor:
            raise AuthorizedCampaignError(
                "duplicate/pre-existing campaign output has a different manifest or contract"
            )
    elif any(raw_root.glob("*.jsonl")):
        raise AuthorizedCampaignError(
            "raw campaign JSONL already exists without a matching descriptor; refusing to append"
        )
    elif any(raw_root.glob("*.summary.json")):
        raise AuthorizedCampaignError(
            "campaign summaries already exist without a matching descriptor; refusing to run"
        )
    if not resume:
        existing_batches = sorted([*raw_root.glob("*.jsonl"), *raw_root.glob("*.summary.json")])
        if existing_batches:
            raise AuthorizedCampaignError(
                "resume is disabled but campaign batch output already exists: "
                + ", ".join(str(path) for path in existing_batches[:3])
            )
    descriptor_path.write_text(
        json.dumps(descriptor, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return descriptor_path


def _write_cell_summaries(rows: Sequence[Mapping[str, Any]], path: pathlib.Path) -> None:
    """Write deterministic JSONL cell summaries."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(
        rows,
        key=lambda row: (
            str(row["scenario_id"]),
            str(row["speed_tier_id"]),
            str(row["planner_id"]),
            int(row["seed"]),
        ),
    )
    path.write_text(
        "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in ordered),
        encoding="utf-8",
    )


def _require_fresh_campaign_promotion_paths(
    cell_summaries_path: pathlib.Path,
    synthesis_path: pathlib.Path,
) -> None:
    """Reject launches that would overwrite promoted or quarantined campaign outputs.

    Raw batches are protected by the matching-descriptor checks in
    ``_prepare_authorized_output_root``. The promoted cell-summary, synthesis, and
    rejected-row paths can be outside that root, so protect them separately before
    any registered episode is launched. A resume may reuse matching raw batches,
    but it must never overwrite a previously promoted or quarantined result.
    """
    rejected_path = cell_summaries_path.with_suffix(".rejected.jsonl")
    named_paths = {
        "cell summaries": cell_summaries_path,
        "synthesis": synthesis_path,
        "rejected cell summaries": rejected_path,
    }
    if len(set(named_paths.values())) != len(named_paths):
        raise AuthorizedCampaignError(
            "authorized campaign requires distinct cell-summary, synthesis, and rejected-row paths"
        )
    existing = [f"{name}={path}" for name, path in named_paths.items() if path.exists()]
    if existing:
        raise AuthorizedCampaignError(
            "authorized campaign refuses to overwrite existing promoted or quarantined artifacts: "
            + ", ".join(existing)
        )


def execute_authorized_campaign(  # noqa: C901, PLR0912, PLR0915
    manifest: CampaignManifest,
    *,
    raw_root: str | pathlib.Path,
    cell_summaries_path: str | pathlib.Path,
    synthesis_path: str | pathlib.Path,
    report_path: str | pathlib.Path | None = None,
    authorization_issue: int,
    resume: bool = False,
) -> dict[str, Any]:
    """Execute the exact #5578 grid after an explicit #6102 authorization.

    This is the only function in this module permitted to launch registered
    episodes. It runs one serial, native map-runner batch per planner/tier pair,
    records a public-safe execution report, rejects any fallback/degraded row, and
    synthesizes only after the complete 2,160-row grid is present.
    """
    if int(authorization_issue) != AUTHORIZED_EXECUTION_ISSUE:
        raise CampaignAuthorizationError(
            "registered issue #5578 execution requires exactly "
            f"--authorization-issue {AUTHORIZED_EXECUTION_ISSUE}"
        )
    raw_path = pathlib.Path(raw_root)
    cell_path = pathlib.Path(cell_summaries_path)
    synthesis_path = pathlib.Path(synthesis_path)
    report_file = (
        pathlib.Path(report_path)
        if report_path is not None
        else raw_path / "campaign_run_report.json"
    )
    execution_provenance = _git_provenance()
    execution_commit = _require_clean_execution_provenance(execution_provenance)
    _require_fresh_campaign_promotion_paths(cell_path, synthesis_path)
    descriptor_path = _prepare_authorized_output_root(raw_path, manifest, resume=resume)
    manifest_path = raw_path / "campaign_manifest.json"
    write_manifest(manifest, manifest_path)

    report: dict[str, Any] = {
        "schema_version": AUTHORIZED_EXECUTION_SCHEMA_VERSION,
        "status": "running",
        "authorization_issue": AUTHORIZED_EXECUTION_ISSUE,
        "manifest_hash": manifest.manifest_hash,
        "expected_cell_count": manifest.expected_cell_count,
        "raw_root": _repo_rel(raw_path),
        "cell_summaries_path": _repo_rel(cell_path),
        "synthesis_path": _repo_rel(synthesis_path),
        "report_path": _repo_rel(report_file),
        "descriptor_path": _repo_rel(descriptor_path),
        "manifest_path": _repo_rel(manifest_path),
        "started_at_utc": _utc_timestamp(),
        "resume": bool(resume),
        "workers": 1,
        "horizon_steps": manifest.horizon_steps,
        "dt_seconds": manifest.dt_seconds,
        "fallback_or_degraded_policy": "reject",
        "batches": [],
        "exclusions": [],
        "git_provenance": execution_provenance,
        "command_environment_manifest": _campaign_command_environment_manifest(manifest),
    }
    rows: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str, str, int]] = set()
    base_scenarios = _authorized_campaign_scenarios(manifest)
    declared_scenario_ids = {scenario.scenario_id for scenario in manifest.scenarios}
    declared_seeds = set(manifest.seeds)
    expected_batch_keys = {
        (scenario_id, int(seed)) for scenario_id in declared_scenario_ids for seed in declared_seeds
    }
    try:
        for tier in manifest.speed_tiers:
            for planner in manifest.planners:
                batch_name = f"{tier.tier_id}__{planner.planner_id}"
                raw_batch_path = raw_path / f"{batch_name}.jsonl"
                scenario_batch = [
                    _authorized_batch_scenario(
                        base_scenarios[scenario.scenario_id],
                        tier=tier,
                        seeds=manifest.seeds,
                        horizon_steps=manifest.horizon_steps,
                    )
                    for scenario in manifest.scenarios
                ]
                summary = _run_native_batch(
                    scenario_batch,
                    raw_batch_path,
                    algo=planner.algorithm,
                    algo_config_path=planner.config_path,
                    horizon_steps=manifest.horizon_steps,
                    dt_seconds=manifest.dt_seconds,
                    resume=resume,
                )
                expected_batch_count = len(manifest.scenarios) * len(manifest.seeds)
                _validate_runner_summary(
                    summary,
                    batch_name=batch_name,
                    expected_count=expected_batch_count,
                    resume=resume,
                )
                batch_records = _read_jsonl_records(raw_batch_path)
                if len(batch_records) != expected_batch_count:
                    raise AuthorizedCampaignError(
                        f"batch {batch_name} has {len(batch_records)} rows; "
                        f"expected {expected_batch_count}"
                    )
                batch_seen: set[tuple[str, int]] = set()
                batch_modes: dict[str, int] = {}
                for record in batch_records:
                    scenario_id = str(record.get("scenario_id"))
                    seed = _episode_integer(record.get("seed"), "seed")
                    if scenario_id not in declared_scenario_ids:
                        raise AuthorizedCampaignError(
                            f"batch {batch_name} contains undeclared scenario: {scenario_id}"
                        )
                    if seed not in declared_seeds:
                        raise AuthorizedCampaignError(
                            f"batch {batch_name} contains undeclared seed: {seed}"
                        )
                    local_key = (scenario_id, seed)
                    if local_key in batch_seen:
                        raise AuthorizedCampaignError(
                            f"duplicate raw episode in batch {batch_name}: {local_key}"
                        )
                    batch_seen.add(local_key)
                    cell = _cell_summary_from_episode(
                        record,
                        scenario_id=scenario_id,
                        tier=tier,
                        planner=planner,
                        seed=seed,
                        horizon_steps=manifest.horizon_steps,
                        dt_seconds=manifest.dt_seconds,
                        expected_repo_commit=execution_commit,
                    )
                    cell_key = (
                        scenario_id,
                        tier.tier_id,
                        planner.planner_id,
                        seed,
                    )
                    if cell_key in seen_keys:
                        raise AuthorizedCampaignError(f"duplicate campaign cell: {cell_key}")
                    seen_keys.add(cell_key)
                    rows.append(cell)
                    mode = str(cell["execution_mode"])
                    batch_modes[mode] = batch_modes.get(mode, 0) + 1
                    if mode != "native":
                        report["exclusions"].append(
                            {
                                "cell": cell_key,
                                "execution_mode": mode,
                                "reason": cell.get("execution_disposition_reason"),
                            }
                        )
                if batch_seen != expected_batch_keys:
                    raise AuthorizedCampaignError(
                        f"batch {batch_name} does not cover the exact scenario/seed block"
                    )
                _write_json_object(
                    raw_path / f"{batch_name}.summary.json",
                    {
                        "schema_version": AUTHORIZED_EXECUTION_SCHEMA_VERSION,
                        "batch": batch_name,
                        "planner_id": planner.planner_id,
                        "algorithm": planner.algorithm,
                        "speed_tier_id": tier.tier_id,
                        "manifest_hash": manifest.manifest_hash,
                        "runner_summary": summary,
                        "row_count": len(batch_records),
                        "execution_modes": batch_modes,
                    },
                )
                report["batches"].append(
                    {
                        "batch": batch_name,
                        "planner_id": planner.planner_id,
                        "algorithm": planner.algorithm,
                        "speed_tier_id": tier.tier_id,
                        "raw_episode_jsonl": _repo_rel(raw_batch_path),
                        "row_count": len(batch_records),
                        "execution_modes": batch_modes,
                        "runner_written": summary.get("written"),
                        "runner_failed_jobs": summary.get("failed_jobs"),
                    }
                )

        expected_keys = {
            (
                scenario.scenario_id,
                tier.tier_id,
                planner.planner_id,
                int(seed),
            )
            for scenario in manifest.scenarios
            for tier in manifest.speed_tiers
            for planner in manifest.planners
            for seed in manifest.seeds
        }
        missing = expected_keys - seen_keys
        extra = seen_keys - expected_keys
        if missing or extra or len(rows) != manifest.expected_cell_count:
            raise AuthorizedCampaignError(
                "authorized campaign grid mismatch: "
                f"rows={len(rows)}, missing={sorted(missing)[:3]}, extra={sorted(extra)[:3]}"
            )

        non_native = [row for row in rows if row["execution_mode"] != "native"]
        if non_native:
            # Quarantine rejected rows away from the canonical cell-summaries artifact so a
            # fallback/degraded/failed run can never populate the path consumed by
            # ``--synthesize``. Only the all-native branch below writes the canonical path.
            rejected_path = cell_path.with_suffix(".rejected.jsonl")
            _write_cell_summaries(rows, rejected_path)
            report["status"] = "rejected_non_native_rows"
            report["native_cell_count"] = len(rows) - len(non_native)
            report["excluded_cell_count"] = len(non_native)
            report["rejected_cell_summaries_path"] = _repo_rel(rejected_path)
            report["finished_at_utc"] = _utc_timestamp()
            _write_json_object(report_file, report)
            raise AuthorizedCampaignError(
                f"authorized campaign rejected {len(non_native)} fallback/degraded/failed rows; "
                "no synthesis was promoted"
            )

        _write_cell_summaries(rows, cell_path)
        synthesis = synthesize_from_cell_summaries(cell_path, output_path=synthesis_path)
        report.update(
            {
                "status": "complete_native",
                "native_cell_count": len(rows),
                "excluded_cell_count": 0,
                "synthesis": synthesis,
                "finished_at_utc": _utc_timestamp(),
            }
        )
        _write_json_object(report_file, report)
        return report
    except AuthorizedCampaignError:
        if report.get("status") == "running":
            report["status"] = "blocked_or_failed"
            report["finished_at_utc"] = _utc_timestamp()
            _write_json_object(report_file, report)
        raise
    except Exception as exc:  # campaign fail-closed boundary: record failure, re-raise (#6690)
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["finished_at_utc"] = _utc_timestamp()
        _write_json_object(report_file, report)
        raise AuthorizedCampaignError(str(exc)) from exc


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
        "authorized_command": (
            "uv run python scripts/benchmark/run_issue_5578_speed_tier_campaign.py "
            "--authorized-full-run --authorization-issue 6102"
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


def _run_authorized_full_run(args: argparse.Namespace) -> int:
    """Run the registered campaign only with the explicit #6102 authorization."""
    if args.authorization_issue != AUTHORIZED_EXECUTION_ISSUE:
        raise CampaignAuthorizationError(
            "authorized campaign mode requires exactly "
            f"--authorization-issue {AUTHORIZED_EXECUTION_ISSUE}"
        )
    manifest = compile_campaign_manifest(config_path=args.config)
    raw_root = pathlib.Path(args.raw_root or DEFAULT_RAW_ROOT)
    # Anchor defaults to the module-level canonical paths so a custom ``--raw-root`` (for
    # example ``--raw-root raw``) cannot drop cell_summaries.jsonl / synthesis.json into the
    # repository root. Explicit ``--cell-summaries-out`` / ``--synthesis-out`` still win.
    cell_summaries_path = pathlib.Path(args.cell_summaries_out or DEFAULT_CELL_SUMMARY_PATH)
    synthesis_path = pathlib.Path(args.synthesis_out or DEFAULT_SYNTHESIS_PATH)
    report = execute_authorized_campaign(
        manifest,
        raw_root=raw_root,
        cell_summaries_path=cell_summaries_path,
        synthesis_path=synthesis_path,
        report_path=args.campaign_report_out,
        authorization_issue=args.authorization_issue,
        resume=args.resume,
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(
            "PASS: issue #5578 authorized native campaign completed "
            f"({report['native_cell_count']} cells, manifest={report['manifest_hash']})"
        )
        print(f"  synthesis: {report['synthesis_path']}")
        print(f"  report: {report['report_path']}")
    return 0


def _run_authorized_runtime_preflight(args: argparse.Namespace) -> int:
    """Exercise the authorized native planner/tier bindings on disjoint seeds."""
    if args.authorization_issue != AUTHORIZED_EXECUTION_ISSUE:
        raise CampaignAuthorizationError(
            "authorized runtime preflight requires exactly "
            f"--authorization-issue {AUTHORIZED_EXECUTION_ISSUE}"
        )
    manifest = compile_campaign_manifest(config_path=args.config)
    output_root = pathlib.Path(
        args.runtime_preflight_out or "output/issue_5578_robot_speed_tier_sweep/runtime_preflight"
    )
    report = run_authorized_runtime_preflight(
        manifest,
        output_root=output_root,
        authorization_issue=args.authorization_issue,
        scenario_id=args.runtime_preflight_scenario,
        seeds=tuple(args.runtime_preflight_seeds),
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(
            "PASS: issue #5578 authorized native runtime preflight completed "
            f"({report['native_cell_count']} cells; {NOT_EVIDENCE_BANNER})."
        )
        print(f"  output: {_repo_rel(output_root)}")
    return 0


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
    if args.authorized_full_run:
        return _run_authorized_full_run(args)
    if args.authorized_runtime_preflight:
        return _run_authorized_runtime_preflight(args)
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
    mode.add_argument(
        "--authorized-full-run",
        action="store_true",
        help=(
            "Execute the registered grid only with the explicit #6102 authorization; "
            "fallback/degraded rows fail closed."
        ),
    )
    mode.add_argument(
        "--authorized-runtime-preflight",
        action="store_true",
        help=(
            "Exercise every planner/tier binding on disjoint seeds only with the explicit "
            "#6102 authorization; this is not benchmark evidence."
        ),
    )
    parser.add_argument(
        "--authorization-issue",
        type=int,
        help=(
            "Required for authorized modes; must be the operational authorization "
            f"issue {AUTHORIZED_EXECUTION_ISSUE}."
        ),
    )
    parser.add_argument(
        "--cell-summaries-out",
        type=pathlib.Path,
        help="Cell-summary output path for the authorized campaign or documented handoff.",
    )
    parser.add_argument(
        "--raw-root",
        type=pathlib.Path,
        help="Raw episode root for --authorized-full-run (defaults under output/).",
    )
    parser.add_argument(
        "--campaign-report-out",
        type=pathlib.Path,
        help="Public-safe authorized campaign report path (defaults inside --raw-root).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a matching authorized campaign descriptor after a partial run.",
    )
    parser.add_argument(
        "--runtime-preflight-out",
        type=pathlib.Path,
        help="Output directory for --authorized-runtime-preflight.",
    )
    parser.add_argument(
        "--runtime-preflight-scenario",
        type=str,
        default=PREFLIGHT_SCENARIO,
        help="Frozen scenario for --authorized-runtime-preflight.",
    )
    parser.add_argument(
        "--runtime-preflight-seeds",
        type=int,
        nargs="+",
        default=[PREFLIGHT_SEEDS[0]],
        help="Disjoint seeds for --authorized-runtime-preflight (outside 111-140).",
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
