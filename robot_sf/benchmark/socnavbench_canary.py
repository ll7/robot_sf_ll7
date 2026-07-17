"""Runnable Robot SF -> SocNavBench cross-suite canary (#5783 / issue #5842).

Issue #5842 hardens the canary introduced by #5783 so the pinned policy is *actually
executed* on the Robot SF path, the exported scenario + staged ETH traversible are
*really consumed* by a SocNavBench source harness, the two suites no longer share a
colliding metric identity for reciprocal ratio definitions, and any fallback path is
recorded as non-success evidence rather than a green receipt.

Scope and claim boundary
------------------------
- This is a **diagnostic sim-to-sim canary**, not a benchmark campaign, not a training run,
  and not a simulator-equivalence or policy-superiority claim. It proves that the same
  pinned policy identity and the same cross-suite metric family can be produced on both sides
  from one mapped scenario and one seed, with suite-specific denominators and reciprocal
  ratio directions preserved and recorded.
- The Robot SF side **executes the pinned SocialForce policy** through the real Robot SF
  environment (headless, CPU, no licensed data) and computes the vendored-style
  ``socnavbench_path_length_ratio`` (distance / displacement) over the resulting trajectory.
- The SocNavBench side **runs a real source harness** that loads the materialized export JSON
  and the staged ETH traversible shape contract, then computes the vendored SocNavBench
  ``cost_functions.path_length_ratio`` (displacement / distance) over the same geometric path.
  Both ratio definitions differ in direction by design; each is recorded with its own metric
  id, numerator, denominator, formula, direction, and mapping class.
- When the licensed ETH asset is absent the SocNavBench path fails closed. A test-only
  ``--allow-synthetic-traversible`` flag lets the no-licensed-data fixture path still exercise
  the real runner against a synthetic traversible, but it is recorded as a fallback and the
  receipt reports ``counts_as_success_evidence: False``. The real canary never sets it.

Fail-closed contract
--------------------
Every gate below returns a nonzero exit / raises rather than silently degrading:
  * missing staged ETH asset or absent export -> blocked (unless the test-only flag is set,
    which is recorded as fallback / non-success);
  * placeholder metadata (``tbd`` / ``blocked_prerequisite`` / ``to_be_selected`` / ``None``)
    in the pinned policy or scenario mapping -> blocked;
  * policy-identity mismatch between the two suite receipts -> blocked (fallback detected);
  * scenario conversion with no mappable map/agents -> blocked;
  * any fallback or degraded adapter path requested or taken -> recorded, never green.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from robot_sf.benchmark.canary_rollout import PolicyRolloutResult, execute_pinned_policy
from robot_sf.benchmark.cross_benchmark_metrics import (
    CROSS_BENCHMARK_CLAIM_BOUNDARY,
    CROSS_BENCHMARK_METRIC_MAPPING_VERSION,
)
from robot_sf.benchmark.metrics import EpisodeData, socnavbench_path_length_ratio
from robot_sf.data.external.socnavbench_eth import (
    ASSET_ID as SOCNAVBENCH_ETH_ASSET_ID,
)
from robot_sf.data.external.socnavbench_eth import (
    SocNavBenchEthDataError,
)
from robot_sf.data.external.socnavbench_eth import (
    is_available as socnavbench_eth_is_available,
)
from robot_sf.data.external.socnavbench_eth import (
    load_traversible as socnavbench_eth_load_traversible,
)
from robot_sf.data.external.socnavbench_eth import (
    resolve_root as socnavbench_eth_resolve_root,
)

MODULE_ROOT = Path(__file__).resolve().parents[2]
EXPORT_SCHEMA_VERSION = "robot_sf.socnavbench_canary_export.v1"
RECEIPT_SCHEMA_VERSION = "robot_sf.cross_suite_canary_receipt.v1"
CANARY_NAME = "socnavbench_cross_suite_canary"
CANARY_VERSION = "1"
PLACEHOLDER_TOKENS = (
    "tbd",
    "to_be_selected",
    "to_be_selected_when_unblocked",
    "blocked_prerequisite",
    "scaffold_only",
    "unvalidated_scaffold",
    "blocked_external_input",
)

# Pinned policy constants. The algo config is read at runtime for provenance.
PINNED_POLICY_ID = "social_force_holonomic"
PINNED_POLICY_VERSION = "tau_low_v1"
PINNED_POLICY_ALGO = "social_force"
PINNED_POLICY_ALGO_CONFIG = (
    MODULE_ROOT / "configs" / "algos" / "social_force_holonomic_tuned_tau_low.yaml"
)

# One concrete, documented cross-suite scenario mapping.
CANARY_ROBOT_SF_SCENARIO_ID = "canary-corridor-uni-low-open"
CANARY_SOCNAVBENCH_SCENARIO_ID = "ETH/canary-corridor-unilowopen"
CANARY_SEED = 0
CANARY_SCENARIO_LIMITATION_FLAGS = (
    "geometry_non_equivalence_eth_vs_robot_sf_svg",
    "pedestrian_dynamics_proxy",
    "no_real_sim_equivalence",
    "single_seed_canary_not_campaign_evidence",
)
CANARY_SCENARIO_MAPPING_QUALITY = "documented_canary_mapping"

# Suite-specific metric identities for the reciprocal path-length-ratio definitions.
# Robot SF computes distance / displacement; SocNavBench computes displacement / distance.
# These MUST be distinct metric IDs — sharing one exact id for reciprocal formulas is a bug.
ROBOT_SF_METRIC_ID = "robot_sf.path_length_ratio.distance_over_displacement"
SOCNAVBENCH_METRIC_ID = "socnavbench.path_length_ratio.displacement_over_distance"
ROBOT_SF_RATIO_FORMULA = "total_path_distance / start_to_goal_displacement"
SOCNAVBENCH_RATIO_FORMULA = "start_to_goal_displacement / total_path_distance"
ROBOT_SF_RATIO_DIRECTION = "distance_over_displacement"
SOCNAVBENCH_RATIO_DIRECTION = "displacement_over_distance"
ROBOT_SF_DENOMINATOR_KIND = "start_to_goal_displacement_m"
SOCNAVBENCH_DENOMINATOR_KIND = "start_to_goal_displacement_m"
ROBOT_SF_MAPPING_CLASS = "approximate"
SOCNAVBENCH_MAPPING_CLASS = "approximate"
SYNTHETIC_TRAVERSIBLE_RESOLUTION_M = 0.05
SYNTHETIC_TRAVERSIBLE_SHAPE = (400, 400)


@dataclass(frozen=True, slots=True)
class PinnedPolicy:
    """Concrete, proven policy identity pinned for the cross-suite canary."""

    policy_id: str
    version: str
    algo: str
    algo_config: str
    config_digest_sha256: str
    source_commit: str
    # Runtime provenance: the exact planner parameters that governed the executed trajectory.
    runtime_planner_config: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        """Return a stable, JSON-safe identity block including runtime provenance."""
        return {
            "policy_id": self.policy_id,
            "version": self.version,
            "algo": self.algo,
            "algo_config": self.algo_config,
            "config_digest_sha256": self.config_digest_sha256,
            "source_commit": self.source_commit,
            "runtime_planner_config": dict(self.runtime_planner_config),
        }

    def matches(self, other: PinnedPolicy) -> bool:
        """Return True only when every identity field is identical (fallback detector)."""
        return (
            self.policy_id == other.policy_id
            and self.version == other.version
            and self.algo == other.algo
            and self.algo_config == other.algo_config
            and self.config_digest_sha256 == other.config_digest_sha256
            and self.source_commit == other.source_commit
            and dict(self.runtime_planner_config) == dict(other.runtime_planner_config)
        )


@dataclass(frozen=True, slots=True)
class ScenarioMapping:
    """Concrete, documented cross-suite scenario mapping with limitation flags."""

    robot_sf_scenario_id: str
    socnavbench_scenario_id: str
    seed: int
    external_asset_id: str
    mapping_quality: str
    limitation_flags: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a stable, JSON-safe mapping block."""
        return {
            "robot_sf_scenario_id": self.robot_sf_scenario_id,
            "socnavbench_scenario_id": self.socnavbench_scenario_id,
            "seed": self.seed,
            "external_asset_id": self.external_asset_id,
            "mapping_quality": self.mapping_quality,
            "limitation_flags": list(self.limitation_flags),
        }


@dataclass(frozen=True, slots=True)
class SuiteMetricSpec:
    """Suite-specific path-length-ratio definition (reciprocal across suites)."""

    metric_id: str
    numerator: str
    denominator_kind: str
    formula: str
    ratio_direction: str
    mapping_class: str

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-safe spec of the suite-specific ratio definition."""
        return {
            "metric_id": self.metric_id,
            "numerator": self.numerator,
            "denominator_kind": self.denominator_kind,
            "formula": self.formula,
            "ratio_direction": self.ratio_direction,
            "mapping_class": self.mapping_class,
        }


class CanaryError(RuntimeError):
    """Raised when the canary cannot run without violating a fail-closed gate."""


def _git_commit_sha(*, repo_root: Path = MODULE_ROOT) -> str:
    """Return the current git commit SHA, or ``dirty:<short>`` fallback for the worktree.

    A deterministic, non-empty string is required so the policy identity block is always
    populated; a worktree detached-head SHA is acceptable for a canary receipt.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        # Worktree/non-git fallback: hash the pinned config so identity stays reproducible.
        return "dirty:" + _config_digest(PINNED_POLICY_ALGO_CONFIG)[:12]


def _config_digest(path: Path) -> str:
    """Return the sha256 hex digest of a config file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _raise_if_placeholder(name: str, value: str | None) -> None:
    """Fail closed when a required pinned field is missing or a placeholder token."""
    if value is None or not str(value).strip():
        raise CanaryError(f"{name} is missing or empty (placeholder not allowed)")
    lowered = str(value).strip().lower()
    for token in PLACEHOLDER_TOKENS:
        if lowered == token or lowered.startswith(f"{token}_") or lowered.endswith(f"_{token}"):
            raise CanaryError(
                f"{name} contains a placeholder token '{token}' (not a concrete value)"
            )


def _repo_relative(path: Path, *, repo_root: Path = MODULE_ROOT) -> str:
    """Return a repo-root-relative POSIX path string."""
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return path.as_posix()


def resolve_pinned_policy(*, repo_root: Path = MODULE_ROOT) -> PinnedPolicy:
    """Resolve the concrete, pinned policy identity for the canary.

    The config digest and source commit are computed at runtime so the identity block
    reflects the real state of the tracked config file and the current git HEAD.

    Returns:
        PinnedPolicy with all identity fields populated (no placeholder tokens allowed).

    Raises:
        CanaryError: If any identity field is missing or a placeholder token.
    """
    config_path = repo_root / "configs" / "algos" / "social_force_holonomic_tuned_tau_low.yaml"
    _raise_if_placeholder("policy_id", PINNED_POLICY_ID)
    _raise_if_placeholder("version", PINNED_POLICY_VERSION)
    _raise_if_placeholder("algo", PINNED_POLICY_ALGO)

    if not config_path.is_file():
        raise CanaryError(f"Pinned policy algo config not found: {config_path}")

    # Runtime provenance comes from the real rollout configuration (see canary_rollout).
    from robot_sf.benchmark.canary_rollout import _resolve_pinned_planner_config  # noqa: PLC0415

    try:
        planner_cfg = _resolve_pinned_planner_config(algo_config=config_path)
    except (OSError, TypeError, ValueError) as exc:
        raise CanaryError(f"Pinned policy planner config is invalid: {exc}") from exc
    from robot_sf.benchmark.canary_rollout import _planner_config_provenance  # noqa: PLC0415

    runtime_config = _planner_config_provenance(planner_cfg)

    return PinnedPolicy(
        policy_id=PINNED_POLICY_ID,
        version=PINNED_POLICY_VERSION,
        algo=PINNED_POLICY_ALGO,
        algo_config=_repo_relative(config_path, repo_root=repo_root),
        config_digest_sha256=_config_digest(config_path),
        source_commit=_git_commit_sha(repo_root=repo_root),
        runtime_planner_config=runtime_config,
    )


def resolve_scenario_mapping() -> ScenarioMapping:
    """Return the concrete, documented cross-suite scenario mapping.

    Returns:
        ScenarioMapping with all fields populated (no placeholder tokens allowed).

    Raises:
        CanaryError: If any mapping field is missing or a placeholder token.
    """
    _raise_if_placeholder("robot_sf_scenario_id", CANARY_ROBOT_SF_SCENARIO_ID)
    _raise_if_placeholder("socnavbench_scenario_id", CANARY_SOCNAVBENCH_SCENARIO_ID)
    return ScenarioMapping(
        robot_sf_scenario_id=CANARY_ROBOT_SF_SCENARIO_ID,
        socnavbench_scenario_id=CANARY_SOCNAVBENCH_SCENARIO_ID,
        seed=CANARY_SEED,
        external_asset_id=SOCNAVBENCH_ETH_ASSET_ID,
        mapping_quality=CANARY_SCENARIO_MAPPING_QUALITY,
        limitation_flags=CANARY_SCENARIO_LIMITATION_FLAGS,
    )


def _start_goal_displacement(robot_positions: np.ndarray, goal: np.ndarray) -> float:
    """Compute straight-line start-to-goal displacement in meters.

    Returns:
        Euclidean distance from the first position to the goal, in meters.
    """
    start = np.asarray(robot_positions[0], dtype=float)
    return float(np.linalg.norm(np.asarray(goal, dtype=float) - start))


def _build_episode_from_trajectory(
    robot_positions: list[tuple[float, float]],
    goal_position: tuple[float, float],
) -> EpisodeData:
    """Build a minimal ``EpisodeData`` from an executed trajectory for metric computation.

    Returns:
        EpisodeData with positions from the trajectory and zeroed-out pedestrian data.
    """
    positions = np.asarray(robot_positions, dtype=float)
    goal = np.asarray(goal_position, dtype=float)
    return EpisodeData(
        robot_pos=positions,
        robot_vel=np.zeros_like(positions),
        robot_acc=np.zeros_like(positions),
        peds_pos=np.zeros((positions.shape[0], 0, 2), dtype=float),
        ped_forces=np.zeros((positions.shape[0], 0, 2), dtype=float),
        goal=goal,
        dt=0.1,
        robot_radius=0.3,
        ped_radius=0.4,
    )


def materialize_socnavbench_export(
    *,
    policy: PinnedPolicy,
    mapping: ScenarioMapping,
    out_dir: Path,
    rollout: PolicyRolloutResult | None = None,
) -> tuple[Path, PolicyRolloutResult]:
    """Write a concrete SocNavBench scenario JSON consumed by the canary runner.

    The export references the staged ETH traversible via the external-data registry root and
    pins the same policy identity that the Robot SF receipt records, so a downstream SocNavBench
    runner consumes a real, mappable scenario (not a preview). It also records the Robot SF
    executed trajectory so the runner consumes the identical geometric path on both sides.

    When ``rollout`` is provided (e.g. from the caller that already ran the policy), it is used
    directly; otherwise ``execute_pinned_policy`` is called once here. Pass the returned rollout
    to ``run_robot_sf_receipt_from_rollout`` to avoid a double execution.

    Args:
        policy: Pinned policy identity.
        mapping: Concrete cross-suite scenario mapping.
        out_dir: Output directory for the export JSON.
        rollout: Optional pre-executed policy rollout result. If None, the policy is executed.

    Returns:
        (export_path, rollout) where rollout is the (possibly freshly executed) policy result.

    Raises:
        CanaryError: If the export cannot be produced without placeholder metadata.
    """
    _raise_if_placeholder("socnavbench_scenario_id", mapping.socnavbench_scenario_id)

    # Execute the pinned policy exactly once and reuse the result across both suite receipts.
    if rollout is None:
        try:
            rollout = execute_pinned_policy(
                seed=mapping.seed, algo_config=PINNED_POLICY_ALGO_CONFIG
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise CanaryError(f"Robot SF policy rollout failed: {exc}") from exc
    if rollout.scenario_id != mapping.robot_sf_scenario_id:
        raise CanaryError(
            "Executed Robot SF scenario does not match the declared mapping: "
            f"{rollout.scenario_id!r} != {mapping.robot_sf_scenario_id!r}."
        )

    robot_positions = [list(pos) for pos in rollout.robot_positions]
    goal = list(rollout.goal_position)

    export = {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "canary": CANARY_NAME,
        "canary_version": CANARY_VERSION,
        "socnavbench_scenario_id": mapping.socnavbench_scenario_id,
        "robot_sf_scenario_id": mapping.robot_sf_scenario_id,
        "executed_robot_sf_scenario_id": rollout.scenario_id,
        "seed": mapping.seed,
        "external_asset_id": mapping.external_asset_id,
        "map": {
            "asset_id": mapping.external_asset_id,
            "traversible": "ETH",
            "resolution_m": None,  # filled by the runner from the staged ETH shape contract
            "width_m": None,
            "length_m": None,
        },
        "robot": {
            "start": robot_positions[0],
            "goal": goal,
        },
        "trajectory": robot_positions,
        "pedestrians": [
            {
                "id": "ped-1",
                "start": [float(goal[0]) - 2.0, float(goal[1]) + 1.5],
                "goal": [float(robot_positions[0][0]) + 2.0, float(robot_positions[0][1]) + 1.5],
            }
        ],
        "policy_identity": policy.to_dict(),
        "mapping_quality": mapping.mapping_quality,
        "limitation_flags": list(mapping.limitation_flags),
        "claim_boundary": CROSS_BENCHMARK_CLAIM_BOUNDARY,
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    export_path = out_dir / f"{mapping.socnavbench_scenario_id.replace('/', '_')}.socnavbench.json"
    export_path.write_text(json.dumps(export, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return export_path, rollout


def run_robot_sf_receipt_from_rollout(
    *,
    policy: PinnedPolicy,
    mapping: ScenarioMapping,
    rollout: PolicyRolloutResult,
) -> dict[str, Any]:
    """Build the Robot SF receipt from an already-executed policy rollout.

    This is the canonical path for ``run_canary``: the rollout is executed once and shared
    across both suite receipts to guarantee that the trajectory and denominator are identical.

    Returns:
        JSON-safe Robot SF receipt dict with policy identity (including runtime provenance),
        mapping, metric value, and the suite-specific ratio definition.
    """
    if rollout.scenario_id != mapping.robot_sf_scenario_id:
        raise CanaryError(
            "Executed Robot SF scenario does not match the declared mapping: "
            f"{rollout.scenario_id!r} != {mapping.robot_sf_scenario_id!r}."
        )
    episode = _build_episode_from_trajectory(rollout.robot_positions, rollout.goal_position)
    try:
        # Robot SF definition: distance / displacement (>= 1.0 for efficient paths).
        ratio = float(socnavbench_path_length_ratio(episode))
    except (AttributeError, IndexError, TypeError, ValueError) as exc:
        raise CanaryError(f"Robot SF metric computation failed: {exc}") from exc

    spec = SuiteMetricSpec(
        metric_id=ROBOT_SF_METRIC_ID,
        numerator="total_path_distance",
        denominator_kind=ROBOT_SF_DENOMINATOR_KIND,
        formula=ROBOT_SF_RATIO_FORMULA,
        ratio_direction=ROBOT_SF_RATIO_DIRECTION,
        mapping_class=ROBOT_SF_MAPPING_CLASS,
    )
    denominator = _start_goal_displacement(episode.robot_pos, episode.goal)
    return {
        "suite": "Robot SF",
        "policy_identity": policy.to_dict(),
        "scenario_mapping": mapping.to_dict(),
        "metric_mapping_version": CROSS_BENCHMARK_METRIC_MAPPING_VERSION,
        "metric_spec": spec.to_dict(),
        "value": ratio,
        "denominator": denominator,
        "trajectory_length": len(rollout.robot_positions),
        "executed_policy": True,
        "claim_boundary": CROSS_BENCHMARK_CLAIM_BOUNDARY,
    }


def run_robot_sf_receipt(
    *,
    policy: PinnedPolicy,
    mapping: ScenarioMapping,
) -> dict[str, Any]:
    """Run the Robot SF side of the canary and return a machine-checkable receipt.

    Executes the **real pinned policy** through the Robot SF environment, then computes the
    vendored-style ``socnavbench_path_length_ratio`` (distance / displacement) over the
    resulting trajectory. Records the suite-specific metric identity, denominator, numerator,
    formula, direction, and mapping class.

    For use in tests or standalone calls. When calling from ``run_canary``, prefer
    ``run_robot_sf_receipt_from_rollout`` to avoid executing the policy twice.

    Returns:
        JSON-safe Robot SF receipt dict with policy identity, mapping, metric value, and
        the suite-specific ratio definition.
    """
    try:
        rollout = execute_pinned_policy(seed=mapping.seed, algo_config=PINNED_POLICY_ALGO_CONFIG)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise CanaryError(f"Robot SF policy rollout failed: {exc}") from exc
    return run_robot_sf_receipt_from_rollout(policy=policy, mapping=mapping, rollout=rollout)


def run_socnavbench_receipt(
    *,
    export_path: Path,
    policy: PinnedPolicy,
    mapping: ScenarioMapping,
    allow_synthetic_traversible: bool = False,
) -> dict[str, Any]:
    """Run the SocNavBench side of the canary and return a machine-checkable receipt.

    Loads the materialized export and **runs a real SocNavBench source harness** that consumes
    the export and the staged ETH traversible shape contract, then computes the vendored
    ``cost_functions.path_length_ratio`` (displacement / distance) over the same geometric path.

    When the licensed ETH asset is absent the runner fails closed unless
    ``allow_synthetic_traversible`` is set for the no-licensed-data test path. In that case the
    run is recorded as a fallback and the receipt reports ``counts_as_success_evidence: False``.
    The ``fallback_forbidden`` flag in the joint receipt reflects whether any fallback was taken
    (it is ``True`` only when no fallback path was activated and the real asset was used).

    Raises:
        CanaryError: On missing export, asset-absent-without-explicit-test-flag, or fallback use.

    Returns:
        JSON-safe SocNavBench receipt dict with policy identity, mapping, metric value, and
        the suite-specific ratio definition.
    """
    export_path = Path(export_path)
    if not export_path.is_file():
        raise CanaryError(f"SocNavBench export missing: {export_path}")
    try:
        export = json.loads(export_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CanaryError(f"SocNavBench export could not be read as JSON: {export_path}") from exc
    if not isinstance(export, dict):
        raise CanaryError("SocNavBench export root must be a mapping.")

    # Fail closed if the export itself carried placeholder metadata.
    _raise_if_placeholder("export.socnavbench_scenario_id", export.get("socnavbench_scenario_id"))
    export_map = export.get("map")
    if not isinstance(export_map, dict):
        raise CanaryError("SocNavBench export map block must be a mapping.")
    if export_map.get("asset_id") != mapping.external_asset_id:
        raise CanaryError("SocNavBench export map asset does not match the scenario mapping.")
    if export_map.get("traversible") != "ETH":
        raise CanaryError("SocNavBench export must select the ETH traversible.")

    # Confirm the staged ETH asset is actually present. The licensed asset is the only real
    # external input; absence means the canary cannot run a runnable invocation.
    asset_available = socnavbench_eth_is_available()
    if not asset_available and not allow_synthetic_traversible:
        root = socnavbench_eth_resolve_root()
        raise CanaryError(
            f"Licensed SocNavBench ETH asset not staged at {root}; cannot run a runnable "
            "SocNavBench invocation. Stage the external asset or stop. Fallback is forbidden."
        )

    # The real source harness consumes the export and the staged ETH traversible shape contract.
    # Without the asset the runner falls back to a synthetic traversible (recorded as fallback).
    traversible_resolution, traversible, synthetic_traversible = _load_traversible_map(
        export, allow_synthetic=allow_synthetic_traversible
    )

    try:
        # The trajectory is embedded in the export by materialize_socnavbench_export so the
        # SocNavBench runner consumes the identical geometric path as the Robot SF receipt.
        trajectory = np.asarray(export["trajectory"], dtype=float)
        goal = np.asarray(export["robot"]["goal"], dtype=float)
        ratio, runner_meta = _run_socnavbench_runner(
            trajectory=trajectory,
            goal=goal,
            traversible=traversible,
            traversible_resolution=traversible_resolution,
            synthetic_traversible=synthetic_traversible,
        )
    except (AttributeError, ImportError, IndexError, KeyError, TypeError, ValueError) as exc:
        raise CanaryError(f"SocNavBench metric computation failed: {exc}") from exc

    spec = SuiteMetricSpec(
        metric_id=SOCNAVBENCH_METRIC_ID,
        numerator="start_to_goal_displacement",
        denominator_kind=SOCNAVBENCH_DENOMINATOR_KIND,
        formula=SOCNAVBENCH_RATIO_FORMULA,
        ratio_direction=SOCNAVBENCH_RATIO_DIRECTION,
        mapping_class=SOCNAVBENCH_MAPPING_CLASS,
    )
    denominator = _start_goal_displacement(trajectory, goal)
    # A fallback is active whenever the test-only synthetic escape is used OR the real asset
    # was not staged. Either case disqualifies the receipt as success evidence.
    fallback = bool(allow_synthetic_traversible) or not asset_available
    return {
        "suite": "SocNavBench",
        "policy_identity": policy.to_dict(),
        "scenario_mapping": mapping.to_dict(),
        "metric_mapping_version": CROSS_BENCHMARK_METRIC_MAPPING_VERSION,
        "metric_spec": spec.to_dict(),
        "value": float(ratio),
        "denominator": denominator,
        "external_asset_id": export.get("external_asset_id"),
        "external_asset_staged": bool(asset_available),
        "traversible_resolution_m": (float(traversible_resolution)),
        "traversible_shape": [int(dim) for dim in traversible.shape],
        "synthetic_traversible": synthetic_traversible,
        "runner": runner_meta,
        "executed_policy": True,
        "is_fallback": fallback,
        "counts_as_success_evidence": (not fallback),
        "trajectory_length": int(trajectory.shape[0]),
        "claim_boundary": CROSS_BENCHMARK_CLAIM_BOUNDARY,
    }


def _load_traversible_map(
    export: dict[str, Any], *, allow_synthetic: bool
) -> tuple[float, np.ndarray, bool]:
    """Load the staged ETH traversible or construct an explicit test-only fallback.

    Returns:
        ``(resolution, traversible, is_synthetic)`` for the runner.
    """
    try:
        resolution, traversible = socnavbench_eth_load_traversible()
        return float(resolution), np.asarray(traversible), False
    except SocNavBenchEthDataError as exc:
        if allow_synthetic:
            return (
                SYNTHETIC_TRAVERSIBLE_RESOLUTION_M,
                np.ones(SYNTHETIC_TRAVERSIBLE_SHAPE, dtype=bool),
                True,
            )
        raise CanaryError(f"SocNavBench ETH traversible could not be loaded: {exc}") from exc


def _run_socnavbench_runner(
    *,
    trajectory: np.ndarray,
    goal: np.ndarray,
    traversible: np.ndarray,
    traversible_resolution: float,
    synthetic_traversible: bool,
) -> tuple[float, dict[str, Any]]:
    """Invoke the vendored SocNavBench source harness over the exported trajectory.

    The vendored SocNavBench metric module imports its own helpers as
    ``from metrics.cost_utils import *``, so its package root
    (``third_party/socnavbench``) must be on ``sys.path`` -- not ``third_party``.

    Args:
        trajectory: (N, 2) executed robot path from the export.
        goal: (2,) goal position consumed by the runner.
        traversible: Validated staged ETH traversible (or explicit test fallback).
        traversible_resolution: Traversible map resolution in meters.
        synthetic_traversible: Whether the explicit test fallback is active.

    Returns:
        ``(ratio, runner_meta)`` where ratio is the SocNavBench definition
        (displacement / distance) and runner_meta records the harness inputs.
    """
    vendored_root = MODULE_ROOT / "third_party" / "socnavbench"
    if str(vendored_root) not in sys.path:
        sys.path.insert(0, str(vendored_root))
    from metrics.cost_functions import path_length_ratio as _sn_path_length_ratio  # noqa: PLC0415

    trajectory = np.asarray(trajectory, dtype=float)
    goal = np.asarray(goal, dtype=float)
    traversible = np.asarray(traversible)
    if trajectory.ndim != 2 or trajectory.shape[1] != 2 or trajectory.shape[0] < 2:
        raise ValueError("SocNavBench trajectory must be a non-empty (N, 2) path.")
    if goal.shape != (2,):
        raise ValueError("SocNavBench goal must contain exactly two coordinates.")
    if traversible.ndim != 2 or 0 in traversible.shape:
        raise ValueError("SocNavBench traversible must be a non-empty 2D array.")
    if not np.isfinite(trajectory).all() or not np.isfinite(goal).all():
        raise ValueError("SocNavBench trajectory and goal must contain finite coordinates.")
    if not np.isfinite(traversible_resolution) or traversible_resolution <= 0:
        raise ValueError("SocNavBench traversible resolution must be finite and positive.")

    # SocNavBench indexes traversibles as [y, x]. The exported scenario is accepted by the
    # source metric harness only when its path and aspirational goal land on traversible cells;
    # this makes the staged map an execution input rather than a presence-only prerequisite.
    points = np.vstack((trajectory, goal))
    grid_xy = np.floor(points / traversible_resolution).astype(int)
    x_indices = grid_xy[:, 0]
    y_indices = grid_xy[:, 1]
    if (
        (x_indices < 0).any()
        or (y_indices < 0).any()
        or (x_indices >= traversible.shape[1]).any()
        or (y_indices >= traversible.shape[0]).any()
    ):
        raise ValueError("SocNavBench export path falls outside the staged traversible bounds.")
    occupied = np.asarray(traversible)[y_indices, x_indices]
    if not np.asarray(occupied, dtype=bool).all():
        raise ValueError("SocNavBench export path intersects a non-traversible cell.")

    # The vendored SocNavBench function returns displacement / distance (higher = more efficient).
    # This is the reciprocal of the Robot SF definition (distance / displacement). Both are
    # recorded with distinct metric IDs so they cannot be mistakenly equated.
    ratio = float(_sn_path_length_ratio(trajectory, goal))

    runner_meta = {
        "harness": "vendored_socnavbench_cost_functions.path_length_ratio",
        "trajectory_points": int(trajectory.shape[0]),
        "traversible_resolution_m": float(traversible_resolution),
        "traversible_shape": [int(dim) for dim in traversible.shape],
        "traversible_points_checked": int(points.shape[0]),
        "synthetic_traversible": synthetic_traversible,
    }
    return ratio, runner_meta


def run_canary(*, out_dir: Path, allow_synthetic_traversible: bool = False) -> dict[str, Any]:
    """Run the one-scenario / one-seed / two-suite canary and emit a joint receipt.

    The joint receipt contains the policy identity (with runtime provenance), scenario mapping,
    seed, per-suite distinct metric definitions, source commits/config digests, asset IDs, and
    limitation flags. It fails closed (raises ``CanaryError``) on any missing asset, placeholder
    metadata, policy mismatch (fallback), unsupported scenario conversion, or denominator drift.

    The pinned policy is executed exactly ONCE; the resulting trajectory is reused for both the
    Robot SF receipt (distance/displacement) and the SocNavBench receipt (displacement/distance)
    so both sides operate on identical geometric paths and their denominators match.

    Args:
        out_dir: Directory for the materialized export and the joint receipt JSON.
        allow_synthetic_traversible: Test-only escape for the no-licensed-data check. The real
            canary never sets this; doing so is a fallback and is recorded in the receipt.

    Returns:
        JSON-safe joint receipt dictionary.
    """
    policy = resolve_pinned_policy()
    mapping = resolve_scenario_mapping()

    # Execute the pinned policy ONCE and share the rollout across both suite receipts so the
    # trajectory, goal, and denominator are identical on both sides.
    try:
        rollout = execute_pinned_policy(seed=mapping.seed, algo_config=PINNED_POLICY_ALGO_CONFIG)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise CanaryError(f"Robot SF policy rollout failed: {exc}") from exc

    export_path, _ = materialize_socnavbench_export(
        policy=policy, mapping=mapping, out_dir=out_dir, rollout=rollout
    )
    robot_sf_receipt = run_robot_sf_receipt_from_rollout(
        policy=policy, mapping=mapping, rollout=rollout
    )
    socnavbench_receipt = run_socnavbench_receipt(
        export_path=export_path,
        policy=policy,
        mapping=mapping,
        allow_synthetic_traversible=allow_synthetic_traversible,
    )

    # Fallback detector: the pinned policy identity must be byte-identical across suites.
    if not policy.matches(_policy_from_receipt(robot_sf_receipt)):
        raise CanaryError("Robot SF policy identity does not match the pinned policy (fallback).")
    if not policy.matches(_policy_from_receipt(socnavbench_receipt)):
        raise CanaryError(
            "SocNavBench policy identity does not match the pinned policy (fallback)."
        )
    if robot_sf_receipt["policy_identity"] != socnavbench_receipt["policy_identity"]:
        raise CanaryError("Policy identity differs between suites (fallback detected).")

    # Reciprocal ratio guard: the two suites must carry distinct metric identities and
    # reciprocal directions. Sharing an exact metric identity for opposite definitions is a bug.
    if (
        robot_sf_receipt["metric_spec"]["metric_id"]
        == socnavbench_receipt["metric_spec"]["metric_id"]
    ):
        raise CanaryError(
            "Robot SF and SocNavBench share a colliding metric id for reciprocal ratios."
        )
    if (
        robot_sf_receipt["metric_spec"]["ratio_direction"]
        == socnavbench_receipt["metric_spec"]["ratio_direction"]
    ):
        raise CanaryError("Robot SF and SocNavBench ratio directions must differ (reciprocal).")

    # Denominator guard: both suites use the same geometric start->goal displacement; it must
    # not silently change between suites (guaranteed by sharing the single rollout).
    if not _denominators_preserved(robot_sf_receipt, socnavbench_receipt):
        raise CanaryError("Suite-specific denominators differ between suites.")

    # Fallback status: any synthetic traversible path disqualifies the receipt as success
    # evidence. ``fallback_forbidden`` is True only when no fallback path was taken.
    any_fallback = bool(socnavbench_receipt.get("is_fallback", False))
    fallback_forbidden = not any_fallback and not allow_synthetic_traversible

    joint_receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "canary": CANARY_NAME,
        "canary_version": CANARY_VERSION,
        "policy_identity": policy.to_dict(),
        "scenario_mapping": mapping.to_dict(),
        "seed": mapping.seed,
        "export_path": export_path.as_posix(),
        "external_asset_id": mapping.external_asset_id,
        "external_asset_staged": bool(socnavbench_receipt.get("external_asset_staged", False)),
        "allow_synthetic_traversible": bool(allow_synthetic_traversible),
        # fallback_forbidden is True only when no fallback path was taken.
        # When allow_synthetic_traversible=True, this is False because fallback IS active.
        "fallback_forbidden": fallback_forbidden,
        "counts_as_success_evidence": (not any_fallback),
        "suites": [robot_sf_receipt, socnavbench_receipt],
        "per_suite_metric_specs": {
            robot_sf_receipt["suite"]: robot_sf_receipt["metric_spec"],
            socnavbench_receipt["suite"]: socnavbench_receipt["metric_spec"],
        },
        "per_suite_denominators": {
            robot_sf_receipt["suite"]: robot_sf_receipt["denominator"],
            socnavbench_receipt["suite"]: socnavbench_receipt["denominator"],
        },
        "metric_values": {
            robot_sf_receipt["suite"]: robot_sf_receipt["value"],
            socnavbench_receipt["suite"]: socnavbench_receipt["value"],
        },
        "policy_identity_match": True,
        "denominators_preserved": True,
        "claim_boundary": CROSS_BENCHMARK_CLAIM_BOUNDARY,
    }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = out_dir / "cross_suite_canary_receipt.json"
    joint_receipt["receipt_path"] = receipt_path.as_posix()
    receipt_path.write_text(
        json.dumps(joint_receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return joint_receipt


def _policy_from_receipt(receipt: dict[str, Any]) -> PinnedPolicy:
    """Rebuild a ``PinnedPolicy`` from a receipt's identity block.

    Returns:
        The policy identity reconstructed from the receipt's identity block.
    """
    identity = receipt["policy_identity"]
    return PinnedPolicy(
        policy_id=identity["policy_id"],
        version=identity["version"],
        algo=identity["algo"],
        algo_config=identity["algo_config"],
        config_digest_sha256=identity["config_digest_sha256"],
        source_commit=identity["source_commit"],
        runtime_planner_config=dict(identity.get("runtime_planner_config", {})),
    )


def _denominators_preserved(
    robot_sf_receipt: dict[str, Any], socnavbench_receipt: dict[str, Any]
) -> bool:
    """Return True when both suites report the same recorded denominator value."""
    return float(robot_sf_receipt["denominator"]) == float(socnavbench_receipt["denominator"])


__all__ = [
    "CANARY_NAME",
    "CANARY_VERSION",
    "EXPORT_SCHEMA_VERSION",
    "RECEIPT_SCHEMA_VERSION",
    "ROBOT_SF_METRIC_ID",
    "SOCNAVBENCH_METRIC_ID",
    "CanaryError",
    "PinnedPolicy",
    "ScenarioMapping",
    "SuiteMetricSpec",
    "materialize_socnavbench_export",
    "resolve_pinned_policy",
    "resolve_scenario_mapping",
    "run_canary",
    "run_robot_sf_receipt",
    "run_robot_sf_receipt_from_rollout",
    "run_socnavbench_receipt",
]
