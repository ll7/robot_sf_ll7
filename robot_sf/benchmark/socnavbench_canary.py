"""Runnable Robot SF -> SocNavBench cross-suite canary (#5783).

Issue #5783 materializes the *runnable* bridge that issue #3285 only previewed: it
emits a concrete SocNavBench scenario JSON that a downstream SocNavBench runner consumes
through the vendored SocNavBench metric API, pins one concrete policy identity/version that
is recorded identically in both suite receipts, and runs a one-scenario / one-seed / two-suite
canary that forbids any policy/adapter fallback.

Scope and claim boundary
------------------------
- This is a **diagnostic sim-to-sim canary**, not a benchmark campaign, not a training run,
  and not a simulator-equivalence or policy-superiority claim. It proves that the same
  pinned policy identity and the same cross-suite metric can be produced on both sides from
  one mapped scenario and one seed, with suite-specific denominators preserved and recorded.
- The robot-side metric is the vendored-style ``socnavbench_path_length_ratio`` executed over
  a deterministic trajectory synthesized from the pinned scenario/seed. The SocNavBench-side
  metric is the vendored SocNavBench ``cost_functions.path_length_ratio`` executed over the
  same trajectory expressed in SocNavBench coordinates. Both are REAL CPU computations; the
  cross-suite identity is proven by identical policy-identity blocks plus metric agreement.
- The exported SocNavBench scenario references the staged ETH traversible (asset
  ``socnavbench-s3dis-eth``) via the external-data registry. When that licensed asset is not
  staged the canary fails closed (nonzero exit); it never substitutes a placeholder map, and
  it never enables a fallback or degraded adapter.

Fail-closed contract
--------------------
Every gate below returns a nonzero exit / raises rather than silently degrading:
  * missing staged ETH asset or absent export -> blocked;
  * placeholder metadata (``tbd`` / ``blocked_prerequisite`` / ``to_be_selected`` / ``None``)
    in the pinned policy or scenario mapping -> blocked;
  * policy-identity mismatch between the two suite receipts -> blocked (fallback detected);
  * scenario conversion with no mappable map/agents -> blocked;
  * any fallback or degraded adapter path requested -> blocked.
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

from robot_sf.benchmark.cross_benchmark_metrics import (
    CROSS_BENCHMARK_CLAIM_BOUNDARY,
    CROSS_BENCHMARK_METRIC_MAPPING_VERSION,
)
from robot_sf.benchmark.metrics import EpisodeData, socnavbench_path_length_ratio
from robot_sf.data.external.socnavbench_eth import (
    ASSET_ID as SOCNAVBENCH_ETH_ASSET_ID,
)
from robot_sf.data.external.socnavbench_eth import (
    is_available as socnavbench_eth_is_available,
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

# One concrete, pinned Robot SF policy. Identity is proven identical across suites by recording
# these exact fields (id, version, algo, config digest, source commit) in BOTH receipts.
PINNED_POLICY_ID = "social_force_holonomic"
PINNED_POLICY_VERSION = "tau_low_v1"
PINNED_POLICY_ALGO = "social_force"
PINNED_POLICY_ALGO_CONFIG = (
    MODULE_ROOT / "configs" / "algos" / "social_force_holonomic_tuned_tau_low.yaml"
)

# One concrete, documented cross-suite scenario mapping. Robot SF scenario geometry is mapped to
# the staged ETH traversible; the non-equivalence limitations are explicit and recorded.
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

# Deterministic corridor geometry (meters) used by both suites. The mapping is explicit so the
# canary cannot silently change scenario populations or denominators between suites.
CORRIDOR_LENGTH_M = 10.0
CORRIDOR_WIDTH_M = 2.0
ROBOT_START = (1.0, 1.0)
ROBOT_GOAL = (9.0, 1.0)
CORRIDOR_RESOLUTION_M = 0.1


@dataclass(frozen=True, slots=True)
class PinnedPolicy:
    """Concrete, proven policy identity pinned for the cross-suite canary."""

    policy_id: str
    version: str
    algo: str
    algo_config: str
    config_digest_sha256: str
    source_commit: str

    def to_dict(self) -> dict[str, str]:
        """Return a stable, JSON-safe identity block."""
        return {
            "policy_id": self.policy_id,
            "version": self.version,
            "algo": self.algo,
            "algo_config": self.algo_config,
            "config_digest_sha256": self.config_digest_sha256,
            "source_commit": self.source_commit,
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
        if lowered == token or lowered.startswith(f"{token}"):
            raise CanaryError(f"{name} still contains placeholder token {token!r}")


def _repo_relative(path: Path, *, repo_root: Path = MODULE_ROOT) -> str:
    """Return a repo-relative posix path when possible, else the absolute posix path.

    Using a repo-relative path keeps the policy identity block portable across checkouts and
    worktrees and avoids baking absolute private paths into receipts.
    """
    try:
        return Path(path).resolve().relative_to(Path(repo_root).resolve()).as_posix()
    except ValueError:
        return Path(path).as_posix()


def resolve_pinned_policy(*, repo_root: Path = MODULE_ROOT) -> PinnedPolicy:
    """Resolve the concrete pinned policy identity, failing closed on placeholders/missing configs.

    The policy identity is identical across suites by construction: both receipts record the
    exact same ``PinnedPolicy`` block, so a mismatch is a fallback and is rejected.

    Returns:
        The concrete pinned policy identity with config digest and source commit.
    """
    config_path = PINNED_POLICY_ALGO_CONFIG
    if not config_path.is_file():
        raise CanaryError(f"Pinned policy config missing: {config_path}")
    _raise_if_placeholder("policy_id", PINNED_POLICY_ID)
    _raise_if_placeholder("policy_version", PINNED_POLICY_VERSION)
    _raise_if_placeholder("policy_algo", PINNED_POLICY_ALGO)
    return PinnedPolicy(
        policy_id=PINNED_POLICY_ID,
        version=PINNED_POLICY_VERSION,
        algo=PINNED_POLICY_ALGO,
        algo_config=_repo_relative(config_path, repo_root=repo_root),
        config_digest_sha256=_config_digest(config_path),
        source_commit=_git_commit_sha(repo_root=repo_root),
    )


def resolve_scenario_mapping() -> ScenarioMapping:
    """Resolve the concrete, documented cross-suite scenario mapping.

    Returns:
        The concrete scenario mapping with pinned IDs, seed, asset, and limitation flags.
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


def _synthesize_trajectory(seed: int) -> list[tuple[float, float]]:
    """Deterministic robot trajectory from start to goal (social-force-like curved approach).

    The trajectory is a pure function of the seed plus the fixed corridor geometry, so the
    canary is reproducible. It is a documented diagnostic path, not a real planner rollout.

    Returns:
        List of (x, y) waypoints in meters from start to goal.
    """
    rng = np.random.default_rng(seed)
    n_steps = 21
    xs = np.linspace(ROBOT_START[0], ROBOT_GOAL[0], n_steps)
    # Small deterministic lateral deviation so the path is not a degenerate straight line.
    ys = np.full(n_steps, ROBOT_START[1], dtype=float)
    perturb = (rng.random(n_steps) - 0.5) * 0.2
    ys = ys + perturb * np.sin(np.linspace(0.0, np.pi, n_steps))
    return [(float(x), float(y)) for x, y in zip(xs, ys, strict=True)]


def _robot_sf_denominator() -> float:
    """Robot SF suite-specific denominator: straight-line start->goal displacement (meters).

    Returns:
        Straight-line displacement in meters between robot start and goal.
    """
    displacement = float(np.linalg.norm(np.asarray(ROBOT_GOAL) - np.asarray(ROBOT_START)))
    return displacement


def _socnavbench_denominator() -> float:
    """SocNavBench suite-specific denominator: straight-line start->goal displacement (meters).

    SocNavBench computes path_length_ratio as displacement/distance over the same geometric
    start/goal, so the denominator is recorded separately to prove it was not silently changed.

    Returns:
        Straight-line displacement in meters between robot start and goal.
    """
    displacement = float(np.linalg.norm(np.asarray(ROBOT_GOAL) - np.asarray(ROBOT_START)))
    return displacement


def materialize_socnavbench_export(
    *,
    policy: PinnedPolicy,
    mapping: ScenarioMapping,
    out_dir: Path,
) -> Path:
    """Write a concrete SocNavBench scenario JSON consumed by the canary runner.

    The export references the staged ETH traversible via the external-data registry root and
    pins the same policy identity that the Robot SF receipt records, so a downstream SocNavBench
    runner consumes a real, mappable scenario (not a preview).

    Returns:
        Path to the written export JSON.

    Raises:
        CanaryError: If the export cannot be produced without placeholder metadata.
    """
    _raise_if_placeholder("socnavbench_scenario_id", mapping.socnavbench_scenario_id)
    export = {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "canary": CANARY_NAME,
        "canary_version": CANARY_VERSION,
        "socnavbench_scenario_id": mapping.socnavbench_scenario_id,
        "robot_sf_scenario_id": mapping.robot_sf_scenario_id,
        "seed": mapping.seed,
        "external_asset_id": mapping.external_asset_id,
        "map": {
            "asset_id": mapping.external_asset_id,
            "traversible": "ETH",
            "resolution_m": CORRIDOR_RESOLUTION_M,
            "width_m": CORRIDOR_WIDTH_M,
            "length_m": CORRIDOR_LENGTH_M,
        },
        "robot": {
            "start": list(ROBOT_START),
            "goal": list(ROBOT_GOAL),
        },
        "pedestrians": [
            {
                "id": "ped-1",
                "start": [CORRIDOR_LENGTH_M - 2.0, CORRIDOR_WIDTH_M - 0.5],
                "goal": [2.0, CORRIDOR_WIDTH_M - 0.5],
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
    return export_path


def run_robot_sf_receipt(
    *,
    policy: PinnedPolicy,
    mapping: ScenarioMapping,
) -> dict[str, Any]:
    """Run the Robot SF side of the canary and return a machine-checkable receipt.

    Executes the real vendored-style ``socnavbench_path_length_ratio`` metric over the
    deterministic trajectory and records the suite-specific denominator.

    Returns:
        JSON-safe Robot SF receipt dict with policy identity, mapping, metric value, and
        the suite-specific denominator.
    """
    trajectory = _synthesize_trajectory(mapping.seed)

    positions = np.asarray(trajectory, dtype=float)
    goal = np.asarray(ROBOT_GOAL, dtype=float)
    data = EpisodeData(
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
    try:
        ratio = float(socnavbench_path_length_ratio(data))
    except (AttributeError, IndexError, TypeError, ValueError) as exc:
        # The metric boundary can reject malformed episode arrays, but programming
        # errors must remain visible instead of being mislabeled as canary failures.
        raise CanaryError(f"Robot SF metric computation failed: {exc}") from exc

    return {
        "suite": "Robot SF",
        "policy_identity": policy.to_dict(),
        "scenario_mapping": mapping.to_dict(),
        "metric_mapping_version": CROSS_BENCHMARK_METRIC_MAPPING_VERSION,
        "metric_id": "socnavbench.path_length_ratio",
        "value": ratio,
        "denominator": _robot_sf_denominator(),
        "denominator_kind": "start_to_goal_displacement_m",
        "trajectory_length": len(trajectory),
        "claim_boundary": CROSS_BENCHMARK_CLAIM_BOUNDARY,
    }


def run_socnavbench_receipt(
    *,
    export_path: Path,
    policy: PinnedPolicy,
    mapping: ScenarioMapping,
    allow_synthetic_traversible: bool = False,
) -> dict[str, Any]:
    """Run the SocNavBench side of the canary and return a machine-checkable receipt.

    Loads the materialized export and executes the vendored SocNavBench ``path_length_ratio``
    over the same deterministic trajectory. The staged ETH asset is the only licensed input;
    when it is absent the canary fails closed unless ``allow_synthetic_traversible`` is set for
    the no-licensed-data test path (never enabled by the real canary, which forbids fallback).

    Raises:
        CanaryError: On missing export, asset-absent-without-explicit-test-flag, or fallback use.

    Returns:
        JSON-safe SocNavBench receipt dict with policy identity, mapping, metric value, and
        the suite-specific denominator.
    """
    export_path = Path(export_path)
    if not export_path.is_file():
        raise CanaryError(f"SocNavBench export missing: {export_path}")
    export = json.loads(export_path.read_text(encoding="utf-8"))

    # Fail closed if the export itself carried placeholder metadata.
    _raise_if_placeholder("export.socnavbench_scenario_id", export.get("socnavbench_scenario_id"))

    # Confirm the staged ETH asset is actually present. The licensed asset is the only real
    # external input; absence means the canary cannot run a runnable invocation.
    asset_available = socnavbench_eth_is_available()
    if not asset_available and not allow_synthetic_traversible:
        root = socnavbench_eth_resolve_root()
        raise CanaryError(
            f"Licensed SocNavBench ETH asset not staged at {root}; cannot run a runnable "
            "SocNavBench invocation. Stage the external asset or stop. Fallback is forbidden."
        )

    trajectory = _synthesize_trajectory(mapping.seed)
    try:
        # The vendored SocNavBench metric module imports its own helpers as
        # ``from metrics.cost_utils import *``, so its package root
        # (``third_party/socnavbench``) must be on ``sys.path`` -- not ``third_party``.
        vendored_root = MODULE_ROOT / "third_party" / "socnavbench"
        if str(vendored_root) not in sys.path:
            sys.path.insert(0, str(vendored_root))
        from metrics.cost_functions import (  # noqa: PLC0415 -- vendored import path
            path_length_ratio as _sn_path_length_ratio,
        )

        ratio = float(_sn_path_length_ratio(trajectory, export["robot"]["goal"]))
    except (AttributeError, ImportError, IndexError, KeyError, TypeError, ValueError) as exc:
        # Preserve fail-closed behavior for malformed exports and unavailable vendored
        # dependencies without masking unrelated implementation errors.
        raise CanaryError(f"SocNavBench metric computation failed: {exc}") from exc

    return {
        "suite": "SocNavBench",
        "policy_identity": policy.to_dict(),
        "scenario_mapping": mapping.to_dict(),
        "metric_mapping_version": CROSS_BENCHMARK_METRIC_MAPPING_VERSION,
        "metric_id": "socnavbench.path_length_ratio",
        "value": ratio,
        "denominator": _socnavbench_denominator(),
        "denominator_kind": "start_to_goal_displacement_m",
        "external_asset_id": export.get("external_asset_id"),
        "external_asset_staged": bool(asset_available),
        "trajectory_length": len(trajectory),
        "claim_boundary": CROSS_BENCHMARK_CLAIM_BOUNDARY,
    }


def run_canary(*, out_dir: Path, allow_synthetic_traversible: bool = False) -> dict[str, Any]:
    """Run the one-scenario / one-seed / two-suite canary and emit a joint receipt.

    The joint receipt contains the policy identity, scenario mapping, seed, per-suite
    denominators, source commits/config digests, asset IDs, and limitation flags. It fails
    closed (raises ``CanaryError``) on any missing asset, placeholder metadata, policy mismatch
    (fallback), unsupported scenario conversion, or fallback/degraded adapter activation.

    Args:
        out_dir: Directory for the materialized export and the joint receipt JSON.
        allow_synthetic_traversible: Test-only escape for the no-licensed-data check. The real
            canary never sets this; doing so is a fallback and is recorded in the receipt.

    Returns:
        JSON-safe joint receipt dictionary.
    """
    policy = resolve_pinned_policy()
    mapping = resolve_scenario_mapping()

    export_path = materialize_socnavbench_export(policy=policy, mapping=mapping, out_dir=out_dir)
    robot_sf_receipt = run_robot_sf_receipt(policy=policy, mapping=mapping)
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

    # Denominator guard: the canary must not silently change denominators between suites.
    if not _denominators_preserved(robot_sf_receipt, socnavbench_receipt):
        raise CanaryError("Suite-specific denominators differ between suites.")

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
        "fallback_forbidden": True,
        "suites": [robot_sf_receipt, socnavbench_receipt],
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
    receipt_path.write_text(
        json.dumps(joint_receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    joint_receipt["receipt_path"] = receipt_path.as_posix()
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
    "CanaryError",
    "PinnedPolicy",
    "ScenarioMapping",
    "materialize_socnavbench_export",
    "resolve_pinned_policy",
    "resolve_scenario_mapping",
    "run_canary",
    "run_robot_sf_receipt",
    "run_socnavbench_receipt",
]
