"""Canonical declarative algorithm contract registry (issue #7676, first slice).

One authoritative record per migrated canonical benchmark algorithm. Each
:class:`AlgorithmContractRecord` owns the algorithm's identity, aliases,
readiness, baseline category, policy semantics, observation contract, upstream
provenance, kinematics profile, and policy-builder owner. The compatibility
facades in :mod:`robot_sf.benchmark.algorithm_metadata`,
:mod:`robot_sf.benchmark.algorithm_readiness`, and
:mod:`robot_sf.benchmark.map_runner_policies.registry` source their migrated
entries from this module so a canonical name cannot drift between owners.

Bounded first slice: ``orca`` plus the SocNav ORCA/HRVO variants, the
``social_navigation_pyenvs_*`` adapters, and the ``gensafenav_*`` adapters.
All other algorithms remain on the legacy path until their family migrates;
see :func:`audit_contract_ownership` for the migration audit surface.

Later-family migration procedure:

1. Capture the family's current readiness, metadata, builder, and model/provenance
   surfaces in a fixture before changing code.
2. Add exactly one validated :class:`AlgorithmContractRecord` and include every
   existing alias; do not infer or rewrite scientific metadata during the move.
3. Declare the builder owner and extend :func:`validate_builder_agreement` when
   the family uses a new registration surface.
4. Rewire compatibility facades to read the record, then add snapshot, alias,
   builder, ordering, and emitted-metadata parity tests.
5. Run :func:`audit_contract_ownership`, the focused benchmark suites, and the
   relevant provenance/readiness gates before migrating another family.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

AlgorithmTier = Literal["baseline-ready", "experimental", "placeholder"]
PolicyBuilderOwner = Literal["legacy_map_runner", "map_runner_policies.socnav_family"]

_RECORD_FIELDS: tuple[str, ...] = (
    "canonical_name",
    "aliases",
    "tier",
    "note",
    "requires_explicit_opt_in",
    "baseline_category",
    "policy_semantics",
    "observation_spec",
    "upstream_reference",
    "kinematics_profile",
    "paper_baseline_eligible",
    "policy_builder_owner",
)
_ALLOWED_TIERS = frozenset({"baseline-ready", "experimental", "placeholder"})
_ALLOWED_POLICY_BUILDER_OWNERS = frozenset(
    {"legacy_map_runner", "map_runner_policies.socnav_family"}
)


@dataclass(frozen=True)
class AlgorithmContractRecord:
    """Strict schema for one canonical algorithm's full benchmark contract.

    Construct records with :meth:`from_mapping` so unknown or missing fields
    fail closed instead of silently dropping contract metadata.
    """

    canonical_name: str
    aliases: tuple[str, ...]
    tier: AlgorithmTier
    note: str
    requires_explicit_opt_in: bool
    baseline_category: str
    policy_semantics: str
    observation_spec: dict[str, Any]
    upstream_reference: dict[str, Any]
    kinematics_profile: dict[str, Any]
    paper_baseline_eligible: bool = False
    policy_builder_owner: PolicyBuilderOwner = "legacy_map_runner"

    def __post_init__(self) -> None:
        """Validate direct construction as well as mapping-based construction."""
        _validate_record(self)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> AlgorithmContractRecord:
        """Build a record from a mapping, rejecting unknown or missing fields.

        Returns:
            AlgorithmContractRecord: The validated frozen record.

        Raises:
            ValueError: When unknown keys are present or required keys are
                missing, when alias/canonical invariants are violated, or when
                required metadata payloads are not non-empty mappings.
        """
        if not isinstance(mapping, Mapping):
            raise ValueError("algorithm contract must be provided as a mapping")
        unknown = sorted(set(mapping) - set(_RECORD_FIELDS))
        if unknown:
            raise ValueError(f"Unknown algorithm-contract fields: {unknown}")
        missing = [
            name
            for name in _RECORD_FIELDS
            if name not in mapping and name != "paper_baseline_eligible"
        ]
        if missing:
            raise ValueError(f"Missing required algorithm-contract fields: {missing}")
        aliases = mapping["aliases"]
        if not isinstance(aliases, (list, tuple)):
            raise ValueError("aliases must be a list or tuple of normalized strings")
        payloads = {
            name: mapping[name]
            for name in ("observation_spec", "upstream_reference", "kinematics_profile")
        }
        for name, payload in payloads.items():
            if not isinstance(payload, Mapping):
                raise ValueError(f"{name} must be a mapping")
        record = cls(
            canonical_name=mapping["canonical_name"],
            aliases=tuple(aliases),
            tier=mapping["tier"],
            note=mapping["note"],
            requires_explicit_opt_in=mapping["requires_explicit_opt_in"],
            baseline_category=mapping["baseline_category"],
            policy_semantics=mapping["policy_semantics"],
            observation_spec=copy.deepcopy(dict(payloads["observation_spec"])),
            upstream_reference=copy.deepcopy(dict(payloads["upstream_reference"])),
            kinematics_profile=copy.deepcopy(dict(payloads["kinematics_profile"])),
            paper_baseline_eligible=mapping.get("paper_baseline_eligible", False),
            policy_builder_owner=mapping.get("policy_builder_owner", "legacy_map_runner"),
        )
        return record

    def snapshot(self) -> dict[str, Any]:
        """Return a deep-copied plain-data view of the record."""
        return {
            "canonical_name": self.canonical_name,
            "aliases": list(self.aliases),
            "tier": self.tier,
            "note": self.note,
            "requires_explicit_opt_in": self.requires_explicit_opt_in,
            "baseline_category": self.baseline_category,
            "policy_semantics": self.policy_semantics,
            "observation_spec": copy.deepcopy(self.observation_spec),
            "upstream_reference": copy.deepcopy(self.upstream_reference),
            "kinematics_profile": copy.deepcopy(self.kinematics_profile),
            "paper_baseline_eligible": self.paper_baseline_eligible,
            "policy_builder_owner": self.policy_builder_owner,
        }


def _require_non_empty_text(value: Any, *, field: str, canonical_name: str) -> None:
    """Require one contract scalar to be a non-empty string."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{canonical_name}: {field} must be a non-empty string")


def _validate_record_identity(record: AlgorithmContractRecord) -> None:
    """Validate names, readiness fields, and builder ownership."""
    if not isinstance(record.canonical_name, str):
        raise ValueError("canonical_name must be a string")
    if not record.canonical_name or record.canonical_name != record.canonical_name.strip().lower():
        raise ValueError(f"canonical_name must be normalized lowercase: {record.canonical_name!r}")
    if not isinstance(record.tier, str) or record.tier not in _ALLOWED_TIERS:
        raise ValueError(f"{record.canonical_name}: unknown readiness tier {record.tier!r}")
    _require_non_empty_text(record.note, field="note", canonical_name=record.canonical_name)
    if not isinstance(record.requires_explicit_opt_in, bool):
        raise ValueError(f"{record.canonical_name}: requires_explicit_opt_in must be boolean")
    _require_non_empty_text(
        record.baseline_category,
        field="baseline_category",
        canonical_name=record.canonical_name,
    )
    _require_non_empty_text(
        record.policy_semantics,
        field="policy_semantics",
        canonical_name=record.canonical_name,
    )
    if not isinstance(record.paper_baseline_eligible, bool):
        raise ValueError(f"{record.canonical_name}: paper_baseline_eligible must be boolean")
    if (
        not isinstance(record.policy_builder_owner, str)
        or record.policy_builder_owner not in _ALLOWED_POLICY_BUILDER_OWNERS
    ):
        raise ValueError(
            f"{record.canonical_name}: unknown policy_builder_owner {record.policy_builder_owner!r}"
        )


def _validate_record_aliases(record: AlgorithmContractRecord) -> None:
    """Validate normalized, unique aliases for one canonical record."""
    if not isinstance(record.aliases, tuple) or not record.aliases:
        raise ValueError(f"{record.canonical_name}: aliases must be a non-empty tuple")
    if any(
        not isinstance(alias, str) or not alias or alias != alias.strip().lower()
        for alias in record.aliases
    ):
        raise ValueError(f"{record.canonical_name}: aliases must be normalized lowercase strings")
    if len(set(record.aliases)) != len(record.aliases):
        raise ValueError(f"{record.canonical_name}: aliases must be unique")
    if record.canonical_name not in record.aliases:
        raise ValueError(f"{record.canonical_name}: aliases must include the canonical name")


def _validate_record_payloads(record: AlgorithmContractRecord) -> None:
    """Validate required fields in observation, provenance, and execution payloads."""
    required_payload_fields = {
        "observation_spec": ("default_mode", "supported_modes", "inputs", "notes"),
        "upstream_reference": ("repo_url", "adapter_boundary"),
        "kinematics_profile": ("planner_command_space", "default_execution_mode"),
    }
    for payload_name, required_fields in required_payload_fields.items():
        payload = getattr(record, payload_name)
        if not isinstance(payload, dict) or not payload:
            raise ValueError(f"{record.canonical_name}: {payload_name} must be a non-empty mapping")
        missing_fields = [field for field in required_fields if field not in payload]
        if missing_fields:
            raise ValueError(
                f"{record.canonical_name}: {payload_name} is missing required fields "
                f"{missing_fields}"
            )


def _validate_record(record: AlgorithmContractRecord) -> None:
    """Validate one record's internal invariants; raise on violation."""
    _validate_record_identity(record)
    _validate_record_aliases(record)
    _validate_record_payloads(record)


def build_alias_index(
    records: tuple[AlgorithmContractRecord, ...],
) -> dict[str, AlgorithmContractRecord]:
    """Build the deterministic alias index, failing closed on collisions.

    Returns:
        dict[str, AlgorithmContractRecord]: Alias-keyed record index in
        declaration order.

    Raises:
        ValueError: On duplicate canonical names, duplicate aliases, an alias
            equal to another record's canonical name, or an empty registry.
    """
    if not records:
        raise ValueError("algorithm contract registry must contain at least one record")
    canonical_names = {record.canonical_name for record in records}
    if len(canonical_names) != len(records):
        duplicates = sorted(
            name for name in canonical_names if sum(r.canonical_name == name for r in records) > 1
        )
        raise ValueError(f"duplicate canonical algorithm name: {duplicates}")
    index: dict[str, AlgorithmContractRecord] = {}
    for record in records:
        for alias in record.aliases:
            if alias != record.canonical_name and alias in canonical_names:
                raise ValueError(f"alias '{alias}' collides with a canonical algorithm name")
            if alias in index:
                raise ValueError(f"duplicate algorithm alias: {alias}")
            index[alias] = record
    return index


MIGRATED_ALGORITHM_RECORDS: tuple[AlgorithmContractRecord, ...] = (
    AlgorithmContractRecord(
        canonical_name="orca",
        aliases=("orca",),
        tier="baseline-ready",
        note="ORCA baseline (requires rvo2 or explicit fallback policy).",
        requires_explicit_opt_in=False,
        baseline_category="classical",
        policy_semantics="orca_adapter",
        observation_spec={
            "default_mode": "socnav_state",
            "supported_modes": ("socnav_state",),
            "inputs": ("robot_state", "goal", "pedestrians"),
            "notes": "Structured Robot SF social-navigation state: robot pose/velocity, route goal, "
            "and pedestrian state when present.",
        },
        upstream_reference={
            "repo_url": "https://github.com/mit-acl/Python-RVO2",
            "commit": "56b245132ea104ee8a621ddf65b8a3dd85028ed2",
            "vendored_path": "third_party/python-rvo2",
            "adapter_boundary": "Use upstream Python-RVO2 to solve reciprocal-avoidance velocity in "
            "world coordinates, then project the selected velocity into Robot SF "
            "unicycle_vw commands.",
        },
        kinematics_profile={
            "planner_command_space": "unicycle_vw",
            "supports_native_commands": False,
            "supports_adapter_commands": True,
            "default_execution_mode": "adapter",
            "default_adapter_name": "ORCAPlannerAdapter",
            "upstream_command_space": "velocity_vector_xy",
            "benchmark_command_space": "unicycle_vw",
            "projection_policy": "heading_safe_velocity_to_unicycle_vw",
            "projection_documented": True,
        },
        paper_baseline_eligible=True,
        policy_builder_owner="legacy_map_runner",
    ),
    AlgorithmContractRecord(
        canonical_name="socnav_orca_nonholonomic",
        aliases=("socnav_orca_nonholonomic",),
        tier="experimental",
        note="SocNav ORCA variant tuned for nonholonomic commitment.",
        requires_explicit_opt_in=True,
        baseline_category="classical",
        policy_semantics="orca_adapter",
        observation_spec={
            "default_mode": "socnav_state",
            "supported_modes": ("socnav_state",),
            "inputs": ("robot_state", "goal", "pedestrians"),
            "notes": "Structured Robot SF social-navigation state: robot pose/velocity, route goal, "
            "and pedestrian state when present.",
        },
        upstream_reference={
            "repo_url": "https://github.com/mit-acl/Python-RVO2",
            "commit": "56b245132ea104ee8a621ddf65b8a3dd85028ed2",
            "vendored_path": "third_party/python-rvo2",
            "adapter_boundary": "Use upstream Python-RVO2 to solve reciprocal-avoidance velocity in "
            "world coordinates, apply nonholonomic commitment heuristics, and "
            "project the selected velocity into Robot SF unicycle_vw commands.",
        },
        kinematics_profile={
            "planner_command_space": "unicycle_vw",
            "supports_native_commands": False,
            "supports_adapter_commands": True,
            "default_execution_mode": "adapter",
            "default_adapter_name": "ORCAPlannerAdapter",
            "upstream_command_space": "velocity_vector_xy",
            "benchmark_command_space": "unicycle_vw",
            "projection_policy": "heading_safe_velocity_to_unicycle_vw",
            "projection_documented": True,
        },
        policy_builder_owner="map_runner_policies.socnav_family",
    ),
    AlgorithmContractRecord(
        canonical_name="socnav_orca_dd",
        aliases=("socnav_orca_dd",),
        tier="experimental",
        note="SocNav ORCA variant tuned for differential-drive compatibility.",
        requires_explicit_opt_in=True,
        baseline_category="classical",
        policy_semantics="orca_adapter",
        observation_spec={
            "default_mode": "socnav_state",
            "supported_modes": ("socnav_state",),
            "inputs": ("robot_state", "goal", "pedestrians"),
            "notes": "Structured Robot SF social-navigation state: robot pose/velocity, route goal, "
            "and pedestrian state when present.",
        },
        upstream_reference={
            "repo_url": "https://github.com/mit-acl/Python-RVO2",
            "commit": "56b245132ea104ee8a621ddf65b8a3dd85028ed2",
            "vendored_path": "third_party/python-rvo2",
            "adapter_boundary": "Use upstream Python-RVO2 to solve reciprocal-avoidance velocity in "
            "world coordinates, tune the result for differential-drive "
            "compatibility, and project it into Robot SF unicycle_vw commands.",
        },
        kinematics_profile={
            "planner_command_space": "unicycle_vw",
            "supports_native_commands": False,
            "supports_adapter_commands": True,
            "default_execution_mode": "adapter",
            "default_adapter_name": "ORCAPlannerAdapter",
            "upstream_command_space": "velocity_vector_xy",
            "benchmark_command_space": "unicycle_vw",
            "projection_policy": "heading_safe_velocity_to_unicycle_vw",
            "projection_documented": True,
        },
        policy_builder_owner="map_runner_policies.socnav_family",
    ),
    AlgorithmContractRecord(
        canonical_name="socnav_orca_relaxed",
        aliases=("socnav_orca_relaxed",),
        tier="experimental",
        note="SocNav ORCA variant with relaxed safety tuning.",
        requires_explicit_opt_in=True,
        baseline_category="classical",
        policy_semantics="orca_adapter",
        observation_spec={
            "default_mode": "socnav_state",
            "supported_modes": ("socnav_state",),
            "inputs": ("robot_state", "goal", "pedestrians"),
            "notes": "Structured Robot SF social-navigation state: robot pose/velocity, route goal, "
            "and pedestrian state when present.",
        },
        upstream_reference={
            "repo_url": "https://github.com/mit-acl/Python-RVO2",
            "commit": "56b245132ea104ee8a621ddf65b8a3dd85028ed2",
            "vendored_path": "third_party/python-rvo2",
            "adapter_boundary": "Use upstream Python-RVO2 to solve reciprocal-avoidance velocity in "
            "world coordinates, apply relaxed safety tuning, and project the "
            "selected velocity into Robot SF unicycle_vw commands.",
        },
        kinematics_profile={
            "planner_command_space": "unicycle_vw",
            "supports_native_commands": False,
            "supports_adapter_commands": True,
            "default_execution_mode": "adapter",
            "default_adapter_name": "ORCAPlannerAdapter",
            "upstream_command_space": "velocity_vector_xy",
            "benchmark_command_space": "unicycle_vw",
            "projection_policy": "heading_safe_velocity_to_unicycle_vw",
            "projection_documented": True,
        },
        policy_builder_owner="map_runner_policies.socnav_family",
    ),
    AlgorithmContractRecord(
        canonical_name="socnav_hrvo",
        aliases=("socnav_hrvo",),
        tier="experimental",
        note="SocNav HRVO variant with hybrid reciprocal velocity obstacles.",
        requires_explicit_opt_in=True,
        baseline_category="classical",
        policy_semantics="hybrid_reciprocal_velocity_obstacle",
        observation_spec={
            "default_mode": "socnav_state",
            "supported_modes": ("socnav_state",),
            "inputs": ("robot_state", "goal", "pedestrians"),
            "notes": "Structured Robot SF social-navigation state: robot pose/velocity, route goal, "
            "and pedestrian state when present.",
        },
        upstream_reference={
            "repo_url": "https://github.com/snape/HRVO",
            "license": "Apache-2.0",
            "reference_repo_url": "https://github.com/atb033/multi_agent_path_planning/blob/master/decentralized/velocity_obstacle/velocity_obstacle.py",
            "adapter_boundary": "Run the local Robot SF HRVO geometry solver inspired by the upstream "
            "HRVO library, then project the selected world-frame velocity into "
            "Robot SF unicycle_vw commands.",
            "provenance_note": "Local implementation informed by upstream references; not a wrapped "
            "upstream runtime.",
        },
        kinematics_profile={
            "planner_command_space": "unicycle_vw",
            "supports_native_commands": False,
            "supports_adapter_commands": True,
            "default_execution_mode": "adapter",
            "default_adapter_name": "HRVOPlannerAdapter",
            "upstream_command_space": "velocity_vector_xy",
            "benchmark_command_space": "unicycle_vw",
            "projection_policy": "heading_safe_velocity_to_unicycle_vw",
            "projection_documented": True,
        },
        policy_builder_owner="map_runner_policies.socnav_family",
    ),
    AlgorithmContractRecord(
        canonical_name="hrvo",
        aliases=("hrvo",),
        tier="experimental",
        note="Local hybrid reciprocal velocity obstacles planner.",
        requires_explicit_opt_in=True,
        baseline_category="classical",
        policy_semantics="hybrid_reciprocal_velocity_obstacle",
        observation_spec={
            "default_mode": "socnav_state",
            "supported_modes": ("socnav_state",),
            "inputs": ("robot_state", "goal", "pedestrians"),
            "notes": "Structured Robot SF social-navigation state: robot pose/velocity, route goal, "
            "and pedestrian state when present.",
        },
        upstream_reference={
            "repo_url": "https://github.com/snape/HRVO",
            "license": "Apache-2.0",
            "reference_repo_url": "https://github.com/atb033/multi_agent_path_planning/blob/master/decentralized/velocity_obstacle/velocity_obstacle.py",
            "adapter_boundary": "Run the local Robot SF HRVO geometry solver inspired by the upstream "
            "HRVO library and VO reference, then project the selected world-frame "
            "velocity into Robot SF unicycle_vw commands.",
            "provenance_note": "Local implementation informed by upstream references; not a wrapped "
            "upstream runtime.",
        },
        kinematics_profile={
            "planner_command_space": "unicycle_vw",
            "supports_native_commands": False,
            "supports_adapter_commands": True,
            "default_execution_mode": "adapter",
            "default_adapter_name": "HRVOPlannerAdapter",
            "upstream_command_space": "velocity_vector_xy",
            "benchmark_command_space": "unicycle_vw",
            "projection_policy": "heading_safe_velocity_to_unicycle_vw",
            "projection_documented": True,
        },
        policy_builder_owner="legacy_map_runner",
    ),
    AlgorithmContractRecord(
        canonical_name="social_navigation_pyenvs_orca",
        aliases=(
            "social_navigation_pyenvs_orca",
            "social_nav_pyenvs_orca",
        ),
        tier="experimental",
        note="Upstream Social-Navigation-PyEnvs non-trainable ORCA wrapper.",
        requires_explicit_opt_in=False,
        baseline_category="classical",
        policy_semantics="upstream_social_navigation_pyenvs_orca_wrapper",
        observation_spec={
            "default_mode": "socnav_state",
            "supported_modes": ("socnav_state",),
            "inputs": ("robot_state", "goal", "pedestrians"),
            "notes": "Structured Robot SF social-navigation state: robot pose/velocity, route goal, "
            "and pedestrian state when present.",
        },
        upstream_reference={
            "repo_url": "https://github.com/TommasoVandermeer/Social-Navigation-PyEnvs",
            "commit": "checked_out_local_probe_2026_03_20",
            "checkout_path": "output/repos/Social-Navigation-PyEnvs",
            "upstream_policy": "crowd_nav.policy_no_train.orca.ORCA",
            "adapter_boundary": "Map Robot SF SocNav observations into the upstream "
            "Social-Navigation-PyEnvs JointState contract, run upstream ORCA "
            "predict(), then project ActionXY into Robot SF unicycle_vw commands.",
        },
        kinematics_profile={
            "planner_command_space": "unicycle_vw",
            "supports_native_commands": False,
            "supports_adapter_commands": True,
            "default_execution_mode": "adapter",
            "default_adapter_name": "SocialNavigationPyEnvsORCAAdapter",
            "upstream_command_space": "velocity_vector_xy",
            "benchmark_command_space": "unicycle_vw",
            "projection_policy": "heading_safe_velocity_to_unicycle_vw",
            "projection_documented": True,
        },
        policy_builder_owner="legacy_map_runner",
    ),
    AlgorithmContractRecord(
        canonical_name="social_navigation_pyenvs_socialforce",
        aliases=(
            "social_navigation_pyenvs_socialforce",
            "social_nav_pyenvs_socialforce",
        ),
        tier="experimental",
        note="Upstream Social-Navigation-PyEnvs non-trainable SocialForce wrapper.",
        requires_explicit_opt_in=False,
        baseline_category="classical",
        policy_semantics="upstream_social_navigation_pyenvs_socialforce_wrapper",
        observation_spec={
            "default_mode": "socnav_state",
            "supported_modes": ("socnav_state",),
            "inputs": ("robot_state", "goal", "pedestrians"),
            "notes": "Structured Robot SF social-navigation state: robot pose/velocity, route goal, "
            "and pedestrian state when present.",
        },
        upstream_reference={
            "repo_url": "https://github.com/TommasoVandermeer/Social-Navigation-PyEnvs",
            "commit": "f9cd244d3e529247ca1031364de22954717b9493",
            "checkout_path": "output/repos/Social-Navigation-PyEnvs",
            "upstream_policy": "crowd_nav.policy_no_train.socialforce.SocialForce",
            "adapter_boundary": "Map Robot SF SocNav observations into the upstream "
            "Social-Navigation-PyEnvs JointState contract, run upstream "
            "SocialForce predict() through an explicit CrowdNav-style "
            "compatibility runtime for socialforce==0.2.3, then project ActionXY "
            "into Robot SF unicycle_vw commands.",
            "runtime_dependency": "socialforce==0.2.3",
            "runtime_strategy": "crowdnav_socialforce_compat_shim",
        },
        kinematics_profile={
            "planner_command_space": "unicycle_vw",
            "supports_native_commands": False,
            "supports_adapter_commands": True,
            "default_execution_mode": "adapter",
            "default_adapter_name": "SocialNavigationPyEnvsForceModelAdapter",
            "upstream_command_space": "velocity_vector_xy",
            "benchmark_command_space": "unicycle_vw",
            "projection_policy": "heading_safe_velocity_to_unicycle_vw",
            "projection_documented": True,
            "runtime_dependency": "socialforce==0.2.3",
            "runtime_strategy": "crowdnav_socialforce_compat_shim",
        },
        policy_builder_owner="legacy_map_runner",
    ),
    AlgorithmContractRecord(
        canonical_name="social_navigation_pyenvs_sfm_helbing",
        aliases=(
            "social_navigation_pyenvs_sfm_helbing",
            "social_nav_pyenvs_sfm_helbing",
        ),
        tier="experimental",
        note="Upstream Social-Navigation-PyEnvs non-trainable SFM-Helbing wrapper.",
        requires_explicit_opt_in=False,
        baseline_category="classical",
        policy_semantics="upstream_social_navigation_pyenvs_sfm_helbing_wrapper",
        observation_spec={
            "default_mode": "socnav_state",
            "supported_modes": ("socnav_state",),
            "inputs": ("robot_state", "goal", "pedestrians"),
            "notes": "Structured Robot SF social-navigation state: robot pose/velocity, route goal, "
            "and pedestrian state when present.",
        },
        upstream_reference={
            "repo_url": "https://github.com/TommasoVandermeer/Social-Navigation-PyEnvs",
            "commit": "f9cd244d3e529247ca1031364de22954717b9493",
            "checkout_path": "output/repos/Social-Navigation-PyEnvs",
            "upstream_policy": "crowd_nav.policy_no_train.sfm_helbing.SFMHelbing",
            "adapter_boundary": "Map Robot SF SocNav observations into the upstream "
            "Social-Navigation-PyEnvs JointState contract, run upstream "
            "SFM-Helbing predict(), then project ActionXY into Robot SF "
            "unicycle_vw commands.",
        },
        kinematics_profile={
            "planner_command_space": "unicycle_vw",
            "supports_native_commands": False,
            "supports_adapter_commands": True,
            "default_execution_mode": "adapter",
            "default_adapter_name": "SocialNavigationPyEnvsForceModelAdapter",
            "upstream_command_space": "velocity_vector_xy",
            "benchmark_command_space": "unicycle_vw",
            "projection_policy": "heading_safe_velocity_to_unicycle_vw",
            "projection_documented": True,
        },
        policy_builder_owner="legacy_map_runner",
    ),
    AlgorithmContractRecord(
        canonical_name="social_navigation_pyenvs_hsfm_new_guo",
        aliases=(
            "social_navigation_pyenvs_hsfm_new_guo",
            "social_nav_pyenvs_hsfm_new_guo",
        ),
        tier="experimental",
        note="Upstream Social-Navigation-PyEnvs non-trainable HSFM-New-Guo wrapper.",
        requires_explicit_opt_in=False,
        baseline_category="classical",
        policy_semantics="upstream_social_navigation_pyenvs_hsfm_wrapper",
        observation_spec={
            "default_mode": "headed_socnav_state",
            "supported_modes": ("headed_socnav_state",),
            "inputs": ("robot_state", "robot_heading", "goal", "pedestrians"),
            "notes": "Structured headed social-navigation state for HSFM-style adapters.",
        },
        upstream_reference={
            "repo_url": "https://github.com/TommasoVandermeer/Social-Navigation-PyEnvs",
            "commit": "f9cd244d3e529247ca1031364de22954717b9493",
            "checkout_path": "output/repos/Social-Navigation-PyEnvs",
            "upstream_policy": "crowd_nav.policy_no_train.hsfm_new_guo.HSFMNewGuo",
            "adapter_boundary": "Map Robot SF SocNav observations into the upstream "
            "Social-Navigation-PyEnvs headed JointState contract, run upstream "
            "HSFM-New-Guo predict(), then project body-frame ActionXYW or "
            "NewHeadedState outputs into Robot SF unicycle_vw commands.",
        },
        kinematics_profile={
            "planner_command_space": "unicycle_vw",
            "supports_native_commands": False,
            "supports_adapter_commands": True,
            "default_execution_mode": "adapter",
            "default_adapter_name": "SocialNavigationPyEnvsHSFMAdapter",
            "upstream_command_space": "body_velocity_xy_plus_omega",
            "benchmark_command_space": "unicycle_vw",
            "projection_policy": "body_velocity_heading_safe_to_unicycle_vw",
            "projection_documented": True,
        },
        policy_builder_owner="legacy_map_runner",
    ),
    AlgorithmContractRecord(
        canonical_name="gensafenav_ours_gst",
        aliases=(
            "gensafenav_ours_gst",
            "gensafe_ours_gst",
            "ours_gst",
        ),
        tier="experimental",
        note="Upstream GenSafeNav constrained learned checkpoint wrapper with fail-fast asset checks.",
        requires_explicit_opt_in=False,
        baseline_category="learning",
        policy_semantics="upstream_gensafenav_checkpoint_wrapper",
        observation_spec={
            "default_mode": "gst_human_state",
            "supported_modes": ("gst_human_state",),
            "inputs": ("robot_state", "goal", "humans"),
            "notes": "GenSafeNav Ours_GST checkpoint input contract.",
        },
        upstream_reference={
            "repo_url": "https://github.com/tasl-lab/GenSafeNav",
            "reference_repo_url": "https://github.com/tasl-lab/SoNIC-Social-Nav",
            "commit": "01baf92",
            "checkout_path": "output/repos/GenSafeNav",
            "default_model_name": "Ours_GST",
            "default_checkpoint": "trained_models/Ours_GST/checkpoints/05207.pt",
            "upstream_policy": "rl.networks.model.Policy[selfAttn_merge_srnn]",
            "adapter_boundary": "Map Robot SF SocNav observations into the GenSafeNav model-only dict "
            "contract, run the upstream constrained selfAttn_merge_srnn "
            "checkpoint with explicit import/runtime shims, and project upstream "
            "ActionXY velocities into Robot SF unicycle_vw commands.",
        },
        kinematics_profile={
            "planner_command_space": "unicycle_vw",
            "supports_native_commands": False,
            "supports_adapter_commands": True,
            "default_execution_mode": "adapter",
            "default_adapter_name": "SonicCrowdNavAdapter",
            "upstream_command_space": "holonomic_velocity_xy",
            "benchmark_command_space": "unicycle_vw",
            "projection_policy": "heading_safe_velocity_to_unicycle_vw",
            "projection_documented": True,
        },
        policy_builder_owner="legacy_map_runner",
    ),
    AlgorithmContractRecord(
        canonical_name="gensafenav_ours_gst_guarded",
        aliases=(
            "gensafenav_ours_gst_guarded",
            "ours_gst_guarded",
        ),
        tier="experimental",
        note=(
            "GenSafeNav Ours_GST wrapper with explicit short-horizon safety guard "
            "and goal fallback for static-risk-heavy slices."
        ),
        requires_explicit_opt_in=False,
        baseline_category="learning",
        policy_semantics="guarded_upstream_gensafenav_checkpoint_wrapper",
        observation_spec={
            "default_mode": "gst_human_state",
            "supported_modes": ("gst_human_state",),
            "inputs": ("robot_state", "goal", "humans", "safety_guard"),
            "notes": "Guarded GenSafeNav Ours_GST checkpoint input contract.",
        },
        upstream_reference={
            "repo_url": "https://github.com/tasl-lab/GenSafeNav",
            "reference_repo_url": "https://github.com/tasl-lab/SoNIC-Social-Nav",
            "commit": "01baf92",
            "checkout_path": "output/repos/GenSafeNav",
            "default_model_name": "Ours_GST",
            "default_checkpoint": "trained_models/Ours_GST/checkpoints/05207.pt",
            "upstream_policy": "rl.networks.model.Policy[selfAttn_merge_srnn]",
            "adapter_boundary": "Run the GenSafeNav model-only Ours_GST checkpoint through the "
            "SoNIC-compatible adapter contract, then apply an explicit "
            "short-horizon safety guard with goal-policy fallback before emitting "
            "Robot SF unicycle_vw commands.",
        },
        kinematics_profile={
            "planner_command_space": "mixed_vw_or_unicycle",
            "supports_native_commands": True,
            "supports_adapter_commands": True,
            "default_execution_mode": "mixed",
            "default_adapter_name": "sonic_guarded_goal_fallback",
            "upstream_command_space": "holonomic_velocity_xy",
            "benchmark_command_space": "unicycle_vw",
            "projection_policy": "heading_safe_velocity_to_unicycle_vw",
            "projection_documented": True,
        },
        policy_builder_owner="legacy_map_runner",
    ),
    AlgorithmContractRecord(
        canonical_name="gensafenav_gst_predictor_rand",
        aliases=(
            "gensafenav_gst_predictor_rand",
            "gensafe_gst_predictor_rand",
            "gst_predictor_rand",
        ),
        tier="experimental",
        note=(
            "Upstream GenSafeNav CrowdNav++-style learned "
            "checkpoint wrapper with fail-fast asset checks."
        ),
        requires_explicit_opt_in=False,
        baseline_category="learning",
        policy_semantics="upstream_gensafenav_checkpoint_wrapper",
        observation_spec={
            "default_mode": "gst_human_state",
            "supported_modes": ("gst_human_state",),
            "inputs": ("robot_state", "goal", "humans"),
            "notes": "GenSafeNav GST_predictor_rand checkpoint input contract.",
        },
        upstream_reference={
            "repo_url": "https://github.com/tasl-lab/GenSafeNav",
            "reference_repo_url": "https://github.com/tasl-lab/SoNIC-Social-Nav",
            "commit": "01baf92",
            "checkout_path": "output/repos/GenSafeNav",
            "default_model_name": "GST_predictor_rand",
            "default_checkpoint": "trained_models/GST_predictor_rand/checkpoints/05207.pt",
            "upstream_policy": "rl.networks.model.Policy[selfAttn_merge_srnn]",
            "adapter_boundary": "Map Robot SF SocNav observations into the GenSafeNav "
            "CrowdNav++-style model-only dict contract, run the upstream "
            "selfAttn_merge_srnn checkpoint with explicit import/runtime shims, "
            "and project upstream ActionXY velocities into Robot SF unicycle_vw "
            "commands.",
        },
        kinematics_profile={
            "planner_command_space": "unicycle_vw",
            "supports_native_commands": False,
            "supports_adapter_commands": True,
            "default_execution_mode": "adapter",
            "default_adapter_name": "SonicCrowdNavAdapter",
            "upstream_command_space": "holonomic_velocity_xy",
            "benchmark_command_space": "unicycle_vw",
            "projection_policy": "heading_safe_velocity_to_unicycle_vw",
            "projection_documented": True,
        },
        policy_builder_owner="legacy_map_runner",
    ),
    AlgorithmContractRecord(
        canonical_name="gensafenav_gst_predictor_rand_guarded",
        aliases=(
            "gensafenav_gst_predictor_rand_guarded",
            "gst_predictor_rand_guarded",
        ),
        tier="experimental",
        note=(
            "GenSafeNav GST_predictor_rand wrapper with explicit short-horizon safety "
            "guard and goal fallback for static-risk-heavy slices."
        ),
        requires_explicit_opt_in=False,
        baseline_category="learning",
        policy_semantics="guarded_upstream_gensafenav_checkpoint_wrapper",
        observation_spec={
            "default_mode": "gst_human_state",
            "supported_modes": ("gst_human_state",),
            "inputs": ("robot_state", "goal", "humans", "safety_guard"),
            "notes": "Guarded GenSafeNav GST_predictor_rand checkpoint input contract.",
        },
        upstream_reference={
            "repo_url": "https://github.com/tasl-lab/GenSafeNav",
            "reference_repo_url": "https://github.com/tasl-lab/SoNIC-Social-Nav",
            "commit": "01baf92",
            "checkout_path": "output/repos/GenSafeNav",
            "default_model_name": "GST_predictor_rand",
            "default_checkpoint": "trained_models/GST_predictor_rand/checkpoints/05207.pt",
            "upstream_policy": "rl.networks.model.Policy[selfAttn_merge_srnn]",
            "adapter_boundary": "Run the GenSafeNav model-only GST_predictor_rand checkpoint through "
            "the SoNIC-compatible adapter contract, then apply an explicit "
            "short-horizon safety guard with goal-policy fallback before emitting "
            "Robot SF unicycle_vw commands.",
        },
        kinematics_profile={
            "planner_command_space": "mixed_vw_or_unicycle",
            "supports_native_commands": True,
            "supports_adapter_commands": True,
            "default_execution_mode": "mixed",
            "default_adapter_name": "sonic_guarded_goal_fallback",
            "upstream_command_space": "holonomic_velocity_xy",
            "benchmark_command_space": "unicycle_vw",
            "projection_policy": "heading_safe_velocity_to_unicycle_vw",
            "projection_documented": True,
        },
        policy_builder_owner="legacy_map_runner",
    ),
)

CONTRACT_RECORDS_BY_NAME: dict[str, AlgorithmContractRecord] = {
    record.canonical_name: record for record in MIGRATED_ALGORITHM_RECORDS
}

ALGORITHM_ALIAS_INDEX: dict[str, AlgorithmContractRecord] = build_alias_index(
    MIGRATED_ALGORITHM_RECORDS
)


def get_contract_record(name: str) -> AlgorithmContractRecord | None:
    """Return the migrated contract record for an algorithm name or alias."""
    return ALGORITHM_ALIAS_INDEX.get(str(name).strip().lower())


def readiness_view(record: AlgorithmContractRecord) -> dict[str, Any]:
    """Project a record onto the readiness-facade field names.

    Returns:
        dict[str, Any]: Plain-data view matching the readiness facade fields.
    """
    return {
        "canonical_name": record.canonical_name,
        "tier": record.tier,
        "aliases": record.aliases,
        "note": record.note,
        "requires_explicit_opt_in": record.requires_explicit_opt_in,
    }


def validate_builder_agreement(*, check_legacy_dispatch: bool = False) -> dict[str, Any]:
    """Fail closed when a record's declared builder owner disagrees with reality.

    Args:
        check_legacy_dispatch: Also verify legacy-owned algorithms appear in the
            map-runner dispatch sets. Off by default because importing the map
            runner pulls the full benchmark runtime into import-light callers.

    Returns:
        A compact agreement report with per-record verdicts.

    Raises:
        RuntimeError: When a record declares a builder owner that does not
            actually build that algorithm.
    """
    report: dict[str, Any] = {"checked": [], "skipped_legacy_checks": []}
    socnav_keys: frozenset[str] | None = None
    for record in MIGRATED_ALGORITHM_RECORDS:
        if record.policy_builder_owner == "map_runner_policies.socnav_family":
            if socnav_keys is None:
                from robot_sf.benchmark.map_runner_policies import socnav_family  # noqa: PLC0415

                socnav_keys = frozenset(socnav_family._SOCNAV_FAMILY_LOOKUP)
            if record.canonical_name not in socnav_keys:
                raise RuntimeError(
                    f"{record.canonical_name} declares builder owner "
                    f"'{record.policy_builder_owner}' but no such registered adapter exists"
                )
            report["checked"].append(record.canonical_name)
        elif record.policy_builder_owner == "legacy_map_runner":
            if not check_legacy_dispatch:
                report["skipped_legacy_checks"].append(record.canonical_name)
                continue
            from robot_sf.benchmark.map_runner.map_runner import _SOCNAV_ALGO_KEYS  # noqa: PLC0415

            if record.canonical_name not in _SOCNAV_ALGO_KEYS:
                raise RuntimeError(
                    f"{record.canonical_name} declares builder owner "
                    f"'{record.policy_builder_owner}' but is absent from the legacy dispatch set"
                )
            report["checked"].append(record.canonical_name)
        else:
            raise RuntimeError(
                f"{record.canonical_name}: unknown policy_builder_owner "
                f"{record.policy_builder_owner!r}"
            )
    return report


def audit_contract_ownership() -> dict[str, Any]:
    """Report which algorithms are registry-owned versus still on the legacy path.

    The audit proves that no migrated canonical name is split across conflicting
    owners by comparing every facade value against the authoritative record.

    Returns:
        A report with migrated and legacy canonical-name lists plus any
        ownership conflicts detected against the live facades.
    """
    from robot_sf.benchmark import algorithm_metadata as _metadata  # noqa: PLC0415
    from robot_sf.benchmark import algorithm_readiness as _readiness  # noqa: PLC0415

    conflicts: list[dict[str, str]] = []
    metadata_attrs = (
        "_BASELINE_CATEGORY_BY_CANONICAL",
        "_POLICY_SEMANTICS_BY_CANONICAL",
        "_OBSERVATION_SPEC_BY_CANONICAL",
        "_UPSTREAM_REFERENCE_BY_CANONICAL",
        "_KINEMATICS_PROFILE_BY_CANONICAL",
    )
    record_fields = {
        "_BASELINE_CATEGORY_BY_CANONICAL": "baseline_category",
        "_POLICY_SEMANTICS_BY_CANONICAL": "policy_semantics",
        "_OBSERVATION_SPEC_BY_CANONICAL": "observation_spec",
        "_UPSTREAM_REFERENCE_BY_CANONICAL": "upstream_reference",
        "_KINEMATICS_PROFILE_BY_CANONICAL": "kinematics_profile",
    }
    for attr in metadata_attrs:
        facade = getattr(_metadata, attr)
        field_name = record_fields[attr]
        for canonical, record in CONTRACT_RECORDS_BY_NAME.items():
            expected = getattr(record, field_name)
            actual = facade.get(canonical)
            if actual != expected:
                conflicts.append({"surface": attr, "canonical": canonical})
    readiness_by_name = {spec.canonical_name: spec for spec in _readiness._ALGORITHMS}
    for canonical, record in CONTRACT_RECORDS_BY_NAME.items():
        spec = readiness_by_name.get(canonical)
        expected = readiness_view(record)
        actual = (
            {
                "canonical_name": spec.canonical_name,
                "tier": spec.tier,
                "aliases": spec.aliases,
                "note": spec.note,
                "requires_explicit_opt_in": spec.requires_explicit_opt_in,
            }
            if spec is not None
            else None
        )
        if actual != expected:
            conflicts.append({"surface": "algorithm_readiness", "canonical": canonical})
    all_canonical = [spec.canonical_name for spec in _readiness._ALGORITHMS]
    return {
        "schema": "algorithm_contract_ownership_audit.v1",
        "migrated": sorted(CONTRACT_RECORDS_BY_NAME),
        "legacy_remaining": [
            name for name in all_canonical if name not in CONTRACT_RECORDS_BY_NAME
        ],
        "conflicts": conflicts,
        "split_ownership_detected": bool(conflicts),
    }


__all__ = [
    "ALGORITHM_ALIAS_INDEX",
    "CONTRACT_RECORDS_BY_NAME",
    "MIGRATED_ALGORITHM_RECORDS",
    "AlgorithmContractRecord",
    "AlgorithmTier",
    "PolicyBuilderOwner",
    "audit_contract_ownership",
    "build_alias_index",
    "get_contract_record",
    "readiness_view",
    "validate_builder_agreement",
]
