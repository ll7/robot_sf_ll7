"""User-facing ``robot-sf planners`` discovery interface.

Import-light CLI for discovering available planners, capabilities, kinematics
interfaces, observation requirements, and readiness status without constructing
planner instances or importing heavy dependencies (PyTorch, Stable-Baselines3,
CARLA, etc.).
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from robot_sf.baselines import BASELINES, SIMPLE_POLICY_ALIASES
from robot_sf.benchmark.algorithm_contract import CONTRACT_RECORDS_BY_NAME
from robot_sf.benchmark.algorithm_metadata import (
    _BASELINE_CATEGORY_BY_CANONICAL,
    _KINEMATICS_PROFILE_BY_CANONICAL,
    _OBSERVATION_SPEC_BY_CANONICAL,
)
from robot_sf.benchmark.algorithm_readiness import (
    _ALGORITHMS,
    AlgorithmReadiness,
    paper_baseline_algorithms,
)

if TYPE_CHECKING:
    import argparse

__all__ = [
    "PlannerDiscoveryEntry",
    "describe_planner_payload",
    "list_planners_payload",
]

SCHEMA_LIST_VERSION = "planner_list.v1"
SCHEMA_DESCRIBE_VERSION = "planner_describe.v1"

# Standalone execution arms or helper drivers that are not standalone planner algorithms.
_EXCLUSIONS = (
    {
        "key": "native_command",
        "reason": "execution_arm",
        "note": (
            "Subprocess execution arm in "
            "robot_sf.benchmark.map_runner.map_runner_native_command; "
            "not a standalone planner algorithm."
        ),
    },
)

# Standalone planner wrappers/adapters present in BASELINES or codebase but not in _ALGORITHMS
_EXTRA_PLANNERS = (
    AlgorithmReadiness(
        canonical_name="random",
        tier="baseline-ready",
        aliases=("random",),
        requires_explicit_opt_in=False,
        note="Uniform random action baseline reference policy.",
    ),
    AlgorithmReadiness(
        canonical_name="fast_pysf_planner",
        tier="experimental",
        aliases=("fast_pysf", "fast_pysf_planner"),
        requires_explicit_opt_in=False,
        note="Direct C++ fast-pysf SocialForce planner wrapper adapter.",
    ),
    AlgorithmReadiness(
        canonical_name="chance_constrained_mpc",
        tier="experimental",
        aliases=("chance_constrained_mpc", "cc_mpc"),
        requires_explicit_opt_in=True,
        note="Chance-constrained MPC local trajectory optimizer with obstacle uncertainty bounds.",
    ),
)


@dataclass(frozen=True)
class PlannerDiscoveryEntry:
    """Discovery catalog entry for a single planner algorithm."""

    canonical_name: str
    aliases: list[str]
    family: str
    tier: str
    status: str
    paper_baseline_eligible: bool
    requires_explicit_opt_in: bool
    execution_mode: str
    command_space: str
    compatible_robot_kinematics: list[str]
    observation_mode_default: str
    observation_modes_supported: list[str]
    observation_inputs: list[str]
    required_extras: list[str]
    required_artifacts: list[str]
    metadata_source: str
    summary: str

    def to_dict(self) -> dict[str, Any]:
        """Convert entry to a JSON-serializable dictionary.

        Returns:
            Dictionary representation of the discovery entry.
        """
        return asdict(self)


def _build_planner_entry(
    spec: AlgorithmReadiness,
    paper_baselines: frozenset[str],
    aliases: list[str],
) -> PlannerDiscoveryEntry:
    """Construct a PlannerDiscoveryEntry for one algorithm specification.

    Returns:
        Structured planner discovery entry.
    """
    canonical = spec.canonical_name
    family = _BASELINE_CATEGORY_BY_CANONICAL.get(canonical, "unknown")
    kinematics = _KINEMATICS_PROFILE_BY_CANONICAL.get(canonical, {})
    obs_spec = _OBSERVATION_SPEC_BY_CANONICAL.get(canonical, {})

    if spec.tier == "placeholder":
        status = "placeholder"
    elif spec.requires_explicit_opt_in:
        status = "experimental_opt_in"
    elif spec.tier == "baseline-ready":
        status = "baseline_ready"
    else:
        status = "experimental"

    paper_eligible = canonical in paper_baselines or getattr(
        CONTRACT_RECORDS_BY_NAME.get(canonical), "paper_baseline_eligible", False
    )

    execution_mode = str(
        kinematics.get("execution_mode") or kinematics.get("default_execution_mode") or "adapter"
    )
    command_space = str(kinematics.get("planner_command_space") or "unknown")
    compatible_kinematics = [str(k) for k in kinematics.get("compatible_robot_kinematics", ())]

    obs_default = str(obs_spec.get("default_mode") or "all")
    obs_supported = [str(m) for m in obs_spec.get("supported_modes", ["all"])]
    obs_inputs = [str(i) for i in obs_spec.get("inputs", ["not_declared"])]

    required_extras: list[str] = []
    if canonical in {
        "orca",
        "hrvo",
        "socnav_orca_nonholonomic",
        "socnav_orca_dd",
        "socnav_orca_relaxed",
        "socnav_hrvo",
    }:
        required_extras.append("rvo2")
    elif canonical == "fast_pysf_planner":
        required_extras.append("fast-pysf")
    elif family == "learning" and canonical != "random":
        required_extras.append("training")

    required_artifacts: list[str] = []
    if canonical in {
        "crowdnav_height",
        "sonic_crowdnav",
        "gensafenav_ours_gst",
        "gensafenav_ours_gst_guarded",
        "gensafenav_gst_predictor_rand",
        "gensafenav_gst_predictor_rand_guarded",
    }:
        required_artifacts.append("checkpoint")

    metadata_source = (
        "algorithm_contract"
        if canonical in CONTRACT_RECORDS_BY_NAME
        else "algorithm_readiness"
        if any(s.canonical_name == canonical for s in _ALGORITHMS)
        else "baseline_registry"
    )

    return PlannerDiscoveryEntry(
        canonical_name=canonical,
        aliases=aliases,
        family=family,
        tier=spec.tier,
        status=status,
        paper_baseline_eligible=paper_eligible,
        requires_explicit_opt_in=spec.requires_explicit_opt_in,
        execution_mode=execution_mode,
        command_space=command_space,
        compatible_robot_kinematics=compatible_kinematics,
        observation_mode_default=obs_default,
        observation_modes_supported=obs_supported,
        observation_inputs=obs_inputs,
        required_extras=sorted(required_extras),
        required_artifacts=sorted(required_artifacts),
        metadata_source=metadata_source,
        summary=spec.note,
    )


def _build_catalog() -> tuple[dict[str, PlannerDiscoveryEntry], dict[str, str]]:
    """Build the validated catalog of canonical planners and alias index.

    Fails closed on duplicate canonical keys or alias collisions.

    Returns:
        Tuple of (entries_dict, alias_index_dict).
    """
    paper_baselines = frozenset(paper_baseline_algorithms())
    entries: dict[str, PlannerDiscoveryEntry] = {}
    alias_index: dict[str, str] = {}

    all_specs: list[AlgorithmReadiness] = list(_ALGORITHMS) + list(_EXTRA_PLANNERS)

    for spec in all_specs:
        canonical = spec.canonical_name
        if canonical in entries:
            raise ValueError(f"Duplicate canonical planner key detected: {canonical}")

        aliases = sorted(set(spec.aliases) | ({canonical}))
        if canonical == "social_force" and "baseline_sf" not in aliases:
            aliases = sorted(set(aliases) | {"baseline_sf"})
        if canonical == "goal":
            aliases = sorted(set(aliases) | set(SIMPLE_POLICY_ALIASES))

        for alias in aliases:
            normalized_alias = alias.strip().lower()
            if normalized_alias in alias_index and alias_index[normalized_alias] != canonical:
                raise ValueError(
                    f"Alias collision detected: '{alias}' maps to both "
                    f"'{alias_index[normalized_alias]}' and '{canonical}'"
                )
            alias_index[normalized_alias] = canonical

        entries[canonical] = _build_planner_entry(spec, paper_baselines, aliases)

    for key in BASELINES:
        normalized_key = key.strip().lower()
        if normalized_key not in alias_index:
            raise ValueError(f"Registered baseline '{key}' missing from planner discovery catalog")

    return entries, alias_index


# Lazily initialized module-level catalog
_ENTRIES: dict[str, PlannerDiscoveryEntry] | None = None
_ALIAS_INDEX_MAP: dict[str, str] | None = None


def _get_catalog() -> tuple[dict[str, PlannerDiscoveryEntry], dict[str, str]]:
    """Return the cached catalog and alias index."""
    global _ENTRIES, _ALIAS_INDEX_MAP
    if _ENTRIES is None or _ALIAS_INDEX_MAP is None:
        _ENTRIES, _ALIAS_INDEX_MAP = _build_catalog()
    return _ENTRIES, _ALIAS_INDEX_MAP


def list_planners_payload() -> dict[str, Any]:
    """Return the structured payload listing all registered planners.

    Returns:
        Structured dictionary matching the planner_list schema.
    """
    entries, _ = _get_catalog()
    sorted_entries = [entries[k].to_dict() for k in sorted(entries.keys())]
    return {
        "schema_version": SCHEMA_LIST_VERSION,
        "count": len(sorted_entries),
        "planners": sorted_entries,
        "exclusions": list(_EXCLUSIONS),
    }


def describe_planner_payload(key: str) -> dict[str, Any]:
    """Return the structured payload describing a single planner by key or alias.

    Returns:
        Structured dictionary matching the planner_describe schema.

    Raises:
        KeyError: If ``key`` does not match any known canonical name or alias.
    """
    entries, alias_index = _get_catalog()
    normalized = str(key).strip().lower()
    canonical = alias_index.get(normalized)
    if canonical is None or canonical not in entries:
        available = sorted(entries.keys())
        raise KeyError(f"Unknown planner '{key}'. Available canonical planners: {available}")

    entry = entries[canonical]
    payload = entry.to_dict()
    payload["schema_version"] = SCHEMA_DESCRIBE_VERSION
    payload["requested_key"] = key
    payload["alias_used"] = key if normalized != canonical else None
    payload["is_alias"] = normalized != canonical
    return payload


def _format_list_planners(payload: dict[str, Any]) -> str:
    """Format planner list for human-friendly terminal display.

    Returns:
        Formatted string for terminal display.
    """
    lines: list[str] = [f"Registered Planners ({payload['count']}):\n"]
    for p in payload["planners"]:
        opt_str = " [requires opt-in]" if p["requires_explicit_opt_in"] else ""
        lines.append(f"- {p['canonical_name']}  [{p['tier']}] ({p['family']}){opt_str}")
        lines.append(f"    aliases: {', '.join(p['aliases'])}")
        lines.append(f"    command space: {p['command_space']}")
        lines.append(f"    execution mode: {p['execution_mode']}")
        if p["required_extras"]:
            lines.append(f"    required extras: {', '.join(p['required_extras'])}")
        if p["required_artifacts"]:
            lines.append(f"    required artifacts: {', '.join(p['required_artifacts'])}")
        lines.append(f"    summary: {p['summary']}")
        lines.append("")

    if payload.get("exclusions"):
        lines.append(f"Exclusions ({len(payload['exclusions'])}):")
        for exc in payload["exclusions"]:
            lines.append(f"- {exc['key']}: {exc['reason']} ({exc['note']})")
        lines.append("")

    return "\n".join(lines)


def _format_describe_planner(payload: dict[str, Any]) -> str:
    """Format single planner description for human-friendly terminal display.

    Returns:
        Formatted string for terminal display.
    """
    lines: list[str] = [f"Planner: {payload['canonical_name']}\n"]
    if payload.get("is_alias"):
        lines.append(
            f"  Requested Alias: {payload['requested_key']} -> {payload['canonical_name']}"
        )
    lines.append(f"  Aliases: {', '.join(payload['aliases'])}")
    lines.append(f"  Family: {payload['family']}")
    lines.append(f"  Tier: {payload['tier']}")
    lines.append(f"  Status: {payload['status']}")
    lines.append(f"  Paper Baseline Eligible: {payload['paper_baseline_eligible']}")
    lines.append(f"  Requires Explicit Opt-In: {payload['requires_explicit_opt_in']}")
    lines.append(f"  Execution Mode: {payload['execution_mode']}")
    lines.append(f"  Command Space: {payload['command_space']}")
    lines.append(
        f"  Compatible Kinematics: {', '.join(payload['compatible_robot_kinematics']) or 'unknown'}"
    )
    lines.append(f"  Observation Mode Default: {payload['observation_mode_default']}")
    lines.append(
        f"  Observation Modes Supported: {', '.join(payload['observation_modes_supported'])}"
    )
    lines.append(f"  Observation Inputs: {', '.join(payload['observation_inputs'])}")
    lines.append(f"  Required Extras: {', '.join(payload['required_extras']) or 'none'}")
    lines.append(f"  Required Artifacts: {', '.join(payload['required_artifacts']) or 'none'}")
    lines.append(f"  Metadata Source: {payload['metadata_source']}")
    lines.append(f"  Summary: {payload['summary']}")
    return "\n".join(lines)


def _handle_planners_list(args: argparse.Namespace) -> int:
    """Handle ``robot-sf planners list`` command.

    Returns:
        Process exit code (0 for success).
    """
    payload = list_planners_payload()
    if getattr(args, "format", "friendly") == "json":
        sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    else:
        sys.stdout.write(_format_list_planners(payload))
    return 0


def _handle_planners_describe(args: argparse.Namespace) -> int:
    """Handle ``robot-sf planners describe <key>`` command.

    Returns:
        Process exit code (0 for success, 2 for error).
    """
    key = getattr(args, "key", "")
    try:
        payload = describe_planner_payload(key)
    except KeyError as exc:
        if getattr(args, "format", "friendly") == "json":
            err_payload = {
                "schema_version": SCHEMA_DESCRIBE_VERSION,
                "status": "error",
                "requested_key": key,
                "error": str(exc),
            }
            sys.stdout.write(json.dumps(err_payload, indent=2, sort_keys=True) + "\n")
        else:
            sys.stderr.write(f"Error: {exc}\n")
        return 2

    if getattr(args, "format", "friendly") == "json":
        sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    else:
        sys.stdout.write(_format_describe_planner(payload) + "\n")
    return 0


def _add_planners_subparser(sub: argparse._SubParsersAction) -> None:
    """Register the ``robot-sf planners`` subcommand tree."""
    planners = sub.add_parser(
        "planners",
        help="List and describe available planners (uv run robot-sf planners ...)",
    )
    planners_sub = planners.add_subparsers(dest="planners_cmd", required=True)

    plist = planners_sub.add_parser("list", help="List registered planners.")
    plist.add_argument(
        "--format",
        choices=("friendly", "json"),
        default="friendly",
        help="Output format (default: friendly).",
    )

    pdescribe = planners_sub.add_parser("describe", help="Describe one planner by key or alias.")
    pdescribe.add_argument("key", help="Planner key or alias to describe.")
    pdescribe.add_argument(
        "--format",
        choices=("friendly", "json"),
        default="friendly",
        help="Output format (default: friendly).",
    )
