"""Read-only readiness/preflight surface for the issue #4142 dense DPCBF comparison.

Issue #4142 asks for a bounded dense dynamic-obstacle comparison of three predeclared
Control Barrier Function (CBF) arms -- unfiltered (``cbf_off``), collision-cone CBF
(``cbf_collision_cone_on``), and the Dynamic Parabolic CBF variant
(``cbf_dynamic_parabolic_v1_on``). The 2026-07-02 campaign gate note requires that the
canonical comparison inputs be predeclared and reviewable *before any campaign can be
authorized*, and that fallback/degraded rows are declared caveats rather than success
evidence.

The comparison inputs live in the predeclared packet
``configs/research/issue_4142_dpcbf_dense_comparison_v1.yaml``. Before this module the
packet was standalone YAML that no code validated: nothing guaranteed the three arms stay
predeclared and fail-closed, that the referenced adapter/scenario configs exist, or that
the fallback exclusion stays in force. This module is that missing validation surface.

It is deliberately *read-only*: it loads the packet and the configs it references,
re-uses the canonical CBF runtime validator
(:func:`robot_sf.benchmark.cbf_safety_filter_runtime.runtime_config_from_mapping`) as the
single source of truth for arm semantics -- per the AGENTS.md canonical-owner rule -- and
derives a fail-closed status. It runs no episodes, launches no campaign, submits no
Slurm/GPU job, and makes no safety-performance or collision-reduction claim.

Status semantics (fail-closed on any structural gap):

- ``prerequisites_incomplete`` -- a required input is missing or invalid: the packet or a
  referenced config is absent/unparseable, an arm fails the canonical runtime validator,
  the three required arms are not all predeclared and distinct, an adapter config does not
  match its arm, or the fallback exclusion is not in force. This is a real, actionable
  failure; the campaign must not be authorized.
- ``inputs_ready_campaign_gated`` -- every packet input is present, valid, mutually
  consistent, and fail-closed. The only remaining blockers are the declared downstream
  gates in :data:`CAMPAIGN_GATES` (no packet-consuming runner is wired to this schema yet,
  and running requires explicit human/Slurm authorization). This is the expected healthy
  state; it confirms the inputs are reviewable, *not* that the comparison may run.

There is intentionally no ``authorized`` / ``ready-to-run`` state: running the dense
comparison is out of scope for this surface and stays gated behind the declared blockers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from robot_sf.benchmark.cbf_safety_filter_runtime import (
    CBF_COLLISION_CONE_ARM,
    CBF_DYNAMIC_PARABOLIC_V1_ARM,
    CBF_OFF_ARM,
    runtime_config_from_mapping,
)
from robot_sf.errors import RobotSfError

#: Output-contract schema for this readiness surface.
SCHEMA_VERSION = "issue-4142-dpcbf-dense-readiness.v1"

#: Schema the predeclared comparison packet must advertise.
PACKET_SCHEMA_VERSION = "robot_sf.issue_4142_dpcbf_dense_comparison.v1"

#: Repository-root-relative path to the predeclared comparison packet.
PACKET_PATH = "configs/research/issue_4142_dpcbf_dense_comparison_v1.yaml"

#: The three arms that must all be predeclared before the comparison is well-formed. The
#: unfiltered baseline and the two CBF variants are compared against each other; missing
#: any one makes the comparison meaningless.
REQUIRED_ARMS: tuple[str, ...] = (
    CBF_OFF_ARM,
    CBF_COLLISION_CONE_ARM,
    CBF_DYNAMIC_PARABOLIC_V1_ARM,
)

#: Row statuses that must be excluded from success evidence (fail-closed caveats). The
#: packet's ``summary_contract.excluded_row_statuses`` must cover at least this set.
REQUIRED_EXCLUDED_ROW_STATUSES: tuple[str, ...] = (
    "fallback",
    "degraded",
    "failed",
    "ineligible",
)

#: Evidence tier the packet must declare; anything stronger would overclaim.
EXPECTED_EVIDENCE_TIER = "bounded_runtime_comparison"

#: Alias map collapsing the runtime and public CBF variant vocabularies into a single
#: family key, so an adapter config and its runtime arm can be compared unambiguously.
_VARIANT_FAMILY = {
    "collision_cone": "collision_cone",
    "collision_cone_cbf_v1": "collision_cone",
    "dynamic_parabolic": "dynamic_parabolic_cbf_v1",
    "dynamic_parabolic_cbf_v1": "dynamic_parabolic_cbf_v1",
}

#: Declared downstream gates that keep the comparison from being authorized here even when
#: every packet input is valid. Surfaced verbatim so an operator (or routing pass) never
#: mistakes ``inputs_ready_campaign_gated`` for a go-ahead to run.
CAMPAIGN_GATES: tuple[str, ...] = (
    "no packet-consuming runner is wired to schema "
    f"'{PACKET_SCHEMA_VERSION}'; the canonical command cannot yet execute the "
    "three-arm comparison",
    "running the dense comparison requires explicit human/Slurm authorization and is "
    "out of scope for this read-only readiness surface",
)


class DpcbfDenseReadinessError(RobotSfError, ValueError):
    """Raised when the packet cannot be loaded or parsed at all."""


@dataclass(frozen=True, slots=True)
class ArmReadiness:
    """Per-arm readiness record for one predeclared comparison arm."""

    arm_key: str
    enabled: bool
    variant: str
    runtime_valid: bool
    algorithm_config_path: str | None
    algorithm_config_exists: bool
    algorithm_config_consistent: bool
    errors: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class DenseComparisonReadiness:
    """Aggregate fail-closed readiness for the issue #4142 dense DPCBF comparison."""

    schema_version: str
    packet_path: str
    packet_schema_ok: bool
    scenario_manifest_path: str | None
    scenario_manifest_exists: bool
    canonical_command: str | None
    evidence_tier: str | None
    fallback_excluded: bool
    required_arms_present: bool
    arms_distinct: bool
    arms: tuple[ArmReadiness, ...]
    status: str
    blockers: tuple[str, ...]
    campaign_gates: tuple[str, ...] = CAMPAIGN_GATES

    @property
    def inputs_ready(self) -> bool:
        """True when every packet input is valid and only downstream gates remain."""
        return self.status == "inputs_ready_campaign_gated"


def load_packet(path: Path) -> dict[str, Any]:
    """Load the comparison packet YAML into a mapping.

    Raises:
        DpcbfDenseReadinessError: if the file is missing, unparseable, or not a mapping.

    Returns:
        The parsed packet mapping.
    """
    if not path.is_file():
        raise DpcbfDenseReadinessError(f"comparison packet not found: {path}")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:  # pragma: no cover - exercised via malformed fixtures
        raise DpcbfDenseReadinessError(f"comparison packet is not valid YAML: {path}") from exc
    if not isinstance(payload, dict):
        raise DpcbfDenseReadinessError(f"comparison packet must be a mapping: {path}")
    return payload


def _config_variant_family(block: Any) -> tuple[bool, str | None]:
    """Read an adapter config ``cbf_safety_filter`` block.

    Returns:
        ``(enabled, variant_family)`` where ``variant_family`` collapses variant aliases.
    """
    if not isinstance(block, dict):
        return False, None
    enabled = bool(block.get("enabled", False))
    raw_variant = block.get("variant", "collision_cone")
    family = _VARIANT_FAMILY.get(str(raw_variant))
    return enabled, family


def _assess_arm(
    arm_payload: Any,
    *,
    algorithm_configs: dict[str, Any],
    repo_root: Path,
) -> ArmReadiness:
    """Validate one ``runtime_cbf_arms`` entry and its adapter config, fail-closed.

    Returns:
        A per-arm readiness record with any structural errors attached.
    """
    errors: list[str] = []
    if not isinstance(arm_payload, dict):
        return ArmReadiness(
            arm_key="<invalid>",
            enabled=False,
            variant="",
            runtime_valid=False,
            algorithm_config_path=None,
            algorithm_config_exists=False,
            algorithm_config_consistent=False,
            errors=("runtime_cbf_arms entry must be a mapping",),
        )

    arm_key = str(arm_payload.get("arm_key", "<missing>"))

    # Reuse the canonical runtime validator as the single source of truth for arm
    # semantics (enabled/arm_key/variant/threshold rules, DPCBF distinct from collision
    # cone). Any drift here fails closed with the canonical error message.
    runtime_valid = True
    resolved_variant = str(arm_payload.get("variant", ""))
    try:
        resolved = runtime_config_from_mapping(arm_payload)
        resolved_variant = str(resolved.variant)
    except (ValueError, TypeError) as exc:
        runtime_valid = False
        errors.append(f"runtime arm rejected by canonical validator: {exc}")

    # Cross-check the adapter config declared for this arm against the runtime arm.
    config_rel = algorithm_configs.get(arm_key)
    config_path = str(config_rel) if config_rel is not None else None
    config_exists = False
    config_consistent = False
    if config_rel is None:
        errors.append(f"no algorithm_config declared for arm '{arm_key}'")
    else:
        config_abs = repo_root / str(config_rel)
        config_exists = config_abs.is_file()
        if not config_exists:
            errors.append(f"algorithm_config for arm '{arm_key}' not found: {config_rel}")
        else:
            config_consistent = _check_config_consistency(
                config_abs,
                arm_key=arm_key,
                enabled=bool(arm_payload.get("enabled", False)),
                runtime_variant=resolved_variant,
                errors=errors,
            )

    return ArmReadiness(
        arm_key=arm_key,
        enabled=bool(arm_payload.get("enabled", False)),
        variant=resolved_variant,
        runtime_valid=runtime_valid,
        algorithm_config_path=config_path,
        algorithm_config_exists=config_exists,
        algorithm_config_consistent=config_consistent,
        errors=tuple(errors),
    )


def _check_config_consistency(
    config_abs: Path,
    *,
    arm_key: str,
    enabled: bool,
    runtime_variant: str,
    errors: list[str],
) -> bool:
    """Confirm the adapter config's CBF block matches the runtime arm. Appends errors.

    Returns:
        True when the adapter config is consistent with the runtime arm, else False.
    """
    try:
        config_payload = yaml.safe_load(config_abs.read_text(encoding="utf-8"))
    except yaml.YAMLError:
        errors.append(f"algorithm_config for arm '{arm_key}' is not valid YAML")
        return False
    if not isinstance(config_payload, dict):
        errors.append(f"algorithm_config for arm '{arm_key}' must be a mapping")
        return False

    cfg_enabled, cfg_family = _config_variant_family(config_payload.get("cbf_safety_filter"))
    if not enabled:
        # cbf_off arm: the adapter must not enable a CBF filter.
        if cfg_enabled:
            errors.append(
                f"arm '{arm_key}' is disabled but its adapter config enables a CBF filter"
            )
            return False
        return True

    if not cfg_enabled:
        errors.append(f"arm '{arm_key}' is enabled but its adapter config disables the CBF filter")
        return False
    runtime_family = _VARIANT_FAMILY.get(runtime_variant)
    if cfg_family != runtime_family:
        errors.append(
            f"arm '{arm_key}' variant mismatch: runtime='{runtime_variant}' "
            f"vs adapter config family='{cfg_family}'"
        )
        return False
    return True


def evaluate_readiness(  # noqa: C901 - one linear fail-closed check per packet field
    repo_root: str | Path = ".",
    packet_path: str | Path = PACKET_PATH,
) -> DenseComparisonReadiness:
    """Evaluate fail-closed readiness of the issue #4142 dense DPCBF comparison packet.

    Args:
        repo_root: Directory that repo-relative packet/config paths resolve against.
        packet_path: Repo-relative (or absolute) path to the comparison packet.

    Returns:
        Aggregate readiness with a fail-closed status and surfaced blockers.
    """
    root = Path(repo_root)
    packet_abs = Path(packet_path)
    if not packet_abs.is_absolute():
        packet_abs = root / packet_abs
    packet = load_packet(packet_abs)

    blockers: list[str] = []

    packet_schema = str(packet.get("schema_version", ""))
    packet_schema_ok = packet_schema == PACKET_SCHEMA_VERSION
    if not packet_schema_ok:
        blockers.append(
            f"packet schema_version '{packet_schema}' != expected '{PACKET_SCHEMA_VERSION}'"
        )

    scenario_rel = packet.get("scenario_manifest")
    scenario_path = str(scenario_rel) if scenario_rel is not None else None
    scenario_exists = bool(scenario_rel) and (root / str(scenario_rel)).is_file()
    if not scenario_rel:
        blockers.append("packet does not declare a scenario_manifest")
    elif not scenario_exists:
        blockers.append(f"scenario_manifest not found: {scenario_rel}")

    algorithm_configs = packet.get("algorithm_configs")
    if not isinstance(algorithm_configs, dict):
        algorithm_configs = {}
        blockers.append("packet does not declare an algorithm_configs mapping")

    raw_arms = packet.get("runtime_cbf_arms")
    arm_payloads = raw_arms if isinstance(raw_arms, list) else []
    if not isinstance(raw_arms, list):
        blockers.append("packet does not declare a runtime_cbf_arms list")

    arms = tuple(
        _assess_arm(arm, algorithm_configs=algorithm_configs, repo_root=root)
        for arm in arm_payloads
    )
    for arm in arms:
        blockers.extend(arm.errors)

    declared_arm_keys = [arm.arm_key for arm in arms]
    arms_distinct = len(declared_arm_keys) == len(set(declared_arm_keys))
    if not arms_distinct:
        blockers.append(f"runtime_cbf_arms contains duplicate arm_key entries: {declared_arm_keys}")
    required_arms_present = set(REQUIRED_ARMS).issubset(set(declared_arm_keys))
    if not required_arms_present:
        missing = sorted(set(REQUIRED_ARMS) - set(declared_arm_keys))
        blockers.append(f"packet is missing required arms: {missing}")

    contract = packet.get("summary_contract")
    contract = contract if isinstance(contract, dict) else {}
    evidence_tier = contract.get("evidence_tier")
    if evidence_tier != EXPECTED_EVIDENCE_TIER:
        blockers.append(
            f"summary_contract.evidence_tier '{evidence_tier}' != '{EXPECTED_EVIDENCE_TIER}'"
        )

    fallback_flag = contract.get("fallback_rows_are_success_evidence", None)
    excluded = contract.get("excluded_row_statuses")
    excluded_set = set(excluded) if isinstance(excluded, list) else set()
    missing_excluded = sorted(set(REQUIRED_EXCLUDED_ROW_STATUSES) - excluded_set)
    fallback_excluded = fallback_flag is False and not missing_excluded
    if fallback_flag is not False:
        blockers.append("summary_contract.fallback_rows_are_success_evidence must be false")
    if missing_excluded:
        blockers.append(
            f"summary_contract.excluded_row_statuses missing fail-closed statuses: {missing_excluded}"
        )

    contract_required = contract.get("required_arms")
    if isinstance(contract_required, list) and set(contract_required) != set(declared_arm_keys):
        blockers.append("summary_contract.required_arms does not match declared runtime_cbf_arms")

    canonical_command = packet.get("canonical_command")
    canonical_command = str(canonical_command) if canonical_command is not None else None

    status = "prerequisites_incomplete" if blockers else "inputs_ready_campaign_gated"

    return DenseComparisonReadiness(
        schema_version=SCHEMA_VERSION,
        packet_path=str(packet_path),
        packet_schema_ok=packet_schema_ok,
        scenario_manifest_path=scenario_path,
        scenario_manifest_exists=scenario_exists,
        canonical_command=canonical_command,
        evidence_tier=evidence_tier if isinstance(evidence_tier, str) else None,
        fallback_excluded=fallback_excluded,
        required_arms_present=required_arms_present,
        arms_distinct=arms_distinct,
        arms=arms,
        status=status,
        blockers=tuple(blockers),
    )


def to_dict(readiness: DenseComparisonReadiness) -> dict[str, Any]:
    """Return a JSON-serializable view of the readiness result."""
    return {
        "schema_version": readiness.schema_version,
        "packet_path": readiness.packet_path,
        "status": readiness.status,
        "inputs_ready": readiness.inputs_ready,
        "packet_schema_ok": readiness.packet_schema_ok,
        "scenario_manifest_path": readiness.scenario_manifest_path,
        "scenario_manifest_exists": readiness.scenario_manifest_exists,
        "canonical_command": readiness.canonical_command,
        "evidence_tier": readiness.evidence_tier,
        "fallback_excluded": readiness.fallback_excluded,
        "required_arms_present": readiness.required_arms_present,
        "arms_distinct": readiness.arms_distinct,
        "arms": [
            {
                "arm_key": arm.arm_key,
                "enabled": arm.enabled,
                "variant": arm.variant,
                "runtime_valid": arm.runtime_valid,
                "algorithm_config_path": arm.algorithm_config_path,
                "algorithm_config_exists": arm.algorithm_config_exists,
                "algorithm_config_consistent": arm.algorithm_config_consistent,
                "errors": list(arm.errors),
            }
            for arm in readiness.arms
        ],
        "blockers": list(readiness.blockers),
        "campaign_gates": list(readiness.campaign_gates),
    }


def render_markdown(readiness: DenseComparisonReadiness) -> str:
    """Render a compact Markdown report for the readiness result.

    Returns:
        A Markdown string leading with the claim boundary and status.
    """
    lines: list[str] = []
    lines.append("# Issue #4142 dense DPCBF comparison readiness")
    lines.append("")
    lines.append(
        "Claim boundary: read-only preflight over predeclared comparison inputs. This "
        "runs no episodes, authorizes no campaign, and makes no safety-performance or "
        "collision-reduction claim."
    )
    lines.append("")
    lines.append(f"- Status: `{readiness.status}`")
    lines.append(f"- Packet: `{readiness.packet_path}` (schema ok: {readiness.packet_schema_ok})")
    lines.append(
        f"- Scenario manifest present: {readiness.scenario_manifest_exists} "
        f"(`{readiness.scenario_manifest_path}`)"
    )
    lines.append(
        f"- Required arms present: {readiness.required_arms_present}; "
        f"distinct: {readiness.arms_distinct}"
    )
    lines.append(
        f"- Evidence tier: `{readiness.evidence_tier}`; fallback/degraded excluded: "
        f"{readiness.fallback_excluded}"
    )
    if readiness.canonical_command:
        lines.append(f"- Canonical command (not executed here): `{readiness.canonical_command}`")
    lines.append("")
    lines.append("## Arms")
    lines.append("")
    lines.append(
        "| arm_key | enabled | variant | runtime valid | config exists | config consistent |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for arm in readiness.arms:
        lines.append(
            f"| `{arm.arm_key}` | {arm.enabled} | `{arm.variant}` | {arm.runtime_valid} | "
            f"{arm.algorithm_config_exists} | {arm.algorithm_config_consistent} |"
        )
    lines.append("")
    if readiness.blockers:
        lines.append("## Blockers (fail-closed)")
        lines.append("")
        for blocker in readiness.blockers:
            lines.append(f"- {blocker}")
        lines.append("")
    lines.append("## Declared campaign gates (remain even when inputs are ready)")
    lines.append("")
    for gate in readiness.campaign_gates:
        lines.append(f"- {gate}")
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "CAMPAIGN_GATES",
    "EXPECTED_EVIDENCE_TIER",
    "PACKET_PATH",
    "PACKET_SCHEMA_VERSION",
    "REQUIRED_ARMS",
    "REQUIRED_EXCLUDED_ROW_STATUSES",
    "SCHEMA_VERSION",
    "ArmReadiness",
    "DenseComparisonReadiness",
    "DpcbfDenseReadinessError",
    "evaluate_readiness",
    "load_packet",
    "render_markdown",
    "to_dict",
]
