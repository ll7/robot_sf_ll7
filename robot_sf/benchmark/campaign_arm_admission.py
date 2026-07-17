"""Fail-closed packet-admission check that resolves every arm through the real loaders.

Plain-language summary: a campaign packet that only validates *shape* (keys present, types
right) cannot catch the S30 failure class, where the declared roster differs from the roster
that actually instantiates. This module resolves each declared arm through the same loaders the
benchmark runtime uses -- ``algorithm_readiness``, ``_ppo_paper_gate_status``, the model
registry's ``resolve_model_path`` / ``benchmark_promotion``, and the real algo-config YAML --
and fails closed when the declared readiness, checkpoint, fallback, or evidence claim cannot be
honoured as written.

This is a generalized admission check: any campaign packet whose ``planner_roster`` lists arms
with ``planner_id`` / ``role`` / ``readiness`` / ``config_path`` (the issue #5302 packet is one
instance) can be checked with :func:`check_campaign_arm_admission`. It is not specific to
issue #5302; the bug it fixes (a validator that checks shape and therefore cannot fail on the
thing it exists to prevent) is generic.

The five admission contracts, mirroring the runtime:

1. **Declared readiness matches the real gate.**
   - ``canonical_baseline`` -> the algorithm's ``algorithm_readiness`` tier is ``baseline-ready``.
   - ``artifact_qualified_only`` -> the resolved model registry entry is ``benchmark_promoted``
     (a ``benchmark_candidate`` or ``not_for_benchmark`` model is NOT artifact-qualified); for
     ``ppo`` the paper-grade gate (``_ppo_paper_gate_status``) must also pass.
   - ``checkpoint_qualified_only`` -> the checkpoint resolves and the resolved model is not
     ``not_for_benchmark``.
   - ``experimental_explicit_opt_in`` -> the algorithm tier is ``baseline-ready`` OR the arm's
     config sets ``allow_testing_algorithms: true`` (mirrors ``require_algorithm_allowed``).
2. **Required checkpoints resolve AND exist.** A ``model_id`` is accepted when its registry
   ``local_path`` is present OR the entry declares a durable remote source
   (``github_release`` / ``wandb_*``) that ``resolve_model_path`` could stage. A naive
   ``local_path`` probe reports false-absent because the registry falls back to the model cache,
   so this check deliberately mirrors :func:`robot_sf.benchmark.campaign_checkpoint_preflight`
   cheap-mode semantics instead of touching the filesystem directly.
3. **Candidate configs load through their real loader** (``yaml.safe_load`` into a mapping).
4. **Fallback flags are consistent with the packet's own execution_boundary.** When
   ``execution_boundary.fallback_or_degraded_success_allowed`` is false, no arm config may enable a
   policy-substitution flag (``fallback_to_goal`` / ``fallback_to_stop`` /
   ``allow_predictor_fallback``); otherwise the arm can silently substitute a different policy --
   a policy-identity violation of the packet's own contract.
5. **Role claims that assert evidence must cite a registered result.** A ``role`` containing
   ``leader`` (or similar superiority language) overstates any arm whose registered evidence is a
   statistical tie; such a role must cite a durable evidence path that establishes the separation,
   or the admission fails.

Claim boundary: this is a fail-closed admission/preference check. It does not run a benchmark,
submit compute, rank planners, or promote a claim. It resolves declared metadata through the real
loaders so a packet cannot be admitted whose arms cannot run as declared.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from robot_sf.benchmark.algorithm_readiness import get_algorithm_readiness
from robot_sf.models import get_registry_entry

# Declared readiness labels that require the resolved model to be artifact-qualified
# (registry ``benchmark_promotion.claim_boundary == benchmark_promoted``). A
# ``benchmark_candidate`` is intentionally NOT sufficient here: declaring an arm
# ``artifact_qualified_only`` while the resolved model is only a candidate is the exact
# declared-roster != instantiated-roster defect this check exists to catch.
_ARTIFACT_QUALIFIED_READINESS_LABELS: frozenset[str] = frozenset({"artifact_qualified_only"})

# Declared readiness labels that require the checkpoint to resolve but do NOT require the model
# to be promoted (a candidate checkpoint that loads is acceptable).
_CHECKPOINT_QUALIFIED_READINESS_LABELS: frozenset[str] = frozenset({"checkpoint_qualified_only"})

# Declared readiness labels that require explicit opt-in for experimental-tier algorithms.
_EXPLICIT_OPT_IN_READINESS_LABELS: frozenset[str] = frozenset(
    {"experimental_explicit_opt_in", "experimental_explicit_opt_in_required"}
)

# Declared readiness labels that require a baseline-ready algorithm tier.
_BASELINE_READINESS_LABELS: frozenset[str] = frozenset({"canonical_baseline"})

# Declared readiness labels that require a cited durable evidence/provenance artifact.
_PROVENANCE_REQUIRED_READINESS_LABELS: frozenset[str] = frozenset(
    {"candidate_requires_existing_provenance"}
)

# Algorithm keys whose readiness is gated by the PPO paper-grade provenance/quality gate when
# declared ``artifact_qualified_only``. Lower-cased for comparison.
_PPO_ARTIFACT_ALGO_KEYS: frozenset[str] = frozenset({"ppo"})

# Registry-entry fields that declare a durable remote source ``resolve_model_path`` can stage.
# Mirrors :data:`robot_sf.benchmark.campaign_checkpoint_preflight._REMOTE_SOURCE_KEYS` to avoid a
# cross-module dependency for a stable, well-known set.
_REMOTE_SOURCE_KEYS: tuple[str, ...] = (
    "github_release",
    "wandb_run_path",
    "wandb_artifact_path",
)

# Model-registry ``benchmark_promotion.claim_boundary`` value that marks an artifact as promoted
# to benchmark baseline (qualified for artifact-qualified readiness claims).
_BENCHMARK_PROMOTED_CLAIM_BOUNDARY = "benchmark_promoted"

# Model-registry ``benchmark_promotion.claim_boundary`` value that explicitly excludes an
# artifact from benchmark use; such a model can never satisfy a qualified readiness label.
_NOT_FOR_BENCHMARK_CLAIM_BOUNDARY = "not_for_benchmark"

# Config keys, at any nesting depth of an arm's algo_config, that enable a policy-substitution
# fallback. When ``execution_boundary.fallback_or_degraded_success_allowed`` is false these must
# all be disabled, otherwise the declared arm can silently run a different policy.
_FALLBACK_FLAG_KEYS: tuple[str, ...] = (
    "fallback_to_goal",
    "fallback_to_stop",
    "allow_predictor_fallback",
)

# Role-language tokens (substring, lower-cased) that assert superiority/leadership over the rest
# of the roster and therefore require a cited registered evidence result that establishes the
# separation. A pure "candidate" or "experimental" role makes no such claim and is not gated.
_LEADER_ROLE_TOKENS: tuple[str, ...] = ("leader", "best", "winner", "dominant", "superior")

# Roster arm fields that may cite a durable evidence/provenance artifact backing a readiness or
# role claim. The first non-empty hit is used.
_EVIDENCE_CITATION_FIELDS: tuple[str, ...] = ("evidence", "evidence_path", "provenance_citation")


class CampaignArmAdmissionError(ValueError):
    """Raised when one or more declared campaign arms cannot instantiate as declared."""


@dataclass(frozen=True)
class ArmAdmissionFinding:
    """One admission-contract failure for a single declared arm.

    Attributes:
        planner_id: The declared arm key.
        contract: Which admission contract failed (e.g. ``readiness``, ``checkpoint``,
            ``config_load``, ``fallback``, ``evidence``).
        declared: What the packet declared (readiness label / role / flag value).
        observed: What the real loaders resolved.
        message: Human-readable explanation of the mismatch and the remedy.
    """

    planner_id: str
    contract: str
    declared: str
    observed: str
    message: str


@dataclass(frozen=True)
class ArmAdmissionReport:
    """Resolution report for a single declared arm across all admission contracts.

    Attributes:
        planner_id: The declared arm key.
        algo: Canonical algorithm resolved for the arm (config ``algo`` field or planner_id).
        config_loaded: Whether the arm's algo_config loaded as a mapping through the real loader.
        findings: Admission-contract failures for this arm (empty when the arm is admissible).
    """

    planner_id: str
    algo: str
    config_loaded: bool
    findings: tuple[ArmAdmissionFinding, ...] = ()


@dataclass(frozen=True)
class CampaignArmAdmissionSummary:
    """Aggregate admission summary for a campaign packet's roster.

    Attributes:
        admissible: True only when every declared arm has no findings.
        arm_count: Number of declared arms checked.
        finding_count: Total number of admission findings across all arms.
        arms: Per-arm admission reports.
        findings: Flat list of all findings (one entry per failed contract per arm).
    """

    admissible: bool
    arm_count: int
    finding_count: int
    arms: tuple[ArmAdmissionReport, ...] = ()
    findings: tuple[ArmAdmissionFinding, ...] = ()

    def failure_messages(self) -> tuple[str, ...]:
        """Return one actionable message per finding, prefixed with the arm and contract."""
        return tuple(
            f"[{finding.planner_id}] {finding.contract}: {finding.message} "
            f"(declared: {finding.declared}; observed: {finding.observed})"
            for finding in self.findings
        )


def _require_mapping(value: Any, *, label: str) -> dict[str, Any]:
    """Return ``value`` as a mapping or raise a :class:`CampaignArmAdmissionError`.

    Args:
        value: The parsed YAML node to type-check.
        label: Human-readable path used in the error message.

    Returns:
        The mapping when ``value`` is a dict.

    Raises:
        CampaignArmAdmissionError: When ``value`` is not a mapping.
    """
    if not isinstance(value, dict):
        raise CampaignArmAdmissionError(f"{label} must be a mapping, got {type(value).__name__}")
    return value


def _normalize_lower(value: Any) -> str:
    """Return a stripped lower-cased string, treating ``None`` as empty."""
    if value is None:
        return ""
    return str(value).strip().lower()


def _resolve_algo(planner_id: str, algo_config: dict[str, Any] | None) -> str:
    """Resolve the canonical algorithm key for an arm.

    The config ``algo`` field wins when present (policy-search candidates declare their mechanism
    there); otherwise the ``planner_id`` itself is the algorithm key.

    Args:
        planner_id: The declared arm key.
        algo_config: The arm's loaded algo_config (may be ``None`` for baseline arms).

    Returns:
        The lower-cased algorithm key to resolve through ``algorithm_readiness``.
    """
    if algo_config is not None:
        config_algo = _normalize_lower(algo_config.get("algo"))
        if config_algo:
            return config_algo
    return _normalize_lower(planner_id)


def _load_arm_algo_config(
    config_path: Any,
    *,
    planner_id: str,
    repo_root: Path,
) -> tuple[dict[str, Any] | None, str | None, ArmAdmissionFinding | None]:
    """Load an arm's algo_config through the real YAML loader.

    Returns:
        A ``(config, resolved_path, finding)`` triple. ``config`` is the parsed mapping (or
        ``None`` when there is no config to load). ``resolved_path`` is the repo-relative config
        path string (or ``None``). ``finding`` is set only when the config exists but cannot be
        loaded as a mapping (the config_load contract failure).
    """
    if config_path is None:
        return None, None, None
    raw_path = str(config_path).strip()
    if not raw_path:
        return None, None, None
    path = repo_root / raw_path
    if not path.is_file():
        # A missing config file is a FileNotFoundError, not a shape error -- it is not an
        # apparently-valid packet. The caller's structural checker already owns that contract.
        raise FileNotFoundError(f"planner config missing: {raw_path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        message = (
            f"arm '{planner_id}' algo_config {raw_path} did not load as a mapping through the "
            "real loader; the arm cannot instantiate as declared"
        )
        return (
            None,
            raw_path,
            ArmAdmissionFinding(
                planner_id=planner_id,
                contract="config_load",
                declared=raw_path,
                observed=type(data).__name__,
                message=message,
            ),
        )
    return data, raw_path, None


def _entry_has_remote_source(entry: dict[str, Any]) -> bool:
    """Return True when a registry entry declares a durable remote source to stage from.

    Mirrors :func:`robot_sf.benchmark.campaign_checkpoint_preflight._entry_has_remote_source`:
    a model that is absent locally but declares ``github_release`` / ``wandb_*`` is stageable by
    ``resolve_model_path`` and therefore resolvable in cheap (network-free) admission mode.
    """
    return any(entry.get(key) for key in _REMOTE_SOURCE_KEYS)


def _resolve_model_id_cheap(model_id: str) -> tuple[bool, str]:
    """Check a ``model_id`` reference without touching the network.

    A checkpoint is resolvable when the registry entry's ``local_path`` exists OR the entry is
    not ``local_only`` and declares a durable remote source. A naive ``local_path`` filesystem
    probe reports false-absent because the registry falls back to the model cache, so this check
    resolves through the registry entry instead of ``Path(local_path).exists()`` alone.

    Args:
        model_id: The registry model id declared in the arm's algo_config.

    Returns:
        A ``(resolvable, detail)`` pair. ``detail`` describes the resolution status.
    """
    try:
        entry = get_registry_entry(model_id)
    except (KeyError, FileNotFoundError, TypeError, ValueError) as exc:
        return False, f"model_id '{model_id}' is not present in the model registry: {exc}"
    local_path = entry.get("local_path")
    if isinstance(local_path, str) and local_path.strip():
        resolved = Path(local_path)
        if not resolved.is_absolute():
            resolved = (Path.cwd() / resolved).resolve()
        if resolved.is_file():
            return True, f"registry local_path present at {resolved}"
    if bool(entry.get("local_only")):
        return (
            False,
            f"model_id '{model_id}' is local_only and its local_path is not present; it cannot "
            "be staged from a remote source",
        )
    if _entry_has_remote_source(entry):
        return (
            True,
            f"model_id '{model_id}' is not cached locally but declares a durable remote source "
            "that resolve_model_path can stage",
        )
    return (
        False,
        f"model_id '{model_id}' has neither a present local_path nor a durable remote source to "
        "stage from",
    )


def _iter_model_ids(algo_config: dict[str, Any]) -> list[str]:
    """Return every ``model_id``-style reference declared anywhere in an algo_config.

    Covers the canonical ``model_id`` plus the predictive / SA-CADRL model-id keys, mirroring
    :data:`robot_sf.benchmark.campaign_checkpoint_preflight._CHECKPOINT_REFERENCE_KEY_PAIRS` so a
    nested prior/predictive checkpoint is still admitted.
    """
    id_keys = ("model_id", "sacadrl_model_id", "predictive_model_id")
    found: list[str] = []

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            for key in id_keys:
                value = node.get(key)
                if isinstance(value, str) and value.strip():
                    found.append(value.strip())
            for value in node.values():
                _walk(value)
        elif isinstance(node, (list, tuple)):
            for item in node:
                _walk(item)

    _walk(algo_config)
    return found


def _resolve_model_claim_boundary(model_id: str) -> str:
    """Return the registry ``benchmark_promotion.claim_boundary`` for a model id.

    Returns an empty string when the model is absent or declares no promotion block, so callers
    can compare against the qualified label uniformly.
    """
    try:
        entry = get_registry_entry(model_id)
    except (KeyError, FileNotFoundError, TypeError, ValueError):
        return ""
    promotion = entry.get("benchmark_promotion")
    if not isinstance(promotion, dict):
        return ""
    return str(promotion.get("claim_boundary", "")).strip().lower()


def _finding(
    *,
    planner_id: str,
    contract: str,
    declared: Any,
    observed: Any,
    message: str,
) -> ArmAdmissionFinding:
    """Build a single :class:`ArmAdmissionFinding` (compact call-site helper).

    Returns:
        ArmAdmissionFinding: The constructed finding.
    """
    return ArmAdmissionFinding(
        planner_id=planner_id,
        contract=contract,
        declared=str(declared),
        observed=str(observed),
        message=message,
    )


def _check_artifact_qualified_readiness(
    *,
    planner_id: str,
    readiness: str,
    algo: str,
    config: dict[str, Any],
) -> list[ArmAdmissionFinding]:
    """Verify an ``artifact_qualified_only`` arm's model is promoted and the ppo gate passes.

    Returns:
        A list of findings (empty when the artifact-qualified contract holds).
    """
    findings: list[ArmAdmissionFinding] = []
    model_ids = _iter_model_ids(config)
    if not model_ids:
        findings.append(
            _finding(
                planner_id=planner_id,
                contract="readiness",
                declared=readiness,
                observed="no model_id in algo_config",
                message=(
                    f"arm '{planner_id}' declares readiness '{readiness}' but its algo_config "
                    "declares no model_id to qualify"
                ),
            )
        )
        return findings
    for model_id in model_ids:
        claim_boundary = _resolve_model_claim_boundary(model_id)
        if claim_boundary != _BENCHMARK_PROMOTED_CLAIM_BOUNDARY:
            findings.append(
                _finding(
                    planner_id=planner_id,
                    contract="readiness",
                    declared=readiness,
                    observed=f"model_id '{model_id}' claim_boundary={claim_boundary!r}",
                    message=(
                        f"arm '{planner_id}' declares readiness '{readiness}' "
                        "(artifact-qualified) but model "
                        f"'{model_id}' is not benchmark_promoted (claim_boundary="
                        f"{claim_boundary!r}); it would instantiate as a candidate/experimental "
                        "policy while declared artifact-qualified"
                    ),
                )
            )
    if algo in _PPO_ARTIFACT_ALGO_KEYS:
        # Lazy import: ``map_runner`` pulls in heavy training-time deps (TensorFlow/torch).
        # ``_ppo_paper_gate_status`` is the canonical, pure-Python paper-grade gate; importing
        # it lazily keeps the admission check fast for non-PPO arms.
        from robot_sf.benchmark.map_runner import _ppo_paper_gate_status  # noqa: PLC0415

        gate_ok, gate_reason = _ppo_paper_gate_status(config)
        if not gate_ok:
            findings.append(
                _finding(
                    planner_id=planner_id,
                    contract="readiness",
                    declared=readiness,
                    observed=f"_ppo_paper_gate_status=(False, {gate_reason!r})",
                    message=(
                        f"arm '{planner_id}' declares readiness '{readiness}' (artifact-"
                        "qualified) for PPO but the paper-grade gate fails: "
                        f"{gate_reason or 'profile is not paper/paper-baseline'}. Provide a "
                        "paper profile with provenance + quality_gate, or relax the readiness "
                        "label."
                    ),
                )
            )
    return findings


def _check_checkpoint_qualified_readiness(
    *,
    planner_id: str,
    readiness: str,
    config: dict[str, Any],
) -> list[ArmAdmissionFinding]:
    """Verify a ``checkpoint_qualified_only`` arm declares a non-excluded checkpoint model.

    Returns:
        A list of findings (empty when the checkpoint-qualified contract holds).
    """
    findings: list[ArmAdmissionFinding] = []
    model_ids = _iter_model_ids(config)
    if not model_ids:
        findings.append(
            _finding(
                planner_id=planner_id,
                contract="readiness",
                declared=readiness,
                observed="no model_id in algo_config",
                message=(
                    f"arm '{planner_id}' declares readiness '{readiness}' but its algo_config "
                    "declares no checkpoint model_id"
                ),
            )
        )
        return findings
    for model_id in model_ids:
        claim_boundary = _resolve_model_claim_boundary(model_id)
        if claim_boundary == _NOT_FOR_BENCHMARK_CLAIM_BOUNDARY:
            findings.append(
                _finding(
                    planner_id=planner_id,
                    contract="readiness",
                    declared=readiness,
                    observed=f"model_id '{model_id}' claim_boundary={claim_boundary!r}",
                    message=(
                        f"arm '{planner_id}' declares readiness '{readiness}' but model "
                        f"'{model_id}' is marked not_for_benchmark"
                    ),
                )
            )
    return findings


def _check_readiness_contract(
    *,
    planner_id: str,
    readiness: str,
    algo: str,
    algo_config: dict[str, Any] | None,
) -> list[ArmAdmissionFinding]:
    """Verify the declared readiness label matches what the real readiness gate computes.

    Args:
        planner_id: The declared arm key.
        readiness: The declared readiness label.
        algo: The resolved canonical algorithm key.
        algo_config: The arm's loaded algo_config (may be ``None``).

    Returns:
        A list of findings (empty when the readiness contract holds).
    """
    config = algo_config or {}

    if readiness in _BASELINE_READINESS_LABELS:
        return _check_baseline_readiness(planner_id=planner_id, readiness=readiness, algo=algo)
    if readiness in _ARTIFACT_QUALIFIED_READINESS_LABELS:
        return _check_artifact_qualified_readiness(
            planner_id=planner_id, readiness=readiness, algo=algo, config=config
        )
    if readiness in _CHECKPOINT_QUALIFIED_READINESS_LABELS:
        return _check_checkpoint_qualified_readiness(
            planner_id=planner_id, readiness=readiness, config=config
        )
    if readiness in _EXPLICIT_OPT_IN_READINESS_LABELS:
        return _check_explicit_opt_in_readiness(
            planner_id=planner_id, readiness=readiness, algo=algo, config=config
        )
    if readiness in _PROVENANCE_REQUIRED_READINESS_LABELS:
        # The readiness contract itself holds for a recognized provenance-required label; the
        # cited-evidence requirement is enforced by the evidence contract below. An empty/missing
        # citation is reported there, not here, so a candidate with cited evidence is admissible.
        return []

    # An unrecognized (or empty) readiness label is itself a finding: the packet is asserting a
    # readiness vocabulary the admission gate does not know how to honour, so it cannot be admitted.
    return [
        _finding(
            planner_id=planner_id,
            contract="readiness",
            declared=readiness or "<missing>",
            observed="unknown readiness label",
            message=(
                f"arm '{planner_id}' declares an unrecognized readiness label "
                f"'{readiness or '<missing>'}'; use one of canonical_baseline, "
                "artifact_qualified_only, checkpoint_qualified_only, "
                "candidate_requires_existing_provenance, or experimental_explicit_opt_in"
            ),
        )
    ]


def _check_baseline_readiness(
    *,
    planner_id: str,
    readiness: str,
    algo: str,
) -> list[ArmAdmissionFinding]:
    """Verify a ``canonical_baseline`` arm resolves to a baseline-ready algorithm tier.

    Returns:
        A list of findings (empty when the baseline-readiness contract holds).
    """
    spec = get_algorithm_readiness(algo)
    tier = spec.tier if spec is not None else None
    if tier == "baseline-ready":
        return []
    return [
        _finding(
            planner_id=planner_id,
            contract="readiness",
            declared=readiness,
            observed=f"tier={tier!r}",
            message=(
                f"arm '{planner_id}' declares readiness '{readiness}' (baseline-ready) "
                f"but algorithm '{algo}' resolves to tier {tier!r}"
            ),
        )
    ]


def _check_explicit_opt_in_readiness(
    *,
    planner_id: str,
    readiness: str,
    algo: str,
    config: dict[str, Any],
) -> list[ArmAdmissionFinding]:
    """Verify an ``experimental_explicit_opt_in`` arm sets the testing opt-in flag.

    Returns:
        A list of findings (empty when the explicit-opt-in contract holds).
    """
    spec = get_algorithm_readiness(algo)
    tier = spec.tier if spec is not None else None
    if tier == "baseline-ready" or _normalize_lower(config.get("allow_testing_algorithms")):
        return []
    return [
        _finding(
            planner_id=planner_id,
            contract="readiness",
            declared=readiness,
            observed=f"tier={tier!r}, allow_testing_algorithms not true",
            message=(
                f"arm '{planner_id}' declares readiness '{readiness}' for an experimental "
                f"algorithm '{algo}' but does not set allow_testing_algorithms: true; "
                "require_algorithm_allowed would reject it"
            ),
        )
    ]


def _check_checkpoint_contract(
    *,
    planner_id: str,
    algo_config: dict[str, Any] | None,
) -> list[ArmAdmissionFinding]:
    """Verify every declared checkpoint reference resolves through the real registry.

    Args:
        planner_id: The declared arm key.
        algo_config: The arm's loaded algo_config (may be ``None``).

    Returns:
        A list of findings (empty when every checkpoint resolves).
    """
    if algo_config is None:
        return []
    findings: list[ArmAdmissionFinding] = []
    for model_id in _iter_model_ids(algo_config):
        resolvable, detail = _resolve_model_id_cheap(model_id)
        if not resolvable:
            findings.append(
                ArmAdmissionFinding(
                    planner_id=planner_id,
                    contract="checkpoint",
                    declared=f"model_id={model_id}",
                    observed=detail,
                    message=(
                        f"arm '{planner_id}' declares checkpoint model_id '{model_id}' which does "
                        f"not resolve through resolve_model_path: {detail}"
                    ),
                )
            )
    return findings


def _walk_enabled_fallback_flags(node: Any) -> list[tuple[str, str]]:
    """Return ``(key, location)`` pairs for every enabled policy-substitution fallback flag.

    Walks nested mappings/sequences so a fallback enabled in a nested block (e.g. a scenario
    override) is still surfaced. A flag is "enabled" only when it is truthy and not literally
    ``False``.
    """
    enabled: list[tuple[str, str]] = []

    def _walk(value: Any, trail: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key in _FALLBACK_FLAG_KEYS and child is not False and bool(child):
                    enabled.append((str(key), f"{trail}.{key}" if trail else str(key)))
                elif isinstance(child, (dict, list)):
                    _walk(child, f"{trail}.{key}" if trail else str(key))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                if isinstance(child, (dict, list)):
                    _walk(child, f"{trail}[{index}]")

    _walk(node, "")
    return enabled


def _check_fallback_contract(
    *,
    planner_id: str,
    algo_config: dict[str, Any] | None,
    fallback_allowed: bool,
) -> list[ArmAdmissionFinding]:
    """Verify arm fallback flags are consistent with the packet's execution_boundary.

    When ``execution_boundary.fallback_or_degraded_success_allowed`` is false, no arm config may
    enable a policy-substitution flag, otherwise the declared arm can silently run a different
    policy -- a policy-identity violation of the packet's own contract.

    Args:
        planner_id: The declared arm key.
        algo_config: The arm's loaded algo_config (may be ``None``).
        fallback_allowed: The packet's ``fallback_or_degraded_success_allowed`` value.

    Returns:
        A list of findings (empty when the fallback contract holds).
    """
    if fallback_allowed or algo_config is None:
        return []
    findings: list[ArmAdmissionFinding] = []
    for key, location in _walk_enabled_fallback_flags(algo_config):
        findings.append(
            ArmAdmissionFinding(
                planner_id=planner_id,
                contract="fallback",
                declared=f"{location}=true",
                observed="execution_boundary.fallback_or_degraded_success_allowed=false",
                message=(
                    f"arm '{planner_id}' enables fallback flag '{location}' while the packet's "
                    "execution_boundary forbids fallback/degraded success; the arm can silently "
                    "substitute a different policy"
                ),
            )
        )
    return findings


def _resolve_evidence_citation(arm: dict[str, Any]) -> str:
    """Return the first evidence/provenance citation declared on an arm, if any."""
    for field_name in _EVIDENCE_CITATION_FIELDS:
        value = arm.get(field_name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _role_asserts_superiority(role: str) -> bool:
    """Return True when a role label asserts leadership/superiority over the roster."""
    role_lower = _normalize_lower(role)
    return any(token in role_lower for token in _LEADER_ROLE_TOKENS)


def _check_evidence_contract(
    *,
    planner_id: str,
    role: str,
    readiness: str,
    evidence_path: str,
    repo_root: Path,
) -> list[ArmAdmissionFinding]:
    """Verify role/readiness claims that assert evidence cite a registered result.

    A ``role`` asserting leadership/superiority overstates any arm whose registered evidence is a
    statistical tie; such a role must cite a durable evidence path that exists. A
    ``candidate_requires_existing_provenance`` readiness likewise requires a cited evidence path.

    Args:
        planner_id: The declared arm key.
        role: The declared role label.
        readiness: The declared readiness label.
        evidence_path: The cited evidence path (repo-relative), if any.
        repo_root: Repository root for resolving the cited path.

    Returns:
        A list of findings (empty when the evidence contract holds).
    """
    findings: list[ArmAdmissionFinding] = []
    requires_evidence = _role_asserts_superiority(role) or (
        readiness in _PROVENANCE_REQUIRED_READINESS_LABELS
    )
    if not requires_evidence:
        return findings
    if not evidence_path:
        claim_kind = "role" if _role_asserts_superiority(role) else "readiness"
        findings.append(
            ArmAdmissionFinding(
                planner_id=planner_id,
                contract="evidence",
                declared=f"{claim_kind}={role or readiness}",
                observed="no evidence citation",
                message=(
                    f"arm '{planner_id}' asserts {claim_kind} '{role or readiness}' that requires "
                    "registered evidence but cites none; provide an evidence/evidence_path field "
                    "pointing at a durable result that establishes the claim"
                ),
            )
        )
        return findings
    evidence_file = repo_root / evidence_path
    # Accept either a durable evidence file or a durable evidence directory (a packet commonly
    # cites a docs/context/evidence/<name>/ folder).
    if not evidence_file.exists():
        findings.append(
            ArmAdmissionFinding(
                planner_id=planner_id,
                contract="evidence",
                declared=f"{role or readiness} -> {evidence_path}",
                observed="cited evidence not found",
                message=(
                    f"arm '{planner_id}' cites evidence '{evidence_path}' that does not resolve "
                    "under the repo root; the superiority/leadership claim is unsupported by a "
                    "registered result"
                ),
            )
        )
    return findings


def _admit_arm(
    arm: dict[str, Any],
    *,
    repo_root: Path,
    fallback_allowed: bool,
) -> ArmAdmissionReport:
    """Resolve a single declared arm through every admission contract.

    Args:
        arm: One ``planner_roster.required`` row (``planner_id`` / ``role`` / ``readiness`` /
            ``config_path``).
        repo_root: Repository root for resolving config and evidence paths.
        fallback_allowed: The packet's ``fallback_or_degraded_success_allowed`` value.

    Returns:
        ArmAdmissionReport: The per-arm resolution report with any findings.
    """
    planner_id = str(arm.get("planner_id", "")).strip()
    readiness = _normalize_lower(arm.get("readiness"))
    role = str(arm.get("role", "")).strip()
    evidence_path = _resolve_evidence_citation(arm)

    config, _resolved_path, load_finding = _load_arm_algo_config(
        arm.get("config_path"), planner_id=planner_id, repo_root=repo_root
    )
    findings: list[ArmAdmissionFinding] = []
    if load_finding is not None:
        findings.append(load_finding)
        # A config that did not load cannot be further resolved; report the load failure and stop.
        return ArmAdmissionReport(
            planner_id=planner_id,
            algo=_resolve_algo(planner_id, None),
            config_loaded=False,
            findings=tuple(findings),
        )

    algo = _resolve_algo(planner_id, config)
    findings.extend(
        _check_readiness_contract(
            planner_id=planner_id, readiness=readiness, algo=algo, algo_config=config
        )
    )
    findings.extend(_check_checkpoint_contract(planner_id=planner_id, algo_config=config))
    findings.extend(
        _check_fallback_contract(
            planner_id=planner_id, algo_config=config, fallback_allowed=fallback_allowed
        )
    )
    findings.extend(
        _check_evidence_contract(
            planner_id=planner_id,
            role=role,
            readiness=readiness,
            evidence_path=evidence_path,
            repo_root=repo_root,
        )
    )
    return ArmAdmissionReport(
        planner_id=planner_id,
        algo=algo,
        config_loaded=True,
        findings=tuple(findings),
    )


def check_campaign_arm_admission(
    packet: dict[str, Any],
    *,
    repo_root: Path | None = None,
    roster_key: str = "planner_roster",
) -> CampaignArmAdmissionSummary:
    """Fail closed when any declared campaign arm cannot instantiate as declared.

    Resolves every arm in the packet's ``planner_roster.required`` list through the real loaders
    (``algorithm_readiness``, ``_ppo_paper_gate_status``, the model registry, the real
    algo-config YAML, and cited evidence paths) and reports admission-contract failures. This is
    the generalized admission check: the issue #5302 oracle-gap packet is one instance, but any
    campaign packet with a ``planner_roster`` of declared arms can be checked the same way.

    Args:
        packet: The parsed campaign packet mapping.
        repo_root: Repository root for resolving config/evidence paths (defaults to this file's
            repo root).
        roster_key: The packet key holding the roster (defaults to ``planner_roster``).

    Returns:
        CampaignArmAdmissionSummary: The aggregate admission summary. ``admissible`` is True only
        when every arm has no findings.

    Raises:
        CampaignArmAdmissionError: When the roster is malformed (not a mapping / missing required
            list). Per-arm admission failures are returned as findings, not raised, so callers can
            report every defect at once.
        FileNotFoundError: When a declared config_path does not exist (a missing file is a file
            error, not an apparently-valid packet).
    """
    root = repo_root or Path(__file__).resolve().parents[2]
    roster = _require_mapping(packet.get(roster_key), label=roster_key)
    required = roster.get("required")
    if not isinstance(required, list):
        raise CampaignArmAdmissionError(f"{roster_key}.required must be a list of declared arms")

    execution = packet.get("execution_boundary")
    fallback_allowed = bool(
        isinstance(execution, dict) and execution.get("fallback_or_degraded_success_allowed")
    )

    arms: list[ArmAdmissionReport] = []
    for entry in required:
        if not isinstance(entry, dict):
            raise CampaignArmAdmissionError(
                f"{roster_key}.required entries must be mappings, got {type(entry).__name__}"
            )
        arms.append(_admit_arm(entry, repo_root=root, fallback_allowed=fallback_allowed))

    all_findings: list[ArmAdmissionFinding] = []
    for arm in arms:
        all_findings.extend(arm.findings)
    return CampaignArmAdmissionSummary(
        admissible=not all_findings,
        arm_count=len(arms),
        finding_count=len(all_findings),
        arms=tuple(arms),
        findings=tuple(all_findings),
    )


__all__ = [
    "ArmAdmissionFinding",
    "ArmAdmissionReport",
    "CampaignArmAdmissionError",
    "CampaignArmAdmissionSummary",
    "check_campaign_arm_admission",
]
