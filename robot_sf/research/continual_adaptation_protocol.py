"""Metadata-only continual-adaptation and revalidation protocol (issue #6582).

Continual-RL research reports rapid adaptation to changed surfaces, payloads,
layouts, actuator behavior, or pedestrian dynamics. Any policy update, however,
changes the *validated* system and can invalidate prior safety evidence. This
module defines the **local, buildable half** of a bounded adaptation protocol: a
metadata-only contract that declares, per adaptation run, what an adaptation job
would have to look like before it can be trusted to preserve safety evidence.

The checker enforces:

* **immutable baseline identity** -- a baseline checkpoint ``identifier`` plus a
  content ``checksum`` so the frozen baseline cannot silently change;
* **immutable safety wrapper** -- the safety-wrapper ``identifier`` plus a
  ``checksum``, and ``mutation_permitted`` must be ``false`` (a manifest that
  grants the adaptation job permission to mutate the wrapper fails closed);
* **declared mutable parameters** -- the parameter prefixes the job may update;
  everything else is frozen;
* **bounded experience budget** -- a finite ``steps`` count; an unbounded budget
  fails closed;
* **disjoint adaptation/evaluation scenarios** -- calibration/adaptation data is
  never counted as held-out evaluation evidence;
* **declared synthetic shift** -- at least one of friction/payload/latency/
  pedestrian/other is named for revalidation;
* **nominal/shift/forgetting thresholds** -- the pre-declared acceptance bounds;
* **promotion gating** -- ``promotion_decision='promote'`` is rejected unless all
  nominal/shift/forgetting result references and the new evidence bundle are
  present.

It also **derives** a new adapted-policy identifier deterministically from the
baseline identity plus a normalized adaptation manifest; the derived identifier
never equals or overwrites the baseline identifier.

It deliberately does **not** launch training, alter a checkpoint, mutate the
safety wrapper, run an evaluation, or promote a policy.
:data:`CONTINUAL_ADAPTATION_EVIDENCE_BOUNDARY` is stamped on every report so a
passing protocol check is never mistaken for an executed adaptation, a promoted
policy, or benchmark/paper evidence. This contract mirrors the
:mod:`robot_sf.research.scenario_prior_staging_contract` pattern.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator

from robot_sf.common.json_pointer import json_pointer
from robot_sf.errors import RobotSfError

CONTINUAL_ADAPTATION_RUN_SCHEMA_VERSION = "continual_adaptation_run.v1"
CONTINUAL_ADAPTATION_RUN_SCHEMA_FILE = (
    Path(__file__).with_name("schemas") / "continual_adaptation_run.v1.json"
)

#: Explicit boundary stamped on every report so a passing protocol check is
#: never mistaken for an executed adaptation, a checkpoint write, a safety-wrapper
#: mutation, a policy promotion, or benchmark/paper evidence.
CONTINUAL_ADAPTATION_EVIDENCE_BOUNDARY = (
    "protocol_contract_only_no_training_no_checkpoint_write_no_safety_wrapper_mutation"
    "_no_policy_promotion_no_benchmark_or_paper_evidence"
)

# Promotion decisions (must match the JSON schema enum).
PROMOTION_REJECT = "reject"
PROMOTION_EXPERIMENTAL = "experimental"
PROMOTION_PROMOTE = "promote"

# Resolved protocol states.
PROTOCOL_STATUS_VALID = "valid"
PROTOCOL_STATUS_INVALID = "invalid"

#: Result/evidence references that must all be present before a 'promote'
#: decision can pass the promotion gate.
_REQUIRED_PROMOTION_REFS = (
    "nominal_result",
    "shift_result",
    "forgetting_result",
    "evidence_bundle",
)


class ContinualAdaptationProtocolError(RobotSfError, ValueError):
    """Raised when a continual-adaptation manifest fails schema checks."""

    def __init__(self, errors: list[str], *, source: str | Path | None = None):
        """Build an actionable validation error from schema messages."""
        self.errors = tuple(errors)
        self.source = str(source) if source is not None else None
        prefix = f"{self.source}: " if self.source else ""
        super().__init__(prefix + "; ".join(errors))


@dataclass(frozen=True, slots=True)
class ContinualAdaptationRunReport:
    """Aggregate result of checking a continual-adaptation run manifest.

    A report with ``protocol_status == 'invalid'`` is fail-closed: it must never
    be treated as a runnable, promotable, or evidence-bearing adaptation. The
    derived adapted-policy identifier is informational only; computing it does
    not write a checkpoint or promote a policy.
    """

    schema_version: str
    run_id: str
    issue: int
    evidence_boundary: str
    baseline_policy_identifier: str
    derived_adapted_policy_identifier: str
    promotion_decision: str
    promotion_ready: bool
    safety_wrapper_mutation_permitted: bool
    experience_budget_bounded: bool
    adaptation_evaluation_disjoint: bool
    protocol_status: str
    blockers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dictionary representation."""
        return asdict(self)


@lru_cache(maxsize=1)
def load_continual_adaptation_run_schema() -> dict[str, Any]:
    """Load the continual-adaptation run JSON schema.

    Returns:
        Parsed JSON Schema dictionary.
    """
    return json.loads(CONTINUAL_ADAPTATION_RUN_SCHEMA_FILE.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _run_validator() -> Draft202012Validator:
    """Return a cached schema validator (schema compilation is reused)."""
    return Draft202012Validator(load_continual_adaptation_run_schema())


def _raise_on_schema_errors(payload: Mapping[str, Any], *, source: str | Path | None) -> None:
    """Raise :class:`ContinualAdaptationProtocolError` if the payload is invalid."""
    validator = _run_validator()
    errors = [
        f"{json_pointer(error.absolute_path)}: {error.message}"
        for error in sorted(validator.iter_errors(payload), key=lambda err: list(err.absolute_path))
    ]
    if errors:
        raise ContinualAdaptationProtocolError(errors, source=source)


def load_continual_adaptation_run(path: str | Path) -> dict[str, Any]:
    """Load and schema-validate a continual-adaptation manifest from JSON or YAML.

    Returns:
        The validated manifest mapping.

    Raises:
        ContinualAdaptationProtocolError: when the file is missing or invalid.
    """
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise ContinualAdaptationProtocolError(["manifest file not found"], source=manifest_path)
    text = manifest_path.read_text(encoding="utf-8")
    try:
        payload = yaml.safe_load(text)
    except yaml.YAMLError as exc:  # pragma: no cover - defensive
        raise ContinualAdaptationProtocolError(
            [f"invalid YAML/JSON: {exc}"], source=manifest_path
        ) from exc
    if not isinstance(payload, Mapping):
        raise ContinualAdaptationProtocolError(["expected a mapping payload"], source=manifest_path)
    _raise_on_schema_errors(payload, source=manifest_path)
    return dict(payload)


def check_continual_adaptation_run(
    manifest: Mapping[str, Any],
    *,
    source: str | Path | None = None,
) -> ContinualAdaptationRunReport:
    """Check a continual-adaptation run manifest against the bounded protocol.

    Schema violations raise :class:`ContinualAdaptationProtocolError`. Cross-field
    / safety fail-closed violations populate ``blockers`` and set
    ``protocol_status='invalid'``; such a report is never promotion-ready and
    must not be treated as evidence.

    The adapted-policy identifier is **derived** deterministically from the
    baseline identity plus a normalized adaptation manifest and never equals or
    overwrites the baseline identifier.

    Args:
        manifest: A ``continual_adaptation_run.v1`` mapping. Schema-validated here.
        source: Optional source path for error messages.

    Returns:
        A structured protocol report.

    Raises:
        ContinualAdaptationProtocolError: when the manifest violates the schema.
    """
    _raise_on_schema_errors(manifest, source=source)

    blockers: list[str] = []

    baseline_identifier = str(manifest["baseline_policy"]["identifier"])
    safety_wrapper = manifest["safety_wrapper"]
    mutation_permitted = bool(safety_wrapper["mutation_permitted"])
    if mutation_permitted:
        blockers.append(
            "safety_wrapper.mutation_permitted is true; the adaptation job must not be allowed to "
            "mutate the safety wrapper"
        )

    budget = manifest["adaptation"]["experience_budget"]
    budget_bounded = _check_experience_budget(budget, blockers)

    adaptation_ids = [str(sid) for sid in manifest["scenarios"]["adaptation"]]
    evaluation_ids = [str(sid) for sid in manifest["scenarios"]["evaluation"]]
    disjoint = _check_scenario_disjoint(adaptation_ids, evaluation_ids, blockers)

    decision = str(manifest["promotion_decision"]["decision"])
    promotion_ready = _check_promotion_gate(decision, manifest.get("results"), blockers)

    derived_identifier = _compute_derived_identifier(baseline_identifier, manifest)

    protocol_status = PROTOCOL_STATUS_VALID if not blockers else PROTOCOL_STATUS_INVALID
    # A derived identifier that somehow collided with the baseline would let the
    # adaptation overwrite the frozen baseline; fail closed defensively.
    if derived_identifier == baseline_identifier:
        blockers.append(
            "derived adapted-policy identifier collides with the baseline identifier; "
            "the adapted identifier must never overwrite the baseline"
        )
        protocol_status = PROTOCOL_STATUS_INVALID
        promotion_ready = False

    return ContinualAdaptationRunReport(
        schema_version=CONTINUAL_ADAPTATION_RUN_SCHEMA_VERSION,
        run_id=str(manifest["run_id"]),
        issue=int(manifest["issue"]),
        evidence_boundary=CONTINUAL_ADAPTATION_EVIDENCE_BOUNDARY,
        baseline_policy_identifier=baseline_identifier,
        derived_adapted_policy_identifier=derived_identifier,
        promotion_decision=decision,
        promotion_ready=promotion_ready,
        safety_wrapper_mutation_permitted=mutation_permitted,
        experience_budget_bounded=budget_bounded,
        adaptation_evaluation_disjoint=disjoint,
        protocol_status=protocol_status,
        blockers=sorted(blockers),
    )


def derive_adapted_policy_identifier(
    manifest: Mapping[str, Any],
    *,
    source: str | Path | None = None,
) -> str:
    """Derive the adapted-policy identifier from a validated manifest.

    The identifier is computed deterministically from the baseline identity plus
    a normalized adaptation manifest and never equals or overwrites the baseline
    identifier. This is a pure derivation: it does not write a checkpoint or
    promote a policy.

    Args:
        manifest: A ``continual_adaptation_run.v1`` mapping. Schema-validated here.
        source: Optional source path for error messages.

    Returns:
        The derived adapted-policy identifier.

    Raises:
        ContinualAdaptationProtocolError: when the manifest violates the schema.
    """
    _raise_on_schema_errors(manifest, source=source)
    baseline_identifier = str(manifest["baseline_policy"]["identifier"])
    return _compute_derived_identifier(baseline_identifier, manifest)


def _compute_derived_identifier(baseline_identifier: str, manifest: Mapping[str, Any]) -> str:
    """Deterministically derive a new adapted-policy identifier.

    The normalized adaptation manifest captures the adaptation recipe (baseline
    identity/checksum, safety-wrapper checksum, mutable parameters, bounded
    budget, scenario IDs, shifts, thresholds) but excludes the promotion decision
    and results, which are outputs rather than adaptation inputs. The digest is
    rendered canonically so the derivation is reproducible across runs and hosts.

    The result is guaranteed to differ from ``baseline_identifier``.

    Returns:
        A new adapted-policy identifier derived from the baseline identity plus the
        normalized adaptation manifest.
    """
    normalized = _normalize_for_derivation(baseline_identifier, manifest)
    digest = hashlib.sha256(
        json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    candidate = f"{baseline_identifier}#continual-adaptation@sha256:{digest[:16]}"
    # Defensive: guarantee the derived identifier never overwrites the baseline.
    # The 16-hex suffix already makes a collision effectively impossible; falling
    # back to the full 64-hex digest breaks any pathological tie.
    if candidate == baseline_identifier:
        candidate = f"{baseline_identifier}#continual-adaptation@sha256:{digest}"
    return candidate


def _normalize_for_derivation(
    baseline_identifier: str, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Build the deterministic adaptation recipe used to derive the identifier.

    Returns:
        A JSON-serializable, key-sorted recipe capturing the baseline identity and
        adaptation inputs (excluding promotion decision and results).
    """
    baseline = manifest["baseline_policy"]
    safety_wrapper = manifest["safety_wrapper"]
    budget = manifest["adaptation"]["experience_budget"]
    return {
        "schema_version": CONTINUAL_ADAPTATION_RUN_SCHEMA_VERSION,
        "baseline_identifier": baseline_identifier,
        "baseline_checksum": _checksum_dict(baseline["checksum"]),
        "safety_wrapper_identifier": str(safety_wrapper["identifier"]),
        "safety_wrapper_checksum": _checksum_dict(safety_wrapper["checksum"]),
        "allowed_parameters": sorted(str(p) for p in manifest["adaptation"]["allowed_parameters"]),
        "experience_budget": {
            "bounded": bool(budget["bounded"]),
            "steps": budget["steps"],
            "units": str(budget["units"]),
        },
        "adaptation_scenarios": sorted(str(sid) for sid in manifest["scenarios"]["adaptation"]),
        "evaluation_scenarios": sorted(str(sid) for sid in manifest["scenarios"]["evaluation"]),
        "shifts": [
            {"id": str(shift["id"]), "kind": str(shift["kind"])}
            for shift in sorted(manifest["shifts"], key=lambda s: (str(s["id"]), str(s["kind"])))
        ],
        "thresholds": {
            "nominal": _threshold_dict(manifest["thresholds"]["nominal"]),
            "shift": _threshold_dict(manifest["thresholds"]["shift"]),
            "forgetting": _threshold_dict(manifest["thresholds"]["forgetting"]),
        },
    }


def _checksum_dict(checksum: Mapping[str, Any]) -> dict[str, str]:
    """Return a deterministic {algorithm, digest} view of a checksum."""
    return {"algorithm": str(checksum["algorithm"]), "digest": str(checksum["digest"])}


def _threshold_dict(threshold: Mapping[str, Any]) -> dict[str, Any]:
    """Return a deterministic {metric, bound, direction} view of a threshold."""
    return {
        "metric": str(threshold["metric"]),
        "bound": threshold["bound"],
        "direction": str(threshold["direction"]),
    }


def _check_experience_budget(budget: Mapping[str, Any], blockers: list[str]) -> bool:
    """Validate that the experience budget is bounded with a finite positive step count.

    Returns:
        ``True`` when the budget is bounded with a finite positive step count.
    """
    bounded = bool(budget["bounded"])
    steps = budget["steps"]
    if not bounded:
        blockers.append(
            "adaptation.experience_budget.bounded is false; the experience budget must be finite"
        )
        return False
    if steps is None:
        blockers.append(
            "adaptation.experience_budget.steps is null for a bounded budget; declare a finite "
            "positive step count"
        )
        return False
    if not isinstance(steps, int) or isinstance(steps, bool) or steps <= 0:
        blockers.append(
            "adaptation.experience_budget.steps must be a finite positive integer for a bounded "
            "budget"
        )
        return False
    return True


def _check_scenario_disjoint(
    adaptation_ids: list[str], evaluation_ids: list[str], blockers: list[str]
) -> bool:
    """Fail closed when adaptation and evaluation scenario IDs overlap.

    Returns:
        ``True`` when the adaptation and evaluation scenario ID sets are disjoint.
    """
    overlap = sorted(set(adaptation_ids) & set(evaluation_ids))
    if overlap:
        blockers.append(
            "scenarios.adaptation and scenarios.evaluation must be disjoint; overlapping IDs: "
            + ", ".join(overlap)
        )
        return False
    return True


def _check_promotion_gate(
    decision: str, results: Mapping[str, Any] | None, blockers: list[str]
) -> bool:
    """Gate promotion on complete result/evidence references.

    Returns:
        ``True`` only when ``decision`` is 'promote' and all required result and
        evidence references are present.
    """
    if decision != PROMOTION_PROMOTE:
        return False
    if results is None:
        blockers.append(
            "promotion_decision is 'promote' but no results block is declared; promotion requires "
            "nominal_result, shift_result, forgetting_result, and evidence_bundle references"
        )
        return False
    promotion_ready = True
    for ref_name in _REQUIRED_PROMOTION_REFS:
        ref = results.get(ref_name)
        if not _is_complete_reference(ref, ref_name):
            blockers.append(
                f"promotion_decision is 'promote' but results.{ref_name} is missing or incomplete; "
                "promotion requires a uri plus checksum"
            )
            promotion_ready = False
    return promotion_ready


def _is_complete_reference(ref: Any, ref_name: str) -> bool:
    """Return ``True`` when a result/evidence reference has a uri and checksum."""
    if not isinstance(ref, Mapping):
        return False
    uri = ref.get("uri")
    checksum = ref.get("checksum")
    if not isinstance(uri, str) or not uri.strip():
        return False
    if not isinstance(checksum, Mapping):
        return False
    algorithm = checksum.get("algorithm")
    digest = checksum.get("digest")
    return (
        isinstance(algorithm, str)
        and bool(algorithm.strip())
        and isinstance(digest, str)
        and bool(digest.strip())
    )
