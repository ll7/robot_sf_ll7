"""Fail-closed answerability checks for research campaign contracts.

The contract answers whether a planned campaign can resolve its declared
question. It does not run a campaign, admit evidence, or replace the existing
research-campaign manifest and figure-quality contracts.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

ANSWERABILITY_SCHEMA = "research_answerability.v1"
PROOF_BINDING_SCHEMA = "research_answerability_proof_binding.v1"
ANSWERABILITY_STATES = (
    "answerable",
    "diagnostic_only",
    "blocked_missing_producer",
    "blocked_underpowered",
    "blocked_analysis_contract",
    "blocked_noncomparable_rows",
    "blocked_artifact_plan",
    "blocked_missing_proof",
    "invalid_contract",
)
PROOF_SURFACES = (
    "producer",
    "preregistration",
    "evidence_contract",
    "analysis",
    "artifact",
    "result_packet",
)
# Decision-capable admission requires a claim-specific minimum.  A generic
# result-packet validator remains optional because the canonical owner is not
# present in every checkout; optionality is surfaced as a warning rather than
# silently promoted to proof.
DECISION_REQUIRED_PROOF_SURFACES = (
    "producer",
    "preregistration",
    "evidence_contract",
    "analysis",
    "artifact",
)
PROOF_STATUSES = ("passed", "unavailable", "failed", "not_run")
_DECISION_VOCABULARY = {"continue", "stop", "inconclusive", "invalid"}
_REQUIRED_SECTIONS = ("question", "estimand", "producers", "analysis", "design", "artifacts")
_REQUIRED_TEXT_FIELDS = {
    "question": (
        "research_question",
        "bounded_claim",
        "negative_result_meaning",
    ),
    "estimand": (
        "primary",
        "reference_or_null",
        "decision_predicates",
        "minimally_important_effect",
    ),
    "analysis": (
        "analysis_unit",
        "resampling_unit",
        "command",
        "multiplicity",
        "sensitivity_plan",
        "dry_run_status",
        "comparability_status",
    ),
    "design": ("mode", "power_status", "budget", "sample_size"),
    "artifacts": (
        "raw_owner",
        "durable_path",
        "durability_status",
        "incomplete_policy",
    ),
}
_PRODUCER_TEXT_FIELDS = (
    "field",
    "producer",
    "source",
    "unit",
    "direction",
    "denominator",
    "pairing_key",
    "missingness_rule",
    "status",
    "execution_mode",
)
_VALID_EXECUTION_MODES = {"native", "adapter", "fallback", "degraded", "unavailable"}
_VALID_PRODUCER_STATUSES = {"available", "unavailable", "missing", "blocked"}
_VALID_DRY_RUN_STATUSES = {"passed", "not_required", "failed", "blocked", "unknown"}
_VALID_COMPARABILITY_STATUSES = {
    "passed",
    "not_required",
    "failed",
    "blocked",
    "unknown",
    "mismatched",
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ProofStatus = Literal["passed", "unavailable", "failed", "not_run"]


class AnswerabilityContractError(ValueError):
    """Raised when a research-answerability contract is structurally invalid."""


@dataclass(frozen=True)
class ProofSurface:
    """Typed admission proof for one answerability surface."""

    status: ProofStatus
    required: bool
    unavailable_reason: str | None = None

    @classmethod
    def from_mapping(cls, value: Any, field: str) -> ProofSurface:
        """Build and validate a proof surface from its contract mapping.

        Returns:
            The validated proof surface.
        """
        surface = _mapping(value, field)
        status = surface.get("status")
        if status not in PROOF_STATUSES:
            raise AnswerabilityContractError(
                f"{field}.status must be one of {list(PROOF_STATUSES)}"
            )
        required = surface.get("required")
        if not isinstance(required, bool):
            raise AnswerabilityContractError(f"{field}.required must be a boolean")
        reason = surface.get("unavailable_reason")
        if status == "unavailable":
            if not isinstance(reason, str) or not reason.strip():
                raise AnswerabilityContractError(
                    f"{field}.unavailable_reason is required when status is unavailable"
                )
            reason = reason.strip()
        elif reason is not None and (not isinstance(reason, str) or not reason.strip()):
            raise AnswerabilityContractError(
                f"{field}.unavailable_reason must be a non-empty string when provided"
            )
        return cls(status=status, required=required, unavailable_reason=reason)


@dataclass(frozen=True)
class AnswerabilityResult:
    """Machine-readable answerability state and conservative reasons."""

    state: str
    reasons: tuple[str, ...]
    warnings: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe result payload."""
        return {
            "schema_version": ANSWERABILITY_SCHEMA,
            "state": self.state,
            "decision_capable": self.state == "answerable",
            "reasons": list(self.reasons),
            "warnings": list(self.warnings),
        }


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AnswerabilityContractError(f"{field} must be a mapping")
    return value


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AnswerabilityContractError(f"{field} must be a non-empty string")
    return value.strip()


def compute_proof_digest(binding: Mapping[str, Any], proof_results: Mapping[str, Any]) -> str:
    """Hash canonical binding inputs and the exact proof results.

    ``proof_results`` is stored in the binding so the strict evaluator can
    detect a later status mutation. The digest intentionally excludes the
    self-referential ``proof_digest`` and the embedded results from the
    binding portion, then includes those results once as the canonical
    ``surfaces`` value.

    Returns:
        The SHA-256 digest of the canonical binding and proof results.
    """
    digest_binding = {
        key: value for key, value in binding.items() if key not in {"proof_digest", "proof_results"}
    }
    payload = {"binding": digest_binding, "surfaces": proof_results}
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _repository_path_candidates(path: Path, repo_root: Path) -> tuple[tuple[Path, Path], ...]:
    """Return lexical and resolved repository-relative candidates for a path."""
    root = repo_root.resolve()
    candidate = path if path.is_absolute() else root / path
    candidates: list[tuple[Path, Path]] = []
    for item in (candidate, candidate.resolve()):
        try:
            relative = item.relative_to(root)
        except ValueError:
            continue
        pair = (item, relative)
        if pair not in candidates:
            candidates.append(pair)
    return tuple(candidates)


def _git_path_is_tracked(path: Path, repo_root: Path) -> bool:
    """Return whether a repository-relative path is present in the Git index."""
    root = repo_root.resolve()
    candidate = path if path.is_absolute() else root / path
    try:
        relative = candidate.relative_to(root)
    except ValueError:
        return False
    try:
        result = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", relative.as_posix()],
            cwd=root,
            capture_output=True,
            check=False,
            text=True,
        )
    except OSError:
        return False
    return result.returncode == 0 and relative.as_posix() in result.stdout.splitlines()


def strict_proof_input_provenance_error(
    path: Path,
    *,
    repo_root: Path,
    field: str,
) -> str | None:
    """Reject fixture inputs and untracked disposable output from strict proof.

    The strict admission path may hash a file successfully even when it is a
    test fixture or an untracked file materialized under ``output/``.  Neither
    condition is acceptable as provenance for decision-capable proof.

    Returns:
        An actionable error message, or ``None`` when provenance is allowed.
    """
    try:
        candidates = _repository_path_candidates(path, repo_root)
    except (OSError, RuntimeError):
        return f"{field} cannot be resolved safely for provenance"
    if any(relative.parts[:2] == ("tests", "fixtures") for _, relative in candidates):
        return f"{field} cannot use tests/fixtures provenance for strict proof"
    output_candidates = [
        candidate for candidate, relative in candidates if relative.parts[:1] == ("output",)
    ]
    try:
        is_file = path.is_file()
    except OSError:
        is_file = False
    if (
        output_candidates
        and is_file
        and not all(_git_path_is_tracked(candidate, repo_root) for candidate in output_candidates)
    ):
        return (
            f"{field} must use a tracked repository file; matching untracked files under "
            "output/ are not valid strict proof inputs"
        )
    return None


def _proof_binding_file_error(
    binding: Mapping[str, Any],
    *,
    field: str,
    digest_field: str,
    repo_root: Path,
) -> str | None:
    """Validate one repository-bound proof input and its digest.

    Returns:
        An error message, or ``None`` when the path and digest match.
    """
    value = binding.get(field)
    if not isinstance(value, str) or not value.strip():
        return f"answerability.proof_binding.{field} must be a non-empty path"
    path = Path(value.strip())
    if path.is_absolute() or ".." in path.parts:
        return f"answerability.proof_binding.{field} must be repository-relative"
    root = repo_root.resolve()
    candidate = root / path
    try:
        resolved = candidate.resolve()
    except (OSError, RuntimeError):
        return f"answerability.proof_binding.{field} cannot be resolved safely"
    if resolved == root or root not in resolved.parents:
        return f"answerability.proof_binding.{field} must resolve within the repository"
    if not resolved.is_file():
        return f"answerability.proof_binding.{field} must name an existing file"
    provenance_error = strict_proof_input_provenance_error(
        resolved,
        repo_root=root,
        field=f"answerability.proof_binding.{field}",
    )
    if provenance_error:
        return provenance_error
    try:
        first = resolved.read_bytes()
        second = resolved.read_bytes()
    except OSError as exc:
        return f"could not read answerability.proof_binding.{field}: {exc}"
    if first != second:
        return f"answerability.proof_binding.{field} changed while being verified"
    actual = hashlib.sha256(first).hexdigest()
    if actual != str(binding.get(digest_field, "")).lower():
        return f"answerability.proof_binding.{digest_field} does not match {field} bytes"
    return None


def _proof_binding_error(  # noqa: C901, PLR0912
    contract: Mapping[str, Any],
    *,
    campaign_id: str | None,
    repo_root: Path | None,
) -> str | None:
    """Return a fail-closed error for a missing or malformed admission binding."""
    binding = contract.get("proof_binding")
    if not isinstance(binding, Mapping):
        return "decision-capable admission requires answerability.proof_binding"
    if binding.get("schema_version") != PROOF_BINDING_SCHEMA:
        return f"answerability.proof_binding.schema_version must be {PROOF_BINDING_SCHEMA}"
    for field in (
        "campaign_id",
        "question",
        "estimand",
        "source_manifest",
        "campaign_config",
        "manifest_sha256",
        "config_sha256",
        "proof_digest",
    ):
        try:
            value = _text(binding.get(field), f"answerability.proof_binding.{field}")
        except AnswerabilityContractError as exc:
            return str(exc)
        if field.endswith("_sha256") or field == "proof_digest":
            if not _SHA256_RE.fullmatch(value.lower()):
                return f"answerability.proof_binding.{field} must be a 64-hex SHA-256"
    if campaign_id is not None and binding["campaign_id"] != campaign_id:
        return "answerability.proof_binding.campaign_id does not match the manifest"
    if binding["question"] != contract["question"]["research_question"]:
        return "answerability.proof_binding.question does not match the contract"
    if binding["estimand"] != contract["estimand"]["primary"]:
        return "answerability.proof_binding.estimand does not match the contract"
    proof_results = binding.get("proof_results")
    proof_surfaces = contract.get("proof_surfaces")
    if not isinstance(proof_results, Mapping) or not isinstance(proof_surfaces, Mapping):
        return "answerability.proof_binding must include canonical proof_results"
    if set(proof_results) != set(PROOF_SURFACES) or set(proof_surfaces) != set(PROOF_SURFACES):
        return "answerability.proof_binding proof_results must name exactly the six proof surfaces"
    for surface in PROOF_SURFACES:
        result = proof_results.get(surface)
        declaration = proof_surfaces.get(surface)
        if not isinstance(result, Mapping) or not isinstance(declaration, Mapping):
            return f"answerability.proof_binding.{surface} proof result must be a mapping"
        if result.get("status") != declaration.get("status"):
            return f"answerability.proof_binding.{surface} status does not match proof results"
        if result.get("required") != declaration.get("required"):
            return (
                f"answerability.proof_binding.{surface} required flag does not match proof results"
            )
    try:
        expected_digest = compute_proof_digest(binding, proof_results)
    except (TypeError, ValueError) as exc:
        return f"answerability.proof_binding proof_results are not canonical JSON: {exc}"
    if binding["proof_digest"].lower() != expected_digest:
        return "answerability.proof_binding.proof_digest does not match proof results"
    if repo_root is None:
        return "strict admission proof verification requires the repository root"
    for field, digest_field in (
        ("source_manifest", "manifest_sha256"),
        ("campaign_config", "config_sha256"),
    ):
        file_error = _proof_binding_file_error(
            binding,
            field=field,
            digest_field=digest_field,
            repo_root=repo_root,
        )
        if file_error:
            return file_error
    return None


def _list(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list) or not value:
        raise AnswerabilityContractError(f"{field} must be a non-empty list")
    return value


def _validate_question(question: Mapping[str, Any]) -> None:
    vocabulary = _list(
        question.get("decision_vocabulary"),
        "answerability.question.decision_vocabulary",
    )
    if not all(isinstance(item, str) and item.strip() for item in vocabulary):
        raise AnswerabilityContractError(
            "answerability.question.decision_vocabulary must contain only non-empty strings"
        )
    unknown_vocabulary = set(vocabulary) - _DECISION_VOCABULARY
    if unknown_vocabulary:
        raise AnswerabilityContractError(
            "answerability.question.decision_vocabulary contains unsupported values: "
            f"{sorted(unknown_vocabulary)}"
        )
    for field in _REQUIRED_TEXT_FIELDS["question"]:
        _text(question.get(field), f"answerability.question.{field}")


def _validate_estimand(estimand: Mapping[str, Any]) -> None:
    for field in _REQUIRED_TEXT_FIELDS["estimand"]:
        _text(estimand.get(field), f"answerability.estimand.{field}")


def _validate_producers(producers: list[Any]) -> None:
    for index, producer_value in enumerate(producers):
        producer = _mapping(producer_value, f"answerability.producers[{index}]")
        for field in _PRODUCER_TEXT_FIELDS:
            _text(producer.get(field), f"answerability.producers[{index}].{field}")
        if producer["status"] not in _VALID_PRODUCER_STATUSES:
            raise AnswerabilityContractError(
                f"answerability.producers[{index}].status must be one of "
                f"{sorted(_VALID_PRODUCER_STATUSES)}"
            )
        if producer["execution_mode"] not in _VALID_EXECUTION_MODES:
            raise AnswerabilityContractError(
                f"answerability.producers[{index}].execution_mode must be one of "
                f"{sorted(_VALID_EXECUTION_MODES)}"
            )
        if not isinstance(producer.get("required", True), bool):
            raise AnswerabilityContractError(
                f"answerability.producers[{index}].required must be a boolean"
            )


def _validate_analysis(analysis: Mapping[str, Any]) -> None:
    for field in _REQUIRED_TEXT_FIELDS["analysis"]:
        _text(analysis.get(field), f"answerability.analysis.{field}")
    if analysis["dry_run_status"] not in _VALID_DRY_RUN_STATUSES:
        raise AnswerabilityContractError(
            "answerability.analysis.dry_run_status must be passed, not_required, failed, blocked, or unknown"
        )
    if analysis["comparability_status"] not in _VALID_COMPARABILITY_STATUSES:
        raise AnswerabilityContractError(
            "answerability.analysis.comparability_status must be passed, not_required, failed, blocked, unknown, or mismatched"
        )


def _validate_design(design: Mapping[str, Any]) -> None:
    for field in _REQUIRED_TEXT_FIELDS["design"]:
        _text(design.get(field), f"answerability.design.{field}")
    if design["mode"] not in {"decision_capable", "diagnostic"}:
        raise AnswerabilityContractError(
            "answerability.design.mode must be decision_capable or diagnostic"
        )
    if design["power_status"] not in {"adequate", "not_required", "underpowered", "unknown"}:
        raise AnswerabilityContractError(
            "answerability.design.power_status must be adequate, not_required, underpowered, or unknown"
        )


def _validate_artifacts(artifacts: Mapping[str, Any]) -> None:
    for field in _REQUIRED_TEXT_FIELDS["artifacts"]:
        _text(artifacts.get(field), f"answerability.artifacts.{field}")
    checksums = artifacts.get("checksums")
    if (
        not isinstance(checksums, list)
        or not checksums
        or not all(isinstance(item, str) and item.strip() for item in checksums)
    ):
        raise AnswerabilityContractError(
            "answerability.artifacts.checksums must be a non-empty list of strings"
        )
    if artifacts["durability_status"] not in {"ready", "planned", "missing", "blocked"}:
        raise AnswerabilityContractError(
            "answerability.artifacts.durability_status must be ready, planned, missing, or blocked"
        )


def _validate_proof_surfaces(contract: Mapping[str, Any]) -> dict[str, ProofSurface]:
    """Validate an explicitly declared proof surface set.

    The section is optional for compatibility with pre-proof contracts. Once
    declared, all six named surfaces are required and each entry is strict.

    Returns:
        The validated proof surfaces, or an empty mapping for legacy contracts.
    """
    value = contract.get("proof_surfaces")
    if value is None:
        return {}
    surfaces = _mapping(value, "answerability.proof_surfaces")
    missing = sorted(set(PROOF_SURFACES) - set(surfaces))
    if missing:
        raise AnswerabilityContractError(
            "answerability.proof_surfaces is missing: " + ", ".join(missing)
        )
    unknown = sorted(set(surfaces) - set(PROOF_SURFACES))
    if unknown:
        raise AnswerabilityContractError(
            "answerability.proof_surfaces contains unsupported values: " + ", ".join(unknown)
        )
    return {
        name: ProofSurface.from_mapping(surfaces[name], f"answerability.proof_surfaces.{name}")
        for name in PROOF_SURFACES
    }


def _proof_findings(
    contract: Mapping[str, Any],
    *,
    enforce_admission_proof: bool,
) -> tuple[list[str], list[str]]:
    proof_surfaces = _validate_proof_surfaces(contract)
    if enforce_admission_proof and not proof_surfaces:
        return list(DECISION_REQUIRED_PROOF_SURFACES), []
    missing = [
        name
        for name, proof in proof_surfaces.items()
        if proof.required and proof.status != "passed"
    ]
    if enforce_admission_proof:
        missing.extend(
            name
            for name in DECISION_REQUIRED_PROOF_SURFACES
            if name not in proof_surfaces
            or not proof_surfaces[name].required
            or proof_surfaces[name].status != "passed"
        )
    optional_not_passed = [
        f"{name}={proof.status}"
        for name, proof in proof_surfaces.items()
        if not proof.required and proof.status != "passed"
    ]
    return missing, optional_not_passed


def validate_answerability_contract(contract: Mapping[str, Any]) -> None:
    """Validate the structural contract without interpreting campaign results."""
    if not isinstance(contract, Mapping):
        raise AnswerabilityContractError("answerability must be a mapping")
    if contract.get("schema_version") != ANSWERABILITY_SCHEMA:
        raise AnswerabilityContractError(
            f"answerability.schema_version must be {ANSWERABILITY_SCHEMA}"
        )
    for section in _REQUIRED_SECTIONS:
        if section == "producers":
            _validate_producers(_list(contract.get(section), f"answerability.{section}"))
        else:
            _mapping(contract.get(section), f"answerability.{section}")
    _validate_question(_mapping(contract["question"], "answerability.question"))
    _validate_estimand(_mapping(contract["estimand"], "answerability.estimand"))
    _validate_analysis(_mapping(contract["analysis"], "answerability.analysis"))
    _validate_design(_mapping(contract["design"], "answerability.design"))
    _validate_artifacts(_mapping(contract["artifacts"], "answerability.artifacts"))
    _validate_proof_surfaces(contract)


def evaluate_answerability(  # noqa: C901, PLR0912
    contract: Mapping[str, Any],
    *,
    enforce_admission_proof: bool = False,
    campaign_id: str | None = None,
    repo_root: Path | None = None,
) -> AnswerabilityResult:
    """Return the most conservative state supported by *contract*.

    Structural defects return ``invalid_contract``. Semantic blockers are
    ordered from missing producers through artifact durability. A diagnostic
    design may be valid without being decision-capable; it returns
    ``diagnostic_only`` and never ``answerable``.
    """
    try:
        validate_answerability_contract(contract)
    except AnswerabilityContractError as exc:
        return AnswerabilityResult("invalid_contract", (str(exc),))

    producers = [dict(_mapping(value, "producer")) for value in contract["producers"]]
    required_producers = [producer for producer in producers if producer.get("required", True)]
    missing_producers = [
        producer["field"]
        for producer in required_producers
        if producer["status"] != "available"
        or producer["execution_mode"] in {"fallback", "degraded", "unavailable"}
    ]
    if missing_producers:
        return AnswerabilityResult(
            "blocked_missing_producer",
            (
                "required producers are missing, unavailable, blocked, or fallback/degraded: "
                + ", ".join(sorted(missing_producers)),
            ),
        )

    analysis = _mapping(contract["analysis"], "answerability.analysis")
    if analysis["dry_run_status"] not in {"passed", "not_required"}:
        return AnswerabilityResult(
            "blocked_analysis_contract",
            (f"analysis dry-run status is {analysis['dry_run_status']!r}",),
        )
    if analysis["comparability_status"] not in {"passed", "not_required"}:
        return AnswerabilityResult(
            "blocked_noncomparable_rows",
            (f"row comparability status is {analysis['comparability_status']!r}",),
        )

    design = _mapping(contract["design"], "answerability.design")
    if design["power_status"] == "underpowered":
        return AnswerabilityResult(
            "blocked_underpowered",
            ("declared executable budget is underpowered for the minimally important effect",),
        )
    if design["power_status"] == "unknown":
        return AnswerabilityResult(
            "blocked_underpowered",
            ("design power or diagnostic-budget classification is unknown",),
        )

    artifacts = _mapping(contract["artifacts"], "answerability.artifacts")
    durability_status = artifacts["durability_status"]
    if durability_status in {"missing", "blocked"}:
        return AnswerabilityResult(
            "blocked_artifact_plan",
            (f"durable evidence plan is {durability_status}",),
        )

    if enforce_admission_proof and design["mode"] == "decision_capable":
        if analysis["dry_run_status"] != "passed":
            return AnswerabilityResult(
                "blocked_analysis_contract",
                (
                    "decision-capable admission requires analysis dry-run status 'passed'; "
                    f"got {analysis['dry_run_status']!r}",
                ),
            )
        if design["power_status"] != "adequate":
            return AnswerabilityResult(
                "blocked_underpowered",
                (
                    "decision-capable admission requires power status 'adequate'; "
                    f"got {design['power_status']!r}",
                ),
            )
    missing_proof, optional_not_passed_proof = _proof_findings(
        contract,
        enforce_admission_proof=(enforce_admission_proof and design["mode"] == "decision_capable"),
    )

    strict_admission = enforce_admission_proof and design["mode"] == "decision_capable"
    binding = contract.get("proof_binding")
    has_bound_results = isinstance(binding, Mapping) and isinstance(
        binding.get("proof_results"), Mapping
    )
    if strict_admission and has_bound_results:
        binding_error = _proof_binding_error(
            contract,
            campaign_id=campaign_id,
            repo_root=repo_root,
        )
        if binding_error:
            return AnswerabilityResult("blocked_missing_proof", (binding_error,))

    warnings_list = []
    optional_non_native = [
        producer["field"]
        for producer in producers
        if not producer.get("required", True)
        and (
            producer["status"] != "available"
            or producer["execution_mode"] not in {"native", "adapter"}
        )
    ]
    if optional_non_native:
        warnings_list.append(
            "optional non-native or non-available producers remain explicit and cannot be "
            "interpreted as zero: " + ", ".join(sorted(optional_non_native))
        )
    if optional_not_passed_proof:
        warnings_list.append(
            "optional proof surfaces are not passed and cannot support admission: "
            + ", ".join(sorted(optional_not_passed_proof))
        )
    warnings = tuple(warnings_list)
    if design["mode"] == "diagnostic" or durability_status == "planned":
        return AnswerabilityResult(
            "diagnostic_only",
            (
                "contract is executable only as a bounded diagnostic, not a decision-capable campaign",
            ),
            warnings,
        )
    if strict_admission and not has_bound_results:
        binding_error = _proof_binding_error(
            contract,
            campaign_id=campaign_id,
            repo_root=repo_root,
        )
        if binding_error and not missing_proof:
            return AnswerabilityResult("blocked_missing_proof", (binding_error,))
    if missing_proof:
        return AnswerabilityResult(
            "blocked_missing_proof",
            (
                "required proof surfaces are missing, not_run, unavailable, or failed: "
                + ", ".join(sorted(missing_proof)),
            ),
            warnings,
        )
    return AnswerabilityResult("answerable", (), warnings)


def answerability_from_manifest(
    manifest: Mapping[str, Any],
    *,
    enforce_admission_proof: bool = False,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Evaluate the optional answerability section of a campaign manifest.

    Returns:
        JSON-safe answerability state, reasons, and warnings.
    """
    contract = manifest.get("answerability")
    if contract is None:
        return {
            "schema_version": ANSWERABILITY_SCHEMA,
            "state": "not_declared",
            "decision_capable": False,
            "reasons": ["manifest does not declare answerability.v1"],
            "warnings": [],
        }
    if not isinstance(contract, Mapping):
        return AnswerabilityResult(
            "invalid_contract", ("answerability must be a mapping",)
        ).as_dict()
    campaign = manifest.get("campaign")
    campaign_id = campaign.get("id") if isinstance(campaign, Mapping) else None
    return evaluate_answerability(
        contract,
        enforce_admission_proof=enforce_admission_proof,
        campaign_id=campaign_id if isinstance(campaign_id, str) else None,
        repo_root=repo_root,
    ).as_dict()


__all__ = [
    "ANSWERABILITY_SCHEMA",
    "ANSWERABILITY_STATES",
    "DECISION_REQUIRED_PROOF_SURFACES",
    "PROOF_BINDING_SCHEMA",
    "PROOF_STATUSES",
    "PROOF_SURFACES",
    "AnswerabilityContractError",
    "AnswerabilityResult",
    "ProofStatus",
    "ProofSurface",
    "answerability_from_manifest",
    "compute_proof_digest",
    "evaluate_answerability",
    "strict_proof_input_provenance_error",
    "validate_answerability_contract",
]
