"""Typed contract for negative-result mechanism-boundary atlas cards."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from robot_sf.common.json_pointer import json_pointer
from robot_sf.errors import RobotSfError

SCHEMA_VERSION = "mechanism_boundary_atlas.v1"
SCHEMA_FILE = Path(__file__).with_name("schemas") / "mechanism_boundary_atlas.v1.json"
PROMOTION_PATTERNS = (
    "benchmark evidence",
    "paper evidence",
    "paper-grade",
    "dissertation-ready",
    "results chapter evidence",
    "establishes",
    "proves",
    "promotes",
    "promotion claim",
)
CONTROLLED_STATES = frozenset(
    {
        "supported_positive",
        "supported_negative",
        "not_supported",
        "inconclusive",
        "invalid_evidence_contract",
        "unavailable",
    }
)
BOUNDARY_LABELS = frozenset(
    {
        "mechanism_not_activated",
        "mechanism_active_no_endpoint_benefit",
        "candidate_or_action_distribution_collapse",
        "adapter_or_representation_mismatch",
        "objective_or_information_insufficiency",
        "scenario_or_threshold_dependence",
        "design_underpowered",
        "missing_required_producer",
        "noncomparable_rows",
        "measurement_instability",
        "artifact_not_durable",
    }
)
LOCAL_ONLY_ROOTS = frozenset({".git", ".venv", "output", "results"})
CODE_CONFIG_IDENTITY_KINDS = frozenset({"path", "commit", "digest"})
SUPPORTED_SOURCE_RESULT_STATES = frozenset(
    {"supported_positive", "supported_negative", "not_supported"}
)


@dataclass(frozen=True, slots=True)
class AtlasValidationIssue:
    """One mechanism-boundary atlas validation issue."""

    path: str
    message: str


@dataclass(frozen=True, slots=True)
class SourceRef:
    """Tracked or explicitly unavailable source reference."""

    path: str
    status: str
    role: str
    sha256: str | None = None
    unavailable_reason: str | None = None
    digest_verified: bool | None = None
    tracked: bool | None = None


@dataclass(frozen=True, slots=True)
class CodeConfigIdentity:
    """Structured code, configuration, commit, or dataset identity record."""

    kind: str
    label: str
    path: str | None = None
    commit: str | None = None
    sha256: str | None = None
    repository: str | None = None


@dataclass(frozen=True, slots=True)
class ResultState:
    """Controlled result-state dimension, separate from mechanism interpretation."""

    controlled_state: str
    evidence_tier: str
    state_reason: str


@dataclass(frozen=True, slots=True)
class MechanismBoundary:
    """Mechanism/evidence-boundary dimension for the case."""

    boundary_labels: list[str]
    mechanism_dimension: str
    status: str
    evidence_boundary: str
    confidence: float
    condition_that_would_change: str


@dataclass(frozen=True, slots=True)
class ScopeBoundary:
    """Scope, estimand, uncertainty, and exclusion limits."""

    scope: str
    estimand: str
    uncertainty: str
    exclusions: list[str]


@dataclass(frozen=True, slots=True)
class ClaimBoundary:
    """Allowed and forbidden claim wording for one card."""

    bounded_claims: list[str]
    forbidden_wording: list[str]
    smallest_next_proof_or_stop: str


@dataclass(frozen=True, slots=True)
class MechanismCaseCard:
    """One source-backed negative-result mechanism-boundary card."""

    case_id: str
    title: str
    question_and_hypothesis: str
    linked_issues: list[int]
    code_or_config_identity: list[CodeConfigIdentity]
    source_refs: list[SourceRef]
    result_state: ResultState
    mechanism_boundary: MechanismBoundary
    mechanism_activation_evidence: str
    observed_result: str
    scope_boundary: ScopeBoundary
    hypotheses_contradicted: list[str]
    hypotheses_still_viable: list[str]
    claim_boundary: ClaimBoundary
    dissertation_admission_status: str
    deferred_or_blocked_outputs: list[str]


@dataclass(frozen=True, slots=True)
class MechanismBoundaryAtlas:
    """Validated atlas payload."""

    schema_version: str
    atlas_id: str
    issue: int
    generated_by: str
    claim_boundary: str
    cards: list[MechanismCaseCard]

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe primitives."""

        return _drop_none(asdict(self))


class MechanismBoundaryAtlasError(RobotSfError, ValueError):
    """Raised when a mechanism-boundary atlas fails validation."""

    def __init__(self, issues: Sequence[AtlasValidationIssue], *, source: Path | None = None):
        """Build an actionable validation error from atlas issues."""

        self.issues = tuple(issues)
        self.source = str(source) if source is not None else None
        prefix = f"{self.source}: " if self.source else ""
        super().__init__(prefix + "; ".join(f"{i.path}: {i.message}" for i in self.issues))


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_schema() -> dict[str, Any]:
    """Load the mechanism-boundary atlas JSON Schema.

    Returns:
        Parsed JSON Schema dictionary.
    """

    return json.loads(SCHEMA_FILE.read_text(encoding="utf-8"))


def load_atlas(path: Path, *, repo_root: Path | None = None) -> MechanismBoundaryAtlas:
    """Load, validate, and type an atlas JSON payload.

    Returns:
        Typed atlas payload.
    """

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise MechanismBoundaryAtlasError(
            [AtlasValidationIssue("/", "expected object payload")],
            source=path,
        )
    issues = validate_atlas_payload(payload, repo_root=repo_root or Path.cwd())
    if issues:
        raise MechanismBoundaryAtlasError(issues, source=path)
    return atlas_from_payload(payload)


def atlas_from_payload(payload: Mapping[str, Any]) -> MechanismBoundaryAtlas:
    """Convert a validated payload into typed dataclasses.

    Returns:
        Typed atlas payload.
    """

    cards = []
    for card in payload["cards"]:
        cards.append(
            MechanismCaseCard(
                case_id=card["case_id"],
                title=card["title"],
                question_and_hypothesis=card["question_and_hypothesis"],
                linked_issues=list(card["linked_issues"]),
                code_or_config_identity=[
                    CodeConfigIdentity(**identity) for identity in card["code_or_config_identity"]
                ],
                source_refs=[SourceRef(**source) for source in card["source_refs"]],
                result_state=ResultState(**card["result_state"]),
                mechanism_boundary=MechanismBoundary(**card["mechanism_boundary"]),
                mechanism_activation_evidence=card["mechanism_activation_evidence"],
                observed_result=card["observed_result"],
                scope_boundary=ScopeBoundary(**card["scope_boundary"]),
                hypotheses_contradicted=list(card["hypotheses_contradicted"]),
                hypotheses_still_viable=list(card["hypotheses_still_viable"]),
                claim_boundary=ClaimBoundary(**card["claim_boundary"]),
                dissertation_admission_status=card["dissertation_admission_status"],
                deferred_or_blocked_outputs=list(card["deferred_or_blocked_outputs"]),
            )
        )
    return MechanismBoundaryAtlas(
        schema_version=payload["schema_version"],
        atlas_id=payload["atlas_id"],
        issue=payload["issue"],
        generated_by=payload["generated_by"],
        claim_boundary=payload["claim_boundary"],
        cards=cards,
    )


def validate_atlas_payload(
    payload: Mapping[str, Any],
    *,
    repo_root: Path,
    verify_sources: bool = True,
) -> list[AtlasValidationIssue]:
    """Validate schema, claim boundaries, and source checksums.

    Returns:
        Validation issues. Empty means valid.
    """

    issues = _schema_issues(payload)
    issues.extend(_semantic_issues(payload))
    issues.extend(_identity_issues(payload, repo_root=repo_root, verify_sources=verify_sources))
    if verify_sources:
        issues.extend(_source_issues(payload, repo_root=repo_root))
    return issues


def build_atlas(
    input_path: Path,
    *,
    repo_root: Path,
    output_path: Path | None = None,
) -> MechanismBoundaryAtlas:
    """Build a deterministic atlas from a checked-in input manifest.

    Returns:
        Typed atlas payload.
    """

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise MechanismBoundaryAtlasError(
            [AtlasValidationIssue("/", "expected object payload")],
            source=input_path,
        )
    output_payload = _payload_with_verified_sources(payload, repo_root=repo_root)
    issues = validate_atlas_payload(output_payload, repo_root=repo_root)
    if issues:
        raise MechanismBoundaryAtlasError(issues, source=input_path)
    atlas = atlas_from_payload(output_payload)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(atlas.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return atlas


def _payload_with_verified_sources(
    payload: Mapping[str, Any],
    *,
    repo_root: Path,
) -> dict[str, Any]:
    """Return a copy with digest_verified/tracked filled for available sources."""

    result = json.loads(json.dumps(payload, sort_keys=True))
    for card in result.get("cards", []):
        for source in card.get("source_refs", []):
            if source.get("status") != "available":
                source["digest_verified"] = False
                source["tracked"] = False
                continue
            source_path = _safe_source_path(repo_root, source.get("path"))
            if source_path is None:
                source["digest_verified"] = False
                source["tracked"] = False
                continue
            source["tracked"] = _is_tracked(repo_root, source["path"])
            source["digest_verified"] = (
                source_path.is_file() and sha256_file(source_path) == source["sha256"]
            )
    return result


def _drop_none(value: Any) -> Any:
    """Return JSON primitives with ``None`` fields omitted from mappings."""

    if isinstance(value, dict):
        return {key: _drop_none(nested) for key, nested in value.items() if nested is not None}
    if isinstance(value, list):
        return [_drop_none(nested) for nested in value]
    return value


def _schema_issues(payload: Mapping[str, Any]) -> list[AtlasValidationIssue]:
    validator = Draft202012Validator(load_schema())
    return [
        AtlasValidationIssue(json_pointer(error.absolute_path), error.message)
        for error in sorted(validator.iter_errors(payload), key=lambda err: list(err.absolute_path))
    ]


def _semantic_issues(payload: Mapping[str, Any]) -> list[AtlasValidationIssue]:  # noqa: C901
    issues: list[AtlasValidationIssue] = []
    top_level_claim_boundary = payload.get("claim_boundary")
    if isinstance(top_level_claim_boundary, str):
        _claim_promotion_text_issues(top_level_claim_boundary, "/claim_boundary", issues)
    cards = payload.get("cards")
    if not isinstance(cards, list):
        return issues
    seen_case_ids: set[str] = set()
    for index, card in enumerate(cards):
        if not isinstance(card, Mapping):
            continue
        prefix = f"/cards/{index}"
        case_id = card.get("case_id")
        if isinstance(case_id, str):
            if case_id in seen_case_ids:
                issues.append(AtlasValidationIssue(f"{prefix}/case_id", f"duplicate {case_id!r}"))
            seen_case_ids.add(case_id)
        _claim_promotion_issues(card, prefix, issues)
        result_state = card.get("result_state")
        mechanism_boundary = card.get("mechanism_boundary")
        if isinstance(result_state, Mapping):
            controlled_state = result_state.get("controlled_state")
            if controlled_state not in CONTROLLED_STATES:
                issues.append(
                    AtlasValidationIssue(
                        f"{prefix}/result_state/controlled_state",
                        "controlled_state must use #7032 exact vocabulary",
                    )
                )
        if isinstance(result_state, Mapping) and isinstance(mechanism_boundary, Mapping):
            issues.extend(_boundary_label_issues(mechanism_boundary, prefix))
            if result_state.get("controlled_state") == mechanism_boundary.get("status"):
                issues.append(
                    AtlasValidationIssue(
                        prefix,
                        "controlled result state must not be reused as mechanism status",
                    )
                )
            if result_state.get("state_reason") == mechanism_boundary.get("evidence_boundary"):
                issues.append(
                    AtlasValidationIssue(
                        prefix,
                        "result_state reason must not duplicate mechanism evidence_boundary",
                    )
                )
        if card.get("result_state") == card.get("mechanism_boundary"):
            issues.append(
                AtlasValidationIssue(
                    prefix,
                    "result_state and mechanism_boundary must remain separate dimensions",
                )
            )
        if (
            isinstance(result_state, Mapping)
            and result_state.get("controlled_state") in SUPPORTED_SOURCE_RESULT_STATES
        ):
            source_refs = card.get("source_refs")
            if not isinstance(source_refs, list) or not any(
                isinstance(source, Mapping) and source.get("status") == "available"
                for source in source_refs
            ):
                issues.append(
                    AtlasValidationIssue(
                        f"{prefix}/source_refs",
                        "supported or not-supported result requires at least one available source",
                    )
                )
    if len(cards) < 6:
        issues.append(AtlasValidationIssue("/cards", "atlas requires at least six case cards"))
    return issues


def _boundary_label_issues(
    mechanism_boundary: Mapping[str, Any],
    card_prefix: str,
) -> list[AtlasValidationIssue]:
    """Validate the issue-controlled mechanism/evidence boundary labels.

    Returns:
        Validation issues for unknown or duplicate labels.
    """

    labels = mechanism_boundary.get("boundary_labels")
    if not isinstance(labels, list):
        return []
    issues: list[AtlasValidationIssue] = []
    label_path = f"{card_prefix}/mechanism_boundary/boundary_labels"
    if not all(isinstance(label, str) for label in labels):
        issues.append(
            AtlasValidationIssue(
                label_path,
                "boundary_labels must contain only strings from #7032 controlled vocabulary",
            )
        )
        return issues
    unknown_labels = set(labels).difference(BOUNDARY_LABELS)
    if unknown_labels:
        issues.append(
            AtlasValidationIssue(
                label_path,
                f"boundary_labels must use #7032 controlled vocabulary: {sorted(unknown_labels)}",
            )
        )
    if len(labels) != len(set(labels)):
        issues.append(
            AtlasValidationIssue(label_path, "boundary_labels must not contain duplicates")
        )
    return issues


def _claim_promotion_issues(
    card: Mapping[str, Any],
    prefix: str,
    issues: list[AtlasValidationIssue],
) -> None:
    text_parts: list[str] = []
    for key in ("title",):
        value = card.get(key)
        if isinstance(value, str):
            text_parts.append(value)
    for section_name in (
        "question_and_hypothesis",
        "result_state",
        "mechanism_boundary",
        "mechanism_activation_evidence",
        "observed_result",
        "scope_boundary",
        "hypotheses_contradicted",
        "hypotheses_still_viable",
    ):
        section = card.get(section_name)
        text_parts.extend(_strings_in(section))
    claim_boundary = card.get("claim_boundary")
    if isinstance(claim_boundary, Mapping):
        text_parts.extend(_strings_in(claim_boundary.get("bounded_claims", [])))
        text_parts.extend(_strings_in(claim_boundary.get("smallest_next_proof_or_stop", "")))
    _claim_promotion_text_issues("\n".join(text_parts), prefix, issues)


def _claim_promotion_text_issues(
    text: str,
    prefix: str,
    issues: list[AtlasValidationIssue],
) -> None:
    """Reject promotion wording in any claim-boundary text surface."""

    normalized_text = text.lower()
    for pattern in PROMOTION_PATTERNS:
        if _contains_unnegated_pattern(normalized_text, pattern):
            issues.append(
                AtlasValidationIssue(
                    prefix,
                    f"claim text contains promotion pattern {pattern!r}",
                )
            )


def _contains_unnegated_pattern(text: str, pattern: str) -> bool:
    """Return whether promotion wording appears outside an explicit negation."""

    negation = re.compile(r"(?:\bnot|\bno|\bnever|\bwithout|\bexcluding)\s+$")
    for match in re.finditer(re.escape(pattern), text):
        context = text[max(0, match.start() - 24) : match.start()]
        if not negation.search(context):
            return True
    return False


def _strings_in(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        result: list[str] = []
        for nested in value.values():
            result.extend(_strings_in(nested))
        return result
    if isinstance(value, list):
        result = []
        for nested in value:
            result.extend(_strings_in(nested))
        return result
    return []


def _identity_issues(
    payload: Mapping[str, Any],
    *,
    repo_root: Path,
    verify_sources: bool,
) -> list[AtlasValidationIssue]:
    """Validate that code/config identities are bound to inspectable records.

    Returns:
        Validation issues for unbound or drifted identity records.
    """

    issues: list[AtlasValidationIssue] = []
    for card_index, card in enumerate(payload.get("cards", [])):
        if not isinstance(card, Mapping):
            continue
        identities = card.get("code_or_config_identity")
        if not isinstance(identities, list):
            continue
        for identity_index, identity in enumerate(identities):
            if not isinstance(identity, Mapping):
                continue
            prefix = f"/cards/{card_index}/code_or_config_identity/{identity_index}"
            kind = identity.get("kind")
            if kind not in CODE_CONFIG_IDENTITY_KINDS:
                continue
            if kind == "path":
                issues.extend(_path_identity_issues(identity, prefix, repo_root))
            elif kind == "commit":
                issues.extend(_commit_identity_issues(identity, prefix, repo_root))
            elif kind == "digest" and verify_sources:
                issues.extend(_digest_identity_issues(identity, prefix, card, repo_root))
    return issues


def _path_identity_issues(
    identity: Mapping[str, Any],
    prefix: str,
    repo_root: Path,
) -> list[AtlasValidationIssue]:
    """Validate a repository path identity and its content digest.

    Returns:
        Validation issues for the path or digest.
    """

    path_value = identity.get("path")
    if not isinstance(path_value, str):
        return []
    identity_path = _safe_source_path(repo_root, path_value)
    if identity_path is None:
        return [
            AtlasValidationIssue(
                f"{prefix}/path",
                "identity path must be repository-root relative and resolve inside the repository",
            )
        ]
    if not identity_path.is_file():
        return [
            AtlasValidationIssue(f"{prefix}/path", "identity path is missing or not a regular file")
        ]

    issues: list[AtlasValidationIssue] = []
    if not _is_tracked(repo_root, path_value):
        issues.append(AtlasValidationIssue(f"{prefix}/path", "identity path is not tracked"))
    expected_sha = identity.get("sha256")
    if isinstance(expected_sha, str):
        actual_sha = sha256_file(identity_path)
        if actual_sha != expected_sha:
            issues.append(
                AtlasValidationIssue(
                    f"{prefix}/sha256",
                    f"identity checksum mismatch: expected {expected_sha}, got {actual_sha}",
                )
            )
    return issues


def _commit_identity_issues(
    identity: Mapping[str, Any],
    prefix: str,
    repo_root: Path,
) -> list[AtlasValidationIssue]:
    """Validate a local commit identity when it claims this repository.

    Returns:
        Validation issues for an unknown local commit.
    """

    commit = identity.get("commit")
    if identity.get("repository") != "this_repository" or not isinstance(commit, str):
        return []
    completed = subprocess.run(
        ["git", "-C", str(repo_root), "cat-file", "-e", f"{commit}^{{commit}}"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if completed.returncode == 0:
        return []
    return [
        AtlasValidationIssue(
            f"{prefix}/commit",
            "this_repository commit identity is not present in the repository",
        )
    ]


def _digest_identity_issues(
    identity: Mapping[str, Any],
    prefix: str,
    card: Mapping[str, Any],
    repo_root: Path,
) -> list[AtlasValidationIssue]:
    """Validate that a non-file digest is recorded by a verified source.

    Returns:
        Validation issues for an unbound digest.
    """

    digest = identity.get("sha256")
    if isinstance(digest, str) and _digest_is_bound_to_source(digest, card, repo_root):
        return []
    return [
        AtlasValidationIssue(
            f"{prefix}/sha256",
            "digest identity must occur in a verified available source record",
        )
    ]


def _digest_is_bound_to_source(
    digest: str,
    card: Mapping[str, Any],
    repo_root: Path,
) -> bool:
    """Return whether a digest is recorded by a verified source manifest."""

    for source in card.get("source_refs", []):
        if not isinstance(source, Mapping) or source.get("status") != "available":
            continue
        path_value = source.get("path")
        source_path = _safe_source_path(repo_root, path_value)
        expected_sha = source.get("sha256")
        if (
            source_path is None
            or not source_path.is_file()
            or not isinstance(path_value, str)
            or not _is_tracked(repo_root, path_value)
            or not isinstance(expected_sha, str)
            or sha256_file(source_path) != expected_sha
        ):
            continue
        try:
            if digest.encode("ascii") in source_path.read_bytes():
                return True
        except (OSError, UnicodeEncodeError):
            continue
    return False


def _source_issues(  # noqa: C901
    payload: Mapping[str, Any],
    *,
    repo_root: Path,
) -> list[AtlasValidationIssue]:
    issues: list[AtlasValidationIssue] = []
    for card_index, card in enumerate(payload.get("cards", [])):
        if not isinstance(card, Mapping):
            continue
        for source_index, source in enumerate(card.get("source_refs", [])):
            if not isinstance(source, Mapping):
                continue
            prefix = f"/cards/{card_index}/source_refs/{source_index}"
            status = source.get("status")
            path_value = source.get("path")
            if not isinstance(path_value, str):
                continue
            source_path = _safe_source_path(repo_root, path_value)
            if source_path is None:
                issues.append(
                    AtlasValidationIssue(
                        f"{prefix}/path",
                        "source path must be repository-root relative and resolve inside the repository",
                    )
                )
                continue
            if status != "available":
                continue
            if not source_path.is_file():
                issues.append(AtlasValidationIssue(f"{prefix}/path", "available source is missing"))
                continue
            if not _is_tracked(repo_root, path_value):
                issues.append(
                    AtlasValidationIssue(f"{prefix}/path", "available source is not tracked")
                )
            expected_sha = source.get("sha256")
            if isinstance(expected_sha, str):
                actual_sha = sha256_file(source_path)
                if actual_sha != expected_sha:
                    issues.append(
                        AtlasValidationIssue(
                            f"{prefix}/sha256",
                            f"checksum mismatch: expected {expected_sha}, got {actual_sha}",
                        )
                    )
    return issues


def _is_absolute_or_traversal(path: str) -> bool:
    candidate = Path(path)
    return candidate.is_absolute() or ".." in candidate.parts


def _safe_source_path(repo_root: Path, path: object) -> Path | None:
    """Resolve a source path only when it stays inside the repository.

    Returns:
        Resolved path, or ``None`` for absolute, traversing, or escaping paths.
    """

    if not isinstance(path, str):
        return None
    try:
        if _is_absolute_or_traversal(path):
            return None
        parts = Path(path).parts
        if not parts or parts[0] in LOCAL_ONLY_ROOTS or ".worktrees" in parts:
            return None
        repository = repo_root.resolve()
        unresolved = repository / path
        current = repository
        for part in parts:
            current /= part
            if current.is_symlink():
                return None
        candidate = unresolved.resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        return None
    try:
        candidate.relative_to(repository)
    except ValueError:
        return None
    return candidate


def _is_tracked(repo_root: Path, path: str) -> bool:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), "ls-files", "--error-unmatch", "--", path],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return False
    return completed.returncode == 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the mechanism-boundary atlas CLI.

    Returns:
        Process exit code.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    try:
        atlas = build_atlas(args.input, repo_root=args.repo_root, output_path=args.output)
    except MechanismBoundaryAtlasError as exc:
        for issue in exc.issues:
            sys.stderr.write(f"{issue.path}: {issue.message}\n")
        return 2
    sys.stdout.write(f"wrote {args.output} ({len(atlas.cards)} cards)\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
