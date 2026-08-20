#!/usr/bin/env python3
"""Validate and resolve fail-closed typed issue dependency packets.

``issue_dependency_packet.v1`` is a small workflow contract, not a second
issue graph.  It records one explicit predicate per dependency, its mandatory
or advisory role, current observation, verdict, exact unblock condition, and
freshness keys.  The evaluator is side-effect-free: REST reads, local Git
queries, and local path reads are bounded inputs to a report and never perform
claims, label writes, downloads, merges, or issue closure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Callable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA = "issue_dependency_packet.v1"
EVALUATION_SCHEMA = "issue_dependency_evaluation.v1"
DEFAULT_REPO = "ll7/robot_sf_ll7"
SHA_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REPOSITORY_RE = re.compile(r"[^/\s]+/[^/\s]+")

DEPENDENCY_KINDS = frozenset(
    {
        "issue_state",
        "pull_request_state",
        "commit_present",
        "path_present",
        "artifact_digest",
        "external_input",
        "environment_capability",
        "human_ruling",
    }
)
VERDICTS = frozenset({"satisfied", "unsatisfied", "unavailable", "conflict", "invalid"})
FRESHNESS_KEYS = frozenset(
    {
        "issue_state",
        "issue_body",
        "pull_request_state",
        "pull_request_head",
        "pull_request_base",
        "commit",
        "path",
        "artifact",
        "external_input",
        "environment",
        "human_ruling",
    }
)
PACKET_FIELDS = frozenset(
    {"schema", "repository", "issue", "contract", "dependencies", "packet_digest"}
)
ROW_FIELDS = frozenset(
    {
        "id",
        "repository",
        "kind",
        "requirement",
        "mandatory",
        "source",
        "observed",
        "verdict",
        "unblock_condition",
        "freshness",
    }
)
STATE_VALUES = frozenset({"OPEN", "CLOSED", "MERGED"})
PATH_TYPES = frozenset({"any", "file", "directory"})

GhRunner = Callable[[list[str]], subprocess.CompletedProcess[str]]
GitRunner = Callable[[list[str]], subprocess.CompletedProcess[str]]


def _is_int(value: object) -> bool:
    """Return whether ``value`` is an integer but not a boolean."""
    return isinstance(value, int) and not isinstance(value, bool)


def _is_sha(value: object) -> bool:
    """Accept only full Git SHA-1 or SHA-256 spellings."""
    return isinstance(value, str) and SHA_RE.fullmatch(value) is not None


def _is_sha256(value: object) -> bool:
    """Accept only lowercase SHA-256 digests."""
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def sha256_bytes(value: bytes) -> str:
    """Return the stable digest used by packet and contract records."""
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    """Digest UTF-8 text without normalizing its bytes."""
    return sha256_bytes(value.encode("utf-8"))


def _canonical_payload(payload: Mapping[str, Any]) -> bytes:
    """Serialize a payload deterministically without its self-digest."""
    unsigned = {key: value for key, value in payload.items() if key != "packet_digest"}
    try:
        rendered = json.dumps(
            unsigned,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"packet is not canonical JSON: {exc}") from exc
    return rendered.encode("utf-8")


def compute_packet_digest(packet: Mapping[str, Any]) -> str:
    """Compute the packet self-digest without modifying the input."""
    return sha256_bytes(_canonical_payload(packet))


def _mapping(value: object, *, field: str, errors: list[str]) -> Mapping[str, Any] | None:
    """Require a JSON object and record a stable error when it is absent."""
    if not isinstance(value, Mapping):
        errors.append(f"{field} must be an object")
        return None
    return value


def _string(value: object, *, field: str, errors: list[str]) -> str | None:
    """Require a non-empty string field."""
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{field} must be a non-empty string")
        return None
    return value


def _path_error(value: object, *, field: str, allow_uri: bool = False) -> str | None:
    """Validate a repository-relative POSIX path or explicitly allowed URI."""
    if not isinstance(value, str) or not value.strip():
        return f"{field} must be a non-empty string"
    text = value.strip()
    if allow_uri and "://" in text:
        return None
    if "\\" in text:
        return f"{field} must use repository-relative POSIX paths"
    path = PurePosixPath(text)
    if not path.parts or path.is_absolute() or "." in path.parts or ".." in path.parts:
        return f"{field} must not escape the repository root"
    return None


def _validate_source(value: object, *, field: str, errors: list[str]) -> None:
    """Validate provenance for the requirement without interpreting its claim."""
    source = _mapping(value, field=field, errors=errors)
    if source is None:
        return
    _string(source.get("kind"), field=f"{field}.kind", errors=errors)
    _string(source.get("ref"), field=f"{field}.ref", errors=errors)
    if "digest" in source and not _is_sha256(source.get("digest")):
        errors.append(f"{field}.digest must be a lowercase SHA-256 digest when present")


def _validate_state_requirement(
    requirement: Mapping[str, Any], *, field: str, errors: list[str]
) -> None:
    """Validate an issue or pull-request state predicate."""
    number = requirement.get("number")
    if not _is_int(number) or number <= 0:
        errors.append(f"{field}.number must be a positive integer")
    if requirement.get("state") not in STATE_VALUES:
        errors.append(f"{field}.state must be one of {sorted(STATE_VALUES)}")
    for exact in ("head_sha", "base_sha"):
        if exact in requirement and not _is_sha(requirement.get(exact)):
            errors.append(f"{field}.{exact} must be a full Git SHA when present")
    for ref in ("head_ref", "base_ref"):
        if ref in requirement:
            _string(requirement.get(ref), field=f"{field}.{ref}", errors=errors)


def _validate_issue_state_requirement(
    requirement: Mapping[str, Any], *, field: str, errors: list[str]
) -> None:
    """Validate an issue state, which cannot be ``MERGED``."""
    _validate_state_requirement(requirement, field=field, errors=errors)
    if requirement.get("state") == "MERGED":
        errors.append(f"{field}.state must be OPEN or CLOSED for an issue")


def _validate_commit_requirement(
    requirement: Mapping[str, Any], *, field: str, errors: list[str]
) -> None:
    """Validate a commit presence and optional ancestry predicate."""
    if not _is_sha(requirement.get("sha")):
        errors.append(f"{field}.sha must be a full Git SHA")
    if "ancestor_of" in requirement and not _is_sha(requirement.get("ancestor_of")):
        errors.append(f"{field}.ancestor_of must be a full Git SHA when present")


def _validate_path_requirement(
    requirement: Mapping[str, Any], *, field: str, errors: list[str]
) -> None:
    """Validate a repository-relative path predicate."""
    if error := _path_error(requirement.get("path"), field=f"{field}.path"):
        errors.append(error)
    if requirement.get("path_type", "any") not in PATH_TYPES:
        errors.append(f"{field}.path_type must be one of {sorted(PATH_TYPES)}")
    if "ref" in requirement:
        _string(requirement.get("ref"), field=f"{field}.ref", errors=errors)


def _validate_artifact_requirement(
    requirement: Mapping[str, Any], *, field: str, errors: list[str]
) -> None:
    """Validate an artifact path/URI, schema, and exact digest predicate."""
    if error := _path_error(requirement.get("path"), field=f"{field}.path", allow_uri=True):
        errors.append(error)
    _string(requirement.get("schema"), field=f"{field}.schema", errors=errors)
    if not _is_sha256(requirement.get("digest")):
        errors.append(f"{field}.digest must be a lowercase SHA-256 digest")


def _validate_external_requirement(
    requirement: Mapping[str, Any], *, field: str, errors: list[str]
) -> None:
    """Validate an external identifier and explicit predicate."""
    _string(requirement.get("identifier"), field=f"{field}.identifier", errors=errors)
    _string(requirement.get("predicate"), field=f"{field}.predicate", errors=errors)
    if "schema" in requirement:
        _string(requirement.get("schema"), field=f"{field}.schema", errors=errors)
    if "digest" in requirement and not _is_sha256(requirement.get("digest")):
        errors.append(f"{field}.digest must be a lowercase SHA-256 digest when present")
    if "uri" in requirement:
        _string(requirement.get("uri"), field=f"{field}.uri", errors=errors)


def _validate_environment_requirement(
    requirement: Mapping[str, Any], *, field: str, errors: list[str]
) -> None:
    """Validate an environment capability and exact predicate."""
    _string(requirement.get("name"), field=f"{field}.name", errors=errors)
    _string(requirement.get("predicate"), field=f"{field}.predicate", errors=errors)


def _validate_ruling_requirement(
    requirement: Mapping[str, Any], *, field: str, errors: list[str]
) -> None:
    """Validate a human-ruling issue number and exact token."""
    number = requirement.get("issue")
    if not _is_int(number) or number <= 0:
        errors.append(f"{field}.issue must be a positive integer")
    _string(requirement.get("token"), field=f"{field}.token", errors=errors)


def _validate_requirement(kind: str, value: object, *, field: str, errors: list[str]) -> None:
    """Validate the exact predicate shape required by one dependency kind."""
    requirement = _mapping(value, field=field, errors=errors)
    if requirement is None:
        return
    validators = {
        "issue_state": _validate_issue_state_requirement,
        "pull_request_state": _validate_state_requirement,
        "commit_present": _validate_commit_requirement,
        "path_present": _validate_path_requirement,
        "artifact_digest": _validate_artifact_requirement,
        "external_input": _validate_external_requirement,
        "environment_capability": _validate_environment_requirement,
        "human_ruling": _validate_ruling_requirement,
    }
    validator = validators.get(kind)
    if validator is None:
        errors.append(f"{field} has unsupported dependency kind {kind!r}")
    else:
        validator(requirement, field=field, errors=errors)


def _validate_row(row: object, *, index: int, repository: str, errors: list[str]) -> str | None:
    """Validate one dependency row and return its identifier when available."""
    field = f"dependencies[{index}]"
    mapping = _mapping(row, field=field, errors=errors)
    if mapping is None:
        return None
    unknown = sorted(str(key) for key in mapping if key not in ROW_FIELDS)
    if unknown:
        errors.append(f"{field} has unsupported field(s): {', '.join(unknown)}")
    identifier = _string(mapping.get("id"), field=f"{field}.id", errors=errors)
    row_repository = _string(mapping.get("repository"), field=f"{field}.repository", errors=errors)
    if row_repository is not None and row_repository != repository:
        errors.append(f"{field}.repository must match packet.repository")
    kind = mapping.get("kind")
    if not isinstance(kind, str) or kind not in DEPENDENCY_KINDS:
        errors.append(f"{field}.kind must be exactly one of {sorted(DEPENDENCY_KINDS)}")
    else:
        _validate_requirement(
            kind, mapping.get("requirement"), field=f"{field}.requirement", errors=errors
        )
    if not isinstance(mapping.get("mandatory"), bool):
        errors.append(f"{field}.mandatory must be boolean")
    _validate_source(mapping.get("source"), field=f"{field}.source", errors=errors)
    if not isinstance(mapping.get("observed"), Mapping):
        errors.append(f"{field}.observed must be an object")
    verdict = mapping.get("verdict")
    if verdict not in VERDICTS:
        errors.append(f"{field}.verdict must be one of {sorted(VERDICTS)}")
    _string(mapping.get("unblock_condition"), field=f"{field}.unblock_condition", errors=errors)
    freshness = mapping.get("freshness")
    if not isinstance(freshness, list) or not freshness:
        errors.append(f"{field}.freshness must be a non-empty array")
    elif any(not isinstance(item, str) or item not in FRESHNESS_KEYS for item in freshness):
        errors.append(f"{field}.freshness contains an unknown key")
    return identifier


def _validate_packet_metadata(
    packet: Mapping[str, Any],
    *,
    errors: list[str],
    expected_repository: str | None,
    expected_issue: int | None,
    contract_text: str | None,
) -> tuple[object, object, str | None]:
    """Validate packet identity and its issue-contract digest."""
    repository = packet.get("repository")
    if not isinstance(repository, str) or REPOSITORY_RE.fullmatch(repository) is None:
        errors.append("repository must be an OWNER/REPO string")
    elif expected_repository is not None and repository != expected_repository:
        errors.append(f"repository {repository} does not match expected {expected_repository}")
    issue = packet.get("issue")
    if not _is_int(issue) or issue <= 0:
        errors.append("issue must be a positive integer")
    elif expected_issue is not None and issue != expected_issue:
        errors.append(f"issue {issue} does not match expected issue {expected_issue}")
    contract = _mapping(packet.get("contract"), field="contract", errors=errors)
    contract_digest: str | None = None
    if contract is not None:
        _string(contract.get("source"), field="contract.source", errors=errors)
        contract_digest = (
            contract.get("digest") if isinstance(contract.get("digest"), str) else None
        )
        if not _is_sha256(contract_digest):
            errors.append("contract.digest must be a lowercase SHA-256 digest")
        elif contract_text is not None and sha256_text(contract_text) != contract_digest:
            errors.append("contract digest does not match the supplied issue body")
    return repository, issue, contract_digest


def _validate_dependency_rows(
    dependencies: object, *, repository: object, errors: list[str]
) -> int:
    """Validate all dependency rows and reject duplicate stable identifiers."""
    if not isinstance(dependencies, list) or not dependencies:
        errors.append("dependencies must be a non-empty array")
        return 0
    identifiers: set[str] = set()
    repository_text = repository if isinstance(repository, str) else ""
    for index, row in enumerate(dependencies):
        identifier = _validate_row(row, index=index, repository=repository_text, errors=errors)
        if identifier is not None:
            if identifier in identifiers:
                errors.append(f"dependencies[{index}].id {identifier!r} is duplicated")
            identifiers.add(identifier)
    return len(dependencies)


def _validate_packet_digest(packet: Mapping[str, Any], *, errors: list[str]) -> object:
    """Validate the packet's canonical self-digest."""
    packet_digest = packet.get("packet_digest")
    if not _is_sha256(packet_digest):
        errors.append("packet_digest must be a lowercase SHA-256 digest")
    else:
        try:
            observed_digest = compute_packet_digest(packet)
        except ValueError as exc:
            errors.append(str(exc))
        else:
            if packet_digest != observed_digest:
                errors.append("packet_digest does not match the canonical packet payload")
    return packet_digest


def validate_packet(
    packet: Mapping[str, Any] | object,
    *,
    expected_repository: str | None = None,
    expected_issue: int | None = None,
    contract_text: str | None = None,
) -> dict[str, Any]:
    """Validate a packet and report every structural or digest error."""
    if not isinstance(packet, Mapping):
        return {"schema": SCHEMA, "ok": False, "errors": ["packet must be an object"]}
    errors: list[str] = []
    unknown = sorted(str(key) for key in packet if key not in PACKET_FIELDS)
    if unknown:
        errors.append("packet has unsupported field(s): " + ", ".join(unknown))
    if packet.get("schema") != SCHEMA:
        errors.append(f"schema must be {SCHEMA!r}")
    repository, issue, contract_digest = _validate_packet_metadata(
        packet,
        errors=errors,
        expected_repository=expected_repository,
        expected_issue=expected_issue,
        contract_text=contract_text,
    )
    dependencies = packet.get("dependencies")
    dependency_count = _validate_dependency_rows(
        dependencies,
        repository=repository,
        errors=errors,
    )
    packet_digest = _validate_packet_digest(packet, errors=errors)
    return {
        "schema": SCHEMA,
        "ok": not errors,
        "errors": errors,
        "repository": repository,
        "issue": issue,
        "contract_digest": contract_digest,
        "packet_digest": packet_digest,
        "dependency_count": dependency_count,
    }


def build_packet(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Build a canonical self-digested packet from a JSON declaration."""
    packet = dict(payload)
    packet["schema"] = SCHEMA
    packet.pop("packet_digest", None)
    packet["packet_digest"] = compute_packet_digest(packet)
    result = validate_packet(packet)
    if not result["ok"]:
        raise ValueError("invalid dependency packet: " + "; ".join(result["errors"]))
    return packet


def _copy_json(value: Any) -> Any:
    """Copy JSON-shaped data without importing a mutable-copy dependency."""
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True))


def _context_group(context: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    """Return one optional observation group, treating absent data as empty."""
    value = context.get(name, {})
    return value if isinstance(value, Mapping) else {}


def _row_key(row: Mapping[str, Any]) -> str:
    """Return the stable key used by kind-specific offline contexts."""
    requirement = row["requirement"]
    kind = row["kind"]
    if kind in {"issue_state", "pull_request_state"}:
        return f"{row['repository']}#{requirement['number']}"
    if kind == "commit_present":
        return f"{row['repository']}@{requirement['sha']}"
    if kind == "path_present":
        return str(requirement["path"])
    if kind == "artifact_digest":
        return str(requirement["path"])
    if kind == "external_input":
        return str(requirement["identifier"])
    if kind == "environment_capability":
        return str(requirement["name"])
    if kind == "human_ruling":
        return f"{row['repository']}#{requirement['issue']}"
    raise ValueError(f"unsupported dependency kind {kind!r}")


def _observation_for_row(
    row: Mapping[str, Any], context: Mapping[str, Any]
) -> Mapping[str, Any] | None:
    """Find an observation by row id first, then by its canonical kind key."""
    observations = _context_group(context, "observations")
    direct = observations.get(row.get("id"))
    if isinstance(direct, Mapping):
        return direct
    groups = {
        "issue_state": "issues",
        "pull_request_state": "pull_requests",
        "commit_present": "commits",
        "path_present": "paths",
        "artifact_digest": "artifacts",
        "external_input": "external_inputs",
        "environment_capability": "environment",
        "human_ruling": "human_rulings",
    }
    group = _context_group(context, groups[row["kind"]])
    observation = group.get(_row_key(row))
    return observation if isinstance(observation, Mapping) else None


def _exact(observed: Mapping[str, Any], key: str, expected: Any) -> bool:
    """Compare an observed scalar literally, with no coercion."""
    return key in observed and observed[key] == expected and type(observed[key]) is type(expected)


def _state_observation(row: Mapping[str, Any], observed: Mapping[str, Any]) -> tuple[str, str]:
    """Evaluate issue or pull-request state and exact revision constraints."""
    if observed.get("available") is False:
        return "unavailable", "the issue or pull-request state could not be observed"
    requirement = row["requirement"]
    expected_state = requirement["state"]
    if not _exact(observed, "state", expected_state):
        return "unsatisfied", f"state must be exactly {expected_state}"
    for requirement_key, observed_key in (
        ("head_sha", "head_sha"),
        ("base_sha", "base_sha"),
        ("head_ref", "head_ref"),
        ("base_ref", "base_ref"),
    ):
        if requirement_key in requirement and not _exact(
            observed, observed_key, requirement[requirement_key]
        ):
            return "conflict", f"{observed_key} must be exactly {requirement[requirement_key]}"
    return "satisfied", "required state and exact revisions match"


def _evaluate_commit(row: Mapping[str, Any], observed: Mapping[str, Any]) -> tuple[str, str]:
    """Evaluate exact commit presence and optional ancestry."""
    requirement = row["requirement"]
    if observed.get("available") is False:
        return "unavailable", "the named commit could not be observed"
    if not _exact(observed, "present", True):
        return "unsatisfied", "the exact commit is not present"
    if "ancestor_of" in requirement:
        if "is_ancestor" not in observed:
            return "unavailable", "commit ancestry was not verified"
        if not _exact(observed, "is_ancestor", True):
            return "conflict", f"commit must be an ancestor of exactly {requirement['ancestor_of']}"
    return "satisfied", "exact commit and ancestry predicate match"


def _evaluate_path(row: Mapping[str, Any], observed: Mapping[str, Any]) -> tuple[str, str]:
    """Evaluate exact local or public path presence and type."""
    if observed.get("available") is False:
        return "unavailable", "the path observation is unavailable"
    if not _exact(observed, "exists", True):
        return "unsatisfied", "the required path is absent"
    expected_type = row["requirement"].get("path_type", "any")
    if expected_type != "any" and not _exact(observed, "type", expected_type):
        return "conflict", f"path type must be exactly {expected_type}"
    return "satisfied", "required path and type match"


def _evaluate_artifact(row: Mapping[str, Any], observed: Mapping[str, Any]) -> tuple[str, str]:
    """Evaluate an artifact schema and exact digest without proving provenance."""
    requirement = row["requirement"]
    if observed.get("available") is False or not observed:
        return "unavailable", "the named artifact was not verified"
    if not _exact(observed, "schema", requirement["schema"]):
        return "conflict", f"artifact schema must be exactly {requirement['schema']}"
    if not _exact(observed, "digest", requirement["digest"]):
        return "conflict", f"artifact digest must be exactly {requirement['digest']}"
    if observed.get("verified") is not True:
        return "unavailable", "the artifact source did not verify the digest"
    return "satisfied", "artifact schema and digest match"


def _evaluate_external(row: Mapping[str, Any], observed: Mapping[str, Any]) -> tuple[str, str]:
    """Evaluate an external-input predicate and optional exact metadata."""
    requirement = row["requirement"]
    if observed.get("available") is not True or observed.get("verified") is not True:
        return "unavailable", "the named external input is not verified"
    if not _exact(observed, "predicate", requirement["predicate"]):
        return "conflict", f"external predicate must be exactly {requirement['predicate']}"
    for key in ("schema", "digest"):
        if key in requirement and not _exact(observed, key, requirement[key]):
            return "conflict", f"external {key} must be exactly {requirement[key]}"
    return "satisfied", "external input predicate and exact metadata match"


def _evaluate_environment(row: Mapping[str, Any], observed: Mapping[str, Any]) -> tuple[str, str]:
    """Evaluate an environment capability predicate."""
    if observed.get("available") is not True:
        return "unavailable", "the environment capability is unavailable"
    predicate = row["requirement"]["predicate"]
    if not _exact(observed, "predicate", predicate):
        return "unsatisfied", f"environment predicate must be exactly {predicate}"
    return "satisfied", "environment capability predicate matches"


def _evaluate_ruling(row: Mapping[str, Any], observed: Mapping[str, Any]) -> tuple[str, str]:
    """Evaluate an exact human-ruling token."""
    requirement = row["requirement"]
    if observed.get("available") is not True:
        return "unavailable", "the named human ruling is not verified"
    if not _exact(observed, "token", requirement["token"]):
        return "conflict", f"human ruling token must be exactly {requirement['token']}"
    return "satisfied", "human ruling token matches exactly"


def _evaluate_state(row: Mapping[str, Any], observed: Mapping[str, Any]) -> tuple[str, str]:
    """Evaluate an issue or pull-request state predicate."""
    return _state_observation(row, observed)


_ROW_EVALUATORS: dict[str, Callable[[Mapping[str, Any], Mapping[str, Any]], tuple[str, str]]] = {
    "issue_state": _evaluate_state,
    "pull_request_state": _evaluate_state,
    "commit_present": _evaluate_commit,
    "path_present": _evaluate_path,
    "artifact_digest": _evaluate_artifact,
    "external_input": _evaluate_external,
    "environment_capability": _evaluate_environment,
    "human_ruling": _evaluate_ruling,
}


def _evaluate_row(
    row: Mapping[str, Any], observed: Mapping[str, Any] | None
) -> tuple[str, str, dict[str, Any]]:
    """Evaluate one validated row and return verdict, reason, and stable observation."""
    if observed is None:
        return "unavailable", "no verified observation was supplied", {}
    current = _copy_json(observed)
    evaluator = _ROW_EVALUATORS.get(row["kind"])
    if evaluator is None:
        return "invalid", f"unsupported dependency kind {row['kind']!r}", current
    verdict, reason = evaluator(row, current)
    return verdict, reason, current


def evaluate_packet(
    packet: Mapping[str, Any] | object,
    context: Mapping[str, Any] | None = None,
    *,
    expected_repository: str | None = None,
    expected_issue: int | None = None,
    contract_text: str | None = None,
) -> dict[str, Any]:
    """Evaluate every packet row against explicit, already-observed inputs.

    The function never contacts GitHub, executes Git, mutates a packet, or
    treats the packet's previously recorded ``observed`` value as current.
    """
    validation = validate_packet(
        packet,
        expected_repository=expected_repository,
        expected_issue=expected_issue,
        contract_text=contract_text,
    )
    if not validation["ok"]:
        return {
            "schema": EVALUATION_SCHEMA,
            "ok": False,
            "verdict": "invalid",
            "errors": validation["errors"],
            "packet_digest": validation.get("packet_digest"),
            "rows": [],
            "mandatory_failures": [],
            "advisory_failures": [],
        }
    assert isinstance(packet, Mapping)
    context_map = context if isinstance(context, Mapping) else {}
    rows: list[dict[str, Any]] = []
    mandatory_failures: list[dict[str, Any]] = []
    advisory_failures: list[dict[str, Any]] = []
    for row in packet["dependencies"]:
        assert isinstance(row, Mapping)
        observed = _observation_for_row(row, context_map)
        verdict, reason, current = _evaluate_row(row, observed)
        result_row = {
            "id": row["id"],
            "repository": row["repository"],
            "kind": row["kind"],
            "mandatory": row["mandatory"],
            "verdict": verdict,
            "reason": reason,
            "unblock_condition": row["unblock_condition"],
            "freshness": row["freshness"],
            "observed": current,
        }
        rows.append(result_row)
        if verdict != "satisfied":
            (mandatory_failures if row["mandatory"] else advisory_failures).append(
                {
                    "id": row["id"],
                    "kind": row["kind"],
                    "verdict": verdict,
                    "reason": reason,
                    "unblock_condition": row["unblock_condition"],
                }
            )
    ok = not mandatory_failures
    return {
        "schema": EVALUATION_SCHEMA,
        "ok": ok,
        "verdict": "satisfied" if ok else "blocked",
        "errors": [],
        "packet_digest": packet["packet_digest"],
        "rows": rows,
        "mandatory_failures": mandatory_failures,
        "advisory_failures": advisory_failures,
    }


def apply_dependency_gate(
    implementability: Mapping[str, Any], evaluation: Mapping[str, Any]
) -> dict[str, Any]:
    """Attach one dependency result to an implementability report.

    This is the small consumer adapter used by the #7611 implementability
    owner.  It deliberately does not re-interpret dependency rows or perform
    a write; a non-satisfied aggregate always removes admission permission.
    """
    result = _copy_json(dict(implementability))
    gate = {
        "schema": EVALUATION_SCHEMA,
        "ok": evaluation.get("ok") is True and evaluation.get("verdict") == "satisfied",
        "verdict": evaluation.get("verdict", "invalid"),
        "packet_digest": evaluation.get("packet_digest"),
        "mandatory_failures": _copy_json(evaluation.get("mandatory_failures", [])),
        "advisory_failures": _copy_json(evaluation.get("advisory_failures", [])),
    }
    result["dependency_gate"] = gate
    if not gate["ok"]:
        result["ready"] = False
        result["write_allowed"] = False
        if result.get("classification") not in {"error", "closed", "already_claimed"}:
            result["classification"] = "needs_dependency"
        reasons = result.setdefault("reasons", [])
        if isinstance(reasons, list):
            for failure in gate["mandatory_failures"]:
                reasons.append(
                    f"dependency {failure['id']}: {failure['reason']}; "
                    f"unblock: {failure['unblock_condition']}"
                )
    return result


# A descriptive alias keeps the adapter discoverable to the #7611 consumer.
merge_dependency_gate = apply_dependency_gate


def _default_gh_runner(command: list[str]) -> subprocess.CompletedProcess[str]:
    """Run one fixed ``gh api`` read with a bounded timeout."""
    return subprocess.run(
        command,
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )


def _default_git_runner(
    command: list[str], *, cwd: Path | None = None
) -> subprocess.CompletedProcess[str]:
    """Run one fixed read-only Git query with a bounded timeout."""
    return subprocess.run(
        command,
        capture_output=True,
        check=False,
        cwd=cwd,
        text=True,
        timeout=30,
    )


def _api_json(
    gh_runner: GhRunner,
    endpoint: str,
) -> tuple[Mapping[str, Any] | list[Any] | None, str | None]:
    """Read one JSON REST endpoint through ``gh api`` without writes."""
    try:
        result = gh_runner(["gh", "api", endpoint])
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f"GitHub REST read failed: {exc}"
    if result.returncode != 0:
        detail = result.stderr.strip() or f"gh api exited with {result.returncode}"
        return None, detail
    try:
        payload = json.loads(result.stdout)
    except (TypeError, json.JSONDecodeError) as exc:
        return None, f"GitHub REST returned invalid JSON: {exc}"
    if not isinstance(payload, (Mapping, list)):
        return None, "GitHub REST returned a non-object/non-array payload"
    return payload, None


def _git_read(
    git_runner: GitRunner,
    command: list[str],
) -> tuple[subprocess.CompletedProcess[str] | None, str | None]:
    """Run one injected or default read-only Git command."""
    try:
        return git_runner(command), None
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f"local Git read failed: {exc}"


def _normalise_issue(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize the small issue fields used by an issue-state predicate."""
    state = payload.get("state")
    number = payload.get("number")
    if not isinstance(state, str) or not _is_int(number):
        raise ValueError("GitHub issue response lacks state or number")
    return {
        "available": True,
        "number": number,
        "state": state.upper(),
        "body_sha256": sha256_text(payload.get("body") or "")
        if isinstance(payload.get("body"), str)
        else None,
        "source": "github_rest",
    }


def _normalise_pull_request(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize PR state, refs, and exact head/base revisions."""
    state = payload.get("state")
    number = payload.get("number")
    head = payload.get("head")
    base = payload.get("base")
    if not isinstance(state, str) or not _is_int(number):
        raise ValueError("GitHub pull-request response lacks state or number")
    if not isinstance(head, Mapping) or not isinstance(base, Mapping):
        raise ValueError("GitHub pull-request response lacks head or base")
    merged_at = payload.get("merged_at")
    normalised_state = "MERGED" if merged_at is not None else state.upper()
    return {
        "available": True,
        "number": number,
        "state": normalised_state,
        "head_sha": head.get("sha"),
        "head_ref": head.get("ref"),
        "base_sha": base.get("sha"),
        "base_ref": base.get("ref"),
        "source": "github_rest",
    }


def _unavailable(message: str, *, source: str = "unavailable") -> dict[str, Any]:
    """Return a stable unavailable observation without hiding the cause."""
    return {"available": False, "reason": message, "source": source}


def _is_github_not_found(error: str | None) -> bool:
    """Return whether a GitHub REST error is a definitive 404 response."""
    text = (error or "").lower()
    return "http 404" in text or "not found" in text


def _rest_mapping(
    gh_runner: GhRunner, endpoint: str, *, label: str
) -> tuple[Mapping[str, Any] | None, str | None]:
    """Fetch one REST object and normalize transport/type errors."""
    payload, error = _api_json(gh_runner, endpoint)
    if error:
        return None, error
    if not isinstance(payload, Mapping):
        return None, f"{label} response was not an object"
    return payload, None


def _resolve_github_issue(row: Mapping[str, Any], gh_runner: GhRunner) -> dict[str, Any]:
    """Resolve one public issue state."""
    requirement = row["requirement"]
    payload, error = _rest_mapping(
        gh_runner,
        f"repos/{row['repository']}/issues/{requirement['number']}",
        label="issue",
    )
    if error or payload is None:
        return _unavailable(error or "issue response was unavailable", source="github_rest")
    try:
        return _normalise_issue(payload)
    except ValueError as exc:
        return _unavailable(str(exc), source="github_rest")


def _resolve_github_pr(row: Mapping[str, Any], gh_runner: GhRunner) -> dict[str, Any]:
    """Resolve one public pull-request state and exact refs."""
    requirement = row["requirement"]
    payload, error = _rest_mapping(
        gh_runner,
        f"repos/{row['repository']}/pulls/{requirement['number']}",
        label="pull-request",
    )
    if error or payload is None:
        return _unavailable(error or "pull-request response was unavailable", source="github_rest")
    try:
        return _normalise_pull_request(payload)
    except ValueError as exc:
        return _unavailable(str(exc), source="github_rest")


def _resolve_github_commit(row: Mapping[str, Any], gh_runner: GhRunner) -> dict[str, Any]:
    """Resolve exact public commit presence."""
    requirement = row["requirement"]
    payload, error = _rest_mapping(
        gh_runner,
        f"repos/{row['repository']}/commits/{requirement['sha']}",
        label="commit",
    )
    if error or payload is None:
        if _is_github_not_found(error):
            return {
                "available": True,
                "present": False,
                "sha": requirement["sha"],
                "source": "github_rest",
            }
        return _unavailable(error or "commit response was unavailable", source="github_rest")
    observed_sha = payload.get("sha")
    return {
        "available": True,
        "present": _exact({"sha": observed_sha}, "sha", requirement["sha"]),
        "sha": observed_sha,
        "source": "github_rest",
    }


def _resolve_github_path(row: Mapping[str, Any], gh_runner: GhRunner) -> dict[str, Any]:
    """Resolve a public repository path through the contents endpoint."""
    requirement = row["requirement"]
    encoded_path = str(requirement["path"]).replace(" ", "%20")
    endpoint = f"repos/{row['repository']}/contents/{encoded_path}"
    ref = requirement.get("ref")
    if isinstance(ref, str) and ref:
        endpoint += f"?ref={ref}"
    payload, error = _api_json(gh_runner, endpoint)
    if error or payload is None:
        if _is_github_not_found(error):
            return {
                "available": True,
                "exists": False,
                "source": "github_rest",
            }
        return _unavailable(error or "contents response was unavailable", source="github_rest")
    if isinstance(payload, list):
        return {
            "available": True,
            "exists": True,
            "type": "directory",
            "source": "github_rest",
        }
    if not isinstance(payload, Mapping):
        return _unavailable("contents response was not an object or array", source="github_rest")
    content_type = payload.get("type")
    return {
        "available": True,
        "exists": True,
        "type": "directory"
        if content_type == "dir"
        else "file"
        if content_type == "file"
        else "other",
        "sha": payload.get("sha"),
        "source": "github_rest",
    }


def _resolve_github_ruling(row: Mapping[str, Any], gh_runner: GhRunner) -> dict[str, Any]:
    """Resolve an exact ruling marker from public issue comments."""
    requirement = row["requirement"]
    endpoint = f"repos/{row['repository']}/issues/{requirement['issue']}/comments?per_page=100"
    payload, error = _api_json(gh_runner, endpoint)
    if error or not isinstance(payload, list):
        return _unavailable(error or "comments response was not an array", source="github_rest")
    token = requirement["token"]
    marker = f"human-ruling: {token}"
    found = any(
        isinstance(comment, Mapping)
        and isinstance(comment.get("body"), str)
        and marker in comment["body"]
        for comment in payload
    )
    return {"available": True, "token": token if found else None, "source": "github_rest"}


def _resolve_github_row(
    row: Mapping[str, Any],
    *,
    gh_runner: GhRunner,
) -> dict[str, Any]:
    """Resolve public issue, PR, commit, path, or ruling state via REST."""
    resolvers = {
        "issue_state": _resolve_github_issue,
        "pull_request_state": _resolve_github_pr,
        "commit_present": _resolve_github_commit,
        "path_present": _resolve_github_path,
        "human_ruling": _resolve_github_ruling,
    }
    resolver = resolvers.get(row["kind"])
    if resolver is not None:
        return resolver(row, gh_runner)
    if row["kind"] in {"artifact_digest", "external_input", "environment_capability"}:
        return _unavailable(
            "no named verifier was supplied; this resolver never downloads or infers inputs",
            source="unavailable",
        )
    return _unavailable(f"unsupported dependency kind {row['kind']!r}")


def _resolve_local_commit(
    row: Mapping[str, Any],
    *,
    git_runner: GitRunner,
) -> dict[str, Any]:
    """Resolve commit presence and optional ancestry with read-only Git."""
    requirement = row["requirement"]
    sha = requirement["sha"]
    present_result, present_error = _git_read(
        git_runner,
        ["git", "cat-file", "-e", f"{sha}^{{commit}}"],
    )
    if present_error or present_result is None:
        return _unavailable(
            present_error or "Git presence query returned no result", source="local_git"
        )
    present = present_result.returncode == 0
    result: dict[str, Any] = {
        "available": True,
        "present": present,
        "sha": sha,
        "source": "local_git",
    }
    if "ancestor_of" in requirement and present:
        target = requirement["ancestor_of"]
        ancestry_result, ancestry_error = _git_read(
            git_runner,
            ["git", "merge-base", "--is-ancestor", sha, target],
        )
        if ancestry_error or ancestry_result is None:
            result["available"] = False
            result["reason"] = ancestry_error or "Git ancestry query returned no result"
        else:
            result["is_ancestor"] = ancestry_result.returncode == 0
            result["ancestor_of"] = target
    elif "ancestor_of" in requirement:
        result["is_ancestor"] = False
        result["ancestor_of"] = requirement["ancestor_of"]
    return result


def _local_path_observation(repo_root: Path, row: Mapping[str, Any]) -> dict[str, Any]:
    """Inspect one repository-relative path without following it outside root."""
    requirement = row["requirement"]
    relative = PurePosixPath(requirement["path"])
    root = repo_root.resolve()
    candidate = (root.joinpath(*relative.parts)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return _unavailable("path resolves outside the repository root", source="local_path")
    try:
        exists = candidate.exists()
    except OSError as exc:
        return _unavailable(f"path read failed: {exc}", source="local_path")
    if not exists:
        return {"available": True, "exists": False, "source": "local_path"}
    try:
        path_type = (
            "directory" if candidate.is_dir() else "file" if candidate.is_file() else "other"
        )
    except OSError as exc:
        return _unavailable(f"path type read failed: {exc}", source="local_path")
    return {
        "available": True,
        "exists": True,
        "type": path_type,
        "source": "local_path",
    }


def _resolve_row(
    row: Mapping[str, Any],
    *,
    repo_root: Path | None,
    gh_runner: GhRunner,
    git_runner: GitRunner,
) -> dict[str, Any]:
    """Resolve one row using local read-only checks where the kind supports them."""
    kind = row["kind"]
    source_kind = row["source"].get("kind")
    if (
        kind == "commit_present"
        and repo_root is not None
        and source_kind
        not in {
            "github",
            "github_rest",
        }
    ):
        return _resolve_local_commit(row, git_runner=git_runner)
    if (
        kind == "path_present"
        and repo_root is not None
        and source_kind
        not in {
            "github",
            "github_rest",
        }
    ):
        return _local_path_observation(repo_root, row)
    return _resolve_github_row(row, gh_runner=gh_runner)


def resolve_packet(
    packet: Mapping[str, Any] | object,
    *,
    context: Mapping[str, Any] | None = None,
    repo_root: Path | str | None = None,
    gh_runner: GhRunner | None = None,
    git_runner: GitRunner | None = None,
    expected_repository: str | None = None,
    expected_issue: int | None = None,
    contract_text: str | None = None,
) -> dict[str, Any]:
    """Resolve missing observations and return the deterministic evaluation.

    Supplied observations always win.  Missing public predicates use ``gh
    api``; local path and Git predicates use the supplied repository root.
    Artifact, external-input, environment, and human-authority decisions are
    never inferred or downloaded.
    """
    validation = validate_packet(
        packet,
        expected_repository=expected_repository,
        expected_issue=expected_issue,
        contract_text=contract_text,
    )
    if not validation["ok"]:
        return evaluate_packet(
            packet,
            context,
            expected_repository=expected_repository,
            expected_issue=expected_issue,
            contract_text=contract_text,
        )
    assert isinstance(packet, Mapping)
    mutable_context = _copy_json(context or {})
    if not isinstance(mutable_context, dict):
        mutable_context = {}
    observations = mutable_context.setdefault("observations", {})
    if not isinstance(observations, dict):
        observations = {}
        mutable_context["observations"] = observations
    resolved_root = Path(repo_root).resolve() if repo_root is not None else None
    read_gh = gh_runner or _default_gh_runner
    if git_runner is not None:
        read_git = git_runner
    elif resolved_root is None:
        read_git = _default_git_runner
    else:

        def read_git(command: list[str]) -> subprocess.CompletedProcess[str]:
            return _default_git_runner(command, cwd=resolved_root)

    for row in packet["dependencies"]:
        assert isinstance(row, Mapping)
        if _observation_for_row(row, mutable_context) is not None:
            continue
        observations[row["id"]] = _resolve_row(
            row,
            repo_root=resolved_root,
            gh_runner=read_gh,
            git_runner=read_git,
        )
    evaluation = evaluate_packet(
        packet,
        mutable_context,
        expected_repository=expected_repository,
        expected_issue=expected_issue,
        contract_text=contract_text,
    )
    evaluation["resolved_context"] = mutable_context
    return evaluation


def _read_json_file(path: str) -> Any:
    """Read one UTF-8 JSON file for the CLI."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json_file(path: str, payload: Mapping[str, Any]) -> None:
    """Write one deterministic UTF-8 JSON file, or stdout for ``-``."""
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path == "-":
        sys.stdout.write(rendered)
    else:
        Path(path).write_text(rendered, encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    """Build the dependency-packet command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate", help="validate one packet without resolving it")
    validate.add_argument("--packet", required=True, help="Packet JSON path.")
    validate.add_argument(
        "--contract-file", help="Optional exact issue-body text for digest checking."
    )
    validate.add_argument("--repo", default=DEFAULT_REPO, help="Expected OWNER/REPO value.")
    validate.add_argument("--issue", type=int, help="Expected issue number.")
    build = subparsers.add_parser("build", help="canonicalize and self-digest a packet declaration")
    build.add_argument("--input", required=True, help="Unsigned packet declaration JSON path.")
    build.add_argument("--output", required=True, help="Output JSON path, or - for stdout.")
    for name, help_text in (
        ("verify", "evaluate a packet against an offline observation context"),
        ("check", "resolve missing public/local predicates and evaluate a packet"),
    ):
        command = subparsers.add_parser(name, help=help_text)
        command.add_argument("--packet", required=True, help="Packet JSON path.")
        command.add_argument("--context", help="Optional observation-context JSON path.")
        command.add_argument("--repo-root", help="Repository root for local path/Git checks.")
        command.add_argument("--repo", default=DEFAULT_REPO, help="Expected OWNER/REPO value.")
        command.add_argument("--issue", type=int, help="Expected issue number.")
        command.add_argument(
            "--contract-file", help="Optional exact issue-body text for digest checking."
        )
    return parser


def _cli_report(args: argparse.Namespace) -> dict[str, Any]:
    """Execute one parsed CLI command and return its JSON report."""
    if args.command == "build":
        packet = build_packet(_read_json_file(args.input))
        _write_json_file(args.output, packet)
        return packet
    packet = _read_json_file(args.packet)
    contract_text = (
        Path(args.contract_file).read_text(encoding="utf-8") if args.contract_file else None
    )
    if args.command == "validate":
        return validate_packet(
            packet,
            expected_repository=args.repo,
            expected_issue=args.issue,
            contract_text=contract_text,
        )
    context = _read_json_file(args.context) if args.context else {}
    if not isinstance(context, Mapping):
        raise ValueError("context JSON must be an object")
    if args.command == "verify":
        return evaluate_packet(
            packet,
            context,
            expected_repository=args.repo,
            expected_issue=args.issue,
            contract_text=contract_text,
        )
    return resolve_packet(
        packet,
        context=context,
        repo_root=args.repo_root,
        expected_repository=args.repo,
        expected_issue=args.issue,
        contract_text=contract_text,
    )


def main(argv: list[str] | None = None) -> int:
    """Run the side-effect-free validation/resolution CLI."""
    args = _build_parser().parse_args(argv)
    try:
        report = _cli_report(args)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        report = {
            "schema": EVALUATION_SCHEMA if args.command in {"verify", "check"} else SCHEMA,
            "ok": False,
            "verdict": "invalid",
            "errors": [str(exc)],
        }
    if args.command != "build" or report.get("ok") is not True:
        sys.stdout.write(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    if args.command == "build":
        return 0 if report.get("ok", True) else 1
    if report.get("ok") is True:
        return 0
    return 2 if report.get("verdict") == "blocked" else 1


if __name__ == "__main__":
    raise SystemExit(main())
