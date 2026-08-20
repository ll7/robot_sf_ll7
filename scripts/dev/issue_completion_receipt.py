#!/usr/bin/env python3
"""Build and verify exact-head issue completion receipts.

The receipt is a generic delivery-integrity contract.  It binds an issue
contract, one exact base/head pair, changed paths, validation inputs, durable
artifacts, acceptance-criterion dispositions, and independent review.  It does
not decide scientific, benchmark, release, licensing, or domain validity.

``validate_receipt`` is the offline fixture path.  ``verify_receipt_against_git``
adds exact Git and optional GitHub pull-request/issue checks.  Both paths fail
closed and return structured evidence rather than treating a successful command
or a producer-authored receipt as acceptance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA = "issue_completion_receipt.v1"
DEFAULT_REPO = "ll7/robot_sf_ll7"
SHA_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REQUIRED_FIELDS = (
    "schema",
    "repository",
    "issue",
    "contract",
    "delivery",
    "covering_pr",
    "diff",
    "validation",
    "validation_inputs",
    "artifacts",
    "acceptance_criteria",
    "residuals",
    "producer",
    "independent_verifier",
    "drift_policy",
    "receipt_digest",
)
VALIDATION_STATUSES = frozenset({"passed", "failed", "skipped", "unavailable", "not_applicable"})
CRITERION_DISPOSITIONS = frozenset(
    {"met", "not_met", "blocked", "skipped", "unavailable", "not_applicable", "deferred"}
)
VERIFIER_STATUSES = frozenset({"verified", "pending", "unavailable", "not_applicable"})
REQUIRED_DRIFT_KEYS = frozenset({"head", "contract", "artifacts", "validation_inputs", "review"})

GitRunner = Callable[[list[str]], subprocess.CompletedProcess[str]]
GhRunner = Callable[[list[str]], subprocess.CompletedProcess[str]]


def _is_int(value: object) -> bool:
    """Return whether ``value`` is an integer but not a boolean."""
    return isinstance(value, int) and not isinstance(value, bool)


def _is_sha(value: object) -> bool:
    """Accept full Git SHA-1 and SHA-256 spellings, never abbreviations."""
    return isinstance(value, str) and SHA_RE.fullmatch(value) is not None


def _is_sha256(value: object) -> bool:
    """Return whether ``value`` is a lowercase SHA-256 digest."""
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def sha256_bytes(value: bytes) -> str:
    """Hash bytes with the receipt's stable digest algorithm."""
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    """Hash UTF-8 text without whitespace normalization."""
    return sha256_bytes(value.encode("utf-8"))


def _canonical_payload(receipt: Mapping[str, Any]) -> bytes:
    """Serialize a receipt deterministically, excluding its self-digest."""
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_digest"}
    try:
        rendered = json.dumps(
            unsigned,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"receipt is not canonical JSON: {exc}") from exc
    return rendered.encode("utf-8")


def compute_receipt_digest(receipt: Mapping[str, Any]) -> str:
    """Compute the self-digest for a receipt without mutating the input."""
    return sha256_bytes(_canonical_payload(receipt))


def _path_error(value: object, *, field: str) -> str | None:
    """Validate one repository-relative artifact path or external URI."""
    if not isinstance(value, str) or not value.strip():
        return f"{field} must be a non-empty string"
    text = value.strip()
    if "://" in text:
        return None
    if "\\" in text:
        return f"{field} must use repository-relative POSIX paths"
    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts:
        return f"{field} must not escape the repository root"
    if text in {"", "."}:
        return f"{field} must name a file or durable artifact"
    return None


def _mapping(value: object, *, field: str, errors: list[str]) -> Mapping[str, Any] | None:
    """Require a mapping and append a stable error when it is absent."""
    if not isinstance(value, Mapping):
        errors.append(f"{field} must be an object")
        return None
    return value


def _string(value: object, *, field: str, errors: list[str], non_empty: bool = True) -> str | None:
    """Require a string field."""
    if not isinstance(value, str) or (non_empty and not value.strip()):
        errors.append(f"{field} must be a non-empty string")
        return None
    return value


def _validate_digest_records(
    records: object,
    *,
    field: str,
    head_sha: str | None,
    errors: list[str],
) -> None:
    """Validate artifact/input path, schema, digest, and exact-head bindings."""
    if not isinstance(records, list):
        errors.append(f"{field} must be an array")
        return
    for index, record in enumerate(records):
        prefix = f"{field}[{index}]"
        row = _mapping(record, field=prefix, errors=errors)
        if row is None:
            continue
        path = row.get("path")
        if error := _path_error(path, field=f"{prefix}.path"):
            errors.append(error)
        _string(row.get("schema"), field=f"{prefix}.schema", errors=errors)
        if not _is_sha256(row.get("digest")):
            errors.append(f"{prefix}.digest must be a lowercase SHA-256 digest")
        captured = row.get("captured_head_sha")
        if not _is_sha(captured):
            errors.append(f"{prefix}.captured_head_sha must be a full Git SHA")
        elif head_sha is not None and captured != head_sha:
            errors.append(
                f"{prefix}.captured_head_sha {captured} does not match delivered head {head_sha}"
            )


def _validate_validation(  # noqa: C901 - one record's fail-closed status matrix
    validation: object,
    *,
    head_sha: str | None,
    errors: list[str],
) -> None:
    """Validate command status records without collapsing unavailable states."""
    if not isinstance(validation, list) or not validation:
        errors.append("validation must be a non-empty array")
        return
    for index, command in enumerate(validation):
        prefix = f"validation[{index}]"
        row = _mapping(command, field=prefix, errors=errors)
        if row is None:
            continue
        _string(row.get("command"), field=f"{prefix}.command", errors=errors)
        _string(row.get("summary"), field=f"{prefix}.summary", errors=errors)
        status = row.get("status")
        if status not in VALIDATION_STATUSES:
            errors.append(f"{prefix}.status must be one of {sorted(VALIDATION_STATUSES)}")
        exit_code = row.get("exit_code")
        if exit_code is not None and not _is_int(exit_code):
            errors.append(f"{prefix}.exit_code must be an integer or null")
        if status == "passed" and exit_code != 0:
            errors.append(f"{prefix}.passed requires exit_code 0")
        if status == "failed" and (not _is_int(exit_code) or exit_code == 0):
            errors.append(f"{prefix}.failed requires a non-zero exit_code")
        if status in {"skipped", "unavailable", "not_applicable"} and exit_code is not None:
            errors.append(f"{prefix}.{status} must use a null exit_code")
        command_head = row.get("head_sha")
        if not _is_sha(command_head):
            errors.append(f"{prefix}.head_sha must be a full Git SHA")
        elif head_sha is not None and command_head != head_sha:
            errors.append(
                f"{prefix}.head_sha {command_head} does not match delivered head {head_sha}"
            )


def _validate_criteria(criteria: object, *, errors: list[str]) -> None:
    """Require exactly one explicit disposition for every criterion identifier."""
    if not isinstance(criteria, list) or not criteria:
        errors.append("acceptance_criteria must be a non-empty array")
        return
    seen: set[str] = set()
    for index, criterion in enumerate(criteria):
        prefix = f"acceptance_criteria[{index}]"
        row = _mapping(criterion, field=prefix, errors=errors)
        if row is None:
            continue
        identifier = row.get("id")
        if not isinstance(identifier, str) or not identifier.strip():
            errors.append(f"{prefix}.id must be a non-empty string")
        elif identifier in seen:
            errors.append(f"{prefix}.id {identifier!r} has more than one disposition")
        else:
            seen.add(identifier)
        if row.get("disposition") not in CRITERION_DISPOSITIONS:
            errors.append(f"{prefix}.disposition must be one of {sorted(CRITERION_DISPOSITIONS)}")
        evidence = row.get("evidence")
        if (
            not isinstance(evidence, list)
            or not evidence
            or any(not isinstance(item, str) or not item.strip() for item in evidence)
        ):
            errors.append(f"{prefix}.evidence must contain at least one non-empty string")


def _validate_string_lists(value: object, *, field: str, errors: list[str]) -> None:
    """Validate a required list of plain strings."""
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        errors.append(f"{field} must be an array of strings")


def validate_receipt(  # noqa: C901, PLR0912, PLR0913, PLR0915 - schema gate
    receipt: Mapping[str, Any] | object,
    *,
    expected_repository: str | None = None,
    expected_issue: int | None = None,
    expected_base_sha: str | None = None,
    expected_head_sha: str | None = None,
    expected_branch: str | None = None,
    issue_contract: str | None = None,
    artifact_root: str | Path | None = None,
    require_independent_verifier: bool = False,
) -> dict[str, Any]:
    """Validate a receipt offline and report all fail-closed errors.

    ``artifact_root`` activates local digest checks for declared artifact and
    validation-input paths.  Without it, the records are structurally checked
    and remain ready for a later Git/PR-backed verification pass.
    """
    if not isinstance(receipt, Mapping):
        return {"schema": SCHEMA, "ok": False, "errors": ["receipt must be an object"]}

    errors: list[str] = []
    missing = [field for field in REQUIRED_FIELDS if field not in receipt]
    errors.extend(f"missing required field: {field}" for field in missing)
    if receipt.get("schema") != SCHEMA:
        errors.append(f"schema must be {SCHEMA!r}")

    repository = receipt.get("repository")
    if not isinstance(repository, str) or not re.fullmatch(r"[^/\s]+/[^/\s]+", repository):
        errors.append("repository must be an OWNER/REPO string")
    elif expected_repository is not None and repository != expected_repository:
        errors.append(f"repository {repository} does not match expected {expected_repository}")

    issue = receipt.get("issue")
    if not _is_int(issue) or issue <= 0:
        errors.append("issue must be a positive integer")
    elif expected_issue is not None and issue != expected_issue:
        errors.append(f"issue {issue} does not match expected issue {expected_issue}")

    contract = _mapping(receipt.get("contract"), field="contract", errors=errors)
    contract_digest: str | None = None
    if contract is not None:
        contract_digest = (
            contract.get("digest") if isinstance(contract.get("digest"), str) else None
        )
        if not _is_sha256(contract_digest):
            errors.append("contract.digest must be a lowercase SHA-256 digest")
        _string(contract.get("source"), field="contract.source", errors=errors)
        if issue_contract is not None and _is_sha256(contract_digest):
            observed_contract_digest = sha256_text(issue_contract)
            if observed_contract_digest != contract_digest:
                errors.append(
                    "contract digest does not match the supplied issue body or contract text"
                )

    delivery = _mapping(receipt.get("delivery"), field="delivery", errors=errors)
    base_sha: str | None = None
    head_sha: str | None = None
    if delivery is not None:
        _string(delivery.get("base_ref"), field="delivery.base_ref", errors=errors)
        branch = _string(delivery.get("branch"), field="delivery.branch", errors=errors)
        base_sha = delivery.get("base_sha") if isinstance(delivery.get("base_sha"), str) else None
        head_sha = delivery.get("head_sha") if isinstance(delivery.get("head_sha"), str) else None
        if not _is_sha(base_sha):
            errors.append("delivery.base_sha must be a full Git SHA")
        if not _is_sha(head_sha):
            errors.append("delivery.head_sha must be a full Git SHA")
        if expected_base_sha is not None and base_sha != expected_base_sha:
            errors.append(
                f"delivery.base_sha {base_sha} does not match expected base {expected_base_sha}"
            )
        if expected_head_sha is not None and head_sha != expected_head_sha:
            errors.append(
                f"delivery.head_sha {head_sha} does not match expected head {expected_head_sha}"
            )
        if expected_branch is not None and branch != expected_branch:
            errors.append(
                f"delivery.branch {branch} does not match expected branch {expected_branch}"
            )

    covering_pr = receipt.get("covering_pr")
    if covering_pr is not None:
        pr = _mapping(covering_pr, field="covering_pr", errors=errors)
        if pr is not None:
            if not _is_int(pr.get("number")) or int(pr["number"]) <= 0:
                errors.append("covering_pr.number must be a positive integer")
            if pr.get("state") not in {"OPEN", "CLOSED", "MERGED"}:
                errors.append("covering_pr.state must be OPEN, CLOSED, or MERGED")
            for field, expected in (("head_sha", head_sha), ("base_sha", base_sha)):
                if field in pr:
                    if not _is_sha(pr.get(field)):
                        errors.append(f"covering_pr.{field} must be a full Git SHA")
                    elif expected is not None and pr.get(field) != expected:
                        errors.append(f"covering_pr.{field} must match delivery.{field}")
            if "head_ref" in pr:
                _string(pr.get("head_ref"), field="covering_pr.head_ref", errors=errors)

    diff = _mapping(receipt.get("diff"), field="diff", errors=errors)
    if diff is not None:
        changed_paths = diff.get("changed_paths")
        if not isinstance(changed_paths, list) or any(
            _path_error(path, field="diff.changed_paths") for path in changed_paths
        ):
            errors.append("diff.changed_paths must be a list of repository-relative paths")
        elif len(set(changed_paths)) != len(changed_paths):
            errors.append("diff.changed_paths must not contain duplicates")
        stat = _mapping(diff.get("stat"), field="diff.stat", errors=errors)
        if stat is not None:
            for field in ("files", "additions", "deletions"):
                if not _is_int(stat.get(field)) or int(stat[field]) < 0:
                    errors.append(f"diff.stat.{field} must be a non-negative integer")

    _validate_validation(receipt.get("validation"), head_sha=head_sha, errors=errors)
    _validate_digest_records(
        receipt.get("validation_inputs"),
        field="validation_inputs",
        head_sha=head_sha,
        errors=errors,
    )
    _validate_digest_records(
        receipt.get("artifacts"),
        field="artifacts",
        head_sha=head_sha,
        errors=errors,
    )
    _validate_criteria(receipt.get("acceptance_criteria"), errors=errors)

    residuals = _mapping(receipt.get("residuals"), field="residuals", errors=errors)
    if residuals is not None:
        for field in ("risks", "deferred", "forbidden_claims"):
            _validate_string_lists(residuals.get(field), field=f"residuals.{field}", errors=errors)

    producer = _mapping(receipt.get("producer"), field="producer", errors=errors)
    if producer is not None:
        _string(producer.get("identity"), field="producer.identity", errors=errors)
        if "head_sha" in producer and producer.get("head_sha") != head_sha:
            errors.append("producer.head_sha must match delivery.head_sha")

    verifier = _mapping(
        receipt.get("independent_verifier"), field="independent_verifier", errors=errors
    )
    if verifier is not None:
        _string(verifier.get("identity"), field="independent_verifier.identity", errors=errors)
        status = verifier.get("status")
        if status not in VERIFIER_STATUSES:
            errors.append(f"independent_verifier.status must be one of {sorted(VERIFIER_STATUSES)}")
        verifier_head = verifier.get("head_sha")
        if not _is_sha(verifier_head):
            errors.append("independent_verifier.head_sha must be a full Git SHA")
        elif head_sha is not None and verifier_head != head_sha:
            errors.append("independent_verifier.head_sha must match delivery.head_sha")
        if require_independent_verifier and status != "verified":
            errors.append("independent verifier is not verified for close or promotion")

    domain_review = receipt.get("domain_review")
    if domain_review is not None:
        domain = _mapping(domain_review, field="domain_review", errors=errors)
        if domain is not None:
            required = domain.get("required")
            if not isinstance(required, bool):
                errors.append("domain_review.required must be boolean")
            if required:
                _string(domain.get("identity"), field="domain_review.identity", errors=errors)
                if domain.get("status") not in VERIFIER_STATUSES:
                    errors.append(
                        f"domain_review.status must be one of {sorted(VERIFIER_STATUSES)}"
                    )
                if require_independent_verifier and domain.get("status") != "verified":
                    errors.append("required domain review is not verified for close or promotion")

    drift_policy = _mapping(receipt.get("drift_policy"), field="drift_policy", errors=errors)
    if drift_policy is not None:
        invalidate_on = drift_policy.get("invalidate_on")
        if not isinstance(invalidate_on, list) or not REQUIRED_DRIFT_KEYS.issubset(
            {item for item in invalidate_on if isinstance(item, str)}
        ):
            errors.append(
                "drift_policy.invalidate_on must include head, contract, artifacts, "
                "validation_inputs, and review"
            )
        if drift_policy.get("post_review") != "invalidate_on_change":
            errors.append("drift_policy.post_review must be invalidate_on_change")

    recorded_digest = receipt.get("receipt_digest")
    if not _is_sha256(recorded_digest):
        errors.append("receipt_digest must be a lowercase SHA-256 digest")
    else:
        try:
            observed_digest = compute_receipt_digest(receipt)
        except ValueError as exc:
            errors.append(str(exc))
        else:
            if recorded_digest != observed_digest:
                errors.append("receipt_digest does not match the canonical receipt payload")

    if artifact_root is not None:
        root = Path(artifact_root).resolve(strict=False)
        for field in ("validation_inputs", "artifacts"):
            records = receipt.get(field)
            if not isinstance(records, list):
                continue
            for index, record in enumerate(records):
                if not isinstance(record, Mapping) or not isinstance(record.get("path"), str):
                    continue
                path = record["path"]
                if "://" in path:
                    continue
                candidate = (root / path).resolve(strict=False)
                if not candidate.is_relative_to(root):
                    errors.append(f"{field}[{index}].path escapes artifact_root")
                    continue
                if not candidate.is_file():
                    errors.append(f"{field}[{index}] declared artifact is unavailable: {path}")
                    continue
                observed = sha256_bytes(candidate.read_bytes())
                if observed != record.get("digest"):
                    errors.append(f"{field}[{index}] digest drift detected for {path}")

    return {
        "schema": SCHEMA,
        "ok": not errors,
        "errors": errors,
        "receipt_digest": receipt.get("receipt_digest"),
        "repository": receipt.get("repository"),
        "issue": receipt.get("issue"),
        "base_sha": base_sha,
        "head_sha": head_sha,
        "independent_verifier_status": (
            verifier.get("status") if isinstance(verifier, Mapping) else None
        ),
    }


def build_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Build a self-digested receipt from an offline declaration."""
    result = dict(payload)
    result["schema"] = SCHEMA
    result.pop("receipt_digest", None)
    result["receipt_digest"] = compute_receipt_digest(result)
    validation = validate_receipt(result)
    if not validation["ok"]:
        raise ValueError("invalid receipt declaration: " + "; ".join(validation["errors"]))
    return result


def admit_completion_receipt(  # noqa: C901 - close/promotion admission is fail-closed
    entry: Mapping[str, Any] | object,
    *,
    expected_repository: str | None = None,
    expected_issue: int | None = None,
    issue_contract: str | None = None,
) -> dict[str, Any]:
    """Decide whether a verified receipt may authorize close or promotion.

    ``entry`` is the receipt plus the JSON result emitted by
    ``verify_receipt_against_git`` under ``verification``.  Requiring that
    result keeps the issue-audit consumer from treating a producer-authored
    offline fixture as exact-head proof.
    """
    if not isinstance(entry, Mapping):
        return {
            "eligible": False,
            "reason": "completion receipt entry must be an object",
            "errors": ["completion receipt entry must be an object"],
        }
    receipt_value = entry.get("receipt", entry)
    receipt = receipt_value if isinstance(receipt_value, Mapping) else None
    verification = entry.get("verification")
    errors: list[str] = []
    if receipt is None:
        errors.append("completion receipt entry must contain a receipt object")
        return {"eligible": False, "reason": errors[0], "errors": errors}
    basic = validate_receipt(
        receipt,
        expected_repository=expected_repository,
        expected_issue=expected_issue,
        issue_contract=issue_contract,
        require_independent_verifier=True,
    )
    errors.extend(basic["errors"])
    if not isinstance(verification, Mapping):
        errors.append("exact-head Git verification result is required")
    elif verification.get("ok") is not True:
        errors.append("exact-head Git verification did not succeed")
    else:
        delivery = receipt.get("delivery") if isinstance(receipt.get("delivery"), Mapping) else {}
        for field in ("receipt_digest", "base_sha", "head_sha", "branch"):
            expected = (
                receipt.get("receipt_digest") if field == "receipt_digest" else delivery.get(field)
            )
            if verification.get(field) != expected:
                errors.append(f"verification.{field} does not match the receipt")
        git_details = verification.get("git")
        if not isinstance(git_details, Mapping) or not isinstance(git_details.get("diff"), Mapping):
            errors.append("exact-head Git diff evidence is missing from verification")
    validation = receipt.get("validation")
    if isinstance(validation, list):
        disallowed = [
            str(row.get("status"))
            for row in validation
            if isinstance(row, Mapping) and row.get("status") != "passed"
        ]
        if disallowed:
            errors.append(
                "completion receipt contains non-passing validation status: "
                + ", ".join(disallowed)
            )
    criteria = receipt.get("acceptance_criteria")
    if isinstance(criteria, list):
        incomplete = [
            str(row.get("id"))
            for row in criteria
            if isinstance(row, Mapping) and row.get("disposition") not in {"met", "not_applicable"}
        ]
        if incomplete:
            errors.append(
                "completion receipt has incomplete acceptance criteria: " + ", ".join(incomplete)
            )
    return {
        "eligible": not errors,
        "reason": "verified exact-head completion receipt" if not errors else errors[0],
        "errors": errors,
        "receipt_digest": receipt.get("receipt_digest"),
        "base_sha": (
            receipt.get("delivery", {}).get("base_sha")
            if isinstance(receipt.get("delivery"), Mapping)
            else None
        ),
        "head_sha": (
            receipt.get("delivery", {}).get("head_sha")
            if isinstance(receipt.get("delivery"), Mapping)
            else None
        ),
        "independent_verifier_status": basic.get("independent_verifier_status"),
    }


def load_receipt(path: str | Path) -> dict[str, Any]:
    """Load one JSON receipt and require an object payload."""
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{source} must contain a JSON object")
    return payload


def _default_git_runner(command: list[str]) -> subprocess.CompletedProcess[str]:
    """Run one bounded Git command."""
    return subprocess.run(command, capture_output=True, text=True, check=False)


def _default_gh_runner(command: list[str]) -> subprocess.CompletedProcess[str]:
    """Run one bounded GitHub CLI command."""
    return subprocess.run(command, capture_output=True, text=True, check=False)


def _git_text(
    repo_root: Path,
    arguments: list[str],
    *,
    runner: GitRunner,
) -> tuple[str | None, str | None]:
    """Return Git stdout or a compact failure."""
    result = runner(["git", "-C", str(repo_root), *arguments])
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip() or f"exit code {result.returncode}"
        return None, detail
    return result.stdout.strip(), None


def _gh_json(
    command: list[str],
    *,
    runner: GhRunner,
) -> tuple[Mapping[str, Any] | None, str | None]:
    """Return one GitHub REST object or a compact failure."""
    result = runner(command)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip() or f"exit code {result.returncode}"
        return None, detail
    try:
        payload = json.loads(result.stdout or "null")
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON: {exc.msg}"
    if not isinstance(payload, Mapping):
        return None, "GitHub response must be an object"
    return payload, None


def _resolve_git_sha(
    repo_root: Path,
    reference: str,
    *,
    runner: GitRunner,
) -> tuple[str | None, str | None]:
    """Resolve one exact commit object from a Git reference."""
    return _git_text(repo_root, ["rev-parse", "--verify", f"{reference}^{{commit}}"], runner=runner)


def _resolve_branch_sha(
    repo_root: Path,
    branch: str,
    *,
    runner: GitRunner,
) -> tuple[str | None, str | None]:
    """Resolve a local or origin-tracking branch without accepting abbreviations."""
    references = [
        branch if branch.startswith("refs/") else f"refs/heads/{branch}",
        f"refs/remotes/origin/{branch.removeprefix('origin/')}",
    ]
    failures: list[str] = []
    for reference in references:
        sha, error = _resolve_git_sha(repo_root, reference, runner=runner)
        if sha:
            return sha, None
        if error:
            failures.append(f"{reference}: {error}")
    return None, "; ".join(failures) or "branch reference unavailable"


def _git_diff_snapshot(
    repo_root: Path,
    *,
    base_sha: str,
    head_sha: str,
    runner: GitRunner,
) -> tuple[dict[str, Any] | None, str | None]:
    """Derive changed paths and a deterministic numstat from an exact pair."""
    paths, error = _git_text(
        repo_root,
        ["diff", "--name-only", "--no-renames", f"{base_sha}..{head_sha}"],
        runner=runner,
    )
    if error:
        return None, f"changed-path comparison failed: {error}"
    numstat, error = _git_text(
        repo_root,
        ["diff", "--numstat", "--no-renames", f"{base_sha}..{head_sha}"],
        runner=runner,
    )
    if error:
        return None, f"diffstat comparison failed: {error}"
    changed_paths = [line for line in (paths or "").splitlines() if line]
    additions = 0
    deletions = 0
    files = 0
    for line in (numstat or "").splitlines():
        parts = line.split("\t", 2)
        if len(parts) != 3:
            return None, f"malformed git numstat row: {line!r}"
        add_text, delete_text, _path = parts
        files += 1
        if add_text != "-":
            try:
                additions += int(add_text)
            except ValueError:
                return None, f"malformed additions count: {add_text!r}"
        if delete_text != "-":
            try:
                deletions += int(delete_text)
            except ValueError:
                return None, f"malformed deletions count: {delete_text!r}"
    return {
        "changed_paths": changed_paths,
        "stat": {"files": files, "additions": additions, "deletions": deletions},
    }, None


def verify_receipt_against_git(  # noqa: C901, PLR0912, PLR0913, PLR0915 - exact-head gate
    receipt: Mapping[str, Any],
    *,
    repo_root: str | Path,
    repository: str | None = None,
    issue_contract: str | None = None,
    pr_snapshot: Mapping[str, Any] | None = None,
    issue_snapshot: Mapping[str, Any] | None = None,
    git_runner: GitRunner | None = None,
    gh_runner: GhRunner | None = None,
    artifact_root: str | Path | None = None,
) -> dict[str, Any]:
    """Verify a receipt against exact Git state and optional GitHub PR state.

    Tests and offline callers can inject ``pr_snapshot``/``issue_snapshot`` and
    runners.  When omitted, the verifier reads the current issue and covering
    PR through REST-backed ``gh api`` calls.  A PR head or branch that has moved
    after the receipt was produced is a hard failure, even when the later commit
    itself is otherwise green.
    """
    root = Path(repo_root).resolve(strict=False)
    git = git_runner or _default_git_runner
    gh = gh_runner or _default_gh_runner
    expected_repository = repository or receipt.get("repository")
    basic = validate_receipt(
        receipt,
        expected_repository=expected_repository if isinstance(expected_repository, str) else None,
        issue_contract=issue_contract,
        artifact_root=artifact_root or root,
    )
    errors = list(basic.get("errors", []))
    details: dict[str, Any] = {"offline": basic, "git": {}, "github": {}}
    delivery = receipt.get("delivery") if isinstance(receipt.get("delivery"), Mapping) else {}
    base_sha = delivery.get("base_sha")
    head_sha = delivery.get("head_sha")
    branch = delivery.get("branch")
    if not _is_sha(base_sha) or not _is_sha(head_sha) or not isinstance(branch, str):
        return {
            "schema": SCHEMA,
            "ok": False,
            "errors": errors or ["delivery binding is malformed"],
            **details,
        }

    for name, sha in (("base", base_sha), ("head", head_sha)):
        resolved, error = _resolve_git_sha(root, sha, runner=git)
        details["git"][f"{name}_resolved"] = resolved
        if error:
            errors.append(f"{name} commit does not exist: {error}")
        elif resolved != sha:
            errors.append(f"{name} commit resolved to {resolved}, not receipt SHA {sha}")

    branch_sha, branch_error = _resolve_branch_sha(root, branch, runner=git)
    details["git"]["branch_resolved"] = branch_sha
    if branch_sha is not None and branch_sha != head_sha:
        errors.append(
            f"branch {branch} moved to {branch_sha}; receipt proof is for earlier head {head_sha}"
        )
    elif branch_sha is None and pr_snapshot is None:
        errors.append(f"delivery branch {branch} is unavailable: {branch_error}")

    observed_diff, diff_error = _git_diff_snapshot(
        root,
        base_sha=base_sha,
        head_sha=head_sha,
        runner=git,
    )
    if diff_error:
        errors.append(diff_error)
    elif observed_diff is not None:
        details["git"]["diff"] = observed_diff
        declared_diff = receipt.get("diff") if isinstance(receipt.get("diff"), Mapping) else {}
        if sorted(declared_diff.get("changed_paths", [])) != sorted(observed_diff["changed_paths"]):
            errors.append("receipt changed_paths do not match the exact base/head Git diff")
        if declared_diff.get("stat") != observed_diff["stat"]:
            errors.append("receipt diff.stat does not match the exact base/head Git diff")

    pr = receipt.get("covering_pr")
    if isinstance(pr, Mapping):
        pr_number = pr.get("number")
        if _is_int(pr_number):
            if pr_snapshot is None:
                pr_snapshot, error = _gh_json(
                    [
                        "gh",
                        "api",
                        f"repos/{expected_repository}/pulls/{pr_number}",
                    ],
                    runner=gh,
                )
                if error:
                    errors.append(f"covering PR lookup failed: {error}")
            if pr_snapshot is not None:
                details["github"]["pull_request"] = dict(pr_snapshot)
                pr_head = (
                    pr_snapshot.get("head", {}).get("sha")
                    if isinstance(pr_snapshot.get("head"), Mapping)
                    else None
                )
                pr_base = (
                    pr_snapshot.get("base", {}).get("sha")
                    if isinstance(pr_snapshot.get("base"), Mapping)
                    else None
                )
                pr_ref = (
                    pr_snapshot.get("head", {}).get("ref")
                    if isinstance(pr_snapshot.get("head"), Mapping)
                    else None
                )
                if pr_head != head_sha:
                    errors.append(
                        f"covering PR head {pr_head} does not match receipt head {head_sha}"
                    )
                if pr_base != base_sha:
                    errors.append(
                        f"covering PR base {pr_base} does not match receipt base {base_sha}"
                    )
                if isinstance(pr_ref, str) and pr_ref != branch:
                    errors.append(
                        f"covering PR branch {pr_ref} does not match receipt branch {branch}"
                    )
                pr_state = str(pr_snapshot.get("state") or "").upper()
                if pr_state and pr_state not in {"OPEN", "CLOSED"}:
                    errors.append(f"covering PR state is malformed: {pr_state}")
                declared_state = str(pr.get("state") or "").upper()
                if declared_state == "OPEN" and pr_state != "OPEN":
                    errors.append("receipt calls covering PR OPEN but GitHub is not open")
                if declared_state == "CLOSED" and pr_state != "CLOSED":
                    errors.append("receipt calls covering PR CLOSED but GitHub is not closed")
                if declared_state == "MERGED" and (
                    pr_state != "CLOSED" or not pr_snapshot.get("merged_at")
                ):
                    errors.append("receipt calls covering PR MERGED but GitHub is not merged")

    if (
        issue_contract is None
        and isinstance(expected_repository, str)
        and _is_int(receipt.get("issue"))
    ):
        if issue_snapshot is None:
            issue_snapshot, error = _gh_json(
                [
                    "gh",
                    "api",
                    f"repos/{expected_repository}/issues/{receipt['issue']}",
                ],
                runner=gh,
            )
            if error:
                errors.append(f"issue contract lookup failed: {error}")
        if issue_snapshot is not None:
            details["github"]["issue"] = dict(issue_snapshot)
            body = issue_snapshot.get("body")
            if not isinstance(body, str):
                errors.append("issue body is unavailable for contract-digest verification")
            elif isinstance(receipt.get("contract"), Mapping):
                expected_digest = receipt["contract"].get("digest")
                if sha256_text(body) != expected_digest:
                    errors.append("issue contract changed after the receipt was reviewed")

    admission = validate_receipt(
        receipt,
        expected_repository=expected_repository if isinstance(expected_repository, str) else None,
        issue_contract=issue_contract,
        artifact_root=artifact_root or root,
        require_independent_verifier=True,
    )
    errors.extend(error for error in admission["errors"] if error not in errors)
    return {
        "schema": SCHEMA,
        "ok": not errors,
        "errors": errors,
        "receipt_digest": receipt.get("receipt_digest"),
        "repository": receipt.get("repository"),
        "issue": receipt.get("issue"),
        "base_sha": base_sha,
        "head_sha": head_sha,
        "branch": branch,
        **details,
    }


def _dump(payload: Mapping[str, Any]) -> None:
    """Print stable JSON for the CLI."""
    print(json.dumps(payload, indent=2, sort_keys=True))


def _load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from a file."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    """Run the offline builder or offline/Git-backed verifier."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build", help="build a self-digested offline receipt")
    build_parser.add_argument("--input", type=Path, required=True)
    build_parser.add_argument("--output", type=Path)
    verify_parser = subparsers.add_parser("verify", help="verify an offline or Git-backed receipt")
    verify_parser.add_argument("--receipt", type=Path, required=True)
    verify_parser.add_argument("--offline", action="store_true")
    verify_parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    verify_parser.add_argument("--repo", default=None)
    verify_parser.add_argument("--issue-contract-file", type=Path)
    verify_parser.add_argument("--artifact-root", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            receipt = build_receipt(_load_json(args.input))
            rendered = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
            if args.output:
                args.output.write_text(rendered, encoding="utf-8")
            else:
                print(rendered, end="")
            return 0

        receipt = load_receipt(args.receipt)
        issue_contract = None
        if args.issue_contract_file:
            issue_contract = args.issue_contract_file.read_text(encoding="utf-8")
        if args.offline:
            result = validate_receipt(
                receipt,
                expected_repository=args.repo,
                issue_contract=issue_contract,
                artifact_root=args.artifact_root,
            )
        else:
            result = verify_receipt_against_git(
                receipt,
                repo_root=args.repo_root,
                repository=args.repo,
                issue_contract=issue_contract,
                artifact_root=args.artifact_root,
            )
        _dump(result)
        return 0 if result.get("ok") else 2
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        _dump({"schema": SCHEMA, "ok": False, "errors": [str(exc)]})
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
