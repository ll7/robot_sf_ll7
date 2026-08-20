#!/usr/bin/env python3
"""Build and compare deterministic goal-autopilot blocker receipts.

The receipt is workflow evidence only.  It does not mutate GitHub state, grant
authority, acquire inputs, or decide whether a scientific or maintainer action
is allowed.  Receipts are intentionally stored outside the repository under
the common Git ``codex-agent-runs/active`` artifact owner.

All fields participating in redispatch decisions use an explicit field-state
wrapper.  Consequently an unavailable value, a missing value, an explicit
``not_applicable`` value, and a successful empty value remain distinct in the
fingerprint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scripts.dev.git_common import resolve_agent_artifact_dir

SCHEMA = "goal_blocker_receipt.v1"
FINGERPRINT_SCHEMA = "goal_blocker_fingerprint.v1"
RECEIPT_SUBDIR = "goal-blocker-receipts"
BLOCKER_CLASSES = frozenset(
    {
        "needs_spec",
        "human_decision",
        "external_input",
        "compute",
        "environment",
        "dependency",
        "validation",
        "authority",
    }
)
FIELD_STATUSES = frozenset({"available", "missing", "unavailable", "not_applicable"})
FINGERPRINT_FIELDS = (
    "issue.body_digest",
    "issue.labels",
    "repository.origin_main_sha",
    "pull_request.number",
    "pull_request.base_sha",
    "pull_request.head_sha",
    "dependency_state",
    "required_inputs",
)
DEFAULT_INVALIDATING_FIELDS = FINGERPRINT_FIELDS
_MISSING = object()


def available(value: Any) -> dict[str, Any]:
    """Wrap a successfully observed value, including empty or false values."""
    return {"status": "available", "value": _canonicalize(value)}


def missing(reason: str = "not_observed") -> dict[str, str]:
    """Return an explicit field state for data that was not supplied."""
    return {"status": "missing", "reason": _reason(reason)}


def unavailable(reason: str) -> dict[str, str]:
    """Return an explicit field state for data that could not be read."""
    return {"status": "unavailable", "reason": _reason(reason)}


def not_applicable(reason: str) -> dict[str, str]:
    """Return an explicit stable field state for a field outside the scope."""
    return {"status": "not_applicable", "reason": _reason(reason)}


def _reason(value: Any) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError("field-state reason must be non-empty")
    return text


def _canonicalize(value: Any) -> Any:
    """Return JSON-compatible, recursively key-sorted data."""
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("canonical JSON object keys must be strings")
            normalized[key] = _canonicalize(item)
        return {key: normalized[key] for key in sorted(normalized)}
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError(f"value is not JSON-compatible: {type(value).__name__}")


def _field_state(value: Any, *, missing_reason: str) -> dict[str, Any]:
    """Normalize a raw value or an explicit field-state wrapper."""
    if value is _MISSING:
        return missing(missing_reason)
    if isinstance(value, Mapping) and "status" in value:
        status = value.get("status")
        if status not in FIELD_STATUSES:
            raise ValueError(f"unknown field status: {status!r}")
        if status == "available":
            if "value" not in value:
                raise ValueError("available field state must contain value")
            return available(value["value"])
        if "reason" not in value:
            raise ValueError(f"{status} field state must contain reason")
        return {
            "status": status,
            "reason": _reason(value["reason"]),
        }
    return available(value)


def _labels_state(value: Any) -> dict[str, Any]:
    state = _field_state(value, missing_reason="issue labels were not observed")
    if state["status"] != "available":
        return state
    labels = state["value"]
    if not isinstance(labels, list) or any(not isinstance(label, str) for label in labels):
        raise ValueError("issue labels must be a list of strings")
    return available(sorted(set(labels)))


def build_fingerprint_inputs(
    *,
    issue_body_digest: Any = _MISSING,
    issue_labels: Any = _MISSING,
    origin_main_sha: Any = _MISSING,
    pr_number: Any = _MISSING,
    base_sha: Any = _MISSING,
    head_sha: Any = _MISSING,
    dependency_state: Any = _MISSING,
    required_inputs: Any = _MISSING,
) -> dict[str, dict[str, Any]]:
    """Build the normalized, versioned input set used by the fingerprint."""
    return {
        "issue.body_digest": _field_state(
            issue_body_digest, missing_reason="issue body digest was not observed"
        ),
        "issue.labels": _labels_state(issue_labels),
        "repository.origin_main_sha": _field_state(
            origin_main_sha, missing_reason="origin/main SHA was not observed"
        ),
        "pull_request.number": _field_state(
            pr_number, missing_reason="relevant pull request number was not observed"
        ),
        "pull_request.base_sha": _field_state(
            base_sha, missing_reason="relevant pull request base SHA was not observed"
        ),
        "pull_request.head_sha": _field_state(
            head_sha, missing_reason="relevant pull request head SHA was not observed"
        ),
        "dependency_state": _field_state(
            dependency_state, missing_reason="dependency state was not observed"
        ),
        "required_inputs": _field_state(
            required_inputs, missing_reason="required-input state was not observed"
        ),
    }


def normalize_fingerprint_inputs(raw: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Normalize a complete input mapping, failing closed on unknown or missing keys."""
    if not isinstance(raw, Mapping):
        raise ValueError("fingerprint inputs must be an object")
    if "fingerprint_inputs" in raw:
        raw = raw["fingerprint_inputs"]
    unknown = sorted(set(raw) - set(FINGERPRINT_FIELDS))
    if unknown:
        raise ValueError(f"unknown fingerprint fields: {unknown}")
    missing_fields = [field for field in FINGERPRINT_FIELDS if field not in raw]
    if missing_fields:
        raise ValueError(f"missing fingerprint fields: {missing_fields}")
    normalized = {
        field: _field_state(raw[field], missing_reason=f"{field} was not observed")
        for field in FINGERPRINT_FIELDS
    }
    labels = normalized["issue.labels"]
    if labels["status"] == "available":
        normalized["issue.labels"] = _labels_state(labels)
    return normalized


def fingerprint(inputs: Mapping[str, Any]) -> str:
    """Return the stable SHA-256 fingerprint for normalized blocker inputs."""
    normalized = normalize_fingerprint_inputs(inputs)
    payload = {
        "schema": FINGERPRINT_SCHEMA,
        "inputs": normalized,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _evidence_items(value: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("evidence must be a sequence of objects")
    result: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValueError(f"evidence[{index}] must be an object")
        source = item.get("source")
        if not isinstance(source, str) or not source.strip():
            raise ValueError(f"evidence[{index}].source must be non-empty")
        result.append(_canonicalize(dict(item)))
    return result


def _required_input(value: str | Mapping[str, Any], unblock_condition: str) -> dict[str, Any]:
    if isinstance(value, str):
        name = value.strip()
        if not name:
            raise ValueError("required input name must be non-empty")
        return {"name": name, "condition": unblock_condition}
    if not isinstance(value, Mapping):
        raise ValueError("required_input must be a string or object")
    normalized = _canonicalize(dict(value))
    name = normalized.get("name")
    condition = normalized.get("condition")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("required_input.name must be non-empty")
    if not isinstance(condition, str) or not condition.strip():
        raise ValueError("required_input.condition must be non-empty")
    return normalized


def build_receipt(  # noqa: C901, PLR0913 - explicit receipt contract fields
    *,
    repository: str,
    issue_number: int,
    blocker_class: str,
    required_input: str | Mapping[str, Any],
    unblock_condition: str,
    evidence: Sequence[Mapping[str, Any]],
    safe_work_completed: Sequence[str],
    next_owner: str,
    retryable: bool,
    recommended_label: str | None = None,
    recommended_state: str | None = None,
    invalidating_fields: Sequence[str] | None = None,
    issue_body_digest: Any = _MISSING,
    issue_labels: Any = _MISSING,
    origin_main_sha: Any = _MISSING,
    pr_number: Any = _MISSING,
    base_sha: Any = _MISSING,
    head_sha: Any = _MISSING,
    dependency_state: Any = _MISSING,
    required_inputs: Any = _MISSING,
) -> dict[str, Any]:
    """Build a validated deterministic blocker receipt."""
    if not isinstance(repository, str) or not repository.strip():
        raise ValueError("repository must be non-empty")
    if type(issue_number) is not int or issue_number < 1:
        raise ValueError("issue_number must be a positive integer")
    if blocker_class not in BLOCKER_CLASSES:
        raise ValueError(f"unknown blocker class: {blocker_class!r}")
    if not isinstance(unblock_condition, str) or not unblock_condition.strip():
        raise ValueError("unblock_condition must be non-empty")
    if not isinstance(next_owner, str) or not next_owner.strip():
        raise ValueError("next_owner must be non-empty")
    if type(retryable) is not bool:
        raise ValueError("retryable must be a boolean")
    for name, value in (
        ("recommended_label", recommended_label),
        ("recommended_state", recommended_state),
    ):
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"{name} must be null or a non-empty string")
    if isinstance(safe_work_completed, (str, bytes)) or not isinstance(
        safe_work_completed, Sequence
    ):
        raise ValueError("safe_work_completed must be a sequence")
    if any(not isinstance(item, str) or not item.strip() for item in safe_work_completed):
        raise ValueError("safe_work_completed entries must be non-empty strings")

    inputs = build_fingerprint_inputs(
        issue_body_digest=issue_body_digest,
        issue_labels=issue_labels,
        origin_main_sha=origin_main_sha,
        pr_number=pr_number,
        base_sha=base_sha,
        head_sha=head_sha,
        dependency_state=dependency_state,
        required_inputs=required_inputs,
    )
    invalidating = list(invalidating_fields or DEFAULT_INVALIDATING_FIELDS)
    if not invalidating or len(set(invalidating)) != len(invalidating):
        raise ValueError("invalidating_fields must be a non-empty unique sequence")
    unknown = sorted(set(invalidating) - set(FINGERPRINT_FIELDS))
    if unknown:
        raise ValueError(f"invalidating_fields contain unknown fields: {unknown}")
    receipt = {
        "schema": SCHEMA,
        "issue": {"repository": repository.strip(), "number": issue_number},
        "observed": {
            "issue_revision": {
                "body_digest": inputs["issue.body_digest"],
                "labels": inputs["issue.labels"],
            },
            "repository": {
                "origin_main_sha": inputs["repository.origin_main_sha"],
            },
            "pull_request": {
                "number": inputs["pull_request.number"],
                "base_sha": inputs["pull_request.base_sha"],
                "head_sha": inputs["pull_request.head_sha"],
            },
            "dependency_state": inputs["dependency_state"],
            "required_inputs": inputs["required_inputs"],
        },
        "fingerprint_inputs": inputs,
        "blocker": {
            "class": blocker_class,
            "required_input": _required_input(required_input, unblock_condition),
            "unblock_condition": unblock_condition.strip(),
            "evidence": _evidence_items(evidence),
            "safe_work_completed": list(safe_work_completed),
            "fingerprint": fingerprint(inputs),
            "retryable": retryable,
            "invalidating_fields": invalidating,
        },
        "recommendation": {
            "label": recommended_label.strip() if isinstance(recommended_label, str) else None,
            "state": recommended_state.strip() if isinstance(recommended_state, str) else None,
            "next_owner": next_owner.strip(),
        },
    }
    validation = validate_receipt(receipt)
    if not validation["ok"]:
        raise ValueError("built receipt failed validation: " + "; ".join(validation["errors"]))
    return receipt


def validate_receipt(receipt: Any) -> dict[str, Any]:  # noqa: C901, PLR0912, PLR0915 - fail-closed contract checks
    """Return ``ok`` and stable validation errors for one receipt object."""
    errors: list[str] = []
    if not isinstance(receipt, Mapping):
        return {"ok": False, "errors": ["receipt must be an object"]}
    if receipt.get("schema") != SCHEMA:
        errors.append(f"schema must be {SCHEMA!r}")
    issue = receipt.get("issue")
    if not isinstance(issue, Mapping):
        errors.append("issue must be an object")
    else:
        if not isinstance(issue.get("repository"), str) or not issue["repository"].strip():
            errors.append("issue.repository must be non-empty")
        if type(issue.get("number")) is not int or issue["number"] < 1:
            errors.append("issue.number must be a positive integer")

    inputs: dict[str, dict[str, Any]] | None = None
    try:
        raw_inputs = receipt.get("fingerprint_inputs")
        if not isinstance(raw_inputs, Mapping):
            raise ValueError("fingerprint_inputs must be an object")
        inputs = normalize_fingerprint_inputs(raw_inputs)
    except (TypeError, ValueError) as exc:
        errors.append(str(exc))

    blocker = receipt.get("blocker")
    if not isinstance(blocker, Mapping):
        errors.append("blocker must be an object")
    else:
        if blocker.get("class") not in BLOCKER_CLASSES:
            errors.append("blocker.class is unknown")
        if (
            not isinstance(blocker.get("unblock_condition"), str)
            or not str(blocker.get("unblock_condition")).strip()
        ):
            errors.append("blocker.unblock_condition must be non-empty")
        try:
            _required_input(
                blocker.get("required_input"), str(blocker.get("unblock_condition", ""))
            )
        except (TypeError, ValueError) as exc:
            errors.append(f"blocker.required_input: {exc}")
        try:
            _evidence_items(blocker.get("evidence", []))
        except (TypeError, ValueError) as exc:
            errors.append(f"blocker.evidence: {exc}")
        work = blocker.get("safe_work_completed")
        if (
            isinstance(work, (str, bytes))
            or not isinstance(work, list)
            or any(not isinstance(item, str) or not item.strip() for item in (work or []))
        ):
            errors.append("blocker.safe_work_completed must be a list of non-empty strings")
        if type(blocker.get("retryable")) is not bool:
            errors.append("blocker.retryable must be a boolean")
        invalidating = blocker.get("invalidating_fields")
        if (
            not isinstance(invalidating, list)
            or not invalidating
            or any(not isinstance(item, str) for item in invalidating)
            or len(set(invalidating)) != len(invalidating)
            or any(item not in FINGERPRINT_FIELDS for item in invalidating)
        ):
            errors.append("blocker.invalidating_fields must name unique fingerprint fields")
        if not isinstance(blocker.get("fingerprint"), str):
            errors.append("blocker.fingerprint must be a string")

    recommendation = receipt.get("recommendation")
    if not isinstance(recommendation, Mapping):
        errors.append("recommendation must be an object")
    elif (
        not isinstance(recommendation.get("next_owner"), str)
        or not str(recommendation.get("next_owner")).strip()
    ):
        errors.append("recommendation.next_owner must be non-empty")

    if inputs is not None and isinstance(blocker, Mapping):
        try:
            expected_fingerprint = fingerprint(inputs)
            if blocker.get("fingerprint") != expected_fingerprint:
                errors.append("blocker.fingerprint does not match fingerprint_inputs")
        except (TypeError, ValueError) as exc:
            errors.append(f"fingerprint computation failed: {exc}")

        observed = receipt.get("observed")
        expected_observed = {
            "issue_revision": {
                "body_digest": inputs["issue.body_digest"],
                "labels": inputs["issue.labels"],
            },
            "repository": {"origin_main_sha": inputs["repository.origin_main_sha"]},
            "pull_request": {
                "number": inputs["pull_request.number"],
                "base_sha": inputs["pull_request.base_sha"],
                "head_sha": inputs["pull_request.head_sha"],
            },
            "dependency_state": inputs["dependency_state"],
            "required_inputs": inputs["required_inputs"],
        }
        if _canonicalize(observed) != _canonicalize(expected_observed):
            errors.append("observed fields do not match fingerprint_inputs")

    return {"ok": not errors, "errors": sorted(set(errors))}


def receipt_directory(path: str | Path | None = None, *, mkdir: bool = False) -> Path:
    """Return the external active-ledger directory used for blocker receipts."""
    if path is not None and str(path):
        directory = Path(path)
        if mkdir:
            directory.mkdir(parents=True, exist_ok=True)
        return directory
    return resolve_agent_artifact_dir(RECEIPT_SUBDIR, mkdir=mkdir)


def receipt_path(issue_number: int, *, directory: str | Path | None = None) -> Path:
    """Return the stable external path for the latest receipt of an issue."""
    if type(issue_number) is not int or issue_number < 1:
        raise ValueError("issue_number must be a positive integer")
    return receipt_directory(directory) / f"issue-{issue_number}.json"


def write_receipt(receipt: Mapping[str, Any], *, path: str | Path | None = None) -> Path:
    """Atomically write one validated receipt outside the repository."""
    validation = validate_receipt(receipt)
    if not validation["ok"]:
        raise ValueError("cannot write invalid receipt: " + "; ".join(validation["errors"]))
    issue = receipt["issue"]
    target = Path(path) if path is not None else receipt_path(int(issue["number"]))
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(_canonicalize(dict(receipt)), indent=2, sort_keys=True) + "\n"
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, target)
    return target


def load_receipt(*, issue_number: int, directory: str | Path | None = None) -> dict[str, Any]:
    """Load the latest external receipt, classifying missing and malformed state."""
    path = receipt_path(issue_number, directory=directory)
    if not path.exists():
        return {"status": "missing", "path": str(path), "receipt": None, "errors": []}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "status": "malformed",
            "path": str(path),
            "receipt": None,
            "errors": [f"receipt read failed: {exc}"],
        }
    validation = validate_receipt(raw)
    if not validation["ok"]:
        return {
            "status": "malformed",
            "path": str(path),
            "receipt": None,
            "errors": validation["errors"],
        }
    return {"status": "available", "path": str(path), "receipt": raw, "errors": []}


def _field_is_comparable(state: Mapping[str, Any]) -> bool:
    return state.get("status") in {"available", "not_applicable"}


def evaluate_redispatch(
    receipt: Any,
    *,
    current_inputs: Mapping[str, Any] | None,
    repository: str | None = None,
    issue_number: int | None = None,
) -> dict[str, Any]:
    """Compare current blocker inputs with a receipt and return a safe action.

    ``blocked_unchanged`` is the only suppressing result.  Missing or
    unavailable current fields, malformed receipts, and identity mismatches
    always return an explicit re-evaluation action.
    """
    validation = validate_receipt(receipt)
    if not validation["ok"]:
        return {
            "decision": "invalid_receipt",
            "action": "re_evaluate",
            "reason": "receipt_invalid",
            "changed_fields": [],
            "errors": validation["errors"],
        }
    assert isinstance(receipt, Mapping)
    identity = receipt["issue"]
    if repository is not None and identity["repository"] != repository:
        return {
            "decision": "stale_receipt",
            "action": "re_evaluate",
            "reason": "receipt_repository_mismatch",
            "changed_fields": [],
            "errors": [],
        }
    if issue_number is not None and identity["number"] != issue_number:
        return {
            "decision": "stale_receipt",
            "action": "re_evaluate",
            "reason": "receipt_issue_mismatch",
            "changed_fields": [],
            "errors": [],
        }
    if current_inputs is None:
        return {
            "decision": "current_state_unavailable",
            "action": "re_evaluate",
            "reason": "current_fingerprint_inputs_unavailable",
            "changed_fields": [],
            "errors": [],
        }
    try:
        current = normalize_fingerprint_inputs(current_inputs)
    except (TypeError, ValueError) as exc:
        return {
            "decision": "current_state_unavailable",
            "action": "re_evaluate",
            "reason": "current_fingerprint_inputs_malformed",
            "changed_fields": [],
            "errors": [str(exc)],
        }
    prior = normalize_fingerprint_inputs(receipt["fingerprint_inputs"])
    invalidating = receipt["blocker"]["invalidating_fields"]
    current_unavailable = [
        field for field in invalidating if not _field_is_comparable(current[field])
    ]
    if current_unavailable:
        return {
            "decision": "current_state_unavailable",
            "action": "re_evaluate",
            "reason": "current_fields_unavailable",
            "changed_fields": current_unavailable,
            "errors": [],
        }
    prior_unavailable = [field for field in invalidating if not _field_is_comparable(prior[field])]
    if prior_unavailable:
        return {
            "decision": "stale_receipt",
            "action": "re_evaluate",
            "reason": "receipt_fields_unavailable",
            "changed_fields": prior_unavailable,
            "errors": [],
        }
    changed = [field for field in invalidating if prior[field] != current[field]]
    if changed:
        return {
            "decision": "blocker_changed",
            "action": "re_evaluate",
            "reason": "invalidating_fields_changed",
            "changed_fields": changed,
            "errors": [],
        }
    return {
        "decision": "blocked_unchanged",
        "action": "no_action",
        "reason": "fingerprint_matches",
        "changed_fields": [],
        "errors": [],
    }


def summarize_redispatch(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize receipt decisions for a compact goal-loop queue report."""
    counts: dict[str, int] = {}
    for row in rows:
        receipt = row.get("blocker_receipt")
        if not isinstance(receipt, Mapping):
            continue
        decision = str(receipt.get("decision", "unknown"))
        counts[decision] = counts.get(decision, 0) + 1
    return {
        "suppressed_redispatch_count": counts.get("blocked_unchanged", 0),
        "re_evaluation_count": sum(
            count for decision, count in counts.items() if decision != "blocked_unchanged"
        ),
        "decision_counts": dict(sorted(counts.items())),
    }


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True, help="receipt JSON path")
    parser.add_argument(
        "--current-inputs",
        type=Path,
        help="optional JSON object containing all fingerprint fields for comparison",
    )
    parser.add_argument("--repository", default=None)
    parser.add_argument("--issue", type=int, default=None)
    args = parser.parse_args()
    try:
        receipt = _load_json(args.receipt)
        payload: dict[str, Any] = {"validation": validate_receipt(receipt)}
        if args.current_inputs is not None:
            payload["redispatch"] = evaluate_redispatch(
                receipt,
                current_inputs=_load_json(args.current_inputs),
                repository=args.repository,
                issue_number=args.issue,
            )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2, sort_keys=True))
        return 1
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["validation"]["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
