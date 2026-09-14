"""Canonical receipt-aware validators for strict answerability proof.

The answerability preflight accepts a producer or analysis receipt only after a
fixed validator has consumed the receipt bytes and compared them with the
manifest.  A manifest cannot select an arbitrary test, command, or AST symbol
as an independent validator.  These checks validate provenance and identity;
they do not run a campaign or make a scientific claim.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from robot_sf.benchmark.research_answerability import strict_proof_input_provenance_error

ReceiptSurface = Literal["producer", "analysis"]

PRODUCER_VALIDATOR_ID = "research_answerability.producer_receipt.v1"
ANALYSIS_VALIDATOR_ID = "research_answerability.analysis_receipt.v1"
VALIDATOR_IDS = {
    "producer": PRODUCER_VALIDATOR_ID,
    "analysis": ANALYSIS_VALIDATOR_ID,
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PRODUCER_RECEIPT_SCHEMA = "research_answerability_producer_receipt.v1"
_ANALYSIS_RECEIPT_SCHEMA = "research_answerability_analysis_receipt.v1"


class ReceiptValidationError(ValueError):
    """Raised when a canonical answerability receipt is not independently valid."""


def _stable_json(path: Path) -> Mapping[str, Any]:
    """Read a receipt twice and parse a strict JSON object."""

    first = path.read_bytes()
    second = path.read_bytes()
    if first != second:
        raise ReceiptValidationError("receipt changed while being consumed")

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ReceiptValidationError(f"receipt contains duplicate key: {key}")
            result[key] = value
        return result

    def invalid(value: str) -> None:
        raise ReceiptValidationError(f"receipt contains non-finite JSON literal: {value}")

    try:
        payload = json.loads(first.decode("utf-8"), object_pairs_hook=pairs, parse_constant=invalid)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReceiptValidationError(f"could not parse receipt: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ReceiptValidationError("receipt must be a JSON object")
    return payload


def _safe_file(repo_root: Path, value: Any, field: str) -> tuple[Path, str]:
    """Resolve and validate a repository-relative source file."""
    if not isinstance(value, str) or not value.strip():
        raise ReceiptValidationError(f"{field}.path must be a non-empty repository-relative path")
    relative = Path(value.strip())
    if relative.is_absolute() or ".." in relative.parts:
        raise ReceiptValidationError(f"{field}.path must be repository-relative")
    root = repo_root.resolve()
    path = root / relative
    try:
        resolved = path.resolve()
    except (OSError, RuntimeError) as exc:
        raise ReceiptValidationError(f"{field}.path cannot be resolved safely") from exc
    if resolved == root or root not in resolved.parents:
        raise ReceiptValidationError(f"{field}.path must resolve within the repository")
    if not resolved.is_file():
        raise ReceiptValidationError(f"{field}.path must name an existing file")
    provenance_error = strict_proof_input_provenance_error(
        resolved,
        repo_root=root,
        field=f"{field} source",
    )
    if provenance_error:
        raise ReceiptValidationError(provenance_error)
    return resolved, relative.as_posix()


def _file_digest(path: Path, field: str) -> str:
    """Return a stable file digest, rejecting changes during the read."""
    first = path.read_bytes()
    second = path.read_bytes()
    if first != second:
        raise ReceiptValidationError(f"{field} changed while being consumed")
    return hashlib.sha256(first).hexdigest()


def _source_digest(source: Mapping[str, Any], *, repo_root: Path, field: str) -> tuple[str, str]:
    """Validate one receipt-bound source path and digest."""
    path, relative = _safe_file(repo_root, source.get("path"), field)
    digest = source.get("sha256")
    if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest.lower()):
        raise ReceiptValidationError(f"{field}.sha256 must be a 64-hex SHA-256")
    actual = _file_digest(path, field)
    if actual != digest.lower():
        raise ReceiptValidationError(f"{field}.sha256 does not match committed source bytes")
    return relative, actual


def _common(
    payload: Mapping[str, Any],
    *,
    surface: ReceiptSurface,
    manifest: Mapping[str, Any],
    identity: Mapping[str, Any],
) -> None:
    """Validate the common receipt identity independently of the caller."""
    expected_schema = (
        _PRODUCER_RECEIPT_SCHEMA if surface == "producer" else _ANALYSIS_RECEIPT_SCHEMA
    )
    if payload.get("schema_version") != expected_schema:
        raise ReceiptValidationError(f"{surface} receipt schema_version is not canonical")
    if payload.get("status") != "passed":
        raise ReceiptValidationError(f"{surface} receipt status must be passed")
    answerability = manifest.get("answerability")
    campaign = manifest.get("campaign")
    if not isinstance(answerability, Mapping) or not isinstance(campaign, Mapping):
        raise ReceiptValidationError("manifest lacks campaign/answerability identity")
    expected = {
        "campaign_id": campaign.get("id"),
        "question": answerability.get("question", {}).get("research_question")
        if isinstance(answerability.get("question"), Mapping)
        else None,
        "estimand": answerability.get("estimand", {}).get("primary")
        if isinstance(answerability.get("estimand"), Mapping)
        else None,
    }
    for field, expected_value in expected.items():
        if not isinstance(expected_value, str) or not expected_value.strip():
            raise ReceiptValidationError(f"manifest identity.{field} is missing")
        if payload.get(field) != expected_value or identity.get(field) != expected_value:
            raise ReceiptValidationError(f"{surface} receipt {field} is not bound to the manifest")


def _validate_producer(  # noqa: C901
    payload: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    identity: Mapping[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    """Validate required producer rows and their committed implementation sources."""
    _common(payload, surface="producer", manifest=manifest, identity=identity)
    answerability = manifest["answerability"]
    producers = answerability.get("producers")
    required = [
        producer
        for producer in producers or []
        if isinstance(producer, Mapping) and producer.get("required", True)
    ]
    expected_fields = sorted(str(producer["field"]) for producer in required)
    if identity.get("producer_fields") != expected_fields:
        raise ReceiptValidationError(
            "producer identity.producer_fields is not bound to the manifest"
        )
    if payload.get("producer_fields") != expected_fields:
        raise ReceiptValidationError("producer receipt producer_fields are not canonical")
    rows = payload.get("producer_rows")
    if not isinstance(rows, list) or len(rows) != len(required):
        raise ReceiptValidationError(
            "producer receipt producer_rows must cover every required producer"
        )
    by_field: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ReceiptValidationError(
                f"producer receipt producer_rows[{index}] must be an object"
            )
        field = row.get("field")
        if not isinstance(field, str) or field in by_field:
            raise ReceiptValidationError("producer receipt producer_rows must have unique fields")
        by_field[field] = row

    source_digests: dict[str, str] = {}
    for producer in required:
        field = str(producer["field"])
        row = by_field.get(field)
        if row is None:
            raise ReceiptValidationError(f"producer receipt is missing required field {field!r}")
        for key in ("producer", "source"):
            if row.get(key) != producer.get(key):
                raise ReceiptValidationError(
                    f"producer receipt {field}.{key} is not bound to the manifest"
                )
        if row.get("status") != "available":
            raise ReceiptValidationError(f"producer receipt {field}.status must be available")
        if row.get("execution_mode") not in {"native", "adapter"}:
            raise ReceiptValidationError(
                f"producer receipt {field}.execution_mode must be native or adapter"
            )
        source_path, digest = _source_digest(
            {"path": row.get("source"), "sha256": row.get("source_sha256")},
            repo_root=repo_root,
            field=f"producer receipt {field}.source",
        )
        if source_path != str(producer["source"]):
            raise ReceiptValidationError(f"producer receipt {field}.source path is not canonical")
        source_digests[field] = digest
    if set(by_field) != set(expected_fields):
        raise ReceiptValidationError("producer receipt contains an unexpected producer field")
    return {
        "validator_id": PRODUCER_VALIDATOR_ID,
        "checked_fields": expected_fields,
        "source_sha256": source_digests,
    }


def _validate_analysis(  # noqa: C901
    payload: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    identity: Mapping[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    """Validate analysis identity, checks, and the committed analysis owner source."""
    _common(payload, surface="analysis", manifest=manifest, identity=identity)
    answerability = manifest["answerability"]
    analysis = answerability.get("analysis")
    if not isinstance(analysis, Mapping):
        raise ReceiptValidationError("manifest answerability.analysis is missing")
    analysis_id = analysis.get("analysis_id")
    if identity.get("analysis_id") != analysis_id or payload.get("analysis_id") != analysis_id:
        raise ReceiptValidationError("analysis receipt analysis_id is not bound to the manifest")
    command = analysis.get("command")
    if not isinstance(command, str) or not command.strip() or payload.get("command") != command:
        raise ReceiptValidationError("analysis receipt command is not bound to the manifest")
    for status_field in ("dry_run_status", "comparability_status"):
        if payload.get(status_field) != "passed":
            raise ReceiptValidationError(f"analysis receipt {status_field} must be passed")
    checks = payload.get("checks")
    if not isinstance(checks, Mapping):
        raise ReceiptValidationError("analysis receipt checks must be an object")
    expected_checks = {"dry_run": "dry_run_status", "comparability": "comparability_status"}
    if set(checks) != set(expected_checks):
        raise ReceiptValidationError(
            "analysis receipt checks must contain dry_run and comparability"
        )
    for check_name, status_field in expected_checks.items():
        check = checks[check_name]
        if not isinstance(check, Mapping) or check.get("status") != payload[status_field]:
            raise ReceiptValidationError(f"analysis receipt checks.{check_name} is not passed")

    source_spec = analysis.get("source")
    source = payload.get("analysis_source")
    if not isinstance(source_spec, str) or not source_spec.strip():
        raise ReceiptValidationError("manifest answerability.analysis.source is required")
    if not isinstance(source, Mapping) or source.get("path") != source_spec:
        raise ReceiptValidationError(
            "analysis receipt analysis_source is not bound to the manifest"
        )
    source_path, digest = _source_digest(
        source,
        repo_root=repo_root,
        field="analysis receipt analysis_source",
    )
    if source_path != source_spec:
        raise ReceiptValidationError("analysis receipt analysis_source path is not canonical")
    return {
        "validator_id": ANALYSIS_VALIDATOR_ID,
        "checked_checks": sorted(checks),
        "source_sha256": digest,
    }


def validate_receipt(
    surface: ReceiptSurface,
    path: Path,
    *,
    manifest: Mapping[str, Any],
    identity: Mapping[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    """Consume and validate one canonical producer or analysis receipt."""
    payload = _stable_json(Path(path))
    if surface == "producer":
        return _validate_producer(
            payload, manifest=manifest, identity=identity, repo_root=repo_root.resolve()
        )
    return _validate_analysis(
        payload, manifest=manifest, identity=identity, repo_root=repo_root.resolve()
    )


def validate_producer_receipt(
    path: Path,
    *,
    manifest: Mapping[str, Any],
    identity: Mapping[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    """Consume and validate a canonical producer receipt."""
    return validate_receipt(
        "producer", path, manifest=manifest, identity=identity, repo_root=repo_root
    )


def validate_analysis_receipt(
    path: Path,
    *,
    manifest: Mapping[str, Any],
    identity: Mapping[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    """Consume and validate a canonical analysis receipt."""
    return validate_receipt(
        "analysis", path, manifest=manifest, identity=identity, repo_root=repo_root
    )


__all__ = [
    "ANALYSIS_VALIDATOR_ID",
    "PRODUCER_VALIDATOR_ID",
    "VALIDATOR_IDS",
    "ReceiptValidationError",
    "validate_analysis_receipt",
    "validate_producer_receipt",
    "validate_receipt",
]
