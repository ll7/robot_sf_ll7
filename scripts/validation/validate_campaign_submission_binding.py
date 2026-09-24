#!/usr/bin/env python3
"""Compare-and-swap binding validator between campaign preflight and submission.

Fails closed when campaign authority-bearing inputs drift between final preflight
and live scheduler submission. Permitted volatile fields are recorded explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "campaign_submission_binding.v1"
PERMITTED_VOLATILE_FIELDS = frozenset(
    {
        "host",
        "hostname",
        "job_id_pending",
        "observed_at_utc",
        "pid",
        "process_id",
        "submission_nonce",
    }
)

AUTHORITY_CATEGORIES = (
    "checkpoint",
    "command_tokens",
    "config",
    "duplicate_state",
    "environment",
    "issue_state",
    "output_root",
    "resource_request",
    "row_ledger",
    "seed_set",
    "source",
)

FLAT_FIELD_MAP = {
    "active_job_ids": ("duplicate_state", "active_job_ids"),
    "admission_status": ("issue_state", "admission_status"),
    "checkpoint_path": ("checkpoint", "path"),
    "checkpoint_sha256": ("checkpoint", "model_sha256"),
    "claim_ref": ("issue_state", "claim_ref"),
    "command_tokens": ("command_tokens", "tokens"),
    "commit": ("source", "commit"),
    "config_path": ("config", "path"),
    "config_sha256": ("config", "content_sha256"),
    "cpus": ("resource_request", "cpus"),
    "duplicate_detected": ("duplicate_state", "duplicate_detected"),
    "expected_rows": ("row_ledger", "expected_rows"),
    "gpus": ("resource_request", "gpus"),
    "is_dirty": ("source", "is_dirty"),
    "issue_number": ("issue_state", "issue_number"),
    "ledger_sha256": ("row_ledger", "ledger_sha256"),
    "lock_sha256": ("environment", "lock_sha256"),
    "model_alias": ("checkpoint", "alias"),
    "nodes": ("resource_request", "nodes"),
    "output_root": ("output_root", "path"),
    "partition": ("resource_request", "partition"),
    "python_version": ("environment", "python_version"),
    "seed_policy": ("seed_set", "seed_policy"),
    "seeds": ("seed_set", "seeds"),
    "source_tree": ("source", "commit"),
    "time_limit": ("resource_request", "time_limit"),
    "tokens": ("command_tokens", "tokens"),
    "untracked_files": ("source", "untracked_files"),
}


def canonical_json(data: Any) -> str:
    """Return stable canonical JSON serialization with sorted keys."""
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_canonical(data: Any) -> str:
    """Return SHA-256 of canonical JSON."""
    return hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest()


def _strip_volatile(data: Any, allowed_volatile: frozenset[str]) -> Any:
    """Recursively remove volatile fields from an arbitrary JSON structure."""
    if isinstance(data, dict):
        cleaned: dict[str, Any] = {}
        for key, value in data.items():
            if key not in allowed_volatile:
                cleaned[key] = _strip_volatile(value, allowed_volatile)
        return cleaned
    if isinstance(data, list):
        return [_strip_volatile(item, allowed_volatile) for item in data]
    return data


def normalize_authority(raw: dict[str, Any], allowed_volatile: frozenset[str]) -> dict[str, Any]:
    """Normalize authority-bearing sections and fields, excluding volatile entries."""
    cleaned = _strip_volatile(raw, allowed_volatile)
    if not isinstance(cleaned, dict):
        return {}

    normalized: dict[str, Any] = {}

    for cat in AUTHORITY_CATEGORIES:
        if cat in cleaned and isinstance(cleaned[cat], dict):
            normalized[cat] = dict(cleaned[cat])

    for key, value in cleaned.items():
        if key in AUTHORITY_CATEGORIES and isinstance(value, dict):
            continue
        if key in FLAT_FIELD_MAP:
            cat, subkey = FLAT_FIELD_MAP[key]
            if cat not in normalized:
                normalized[cat] = {}
            normalized[cat][subkey] = value
        else:
            normalized[key] = value

    return {k: normalized[k] for k in sorted(normalized.keys())}


def sanitize_value(field: str, val: Any) -> Any:
    """Sanitize secrets, passwords, or tokens for safe diagnostic display."""
    field_lower = field.lower()
    for sensitive in ("token", "secret", "password", "key", "credential", "auth"):
        if sensitive in field_lower and isinstance(val, str) and len(val) > 4:
            digest = hashlib.sha256(val.encode("utf-8")).hexdigest()[:12]
            return f"<redacted len={len(val)} sha256={digest}>"
    if isinstance(val, (dict, list)):
        return val
    return val


def find_authority_differences(
    expected_data: Any,
    observed_data: Any,
    path: str = "",
) -> list[dict[str, Any]]:
    """Recursively compare authority data and collect sorted differences."""
    differences: list[dict[str, Any]] = []
    if isinstance(expected_data, dict) and isinstance(observed_data, dict):
        all_keys = sorted(set(expected_data.keys()) | set(observed_data.keys()))
        for key in all_keys:
            sub_path = f"{path}.{key}" if path else key
            if key not in expected_data:
                differences.append(
                    {
                        "diagnostic": f"field added in live submission: {sub_path}",
                        "expected": None,
                        "field": sub_path,
                        "observed": sanitize_value(sub_path, observed_data[key]),
                    }
                )
            elif key not in observed_data:
                differences.append(
                    {
                        "diagnostic": f"field missing in live submission: {sub_path}",
                        "expected": sanitize_value(sub_path, expected_data[key]),
                        "field": sub_path,
                        "observed": None,
                    }
                )
            else:
                differences.extend(
                    find_authority_differences(
                        expected_data[key],
                        observed_data[key],
                        path=sub_path,
                    )
                )
    elif isinstance(expected_data, list) and isinstance(observed_data, list):
        if expected_data != observed_data:
            differences.append(
                {
                    "diagnostic": (
                        f"list mismatch in {path}: expected {expected_data}, observed {observed_data}"
                    ),
                    "expected": sanitize_value(path, expected_data),
                    "field": path,
                    "observed": sanitize_value(path, observed_data),
                }
            )
    elif expected_data != observed_data:
        differences.append(
            {
                "diagnostic": (
                    f"value mismatch in {path}: expected {expected_data!r}, observed {observed_data!r}"
                ),
                "expected": sanitize_value(path, expected_data),
                "field": path,
                "observed": sanitize_value(path, observed_data),
            }
        )
    return differences


def _check_safety_invariants(
    submission_auth: dict[str, Any], differences: list[dict[str, Any]]
) -> None:
    """Enforce explicit fail-closed safety invariants on live submission."""
    dup_state = submission_auth.get("duplicate_state")
    if isinstance(dup_state, dict) and dup_state.get("duplicate_detected") is True:
        if not any(d["field"] == "duplicate_state.duplicate_detected" for d in differences):
            differences.append(
                {
                    "diagnostic": "duplicate job execution detected in live submission packet",
                    "expected": False,
                    "field": "duplicate_state.duplicate_detected",
                    "observed": True,
                }
            )

    source = submission_auth.get("source")
    if isinstance(source, dict) and source.get("is_dirty") is True:
        if not any(d["field"] == "source.is_dirty" for d in differences):
            differences.append(
                {
                    "diagnostic": "working tree is dirty at live submission boundary",
                    "expected": False,
                    "field": "source.is_dirty",
                    "observed": True,
                }
            )


def extract_volatile_diffs(
    preflight_raw: dict[str, Any],
    submission_raw: dict[str, Any],
    allowed_volatile: frozenset[str],
) -> list[str]:
    """Find which permitted volatile fields differ between preflight and submission."""
    differing_volatile: list[str] = []
    for field in sorted(allowed_volatile):
        v_pre = preflight_raw.get(field)
        v_sub = submission_raw.get(field)
        if v_pre != v_sub:
            differing_volatile.append(field)
    return differing_volatile


def validate_submission_binding(
    preflight_raw: dict[str, Any],
    submission_raw: dict[str, Any],
    allowed_volatile: frozenset[str] = PERMITTED_VOLATILE_FIELDS,
) -> dict[str, Any]:
    """Compare preflight and live submission packets, generating a binding receipt."""
    preflight_auth = normalize_authority(preflight_raw, allowed_volatile)
    submission_auth = normalize_authority(submission_raw, allowed_volatile)

    preflight_digest = sha256_canonical(preflight_auth)
    submission_digest = sha256_canonical(submission_auth)

    differences = find_authority_differences(preflight_auth, submission_auth)
    _check_safety_invariants(submission_auth, differences)
    differences.sort(key=lambda item: item["field"])

    first_diff = differences[0]["field"] if differences else None
    passed = len(differences) == 0 and (preflight_digest == submission_digest)

    volatile_diffs = extract_volatile_diffs(preflight_raw, submission_raw, allowed_volatile)
    authority_keys = sorted(set(preflight_auth.keys()) | set(submission_auth.keys()))
    now_utc = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    return {
        "authority_fields_checked": authority_keys,
        "differing_fields": differences,
        "first_differing_field": first_diff,
        "preflight_digest": preflight_digest,
        "recomputed_at_utc": now_utc,
        "schema": SCHEMA_VERSION,
        "status": "matched" if passed else "drift_detected",
        "submission_digest": submission_digest,
        "verdict": "pass" if passed else "fail",
        "volatile_fields_allowed": sorted(allowed_volatile),
        "volatile_fields_observed": volatile_diffs,
    }


def _load_json_packet(path: Path) -> dict[str, Any]:
    """Read and validate a JSON packet file."""
    if not path.is_file():
        raise FileNotFoundError(f"Packet file not found: {path}")
    content = path.read_text(encoding="utf-8")
    data = json.loads(content)
    if not isinstance(data, dict):
        raise ValueError(f"Packet must be a JSON object: {path}")
    return data


def format_text_report(receipt: dict[str, Any]) -> str:
    """Format binding receipt for terminal display."""
    lines: list[str] = []
    verdict = receipt["verdict"].upper()
    status = receipt["status"]
    lines.append(f"Campaign submission binding check: {verdict} ({status})")
    lines.append(f"  Preflight digest:   {receipt['preflight_digest']}")
    lines.append(f"  Submission digest:  {receipt['submission_digest']}")
    lines.append(f"  Authority fields:   {len(receipt['authority_fields_checked'])} checked")
    if receipt["volatile_fields_observed"]:
        obs = ", ".join(receipt["volatile_fields_observed"])
        lines.append(
            f"  Volatile fields:    {len(receipt['volatile_fields_observed'])} permitted differences ({obs})"
        )
    if receipt["differing_fields"]:
        lines.append(f"  First difference:   {receipt['first_differing_field']}")
        lines.append(f"  Total differences:  {len(receipt['differing_fields'])}")
        for diff in receipt["differing_fields"][:5]:
            lines.append(f"    - {diff['field']}: {diff['diagnostic']}")
        if len(receipt["differing_fields"]) > 5:
            remaining = len(receipt["differing_fields"]) - 5
            lines.append(f"    ... and {remaining} more difference(s)")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for submission binding validator."""
    parser = argparse.ArgumentParser(
        description="Verify campaign inputs drift neither subtly nor completely between preflight and submission."
    )
    parser.add_argument(
        "--preflight",
        required=True,
        type=Path,
        help="Path to preflight receipt JSON",
    )
    parser.add_argument(
        "--submission",
        required=True,
        type=Path,
        help="Path to live submission packet JSON",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        default=True,
        help="Run validation in check-only mode without mutating files or scheduler",
    )
    parser.add_argument(
        "--receipt-output",
        type=Path,
        default=None,
        help="Optional path to write binding receipt JSON",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (text or json)",
    )
    parser.add_argument(
        "--allow-volatile",
        action="append",
        default=[],
        help="Additional permitted volatile field name(s)",
    )
    return parser


def main() -> int:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args()

    try:
        preflight_raw = _load_json_packet(args.preflight)
        submission_raw = _load_json_packet(args.submission)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as err:
        sys.stderr.write(f"Error loading packet files: {err}\n")
        return 2

    allowed_volatile = PERMITTED_VOLATILE_FIELDS
    if args.allow_volatile:
        allowed_volatile = allowed_volatile | frozenset(args.allow_volatile)

    receipt = validate_submission_binding(
        preflight_raw,
        submission_raw,
        allowed_volatile=allowed_volatile,
    )

    if args.receipt_output is not None:
        args.receipt_output.parent.mkdir(parents=True, exist_ok=True)
        args.receipt_output.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    if args.format == "json":
        sys.stdout.write(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    else:
        sys.stdout.write(format_text_report(receipt) + "\n")

    return 0 if receipt["verdict"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
