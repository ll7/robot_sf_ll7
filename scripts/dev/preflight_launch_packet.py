#!/usr/bin/env python3
"""Preflight a heterogeneous launch-packet YAML for run-ready discovery."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from collections.abc import Iterable

SCHEMA_VERSION = "launch-packet-preflight.v1"
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
ISSUE_RE = re.compile(r"issue[_-](\d+)|#(\d+)", re.IGNORECASE)
PATH_TOKEN_RE = re.compile(r"^(?:configs|scripts|docs|experiments|robot_sf|tests)/.+")
PATH_KEY_RE = re.compile(
    r"(^path$|_path$|_config$|_manifest$|_matrix$|_script$|^config$|^launcher$|^script$)"
)
PLACEHOLDER_RE = re.compile(r"(<[^>]+>|\{[^}]+\}|TODO|TBD)", re.IGNORECASE)
COMMAND_KEYS = (
    "campaign_command",
    "report_command",
    "validation_command",
    "slurm_command_shape",
    "command",
)
CLAIM_KEYS = ("claim_gate", "claim_map_gate", "execution_boundary", "no_claim_statement")
SEED_KEYS = ("seed_budget", "seed_policy", "seeds")


@dataclass(frozen=True, slots=True)
class PathCheck:
    """One checked path and optional checksum."""

    field: str
    path: str
    exists: bool
    sha256_expected: str | None = None
    sha256_observed: str | None = None
    sha256_matches: bool | None = None


def _sha256(path: Path) -> str:
    """Return SHA-256 digest for *path*."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_path(repo_root: Path, raw_path: str) -> Path:
    """Resolve a launch-packet path relative to repository root."""

    path = Path(raw_path)
    return path if path.is_absolute() else repo_root / path


def _looks_like_placeholder(value: str) -> bool:
    """Return whether value contains an unresolved placeholder."""

    return bool(PLACEHOLDER_RE.search(value))


def _looks_like_repo_path(value: str) -> bool:
    """Return whether a scalar string is a repository path worth preflighting."""

    if "\n" in value or _looks_like_placeholder(value):
        return False
    return bool(PATH_TOKEN_RE.match(value.strip()))


def _iter_mappings(payload: Any, prefix: str = "") -> Iterable[tuple[str, dict[str, Any]]]:
    """Yield nested mappings with dotted prefixes."""

    if isinstance(payload, dict):
        yield prefix, payload
        for key, value in payload.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from _iter_mappings(value, next_prefix)
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            yield from _iter_mappings(value, f"{prefix}[{index}]")


def _collect_checksum_pairs(payload: dict[str, Any]) -> dict[str, tuple[str, str]]:
    """Collect path -> expected sha256 declarations from common packet shapes."""

    checks: dict[str, tuple[str, str]] = {}
    for prefix, mapping in _iter_mappings(payload):
        raw_checksums = mapping.get("checksums")
        if isinstance(raw_checksums, dict):
            for raw_path, raw_sha in raw_checksums.items():
                if isinstance(raw_path, str) and isinstance(raw_sha, str):
                    field = f"{prefix}.checksums.{raw_path}" if prefix else f"checksums.{raw_path}"
                    checks[field] = (raw_path, raw_sha.strip())
        for key, value in mapping.items():
            key_text = str(key)
            if not key_text.endswith("_sha256") or not isinstance(value, str):
                continue
            base_key = key_text[: -len("_sha256")]
            raw_path = mapping.get(base_key)
            if isinstance(raw_path, str):
                field = f"{prefix}.{base_key}" if prefix else base_key
                checks[field] = (raw_path, value.strip())
    return checks


def _collect_path_refs(payload: Any, *, prefix: str = "") -> dict[str, str]:
    """Collect likely repository path references from keys and command tokens."""

    refs: dict[str, str] = {}
    if isinstance(payload, dict):
        for key, value in payload.items():
            key_text = str(key)
            field = f"{prefix}.{key_text}" if prefix else key_text
            if isinstance(value, str):
                if PATH_KEY_RE.search(key_text) and _looks_like_repo_path(value):
                    refs[field] = value.strip()
                if key_text in COMMAND_KEYS:
                    refs.update(_command_path_refs(value, field=field))
            elif isinstance(value, (dict, list)):
                refs.update(_collect_path_refs(value, prefix=field))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            refs.update(_collect_path_refs(value, prefix=f"{prefix}[{index}]"))
    return refs


def _command_path_refs(command: str, *, field: str) -> dict[str, str]:
    """Extract repository path tokens from a shell-ish command string."""

    refs: dict[str, str] = {}
    try:
        tokens = shlex.split(command, comments=True, posix=True)
    except ValueError:
        tokens = command.split()
    output_flags = {"--output", "--output-dir", "--output-root", "--report-dir", "--artifact-dir"}
    for index, token in enumerate(tokens):
        previous = tokens[index - 1] if index > 0 else ""
        if previous in output_flags:
            continue
        clean = token.strip().strip("'\"").rstrip(",.:;)]}")
        if _looks_like_repo_path(clean):
            refs[f"{field}.token[{index}]"] = clean
    return refs


def _check_path(  # noqa: C901
    repo_root: Path, *, field: str, path_text: str, expected_sha: str | None = None
) -> tuple[PathCheck, list[str]]:
    """Check one path and optional SHA-256."""

    reasons: list[str] = []
    exists = False
    observed: str | None = None
    matches: bool | None = None
    if _looks_like_placeholder(path_text):
        reasons.append(f"{field}: path contains placeholder: {path_text}")
    else:
        path = _repo_path(repo_root, path_text)
        if path.is_symlink():
            reasons.append(f"{field}: path is a symlink and is not allowed: {path_text}")
            path = path.resolve(strict=False)
        resolved_path = path.resolve(strict=False)
        if not resolved_path.is_relative_to(repo_root):
            reasons.append(f"{field}: path escapes repository root: {path_text}")
        else:
            exists = resolved_path.is_file()
            if not exists:
                reasons.append(f"{field}: path is not an existing file: {path_text}")
        if expected_sha is not None:
            if not SHA256_RE.fullmatch(expected_sha):
                reasons.append(f"{field}: sha256 is not a 64-character hex digest")
                matches = False
            elif exists:
                try:
                    observed = _sha256(resolved_path)
                except OSError as exc:
                    reasons.append(f"{field}: cannot read path for sha256: {exc}")
                    matches = False
                else:
                    matches = observed == expected_sha.lower()
                    if not matches:
                        reasons.append(
                            f"{field}: sha256 mismatch expected {expected_sha.lower()} observed {observed}"
                        )
            elif not matches:
                matches = False
    return (
        PathCheck(
            field=field,
            path=path_text,
            exists=exists,
            sha256_expected=expected_sha,
            sha256_observed=observed,
            sha256_matches=matches,
        ),
        reasons,
    )


def _extract_issue(packet_path: Path, payload: dict[str, Any]) -> int | None:
    """Extract issue number from payload or filename."""

    issue = payload.get("issue")
    if isinstance(issue, int):
        return issue
    if isinstance(issue, str) and issue.isdigit():
        return int(issue)
    match = ISSUE_RE.search(packet_path.name)
    if match:
        return int(next(group for group in match.groups() if group))
    return None


def _first_command(payload: dict[str, Any]) -> str | None:
    """Return first recognized command string."""

    for key in COMMAND_KEYS:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _seed_budget(payload: dict[str, Any]) -> Any:
    """Return declared seed budget surface if present."""

    for key in SEED_KEYS:
        if key in payload:
            return payload[key]
    return None


def preflight_launch_packet(  # noqa: C901
    packet_path: Path, *, repo_root: Path | None = None
) -> dict[str, Any]:
    """Preflight one launch packet and return a machine-readable report."""

    repo_root = (repo_root or Path.cwd()).resolve()
    packet_path = packet_path if packet_path.is_absolute() else repo_root / packet_path
    reasons: list[str] = []
    path_checks: list[PathCheck] = []

    try:
        payload = yaml.safe_load(packet_path.read_text(encoding="utf-8")) or {}
    except OSError as exc:
        return {
            "schema_version": SCHEMA_VERSION,
            "packet_path": str(packet_path),
            "ready": False,
            "reasons": [f"cannot read packet: {exc}"],
        }
    except yaml.YAMLError as exc:
        return {
            "schema_version": SCHEMA_VERSION,
            "packet_path": str(packet_path),
            "ready": False,
            "reasons": [f"invalid YAML: {exc}"],
        }
    if not isinstance(payload, dict):
        reasons.append("packet YAML must be a mapping")
        payload = {}

    packet_rel = (
        packet_path.relative_to(repo_root) if packet_path.is_relative_to(repo_root) else packet_path
    )
    schema = payload.get("schema_version")
    if not isinstance(schema, str) or not schema.strip():
        reasons.append("missing schema_version")
    issue = _extract_issue(packet_path, payload)
    if issue is None:
        reasons.append("missing issue number")
    command = _first_command(payload)
    if command is None:
        reasons.append("missing launch/validation command")
    if not any(key in payload for key in CLAIM_KEYS):
        reasons.append("missing claim gate or execution boundary")

    path_refs = _collect_path_refs(payload)
    checksum_pairs = _collect_checksum_pairs(payload)
    for field, (path_text, sha) in checksum_pairs.items():
        path_refs.pop(field, None)
        check, check_reasons = _check_path(
            repo_root, field=field, path_text=path_text, expected_sha=sha
        )
        path_checks.append(check)
        reasons.extend(check_reasons)
    for field, path_text in path_refs.items():
        check, check_reasons = _check_path(repo_root, field=field, path_text=path_text)
        path_checks.append(check)
        reasons.extend(check_reasons)

    placeholders = [
        field
        for field, value in _collect_scalar_strings(payload).items()
        if _looks_like_placeholder(value)
    ]
    if placeholders:
        reasons.append(f"placeholder values remain: {', '.join(placeholders[:8])}")

    ready = not reasons
    return {
        "schema_version": SCHEMA_VERSION,
        "packet_path": str(packet_rel),
        "packet_schema_version": schema,
        "issue": issue,
        "kind": _packet_kind(packet_path),
        "ready": ready,
        "reasons": reasons,
        "configs": [asdict(check) for check in path_checks],
        "seed_budget": _seed_budget(payload),
        "command": command,
        "claim_gate": payload.get("claim_gate")
        or payload.get("claim_map_gate")
        or payload.get("execution_boundary"),
    }


def _packet_kind(packet_path: Path) -> str:
    """Infer packet kind from path."""

    parts = set(packet_path.parts)
    if "benchmarks" in parts:
        return "benchmark"
    if "training" in parts:
        return "training"
    return "launch_packet"


def _collect_scalar_strings(payload: Any, *, prefix: str = "") -> dict[str, str]:
    """Collect scalar strings for placeholder detection."""

    out: dict[str, str] = {}
    if isinstance(payload, dict):
        for key, value in payload.items():
            field = f"{prefix}.{key}" if prefix else str(key)
            out.update(_collect_scalar_strings(value, prefix=field))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            out.update(_collect_scalar_strings(value, prefix=f"{prefix}[{index}]"))
    elif isinstance(payload, str):
        out[prefix] = payload
    return out


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", required=True, type=Path, help="Launch-packet YAML path.")
    parser.add_argument("--repo-root", default=Path.cwd(), type=Path, help="Repository root.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""

    args = _parse_args(argv)
    report = preflight_launch_packet(args.packet, repo_root=args.repo_root)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report.get("ready") else 2


if __name__ == "__main__":
    raise SystemExit(main())
