"""Versioned result-job durability gate contract.

A completed result job that wants to admit an ``analysis:`` successor issue MUST
publish, before that successor is created or admitted:

1. a checksummed raw or sufficient-derived input (private-safe pointer if the
   rows must stay private);
2. a versioned schema for that input;
3. a canonical rerun command that regenerates the analysis from it; and
4. a durable public-safe pointer (tracked path or registry entry).

This module defines the ``result_job_durability.v1`` manifest and a probe that
verifies all four properties hold on a clean checkout. It is the producer-side
counterpart to the orchestrator's admission discipline
(ll7/codex-orchestrator#129): a result job fails closed here, so it cannot emit a
phantom-ready successor whose decisive input lives only on a compute host.

Motivating issue: #5936 (instances #5890, #5891, #5912).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator

from robot_sf.common.json_pointer import json_pointer

RESULT_JOB_DURABILITY_SCHEMA_VERSION = "result_job_durability.v1"
RESULT_JOB_DURABILITY_SCHEMA_FILE = (
    Path(__file__).with_name("schemas") / "result_job_durability.v1.json"
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

#: Path prefixes that name disposable local state. A pointer rooted here is not
#: durable on a clean checkout and is rejected unless paired with a tracked or
#: release pointer. Mirrors the artifact_catalog durability boundary.
_LOCAL_ONLY_PREFIXES = (
    "output/",
    "results/",
    ".git/",
    ".venv/",
    "/tmp/",
    "/var/tmp/",
    "/home/",
)

#: pointer_kind values that name a durable pointer but require a hydration step
#: before the input can be checksummed on a clean checkout.
_HYDRATION_REQUIRED_POINTER_KINDS = ("registry_entry", "release_artifact")


@dataclass(frozen=True, slots=True)
class DurabilityIssue:
    """One durability-gate validation issue."""

    gate: str
    path: str
    message: str
    remedy: str | None = None


@dataclass(frozen=True, slots=True)
class DurabilityVerdict:
    """Structured probe verdict for the durability gate.

    Attributes:
        ok: True when every gate property holds and the manifest is durable.
        gate_id: Stable manifest identifier.
        gate_results: Per-gate pass/fail flags.
        issues: All validation issues (empty when ok).
    """

    ok: bool
    gate_id: str | None
    gate_results: dict[str, bool]
    issues: tuple[DurabilityIssue, ...]

    def to_dict(self) -> dict[str, Any]:
        """Convert the verdict to JSON-safe primitives.

        Returns:
            Dictionary representation of the verdict.
        """

        return {
            "ok": self.ok,
            "gate_id": self.gate_id,
            "gate_results": dict(self.gate_results),
            "issues": [asdict(issue) for issue in self.issues],
        }


class ResultJobDurabilityError(ValueError):
    """Raised when a result-job durability manifest fails validation."""

    def __init__(self, issues: list[DurabilityIssue], *, source: str | Path | None = None):
        """Build an actionable validation error from durability issues."""

        self.issues = tuple(issues)
        self.source = str(source) if source is not None else None
        prefix = f"{self.source}: " if self.source else ""
        super().__init__(
            prefix
            + "; ".join(
                f"{issue.gate}/{issue.path}: {issue.message}"
                + (f" (remedy: {issue.remedy})" if issue.remedy else "")
                for issue in issues
            )
        )


def load_result_job_durability_schema() -> dict[str, Any]:
    """Load the public ``result_job_durability.v1`` JSON Schema.

    Returns:
        Parsed JSON Schema dictionary.
    """

    return json.loads(RESULT_JOB_DURABILITY_SCHEMA_FILE.read_text(encoding="utf-8"))


def load_result_job_durability(path: Path) -> DurabilityVerdict:
    """Load a durability manifest and return its probe verdict.

    Returns:
        Structured durability verdict (use ``verdict.ok``).
    """

    try:
        text = path.read_text(encoding="utf-8")
        payload = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        return DurabilityVerdict(
            ok=False,
            gate_id=None,
            gate_results=_empty_gate_results(),
            issues=(
                DurabilityIssue(
                    gate="manifest",
                    path="/",
                    message=f"failed to load manifest: {exc}",
                    remedy="provide a readable result_job_durability.v1 YAML/JSON manifest",
                ),
            ),
        )
    if not isinstance(payload, Mapping):
        return DurabilityVerdict(
            ok=False,
            gate_id=None,
            gate_results=_empty_gate_results(),
            issues=(
                DurabilityIssue(
                    gate="manifest",
                    path="/",
                    message="expected a mapping payload",
                    remedy="provide a result_job_durability.v1 mapping",
                ),
            ),
        )
    return result_job_durability_from_dict(payload, manifest_path=path)


def result_job_durability_from_dict(
    payload: Mapping[str, Any],
    *,
    manifest_path: Path,
) -> DurabilityVerdict:
    """Validate and probe a durability manifest mapping.

    Returns:
        Structured durability verdict. Empty issues + all gate_results True means durable.
    """

    issues = _schema_validation_issues(payload)
    issues.extend(_semantic_validation_issues(payload, manifest_path=manifest_path))
    gate_id = payload.get("gate_id") if isinstance(payload.get("gate_id"), str) else None
    gate_results = _gate_results_from_issues(issues)
    return DurabilityVerdict(
        ok=not issues,
        gate_id=gate_id,
        gate_results=gate_results,
        issues=tuple(issues),
    )


def _empty_gate_results() -> dict[str, bool]:
    """Return all-gates-unknown flags for an unloadable manifest.

    Returns:
        Mapping of each gate name to ``False``.
    """

    return dict.fromkeys(_GATE_ORDER, False)


#: Order the four gate properties are surfaced, matching the issue statement.
_GATE_ORDER = ("checksum", "schema", "rerun_command", "durable_pointer")


def _gate_results_from_issues(issues: list[DurabilityIssue]) -> dict[str, bool]:
    """Reduce validation issues to per-gate pass/fail flags.

    Returns:
        Mapping of each gate name to ``True`` when it has no failing issue.
    """

    failed = {issue.gate for issue in issues if issue.gate in _GATE_ORDER}
    return {gate: gate not in failed for gate in _GATE_ORDER}


def _schema_validation_issues(payload: Mapping[str, Any]) -> list[DurabilityIssue]:
    """Return JSON Schema validation issues mapped onto the four gates."""

    validator = Draft202012Validator(load_result_job_durability_schema())
    mapped: list[DurabilityIssue] = []
    for error in sorted(validator.iter_errors(payload), key=lambda err: list(err.absolute_path)):
        gate, path = _schema_error_to_gate(error)
        mapped.append(
            DurabilityIssue(
                gate=gate,
                path=path,
                message=error.message,
                remedy=_remedy_for_gate(gate),
            )
        )
    return mapped


_FIELD_TO_GATE = {
    "sha256": "checksum",
    "input_schema": "schema",
    "rerun_command": "rerun_command",
    "analysis_input": "durable_pointer",
}


def _schema_error_to_gate(error: Any) -> tuple[str, str]:
    """Map a JSON Schema error onto a gate name and a JSON pointer.

    A ``required`` error is reported at the document root with an empty path,
    so the missing field name is recovered from the error message and mapped
    onto its owning gate rather than the generic ``manifest`` gate.

    Returns:
        Tuple of ``(gate_name, json_pointer)``.
    """

    pointer = json_pointer(error.absolute_path)
    parts = [str(part) for part in error.absolute_path]
    for field, gate in _FIELD_TO_GATE.items():
        if field in parts:
            return gate, pointer
    message = str(getattr(error, "message", ""))
    required_match = re.search(r"'([a-z_]+)' is a required property", message)
    if required_match:
        gate = _FIELD_TO_GATE.get(required_match.group(1))
        if gate is not None:
            return gate, f"/{required_match.group(1)}"
    return "manifest", pointer


def _remedy_for_gate(gate: str) -> str | None:
    """Return the concrete fix a producer should apply for a failed gate."""

    return {
        "checksum": "publish the analysis input and record its sha256",
        "schema": "declare a versioned input schema that resolves on a clean checkout",
        "rerun_command": "declare the canonical command that regenerates the analysis from the input",
        "durable_pointer": (
            "publish the input to a tracked path or registry entry; "
            "local-only (output/, results/, worktree) pointers are not durable"
        ),
        "manifest": "fix the result_job_durability.v1 manifest structure",
    }.get(gate)


def _semantic_validation_issues(
    payload: Mapping[str, Any],
    *,
    manifest_path: Path,
) -> list[DurabilityIssue]:
    """Return cross-field and filesystem probe issues."""

    issues: list[DurabilityIssue] = []
    if not isinstance(payload.get("analysis_input"), Mapping):
        return issues
    analysis_input = payload["analysis_input"]
    pointer_text = analysis_input.get("pointer")
    pointer_kind = analysis_input.get("pointer_kind")
    private_safe = bool(analysis_input.get("private_safe", False))

    issues.extend(
        _probe_durable_pointer(
            pointer_text, pointer_kind, private_safe, manifest_path=manifest_path
        )
    )

    sha_value = payload.get("sha256")
    if isinstance(sha_value, str) and _SHA256_RE.fullmatch(sha_value.strip()):
        issues.extend(_probe_checksum(pointer_text, pointer_kind, sha_value, manifest_path))

    if isinstance(payload.get("input_schema"), Mapping):
        issues.extend(_probe_input_schema(payload["input_schema"], manifest_path))

    if isinstance(payload.get("rerun_command"), str):
        issues.extend(_probe_rerun_command(payload["rerun_command"], pointer_text))

    return issues


def _probe_durable_pointer(
    pointer_text: Any,
    pointer_kind: Any,
    private_safe: bool,
    *,
    manifest_path: Path,
) -> list[DurabilityIssue]:
    """Verify the analysis-input pointer is durable and resolvable.

    Returns:
        Durability issues for the durable-pointer gate (empty when durable).
    """

    issues: list[DurabilityIssue] = []
    if not isinstance(pointer_text, str) or not pointer_text.strip():
        return issues
    path_text = pointer_text.strip()
    pointer = "/analysis_input/pointer"

    if _is_local_only_path(path_text):
        issues.append(
            DurabilityIssue(
                gate="durable_pointer",
                path=pointer,
                message=(
                    f"local-only analysis input pointer is not durable: {path_text}"
                    + (" (raw rows must stay private)" if private_safe else "")
                ),
                remedy=_remedy_for_gate("durable_pointer"),
            )
        )
        return issues

    if pointer_kind in _HYDRATION_REQUIRED_POINTER_KINDS:
        # A registry/release pointer is durable by reference; require a hydration
        # command but do not insist the artifact is checked into the repository.
        return issues

    if pointer_kind == "tracked_path":
        resolved = _resolve_manifest_path(manifest_path, path_text)
        if not resolved.exists():
            issues.append(
                DurabilityIssue(
                    gate="durable_pointer",
                    path=pointer,
                    message=f"tracked analysis input does not resolve on clean checkout: {path_text}",
                    remedy="promote the artifact to the tracked path or switch pointer_kind to registry_entry/release_artifact",
                )
            )
        elif not resolved.is_file():
            issues.append(
                DurabilityIssue(
                    gate="durable_pointer",
                    path=pointer,
                    message=f"tracked analysis input is not a file: {path_text}",
                )
            )
        return issues

    return issues


def _probe_checksum(
    pointer_text: Any,
    pointer_kind: Any,
    sha_value: str,
    manifest_path: Path,
) -> list[DurabilityIssue]:
    """Verify the declared checksum matches the resolved artifact.

    Returns:
        Durability issues for the checksum gate (empty when the digest matches).
    """

    issues: list[DurabilityIssue] = []
    if not isinstance(pointer_text, str) or not pointer_text.strip():
        return issues
    path_text = pointer_text.strip()

    if pointer_kind != "tracked_path":
        # registry_entry/release_artifact inputs are checksummed out-of-band at
        # hydration time; the gate trusts the declared digest here and fails at
        # hydration if it diverges. We still confirm the digest is well-formed
        # (the schema layer enforces shape; this is a defensive no-op).
        return issues

    resolved = _resolve_manifest_path(manifest_path, path_text)
    if not resolved.is_file():
        return issues  # surfaced by the durable_pointer probe

    actual_sha = sha256_file(resolved)
    if actual_sha != sha_value.strip():
        issues.append(
            DurabilityIssue(
                gate="checksum",
                path="/sha256",
                message=(
                    f"checksum mismatch for {path_text}: expected {sha_value.strip()}, "
                    f"got {actual_sha}"
                ),
                remedy="re-publish the analysis input and update sha256 to match",
            )
        )
    return issues


def _probe_input_schema(
    input_schema: Mapping[str, Any],
    manifest_path: Path,
) -> list[DurabilityIssue]:
    """Verify the versioned input schema is declared and resolvable when given.

    Returns:
        Durability issues for the schema gate (empty when declared and resolvable).
    """

    issues: list[DurabilityIssue] = []
    schema_path = input_schema.get("schema_path")
    if not isinstance(schema_path, str) or not schema_path.strip():
        return issues  # schema_name+schema_version is sufficient; path is optional
    path_text = schema_path.strip()
    if _is_local_only_path(path_text):
        issues.append(
            DurabilityIssue(
                gate="schema",
                path="/input_schema/schema_path",
                message=f"input schema path is not durable: {path_text}",
                remedy="point schema_path at a tracked repository schema",
            )
        )
        return issues
    resolved = _resolve_manifest_path(manifest_path, path_text)
    if not resolved.exists():
        issues.append(
            DurabilityIssue(
                gate="schema",
                path="/input_schema/schema_path",
                message=f"input schema does not resolve on clean checkout: {path_text}",
                remedy="commit the schema or drop schema_path and rely on schema_name+schema_version",
            )
        )
    return issues


def _probe_rerun_command(rerun_command: str, pointer_text: Any) -> list[DurabilityIssue]:
    """Verify the canonical rerun command is present and not a host-only stub.

    Returns:
        Durability issues for the rerun-command gate (empty when reproducible).
    """

    issues: list[DurabilityIssue] = []
    command = rerun_command.strip()
    if not command:
        return issues
    # A rerun command that names the raw host-only artifact as its sole input
    # cannot reproduce the analysis from the durable pointer alone.
    if isinstance(pointer_text, str) and _is_local_only_path(pointer_text.strip()):
        issues.append(
            DurabilityIssue(
                gate="rerun_command",
                path="/rerun_command",
                message=(
                    "rerun_command cannot reproduce the analysis from a local-only input pointer; "
                    "hydrate from the durable pointer first"
                ),
                remedy="make the rerun command consume the durable pointer or its hydration target",
            )
        )
    return issues


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest for a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_manifest_path(manifest_path: Path, path_text: str) -> Path:
    """Resolve a manifest reference relative to manifest dir or repository root.

    Returns:
        Absolute resolved path.
    """

    manifest_relative = (manifest_path.parent / path_text).resolve()
    if manifest_relative.exists():
        return manifest_relative
    return (_repo_root_for(manifest_path) / path_text).resolve()


def _repo_root_for(path: Path) -> Path:
    """Return the nearest repository root, falling back to the manifest dir."""

    resolved = path.resolve()
    for parent in (resolved.parent, *resolved.parents):
        if (parent / ".git").exists():
            return parent
    return resolved.parent


def _is_local_only_path(value: str) -> bool:
    """Return whether a path points at disposable local state.

    Mirrors artifact_catalog._is_local_only_path so the durability boundary is
    identical across the two surfaces.
    """

    path = Path(value.strip())
    parts = path.parts
    local_roots = {prefix.strip("/") for prefix in _LOCAL_ONLY_PREFIXES}
    local_roots.discard("")
    if path.is_absolute():
        return len(parts) > 1 and parts[1] in local_roots
    return bool(parts) and (parts[0] in local_roots or any(".worktrees" in part for part in parts))


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the durability-gate probe parser.

    Returns:
        Configured argument parser.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Probe a result_job_durability.v1 manifest. A result job MUST pass "
            "before any analysis: successor issue is created or admitted."
        ),
    )
    parser.add_argument("manifest", type=Path, help="result_job_durability.v1 YAML/JSON path.")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a JSON probe verdict instead of a human-readable report.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Probe one durability manifest and return a shell-friendly exit code.

    Returns:
        ``0`` when the manifest is durable, otherwise ``2``.
    """

    args = build_arg_parser().parse_args(argv)
    verdict = load_result_job_durability(args.manifest)
    report = verdict.to_dict()
    if args.json:
        sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    else:
        gate_id = report["gate_id"] or "<unknown>"
        status = "DURABLE" if report["ok"] else "NOT DURABLE"
        sys.stdout.write(f"result-job durability gate [{gate_id}]: {status}\n")
        for gate, passed in report["gate_results"].items():
            sys.stdout.write(f"  {gate}: {'pass' if passed else 'fail'}\n")
        for issue in report["issues"]:
            remedy = f"  -> remedy: {issue['remedy']}" if issue.get("remedy") else ""
            sys.stdout.write(f"  {issue['gate']}/{issue['path']}: {issue['message']}{remedy}\n")
    return 0 if report["ok"] else 2


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
