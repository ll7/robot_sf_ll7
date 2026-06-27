#!/usr/bin/env python3
"""Re-export readiness preflight for stale dissertation table bundles.

Issue #3203 motivation: a stale dissertation export bundle (for example the
scenario-horizon tables) must never be silently treated as current evidence
just because a manifest exists. Before anyone re-runs the bounded campaign that
regenerates the tables, an operator needs a conservative, read-only answer to
two questions:

1. *Does this bundle even need a re-export?* (Is the payload present and do its
   checksums still match?)
2. *If a re-export is needed, can it be reproduced here?* (Are the required
   inputs -- the campaign config and the generation script -- actually present
   in this checkout, and is the source-commit provenance recorded?)

This preflight composes the existing freshness classifier in
``scripts/tools/stale_artifact_detector.py`` (the canonical owner for artifact
freshness) with a *required-input availability* check, and reduces the two
signals to a single readiness state:

* ``fresh`` -- payload present and checksums match; no re-export needed.
* ``stale`` -- a re-export is needed and every required input is present, so the
  bounded campaign can be reproduced here (re-export is unblocked).
* ``blocked`` -- a re-export is needed but at least one required input is missing
  or the provenance cannot be reconstructed, so the re-export must be reported
  as blocked rather than run against unverifiable provenance.

This tool is intentionally read-only. It does **not** run the campaign,
regenerate any table, or edit dissertation claims. It only reports status and
the concrete inputs a re-export would require.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from scripts.tools import stale_artifact_detector as detector

try:
    import yaml
except ImportError:  # pragma: no cover - dependency is present in repo env
    yaml = None

SCHEMA_VERSION = "reexport_readiness_preflight.v1"

# Exceptions that mean a manifest could not be loaded/parsed. ``json.JSONDecodeError``
# is a ``ValueError`` subclass; ``yaml.YAMLError`` is added when PyYAML is available.
_LOAD_ERRORS: tuple[type[Exception], ...] = (OSError, ValueError)
if yaml is not None:  # pragma: no branch - PyYAML is present in the repo env
    _LOAD_ERRORS = (*_LOAD_ERRORS, yaml.YAMLError)

# Freshness states that mean the bundle payload is safe to cite as-is and does
# not require a re-export. Everything else implies a re-export is needed.
_FRESH_STATES = frozenset(
    {
        detector.ArtifactState.CURRENT,
        detector.ArtifactState.HISTORICAL_VALID,
    }
)


class ReexportReadiness(StrEnum):
    """Operator-facing re-export readiness for a dissertation table bundle."""

    FRESH = "fresh"
    STALE = "stale"
    BLOCKED = "blocked"


@dataclass(frozen=True, slots=True)
class RequiredInput:
    """One input a re-export needs, with its resolved availability."""

    name: str
    kind: str
    value: str | None
    present: bool
    detail: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view."""
        return {
            "name": self.name,
            "kind": self.kind,
            "value": self.value,
            "present": self.present,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class BundleReadinessReport:
    """Readiness assessment for one bundle manifest."""

    manifest_path: str
    state: ReexportReadiness
    reexport_needed: bool
    reasons: list[str]
    required_inputs: list[RequiredInput]
    freshness: dict[str, Any]
    generated_at_utc: str
    schema: str = SCHEMA_VERSION

    @property
    def missing_inputs(self) -> list[RequiredInput]:
        """Return required inputs that are not present in this checkout."""
        return [item for item in self.required_inputs if not item.present]

    @property
    def exit_code(self) -> int:
        """Return non-zero only when a re-export is blocked.

        A ``stale`` bundle is an actionable-but-unblocked state (the operator can
        re-run the bounded campaign), so it is not treated as a hard failure. A
        ``blocked`` bundle is a hard failure because the re-export cannot be
        reproduced from the recorded provenance.
        """
        return int(self.state is ReexportReadiness.BLOCKED)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable report."""
        return {
            "schema": self.schema,
            "generated_at_utc": self.generated_at_utc,
            "manifest_path": self.manifest_path,
            "state": self.state.value,
            "reexport_needed": self.reexport_needed,
            "reasons": self.reasons,
            "required_inputs": [item.to_dict() for item in self.required_inputs],
            "missing_inputs": [item.name for item in self.missing_inputs],
            "freshness": self.freshness,
        }


def _load_raw_payload(manifest_path: Path) -> Any:
    """Load the raw manifest payload (JSON or YAML) without flattening."""
    text = manifest_path.read_text(encoding="utf-8")
    if manifest_path.suffix.lower() == ".json":
        return json.loads(text)
    if yaml is None:  # pragma: no cover - dependency is present in repo env
        raise RuntimeError("PyYAML is required for YAML manifests")
    return yaml.safe_load(text)


def _iter_entries(payload: Any) -> list[dict[str, Any]]:
    """Return candidate entries from a manifest payload for metadata scraping."""
    entries: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        entries.append(payload)
        artifacts = payload.get("artifacts")
        if isinstance(artifacts, list):
            entries.extend(item for item in artifacts if isinstance(item, dict))
    elif isinstance(payload, list):
        entries.extend(item for item in payload if isinstance(item, dict))
    return entries


def _first_field(entries: list[dict[str, Any]], *names: str) -> str | None:
    """Return the first non-empty string value for any of ``names``."""
    for entry in entries:
        for name in names:
            value = entry.get(name)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _extract_option_value(command: str, *flags: str) -> str | None:
    """Return the value following any of ``flags`` in a shell command string."""
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()
    for index, token in enumerate(tokens):
        for flag in flags:
            if token == flag and index + 1 < len(tokens):
                return tokens[index + 1]
            prefix = f"{flag}="
            if token.startswith(prefix):
                return token[len(prefix) :]
    return None


def _extract_script_path(command: str) -> str | None:
    """Return the first ``*.py`` script path token in a command string.

    Handles both direct ``python scripts/foo.py`` and ``python -m pkg.mod``
    forms; for ``-m`` invocations the module path is returned as a dotted
    reference rather than a filesystem path.
    """
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()
    for index, token in enumerate(tokens):
        if token == "-m" and index + 1 < len(tokens):
            return tokens[index + 1]
        if token.endswith(".py") and not token.startswith("-"):
            return token
    return None


def _resolve_under_root(path_text: str, repo_root: Path) -> Path:
    """Resolve a possibly-relative manifest path against ``repo_root``."""
    candidate = Path(path_text)
    if candidate.is_absolute():
        return candidate
    return repo_root / candidate


_PLACEHOLDER_RE = re.compile(r"<[^>]+>")


def _looks_like_placeholder(value: str) -> bool:
    """Return True when a recorded path is a placeholder, not a real input."""
    return bool(_PLACEHOLDER_RE.search(value))


def _required_inputs(
    entries: list[dict[str, Any]],
    *,
    repo_root: Path,
) -> list[RequiredInput]:
    """Derive the concrete inputs a re-export needs and check their presence."""
    inputs: list[RequiredInput] = []

    command = _first_field(entries, "generation_command")
    source_commit = _first_field(entries, "source_commit")

    # Campaign config: the bounded campaign cannot be reproduced without it.
    config_value = _extract_option_value(command, "--config") if command else None
    if config_value is None:
        inputs.append(
            RequiredInput(
                name="campaign_config",
                kind="config",
                value=None,
                present=False,
                detail="no --config recorded in generation_command",
            )
        )
    elif _looks_like_placeholder(config_value):
        inputs.append(
            RequiredInput(
                name="campaign_config",
                kind="config",
                value=config_value,
                present=False,
                detail="recorded config path is a placeholder",
            )
        )
    else:
        config_path = _resolve_under_root(config_value, repo_root)
        present = config_path.is_file()
        inputs.append(
            RequiredInput(
                name="campaign_config",
                kind="config",
                value=config_value,
                present=present,
                detail="config file present" if present else "config file missing",
            )
        )

    # Generation script: the entry point that produces the tables.
    script_value = _extract_script_path(command) if command else None
    if script_value is None:
        inputs.append(
            RequiredInput(
                name="generation_script",
                kind="script",
                value=None,
                present=False,
                detail="no generation script found in generation_command",
            )
        )
    elif script_value.endswith(".py"):
        script_path = _resolve_under_root(script_value, repo_root)
        present = script_path.is_file()
        inputs.append(
            RequiredInput(
                name="generation_script",
                kind="script",
                value=script_value,
                present=present,
                detail="script present" if present else "script missing",
            )
        )
    else:
        # A ``-m`` module reference; treat as present-as-recorded provenance.
        inputs.append(
            RequiredInput(
                name="generation_script",
                kind="module",
                value=script_value,
                present=True,
                detail="module reference recorded",
            )
        )

    # Source commit: provenance anchor for a reproducible re-run.
    inputs.append(
        RequiredInput(
            name="source_commit",
            kind="provenance",
            value=source_commit,
            present=source_commit is not None,
            detail="source_commit recorded"
            if source_commit is not None
            else "no source_commit recorded",
        )
    )

    return inputs


def assess_bundle(
    manifest_path: Path,
    *,
    repo_root: Path | None = None,
    current_schema: str = detector.CURRENT_ARTIFACT_SCHEMA,
) -> BundleReadinessReport:
    """Assess re-export readiness for one bundle manifest.

    Composes the canonical freshness classifier with a required-input
    availability check and reduces both to a single readiness state.
    """
    repo_root = repo_root or manifest_path.resolve().parent
    reasons: list[str] = []

    try:
        payload = _load_raw_payload(manifest_path)
        entries = _iter_entries(payload)
    except _LOAD_ERRORS as exc:  # malformed manifest -> cannot reproduce
        return BundleReadinessReport(
            manifest_path=str(manifest_path),
            state=ReexportReadiness.BLOCKED,
            reexport_needed=True,
            reasons=[f"cannot load manifest: {exc}"],
            required_inputs=[],
            freshness={"error": str(exc)},
            generated_at_utc=datetime.now(UTC).isoformat(),
        )

    freshness_report = detector.scan_manifest(manifest_path, current_schema=current_schema)
    fresh = bool(freshness_report.results) and all(
        result.state in _FRESH_STATES for result in freshness_report.results
    )
    reexport_needed = not fresh

    required_inputs = _required_inputs(entries, repo_root=repo_root)
    missing = [item for item in required_inputs if not item.present]

    summary_counts = {state.value: 0 for state in detector.ArtifactState}
    for result in freshness_report.results:
        summary_counts[result.state.value] += 1
    freshness_summary = {
        "summary": summary_counts,
        "results": [
            {"artifact_id": result.artifact_id, "state": result.state.value}
            for result in freshness_report.results
        ],
    }

    if not reexport_needed:
        reasons.append("payload present and checksums match; no re-export needed")
        state = ReexportReadiness.FRESH
    elif missing:
        reasons.append("re-export needed but required inputs are missing")
        reasons.extend(f"missing {item.name}: {item.detail}" for item in missing)
        state = ReexportReadiness.BLOCKED
    else:
        reasons.append("re-export needed; all required inputs present (re-export unblocked)")
        state = ReexportReadiness.STALE

    return BundleReadinessReport(
        manifest_path=str(manifest_path),
        state=state,
        reexport_needed=reexport_needed,
        reasons=reasons,
        required_inputs=required_inputs,
        freshness=freshness_summary,
        generated_at_utc=datetime.now(UTC).isoformat(),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only re-export readiness preflight for stale dissertation table "
            "bundles. Reports fresh/stale/blocked status and required re-export inputs; "
            "never runs the campaign or edits claims."
        )
    )
    parser.add_argument("manifest", type=Path, help="Bundle manifest JSON/YAML file.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root used to resolve relative input paths (default: manifest dir).",
    )
    parser.add_argument(
        "--current-schema",
        default=detector.CURRENT_ARTIFACT_SCHEMA,
        help="Schema version treated as current for freshness classification.",
    )
    parser.add_argument("--json-out", type=Path, help="Optional JSON report output path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = _build_parser().parse_args(argv)
    report = assess_bundle(
        args.manifest,
        repo_root=args.repo_root,
        current_schema=args.current_schema,
    )
    payload = json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return report.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
