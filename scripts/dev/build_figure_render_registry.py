#!/usr/bin/env python3
"""Canonical figure-render registry builder for issue #6769.

Inventory and classify candidate figure-render / reproduction commands from exactly three
declared committed source classes under ``docs/context/evidence/``:

1. a file named ``build_command*.txt``;
2. a JSON or YAML manifest with a field named exactly ``render_command`` or ``build_command``;
3. a README section already linked by ``docs/context/catalog.yaml`` whose fenced code block is
   explicitly identified (by the nearest preceding non-empty line) as a reproduction or render
   command.

The generator never executes any renderer. It discovers commands, stores each as an argument
vector, classifies recurrence eligibility with deterministic rules, and emits a versioned registry
plus a compact audit. ``--check`` regenerates in memory and exits non-zero on drift without
rewriting files.

The domain-aware approval for this tool is narrow: ``recurrence_eligible`` means only that a
command is safe, deterministic, local, and backed by committed inputs for future recurrence
testing. It is NOT scientific figure eligibility, publication approval, visual correctness, or
benchmark evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from robot_sf.evidence.writers import review_marker_json
from robot_sf.evidence.writers import write_json as write_evidence_json

# evidence-writer-exempt: figure_render_registry.v1.yaml is a versioned workflow artifact under
# docs/context/ (not the docs/context/evidence/ tree, so the evidence write_* helpers do not
# apply to it); the audit JSON is written through the shared write_evidence_json writer.
REGISTRY_VERSION = 1
ISSUE_NUMBER = 6769
REPO_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_ROOT = REPO_ROOT / "docs" / "context" / "evidence"
CATALOG_PATH = REPO_ROOT / "docs" / "context" / "catalog.yaml"
REGISTRY_PATH = REPO_ROOT / "docs" / "context" / "figure_render_registry.v1.yaml"
AUDIT_PATH = EVIDENCE_ROOT / "issue_6769_figure_render_registry_audit.json"

SOURCE_CLASS_BUILD_CMD = "build_command_txt"
SOURCE_CLASS_MANIFEST = "manifest_field"
SOURCE_CLASS_README = "readme_reproduction"

# Nearest preceding non-empty line must match this to qualify a fenced README block.
README_LABEL_RE = re.compile(r"(reproduc|render)", re.IGNORECASE)
PLACEHOLDER_RE = re.compile(r"<[^>]+>")
SHELL_METACHAR_RE = re.compile(r"(&&|\|\||;|\||>|<|`|\$\()")
ENV_ASSIGN_RE = re.compile(r"^[A-Z_][A-Z0-9_]*=")
NETWORK_RE = re.compile(r"(https?://|(^|\s)(curl|wget|scp|rsync|ssh)(\s|$))", re.IGNORECASE)
SLURM_RE = re.compile(r"(^|\s)(sbatch|srun|salloc|squeue)(\s|$)", re.IGNORECASE)

# Flags whose argument value names an output path or root.
OUTPUT_FLAGS = {
    "--output",
    "--out",
    "--output-dir",
    "--out-dir",
    "--output-root",
    "--out-json",
    "--out-md",
    "--out-csv",
    "--out-dir-json",
}

ALLOWED_EXCLUSION_REASONS = frozenset(
    {
        "external_input",
        "requires_slurm",
        "requires_network",
        "missing_committed_fixture",
        "non_deterministic_contract",
        "unsafe_command",
        "historical_only",
        "superseded",
    }
)


def now_iso() -> str:
    """Return the current UTC time as a second-precision ISO-8601 string."""
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def git_head() -> str | None:
    """Return the current repository HEAD commit SHA, or ``None`` if unavailable."""
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, OSError):
        return None


def sha256_of(path: Path) -> str | None:
    """Return the SHA-256 hex digest of a file, or ``None`` if it is not a readable file."""
    full = path if path.is_absolute() else REPO_ROOT / path
    if not full.is_file():
        return None
    digest = hashlib.sha256()
    with full.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class CommandSource:
    """A discovered command source (file, manifest field, or labeled README block)."""

    source_class: str
    source_path: str  # repo-relative
    raw_commands: list[str] = field(default_factory=list)


@dataclass
class Entry:
    """A classified registry entry for a single discovered command."""

    id: str
    source_class: str
    source_path: str
    command_index: int
    command: list[str]
    working_dir: str
    inputs: list[dict[str, str]]
    expected_outputs: list[str]
    timeout_seconds: int
    environment: dict[str, str]
    recurrence_eligible: bool
    exclusion_reason: str | None
    classification_notes: list[str]
    provenance_status: str
    last_verified_commit: str | None


def discover_build_command_txt() -> list[CommandSource]:
    """Discover commands from ``build_command*.txt`` files under the evidence root."""
    sources: list[CommandSource] = []
    for path in sorted(EVIDENCE_ROOT.rglob("build_command*.txt")):
        if not path.is_file():
            continue
        rel = path.relative_to(REPO_ROOT).as_posix()
        text = path.read_text(encoding="utf-8")
        text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
        logical_lines = [
            line for line in text.splitlines() if line.strip() and not line.strip().startswith("#")
        ]
        commands = _group_commands("\n".join(logical_lines))
        sources.append(CommandSource(SOURCE_CLASS_BUILD_CMD, rel, commands))
    return sources


def discover_manifest_fields() -> list[CommandSource]:
    """Discover commands from manifests with an exact ``render_command``/``build_command`` field."""
    sources: list[CommandSource] = []
    candidates = sorted(
        list(EVIDENCE_ROOT.rglob("*.json"))
        + list(EVIDENCE_ROOT.rglob("*.yaml"))
        + list(EVIDENCE_ROOT.rglob("*.yml"))
    )
    for path in candidates:
        if not path.is_file():
            continue
        try:
            if path.suffix == ".json":
                data = json.loads(path.read_text(encoding="utf-8"))
            else:
                data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, yaml.YAMLError, OSError):
            continue
        commands = _extract_named_fields(data)
        if commands:
            rel = path.relative_to(REPO_ROOT).as_posix()
            sources.append(CommandSource(SOURCE_CLASS_MANIFEST, rel, commands))
    return sources


def _extract_named_fields(node: Any) -> list[str]:
    """Recursively collect values of keys named exactly ``render_command``/``build_command``."""
    found: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            if (
                key in ("render_command", "build_command")
                and isinstance(value, str)
                and value.strip()
            ):
                found.append(value.strip())
            else:
                found.extend(_extract_named_fields(value))
    elif isinstance(node, list):
        for item in node:
            found.extend(_extract_named_fields(item))
    return found


def discover_readme_reproduction() -> list[CommandSource]:
    """Discover commands from catalog-linked READMEs with explicitly-labeled fenced blocks."""
    sources: list[CommandSource] = []
    for rel in _catalog_linked_readmes():
        path = REPO_ROOT / rel
        if not path.is_file():
            continue
        commands = _extract_labeled_fenced(path)
        if commands:
            sources.append(CommandSource(SOURCE_CLASS_README, rel, commands))
    return sources


def _catalog_linked_readmes() -> list[str]:
    """Return the sorted, de-duplicated set of README paths linked by the context catalog."""
    try:
        text = CATALOG_PATH.read_text(encoding="utf-8")
    except OSError:
        return []
    return sorted(set(re.findall(r"docs/context/evidence/\S+/README\.md", text)))


def _extract_labeled_fenced(path: Path) -> list[str]:
    """Extract commands from fenced blocks whose nearest preceding line is a render/reproduc label."""
    lines = path.read_text(encoding="utf-8").splitlines()
    commands: list[str] = []
    in_block = False
    block: list[str] = []
    fence_open_idx = None
    for idx, line in enumerate(lines):
        if line.lstrip().startswith("```"):
            if not in_block:
                in_block = True
                block = []
                fence_open_idx = idx
            else:
                probe = fence_open_idx - 1 if fence_open_idx is not None else idx - 1
                while probe >= 0 and not lines[probe].strip():
                    probe -= 1
                label = lines[probe] if probe >= 0 else ""
                if label and README_LABEL_RE.search(label):
                    commands.extend(_group_commands("\n".join(block)))
                in_block = False
                block = []
        elif in_block:
            block.append(line)
    return commands


def _group_commands(text: str) -> list[str]:
    """Group raw text into commands, honoring backslash continuations and blank-line separators.

    Shell-style ``#`` comment lines are dropped.
    """
    groups: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        if not line.strip():
            if current:
                groups.append(" ".join(current))
                current = []
            continue
        if line.lstrip().startswith("#"):
            continue
        if line.rstrip().endswith("\\"):
            current.append(line.rstrip()[:-1].strip())
        else:
            current.append(line.strip())
            groups.append(" ".join(current))
            current = []
    if current:
        groups.append(" ".join(current))
    return groups


def _detect_trigger(raw: str, argv: list[str]) -> tuple[str | None, list[str], dict[str, str]]:
    """Run ordered command-string trigger checks and return (reason, notes, env_prefix).

    Placeholder paths are detected before shell-metacharacter redirection so a closed ``<...>``
    placeholder is classified as a missing fixture rather than redirection.
    """
    notes: list[str] = []
    env_prefix: dict[str, str] = {}
    reason: str | None = None

    while argv and ENV_ASSIGN_RE.match(argv[0]) and "=" in argv[0]:
        key, _, value = argv.pop(0).partition("=")
        env_prefix[key] = value
    if env_prefix:
        notes.append("command uses shell environment-assignment prefix")
        return "unsafe_command", notes, env_prefix

    if PLACEHOLDER_RE.search(raw):
        notes.append("command references a placeholder path")
        return "missing_committed_fixture", notes, env_prefix
    if SHELL_METACHAR_RE.search(raw):
        notes.append("command contains shell metacharacters")
        return "unsafe_command", notes, env_prefix
    if SLURM_RE.search(raw):
        notes.append("command submits to a workload scheduler")
        return "requires_slurm", notes, env_prefix
    if NETWORK_RE.search(raw):
        notes.append("command requires network access")
        return "requires_network", notes, env_prefix
    for token in argv:
        if token.startswith("/") and not token.startswith("//"):
            notes.append("command uses an absolute path argument")
            return "unsafe_command", notes, env_prefix
    return reason, notes, env_prefix


def _collect_inputs_outputs(
    argv: list[str],
) -> tuple[list[dict[str, str]], list[str], list[str], str | None]:
    """Collect committed inputs and declared expected outputs from the argument vector.

    Returns (inputs, expected_outputs, notes, reason) where ``reason`` is set when a referenced
    path is missing or an output targets the untracked ``output/`` root.
    """
    notes: list[str] = []
    inputs: list[dict[str, str]] = []
    expected_outputs: list[str] = []
    reason: str | None = None
    idx = 0
    while idx < len(argv):
        token = argv[idx]
        if token in OUTPUT_FLAGS and idx + 1 < len(argv):
            expected_outputs.append(argv[idx + 1])
            idx += 2
            continue
        candidate = REPO_ROOT / token if not Path(token).is_absolute() else Path(token)
        try:
            rel = candidate.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
        except ValueError:
            rel = None
        if rel is not None and (REPO_ROOT / rel).is_file():
            digest = sha256_of(REPO_ROOT / rel)
            if digest:
                inputs.append({"path": rel, "sha256": digest})
            else:
                notes.append(f"input not readable: {rel}")
                reason = reason or "missing_committed_fixture"
        elif (
            rel is not None
            and not (REPO_ROOT / rel).exists()
            and "/" in token
            and not token.startswith("-")
        ):
            notes.append(f"referenced path missing: {rel}")
            reason = reason or "missing_committed_fixture"
        idx += 1

    for output in expected_outputs:
        if output.startswith("output/") or output == "output":
            notes.append("command writes to the untracked output/ root")
            reason = reason or "missing_committed_fixture"
            break
    return inputs, expected_outputs, notes, reason


def classify(source: CommandSource, command_index: int, raw: str, head: str | None) -> Entry:
    """Classify a single raw command string into a registry entry."""
    notes: list[str] = []
    reason: str | None = None
    try:
        argv = shlex.split(raw, posix=True)
    except ValueError as exc:
        argv = []
        notes.append(f"shlex parse error: {exc}")
        reason = "unsafe_command"

    if reason is None:
        trigger_reason, trigger_notes, env_prefix = _detect_trigger(raw, argv)
        notes.extend(trigger_notes)
        reason = trigger_reason
    else:
        env_prefix = {}

    inputs: list[dict[str, str]] = []
    expected_outputs: list[str] = []
    if reason is None:
        inputs, expected_outputs, io_notes, io_reason = _collect_inputs_outputs(argv)
        notes.extend(io_notes)
        reason = io_reason

    if reason is None and not expected_outputs:
        notes.append("command declares no explicit expected output")
        reason = "non_deterministic_contract"

    eligible = reason is None
    return Entry(
        id=_stable_id(source, command_index, raw),
        source_class=source.source_class,
        source_path=source.source_path,
        command_index=command_index,
        command=argv,
        working_dir=".",
        inputs=inputs,
        expected_outputs=expected_outputs,
        timeout_seconds=600,
        environment=env_prefix,
        recurrence_eligible=eligible,
        exclusion_reason=reason,
        classification_notes=notes,
        provenance_status="discovered",
        last_verified_commit=head,
    )


def _stable_id(source: CommandSource, command_index: int, raw: str) -> str:
    """Derive a stable entry id from the source path, command index, and command digest."""
    base = source.source_path.replace("/", "_").replace(".", "_").strip("_")
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8]
    return f"{base}__cmd{command_index + 1}_{digest}"


def discover_all() -> list[CommandSource]:
    """Discover sources across all three declared source classes."""
    return (
        discover_build_command_txt() + discover_manifest_fields() + discover_readme_reproduction()
    )


def build_registry() -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the registry and audit documents from current discovery rules."""
    head = git_head()
    sources = discover_all()
    entries: list[Entry] = []
    seen: dict[str, str] = {}
    for source in sources:
        for idx, raw in enumerate(source.raw_commands):
            entry = classify(source, idx, raw, head)
            if entry.id in seen:
                raise SystemExit(
                    f"fail-closed: duplicate command identity '{entry.id}' from "
                    f"{entry.source_path} and {seen[entry.id]}"
                )
            seen[entry.id] = entry.source_path
            entries.append(entry)
    entries.sort(key=lambda e: (e.source_class, e.source_path, e.command_index))

    registry = {
        "version": REGISTRY_VERSION,
        "generated_at": now_iso(),
        "generator": "scripts/dev/build_figure_render_registry.py",
        "issue": ISSUE_NUMBER,
        "claim_boundary": (
            "Workflow reproducibility metadata only. No figure, benchmark, or publication "
            "claim is produced. recurrence_eligible means only safe, deterministic, local, "
            "committed-input recurrence candidacy."
        ),
        "provenance": {"source_commit": head, "issue": ISSUE_NUMBER},
        "entries": [_entry_to_dict(e) for e in entries],
    }
    audit = _build_audit(entries, registry["generated_at"], head)
    return registry, audit


def _build_audit(entries: list[Entry], generated_at: str, head: str | None) -> dict[str, Any]:
    """Compile the compact audit summary from classified entries."""
    eligible = [e for e in entries if e.recurrence_eligible]
    reason_counts = Counter(e.exclusion_reason for e in entries if e.exclusion_reason)
    class_counts = Counter(e.source_class for e in entries)
    return {
        "schema": "issue_6769_figure_render_registry_audit.v1",
        "generated_at": generated_at,
        "source_commit": head,
        "issue": ISSUE_NUMBER,
        "total_entries": len(entries),
        "eligible_count": len(eligible),
        "ineligible_count": len(entries) - len(eligible),
        "source_class_counts": dict(sorted(class_counts.items())),
        "exclusion_reason_counts": dict(sorted(reason_counts.items())),
        "eligible_ids": [e.id for e in eligible],
        "route_evidence_only": True,
    }


def _entry_to_dict(entry: Entry) -> dict[str, Any]:
    """Serialize an entry to its registry representation."""
    return {
        "id": entry.id,
        "source_class": entry.source_class,
        "source_path": entry.source_path,
        "command_index": entry.command_index,
        "command": entry.command,
        "working_dir": entry.working_dir,
        "inputs": entry.inputs,
        "expected_outputs": entry.expected_outputs,
        "timeout_seconds": entry.timeout_seconds,
        "environment": entry.environment,
        "recurrence_eligible": entry.recurrence_eligible,
        "exclusion_reason": entry.exclusion_reason,
        "classification_notes": entry.classification_notes,
        "provenance_status": entry.provenance_status,
        "last_verified_commit": entry.last_verified_commit,
    }


def write_outputs(registry: dict[str, Any], audit: dict[str, Any]) -> None:
    """Write the registry YAML and audit JSON to their canonical paths."""
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    registry_text = yaml.safe_dump(
        registry, sort_keys=False, default_flow_style=False, allow_unicode=True, width=100
    )
    REGISTRY_PATH.write_text(registry_text, encoding="utf-8")
    write_evidence_json(AUDIT_PATH, audit, catalog_area="workflow_evidence")


def check_drift() -> int:
    """Return 0 if committed outputs match current discovery rules, else 1."""
    registry, audit = build_registry()
    if not REGISTRY_PATH.is_file() or not AUDIT_PATH.is_file():
        print("drift: registry or audit file missing", file=sys.stderr)
        return 1
    committed_registry = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))
    committed_audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    reg_cmp = {k: v for k, v in registry.items() if k != "generated_at"}
    committed_reg_cmp = {k: v for k, v in committed_registry.items() if k != "generated_at"}
    # The shared evidence writer prepends a review_marker to the on-disk audit; mirror it before
    # comparing so drift detection is byte-stable across regenerations.
    audit_with_marker = {"review_marker": review_marker_json(), **audit}
    audit_cmp = {k: v for k, v in audit_with_marker.items() if k != "generated_at"}
    committed_audit_cmp = {k: v for k, v in committed_audit.items() if k != "generated_at"}
    if reg_cmp != committed_reg_cmp or audit_cmp != committed_audit_cmp:
        print("drift: regenerated registry/audit differs from committed files", file=sys.stderr)
        return 1
    print("ok: registry and audit are in sync with discovery rules")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point: build, write, or check the figure-render registry."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="detect drift without rewriting files")
    parser.add_argument("--write", action="store_true", help="write registry and audit files")
    args = parser.parse_args(argv)
    if args.check:
        return check_drift()
    registry, audit = build_registry()
    if args.write:
        write_outputs(registry, audit)
        print(f"wrote {REGISTRY_PATH.relative_to(REPO_ROOT)}")
        print(f"wrote {AUDIT_PATH.relative_to(REPO_ROOT)}")
    else:
        print(yaml.safe_dump(registry, sort_keys=False, default_flow_style=False, width=100))
    print(
        f"summary: entries={audit['total_entries']} eligible={audit['eligible_count']} "
        f"ineligible={audit['ineligible_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
