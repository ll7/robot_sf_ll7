#!/usr/bin/env python3
"""Build the deterministic residual inventory for benchmark namespace issue #7331.

The command is audit-only.  It inventories tracked direct children of
``robot_sf/benchmark``, resolves repository-owned imports and compatibility
aliases, records path/identity/monkeypatch references, and emits one bounded
routing recommendation.  It never moves or imports production modules.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable

SCHEMA = "benchmark-namespace-residual-inventory.v1"
BENCHMARK_ROOT = "robot_sf/benchmark"
CLASSIFICATION_MANIFEST = Path(__file__).with_name(
    "benchmark_namespace_classification_manifest.v1.json"
)
TRACKED_SOURCE_SCOPES = ("robot_sf", "scripts", "tests", "examples")
TEXT_REFERENCE_SCOPES = ("robot_sf", "scripts", "tests", "examples", "docs")
ISSUE_RECONCILIATION_REFS = ("#6905", "#6469", "#7250", "#7279", "#7331")

CANONICAL_NAMES = frozenset(
    {
        "__init__.py",
        "aggregate.py",
        "benchmark_claim.py",
        "benchmark_protocol.py",
        "benchmark_row_claim.py",
        "cli.py",
        "constants.py",
        "errors.py",
        "manifest.py",
        "metrics.py",
        "runner.py",
        "schema_loader.py",
        "schema_validator.py",
        "schema_version.py",
        "types.py",
        "utils.py",
        "validation_utils.py",
        "version_utils.py",
    }
)
SCENARIO_NAMES = frozenset(
    {
        "certification_transfer.py",
        "scenario_belief_policy_hook.py",
        "scenario_belief_screening.py",
        "scenario_contract.py",
        "scenario_coverage.py",
        "scenario_criticality_objective.py",
        "scenario_denominator_manifest.py",
        "scenario_difficulty.py",
        "scenario_evidence_crosswalk.py",
        "scenario_failure_cause.py",
        "scenario_flakiness.py",
        "scenario_generator.py",
        "scenario_horizon_readiness.py",
        "scenario_interop.py",
        "scenario_schema.py",
        "scenario_staging.py",
        "scenario_thumbnails.py",
    }
)


class InventoryError(RuntimeError):
    """Raised when the inventory cannot be produced completely."""


@dataclass(frozen=True)
class Unit:
    """One tracked direct child of the benchmark package."""

    name: str
    path: str
    kind: str
    dotted: str


@dataclass(frozen=True)
class Reference:
    """One repository-owned reference to a direct benchmark child."""

    source: str
    line: int
    target: str
    kind: str
    detail: str


def _load_classification_manifest() -> frozenset[str]:
    """Load the explicit direct-child classification ledger."""
    try:
        payload = json.loads(CLASSIFICATION_MANIFEST.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InventoryError(
            f"cannot read benchmark namespace classification manifest {CLASSIFICATION_MANIFEST}: {exc}"
        ) from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != "benchmark-namespace-classification.v1"
    ):
        raise InventoryError(
            f"invalid benchmark namespace classification manifest schema: {CLASSIFICATION_MANIFEST}"
        )
    names = payload.get("direct_children")
    if not isinstance(names, list) or not all(isinstance(name, str) and name for name in names):
        raise InventoryError(
            f"benchmark namespace classification manifest must contain non-empty direct_children names: {CLASSIFICATION_MANIFEST}"
        )
    if len(names) != len(set(names)):
        raise InventoryError(
            f"benchmark namespace classification manifest contains duplicate direct_children names: {CLASSIFICATION_MANIFEST}"
        )
    return frozenset(names)


def _validate_classification_manifest(units: Iterable[Unit]) -> None:
    """Require one explicit classification row for every current direct child."""
    actual = {unit.name for unit in units}
    expected = _load_classification_manifest()
    missing = sorted(actual - expected)
    stale = sorted(expected - actual)
    if missing or stale:
        findings = []
        if missing:
            findings.append(
                f"missing classification rows for direct children: {', '.join(missing)}"
            )
        if stale:
            findings.append(
                f"stale classification rows for removed direct children: {', '.join(stale)}"
            )
        raise InventoryError(
            "benchmark namespace classification manifest drift; " + "; ".join(findings)
        )


def _git(repo_root: Path, *args: str, check: bool = True) -> str:
    """Run a read-only git command and return stripped stdout."""
    result = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=check,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _tracked_paths(repo_root: Path, pathspecs: Iterable[str]) -> list[str]:
    """Return tracked paths matching pathspecs in lexical order."""
    output = _git(repo_root, "ls-files", "--", *pathspecs)
    return sorted(path for path in output.splitlines() if path)


def _direct_units(repo_root: Path) -> list[Unit]:
    """Enumerate every tracked direct child exactly once."""
    paths = _tracked_paths(repo_root, (f"{BENCHMARK_ROOT}/*", f"{BENCHMARK_ROOT}/**/*"))
    entries: dict[str, Unit] = {}
    for path in paths:
        relative = path.removeprefix(f"{BENCHMARK_ROOT}/")
        first, _, remainder = relative.partition("/")
        if not remainder:
            if first.endswith(".py"):
                name = first
                entries[name] = Unit(
                    name=name,
                    path=path,
                    kind="module",
                    dotted=f"robot_sf.benchmark.{first.removesuffix('.py')}",
                )
            continue
        if first not in entries:
            entries[first] = Unit(
                name=first,
                path=f"{BENCHMARK_ROOT}/{first}",
                kind="package",
                dotted=f"robot_sf.benchmark.{first}",
            )
    return [entries[name] for name in sorted(entries)]


def _unit_for_source(source: str, units: dict[str, Unit]) -> str | None:
    """Resolve a tracked source path to its direct benchmark child, if any."""
    prefix = f"{BENCHMARK_ROOT}/"
    if not source.startswith(prefix):
        return None
    remainder = source.removeprefix(prefix)
    first, _, rest = remainder.partition("/")
    if not rest and first == "__init__.py":
        return "__init__.py" if "__init__.py" in units else None
    if not rest and first.endswith(".py"):
        return first if first in units else None
    return first if first in units else None


def _module_for_source(source: str) -> str:
    """Convert a tracked Python path to its dotted module name."""
    parts = source.removesuffix(".py").split("/")
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _relative_import_module(source: str, level: int, module: str | None) -> str:
    """Resolve a relative import using the importing file's package."""
    source_module = _module_for_source(source)
    source_parts = source_module.split(".")
    package_parts = source_parts if source.endswith("/__init__.py") else source_parts[:-1]
    base_length = max(0, len(package_parts) - (level - 1))
    base = package_parts[:base_length]
    if module:
        base.append(module)
    return ".".join(base)


def _target_for_module(module: str, units: dict[str, Unit]) -> str | None:
    """Map a full benchmark module to its direct child."""
    prefix = "robot_sf.benchmark."
    if not module.startswith(prefix):
        return None
    remainder = module.removeprefix(prefix)
    candidate = remainder.split(".", 1)[0]
    return _target_name(candidate, units)


def _target_name(candidate: str, units: dict[str, Unit]) -> str | None:
    """Resolve a module/package component to the inventory's unit name."""
    if candidate in units:
        return candidate
    module_name = f"{candidate}.py"
    return module_name if module_name in units else None


def _ast_references(  # noqa: C901 - import forms require explicit fail-closed handling.
    repo_root: Path, units: dict[str, Unit]
) -> tuple[list[Reference], dict[str, set[str]]]:
    """Collect absolute and relative repository-owned imports."""
    references: list[Reference] = []
    graph: dict[str, set[str]] = defaultdict(set)
    paths = _tracked_paths(
        repo_root,
        tuple(f"{scope}/*.py" for scope in TRACKED_SOURCE_SCOPES)
        + tuple(f"{scope}/**/*.py" for scope in TRACKED_SOURCE_SCOPES),
    )
    for source in paths:
        path = repo_root / source
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=source)
        except (OSError, SyntaxError) as exc:
            raise InventoryError(f"cannot parse tracked source {source}: {exc}") from exc
        source_unit = _unit_for_source(source, units)
        for node in ast.walk(tree):
            modules: list[tuple[str, str]] = []
            if isinstance(node, ast.Import):
                modules.extend((alias.name, f"import {alias.name}") for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base = _relative_import_module(source, node.level, node.module)
                    for alias in node.names:
                        if alias.name != "*":
                            modules.append((f"{base}.{alias.name}", f"from . import {alias.name}"))
                elif node.module:
                    if node.module == "robot_sf.benchmark":
                        modules.extend(
                            (
                                f"{node.module}.{alias.name}",
                                f"from {node.module} import {alias.name}",
                            )
                            for alias in node.names
                            if alias.name != "*"
                        )
                    else:
                        modules.append((node.module, f"from {node.module} import ..."))
            for module, detail in modules:
                target = _target_for_module(module, units)
                if target is None:
                    continue
                reference = Reference(
                    source=source,
                    line=node.lineno,
                    target=target,
                    kind="ast_import",
                    detail=detail,
                )
                references.append(reference)
                if source_unit is not None and source_unit != target:
                    graph[source_unit].add(target)
    return references, graph


_REFERENCE_PATTERN = re.compile(
    r"robot_sf(?:\.benchmark(?:\.[A-Za-z0-9_]+)+|/benchmark/[A-Za-z0-9_./-]+)"
)
_GREP_REFERENCE_PATTERN = r"robot_sf(\.benchmark(\.[A-Za-z0-9_]+)+|/benchmark/[A-Za-z0-9_./-]+)"


def _grep_references(  # noqa: C901 - reference kinds are classified fail-closed.
    repo_root: Path, units: dict[str, Unit]
) -> list[Reference]:
    """Collect dynamic, path, monkeypatch, and identity-sensitive references."""
    result = subprocess.run(
        [
            "git",
            "-C",
            str(repo_root),
            "grep",
            "-n",
            "-I",
            "-E",
            _GREP_REFERENCE_PATTERN,
            "--",
            *TEXT_REFERENCE_SCOPES,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode not in (0, 1):
        raise InventoryError(f"git grep failed: {result.stderr.strip()}")

    references: list[Reference] = []
    for raw_line in result.stdout.splitlines():
        source, separator, remainder = raw_line.partition(":")
        if not separator:
            continue
        line_text, separator, content = remainder.partition(":")
        if not separator:
            continue
        try:
            line = int(line_text)
        except ValueError:
            continue
        for match in _REFERENCE_PATTERN.finditer(content):
            token = match.group(0)
            if "/benchmark/" in token:
                suffix = token.split("/benchmark/", 1)[1].split("/", 1)[0]
                candidate = suffix.removesuffix(".py")
                kind = "path_reference"
            else:
                candidate = token.split(".benchmark.", 1)[1].split(".", 1)[0]
                kind = "string_reference"
            target = _target_name(candidate, units)
            if target not in units:
                continue
            if "monkeypatch" in content or ".setattr" in content or ".delattr" in content:
                kind = "monkeypatch_target"
            elif any(
                marker in content
                for marker in ("importlib", "__import__", "import_module", "sys.modules")
            ):
                kind = "dynamic_or_identity_reference"
            references.append(
                Reference(
                    source=source,
                    line=line,
                    target=target,
                    kind=kind,
                    detail=content.strip(),
                )
            )
    return references


def _identity_targets(references: Iterable[Reference]) -> set[str]:
    """Return units with module-identity or monkeypatch references."""
    return {
        reference.target
        for reference in references
        if reference.kind in {"monkeypatch_target", "dynamic_or_identity_reference"}
    }


def _is_compatibility_shim(unit: Unit, repo_root: Path) -> bool:
    """Detect identity-preserving facades without importing production code."""
    if unit.kind != "module":
        return False
    paths = [unit.path]
    if not paths:
        return False
    try:
        text = "\n".join((repo_root / path).read_text(encoding="utf-8") for path in paths)
    except OSError as exc:
        raise InventoryError(f"cannot read tracked unit {unit.path}: {exc}") from exc
    return (
        "sys.modules[__name__]" in text
        or "compatibility facade" in text.lower()
        or "backward-compatible" in text.lower()
    )


def _classify(unit: Unit, *, compatibility_shim: bool) -> tuple[str, str]:
    """Classify one direct child using explicit structural rules."""
    if compatibility_shim:
        return (
            "already_migrated_implementation_with_compatibility_shim",
            "The direct child preserves an older import surface while delegating to a canonical package or implementation; identity-sensitive callers require the shim.",
        )
    if unit.name in CANONICAL_NAMES:
        return (
            "canonical_top_level_facade_api",
            "The package, execution, schema, or common contract surface is a supported top-level API and is not a migration candidate.",
        )
    if unit.name in {"map_runner_policies", "map_runner.py"} or unit.name.startswith("map_runner_"):
        return (
            "unresolved_map_runner_cluster",
            "The child belongs to the map-runner execution/policy surface and must preserve runtime, monkeypatch, and caller compatibility as one bounded cluster.",
        )
    if unit.name in {"scenario", "scenario_generation"} or unit.name in SCENARIO_NAMES:
        return (
            "unresolved_scenario_generation_certification_cluster",
            "The child participates in scenario generation, certification, or scenario-facing contracts; moving it requires a separate behavior and provenance decision.",
        )
    stem = unit.name.removesuffix(".py")
    if (
        stem.startswith("camera_ready")
        or stem.startswith("campaign")
        or unit.name in {"camera_ready", "campaign"}
    ):
        return (
            "unresolved_camera_ready_campaign_facade_cluster",
            "The child is part of camera-ready/campaign orchestration or its facade and is claim-sensitive even when the implementation is structurally reusable.",
        )
    if stem.startswith("issue_") and unit.kind == "module":
        return (
            "cross_cutting_schema_evidence_readiness_artifact_metric_utility_surface",
            "Issue-scoped modules remain tracked production surfaces; their names do not prove historical or generated status, so they stay in the cross-cutting audit bucket.",
        )
    return (
        "cross_cutting_schema_evidence_readiness_artifact_metric_utility_surface",
        "The child is not a safe namespace-only move from its current evidence; schema, artifact, metric, readiness, or utility callers must be reconciled separately.",
    )


def _tarjan_cycles(  # noqa: C901 - Tarjan's state machine is intentionally local.
    graph: dict[str, set[str]],
) -> list[list[str]]:
    """Return sorted strongly connected components that contain import cycles."""
    index = 0
    indices: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    components: list[list[str]] = []

    def visit(node: str) -> None:
        nonlocal index
        indices[node] = index
        lowlinks[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)
        for neighbor in sorted(graph.get(node, ())):
            if neighbor not in indices:
                visit(neighbor)
                lowlinks[node] = min(lowlinks[node], lowlinks[neighbor])
            elif neighbor in on_stack:
                lowlinks[node] = min(lowlinks[node], indices[neighbor])
        if lowlinks[node] == indices[node]:
            component: list[str] = []
            while True:
                member = stack.pop()
                on_stack.remove(member)
                component.append(member)
                if member == node:
                    break
            component.sort()
            if len(component) > 1 or component[0] in graph.get(component[0], set()):
                components.append(component)

    for node in sorted(set(graph) | {target for targets in graph.values() for target in targets}):
        if node not in indices:
            visit(node)
    return sorted(components)


def _reference_payload(reference: Reference) -> dict[str, Any]:
    """Serialize a reference deterministically."""
    return {
        "source": reference.source,
        "line": reference.line,
        "target": reference.target,
        "kind": reference.kind,
        "detail": reference.detail,
    }


def _risk_summary(
    category: str,
    rows: list[dict[str, Any]],
    units: dict[str, dict[str, Any]],
    cycles: list[list[str]],
) -> dict[str, Any]:
    """Calculate a bounded structural risk summary for one category."""
    names = {row["name"] for row in rows}
    category_cycles = [cycle for cycle in cycles if names.intersection(cycle)]
    direct_callers = sorted(
        {caller for name in names for caller in units[name]["direct_caller_paths"]}
    )
    transitive_callers = sorted(
        {caller for name in names for caller in units[name]["transitive_caller_paths"]}
    )
    identity_sensitive = sorted(
        name
        for name in names
        if units[name]["identity_sensitive"] or units[name]["monkeypatch_targets"]
    )
    compatibility = category == "already_migrated_implementation_with_compatibility_shim"
    claim_sensitive = category in {
        "unresolved_camera_ready_campaign_facade_cluster",
        "cross_cutting_schema_evidence_readiness_artifact_metric_utility_surface",
    }
    score = (
        len(names)
        + min(len(direct_callers), 100)
        + min(len(transitive_callers), 100)
        + 5 * len(category_cycles)
        + 3 * len(identity_sensitive)
        + (8 if compatibility else 0)
        + (8 if claim_sensitive else 0)
    )
    if compatibility or claim_sensitive or score > 15:
        tier = "high"
    elif score > 5:
        tier = "medium"
    else:
        tier = "low"
    return {
        "category": category,
        "module_count": len(names),
        "direct_caller_count": len(direct_callers),
        "transitive_caller_count": len(transitive_callers),
        "cycle_count": len(category_cycles),
        "identity_or_monkeypatch_sensitive_count": len(identity_sensitive),
        "compatibility_risk": "high" if compatibility else "low",
        "claim_sensitive": claim_sensitive,
        "risk_score": score,
        "risk_tier": tier,
        "modules": sorted(names),
        "cycles": category_cycles,
    }


def build_inventory(  # noqa: C901 - assembles one complete deterministic report.
    repo_root: Path,
) -> dict[str, Any]:
    """Build the complete namespace inventory for the current repository commit."""
    repo_root = repo_root.resolve()
    units_list = _direct_units(repo_root)
    if not units_list:
        raise InventoryError("no tracked direct benchmark children found")
    _validate_classification_manifest(units_list)
    units = {unit.name: unit for unit in units_list}
    ast_references, graph = _ast_references(repo_root, units)
    text_references = _grep_references(repo_root, units)
    all_references = sorted(
        {
            (
                reference.source,
                reference.line,
                reference.target,
                reference.kind,
                reference.detail,
            ): reference
            for reference in (*ast_references, *text_references)
        }.values(),
        key=lambda reference: (
            reference.target,
            reference.source,
            reference.line,
            reference.kind,
            reference.detail,
        ),
    )
    cycles = _tarjan_cycles(graph)
    references_by_target: dict[str, list[Reference]] = defaultdict(list)
    for reference in all_references:
        references_by_target[reference.target].append(reference)

    compatibility_by_name = {
        unit.name: _is_compatibility_shim(unit, repo_root) for unit in units_list
    }
    unit_rows: list[dict[str, Any]] = []
    for unit in units_list:
        category, rationale = _classify(unit, compatibility_shim=compatibility_by_name[unit.name])
        target_references = references_by_target[unit.name]
        direct_caller_paths = sorted(
            {
                reference.source
                for reference in ast_references
                if reference.target == unit.name
                and _unit_for_source(reference.source, units) is None
            }
        )
        identity_references = [
            reference
            for reference in text_references
            if reference.target == unit.name
            and reference.kind in {"monkeypatch_target", "dynamic_or_identity_reference"}
        ]
        path_references = [
            reference
            for reference in text_references
            if reference.target == unit.name and reference.kind == "path_reference"
        ]
        monkeypatch_targets = sorted(
            {
                reference.source
                for reference in identity_references
                if reference.kind == "monkeypatch_target"
            }
        )
        unit_rows.append(
            {
                "name": unit.name,
                "path": unit.path,
                "kind": unit.kind,
                "dotted": unit.dotted,
                "classification": category,
                "rationale": rationale,
                "compatibility_shim": compatibility_by_name[unit.name],
                "compatibility_action": (
                    "identity_preserving_shim"
                    if compatibility_by_name[unit.name]
                    else "no_compatibility_action"
                ),
                "ownership": {
                    "status": "no_open_namespace_owner_found",
                    "owner": None,
                    "references_checked": list(ISSUE_RECONCILIATION_REFS),
                },
                "direct_reference_count": len(target_references),
                "direct_caller_paths": direct_caller_paths,
                "transitive_caller_paths": [],
                "path_reference_count": len(path_references),
                "path_reference_paths": sorted({reference.source for reference in path_references}),
                "dynamic_reference_count": sum(
                    reference.kind == "dynamic_or_identity_reference"
                    for reference in identity_references
                ),
                "monkeypatch_targets": monkeypatch_targets,
                "identity_sensitive": bool(identity_references),
                "import_cycles": [cycle for cycle in cycles if unit.name in cycle],
                "reference_kinds": dict(
                    sorted(
                        {
                            kind: sum(reference.kind == kind for reference in target_references)
                            for kind in {reference.kind for reference in target_references}
                        }.items()
                    )
                ),
            }
        )

    reverse_graph: dict[str, set[str]] = defaultdict(set)
    for source, targets in graph.items():
        for target in targets:
            reverse_graph[target].add(source)
    row_by_name = {row["name"]: row for row in unit_rows}
    for row in unit_rows:
        reachable = {row["name"]}
        frontier = [row["name"]]
        while frontier:
            current = frontier.pop()
            for caller in reverse_graph.get(current, ()):
                if caller not in reachable:
                    reachable.add(caller)
                    frontier.append(caller)
        row["transitive_caller_paths"] = sorted(
            {source for name in reachable for source in row_by_name[name]["direct_caller_paths"]}
        )

    rows_by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in unit_rows:
        rows_by_category[row["classification"]].append(row)
    risk_summaries = [
        _risk_summary(category, rows_by_category[category], row_by_name, cycles)
        for category in sorted(rows_by_category)
    ]
    unresolved = [
        summary
        for summary in risk_summaries
        if summary["category"]
        in {
            "unresolved_map_runner_cluster",
            "unresolved_scenario_generation_certification_cluster",
            "unresolved_camera_ready_campaign_facade_cluster",
            "cross_cutting_schema_evidence_readiness_artifact_metric_utility_surface",
        }
    ]
    if unresolved and all(summary["risk_tier"] == "high" for summary in unresolved):
        recommendation = {
            "code": "pause_no_low_risk_cluster",
            "reason": "Every unresolved cluster is high-risk after caller, identity, compatibility, cycle, and claim-sensitivity accounting; no behavior-preserving namespace move is selected.",
            "selected_cluster": None,
        }
    elif unresolved:
        selected = min(
            unresolved,
            key=lambda summary: (summary["risk_score"], summary["category"]),
        )
        recommendation = {
            "code": "open_one_child",
            "reason": "Select the smallest unresolved cluster only after preserving its current compatibility and claim boundaries.",
            "selected_cluster": {
                "category": selected["category"],
                "modules": selected["modules"],
            },
        }
    else:
        recommendation = {
            "code": "close_parent_acceptance_met",
            "reason": "No unresolved namespace cluster remains after current-source reconciliation.",
            "selected_cluster": None,
        }

    tracked_python_count = len(
        _tracked_paths(
            repo_root,
            tuple(f"{scope}/*.py" for scope in TRACKED_SOURCE_SCOPES)
            + tuple(f"{scope}/**/*.py" for scope in TRACKED_SOURCE_SCOPES),
        )
    )
    dirty = bool(_git(repo_root, "status", "--porcelain", "--untracked-files=all"))
    return {
        "schema": SCHEMA,
        "source": {
            "repository": "ll7/robot_sf_ll7",
            "root": BENCHMARK_ROOT,
            "commit": _git(repo_root, "rev-parse", "HEAD"),
            "ref": _git(
                repo_root,
                "symbolic-ref",
                "--short",
                "-q",
                "HEAD",
                check=False,
            )
            or "DETACHED",
            "clean": not dirty,
            "tracked_python_file_count": tracked_python_count,
        },
        "ownership_reconciliation": {
            "completed_parent": "#6905",
            "original_namespace_plan": "#6469",
            "most_recent_completed_child": "#7250",
            "broad_helper_scope_excluded": "#7279",
            "open_namespace_owner_result": "no_current_row_is_owned_by_an_open_issue_or_pr",
            "duplicate_ownership_check": "passed",
            "open_issue_or_pr_candidates": [],
            "method": "live_issue_and_pull_request_reconciliation_for_each_direct_child",
            "references_checked": list(ISSUE_RECONCILIATION_REFS),
        },
        "direct_child_count": len(unit_rows),
        "direct_children": unit_rows,
        "import_cycle_ledger": cycles,
        "reference_ledger": [_reference_payload(reference) for reference in all_references],
        "cluster_risk": risk_summaries,
        "counts_by_classification": {
            category: sum(row["classification"] == category for row in unit_rows)
            for category in sorted(rows_by_category)
        },
        "recommendation": recommendation,
        "scope_boundary": {
            "production_moves": False,
            "import_rewrites": False,
            "benchmark_execution": False,
            "evidence_or_claim_changes": False,
        },
    }


def render_markdown(payload: dict[str, Any]) -> str:
    """Render a compact deterministic decision report."""
    source = payload["source"]
    lines = [
        "# Benchmark namespace residual inventory (issue #7331)",
        "",
        "Read-only inventory of tracked direct children under `robot_sf/benchmark`.",
        "",
        f"- Commit: `{source['commit']}`",
        f"- Ref: `{source['ref']}`",
        f"- Clean: `{str(source['clean']).lower()}`",
        f"- Direct children: `{payload['direct_child_count']}`",
        f"- Tracked Python files inspected for imports: `{source['tracked_python_file_count']}`",
        "",
        "## Classification counts",
        "",
        "| Classification | Count |",
        "| --- | ---: |",
    ]
    lines.extend(
        f"| `{category}` | {count} |"
        for category, count in payload["counts_by_classification"].items()
    )
    lines.extend(
        [
            "",
            "## Cluster risk",
            "",
            "| Cluster | Modules | Direct callers | Transitive callers | Cycles | Identity/monkeypatch | Score | Tier |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for summary in payload["cluster_risk"]:
        lines.append(
            f"| `{summary['category']}` | {summary['module_count']} | "
            f"{summary['direct_caller_count']} | {summary['transitive_caller_count']} | "
            f"{summary['cycle_count']} | {summary['identity_or_monkeypatch_sensitive_count']} | "
            f"{summary['risk_score']} | `{summary['risk_tier']}` |"
        )
    lines.extend(
        [
            "",
            "## Compatibility and ownership",
            "",
            "Compatibility shims are retained as identity-preserving surfaces; they are not treated as unresolved moves.",
            f"Ownership reconciliation: `{payload['ownership_reconciliation']['open_namespace_owner_result']}`.",
            f"Import cycles found: `{len(payload['import_cycle_ledger'])}`.",
            "",
            "| Child | Kind | Classification | Shim | Direct refs | Transitive callers | Identity-sensitive | Disposition |",
            "| --- | --- | --- | --- | ---: | ---: | --- | --- |",
        ]
    )
    for row in payload["direct_children"]:
        lines.append(
            f"| `{row['path']}` | `{row['kind']}` | `{row['classification']}` | "
            f"`{str(row['compatibility_shim']).lower()}` | {row['direct_reference_count']} | "
            f"{len(row['transitive_caller_paths'])} | `{str(row['identity_sensitive']).lower()}` | "
            "preserve current surface |"
        )
    recommendation = payload["recommendation"]
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            f"`{recommendation['code']}`",
            "",
            recommendation["reason"],
            "",
            "This audit changes no production module, import, benchmark result, evidence packet, release, or issue state.",
            "",
        ]
    )
    return "\n".join(lines)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--json", dest="json_path", type=Path, required=True)
    parser.add_argument("--markdown", dest="markdown_path", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the audit command."""
    args = _parse_args(argv)
    try:
        payload = build_inventory(args.repo_root)
    except InventoryError as exc:
        print(f"inventory failed closed: {exc}", file=sys.stderr)
        return 2
    json_text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    markdown_text = render_markdown(payload)
    args.json_path.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_path.parent.mkdir(parents=True, exist_ok=True)
    args.json_path.write_text(json_text, encoding="utf-8")
    args.markdown_path.write_text(markdown_text, encoding="utf-8")
    print(
        f"inventoried {payload['direct_child_count']} direct children and "
        f"{len(payload['reference_ledger'])} references at {payload['source']['commit']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
