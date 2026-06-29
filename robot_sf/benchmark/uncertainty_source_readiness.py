"""Inventory readiness for issue #3557 uncertainty-source episode runs.

Plain-language summary: issue #3557 needs episode-level retained-vs-dropped safety
contrasts across several ``ScenarioBelief`` uncertainty sources. This module does
not run those episodes and does not claim the effect generalizes. It only reports
which sources already have the three prerequisites needed before a source can be
scheduled safely: a condition builder, a scenario hook, and the expected surrogate
output contract.
"""

from __future__ import annotations

import ast
import importlib
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "uncertainty_source_readiness.v1"
UNCERTAINTY_SOURCE_READINESS_SCHEMA = SCHEMA_VERSION
ISSUE = 3557

EXISTENCE_DEGRADATION = "existence_degradation"
VISIBILITY_OCCLUSION = "visibility_occlusion"
COVARIANCE_INFLATION = "covariance_inflation"
CLASS_PROBABILITY_AMBIGUITY = "class_probability_ambiguity"
TRACKING_NOISE = "tracking_noise"

SOURCE_READY = "ready"
MISSING_CONDITION_BUILDER = "missing_condition_builder"
MISSING_SCENARIO_HOOK = "missing_scenario_hook"
MISSING_SURROGATE_OUTPUT = "missing_surrogate_output"

_ISSUE_3471_HARNESS = "scripts.validation.run_scenario_belief_episode_safety_issue_3471"
_GENERALIZATION_LAYER = "robot_sf.representation.uncertainty_source_generalization"

EXPECTED_SURROGATE_OUTPUTS = (
    "source",
    "retained_unsafe_commit_rate",
    "dropped_unsafe_commit_rate",
    "min_separation_delta_m",
    "n_episodes",
)
_GENERALIZATION_SURROGATE_OUTPUTS = frozenset(EXPECTED_SURROGATE_OUTPUTS)

_REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True, slots=True)
class ReadinessComponent:
    """One source prerequisite and the local owner expected to satisfy it."""

    present: bool
    owner: str | None
    evidence: str

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-ready component metadata."""

        return {
            "present": self.present,
            "owner": self.owner,
            "evidence": self.evidence,
        }


@dataclass(frozen=True, slots=True)
class UncertaintySourceReadiness:
    """Readiness inventory row for one uncertainty source."""

    source: str
    condition_builder: ReadinessComponent
    scenario_hook: ReadinessComponent
    expected_surrogate_outputs: ReadinessComponent

    @property
    def ready(self) -> bool:
        """Whether this source has every prerequisite for an episode-level run."""

        return (
            self.condition_builder.present
            and self.scenario_hook.present
            and self.expected_surrogate_outputs.present
        )

    @property
    def status(self) -> str:
        """Return the first missing readiness class, or ``ready``."""

        if not self.condition_builder.present:
            return MISSING_CONDITION_BUILDER
        if not self.scenario_hook.present:
            return MISSING_SCENARIO_HOOK
        if not self.expected_surrogate_outputs.present:
            return MISSING_SURROGATE_OUTPUT
        return SOURCE_READY

    def as_dict(self) -> dict[str, Any]:
        """Return deterministic JSON-ready source inventory."""

        return {
            "source": self.source,
            "ready": self.ready,
            "status": self.status,
            "condition_builder": self.condition_builder.as_dict(),
            "scenario_hook": self.scenario_hook.as_dict(),
            "expected_surrogate_outputs": self.expected_surrogate_outputs.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class UncertaintySourceReadinessReport:
    """Issue #3557 source-readiness report."""

    sources: tuple[UncertaintySourceReadiness, ...]

    @property
    def ready_sources(self) -> tuple[str, ...]:
        """Sources with all prerequisites present."""

        return tuple(row.source for row in self.sources if row.ready)

    @property
    def blocked_sources(self) -> tuple[str, ...]:
        """Sources missing at least one prerequisite."""

        return tuple(row.source for row in self.sources if not row.ready)

    def as_dict(self) -> dict[str, Any]:
        """Return deterministic JSON-ready report."""

        return {
            "schema_version": SCHEMA_VERSION,
            "issue": ISSUE,
            "ready": not self.blocked_sources,
            "ready_sources": list(self.ready_sources),
            "blocked_sources": list(self.blocked_sources),
            "sources": [row.as_dict() for row in self.sources],
            "claim_boundary": (
                "Readiness inventory only; does not run the episode harness, "
                "does not submit Slurm/GPU work, and does not claim uncertainty-drop "
                "safety effect generalization."
            ),
        }


@dataclass(frozen=True, slots=True)
class SourceReadinessSpec:
    """Declarative owner inventory for one uncertainty source."""

    source: str
    condition_builder_owner: str | None
    scenario_hook_owner: str | None
    surrogate_output_owner: str | None


@dataclass(frozen=True, slots=True)
class UncertaintySourceRunSpec:
    """Backward-compatible inventory contract for one candidate uncertainty source."""

    source: str
    condition_builder: str | None
    scenario_hook: str | None
    expected_surrogate_outputs: tuple[str, ...] = tuple(sorted(_GENERALIZATION_SURROGATE_OUTPUTS))


DEFAULT_SOURCE_SPECS = (
    SourceReadinessSpec(
        source=EXISTENCE_DEGRADATION,
        condition_builder_owner=f"{_ISSUE_3471_HARNESS}:build_belief_for_mode",
        scenario_hook_owner=f"{_ISSUE_3471_HARNESS}:run_episode",
        surrogate_output_owner=f"{_GENERALIZATION_LAYER}:SourceContrast",
    ),
    SourceReadinessSpec(
        source=VISIBILITY_OCCLUSION,
        condition_builder_owner=None,
        scenario_hook_owner=None,
        surrogate_output_owner=f"{_GENERALIZATION_LAYER}:SourceContrast",
    ),
    SourceReadinessSpec(
        source=COVARIANCE_INFLATION,
        condition_builder_owner=None,
        scenario_hook_owner=None,
        surrogate_output_owner=f"{_GENERALIZATION_LAYER}:SourceContrast",
    ),
    SourceReadinessSpec(
        source=CLASS_PROBABILITY_AMBIGUITY,
        condition_builder_owner=None,
        scenario_hook_owner=None,
        surrogate_output_owner=f"{_GENERALIZATION_LAYER}:SourceContrast",
    ),
    SourceReadinessSpec(
        source=TRACKING_NOISE,
        condition_builder_owner=None,
        scenario_hook_owner=None,
        surrogate_output_owner=f"{_GENERALIZATION_LAYER}:SourceContrast",
    ),
)

DEFAULT_UNCERTAINTY_SOURCE_SPECS: tuple[UncertaintySourceRunSpec, ...] = (
    UncertaintySourceRunSpec(
        source="existence_degraded",
        condition_builder="_condition_existence_degraded",
        scenario_hook="build_belief_for_mode",
    ),
    UncertaintySourceRunSpec(
        source="visibility_limited",
        condition_builder="_condition_visibility_limited",
        scenario_hook=None,
    ),
    UncertaintySourceRunSpec(
        source="covariance_inflated",
        condition_builder="_condition_covariance_inflated",
        scenario_hook=None,
    ),
    UncertaintySourceRunSpec(
        source="class_probability",
        condition_builder="_condition_class_probability",
        scenario_hook=None,
    ),
    UncertaintySourceRunSpec(
        source="tracking_noise",
        condition_builder=None,
        scenario_hook=None,
    ),
)


def _resolve_owner(owner: str | None) -> tuple[bool, str]:
    """Return whether ``module:attribute`` exists without executing benchmark runs."""

    if owner is None:
        return False, "no owner registered"
    module_name, separator, attribute = owner.partition(":")
    if not module_name or separator != ":" or not attribute:
        return False, "owner must use 'module:attribute' form"
    source_path = _module_source_path(module_name)
    if source_path is not None:
        try:
            tree = ast.parse(source_path.read_text())
        except OSError as exc:
            return False, f"cannot read {source_path}: {exc}"
        if not _module_ast_defines(tree, attribute):
            return False, f"{module_name} has no top-level attribute {attribute!r}"
        return True, f"resolved {owner} by static source inspection"

    # Fallback for extension modules or generated modules without a source file.
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        return False, f"cannot import {module_name}: {exc}"
    if not hasattr(module, attribute):
        return False, f"{module_name} has no attribute {attribute!r}"
    return True, f"resolved {owner}"


def _module_source_path(module_name: str) -> Path | None:
    """Return local source path for a dotted module without importing parents.

    Returns:
        Python source path when the module resolves under this repository root.
    """

    relative = Path(*module_name.split("."))
    module_file = _REPO_ROOT / relative.with_suffix(".py")
    if module_file.exists():
        return module_file
    package_file = _REPO_ROOT / relative / "__init__.py"
    if package_file.exists():
        return package_file
    return None


def _module_ast_defines(tree: ast.Module, attribute: str) -> bool:
    """Return whether a module AST defines a top-level symbol.

    Returns:
        True when a function, class, or assignment defines ``attribute``.
    """

    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            if node.name == attribute:
                return True
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == attribute:
                    return True
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == attribute:
                return True
    return False


def _class_field_names_from_source(owner: str) -> set[str] | None:
    """Return dataclass-style annotated field names from source when available.

    Returns:
        Field names when the owner can be statically inspected; otherwise ``None``.
    """

    module_name, _separator, attribute = owner.partition(":")
    source_path = _module_source_path(module_name)
    if source_path is None:
        return None
    try:
        tree = ast.parse(source_path.read_text())
    except OSError:
        return None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == attribute:
            return {
                body_node.target.id
                for body_node in node.body
                if isinstance(body_node, ast.AnnAssign) and isinstance(body_node.target, ast.Name)
            }
    return None


def _source_contrast_has_expected_fields(owner: str | None) -> tuple[bool, str]:
    """Verify the decision-layer surrogate contrast exposes required fields.

    Returns:
        Pair of readiness boolean and human-readable evidence.
    """

    present, evidence = _resolve_owner(owner)
    if not present or owner is None:
        return present, evidence
    field_names = _class_field_names_from_source(owner)
    if field_names is None:
        module_name, _separator, attribute = owner.partition(":")
        contrast_type = getattr(importlib.import_module(module_name), attribute)
        field_names = {field.name for field in fields(contrast_type)}
    missing = sorted(set(EXPECTED_SURROGATE_OUTPUTS) - field_names)
    if missing:
        return False, f"{owner} missing fields {missing}"
    return True, f"{owner} exposes {sorted(EXPECTED_SURROGATE_OUTPUTS)}"


def inspect_uncertainty_source_readiness(
    specs: tuple[SourceReadinessSpec, ...] = DEFAULT_SOURCE_SPECS,
) -> UncertaintySourceReadinessReport:
    """Inspect prerequisite owners for issue #3557 uncertainty-source runs.

    Returns:
        Readiness report for the supplied uncertainty-source specs.
    """

    rows: list[UncertaintySourceReadiness] = []
    for spec in specs:
        builder_present, builder_evidence = _resolve_owner(spec.condition_builder_owner)
        hook_present, hook_evidence = _resolve_owner(spec.scenario_hook_owner)
        output_present, output_evidence = _source_contrast_has_expected_fields(
            spec.surrogate_output_owner
        )
        rows.append(
            UncertaintySourceReadiness(
                source=spec.source,
                condition_builder=ReadinessComponent(
                    present=builder_present,
                    owner=spec.condition_builder_owner,
                    evidence=builder_evidence,
                ),
                scenario_hook=ReadinessComponent(
                    present=hook_present,
                    owner=spec.scenario_hook_owner,
                    evidence=hook_evidence,
                ),
                expected_surrogate_outputs=ReadinessComponent(
                    present=output_present,
                    owner=spec.surrogate_output_owner,
                    evidence=output_evidence,
                ),
            )
        )
    return UncertaintySourceReadinessReport(sources=tuple(rows))


def _missing_expected_outputs(spec: UncertaintySourceRunSpec) -> list[str]:
    """Return required surrogate fields absent from ``spec``.

    Returns:
        Sorted missing surrogate output field names.
    """

    return sorted(_GENERALIZATION_SURROGATE_OUTPUTS - set(spec.expected_surrogate_outputs))


def _discover_condition_builders() -> frozenset[str]:
    """Return condition-builder symbols present in the #2546 diagnostic owner.

    Returns:
        Available condition-builder names, or an empty set when the owner cannot load.
    """

    try:
        module = importlib.import_module(
            "scripts.analysis.run_scenario_belief_uncertainty_diagnostic_issue_2546"
        )
    except (ImportError, SyntaxError):
        return frozenset()
    return frozenset(name for name in dir(module) if name.startswith("_condition_"))


def _discover_scenario_hooks() -> frozenset[str]:
    """Return episode scenario-hook symbols present in the #3471 runner owner.

    Returns:
        Available scenario-hook names, or an empty set when the owner cannot load.
    """

    try:
        module = importlib.import_module(
            "scripts.validation.run_scenario_belief_episode_safety_issue_3471"
        )
    except (ImportError, SyntaxError):
        return frozenset()
    return frozenset(
        {"build_belief_for_mode"} if hasattr(module, "build_belief_for_mode") else set()
    )


def classify_uncertainty_source_readiness(
    spec: UncertaintySourceRunSpec,
    *,
    known_condition_builders: set[str] | frozenset[str] | None = None,
    known_scenario_hooks: set[str] | frozenset[str] | None = None,
) -> dict[str, Any]:
    """Classify one legacy source spec as ready or blocked.

    Returns:
        Backward-compatible source readiness row.
    """

    builders = (
        _discover_condition_builders()
        if known_condition_builders is None
        else known_condition_builders
    )
    hooks = _discover_scenario_hooks() if known_scenario_hooks is None else known_scenario_hooks
    missing: list[str] = []

    condition_builder_present = bool(
        spec.condition_builder is not None and spec.condition_builder in builders
    )
    if not condition_builder_present:
        missing.append("condition_builder")

    scenario_hook_present = bool(spec.scenario_hook is not None and spec.scenario_hook in hooks)
    if not scenario_hook_present:
        missing.append("scenario_hook")

    missing_outputs = _missing_expected_outputs(spec)
    expected_surrogate_outputs_present = not missing_outputs
    if missing_outputs:
        missing.append("expected_surrogate_outputs")

    return {
        "source": spec.source,
        "status": "ready" if not missing else "blocked",
        "missing": sorted(set(missing)),
        "condition_builder": spec.condition_builder,
        "condition_builder_present": condition_builder_present,
        "scenario_hook": spec.scenario_hook,
        "scenario_hook_present": scenario_hook_present,
        "expected_surrogate_outputs": list(spec.expected_surrogate_outputs),
        "expected_surrogate_outputs_present": expected_surrogate_outputs_present,
        "missing_expected_surrogate_outputs": missing_outputs,
    }


def build_uncertainty_source_readiness_inventory(
    specs: tuple[UncertaintySourceRunSpec, ...] | list[UncertaintySourceRunSpec] = (
        DEFAULT_UNCERTAINTY_SOURCE_SPECS
    ),
) -> dict[str, Any]:
    """Build the backward-compatible issue #3557 readiness inventory.

    Returns:
        Legacy JSON-ready inventory used by existing benchmark tests and callers.
    """

    if not specs:
        raise ValueError("at least one uncertainty source spec is required")

    known_condition_builders = _discover_condition_builders()
    known_scenario_hooks = _discover_scenario_hooks()
    rows = [
        classify_uncertainty_source_readiness(
            spec,
            known_condition_builders=known_condition_builders,
            known_scenario_hooks=known_scenario_hooks,
        )
        for spec in specs
    ]
    ready_sources = sorted(row["source"] for row in rows if row["status"] == "ready")
    blocked_sources = sorted(row["source"] for row in rows if row["status"] == "blocked")

    return {
        "schema_version": UNCERTAINTY_SOURCE_READINESS_SCHEMA,
        "issue": ISSUE,
        "claim_boundary": "readiness_inventory_only",
        "not_benchmark_evidence": True,
        "runs_executed": False,
        "surrogate_semantics_changed": False,
        "sources": rows,
        "ready_sources": ready_sources,
        "blocked_sources": blocked_sources,
        "all_sources_ready": not blocked_sources,
    }


__all__ = [
    "CLASS_PROBABILITY_AMBIGUITY",
    "COVARIANCE_INFLATION",
    "DEFAULT_SOURCE_SPECS",
    "DEFAULT_UNCERTAINTY_SOURCE_SPECS",
    "EXISTENCE_DEGRADATION",
    "EXPECTED_SURROGATE_OUTPUTS",
    "ISSUE",
    "MISSING_CONDITION_BUILDER",
    "MISSING_SCENARIO_HOOK",
    "MISSING_SURROGATE_OUTPUT",
    "SCHEMA_VERSION",
    "SOURCE_READY",
    "TRACKING_NOISE",
    "UNCERTAINTY_SOURCE_READINESS_SCHEMA",
    "VISIBILITY_OCCLUSION",
    "ReadinessComponent",
    "SourceReadinessSpec",
    "UncertaintySourceReadiness",
    "UncertaintySourceReadinessReport",
    "UncertaintySourceRunSpec",
    "build_uncertainty_source_readiness_inventory",
    "classify_uncertainty_source_readiness",
    "inspect_uncertainty_source_readiness",
]
