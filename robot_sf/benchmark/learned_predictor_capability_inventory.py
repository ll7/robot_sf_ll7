"""Read-only capability inventory for the learned probabilistic graph predictor v1 lane.

This module is a **preflight/inventory surface only**. It does not implement, train, or
run any predictor, it does not change planner behavior, and it never asserts that the
learned-prediction training lane is unblocked. It enumerates the *code-level*
prerequisites that a v1 learned probabilistic graph predictor would have to extend
(interface protocol, ``ForecastBatch.v1`` contract, dataset recorder, model-artifact
registry export, and the readiness evidence gate) and reports whether each hook is
present in the current checkout.

Scope boundary (issue #2844)
----------------------------
The learned-predictor training lane is gated by the maintainer-defined readiness
*evidence* gate validated by
``scripts/validation/validate_learned_prediction_readiness.py`` (calibration,
transferability, and closed-loop coupling must all recommend ``continue``). This
inventory is complementary and strictly narrower: it surfaces whether the *wiring*
(the surfaces a new predictor would extend) is present, which is a different question
from whether the lane has continue-grade evidence.

A hook reported as ``present`` means only "the surface to extend exists in this
checkout"; it does **not** mean the lane is unblocked. ``unblocks_training`` is always
``False`` in the emitted report, with a pointer to the evidence gate that owns the
unblock decision. This matches the 2026-06-23 readiness-gate audit conclusion: the lane
is blocked on evidence, not on missing wiring.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Literal

ISSUE = 2844
LANE = "learned_probabilistic_graph_predictor_v1"

# Pointer to the canonical owner of the *unblock* decision. This inventory never makes
# that decision; it only reports code-surface presence.
READINESS_EVIDENCE_GATE = "scripts/validation/validate_learned_prediction_readiness.py"

HookCategory = Literal["interface", "contract", "dataset", "registry", "readiness_gate"]
HookRequirement = Literal["importable", "file"]

# Importer signature: maps a module name to an imported module (injectable for tests).
Importer = Callable[[str], ModuleType]


@dataclass(frozen=True)
class CapabilityHook:
    """One code-level prerequisite a v1 learned predictor would extend or rely on.

    Attributes:
        name: Stable identifier for the hook (used as the report key).
        category: Which prerequisite surface the hook belongs to.
        requirement: How presence is checked. ``"importable"`` resolves a
            ``module:symbol`` target via import; ``"file"`` checks a repo-relative path.
        target: For ``"importable"`` hooks, a ``"module:symbol"`` string; for ``"file"``
            hooks, a repository-root-relative path string.
        description: Human-facing note on why the hook matters for the v1 predictor.
    """

    name: str
    category: HookCategory
    requirement: HookRequirement
    target: str
    description: str


@dataclass(frozen=True)
class HookStatus:
    """Resolution result for a single :class:`CapabilityHook`."""

    hook: CapabilityHook
    present: bool
    detail: str


# Canonical v1 prerequisite surfaces. These are *existing* owners (per the issue's
# "Verified entry points" and AGENTS.md Canonical Owner Check) — this inventory points
# at them rather than duplicating their logic.
LEARNED_PREDICTOR_V1_HOOKS: tuple[CapabilityHook, ...] = (
    CapabilityHook(
        name="predictor_protocol",
        category="interface",
        requirement="importable",
        target="robot_sf.nav.predictive_types:ProbabilisticPredictor",
        description=(
            "Shared runtime-checkable predictor protocol a learned predictor must satisfy "
            "(predict(observation) -> ProbabilisticPrediction)."
        ),
    ),
    CapabilityHook(
        name="prediction_container",
        category="interface",
        requirement="importable",
        target="robot_sf.nav.predictive_types:ProbabilisticPrediction",
        description="Probabilistic prediction container returned by the predict() interface.",
    ),
    CapabilityHook(
        name="baseline_predictor_surface",
        category="interface",
        requirement="importable",
        target="robot_sf.nav.baseline_probabilistic_predictor:BaselineProbabilisticPredictor",
        description=(
            "Deterministic baseline predictor; the module a learned v1 predictor extends "
            "alongside under robot_sf/nav/."
        ),
    ),
    CapabilityHook(
        name="forecast_batch_contract",
        category="contract",
        requirement="importable",
        target="robot_sf.benchmark.forecast_batch:ForecastBatch",
        description="ForecastBatch.v1 output interchange contract for forecast artifacts.",
    ),
    CapabilityHook(
        name="forecast_batch_schema_version",
        category="contract",
        requirement="importable",
        target="robot_sf.benchmark.forecast_batch:FORECAST_BATCH_SCHEMA_VERSION",
        description="Pinned ForecastBatch schema-version constant for provenance validation.",
    ),
    CapabilityHook(
        name="forecast_dataset_recorder",
        category="dataset",
        requirement="importable",
        target=(
            "robot_sf.benchmark.forecast_dataset_recorder:"
            "record_forecast_dataset_from_trace_exports"
        ),
        description="Durable forecast dataset recorder (train from manifest, not ad-hoc output/).",
    ),
    CapabilityHook(
        name="dataset_builder_script",
        category="dataset",
        requirement="file",
        target="scripts/training/build_predictive_mixed_dataset.py",
        description="Split-manifest dataset builder entry point for the forecast lane.",
    ),
    CapabilityHook(
        name="model_artifact_registry",
        category="registry",
        requirement="importable",
        target="robot_sf.benchmark.local_model_artifacts:classify_local_model_references",
        description="Durable model-pointer classifier guarding against ad-hoc output/ model paths.",
    ),
    CapabilityHook(
        name="readiness_evidence_gate",
        category="readiness_gate",
        requirement="file",
        target=READINESS_EVIDENCE_GATE,
        description="Fail-closed readiness evidence gate that owns the lane unblock decision.",
    ),
    CapabilityHook(
        name="readiness_contract_doc",
        category="readiness_gate",
        requirement="file",
        target="docs/context/issue_2768_learned_prediction_readiness.md",
        description="Readiness contract documenting the prerequisites the gate enforces.",
    ),
)


def _split_target(target: str) -> tuple[str, str]:
    """Split a ``"module:symbol"`` target into its parts, validating the format.

    Returns:
        The ``(module_name, symbol)`` pair parsed from the target.
    """
    module_name, sep, symbol = target.partition(":")
    if not sep or not module_name or not symbol:
        raise ValueError(f"importable target must be 'module:symbol', got: {target!r}")
    return module_name, symbol


def _resolve_importable(target: str, importer: Importer) -> tuple[bool, str]:
    """Resolve an importable ``module:symbol`` hook.

    Returns:
        A ``(present, detail)`` pair; ``present`` is ``False`` on import failure or a
        missing attribute, with ``detail`` describing the reason.
    """
    module_name, symbol = _split_target(target)
    try:
        module = importer(module_name)
    except Exception as error:  # noqa: BLE001 - report any import failure as missing
        return False, f"import failed for {module_name}: {error.__class__.__name__}: {error}"
    if not hasattr(module, symbol):
        return False, f"module {module_name} has no attribute {symbol!r}"
    return True, f"resolved {target}"


def _resolve_file(target: str, repo_root: Path) -> tuple[bool, str]:
    """Resolve a repo-relative file hook.

    Returns:
        A ``(present, detail)`` pair indicating whether the file exists under
        ``repo_root``.
    """
    candidate = (repo_root / target).resolve()
    if candidate.exists():
        return True, f"found {target}"
    return False, f"missing file: {target}"


def resolve_hook(
    hook: CapabilityHook,
    *,
    repo_root: Path,
    importer: Importer,
) -> HookStatus:
    """Resolve a single hook against the current checkout (or an injected importer).

    Returns:
        A :class:`HookStatus` recording presence and a human-readable detail string.
    """
    if hook.requirement == "importable":
        present, detail = _resolve_importable(hook.target, importer)
    elif hook.requirement == "file":
        present, detail = _resolve_file(hook.target, repo_root)
    else:  # defensive: keep exhaustive over HookRequirement
        raise ValueError(f"unknown requirement: {hook.requirement!r}")
    return HookStatus(hook=hook, present=present, detail=detail)


def build_inventory(
    hooks: Sequence[CapabilityHook] = LEARNED_PREDICTOR_V1_HOOKS,
    *,
    repo_root: Path | None = None,
    importer: Importer | None = None,
) -> dict:
    """Build the read-only capability-inventory report.

    Args:
        hooks: Capability hooks to resolve. Defaults to the canonical v1 hook set; tests
            may pass synthetic hooks for deterministic, dependency-free checks.
        repo_root: Repository root used to resolve ``"file"`` hooks. Defaults to the
            current working directory.
        importer: Module importer for ``"importable"`` hooks. Defaults to
            :func:`importlib.import_module`; injectable for synthetic tests.

    Returns:
        A JSON-serializable report. ``unblocks_training`` is always ``False``: capability
        presence is a wiring signal, never a readiness/unblock claim. The unblock
        decision is owned by the readiness evidence gate at
        :data:`READINESS_EVIDENCE_GATE`.
    """
    root = (repo_root or Path.cwd()).resolve()
    resolve_importer: Importer = importer or importlib.import_module

    statuses = [resolve_hook(hook, repo_root=root, importer=resolve_importer) for hook in hooks]
    missing = [status for status in statuses if not status.present]

    return {
        "issue": ISSUE,
        "lane": LANE,
        # Capability wiring is "complete" only when every hook resolves; this is NOT a
        # training-readiness verdict.
        "capability_status": "complete" if not missing else "incomplete",
        # Honest, constant boundary: this surface never unblocks training.
        "unblocks_training": False,
        "unblock_owner": READINESS_EVIDENCE_GATE,
        "summary": {
            "total": len(statuses),
            "present": len(statuses) - len(missing),
            "missing": len(missing),
        },
        "hooks": [
            {
                "name": status.hook.name,
                "category": status.hook.category,
                "requirement": status.hook.requirement,
                "target": status.hook.target,
                "description": status.hook.description,
                "present": status.present,
                "detail": status.detail,
            }
            for status in statuses
        ],
        "missing_hooks": [status.hook.name for status in missing],
    }


def iter_hook_targets(hooks: Iterable[CapabilityHook] = LEARNED_PREDICTOR_V1_HOOKS) -> list[str]:
    """Return the resolvable targets for the given hooks (helper for tooling/tests)."""
    return [hook.target for hook in hooks]
