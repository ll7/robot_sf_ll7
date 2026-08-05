"""Fail-closed research-lane launcher for the continual-adaptation protocol (#6659).

This module wires the metadata-only continual-adaptation protocol contract
(:mod:`robot_sf.research.continual_adaptation_protocol`) to a research-lane
launcher. It loads a ``continual_adaptation_run.v1`` manifest, **refuses to
proceed unless** :func:`check_continual_adaptation_run` reports
``protocol_status == 'valid'``, and then echoes the bounded adaptation plus the
nominal/shift/forgetting evaluation surfaces as **diagnostic** outputs under
``output/``.

The launcher is deliberately diagnostic-only:

* it trains nothing and writes no checkpoint;
* it mutates no safety wrapper;
* it computes no evaluation metric;
* it emits **no promotion decision** and generates **no evidence bundle**;
* it makes **no benchmark or paper claim**.

Every output stamps :data:`CONTINUAL_ADAPTATION_EVIDENCE_BOUNDARY` so a
diagnostic launch is never mistaken for an executed adaptation, a promoted
policy, or benchmark/paper evidence. Promotion, evidence-bundle generation, and
benchmark-campaign integration are deferred to a separate approval-gated issue.

The merged contract validator and its JSON schema are *consumed*, never
modified; this launcher gates on them.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from robot_sf.common.logging import get_logger
from robot_sf.research.artifact_paths import get_artifact_root
from robot_sf.research.continual_adaptation_protocol import (
    CONTINUAL_ADAPTATION_EVIDENCE_BOUNDARY,
    CONTINUAL_ADAPTATION_RUN_SCHEMA_VERSION,
    PROTOCOL_STATUS_VALID,
    ContinualAdaptationProtocolError,
    check_continual_adaptation_run,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = get_logger(__name__)

#: Launcher mode stamped on every diagnostic output. The launcher never executes
#: training or evaluation; it only echoes the bounded protocol as diagnostics.
CONTINUAL_ADAPTATION_LAUNCHER_MODE = "diagnostic_only"

#: The launcher emits no promotion decision. Promotion is deferred to a separate
#: approval-gated issue; this constant makes that explicit on every report.
LAUNCHER_PROMOTION_DECISION_NONE = "none_diagnostic_only"

#: Per-surface diagnostic status: the adaptation/evaluation is declared, not run.
DIAGNOSTIC_STATUS_NOT_EXECUTED = "diagnostic_only_not_executed"

#: Evaluation surfaces the bounded revalidation protocol declares.
EVALUATION_SURFACES = ("nominal", "shift", "forgetting")

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True, slots=True)
class ContinualAdaptationDiagnosticReport:
    """Aggregate diagnostic output of a fail-closed continual-adaptation launch.

    A report is only ever produced for a manifest whose protocol check returned
    ``protocol_status == 'valid'``; an invalid manifest fails closed before any
    output is written. The report describes the bounded adaptation and the
    nominal/shift/forgetting evaluation surfaces as diagnostics only: it never
    records an executed adaptation, a computed metric, a checkpoint write, a
    safety-wrapper mutation, a promotion decision, or an evidence bundle, and it
    is never benchmark or paper evidence.
    """

    schema_version: str
    run_id: str
    issue: int
    evidence_boundary: str
    launcher_mode: str
    baseline_policy_identifier: str
    derived_adapted_policy_identifier: str
    protocol_status: str
    emits_promotion_decision: bool
    evidence_bundle_generated: bool
    makes_benchmark_or_paper_claim: bool
    adaptation: dict[str, Any]
    evaluations: dict[str, Any]
    output_files: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dictionary representation."""
        return asdict(self)


def get_continual_adaptation_output_root() -> Path:
    """Return the canonical diagnostic output root under the artifact root.

    Returns:
        ``<artifact_root>/continual_adaptation_diagnostics`` where the artifact
        root defaults to ``output/`` (gitignored) and honors the
        ``ROBOT_SF_ARTIFACT_ROOT`` override.
    """
    return get_artifact_root() / "continual_adaptation_diagnostics"


def _validated_artifact_root() -> Path:
    """Resolve the artifact root and reject repository-local tracked locations.

    External roots are supported for temporary directories and controlled artifact
    stores. A root inside this repository must stay below the canonical ``output/``
    directory so an environment override cannot redirect diagnostics into tracked or
    forbidden paths.

    Returns:
        The resolved, validated artifact root.

    Raises:
        ContinualAdaptationProtocolError: when a repository-local artifact root is
            outside the canonical ``output/`` directory.
    """
    artifact_root = get_artifact_root().expanduser().resolve()
    repository_root = _REPOSITORY_ROOT.resolve()
    canonical_repository_artifact_root = (repository_root / "output").resolve()

    try:
        artifact_root.relative_to(repository_root)
    except ValueError:
        return artifact_root

    try:
        artifact_root.relative_to(canonical_repository_artifact_root)
    except ValueError:
        raise ContinualAdaptationProtocolError(
            [
                f"configured artifact root {artifact_root} is inside the repository but "
                f"outside canonical output {canonical_repository_artifact_root}; diagnostic "
                "outputs must never target tracked or forbidden repository paths"
            ]
        ) from None
    return artifact_root


def _resolve_output_dir(output_dir: str | Path, *, namespace_root: Path | None = None) -> Path:
    """Resolve an output directory and enforce the repository artifact boundary.

    Every output must stay under the validated configured artifact root. External
    temporary directories remain valid when they are configured as that root. The
    default run path has the narrower continual-adaptation diagnostic namespace as
    its required root so a path-valued ``run_id`` cannot reach a sibling namespace.

    Args:
        output_dir: Requested diagnostic output directory.
        namespace_root: Optional narrower namespace that must contain the output.

    Returns:
        The resolved output directory.

    Raises:
        ContinualAdaptationProtocolError: when the artifact root is invalid or the
            output escapes its required boundary.
    """
    resolved = Path(output_dir).expanduser().resolve()
    artifact_root = _validated_artifact_root()
    required_root = (
        Path(namespace_root).expanduser().resolve() if namespace_root is not None else artifact_root
    )
    try:
        required_root.relative_to(artifact_root)
    except ValueError:
        raise ContinualAdaptationProtocolError(
            [
                f"diagnostic output namespace {required_root} escapes the configured "
                f"artifact root {artifact_root}"
            ]
        ) from None

    try:
        resolved.relative_to(required_root)
    except ValueError:
        boundary_name = (
            "diagnostic output namespace"
            if namespace_root is not None
            else "configured artifact root"
        )
        raise ContinualAdaptationProtocolError(
            [
                f"output_dir {resolved} escapes the {boundary_name} {required_root}; "
                "diagnostic outputs must remain below that boundary"
            ]
        ) from None

    return resolved


def _write_json_atomically(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON without following a pre-existing destination symlink.

    The temporary file is created beside the destination and atomically replaces
    the final path. ``os.replace`` replaces a symlink at the destination itself;
    it does not follow the symlink to an external target.

    Args:
        path: Destination path below the validated output directory.
        payload: JSON-serializable payload to write.
    """
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def build_adaptation_diagnostic(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Build the diagnostic record for the bounded adaptation.

    The record echoes the declared mutable parameter prefixes, the bounded
    experience budget, and the adaptation scenario IDs. It is a diagnostic only:
    no training is executed, no checkpoint is written, and the safety wrapper is
    not mutated.

    Args:
        manifest: A schema-valid ``continual_adaptation_run.v1`` mapping.

    Returns:
        A JSON-safe diagnostic record stamped with the evidence boundary.
    """
    adaptation = manifest["adaptation"]
    budget = adaptation["experience_budget"]
    return {
        "evidence_boundary": CONTINUAL_ADAPTATION_EVIDENCE_BOUNDARY,
        "status": DIAGNOSTIC_STATUS_NOT_EXECUTED,
        "allowed_parameters": [str(p) for p in adaptation["allowed_parameters"]],
        "experience_budget": {
            "bounded": bool(budget["bounded"]),
            "steps": budget["steps"],
            "units": str(budget["units"]),
        },
        "adaptation_scenarios": [str(s) for s in manifest["scenarios"]["adaptation"]],
        "training_executed": False,
        "checkpoint_written": False,
        "safety_wrapper_mutated": False,
    }


def build_evaluation_diagnostics(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Build the nominal/shift/forgetting diagnostic evaluation records.

    Each record echoes the held-out evaluation scenario IDs and the pre-declared
    acceptance threshold for its surface; the ``shift`` surface additionally
    echoes the declared synthetic shifts. No metric is computed and no record is
    evidence.

    Args:
        manifest: A schema-valid ``continual_adaptation_run.v1`` mapping.

    Returns:
        A mapping of surface name (``nominal``/``shift``/``forgetting``) to a
        JSON-safe diagnostic record stamped with the evidence boundary.
    """
    evaluation_scenarios = [str(s) for s in manifest["scenarios"]["evaluation"]]
    thresholds = manifest["thresholds"]
    evaluations: dict[str, Any] = {}
    for surface in EVALUATION_SURFACES:
        threshold = thresholds[surface]
        record: dict[str, Any] = {
            "evidence_boundary": CONTINUAL_ADAPTATION_EVIDENCE_BOUNDARY,
            "status": DIAGNOSTIC_STATUS_NOT_EXECUTED,
            "evaluation_scenarios": list(evaluation_scenarios),
            "threshold": {
                "metric": str(threshold["metric"]),
                "bound": threshold["bound"],
                "direction": str(threshold["direction"]),
            },
            "metric_computed": False,
            "evidence": False,
        }
        if surface == "shift":
            record["shifts"] = [
                {
                    "id": str(shift["id"]),
                    "kind": str(shift["kind"]),
                    "parameters": shift.get("parameters", {}),
                }
                for shift in manifest["shifts"]
            ]
        evaluations[surface] = record
    return evaluations


def run_continual_adaptation_diagnostics(
    manifest: Mapping[str, Any],
    *,
    source: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> ContinualAdaptationDiagnosticReport:
    """Gate on the protocol contract and write diagnostic-only outputs.

    The manifest is checked with :func:`check_continual_adaptation_run`. When the
    report is not ``protocol_status == 'valid'`` the launcher fails closed by
    raising :class:`ContinualAdaptationProtocolError` and writes nothing.
    Otherwise it writes ``adaptation.json``, ``nominal.json``, ``shift.json``,
    ``forgetting.json``, and an aggregate ``report.json`` under ``output_dir``,
    each stamped with :data:`CONTINUAL_ADAPTATION_EVIDENCE_BOUNDARY`.

    The launcher emits no promotion decision, generates no evidence bundle, and
    makes no benchmark or paper claim regardless of the manifest's own
    ``promotion_decision``.

    Args:
        manifest: A ``continual_adaptation_run.v1`` mapping. Schema-validated by
            the protocol checker.
        source: Optional source path used only for error messages.
        output_dir: Diagnostic output directory below the configured artifact root. Defaults to
            ``<artifact_root>/continual_adaptation_diagnostics/<run_id>``.
            Repository-local artifact roots must remain under canonical ``output/``.

    Returns:
        The diagnostic report describing the (un-executed) bounded adaptation and
        evaluation surfaces.

    Raises:
        ContinualAdaptationProtocolError: when the manifest violates the schema
            or the protocol check is not ``valid`` (fail-closed).
    """
    report = check_continual_adaptation_run(manifest, source=source)
    if report.protocol_status != PROTOCOL_STATUS_VALID:
        raise ContinualAdaptationProtocolError(report.blockers, source=source)

    adaptation = build_adaptation_diagnostic(manifest)
    evaluations = build_evaluation_diagnostics(manifest)

    run_id = str(manifest["run_id"])
    diagnostic_output_root = get_continual_adaptation_output_root()
    target_dir = _resolve_output_dir(
        Path(output_dir) if output_dir is not None else diagnostic_output_root / run_id,
        namespace_root=diagnostic_output_root if output_dir is None else None,
    )
    target_dir.mkdir(parents=True, exist_ok=True)

    output_files = [
        str(target_dir / f"{name}.json") for name in ("adaptation", *EVALUATION_SURFACES, "report")
    ]
    diagnostic_report = ContinualAdaptationDiagnosticReport(
        schema_version=CONTINUAL_ADAPTATION_RUN_SCHEMA_VERSION,
        run_id=run_id,
        issue=int(manifest["issue"]),
        evidence_boundary=CONTINUAL_ADAPTATION_EVIDENCE_BOUNDARY,
        launcher_mode=CONTINUAL_ADAPTATION_LAUNCHER_MODE,
        baseline_policy_identifier=report.baseline_policy_identifier,
        derived_adapted_policy_identifier=report.derived_adapted_policy_identifier,
        protocol_status=report.protocol_status,
        emits_promotion_decision=False,
        evidence_bundle_generated=False,
        makes_benchmark_or_paper_claim=False,
        adaptation=adaptation,
        evaluations=evaluations,
        output_files=output_files,
    )

    payloads: dict[str, dict[str, Any]] = {
        "adaptation": adaptation,
        "nominal": evaluations["nominal"],
        "shift": evaluations["shift"],
        "forgetting": evaluations["forgetting"],
        "report": diagnostic_report.to_dict(),
    }
    for name, payload in payloads.items():
        _write_json_atomically(target_dir / f"{name}.json", payload)

    logger.info(
        "Wrote continual-adaptation diagnostics",
        run_id=run_id,
        output_dir=str(target_dir),
        evidence_boundary=CONTINUAL_ADAPTATION_EVIDENCE_BOUNDARY,
    )
    return diagnostic_report


def render_markdown(report: ContinualAdaptationDiagnosticReport) -> str:
    """Render a human-readable diagnostic summary.

    Args:
        report: A produced :class:`ContinualAdaptationDiagnosticReport`.

    Returns:
        A Markdown string. The summary states the diagnostic-only boundary and
        never asserts an executed adaptation, a promotion, or evidence.
    """
    budget = report.adaptation["experience_budget"]
    lines = [
        f"# Continual-adaptation diagnostic launch: {report.run_id}",
        "",
        f"- Evidence boundary: `{report.evidence_boundary}`",
        f"- Launcher mode: `{report.launcher_mode}`",
        f"- Protocol status: `{report.protocol_status}`",
        f"- Baseline policy: `{report.baseline_policy_identifier}`",
        "- Derived adapted policy (informational only): "
        f"`{report.derived_adapted_policy_identifier}`",
        f"- Emits promotion decision: {report.emits_promotion_decision}",
        f"- Evidence bundle generated: {report.evidence_bundle_generated}",
        f"- Benchmark/paper claim: {report.makes_benchmark_or_paper_claim}",
        "",
        "Diagnostic only: trains nothing, computes no metric, writes no checkpoint, mutates no "
        "safety wrapper, emits no promotion decision, and produces no evidence bundle.",
        "",
        "## Bounded adaptation (diagnostic)",
        f"- Status: `{report.adaptation['status']}`",
        f"- Allowed parameters: {', '.join(report.adaptation['allowed_parameters'])}",
        f"- Experience budget: {budget['steps']} {budget['units']} (bounded={budget['bounded']})",
        "- Adaptation scenarios: " + ", ".join(report.adaptation["adaptation_scenarios"]),
        "",
        "## Evaluation surfaces (diagnostic)",
    ]
    for surface in EVALUATION_SURFACES:
        evaluation = report.evaluations[surface]
        threshold = evaluation["threshold"]
        lines.append(
            f"- {surface}: `{evaluation['status']}`; threshold "
            f"{threshold['metric']} {threshold['direction']} {threshold['bound']}; "
            f"scenarios {', '.join(evaluation['evaluation_scenarios'])}"
        )
    lines.append("")
    lines.append("## Outputs")
    lines.extend(f"- {path}" for path in report.output_files)
    return "\n".join(lines)
