"""Capability inventory / preflight checker for the forecast research lane.

The forecast lane (epic #2835) is assembled from many independently landed
components: the ``ForecastBatch.v1`` contract and its JSON schema, a
predictor-agnostic metric surface, deterministic baselines, calibration and
conformal diagnostics, dataset recording, observation-tier adapters, the
forecast-to-risk adapter, the transferability stress matrix, and the live
same-seed closed-loop replay gate.

Because those pieces land across many PRs and machines, it is easy to lose track
of which lane surfaces are actually importable on a given checkout, and which are
missing or broken. This module is a **read-only, fail-closed surface detector**:
it declares the canonical lane components, probes each one (file presence, import,
required public symbols), and reports missing capabilities as explicit blockers.

It deliberately does **not** run predictors, training, benchmarks, or any
forecast evaluation. It only introspects what is present, so it is safe to run on
a shared machine. It composes the canonical owner modules under
``robot_sf/benchmark/`` rather than re-deriving any forecast capability.

Run it as a preflight before forecast-lane work::

    python -m robot_sf.benchmark.forecast_lane_inventory
    python -m robot_sf.benchmark.forecast_lane_inventory --json

Exit code ``0`` means every **required** lane capability is present and
importable. A non-zero exit means at least one required capability is missing or
broken; the report names the blocker and its canonical owner so the gap can be
fixed before downstream forecast work depends on it. Optional (gated/expansion)
capabilities never fail the check; they are reported as informational rows.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Repo root, so the schema-file probe works regardless of the caller's CWD.
_REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ForecastCapabilitySpec:
    """Declared expectation for one forecast-lane capability.

    Attributes:
        capability_id: Stable identifier for the lane surface.
        sublane: Coarse grouping (contract, metrics, baselines, ...).
        module: Importable module path that owns the capability.
        symbols: Public names that must exist on ``module``.
        required: When True a missing/broken capability fails the preflight.
            Optional rows are gated/expansion surfaces reported as informational.
        files: Repo-root-relative files that must exist (schema JSON, scripts).
        owner: Human-readable pointer to fix when the capability is missing.
        note: Short description of why the capability matters to the lane.
    """

    capability_id: str
    sublane: str
    module: str
    symbols: tuple[str, ...]
    required: bool
    files: tuple[str, ...] = ()
    owner: str = ""
    note: str = ""


@dataclass(frozen=True)
class ForecastLaneStatusSpec:
    """One checked-progress row for the forecast research lane epic."""

    requirement_id: str
    requirement: str
    current_artifacts: tuple[str, ...]
    status: str
    remaining_blocker: str
    next_action: str
    learned_predictor_gate: bool = False

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-serializable status row."""
        return {
            "requirement_id": self.requirement_id,
            "requirement": self.requirement,
            "current_artifacts": list(self.current_artifacts),
            "status": self.status,
            "remaining_blocker": self.remaining_blocker,
            "next_action": self.next_action,
            "learned_predictor_gate": self.learned_predictor_gate,
        }


# Canonical forecast-lane registry. Each entry points at an existing owner module
# under robot_sf/benchmark/; this inventory never reimplements those surfaces.
_FORECAST_CAPABILITIES: tuple[ForecastCapabilitySpec, ...] = (
    ForecastCapabilitySpec(
        capability_id="forecast_batch_contract",
        sublane="contract",
        module="robot_sf.benchmark.forecast_batch",
        symbols=(
            "FORECAST_BATCH_SCHEMA_VERSION",
            "ForecastBatch",
            "ActorForecast",
            "ForecastBatchProvenance",
            "CoordinateFrame",
            "load_forecast_batch",
            "save_forecast_batch",
            "validate_forecast_batch",
        ),
        required=True,
        owner="robot_sf/benchmark/forecast_batch.py",
        note="ForecastBatch.v1 typed contract and provenance dataclasses.",
    ),
    ForecastCapabilitySpec(
        capability_id="forecast_batch_schema",
        sublane="contract",
        module="robot_sf.benchmark.schemas.forecast_batch_schema",
        symbols=("ForecastBatchSchema",),
        required=True,
        files=("robot_sf/benchmark/schemas/forecast_batch.schema.v1.json",),
        owner="robot_sf/benchmark/schemas/forecast_batch_schema.py",
        note="JSON-schema validator for ForecastBatch.v1 artifacts.",
    ),
    ForecastCapabilitySpec(
        capability_id="forecast_metrics",
        sublane="metrics",
        module="robot_sf.benchmark.forecast_metrics",
        symbols=(
            "ForecastMetricRow",
            "evaluate_forecast_batch",
            "format_forecast_metrics_markdown",
        ),
        required=True,
        owner="robot_sf/benchmark/forecast_metrics.py",
        note="Predictor-agnostic forecast metric evaluation surface.",
    ),
    ForecastCapabilitySpec(
        capability_id="deterministic_baselines",
        sublane="baselines",
        module="robot_sf.benchmark.pedestrian_forecast",
        symbols=(
            "BASELINE_FUNCTIONS",
            "compute_batch_forecast_metrics",
            "is_pedestrian_actor",
        ),
        required=True,
        owner="robot_sf/benchmark/pedestrian_forecast.py",
        note="Deterministic pedestrian forecast baselines (CV and variants).",
    ),
    ForecastCapabilitySpec(
        capability_id="baseline_comparison",
        sublane="baselines",
        module="robot_sf.benchmark.forecast_baseline_comparison",
        symbols=("compare_forecast_baselines", "ForecastBaselineComparison"),
        required=True,
        files=("scripts/benchmark/run_forecast_baseline_comparison.py",),
        owner="robot_sf/benchmark/forecast_baseline_comparison.py",
        note="Baseline comparison/leaderboard core required before learned models.",
    ),
    ForecastCapabilitySpec(
        capability_id="observation_adapters",
        sublane="observation",
        module="robot_sf.benchmark.forecast_observation_adapters",
        symbols=(
            "ForecastObservationAdapter",
            "OracleFullStateForecastAdapter",
            "TrackedAgentsForecastAdapter",
            "build_constant_velocity_forecast_batch",
        ),
        required=True,
        owner="robot_sf/benchmark/forecast_observation_adapters.py",
        note="Observation-tier adapters separating oracle from tracked inputs.",
    ),
    ForecastCapabilitySpec(
        capability_id="dataset_recorder",
        sublane="dataset",
        module="robot_sf.benchmark.forecast_dataset_recorder",
        symbols=(
            "record_forecast_dataset_from_trace_exports",
            "validate_forecast_dataset_manifest",
        ),
        required=True,
        owner="robot_sf/benchmark/forecast_dataset_recorder.py",
        note="Forecast dataset recorder and split-manifest helpers.",
    ),
    ForecastCapabilitySpec(
        capability_id="calibration_report",
        sublane="calibration",
        module="robot_sf.benchmark.forecast_calibration_report",
        symbols=(
            "build_forecast_calibration_report",
            "format_forecast_calibration_markdown",
        ),
        required=True,
        owner="robot_sf/benchmark/forecast_calibration_report.py",
        note="Calibration/reliability summaries for forecast metric artifacts.",
    ),
    ForecastCapabilitySpec(
        capability_id="conformal_pilot",
        sublane="calibration",
        module="robot_sf.benchmark.forecast_conformal_pilot",
        symbols=("build_forecast_conformal_pilot_report",),
        required=False,
        owner="robot_sf/benchmark/forecast_conformal_pilot.py",
        note="Smoke-only conformal/reachable-set tube diagnostics (pilot).",
    ),
    ForecastCapabilitySpec(
        capability_id="risk_adapter",
        sublane="risk",
        module="robot_sf.benchmark.forecast_risk_adapter",
        symbols=("ForecastRiskSignal", "compute_forecast_risk"),
        required=False,
        owner="robot_sf/benchmark/forecast_risk_adapter.py",
        note="Forecast-to-risk signal adapter (gated; not default behavior).",
    ),
    ForecastCapabilitySpec(
        capability_id="transferability_matrix",
        sublane="transfer",
        module="robot_sf.benchmark.forecast_transferability_stress_matrix",
        symbols=("build_forecast_transferability_stress_matrix",),
        required=False,
        owner="robot_sf/benchmark/forecast_transferability_stress_matrix.py",
        note="Transferability stress matrix (required before paper-facing claims).",
    ),
    ForecastCapabilitySpec(
        capability_id="closed_loop_replay_gate",
        sublane="closed_loop",
        module="robot_sf.benchmark.live_forecast_replay_gate",
        symbols=(
            "LiveForecastReplayGateConfig",
            "classify_live_forecast_replay_run",
            "check_native_live_path_eligibility",
        ),
        required=True,
        files=("scripts/benchmark/run_forecast_risk_coupling_gate.py",),
        owner="robot_sf/benchmark/live_forecast_replay_gate.py",
        note="Same-seed closed-loop coupling gate (the hard expansion boundary).",
    ),
)


_VALID_LANE_STATUSES = frozenset({"done", "partial", "unresolved", "blocked"})


# Checked-progress ledger requested on issue #2835. This is deliberately durable
# research-lane state, not transient queue routing state or campaign lineage.
_FORECAST_LANE_STATUS_ROWS: tuple[ForecastLaneStatusSpec, ...] = (
    ForecastLaneStatusSpec(
        requirement_id="forecast_batch_v1",
        requirement="ForecastBatch.v1 artifact contract",
        current_artifacts=("#2836", "#2849", "robot_sf/benchmark/forecast_batch.py"),
        status="done",
        remaining_blocker="None for the schema/provenance contract.",
        next_action="Keep downstream forecast artifacts on ForecastBatch.v1.",
    ),
    ForecastLaneStatusSpec(
        requirement_id="observation_adapters",
        requirement="Observation-level forecast adapters",
        current_artifacts=("#2838", "#2860", "robot_sf/benchmark/forecast_observation_adapters.py"),
        status="done",
        remaining_blocker="None for oracle/full-state/tracked observation separation.",
        next_action="Extend only when a new deployable observation tier lands.",
    ),
    ForecastLaneStatusSpec(
        requirement_id="motion_rich_traces",
        requirement="Motion-rich forecast-evaluable trace families",
        current_artifacts=("#2774", "#2853", "#2884"),
        status="partial",
        remaining_blocker="Coverage is useful but not a final transfer-aware scenario matrix.",
        next_action="Use fixture gaps from transferability rows to choose the next trace family.",
    ),
    ForecastLaneStatusSpec(
        requirement_id="baseline_ladder",
        requirement="CV, semantic-CV, and interaction-aware baseline ladder",
        current_artifacts=("#2758", "#2781", "#2915"),
        status="partial",
        remaining_blocker="Baseline comparison is diagnostic until same-seed planner consumption is proven.",
        next_action="Keep learned predictors gated until closed-loop rows compare these baselines.",
        learned_predictor_gate=True,
    ),
    ForecastLaneStatusSpec(
        requirement_id="calibration_and_risk",
        requirement="Calibration, reliability, and forecast-risk readiness",
        current_artifacts=("#2841", "#2865", "#2869"),
        status="blocked",
        remaining_blocker="Forecast-risk scoring rows are gated by eligible risk-filtered planner evidence.",
        next_action="Populate risk-scoring-eligible rows through the closed-loop coupling path first.",
        learned_predictor_gate=True,
    ),
    ForecastLaneStatusSpec(
        requirement_id="transferability_matrix",
        requirement="Transferability stress matrix",
        current_artifacts=("#2847", "#2866", "#2887"),
        status="partial",
        remaining_blocker="Some matrix cells remain explicitly unavailable or diagnostic-only.",
        next_action="Fill high-value blocked cells after observation, metric, and fixture owners are ready.",
        learned_predictor_gate=True,
    ),
    ForecastLaneStatusSpec(
        requirement_id="closed_loop_gate",
        requirement="Same-seed closed-loop planner coupling gate",
        current_artifacts=("#2843", "#2902", "#2916", "#2966"),
        status="unresolved",
        remaining_blocker=(
            "Forecast improvement alone has not established non-regressive planner safety/progress."
        ),
        next_action="Run CPU/local gate slices only; no learned-heavy expansion until verdict is continue.",
        learned_predictor_gate=True,
    ),
    ForecastLaneStatusSpec(
        requirement_id="learned_predictor",
        requirement="Learned probabilistic predictor expansion",
        current_artifacts=("#2844", "#2845"),
        status="blocked",
        remaining_blocker="Requires closed-loop gate continue verdict plus calibration/transfer readiness.",
        next_action="Keep to spec/scoping work; do not launch training from this epic.",
        learned_predictor_gate=True,
    ),
    ForecastLaneStatusSpec(
        requirement_id="final_synthesis",
        requirement="Final forecast-lane synthesis",
        current_artifacts=("#2864", "#2881", "#2929"),
        status="partial",
        remaining_blocker="Needs updated synthesis after gate and transfer rows settle.",
        next_action="Consolidate blockers and next empirical action before any claim promotion.",
        learned_predictor_gate=True,
    ),
)


@dataclass
class CapabilityProbeResult:
    """Outcome of probing a single forecast capability.

    ``status`` is one of ``present``, ``missing_module``, ``missing_symbols``,
    or ``missing_files``. ``blockers`` lists the concrete reasons a capability is
    not fully present, so the report can name each gap explicitly.
    """

    capability_id: str
    sublane: str
    required: bool
    status: str
    owner: str
    note: str
    blockers: list[str] = field(default_factory=list)

    @property
    def present(self) -> bool:
        """True when the capability is fully importable with all symbols/files."""
        return self.status == "present"

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view of the probe result."""
        return {
            "capability_id": self.capability_id,
            "sublane": self.sublane,
            "required": self.required,
            "status": self.status,
            "owner": self.owner,
            "note": self.note,
            "blockers": list(self.blockers),
        }


def _probe_capability(spec: ForecastCapabilitySpec, *, repo_root: Path) -> CapabilityProbeResult:
    """Probe one capability: import the module, then check symbols and files.

    Import errors are captured as blockers rather than propagated, so a single
    broken module never aborts the whole inventory. The probe is read-only: it
    imports the module (which forecast modules support without side effects) and
    inspects attributes only.

    Returns:
        The probe result for ``spec`` with its status and any blockers.
    """

    blockers: list[str] = []
    status = "present"

    # Required companion files (schema JSON, runnable scripts).
    for rel_path in spec.files:
        if not (repo_root / rel_path).is_file():
            blockers.append(f"missing file: {rel_path}")
    if blockers:
        status = "missing_files"

    module: Any = None
    try:
        module = importlib.import_module(spec.module)
    except Exception as exc:  # noqa: BLE001 - report any import failure as a blocker
        blockers.append(f"import failed: {spec.module}: {exc!r}")
        return CapabilityProbeResult(
            capability_id=spec.capability_id,
            sublane=spec.sublane,
            required=spec.required,
            status="missing_module",
            owner=spec.owner,
            note=spec.note,
            blockers=blockers,
        )

    missing_symbols = [name for name in spec.symbols if not hasattr(module, name)]
    if missing_symbols:
        blockers.append(f"missing symbols in {spec.module}: {', '.join(missing_symbols)}")
        status = "missing_symbols"

    return CapabilityProbeResult(
        capability_id=spec.capability_id,
        sublane=spec.sublane,
        required=spec.required,
        status=status,
        owner=spec.owner,
        note=spec.note,
        blockers=blockers,
    )


def build_forecast_lane_inventory(*, repo_root: Path | None = None) -> dict[str, Any]:
    """Probe every declared forecast-lane capability and summarize the result.

    Args:
        repo_root: Repository root used to resolve declared companion files.
            Defaults to the repository containing this module.

    Returns:
        A JSON-serializable report with per-capability probe results plus a
        summary. ``ok`` is True only when every **required** capability is
        present; optional capabilities never flip ``ok`` to False.
    """

    root = repo_root if repo_root is not None else _REPO_ROOT
    results = [_probe_capability(spec, repo_root=root) for spec in _FORECAST_CAPABILITIES]

    required_results = [r for r in results if r.required]
    optional_results = [r for r in results if not r.required]
    missing_required = [r for r in required_results if not r.present]
    missing_optional = [r for r in optional_results if not r.present]

    return {
        "schema": "forecast_lane_inventory.v1",
        "ok": not missing_required,
        "capabilities": [r.as_dict() for r in results],
        "summary": {
            "total": len(results),
            "required": len(required_results),
            "required_present": len(required_results) - len(missing_required),
            "required_missing": len(missing_required),
            "optional": len(optional_results),
            "optional_present": len(optional_results) - len(missing_optional),
            "optional_missing": len(missing_optional),
            "missing_required_ids": [r.capability_id for r in missing_required],
            "missing_optional_ids": [r.capability_id for r in missing_optional],
        },
    }


def build_forecast_lane_status() -> dict[str, Any]:
    """Return the checked-progress ledger for forecast-lane issue #2835.

    The status report answers a different question than the capability inventory:
    not "does this module import?", but "which epic requirements are complete
    enough to unblock learned-predictor work?". It remains read-only and does not
    execute campaigns, training, predictors, or benchmark evaluation.
    """
    rows = list(_FORECAST_LANE_STATUS_ROWS)
    invalid_rows = [row for row in rows if row.status not in _VALID_LANE_STATUSES]
    learned_gate_blockers = [
        row
        for row in rows
        if row.learned_predictor_gate and row.status in {"partial", "unresolved", "blocked"}
    ]
    status_counts = {
        status: sum(row.status == status for row in rows) for status in _VALID_LANE_STATUSES
    }
    return {
        "schema": "forecast_lane_status.v1",
        "ok": not invalid_rows,
        "learned_predictor_unblocked": not learned_gate_blockers,
        "issue": 2835,
        "claim_boundary": (
            "Forecast-lane infrastructure is diagnostic until same-seed closed-loop "
            "planner evidence establishes non-regressive safety/progress under explicit "
            "fallback/degraded accounting."
        ),
        "requirements": [row.as_dict() for row in rows],
        "summary": {
            "total": len(rows),
            "status_counts": status_counts,
            "invalid_status_ids": [row.requirement_id for row in invalid_rows],
            "learned_predictor_blocker_ids": [row.requirement_id for row in learned_gate_blockers],
        },
    }


def format_status_markdown(report: dict[str, Any]) -> str:
    """Render compact Markdown for the forecast-lane checked-progress ledger.

    Returns:
        Markdown status table plus learned-predictor blockers when present.
    """
    lines: list[str] = []
    verdict = "UNBLOCKED" if report["learned_predictor_unblocked"] else "BLOCKED"
    lines.append(f"# Forecast lane checked-progress ledger: learned predictor {verdict}")
    lines.append("")
    lines.append(report["claim_boundary"])
    lines.append("")
    lines.append("| Requirement | Status | Current artifact | Remaining blocker | Next action |")
    lines.append("| --- | --- | --- | --- | --- |")
    for row in report["requirements"]:
        artifacts = ", ".join(row["current_artifacts"])
        lines.append(
            f"| {row['requirement']} | {row['status']} | {artifacts} | "
            f"{row['remaining_blocker']} | {row['next_action']} |"
        )
    blocker_ids = report["summary"]["learned_predictor_blocker_ids"]
    if blocker_ids:
        lines.append("")
        lines.append("## Learned-predictor blockers")
        for blocker_id in blocker_ids:
            lines.append(f"- `{blocker_id}`")
    return "\n".join(lines)


def format_inventory_markdown(report: dict[str, Any]) -> str:
    """Render a compact human-readable inventory report.

    Returns:
        A Markdown string with the overall verdict, a per-capability table, and a
        blockers section when any capability is missing.
    """

    lines: list[str] = []
    summary = report["summary"]
    overall = "PASS" if report["ok"] else "FAIL"
    lines.append(f"# Forecast lane capability inventory: {overall}")
    lines.append("")
    lines.append(
        f"Required present: {summary['required_present']}/{summary['required']} | "
        f"Optional present: {summary['optional_present']}/{summary['optional']}"
    )
    lines.append("")
    lines.append("| Capability | Sublane | Required | Status | Owner |")
    lines.append("| --- | --- | --- | --- | --- |")
    for cap in report["capabilities"]:
        flag = "yes" if cap["required"] else "no"
        mark = "ok" if cap["status"] == "present" else cap["status"]
        lines.append(
            f"| {cap['capability_id']} | {cap['sublane']} | {flag} | {mark} | {cap['owner']} |"
        )
    blockers = [
        (cap["capability_id"], blocker)
        for cap in report["capabilities"]
        for blocker in cap["blockers"]
    ]
    if blockers:
        lines.append("")
        lines.append("## Blockers")
        for capability_id, blocker in blockers:
            lines.append(f"- `{capability_id}`: {blocker}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: print inventory or status report and return exit code.

    Returns:
        In default inventory mode, ``0`` if required capability is present,
        otherwise ``1``. In status mode, ``0`` if the ledger itself is valid;
        learned-predictor blockers are reported in the payload but do not make
        the command fail because they are expected issue state.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Read-only forecast-lane capability inventory / status preflight. "
            "Reports missing forecast components and epic blockers explicitly; "
            "does not run predictors, training, or benchmarks."
        )
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the machine-readable report as JSON.",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Emit checked-progress ledger for issue #2835 instead of import inventory.",
    )
    args = parser.parse_args(argv)

    report = build_forecast_lane_status() if args.status else build_forecast_lane_inventory()
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))  # noqa: T201 - CLI output
    elif args.status:
        print(format_status_markdown(report))  # noqa: T201 - CLI output
    else:
        print(format_inventory_markdown(report))  # noqa: T201 - CLI output

    return 0 if report["ok"] else 1


if __name__ == "__main__":  # pragma: no cover - exercised via the CLI test
    sys.exit(main())
