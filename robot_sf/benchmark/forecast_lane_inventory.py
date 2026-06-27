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
    """CLI entry point: print the inventory report and return an exit code.

    Returns:
        ``0`` when every required capability is present, otherwise ``1``.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Read-only forecast-lane capability inventory / preflight. "
            "Reports missing forecast components as explicit blockers; does not "
            "run predictors, training, or benchmarks."
        )
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the machine-readable inventory report as JSON.",
    )
    args = parser.parse_args(argv)

    report = build_forecast_lane_inventory()
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))  # noqa: T201 - CLI output
    else:
        print(format_inventory_markdown(report))  # noqa: T201 - CLI output

    return 0 if report["ok"] else 1


if __name__ == "__main__":  # pragma: no cover - exercised via the CLI test
    sys.exit(main())
