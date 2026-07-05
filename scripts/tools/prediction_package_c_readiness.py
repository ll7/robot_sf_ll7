#!/usr/bin/env python3
"""Issue #3080 readiness helper for closed-loop prediction Package C.

Package C coordinates a same-seed comparison of four forecast arms
(``no_forecast``, ``cv``, ``semantic_cv``, ``interaction_aware``) across three
coordination stages: open-loop forecast analysis (#2915), live observation
perturbation replay (#2777), and closed-loop forecast-risk coupling (#2916).

This helper does not execute campaigns, alter predictor semantics, or claim
forecast performance. It only inspects repository inputs and supplied #2916
artifacts, then reports fail-closed per-arm ``ready`` / ``blocked`` /
``missing`` status.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

SCHEMA_VERSION = "prediction-package-c-readiness.v1"
ISSUE = 3080
ArmStatus = Literal["ready", "blocked", "missing"]

REPO_ROOT = Path(__file__).resolve().parents[2]

CONFIG_OPEN_LOOP = "configs/research/forecast_baseline_comparison_issue_2915.yaml"
CONFIG_CLOSED_LOOP = "configs/research/forecast_risk_coupling_issue_2916.yaml"
SCRIPT_OBSERVATION_REPLAY = "scripts/benchmark/run_observation_noise_envelope.py"
DEFAULT_OBSERVATION_REPLAY_OUTPUT_ROOT = (
    "docs/context/evidence/issue_2755_observation_noise_envelope_2026-06-13"
)
DEFAULT_CLOSED_LOOP_OUTPUT_ROOT = "docs/context/evidence/issue_2916_forecast_risk_coupling_2026-06-23"

REQUIRED_CODE: tuple[str, ...] = (
    "robot_sf/benchmark/forecast_metrics.py",
    "robot_sf/benchmark/forecast_batch.py",
    "robot_sf/benchmark/pedestrian_forecast.py",
    "robot_sf/benchmark/runner.py",
    SCRIPT_OBSERVATION_REPLAY,
    "scripts/tools/campaign_result_store.py",
)
REQUIRED_CONFIGS: tuple[str, ...] = (CONFIG_OPEN_LOOP, CONFIG_CLOSED_LOOP)

RESULT_STORE_SIGNAL_FILE = "summary.json"
COUPLING_REPORT_FILE = "forecast_risk_coupling_gate_report.json"
EXPECTED_COUPLING_ROWS: dict[str, str] = {
    "no_forecast": "none",
    "cv_risk": "constant_velocity",
    "semantic_risk": "semantic_cv",
    "interaction_risk": "interaction_aware_cv",
}
REQUIRED_COUPLING_METRICS: tuple[str, ...] = (
    "collision",
    "near_miss",
    "safety_events",
    "progress_m",
    "stop_yield_timing_steps",
    "false_positive_stops",
    "runtime_s",
    "snqi",
)
VALID_COUPLING_VERDICTS = {"continue", "revise", "stop"}


@dataclass(frozen=True, slots=True)
class PackageCArm:
    """Declarative spec for one Package C forecast arm."""

    arm: str
    forecast_variant: str
    risk_source: str
    baseline_id: str | None
    description: str


ARMS: tuple[PackageCArm, ...] = (
    PackageCArm(
        arm="no_forecast",
        forecast_variant="none",
        risk_source="none",
        baseline_id=None,
        description="Control arm: no forecast risk fed to the gate.",
    ),
    PackageCArm(
        arm="cv",
        forecast_variant="cv",
        risk_source="constant_velocity",
        baseline_id="constant_velocity_gaussian_baseline",
        description="Constant-velocity Gaussian motion-extrapolation forecast.",
    ),
    PackageCArm(
        arm="semantic_cv",
        forecast_variant="semantic",
        risk_source="semantic_cv",
        baseline_id="semantic_cv_baseline",
        description="Semantic CV forecast signal with intent-aware adjustments.",
    ),
    PackageCArm(
        arm="interaction_aware",
        forecast_variant="interaction_aware",
        risk_source="interaction_aware_cv",
        baseline_id="interaction_aware_cv_baseline",
        description="Interaction-aware CV forecast using neighbor context.",
    ),
)


@dataclass(frozen=True, slots=True)
class ArmReadiness:
    """Fail-closed readiness verdict for a Package C arm."""

    arm: str
    forecast_variant: str
    risk_source: str
    baseline_id: str | None
    status: ArmStatus
    reason: str
    present_inputs: list[str] = field(default_factory=list)
    missing_inputs: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class CouplingArtifactReadiness:
    """Validated readiness signal for the Package C #2916 coupling artifact."""

    available: bool
    path: str | None
    blockers: list[str] = field(default_factory=list)


def _exists(repo_root: Path, rel_path: str) -> bool:
    """Return True when ``rel_path`` exists under ``repo_root``."""
    return (repo_root / rel_path).exists()


def _baseline_declared(repo_root: Path, baseline_id: str) -> bool:
    """Return True when ``baseline_id`` is registered in the forecast module."""
    source_path = repo_root / "robot_sf/benchmark/pedestrian_forecast.py"
    if not source_path.exists():
        return False
    return baseline_id in source_path.read_text(encoding="utf-8")


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML config dictionary, returning ``{}`` if absent or malformed."""
    if not path.exists():
        return {}
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object, returning ``{}`` if absent or malformed."""
    if not path.exists():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _collect_seed_plan(repo_root: Path) -> list[int]:
    """Return the union of declared seeds across coordination configs."""
    seeds: set[int] = set()
    open_loop = _load_yaml(repo_root / CONFIG_OPEN_LOOP)
    for seed in open_loop.get("seeds", []) or []:
        if isinstance(seed, int):
            seeds.add(seed)

    closed_loop = _load_yaml(repo_root / CONFIG_CLOSED_LOOP)
    fixture = closed_loop.get("fixture", {})
    if isinstance(fixture, dict) and isinstance(fixture.get("seed"), int):
        seeds.add(fixture["seed"])
    return sorted(seeds)


def _missing_seed_contracts(repo_root: Path) -> list[str]:
    """Return missing same-seed coordination contracts."""
    missing: list[str] = []
    open_loop = _load_yaml(repo_root / CONFIG_OPEN_LOOP)
    open_loop_seeds = {seed for seed in open_loop.get("seeds", []) or [] if isinstance(seed, int)}
    if not open_loop_seeds:
        missing.append(f"{CONFIG_OPEN_LOOP}::seeds")

    closed_loop = _load_yaml(repo_root / CONFIG_CLOSED_LOOP)
    fixture = closed_loop.get("fixture", {})
    closed_loop_seed = fixture.get("seed") if isinstance(fixture, dict) else None
    if not isinstance(closed_loop_seed, int):
        missing.append(f"{CONFIG_CLOSED_LOOP}::fixture.seed")
    elif open_loop_seeds and closed_loop_seed not in open_loop_seeds:
        missing.append(
            f"{CONFIG_CLOSED_LOOP}::fixture.seed={closed_loop_seed} not declared in "
            f"{CONFIG_OPEN_LOOP}::seeds"
        )
    return missing


def _collect_output_roots(repo_root: Path) -> list[str]:
    """Return declared output/evidence roots from Package C coordination configs."""
    roots: list[str] = [
        DEFAULT_OBSERVATION_REPLAY_OUTPUT_ROOT,
        DEFAULT_CLOSED_LOOP_OUTPUT_ROOT,
    ]
    open_loop = _load_yaml(repo_root / CONFIG_OPEN_LOOP)
    output = open_loop.get("output", {})
    if isinstance(output, dict):
        evidence_dir = output.get("evidence_dir")
        if isinstance(evidence_dir, str):
            roots.insert(0, evidence_dir)
    return roots


def _coupling_store_available(coupling_result_store: Path | None) -> bool:
    """Return True when a canonical durable #2916 result-store marker is present."""
    if coupling_result_store is None:
        return False
    return (coupling_result_store / RESULT_STORE_SIGNAL_FILE).exists()


def _resolve_coupling_report_path(
    coupling_report: Path | None,
    coupling_result_store: Path | None,
) -> Path | None:
    """Resolve an explicit #2916 report path or the canonical report in a store."""
    if coupling_report is not None:
        return coupling_report
    if coupling_result_store is None:
        return None
    candidate = coupling_result_store / COUPLING_REPORT_FILE
    return candidate if candidate.exists() else None


def _validate_report_header(report: dict[str, Any], report_path: Path) -> list[str]:
    """Return blockers for top-level #2916 report metadata."""
    blockers: list[str] = []
    if report.get("issue") != 2916:
        blockers.append(f"{report_path}::issue must be 2916")
    if report.get("claim_boundary") != "diagnostic_only":
        blockers.append(f"{report_path}::claim_boundary must be diagnostic_only")
    if report.get("paper_grade") is not False:
        blockers.append(f"{report_path}::paper_grade must be false")
    return blockers


def _validate_report_fixture(
    report: dict[str, Any],
    report_path: Path,
    *,
    repo_seed_plan: list[int],
) -> tuple[list[str], int | None, Any]:
    """Return blockers plus expected fixture seed and scenario."""
    blockers: list[str] = []
    config = report.get("config")
    fixture = config.get("fixture", {}) if isinstance(config, dict) else {}
    fixture_seed = fixture.get("seed") if isinstance(fixture, dict) else None
    if not isinstance(fixture_seed, int):
        blockers.append(f"{report_path}::config.fixture.seed missing")
    elif fixture_seed not in repo_seed_plan:
        blockers.append(
            f"{report_path}::config.fixture.seed={fixture_seed} not in Package C seed plan"
        )
    scenario_id = fixture.get("scenario_id") if isinstance(fixture, dict) else None
    if not scenario_id:
        blockers.append(f"{report_path}::config.fixture.scenario_id missing")
    return blockers, fixture_seed if isinstance(fixture_seed, int) else None, scenario_id


def _validate_report_verdict(report: dict[str, Any], report_path: Path) -> list[str]:
    """Return blockers for the #2916 continue/revise/stop verdict."""
    verdict = report.get("verdict")
    decision = verdict.get("decision") if isinstance(verdict, dict) else None
    if decision not in VALID_COUPLING_VERDICTS:
        return [f"{report_path}::verdict.decision must be continue|revise|stop"]
    return []


def _observed_report_rows(
    report: dict[str, Any], report_path: Path
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    """Return row-shape blockers and row mapping keyed by row name."""
    rows = report.get("rows")
    if not isinstance(rows, list):
        return [f"{report_path}::rows missing"], {}

    observed_rows: dict[str, dict[str, Any]] = {}
    for row in rows:
        if isinstance(row, dict) and isinstance(row.get("row"), str):
            observed_rows[row["row"]] = row
    return [], observed_rows


def _validate_single_report_row(
    row_name: str,
    row: dict[str, Any],
    report_path: Path,
    *,
    expected_risk_source: str,
    fixture_seed: int | None,
    scenario_id: Any,
) -> list[str]:
    """Return blockers for one #2916 coupling row."""
    blockers: list[str] = []
    if row.get("risk_source") != expected_risk_source:
        blockers.append(f"{report_path}::{row_name}.risk_source must be {expected_risk_source}")
    if row.get("seed") != fixture_seed:
        blockers.append(f"{report_path}::{row_name}.seed must match fixture seed")
    if row.get("scenario_id") != scenario_id:
        blockers.append(f"{report_path}::{row_name}.scenario_id must match fixture scenario")
    metrics = row.get("metrics")
    if not isinstance(metrics, dict):
        return [*blockers, f"{report_path}::{row_name}.metrics missing"]
    for metric in REQUIRED_COUPLING_METRICS:
        if metric not in metrics:
            blockers.append(f"{report_path}::{row_name}.metrics.{metric} missing")
    return blockers


def _validate_report_rows(
    observed_rows: dict[str, dict[str, Any]],
    report_path: Path,
    *,
    fixture_seed: int | None,
    scenario_id: Any,
) -> list[str]:
    """Return blockers for expected #2916 rows and primary outcome metrics."""
    blockers: list[str] = []
    missing_rows = sorted(set(EXPECTED_COUPLING_ROWS) - set(observed_rows))
    extra_rows = sorted(set(observed_rows) - set(EXPECTED_COUPLING_ROWS))
    if missing_rows:
        blockers.append(f"{report_path} missing coupling row(s): {missing_rows}")
    if extra_rows:
        blockers.append(f"{report_path} has unexpected coupling row(s): {extra_rows}")

    for row_name, expected_risk_source in EXPECTED_COUPLING_ROWS.items():
        row = observed_rows.get(row_name)
        if row is None:
            continue
        blockers.extend(
            _validate_single_report_row(
                row_name,
                row,
                report_path,
                expected_risk_source=expected_risk_source,
                fixture_seed=fixture_seed,
                scenario_id=scenario_id,
            )
        )
    return blockers


def _validate_coupling_report(
    report_path: Path | None,
    *,
    repo_seed_plan: list[int],
) -> CouplingArtifactReadiness:
    """Validate the #2916 report contract required before Package C assembly."""
    if report_path is None:
        return CouplingArtifactReadiness(available=False, path=None)

    blockers: list[str] = []
    report = _load_json(report_path)
    if not report:
        return CouplingArtifactReadiness(
            available=False,
            path=str(report_path),
            blockers=[f"{report_path} missing or not a JSON object"],
        )

    blockers.extend(_validate_report_header(report, report_path))
    fixture_blockers, fixture_seed, scenario_id = _validate_report_fixture(
        report,
        report_path,
        repo_seed_plan=repo_seed_plan,
    )
    blockers.extend(fixture_blockers)
    blockers.extend(_validate_report_verdict(report, report_path))
    row_blockers, observed_rows = _observed_report_rows(report, report_path)
    blockers.extend(row_blockers)
    blockers.extend(
        _validate_report_rows(
            observed_rows,
            report_path,
            fixture_seed=fixture_seed,
            scenario_id=scenario_id,
        )
    )

    return CouplingArtifactReadiness(
        available=not blockers,
        path=str(report_path),
        blockers=blockers,
    )


def assess_arm(
    arm: PackageCArm,
    repo_root: Path,
    *,
    coupling_artifact: CouplingArtifactReadiness,
) -> ArmReadiness:
    """Return the fail-closed readiness verdict for one arm."""
    required = list(REQUIRED_CONFIGS) + list(REQUIRED_CODE)
    present = [rel for rel in required if _exists(repo_root, rel)]
    missing = [rel for rel in required if not _exists(repo_root, rel)]
    if not missing:
        missing.extend(_missing_seed_contracts(repo_root))

    if arm.baseline_id is not None and not _baseline_declared(repo_root, arm.baseline_id):
        missing.append(f"robot_sf/benchmark/pedestrian_forecast.py::{arm.baseline_id}")

    if missing:
        return ArmReadiness(
            arm=arm.arm,
            forecast_variant=arm.forecast_variant,
            risk_source=arm.risk_source,
            baseline_id=arm.baseline_id,
            status="missing",
            reason="required Package C input(s) absent; cannot preflight this arm",
            present_inputs=sorted(present),
            missing_inputs=sorted(missing),
            blockers=[],
        )

    if not coupling_artifact.available:
        blocker = (
            "#2916 closed-loop forecast-risk coupling has not supplied a durable, "
            "validated coupling report/result store; supply --coupling-report or "
            "--coupling-result-store once #2916 durable artifacts are available"
        )
        return ArmReadiness(
            arm=arm.arm,
            forecast_variant=arm.forecast_variant,
            risk_source=arm.risk_source,
            baseline_id=arm.baseline_id,
            status="blocked",
            reason="inputs wired; Package C assembly gated on #2916 durable artifacts",
            present_inputs=sorted(present),
            missing_inputs=[],
            blockers=[blocker, *coupling_artifact.blockers],
        )

    return ArmReadiness(
        arm=arm.arm,
        forecast_variant=arm.forecast_variant,
        risk_source=arm.risk_source,
        baseline_id=arm.baseline_id,
        status="ready",
        reason="inputs wired and validated #2916 coupling report present",
        present_inputs=sorted(present),
        missing_inputs=[],
        blockers=[],
    )


def assess_package_c_readiness(
    repo_root: Path | None = None,
    *,
    coupling_result_store: Path | None = None,
    coupling_report: Path | None = None,
) -> dict[str, Any]:
    """Assess Package C readiness for all four arms."""
    root = repo_root or REPO_ROOT
    seed_plan = _collect_seed_plan(root)
    coupling_store_available = _coupling_store_available(coupling_result_store)
    coupling_report_path = _resolve_coupling_report_path(coupling_report, coupling_result_store)
    coupling_artifact = _validate_coupling_report(
        coupling_report_path,
        repo_seed_plan=seed_plan,
    )
    if coupling_store_available and coupling_report_path is None:
        coupling_artifact = CouplingArtifactReadiness(
            available=True,
            path=str(coupling_result_store / RESULT_STORE_SIGNAL_FILE),
        )

    arms = [assess_arm(arm, root, coupling_artifact=coupling_artifact) for arm in ARMS]
    if any(arm.status == "missing" for arm in arms):
        overall: ArmStatus = "missing"
    elif any(arm.status == "blocked" for arm in arms):
        overall = "blocked"
    else:
        overall = "ready"

    return {
        "schema_version": SCHEMA_VERSION,
        "issue": ISSUE,
        "claim_boundary": (
            "coordination preflight only; no benchmark campaign executed and no forecast "
            "or paper-facing performance claim made"
        ),
        "overall_status": overall,
        "seed_plan": seed_plan,
        "output_roots": _collect_output_roots(root),
        "coupling_result_store_available": coupling_store_available,
        "coupling_report_available": coupling_artifact.available,
        "coupling_report_path": coupling_artifact.path,
        "coupling_report_blockers": coupling_artifact.blockers,
        "required_configs": list(REQUIRED_CONFIGS),
        "required_code": list(REQUIRED_CODE),
        "arms": [asdict(arm) for arm in arms],
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Render a compact Markdown summary readiness report."""
    lines = [
        f"# Issue #{report['issue']}: Prediction Package C Readiness Preflight",
        "",
        f"**Claim boundary:** {report['claim_boundary']}",
        "",
        f"- **Overall status:** `{report['overall_status']}`",
        f"- **Seed plan (same-seed):** {report['seed_plan']}",
        f"- **Output roots:** {report['output_roots'] or '(none declared)'}",
        f"- **Coupling result store available:** {report['coupling_result_store_available']}",
        f"- **Coupling report available:** {report['coupling_report_available']}",
        f"- **Coupling report path:** {report['coupling_report_path'] or '(none supplied)'}",
        "",
        "## Arms",
        "",
        "| arm | variant | risk_source | baseline | status |",
        "| --- | --- | --- | --- | --- |",
    ]
    for arm in report["arms"]:
        lines.append(
            f"| {arm['arm']} | {arm['forecast_variant']} | {arm['risk_source']} | "
            f"{arm['baseline_id'] or '-'} | `{arm['status']}` |"
        )
    blocked = [arm for arm in report["arms"] if arm["blockers"]]
    if blocked:
        lines += ["", "## Blockers", ""]
        for arm in blocked:
            for blocker in arm["blockers"]:
                lines.append(f"- `{arm['arm']}`: {blocker}")
    return "\n".join(lines)


def write_report_outputs(
    report: dict[str, Any],
    *,
    output_json: Path | None = None,
    output_markdown: Path | None = None,
) -> None:
    """Persist the readiness report to the requested durable output paths."""
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if output_markdown is not None:
        output_markdown.parent.mkdir(parents=True, exist_ok=True)
        output_markdown.write_text(render_markdown(report) + "\n", encoding="utf-8")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--coupling-result-store",
        type=Path,
        default=None,
        help="Path to durable #2916 campaign result store.",
    )
    parser.add_argument(
        "--coupling-report",
        type=Path,
        default=None,
        help="Path to #2916 forecast-risk coupling gate report JSON.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write readiness report JSON.",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=None,
        help="Optional path to write readiness report Markdown.",
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Print Markdown summary instead of JSON.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run Package C readiness preflight from the CLI."""
    args = _parse_args(argv)
    report = assess_package_c_readiness(
        coupling_result_store=args.coupling_result_store,
        coupling_report=args.coupling_report,
    )

    write_report_outputs(
        report,
        output_json=args.output_json,
        output_markdown=args.output_markdown,
    )

    if args.markdown:
        print(render_markdown(report))
    else:
        print(json.dumps(report, indent=2))
    return 0 if report["overall_status"] == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
