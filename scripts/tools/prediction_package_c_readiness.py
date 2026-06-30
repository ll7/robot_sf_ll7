#!/usr/bin/env python3
"""Issue #3080: readiness / preflight helper for closed-loop prediction Package C.

Package C coordinates a same-seed comparison of four forecast arms
(``no_forecast``, ``cv``, ``semantic_cv``, ``interaction_aware``) across three
coordination stages: open-loop forecast analysis (#2915), live observation
perturbation replay (#2777), and closed-loop forecast-risk coupling (#2916).

This helper does **not** execute any benchmark campaign, alter predictor
semantics, or claim forecast performance.  It only inspects the repository for
the inputs each arm needs and reports a fail-closed ``ready`` / ``blocked`` /
``missing`` status per arm, together with the required configs, the declared
seed plan, the declared output roots, and the named blockers.

Status vocabulary (fail-closed):

- ``missing``  - a required config or code entry point is absent on disk.
- ``blocked``  - all inputs are present, but a named external blocker (the
  #2916 closed-loop coupling execution producing a durable result store) has
  not cleared yet.  This is the default for Package C per the issue-audit on
  2026-06-22: the comparison is gated solely on #2916 execution producing
  durable artifacts.
- ``ready``    - all inputs present and the coupling result store is available.

Usage::

    uv run python scripts/tools/prediction_package_c_readiness.py
    uv run python scripts/tools/prediction_package_c_readiness.py \\
        --coupling-result-store output/issue_2916_coupling/result_store
    uv run python scripts/tools/prediction_package_c_readiness.py \\
        --output-json output/issue_3080_readiness/summary.json
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
ALLOWED_ARM_STATUSES: tuple[ArmStatus, ...] = ("ready", "blocked", "missing")

REPO_ROOT = Path(__file__).resolve().parents[2]

# Open-loop forecast comparison (#2915) config.
CONFIG_OPEN_LOOP = "configs/research/forecast_baseline_comparison_issue_2915.yaml"
# Closed-loop forecast-risk coupling (#2916) config.
CONFIG_CLOSED_LOOP = "configs/research/forecast_risk_coupling_issue_2916.yaml"
# Live observation perturbation replay entry point (#2777).
SCRIPT_OBSERVATION_REPLAY = "scripts/benchmark/run_observation_noise_envelope.py"
DEFAULT_OBSERVATION_REPLAY_OUTPUT_ROOT = (
    "docs/context/evidence/issue_2755_observation_noise_envelope_2026-06-13"
)
DEFAULT_CLOSED_LOOP_OUTPUT_ROOT = "output/issue_2916_coupling_gate"

# Required code entry points shared by every arm.  Each maps to a coordination
# stage named in the issue's verified entry points.
REQUIRED_CODE: tuple[str, ...] = (
    "robot_sf/benchmark/forecast_metrics.py",  # open-loop metrics (#2915)
    "robot_sf/benchmark/forecast_batch.py",  # ForecastBatch.v1 artifact (#2915)
    "robot_sf/benchmark/pedestrian_forecast.py",  # CV-family baselines (arms)
    "robot_sf/benchmark/runner.py",  # closed-loop coupling harness (#2916)
    SCRIPT_OBSERVATION_REPLAY,  # observation perturbation replay (#2777)
    "scripts/tools/campaign_result_store.py",  # canonical result store
)

# Required configs shared by every arm.
REQUIRED_CONFIGS: tuple[str, ...] = (CONFIG_OPEN_LOOP, CONFIG_CLOSED_LOOP)

# Canonical file that signals a durable campaign result store exists; mirrors
# scripts/tools/campaign_result_store.REQUIRED_STORE_FILES without importing the
# pandas-backed module.
RESULT_STORE_SIGNAL_FILE = "summary.json"


@dataclass(frozen=True, slots=True)
class PackageCArm:
    """Declarative spec for one Package C forecast arm.

    ``baseline_id`` is the canonical deterministic CV-family baseline in
    ``robot_sf/benchmark/pedestrian_forecast.py``; ``None`` for the no-forecast
    control arm, which feeds no forecast risk to the gate.
    """

    arm: str
    forecast_variant: str
    risk_source: str
    baseline_id: str | None
    description: str


# The four Package C arms, in comparison order.  Variant / risk-source / baseline
# names mirror configs/research/forecast_risk_coupling_issue_2916.yaml so the
# preflight stays aligned with the coupling rows that actually execute.
ARMS: tuple[PackageCArm, ...] = (
    PackageCArm(
        arm="no_forecast",
        forecast_variant="none",
        risk_source="none",
        baseline_id=None,
        description="Control arm: no forecast risk fed to the gate (baseline outcome).",
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
        description="Semantic CV forecast with signal/intent-aware adjustments.",
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
    """Fail-closed readiness verdict for a single Package C arm."""

    arm: str
    forecast_variant: str
    risk_source: str
    baseline_id: str | None
    status: ArmStatus
    reason: str
    present_inputs: list[str] = field(default_factory=list)
    missing_inputs: list[str] = field(default_factory=list)
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
    """Load a YAML config to a dict, returning {} when absent or malformed."""
    if not path.exists():
        return {}
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _collect_seed_plan(repo_root: Path) -> list[int]:
    """Return the sorted union of declared seeds across both coordination configs.

    The open-loop config (#2915) declares a ``seeds`` list; the coupling config
    (#2916) declares a single ``fixture.seed``.  Package C requires identical
    seeds across arms, so the union documents the same-seed plan to honor.
    """
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
    """Return missing same-seed contract fields from coordination configs."""
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
            f"{CONFIG_CLOSED_LOOP}::fixture.seed={closed_loop_seed} "
            f"not declared in {CONFIG_OPEN_LOOP}::seeds"
        )

    return missing


def _collect_output_roots(repo_root: Path) -> list[str]:
    """Return declared output/evidence roots from the coordination configs.

    Package C spans three stages. The open-loop config (#2915) declares its
    evidence directory in YAML, while observation replay (#2777) and closed-loop
    coupling (#2916) expose default runner roots. Local ``output/`` paths are
    not durable evidence; durable results must point at tracked evidence, a
    manifest, or an approved external artifact URI.
    """
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
    """Return True when a durable #2916 coupling result store is present.

    A durable store is signalled by the canonical ``summary.json`` file, the
    same marker the campaign result store writes.  Absence keeps Package C
    fail-closed at ``blocked`` rather than ``ready``.
    """
    if coupling_result_store is None:
        return False
    return (coupling_result_store / RESULT_STORE_SIGNAL_FILE).exists()


def assess_arm(
    arm: PackageCArm,
    repo_root: Path,
    *,
    coupling_store_available: bool,
) -> ArmReadiness:
    """Return the fail-closed readiness verdict for one arm.

    Resolution order is fail-closed: any missing required input yields
    ``missing``; otherwise an uncleared coupling blocker yields ``blocked``;
    only a fully wired arm with a durable coupling store yields ``ready``.
    """
    required = list(REQUIRED_CONFIGS) + list(REQUIRED_CODE)
    present = [rel for rel in required if _exists(repo_root, rel)]
    missing = [rel for rel in required if not _exists(repo_root, rel)]
    if not missing:
        missing.extend(_missing_seed_contracts(repo_root))

    # The arm-specific deterministic baseline must be registered (control arm
    # has no baseline and is always satisfied).
    if arm.baseline_id is not None and not _baseline_declared(repo_root, arm.baseline_id):
        missing.append(
            f"robot_sf/benchmark/pedestrian_forecast.py::{arm.baseline_id}",
        )

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

    if not coupling_store_available:
        blocker = (
            "#2916 closed-loop forecast-risk coupling has not produced a durable "
            "campaign result store (issue-audit 2026-06-22); supply "
            "--coupling-result-store once #2916 lands durable artifacts"
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
            blockers=[blocker],
        )

    return ArmReadiness(
        arm=arm.arm,
        forecast_variant=arm.forecast_variant,
        risk_source=arm.risk_source,
        baseline_id=arm.baseline_id,
        status="ready",
        reason="inputs wired and durable #2916 coupling store present",
        present_inputs=sorted(present),
        missing_inputs=[],
        blockers=[],
    )


def assess_package_c_readiness(
    repo_root: Path | None = None,
    *,
    coupling_result_store: Path | None = None,
) -> dict[str, Any]:
    """Assess Package C preflight readiness for all four arms.

    Returns a JSON-serializable report with per-arm verdicts, the declared
    same-seed plan, the declared output roots, the required inputs, and an
    overall fail-closed status.  This inspects the repository only; it never
    executes a benchmark campaign or claims forecast performance.
    """
    root = repo_root or REPO_ROOT
    coupling_available = _coupling_store_available(coupling_result_store)

    arms = [assess_arm(arm, root, coupling_store_available=coupling_available) for arm in ARMS]

    if any(a.status == "missing" for a in arms):
        overall: ArmStatus = "missing"
    elif any(a.status == "blocked" for a in arms):
        overall = "blocked"
    else:
        overall = "ready"

    return {
        "schema_version": SCHEMA_VERSION,
        "issue": ISSUE,
        "claim_boundary": (
            "Preflight/readiness inventory only. Does NOT execute a benchmark "
            "campaign, alter predictor semantics, or claim forecast performance. "
            "Statuses are fail-closed: blocked/missing are never success evidence."
        ),
        "overall_status": overall,
        "allowed_arm_statuses": list(ALLOWED_ARM_STATUSES),
        "coordination_stages": {
            "open_loop_forecast": {"issue": 2915, "config": CONFIG_OPEN_LOOP},
            "observation_perturbation_replay": {"issue": 2777, "script": SCRIPT_OBSERVATION_REPLAY},
            "closed_loop_risk_coupling": {"issue": 2916, "config": CONFIG_CLOSED_LOOP},
        },
        "required_configs": list(REQUIRED_CONFIGS),
        "required_code": list(REQUIRED_CODE),
        "seed_plan": _collect_seed_plan(root),
        "output_roots": _collect_output_roots(root),
        "coupling_result_store_available": coupling_available,
        "arms": [asdict(a) for a in arms],
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Render a compact Markdown summary of a readiness report."""
    lines = [
        f"# Issue #{report['issue']}: Prediction Package C Readiness Preflight",
        "",
        f"**Claim boundary:** {report['claim_boundary']}",
        "",
        f"- **Overall status:** `{report['overall_status']}`",
        f"- **Seed plan (same-seed):** {report['seed_plan']}",
        f"- **Output roots:** {report['output_roots'] or '(none declared)'}",
        f"- **Coupling result store available:** {report['coupling_result_store_available']}",
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
    blocked = [a for a in report["arms"] if a["blockers"]]
    if blocked:
        lines += ["", "## Blockers", ""]
        for arm in blocked:
            for blocker in arm["blockers"]:
                lines.append(f"- `{arm['arm']}`: {blocker}")
    return "\n".join(lines)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--coupling-result-store",
        type=Path,
        default=None,
        help="Path to a durable #2916 coupling campaign result store.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write the readiness report JSON.",
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Print a Markdown summary instead of JSON.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the Package C readiness preflight from the CLI.

    Returns 0 only when every arm is ``ready``; a fail-closed ``blocked`` or
    ``missing`` overall status returns 1 so the preflight can gate later steps.
    """
    args = _parse_args(argv)
    report = assess_package_c_readiness(coupling_result_store=args.coupling_result_store)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    if args.markdown:
        print(render_markdown(report))
    else:
        print(json.dumps(report, indent=2, sort_keys=True))

    return 0 if report["overall_status"] == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
