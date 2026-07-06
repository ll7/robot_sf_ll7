#!/usr/bin/env python3
"""Report local prerequisite readiness for the predictive hard-case tournament (#3215).

The epic #3215 runs three hard-case levers concurrently as a tournament -- Selection
(#3204), Authority (#3213), and Model (#3214) -- under a shared benchmark protocol, then
synthesizes a single decision. The full campaign is SLURM/GPU-gated; this helper
deliberately does **not** submit jobs, run the tournament, or rank arms.

Instead it answers a narrow, repeatable question: *are the local prerequisites in place to
launch each arm?* For every arm and for the shared protocol it inventories the expected
configs, harness scripts, and output path, then classifies the arm as ``ready`` (all local
prerequisites present) or ``blocked`` (one or more missing). The report is presence-only and
fail-closed: it never marks tournament execution authorized, even when local
prerequisites are present.

Example:
    uv run python scripts/tools/predictive_tournament_readiness.py
    uv run python scripts/tools/predictive_tournament_readiness.py --json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Shared-protocol prerequisites every arm depends on. Frozen by the launch packet in #3215:
# fixed seed fixture, the campaign harness, the compact result store, the no-fallback algo
# config builder, and the portfolio scenario set used as the map-runner gate.
SHARED_PROTOCOL_PATHS: tuple[Path, ...] = (
    Path("configs/benchmarks/predictive_hard_seeds_v1.yaml"),
    Path("configs/benchmarks/predictive_sweep_planner_grid_v1.yaml"),
    Path("scripts/validation/run_predictive_success_campaign.py"),
    Path("scripts/tools/campaign_result_store.py"),
    Path("robot_sf/benchmark/predictive_planner_config.py"),
    Path("configs/scenarios/sets/predictive_hardcase_portfolio_v1.yaml"),
)

# The tournament run is gated on more than local files: the three child bets must be ready
# and any compute submission must happen outside this read-only helper. These are
# standing run blockers recorded so a "prerequisites ready" report is never mistaken for
# "authorized to launch".
RUN_GATES: tuple[str, ...] = (
    "child bets #3204 / #3213 / #3214 must reach their own decision rules",
    "compute submission must happen outside this read-only helper under #3144 capacity policy",
    "Autonomous Usage Stop Guard must permit unattended execution",
)


@dataclass(frozen=True)
class ArmSpec:
    """Static definition of one tournament arm and its local prerequisites."""

    arm_id: str
    display_name: str
    child_issue: int
    description: str
    # Configs and harness scripts that must exist locally before the arm can be staged.
    required_paths: tuple[Path, ...]
    # Where this arm's campaign results are expected to be written (informational; the path
    # need not exist yet -- the run that creates it is SLURM-gated and out of scope here).
    expected_output_path: Path
    notes: str = ""


@dataclass(frozen=True)
class ProgressionSpec:
    """Static definition for the next #3215 progression packet."""

    packet_id: str
    display_name: str
    description: str
    required_paths: tuple[Path, ...]
    expected_output_path: Path
    forecast_arms: tuple[str, ...]
    outcomes: tuple[str, ...]
    evidence_tier: str
    notes: str = ""


ARMS: tuple[ArmSpec, ...] = (
    ArmSpec(
        arm_id="selection",
        display_name="Selection (proxy-based checkpoint selection)",
        child_issue=3204,
        description=(
            "Proxy-based checkpoint selection vs hard-set success: pick the checkpoint whose "
            "cheap proxy best predicts hard-case success."
        ),
        required_paths=(
            Path("scripts/research/analyze_predictive_checkpoint_proxy.py"),
            Path("configs/research/predictive_checkpoint_proxy_v1.yaml"),
            Path("configs/benchmarks/predictive_hard_seeds_v1.yaml"),
        ),
        expected_output_path=Path("output/tmp/predictive_planner/campaigns/tournament_selection"),
        notes="Proxy analysis is local; the success comparison it selects against is SLURM-gated.",
    ),
    ArmSpec(
        arm_id="authority",
        display_name="Authority (maneuver-authority / action-lattice sweep)",
        child_issue=3213,
        description=(
            "Maneuver-authority / action-lattice / kinematic sweep over the hard-case seeds "
            "to test whether more action authority closes the plateau."
        ),
        required_paths=(
            Path("configs/benchmarks/predictive_hardcase_authority_grid_issue_3213.yaml"),
            Path("configs/algos/hardcase_authority"),
        ),
        expected_output_path=Path("output/tmp/predictive_planner/campaigns/tournament_authority"),
        notes="Algo dir provides the fully-specified --algo-config variants (no implicit fallback).",
    ),
    ArmSpec(
        arm_id="model",
        display_name="Model (hard-case-focused retraining)",
        child_issue=3214,
        description=(
            "Hard-case-focused retraining with crossing-conflict augmentation, then evaluated "
            "under the shared protocol."
        ),
        required_paths=(
            Path("configs/training/predictive/predictive_retraining_readiness_issue_3214.yaml"),
            Path(
                "configs/training/predictive/predictive_crossing_conflict_weighted_issue_3254.yaml"
            ),
        ),
        expected_output_path=Path("output/tmp/predictive_planner/campaigns/tournament_model"),
        notes=(
            "Checks the frozen #3254 weighted crossing-conflict config for the #3214 "
            "model arm; retraining itself remains compute-gated."
        ),
    ),
)


NEXT_PROGRESSION = ProgressionSpec(
    packet_id="scenario_family_oracle_arm",
    display_name="Scenario-family promotion packet with oracle forecast arm",
    description=(
        "Post-synthesis progression from the #3215 negative result: promote the "
        "forecast-risk gate from a single fixture to a scenario-family packet and "
        "include an oracle future-trajectory arm so later runs can separate planner "
        "limits from prediction limits and scenario infeasibility."
    ),
    required_paths=(
        Path("configs/benchmarks/predictive_scenario_family_oracle_arm_issue_3215.yaml"),
        Path("configs/scenarios/sets/predictive_hardcase_portfolio_v1.yaml"),
        Path("scripts/benchmark/run_forecast_risk_coupling_gate.py"),
        Path("robot_sf/benchmark/forecast_risk_adapter.py"),
    ),
    expected_output_path=Path("output/tmp/predictive_planner/campaigns/scenario_family_oracle_arm"),
    forecast_arms=("none", "constant_velocity", "interaction_aware", "oracle_future"),
    outcomes=(
        "collision_rate",
        "near_miss_rate",
        "false_positive_stop_rate",
        "progress_loss",
        "stop_timing",
        "forecast_risk_calibration",
    ),
    evidence_tier="diagnostic-only",
    notes=(
        "Presence-only packet readiness; does not generate the scenario family, "
        "run paired seeds, or promote dissertation/paper claims."
    ),
)


@dataclass
class PathStatus:
    """Presence record for a single expected prerequisite path."""

    path: str
    exists: bool


@dataclass
class ComponentReadiness:
    """Readiness classification for one arm or for the shared protocol."""

    component_id: str
    display_name: str
    status: str  # "ready" | "blocked"
    paths: list[PathStatus]
    missing_paths: list[str] = field(default_factory=list)
    child_issue: int | None = None
    description: str = ""
    expected_output_path: str | None = None
    notes: str = ""


def _classify_paths(repo_root: Path, paths: tuple[Path, ...]) -> tuple[list[PathStatus], list[str]]:
    """Return per-path presence records and the list of missing relative paths.

    A path counts as present if the file or directory exists. Directories are valid
    prerequisites (e.g. the authority algo-config family lives in a directory).
    """
    statuses: list[PathStatus] = []
    missing: list[str] = []
    for rel in paths:
        exists = (repo_root / rel).exists()
        statuses.append(PathStatus(path=rel.as_posix(), exists=exists))
        if not exists:
            missing.append(rel.as_posix())
    return statuses, missing


def evaluate_shared_protocol(repo_root: Path) -> ComponentReadiness:
    """Classify the shared-protocol prerequisites that every arm depends on."""
    statuses, missing = _classify_paths(repo_root, SHARED_PROTOCOL_PATHS)
    return ComponentReadiness(
        component_id="shared_protocol",
        display_name="Shared benchmark protocol (predictive_hard_seeds_v1)",
        status="ready" if not missing else "blocked",
        paths=statuses,
        missing_paths=missing,
        description=(
            "Fixed seed fixture, campaign harness, result store, no-fallback algo-config "
            "builder, and portfolio scenario set shared across all three arms."
        ),
    )


def evaluate_arm(repo_root: Path, arm: ArmSpec) -> ComponentReadiness:
    """Classify a single tournament arm as ready or blocked on local prerequisites."""
    statuses, missing = _classify_paths(repo_root, arm.required_paths)
    return ComponentReadiness(
        component_id=arm.arm_id,
        display_name=arm.display_name,
        status="ready" if not missing else "blocked",
        paths=statuses,
        missing_paths=missing,
        child_issue=arm.child_issue,
        description=arm.description,
        expected_output_path=arm.expected_output_path.as_posix(),
        notes=arm.notes,
    )


def evaluate_next_progression(repo_root: Path) -> dict:
    """Classify post-synthesis scenario-family/oracle-arm packet readiness."""
    statuses, missing = _classify_paths(repo_root, NEXT_PROGRESSION.required_paths)
    return {
        "id": NEXT_PROGRESSION.packet_id,
        "display_name": NEXT_PROGRESSION.display_name,
        "status": "ready" if not missing else "blocked",
        "paths": [{"path": p.path, "exists": p.exists} for p in statuses],
        "expected_configs": [p.path for p in statuses if p.path.startswith("configs/")],
        "blockers": [
            {"path": path, "reason": "required scenario-family packet input is missing"}
            for path in missing
        ],
        "missing_paths": missing,
        "description": NEXT_PROGRESSION.description,
        "expected_output_path": NEXT_PROGRESSION.expected_output_path.as_posix(),
        "expected_output_paths": [NEXT_PROGRESSION.expected_output_path.as_posix()],
        "forecast_arms": list(NEXT_PROGRESSION.forecast_arms),
        "outcomes": list(NEXT_PROGRESSION.outcomes),
        "evidence_tier": NEXT_PROGRESSION.evidence_tier,
        "notes": NEXT_PROGRESSION.notes,
    }


def evaluate_readiness(repo_root: Path = REPO_ROOT) -> dict:
    """Build the full presence-only readiness report for the #3215 tournament.

    The returned payload separates *local prerequisite* readiness (which this helper can
    verify) from *run authorization* (which it cannot and must never assert): ``run_authorized``
    is always ``False`` and ``run_gates`` lists the standing blockers to actually launching.
    """
    shared = evaluate_shared_protocol(repo_root)
    arms = [evaluate_arm(repo_root, arm) for arm in ARMS]
    next_progression = evaluate_next_progression(repo_root)

    components_ready = shared.status == "ready" and all(a.status == "ready" for a in arms)
    prerequisites_status = "ready" if components_ready else "blocked"

    return {
        "issue": 3215,
        "report": "predictive-tournament-readiness",
        "scope": "presence-only local prerequisite check; does not submit, run, or rank",
        "prerequisites_status": prerequisites_status,
        "run_authorized": False,
        "run_gates": list(RUN_GATES),
        "shared_protocol": _component_to_dict(shared),
        "arms": [_component_to_dict(a) for a in arms],
        "next_progression": next_progression,
    }


def _component_to_dict(component: ComponentReadiness) -> dict:
    """Serialize a ComponentReadiness to a plain JSON-friendly dict."""
    payload: dict = {
        "id": component.component_id,
        "display_name": component.display_name,
        "status": component.status,
        "paths": [{"path": p.path, "exists": p.exists} for p in component.paths],
        "expected_configs": [p.path for p in component.paths if p.path.startswith("configs/")],
        "blockers": [
            {"path": path, "reason": "required prerequisite path is missing"}
            for path in component.missing_paths
        ],
        "missing_paths": component.missing_paths,
        "description": component.description,
    }
    if component.child_issue is not None:
        payload["child_issue"] = component.child_issue
    if component.expected_output_path is not None:
        payload["expected_output_path"] = component.expected_output_path
        payload["expected_output_paths"] = [component.expected_output_path]
    if component.notes:
        payload["notes"] = component.notes
    return payload


def render_text(report: dict) -> str:
    """Render a compact human-readable readiness summary."""
    lines: list[str] = []
    lines.append("Predictive hard-case tournament readiness (#3215)")
    lines.append(f"  scope: {report['scope']}")
    lines.append(f"  local prerequisites: {report['prerequisites_status'].upper()}")
    lines.append(f"  run authorized: {report['run_authorized']} (presence-only check)")
    lines.append("")

    shared = report["shared_protocol"]
    lines.append(f"[shared protocol] {shared['display_name']}: {shared['status'].upper()}")
    for path in shared["paths"]:
        mark = "ok " if path["exists"] else "MISS"
        lines.append(f"    [{mark}] {path['path']}")
    lines.append("")

    for arm in report["arms"]:
        header = f"[arm:{arm['id']}] {arm['display_name']} (child #{arm['child_issue']})"
        lines.append(f"{header}: {arm['status'].upper()}")
        for path in arm["paths"]:
            mark = "ok " if path["exists"] else "MISS"
            lines.append(f"    [{mark}] {path['path']}")
        lines.append(f"    expected output: {arm['expected_output_path']}")
        if arm["missing_paths"]:
            lines.append(f"    blockers: {', '.join(arm['missing_paths'])}")
        lines.append("")

    progression = report["next_progression"]
    lines.append(
        f"[next progression] {progression['display_name']}: {progression['status'].upper()}"
    )
    lines.append(f"  evidence tier: {progression['evidence_tier']}")
    lines.append(f"  forecast arms: {', '.join(progression['forecast_arms'])}")
    lines.append(f"  outcomes: {', '.join(progression['outcomes'])}")
    for path in progression["paths"]:
        mark = "ok " if path["exists"] else "MISS"
        lines.append(f"  [{mark}] {path['path']}")
    lines.append(f"  expected output: {progression['expected_output_path']}")
    if progression["missing_paths"]:
        lines.append(f"  blockers: {', '.join(progression['missing_paths'])}")
    if progression.get("notes"):
        lines.append(f"  note: {progression['notes']}")
    lines.append("")

    lines.append("Run gates (must clear before launching; out of scope for this helper):")
    for gate in report["run_gates"]:
        lines.append(f"  - {gate}")
    return "\n".join(lines)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the machine-readable JSON report instead of the text summary.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root to evaluate prerequisites against (defaults to this checkout).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Print the readiness report; exit 0 when local prerequisites are ready, else 1.

    The non-zero exit on blocked prerequisites is a fail-closed signal for callers staging the
    tournament. It says nothing about run authorization, which stays gated on ``run_gates``.
    """
    args = _parse_args(argv)
    report = evaluate_readiness(args.repo_root)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(render_text(report))
    return 0 if report["prerequisites_status"] == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
