#!/usr/bin/env python3
"""Run or fail-close the issue #2777 live observation-perturbation replay batch."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

SCHEMA_VERSION = "issue_2777_observation_noise_live_replay.v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = Path("docs/context/evidence/issue_2777_live_observation_noise_replay")
DEFAULT_SCENARIO_MATRIX = Path(
    "configs/scenarios/sets/issue_3201_pedestrian_dominated_observation_noise.yaml"
)
DEFAULT_CANDIDATE = "risk_surface_dwa_v0"
DEFAULT_STAGE = "smoke"
TRACE_RUNNER = Path("scripts/validation/run_policy_search_step_diagnostics.py")
REQUIRED_CONDITIONS = (
    "noop",
    "low_noise",
    "medium_noise",
    "missed_detection_only",
    "occlusion_only",
    "delay_only",
    "combined",
)
PROGRESS_FIELDS = (
    "net_goal_progress",
    "best_goal_progress",
    "closest_robot_ped_distance",
    "closest_robot_ped_step",
    "collision_flag_counts",
    "progress_step_count",
    "regression_step_count",
    "stagnant_step_count",
    "longest_stagnant_run",
)


@dataclass(frozen=True)
class Condition:
    """One #2755 perturbation family expressed as diagnostics-runner CLI flags."""

    name: str
    description: str
    flags: tuple[str, ...] = ()


CONDITIONS = (
    Condition("noop", "Unperturbed live planner/environment replay."),
    Condition(
        "low_noise",
        "Bounded Gaussian pedestrian-position perturbation, std=0.10 m, bound=0.20 m.",
        (
            "--observation-noise-std-m",
            "0.10",
            "--observation-noise-bound-m",
            "0.20",
            "--observation-perturbation-seed",
            "2755",
        ),
    ),
    Condition(
        "medium_noise",
        "Bounded Gaussian pedestrian-position perturbation, std=0.30 m, bound=0.60 m.",
        (
            "--observation-noise-std-m",
            "0.30",
            "--observation-noise-bound-m",
            "0.60",
            "--observation-perturbation-seed",
            "2755",
        ),
    ),
    Condition(
        "missed_detection_only",
        "All live pedestrians removed from planner input by missed-detection probability 1.0.",
        ("--missed-detection-probability", "1.0", "--observation-perturbation-seed", "2755"),
    ),
    Condition(
        "occlusion_only",
        "All live pedestrians occluded from planner input with a zero-distance occlusion gate.",
        ("--occlusion-distance-m", "0.0"),
    ),
    Condition(
        "delay_only",
        "Two-step delayed pedestrian observation, preserving the #2755 expected lag.",
        ("--observation-delay-steps", "2"),
    ),
    Condition(
        "combined",
        "Medium Gaussian perturbation plus full live-pedestrian occlusion.",
        (
            "--observation-noise-std-m",
            "0.30",
            "--observation-noise-bound-m",
            "0.60",
            "--occlusion-distance-m",
            "0.0",
            "--observation-perturbation-seed",
            "2755",
        ),
    ),
)


def _repo_path(path: Path) -> Path:
    """Resolve a repository-relative path."""
    return path if path.is_absolute() else REPO_ROOT / path


def _scenario_matrix_text(matrix_path: Path) -> str:
    """Return scenario matrix text with includes expanded one level when possible."""
    raw_text = matrix_path.read_text(encoding="utf-8")
    try:
        payload = yaml.safe_load(raw_text) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"{_durable_ref(matrix_path)} is not valid YAML: {exc}") from exc
    if not isinstance(payload, dict):
        return raw_text
    base_payload = dict(payload)
    base_payload.pop("includes", None)
    text = yaml.safe_dump(base_payload, sort_keys=False)
    includes = payload.get("includes", []) or []
    if not isinstance(includes, list):
        return text
    for include in includes:
        include_path = (matrix_path.parent / str(include)).resolve()
        if include_path.exists():
            text += "\n" + include_path.read_text(encoding="utf-8")
    return text


def _fixture_contract(matrix_path: Path) -> dict[str, Any]:
    """Check whether a live matrix appears to preserve the #2755 fixture boundary."""
    matrix_error = None
    try:
        matrix_text = _scenario_matrix_text(matrix_path)
    except ValueError as exc:
        matrix_text = ""
        matrix_error = str(exc)
    has_occluded_fixture = (
        "issue_2756_occluded_emergence" in matrix_text
        or "deterministic_occluded_emergence" in matrix_text
    )
    if matrix_error:
        blocker = matrix_error
    elif has_occluded_fixture:
        blocker = None
    else:
        blocker = (
            "No checked-in live scenario matrix preserving the #2755/#2756 "
            "occluded-emergence fixture boundary was found in the selected matrix."
        )
    return {
        "required_source_issue": 2756,
        "required_scenario": "issue_2756_occluded_emergence",
        "required_family": "occluded_emergence/deterministic_occluded_emergence",
        "first_visible_step": 5,
        "delay_steps": 2,
        "delay_only_expected_first_observed_step": 7,
        "scenario_matrix": _durable_ref(matrix_path),
        "satisfied": has_occluded_fixture,
        "blocker": blocker,
    }


def _write_generated_funnel(
    *,
    output_dir: Path,
    scenario_matrix: Path,
    stage: str,
    horizon: int,
) -> Path:
    """Write a tiny funnel config that points diagnostics at the selected matrix."""
    output_dir.mkdir(parents=True, exist_ok=True)
    funnel = {
        "stage_order": [stage],
        "stages": {
            stage: {
                "scenario_matrix": _repo_path(scenario_matrix).as_posix(),
                "seed_list": [3233],
                "benchmark_profile": "experimental",
                "horizon": int(horizon),
                "dt": 0.1,
                "workers": 1,
                "requires_slurm": False,
            }
        },
    }
    path = output_dir / "generated_policy_search_funnel.yaml"
    path.write_text(yaml.safe_dump(funnel, sort_keys=False), encoding="utf-8")
    return path


def _condition_command(
    *,
    condition: Condition,
    output_dir: Path,
    funnel_config: Path,
    args: argparse.Namespace,
) -> list[str]:
    """Build one live diagnostics subprocess command."""
    condition_dir = output_dir / "traces" / condition.name
    command = [
        sys.executable,
        str(_repo_path(TRACE_RUNNER)),
        "--candidate",
        args.candidate,
        "--stage",
        args.stage,
        "--candidate-registry",
        str(_repo_path(args.candidate_registry)),
        "--funnel-config",
        str(funnel_config),
        "--scenario-index",
        str(args.scenario_index),
        "--seed-index",
        str(args.seed_index),
        "--horizon",
        str(args.horizon),
        "--output-dir",
        str(condition_dir),
    ]
    if args.scenario_name:
        command.extend(["--scenario-name", args.scenario_name])
    if args.seed is not None:
        command.extend(["--seed", str(args.seed)])
    command.extend(condition.flags)
    return command


def _durable_ref(path: Path) -> str:
    """Return a report-safe path reference."""
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _load_trace(path: Path) -> dict[str, Any]:
    """Load a diagnostics trace."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("steps"), list):
        raise ValueError(f"{path} is not a diagnostics trace")
    return payload


def _mapping(value: Any) -> dict[str, Any]:
    """Return mapping values and coerce JSON null to empty."""
    return value if isinstance(value, dict) else {}


def _commands(trace: dict[str, Any]) -> list[Any]:
    """Return selected policy commands from a diagnostics trace."""
    return [_mapping(row).get("policy_command") for row in trace.get("steps", [])]


def _observation_totals(trace: dict[str, Any]) -> dict[str, Any]:
    """Summarize perturbation metadata across trace rows."""
    totals = {
        "missed_actor_observations_total": 0,
        "occluded_actor_observations_total": 0,
        "min_observed_actor_count": None,
        "max_observed_actor_count": None,
        "noise_profiles": set(),
    }
    observed_counts: list[int] = []
    for row in trace.get("steps", []):
        meta = _mapping(row.get("observation_perturbation"))
        totals["missed_actor_observations_total"] += int(meta.get("missed_actor_count", 0) or 0)
        totals["occluded_actor_observations_total"] += int(meta.get("occluded_actor_count", 0) or 0)
        observed_counts.append(int(meta.get("observed_actor_count", 0) or 0))
        profile = meta.get("noise_profile")
        if profile:
            totals["noise_profiles"].add(str(profile))
    totals["min_observed_actor_count"] = min(observed_counts) if observed_counts else None
    totals["max_observed_actor_count"] = max(observed_counts) if observed_counts else None
    totals["noise_profiles"] = sorted(totals["noise_profiles"])
    return totals


def _progress_delta(noop: dict[str, Any], condition: dict[str, Any]) -> dict[str, Any]:
    """Compare selected progress/risk summary fields."""
    noop_summary = _mapping(noop.get("progress_summary"))
    condition_summary = _mapping(condition.get("progress_summary"))
    return {
        field: {
            "noop": noop_summary.get(field),
            "condition": condition_summary.get(field),
            "changed": noop_summary.get(field) != condition_summary.get(field),
        }
        for field in PROGRESS_FIELDS
    }


def _classification(
    *,
    noop_trace: dict[str, Any],
    condition_trace: dict[str, Any],
    fixture_contract_satisfied: bool,
) -> dict[str, str]:
    """Classify one live condition against the no-op trace."""
    progress_delta = _progress_delta(noop_trace, condition_trace)
    command_changed = _commands(noop_trace) != _commands(condition_trace)
    progress_changed = any(item["changed"] for item in progress_delta.values())
    observation_changed = _observation_totals(noop_trace) != _observation_totals(condition_trace)
    closest = _mapping(noop_trace.get("progress_summary")).get("closest_robot_ped_distance")
    near_field = isinstance(closest, (int, float)) and closest <= 2.0
    if not fixture_contract_satisfied:
        return {
            "label": "diagnostic_only",
            "rationale": (
                "Live replay ran on a proxy scenario, not the #2755/#2756 "
                "occluded-emergence fixture boundary."
            ),
        }
    if command_changed or progress_changed:
        return {
            "label": "diagnostic_only",
            "rationale": (
                "Perturbation changed selected commands or progress/risk fields. "
                "This is live behavior evidence, but one seed is not enough for "
                "a benchmark-strength robustness claim."
            ),
        }
    if observation_changed and near_field:
        return {
            "label": "policy_insensitive",
            "rationale": (
                "Perturbation changed planner-input observations in a near-field trace, "
                "but selected commands and progress/risk summaries were identical."
            ),
        }
    return {
        "label": "scenario_too_weak",
        "rationale": (
            "The live condition did not expose a near-field behavior difference "
            "against the no-op trace."
        ),
    }


def _compare_condition(
    *,
    noop_trace_path: Path,
    condition_trace_path: Path,
    fixture_contract_satisfied: bool,
) -> dict[str, Any]:
    """Compare one condition trace against the no-op trace."""
    noop_trace = _load_trace(noop_trace_path)
    condition_trace = _load_trace(condition_trace_path)
    return {
        "trace": _durable_ref(condition_trace_path),
        "scenario": {
            "noop": noop_trace.get("scenario_id"),
            "condition": condition_trace.get("scenario_id"),
            "same": noop_trace.get("scenario_id") == condition_trace.get("scenario_id"),
        },
        "seed": {
            "noop": noop_trace.get("seed"),
            "condition": condition_trace.get("seed"),
            "same": noop_trace.get("seed") == condition_trace.get("seed"),
        },
        "planner_mode": {
            "candidate": condition_trace.get("candidate"),
            "algo": condition_trace.get("algo"),
            "stage": condition_trace.get("stage"),
        },
        "observation_summary": {
            "noop": _observation_totals(noop_trace),
            "condition": _observation_totals(condition_trace),
        },
        "command_summary": {
            "sequence_changed": _commands(noop_trace) != _commands(condition_trace),
            "noop_first": _commands(noop_trace)[0] if _commands(noop_trace) else None,
            "condition_first": _commands(condition_trace)[0]
            if _commands(condition_trace)
            else None,
            "noop_last": _commands(noop_trace)[-1] if _commands(noop_trace) else None,
            "condition_last": _commands(condition_trace)[-1]
            if _commands(condition_trace)
            else None,
        },
        "progress_delta": _progress_delta(noop_trace, condition_trace),
        "classification": _classification(
            noop_trace=noop_trace,
            condition_trace=condition_trace,
            fixture_contract_satisfied=fixture_contract_satisfied,
        ),
    }


def _fail_closed_report(
    *,
    output_dir: Path,
    args: argparse.Namespace,
    fixture_contract: dict[str, Any],
    blocker: str,
) -> dict[str, Any]:
    """Return a fail-closed report without running live diagnostics."""
    return {
        "schema_version": SCHEMA_VERSION,
        "issue": 2777,
        "status": "fail_closed",
        "classification": {
            "label": "blocked",
            "rationale": blocker,
        },
        "claim_boundary": (
            "No benchmark-facing robustness claim. The command failed closed before "
            "live replay because the #2755/#2756 occluded-emergence fixture boundary "
            "could not be preserved."
        ),
        "fixture_contract": fixture_contract,
        "run_config": _run_config(args=args, output_dir=output_dir),
        "conditions": [
            {
                "name": condition.name,
                "description": condition.description,
                "status": "blocked",
                "blocker": blocker,
            }
            for condition in CONDITIONS
        ],
        "blockers": [blocker],
    }


def _run_config(*, args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    """Return the stable command configuration payload."""
    return {
        "candidate": args.candidate,
        "stage": args.stage,
        "scenario_matrix": _durable_ref(_repo_path(args.scenario_matrix)),
        "scenario_name": args.scenario_name,
        "scenario_index": args.scenario_index,
        "seed": args.seed,
        "seed_index": args.seed_index,
        "horizon": args.horizon,
        "output_dir": _durable_ref(output_dir),
        "allow_non_occluded_live_fixture": bool(args.allow_non_occluded_live_fixture),
        "dry_run": bool(args.dry_run),
    }


def _write_outputs(report: dict[str, Any], output_dir: Path) -> None:
    """Write JSON and Markdown report artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output_dir / "README.md").write_text(_markdown(report), encoding="utf-8")


def _markdown(report: dict[str, Any]) -> str:
    """Render a compact Markdown report."""
    lines = [
        "# Issue #2777 Live Observation-Noise Replay",
        "",
        "## Status",
        "",
        f"- Status: `{report['status']}`",
        f"- Classification: `{report['classification']['label']}`",
        f"- Rationale: {report['classification']['rationale']}",
        "",
        "## Claim Boundary",
        "",
        report["claim_boundary"],
        "",
        "## Fixture Contract",
        "",
    ]
    contract = report["fixture_contract"]
    for key in (
        "required_scenario",
        "required_family",
        "first_visible_step",
        "delay_steps",
        "delay_only_expected_first_observed_step",
        "scenario_matrix",
        "satisfied",
        "blocker",
    ):
        lines.append(f"- `{key}`: `{contract.get(key)}`")
    lines.extend(
        [
            "",
            "## Conditions",
            "",
            "| Condition | Status | Classification | Caveat |",
            "|---|---|---|---|",
        ]
    )
    for condition in report["conditions"]:
        classification = _mapping(condition.get("classification"))
        label = classification.get("label") or condition.get("status", "")
        lines.append(
            f"| `{condition['name']}` | `{condition['status']}` | "
            f"`{label}` | "
            f"{condition.get('blocker') or classification.get('rationale', '')} |"
        )
    if report.get("blockers"):
        lines.extend(["", "## Blockers", ""])
        lines.extend(f"- {blocker}" for blocker in report["blockers"])
    lines.append("")
    return "\n".join(lines)


def _planned_report(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    fixture_contract: dict[str, Any],
    funnel_config: Path,
) -> dict[str, Any]:
    """Return a dry-run report with planned live commands."""
    return {
        "schema_version": SCHEMA_VERSION,
        "issue": 2777,
        "status": "diagnostic_only",
        "classification": {
            "label": "diagnostic_only",
            "rationale": "Dry run only; no live planner/environment replay was executed.",
        },
        "claim_boundary": (
            "Dry-run command plan only. This makes no benchmark-facing robustness claim."
        ),
        "fixture_contract": fixture_contract,
        "run_config": _run_config(args=args, output_dir=output_dir),
        "conditions": [
            {
                "name": condition.name,
                "description": condition.description,
                "status": "planned",
                "command": _condition_command(
                    condition=condition,
                    output_dir=output_dir,
                    funnel_config=funnel_config,
                    args=args,
                ),
            }
            for condition in CONDITIONS
        ],
        "blockers": [] if fixture_contract["satisfied"] else [fixture_contract["blocker"]],
    }


def _execute_conditions(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    funnel_config: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Run all condition subprocesses and collect raw condition statuses."""
    conditions: list[dict[str, Any]] = []
    blockers: list[str] = []
    for condition in CONDITIONS:
        command = _condition_command(
            condition=condition,
            output_dir=output_dir,
            funnel_config=funnel_config,
            args=args,
        )
        try:
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
                check=False,
                timeout=args.timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            blocker = (
                f"{condition.name} live replay timed out after "
                f"{args.timeout_seconds:.1f}s; stderr: {str(exc.stderr or '')[:500]}"
            )
            blockers.append(blocker)
            conditions.append(
                {
                    "name": condition.name,
                    "description": condition.description,
                    "status": "blocked",
                    "command": command,
                    "blocker": blocker,
                }
            )
            continue
        trace_path = output_dir / "traces" / condition.name / "trace.json"
        report_path = output_dir / "traces" / condition.name / "report.md"
        if completed.returncode != 0 or not trace_path.exists():
            blocker = (
                f"{condition.name} live replay failed with exit {completed.returncode}; "
                f"stderr: {completed.stderr.strip()[:500]}"
            )
            blockers.append(blocker)
            conditions.append(
                {
                    "name": condition.name,
                    "description": condition.description,
                    "status": "blocked",
                    "command": command,
                    "blocker": blocker,
                }
            )
            continue
        conditions.append(
            {
                "name": condition.name,
                "description": condition.description,
                "status": "live_replay",
                "command": command,
                "trace": _durable_ref(trace_path),
                "report": _durable_ref(report_path),
            }
        )
    return conditions, blockers


def _attach_condition_comparisons(
    *,
    output_dir: Path,
    conditions: list[dict[str, Any]],
    fixture_contract_satisfied: bool,
) -> list[str]:
    """Attach no-op comparisons to completed live conditions."""
    blockers: list[str] = []
    noop = next((item for item in conditions if item["name"] == "noop"), None)
    if noop is None or noop.get("status") != "live_replay":
        blockers.append("No no-op live replay trace was available for condition comparison.")
    else:
        noop_trace = output_dir / "traces" / "noop" / "trace.json"
        for item in conditions:
            if item["name"] == "noop" or item.get("status") != "live_replay":
                continue
            comparison = _compare_condition(
                noop_trace_path=noop_trace,
                condition_trace_path=output_dir / "traces" / item["name"] / "trace.json",
                fixture_contract_satisfied=fixture_contract_satisfied,
            )
            item.update(comparison)
    return blockers


def _final_classification(
    *,
    blockers: list[str],
    fixture_contract_satisfied: bool,
    conditions: list[dict[str, Any]],
) -> tuple[str, dict[str, str]]:
    """Resolve the top-level status/classification for the issue report."""
    if blockers:
        return "fail_closed", {"label": "blocked", "rationale": blockers[0]}
    if not fixture_contract_satisfied:
        return (
            "diagnostic_only",
            {
                "label": "diagnostic_only",
                "rationale": (
                    "Live replay completed on a proxy scenario, not the #2755/#2756 "
                    "occluded-emergence fixture."
                ),
            },
        )
    labels = {
        _mapping(item.get("classification")).get("label")
        for item in conditions
        if item["name"] != "noop"
    }
    if "diagnostic_only" in labels:
        label = "diagnostic_only"
    elif "policy_insensitive" in labels:
        label = "policy_insensitive"
    elif "scenario_too_weak" in labels:
        label = "scenario_too_weak"
    else:
        label = "robustness_evidence"
    return (
        "live_replay",
        {
            "label": label,
            "rationale": "All seven perturbation-family live replays completed.",
        },
    )


def run_live_batch(args: argparse.Namespace) -> dict[str, Any]:
    """Run the seven live diagnostics conditions and return the summary report."""
    output_dir = args.output_dir
    output_dir = _repo_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scenario_matrix = _repo_path(args.scenario_matrix)
    fixture_contract = _fixture_contract(scenario_matrix)
    if not fixture_contract["satisfied"] and not args.allow_non_occluded_live_fixture:
        return _fail_closed_report(
            output_dir=output_dir,
            args=args,
            fixture_contract=fixture_contract,
            blocker=str(fixture_contract["blocker"]),
        )

    funnel_config = _write_generated_funnel(
        output_dir=output_dir,
        scenario_matrix=args.scenario_matrix,
        stage=args.stage,
        horizon=args.horizon,
    )
    if args.dry_run:
        return _planned_report(
            args=args,
            output_dir=output_dir,
            fixture_contract=fixture_contract,
            funnel_config=funnel_config,
        )

    conditions, blockers = _execute_conditions(
        args=args,
        output_dir=output_dir,
        funnel_config=funnel_config,
    )
    blockers.extend(
        _attach_condition_comparisons(
            output_dir=output_dir,
            conditions=conditions,
            fixture_contract_satisfied=bool(fixture_contract["satisfied"]),
        )
    )
    status, classification = _final_classification(
        blockers=blockers,
        fixture_contract_satisfied=bool(fixture_contract["satisfied"]),
        conditions=conditions,
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "issue": 2777,
        "status": status,
        "classification": classification,
        "claim_boundary": (
            "Stress-slice live planner/environment replay for one scenario, one seed, "
            "and seven #2755 perturbation families. Treat as benchmark-facing only "
            "when fixture_contract.satisfied is true; otherwise diagnostic-only."
        ),
        "fixture_contract": fixture_contract,
        "run_config": _run_config(args=args, output_dir=output_dir),
        "conditions": conditions,
        "blockers": blockers,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--scenario-matrix", type=Path, default=DEFAULT_SCENARIO_MATRIX)
    parser.add_argument("--candidate", default=DEFAULT_CANDIDATE)
    parser.add_argument(
        "--candidate-registry",
        type=Path,
        default=Path("docs/context/policy_search/candidate_registry.yaml"),
    )
    parser.add_argument("--stage", default=DEFAULT_STAGE)
    parser.add_argument("--scenario-name", default=None)
    parser.add_argument("--scenario-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seed-index", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument(
        "--allow-non-occluded-live-fixture",
        action="store_true",
        help="Run a diagnostic proxy live replay even when the #2755 fixture boundary is absent.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = parse_args(argv)
    args.output_dir = _repo_path(args.output_dir)
    if tuple(condition.name for condition in CONDITIONS) != REQUIRED_CONDITIONS:
        raise RuntimeError("Issue #2777 condition set drifted from the required seven families")
    report = run_live_batch(args)
    _write_outputs(report, args.output_dir)
    print(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "status": report["status"],
                "classification": report["classification"],
                "summary": _durable_ref(args.output_dir / "summary.json"),
                "readme": _durable_ref(args.output_dir / "README.md"),
            },
            sort_keys=True,
        )
    )
    return 0 if report["status"] != "fail_closed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
