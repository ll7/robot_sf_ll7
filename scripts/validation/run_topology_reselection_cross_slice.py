#!/usr/bin/env python3
"""Run a manifest-driven cross-slice topology reselection diagnostic."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_DEFAULT_MANIFEST = Path("configs/policy_search/topology_reselection_cross_slice_issue_2716.yaml")
_DEFAULT_REGISTRY = Path("docs/context/policy_search/candidate_registry.yaml")


@dataclass(frozen=True)
class SweepRow:
    """One candidate/slice/threshold diagnostic row."""

    slice_id: str
    scenario_name: str
    slice_role: str
    source_surface: str
    candidate_role: str
    candidate: str
    threshold_m: float | None
    output_dir: Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed CLI arguments.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=_DEFAULT_MANIFEST)
    parser.add_argument("--candidate-registry", type=Path, default=_DEFAULT_REGISTRY)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/diagnostics/issue_2716_topology_reselection_cross_slice"),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse existing row trace artifacts instead of launching diagnostics when present.",
    )
    parser.add_argument("--max-runs", type=int, default=None)
    return parser.parse_args(argv)


def load_manifest(path: Path) -> dict[str, Any]:
    """Load and minimally validate a cross-slice manifest.

    Returns:
        Manifest mapping.
    """

    manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise TypeError(f"Manifest must be a mapping: {path}")
    if manifest.get("schema") != "topology_reselection_cross_slice.v1":
        raise ValueError("Unsupported manifest schema")
    slices = manifest.get("slices")
    if not isinstance(slices, list) or len(slices) < 4:
        raise ValueError("Manifest must include at least four slices")
    hard_count = sum(1 for row in slices if row.get("role") == "hard")
    control_count = sum(1 for row in slices if row.get("role") == "negative_control")
    if hard_count < 3 or control_count < 1:
        raise ValueError("Manifest must include >=3 hard slices and >=1 negative-control slice")
    return manifest


def build_rows(manifest: dict[str, Any], output_dir: Path) -> list[SweepRow]:
    """Expand a manifest into deterministic sweep rows.

    Returns:
        Ordered sweep rows.
    """

    candidates = manifest["candidates"]
    thresholds = [float(value) for value in manifest.get("progress_gate_thresholds_m", [])]
    rows: list[SweepRow] = []
    for slice_row in manifest["slices"]:
        slice_id = str(slice_row["id"])
        scenario_name = str(slice_row["scenario_name"])
        slice_role = str(slice_row["role"])
        for candidate_role in ("baseline", "reuse_penalty"):
            candidate = str(candidates[candidate_role])
            rows.append(
                SweepRow(
                    slice_id=slice_id,
                    scenario_name=scenario_name,
                    slice_role=slice_role,
                    source_surface=str(slice_row["source_surface"]),
                    candidate_role=candidate_role,
                    candidate=candidate,
                    threshold_m=None,
                    output_dir=output_dir / slice_id / candidate_role,
                )
            )
        for threshold in thresholds:
            rows.append(
                SweepRow(
                    slice_id=slice_id,
                    scenario_name=scenario_name,
                    slice_role=slice_role,
                    source_surface=str(slice_row["source_surface"]),
                    candidate_role="progress_gated",
                    candidate=str(candidates["progress_gated"]),
                    threshold_m=threshold,
                    output_dir=output_dir / slice_id / f"progress_gated_threshold_{threshold:g}",
                )
            )
    return rows


def materialize_threshold_candidate_registry(
    *,
    source_registry: Path,
    base_candidate: str,
    threshold_m: float,
    work_dir: Path,
) -> tuple[Path, str]:
    """Create a temporary candidate registry with a threshold-specific candidate.

    Returns:
        Temporary registry path and generated candidate name.
    """

    registry = yaml.safe_load(source_registry.read_text(encoding="utf-8"))
    if not isinstance(registry, dict) or not isinstance(registry.get("candidates"), dict):
        raise TypeError(f"Candidate registry must contain candidates mapping: {source_registry}")
    candidates = registry["candidates"]
    if base_candidate not in candidates:
        raise KeyError(f"Candidate not found in registry: {base_candidate}")

    base_entry = copy.deepcopy(candidates[base_candidate])
    base_config_path = Path(str(base_entry["candidate_config_path"]))
    base_config = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))
    if not isinstance(base_config, dict):
        raise TypeError(f"Candidate config must be a mapping: {base_config_path}")
    params = dict(base_config.get("params") or {})
    params["primary_route_progress_gate_threshold_m"] = float(threshold_m)
    generated_name = f"{base_candidate}_threshold_{threshold_m:g}".replace(".", "p")
    generated_config = dict(base_config)
    generated_config["name"] = generated_name
    generated_config["params"] = params

    work_dir.mkdir(parents=True, exist_ok=True)
    generated_config_path = work_dir / f"{generated_name}.yaml"
    generated_registry_path = work_dir / "candidate_registry.yaml"
    generated_config_path.write_text(
        yaml.safe_dump(generated_config, sort_keys=False), encoding="utf-8"
    )
    generated_entry = copy.deepcopy(base_entry)
    generated_entry["candidate_config_path"] = str(generated_config_path)
    generated_entry["issue"] = 2716
    candidates[generated_name] = generated_entry
    generated_registry_path.write_text(yaml.safe_dump(registry, sort_keys=False), encoding="utf-8")
    return generated_registry_path, generated_name


def materialize_slice_funnel(*, manifest: dict[str, Any], row: SweepRow, work_dir: Path) -> Path:
    """Create a temporary funnel whose stage points at one slice source surface.

    Returns:
        Temporary funnel config path.
    """

    work_dir.mkdir(parents=True, exist_ok=True)
    funnel_path = work_dir / "funnel.yaml"
    funnel = {
        "stage_order": ["issue_2716_slice"],
        "stages": {
            "issue_2716_slice": {
                "scenario_matrix": row.source_surface,
                "seed_list": [int(manifest.get("seed", 111))],
                "benchmark_profile": "experimental",
                "horizon": int(manifest.get("horizon", 160)),
                "dt": 0.1,
                "workers": 1,
                "requires_slurm": False,
                "paper_facing": False,
                "claim_boundary": manifest["claim_boundary"],
            }
        },
    }
    funnel_path.write_text(yaml.safe_dump(funnel, sort_keys=False), encoding="utf-8")
    return funnel_path


def command_for_row(
    *,
    row: SweepRow,
    manifest: dict[str, Any],
    candidate_registry: Path,
    temp_root: Path,
) -> list[str]:
    """Build the diagnostic command for one sweep row.

    Returns:
        Command vector.
    """

    candidate = row.candidate
    registry = candidate_registry
    row_temp_root = temp_root / row.slice_id / row.output_dir.name
    funnel_config = materialize_slice_funnel(manifest=manifest, row=row, work_dir=row_temp_root)
    if row.threshold_m is not None:
        registry, candidate = materialize_threshold_candidate_registry(
            source_registry=candidate_registry,
            base_candidate=row.candidate,
            threshold_m=row.threshold_m,
            work_dir=row_temp_root,
        )
    return [
        sys.executable,
        "scripts/validation/run_topology_hypothesis_diagnostics.py",
        "--candidate",
        candidate,
        "--candidate-registry",
        str(registry),
        "--funnel-config",
        str(funnel_config),
        "--stage",
        "issue_2716_slice",
        "--scenario-name",
        row.scenario_name,
        "--seed",
        str(int(manifest.get("seed", 111))),
        "--horizon",
        str(int(manifest.get("horizon", 160))),
        "--max-hypotheses",
        str(int(manifest.get("max_hypotheses", 3))),
        "--min-hypotheses",
        str(int(manifest.get("min_hypotheses", 2))),
        "--block-radius-cells",
        str(int(manifest.get("block_radius_cells", 3))),
        "--block-stride-cells",
        str(int(manifest.get("block_stride_cells", 8))),
        "--output-dir",
        str(row.output_dir),
    ]


def run_row(command: list[str]) -> dict[str, Any]:
    """Run one diagnostic command and load its trace summary.

    Returns:
        Row result payload.
    """

    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    stdout_payload: dict[str, Any] = {}
    if completed.stdout.strip():
        stdout_payload = json.loads(completed.stdout)
    trace_raw = stdout_payload.get("trace")
    trace_path = Path(trace_raw) if isinstance(trace_raw, str) and trace_raw else None
    trace_payload = (
        json.loads(trace_path.read_text(encoding="utf-8"))
        if trace_path is not None and trace_path.exists()
        else {}
    )
    return {
        "returncode": completed.returncode,
        "stdout": stdout_payload,
        "stderr_excerpt": completed.stderr[-2000:],
        "trace": str(trace_path) if trace_path is not None else None,
        "report": stdout_payload.get("report"),
        "summary": trace_payload.get("summary", {}),
        "diagnostic_status": trace_payload.get("diagnostic_status", "command_failed"),
    }


def load_existing_row(row: SweepRow) -> dict[str, Any] | None:
    """Load an existing row trace from ``row.output_dir`` when available.

    Returns:
        Existing row result, or None when no trace exists.
    """

    trace_path = row.output_dir / "topology_hypotheses.json"
    if not trace_path.exists():
        return None
    trace_payload = json.loads(trace_path.read_text(encoding="utf-8"))
    return {
        "returncode": 0,
        "stdout": {
            "trace": str(trace_path),
            "report": str(row.output_dir / "topology_hypotheses.md"),
            "diagnostic_status": trace_payload.get("diagnostic_status", "unknown"),
        },
        "stderr_excerpt": "",
        "trace": str(trace_path),
        "report": str(row.output_dir / "topology_hypotheses.md"),
        "summary": trace_payload.get("summary", {}),
        "diagnostic_status": trace_payload.get("diagnostic_status", "unknown"),
    }


def row_metrics(row: SweepRow, result: dict[str, Any]) -> dict[str, Any]:
    """Extract requested cross-slice metrics from a diagnostic result.

    Returns:
        Compact metric mapping.
    """

    summary = result.get("summary") or {}
    corrective = summary.get("corrective_behavior") or {}
    terminal = corrective.get("terminal_outcome") or {}
    reuse = summary.get("topology_reuse_penalty") or {}
    outcome = terminal.get("outcome")
    success = bool(terminal.get("success", False))
    terminal_step = terminal.get("step")
    horizon_exhausted = outcome == "horizon_exhausted"
    return {
        "slice_id": row.slice_id,
        "scenario_name": row.scenario_name,
        "slice_role": row.slice_role,
        "candidate_role": row.candidate_role,
        "candidate": row.candidate,
        "threshold_m": row.threshold_m,
        "diagnostic_status": result.get("diagnostic_status"),
        "success": success,
        "terminal_outcome": outcome,
        "route_progress_m": corrective.get("max_route_progress_delta_m"),
        "deadlock_duration_steps": int(terminal_step or 0) if horizon_exhausted else 0,
        "oscillation_count": int(corrective.get("hypothesis_switch_count", 0)),
        "topology_switch_count": int(corrective.get("hypothesis_switch_count", 0)),
        "detour_cost_proxy": int(corrective.get("non_primary_topology_command_steps", 0)),
        "collision_rate": 1.0
        if terminal.get("is_pedestrian_collision")
        or terminal.get("is_obstacle_collision")
        or terminal.get("is_robot_collision")
        else 0.0,
        "time_to_clear_step": int(terminal_step) if success and terminal_step is not None else None,
        "topology_command_steps": int(corrective.get("topology_command_steps", 0)),
        "non_primary_topology_command_steps": int(
            corrective.get("non_primary_topology_command_steps", 0)
        ),
        "reuse_penalty_applied_steps": int(reuse.get("applied_steps", 0)),
        "progress_gate_satisfied_steps": int(reuse.get("progress_gate_satisfied_steps", 0)),
        "progress_suppressed_steps": int(reuse.get("progress_suppressed_steps", 0)),
    }


def classify_report(metrics: list[dict[str, Any]]) -> tuple[str, str]:
    """Classify progress-gated reselection from aggregate hard/control metrics.

    Returns:
        Classification and rationale.
    """

    progress_rows = [row for row in metrics if row["candidate_role"] == "progress_gated"]
    reuse_rows = [row for row in metrics if row["candidate_role"] == "reuse_penalty"]
    control_rows = [row for row in progress_rows if row["slice_role"] == "negative_control"]
    hard_progress = [row for row in progress_rows if row["slice_role"] == "hard"]
    if not progress_rows or not reuse_rows:
        return "blocked", "Missing progress-gated or reuse-penalty rows."
    if any(row["diagnostic_status"] != "diagnostic_complete" for row in progress_rows):
        return "blocked", "At least one progress-gated row did not produce diagnostic evidence."
    if any(row["topology_command_steps"] == 0 for row in hard_progress):
        return "stop", "A hard slice lost topology-command influence under progress gating."
    if any(row["topology_switch_count"] > 0 for row in control_rows):
        return "revise", "Negative-control row showed topology reselection/switching."
    hard_successes = [row for row in hard_progress if row["success"]]
    if not hard_successes:
        return (
            "revise",
            "Hard progress-gated rows retained diagnostic influence but all remained horizon_exhausted.",
        )

    hard_slice_ids = {row.get("slice_id") for row in hard_progress}
    cleared_hard_slice_ids = {row.get("slice_id") for row in hard_successes}
    if hard_slice_ids and cleared_hard_slice_ids == hard_slice_ids:
        return (
            "promote",
            "Every hard slice had at least one progress-gated clearance without control switching.",
        )
    return (
        "revise",
        "Progress-gated rows ran, but at least one hard slice did not clear.",
    )


def report_markdown(report: dict[str, Any]) -> str:
    """Render a compact Markdown decision table.

    Returns:
        Markdown report text.
    """

    lines = [
        "# Issue 2716 Topology Reselection Cross-Slice Diagnostic",
        "",
        f"Claim boundary: `{report['claim_boundary']}`.",
        f"Classification: `{report['classification']}`.",
        "",
        report["classification_rationale"],
        "",
        "## Decision Table",
        "",
        "| Slice | Role | Candidate | Threshold | Status | Outcome | Progress m | Switches | Deadlock steps | Collision rate |",
        "|---|---|---|---:|---|---|---:|---:|---:|---:|",
    ]
    for row in report["rows"]:
        lines.append(
            "| "
            f"{row['slice_id']} | {row['slice_role']} | {row['candidate_role']} | "
            f"{row['threshold_m'] if row['threshold_m'] is not None else 'NA'} | "
            f"{row['diagnostic_status']} | {row['terminal_outcome']} | "
            f"{row['route_progress_m'] if row['route_progress_m'] is not None else 'NA'} | "
            f"{row['topology_switch_count']} | {row['deadlock_duration_steps']} | "
            f"{row['collision_rate']} |"
        )
    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- Evidence is diagnostic-only and fail-closed; failed, unavailable, or degraded rows are not success evidence.",
            "- `detour_cost_proxy` is the non-primary topology-command step count, not a path-optimality proof.",
            "- `oscillation_count` uses topology hypothesis switch count from existing diagnostics.",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Run the manifest sweep and write JSON/Markdown outputs."""

    args = parse_args(argv)
    manifest = load_manifest(args.manifest)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = build_rows(manifest, output_dir / "runs")
    if args.max_runs is not None:
        rows = rows[: max(args.max_runs, 0)]

    row_results: list[dict[str, Any]] = []
    temp_root = output_dir / "_generated_inputs"
    for row in rows:
        command = command_for_row(
            row=row,
            manifest=manifest,
            candidate_registry=args.candidate_registry,
            temp_root=temp_root,
        )
        if args.dry_run:
            row_results.append({"row": row, "command": command, "result": None})
            continue
        existing = load_existing_row(row) if args.reuse_existing else None
        row_results.append({"row": row, "command": command, "result": existing or run_row(command)})

    metrics = [
        row_metrics(item["row"], item["result"])
        for item in row_results
        if item.get("result") is not None
    ]
    classification, rationale = (
        ("dry_run", "Commands were generated but not executed.")
        if args.dry_run
        else classify_report(metrics)
    )
    report = {
        "schema": "topology_reselection_cross_slice_report.v1",
        "issue": 2716,
        "manifest": str(args.manifest),
        "claim_boundary": manifest["claim_boundary"],
        "classification": classification,
        "classification_rationale": rationale,
        "dry_run": bool(args.dry_run),
        "commands": [
            {
                "slice_id": item["row"].slice_id,
                "candidate_role": item["row"].candidate_role,
                "threshold_m": item["row"].threshold_m,
                "command": item["command"],
            }
            for item in row_results
        ],
        "rows": metrics,
    }
    json_path = output_dir / "topology_reselection_cross_slice_report.json"
    md_path = output_dir / "topology_reselection_cross_slice_report.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(report_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {"report": str(json_path), "markdown": str(md_path), "classification": classification},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
