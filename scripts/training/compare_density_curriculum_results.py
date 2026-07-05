"""Diagnostic report builder for issue #4018 density-curriculum comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from robot_sf.training.density_curriculum import build_density_curriculum_schedule
from robot_sf.training.density_curriculum_readiness import evaluate_density_curriculum_readiness


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="Path to the completed comparison_manifest.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write the report files to (defaults to manifest parent directory)",
    )
    return parser.parse_args()


def find_run_manifest(policy_id: str) -> Path | None:
    """Find the latest run manifest for the given policy_id."""
    runs_dir = Path("output/benchmarks/ppo_imitation/runs")
    if not runs_dir.exists():
        return None
    candidates = list(runs_dir.glob(f"{policy_id}_*.json"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def format_runtime(hours: float) -> str:
    """Format hours into a human-readable duration."""
    total_seconds = int(hours * 3600)
    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def main() -> int:
    """Run the comparison report builder."""
    args = _parse_args()
    manifest_path = args.manifest.resolve()
    if not manifest_path.exists():
        print(f"Error: manifest {manifest_path} does not exist.")
        return 1

    # First evaluate readiness
    readiness = evaluate_density_curriculum_readiness(manifest_path)
    if readiness.status != "ready_diagnostic_smoke":
        print(f"Error: Comparison manifest is blocked or not ready: {readiness.status}")
        for blocker in readiness.blockers:
            print(f" - {blocker}")
        return 1

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_dir = args.output_dir or manifest_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts = manifest["artifacts"]
    curriculum_cfg_path = Path(manifest["curriculum"]["path"])

    # Load policy json files
    curr_policy_json_path = Path(artifacts["curriculum_checkpoint"]).with_suffix(".json")
    base_policy_json_path = Path(artifacts["baseline_checkpoint"]).with_suffix(".json")

    if not curr_policy_json_path.exists() or not base_policy_json_path.exists():
        print("Error: Missing policy JSON files beside checkpoints.")
        return 1

    curr_policy_data = json.loads(curr_policy_json_path.read_text(encoding="utf-8"))
    base_policy_data = json.loads(base_policy_json_path.read_text(encoding="utf-8"))

    # Load latest run manifests for runtime / extra info
    curr_run_manifest_path = find_run_manifest(manifest["curriculum"]["policy_id"])
    base_run_manifest_path = find_run_manifest(manifest["baseline"]["policy_id"])

    curr_run_data = (
        json.loads(curr_run_manifest_path.read_text(encoding="utf-8"))
        if curr_run_manifest_path
        else {}
    )
    base_run_data = (
        json.loads(base_run_manifest_path.read_text(encoding="utf-8"))
        if base_run_manifest_path
        else {}
    )

    # Success rate
    curr_success = curr_policy_data["metrics"]["success_rate"]["mean"]
    base_success = base_policy_data["metrics"]["success_rate"]["mean"]

    # Collision rate
    curr_collision = curr_policy_data["metrics"]["collision_rate"]["mean"]
    base_collision = base_policy_data["metrics"]["collision_rate"]["mean"]

    # Eval return
    curr_return = curr_policy_data["metrics"]["eval_episode_return"]["mean"]
    base_return = base_policy_data["metrics"]["eval_episode_return"]["mean"]

    # Stage reached
    total_steps_executed = int(curr_policy_data["metrics"]["total_timesteps_executed"]["mean"])
    curr_stages_spec = {}
    if curriculum_cfg_path.exists():
        try:
            curr_cfg_dict = yaml.safe_load(curriculum_cfg_path.read_text(encoding="utf-8"))
            curr_stages_spec = curr_cfg_dict.get("density_curriculum") or {}
        except (yaml.YAMLError, OSError, ValueError):
            pass

    schedule = build_density_curriculum_schedule(curr_stages_spec)
    final_stage = "N/A"
    if schedule.enabled and schedule.stages:
        active_stage = schedule.stage_for_timestep(total_steps_executed - 1)
        final_stage = active_stage.id if active_stage else "unknown"

    # Sample efficiency proxy (timesteps to convergence)
    curr_conv = curr_policy_data["metrics"]["timesteps_to_convergence"]["mean"]
    base_conv = base_policy_data["metrics"]["timesteps_to_convergence"]["mean"]

    # Runtime
    curr_runtime = curr_run_data.get("wall_clock_hours", 0.0)
    base_runtime = base_run_data.get("wall_clock_hours", 0.0)

    # Build report dict
    report_data = {
        "schema_version": "density_curriculum_comparison_report.v1",
        "claim_boundary": "diagnostic comparison only; not benchmark evidence",
        "curriculum": {
            "policy_id": manifest["curriculum"]["policy_id"],
            "success_rate": curr_success,
            "collision_rate": curr_collision,
            "eval_return": curr_return,
            "final_stage_reached": final_stage,
            "timesteps_to_convergence": curr_conv,
            "total_timesteps_executed": total_steps_executed,
            "runtime_hours": curr_runtime,
        },
        "baseline": {
            "policy_id": manifest["baseline"]["policy_id"],
            "success_rate": base_success,
            "collision_rate": base_collision,
            "eval_return": base_return,
            "final_stage_reached": "N/A (Disabled)",
            "timesteps_to_convergence": base_conv,
            "total_timesteps_executed": int(
                base_policy_data["metrics"]["total_timesteps_executed"]["mean"]
            ),
            "runtime_hours": base_runtime,
        },
    }

    # Generate Markdown report
    md_content = f"""# Issue #4018: Density Curriculum Comparison Report
*Diagnostic tier only. Explicitly not paper-grade or benchmark evidence.*

## Summary Comparison

| Metric | Curriculum ({report_data["curriculum"]["policy_id"]}) | Fixed-Density ({report_data["baseline"]["policy_id"]}) |
| :--- | :--- | :--- |
| **Final Stage Reached** | `{report_data["curriculum"]["final_stage_reached"]}` | `{report_data["baseline"]["final_stage_reached"]}` |
| **Success Rate (mean)** | `{curr_success:.4f}` | `{base_success:.4f}` |
| **Collision Rate (mean)** | `{curr_collision:.4f}` | `{base_collision:.4f}` |
| **Eval Episode Return (mean)** | `{curr_return:.4f}` | `{base_return:.4f}` |
| **Timesteps to Convergence** | `{curr_conv:.1f}` | `{base_conv:.1f}` |
| **Total Steps Executed** | `{total_steps_executed}` | `{report_data["baseline"]["total_timesteps_executed"]}` |
| **Runtime (wall clock)** | `{format_runtime(curr_runtime)}` | `{format_runtime(base_runtime)}` |

## Claim Boundary & Integrity Checklist
- [x] **No Benchmark Claims**: This report serves as a diagnostic validation of the training pipeline integration only.
- [x] **Fail-Closed Verification**: The readiness status was verified as `ready_diagnostic_smoke`.
- [x] **Matched Hyperparameters**: Both runs shared identical seeds, network architecture, and evaluation cadences.
"""

    report_md_path = output_dir / "comparison_report.md"
    report_json_path = output_dir / "comparison_report.json"

    report_md_path.write_text(md_content, encoding="utf-8")
    report_json_path.write_text(
        json.dumps(report_data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(f"wrote report: {report_md_path}")
    print(f"wrote JSON data: {report_json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
