"""Produce compact, fail-closed FXB measurements from preserved diagnostic probes."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.validation.probe_issue_10007_fxb import SCENARIOS


def summarize(directory: Path) -> dict:
    """Require all 30 requested rows and preserve outcomes plus command diagnostics."""
    result = {
        "provenance": json.loads((directory / "provenance.json").read_text()),
        "scenarios": {},
    }
    for scenario in SCENARIOS:
        rows, actions = [], []
        hashes = {}
        for seed in range(1001, 1011):
            record_path = directory / f"{scenario}_{seed}.jsonl"
            record = json.loads(record_path.read_text())
            if record["seed"] != seed or record["scenario_id"] != scenario:
                raise ValueError("Unexpected scenario/seed in evidence")
            action_path = directory / f"{scenario}_{seed}_actions.json"
            episode_actions = json.loads(action_path.read_text())
            if len(episode_actions) != record["steps"]:
                raise ValueError("Missing action observations")
            for source in (record_path, action_path):
                hashes[source.name] = hashlib.sha256(source.read_bytes()).hexdigest()
            rows.append(
                {
                    "seed": seed,
                    "outcome": record["outcome"],
                    "steps": record["steps"],
                    "min_clearance": record["metrics"]["min_clearance"],
                    "jerk_mean": record["metrics"]["jerk_mean"],
                }
            )
            actions.extend(episode_actions)
        data = {"episodes": rows, "source_sha256": hashes, "action_count": len(actions)}
        data["outcomes"] = {
            key: sum(row["outcome"][key] for row in rows)
            for key in ("route_complete", "collision_event", "timeout_event")
        }
        for key in ("min_clearance", "jerk_mean"):
            data[f"mean_{key}"] = float(np.mean([row[key] for row in rows]))
        if "resolved_dt" in actions[0]:
            data["resolved_dt_values"] = sorted({row["resolved_dt"] for row in actions})
        else:
            traces = [row["adapter_trace"] for row in actions]
            trace_angles = [
                row["angle_error_rad"] for row in traces if row["angle_error_rad"] is not None
            ]
            data["adapter_trace_angle_p95"] = float(np.percentile(trace_angles, 95))
            data["backward_command_gt_0_5_count"] = sum(
                row["angle_error_rad"] is not None
                and row["angle_error_rad"] > np.pi / 2
                and row["executed_speed_mps"] > 0.5
                for row in traces
            )
            physical_bad = 0
            for row in actions:
                trace = row["adapter_trace"]
                velocity = np.array(trace["planned_velocity_world_mps"])
                if trace["planned_speed_mps"] > 1e-6:
                    err = np.arctan2(velocity[1], velocity[0]) - row["heading"]
                    err = np.arctan2(np.sin(err), np.cos(err))
                    physical_bad += abs(err) > np.pi / 2 and row["physical_speed"] > 0.5
            data["backward_physical_speed_gt_0_5_count"] = int(physical_bad)
            data["max_planned_speed"] = max(row["planned_speed_mps"] for row in traces)
        data["seed_1001_first_10_actions"] = json.loads(
            (directory / f"{scenario}_1001_actions.json").read_text()
        )[:10]
        result["scenarios"][scenario] = data
    return result


def main():
    """Summarize each named source directory, rejecting incomplete evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("directories", nargs="+")
    args = parser.parse_args()
    result = {
        "evidence_status": "diagnostic-only",
        "runs": {name: summarize(args.root / name) for name in args.directories},
    }
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
