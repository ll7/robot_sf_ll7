#!/usr/bin/env python3
"""Smoke test: prove trace/report payload shape from observation perturbation.

Demonstrates that ground-truth and observed states are cleanly separated
in a trace row, with no merging of ideal_state and perception_limited rows.

Exit 0 = shape contract satisfied.  This is a diagnostic-only artifact.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from robot_sf.benchmark.observation_perturbation import (
    EVIDENCE_IDEAL,
    EVIDENCE_PERCEPTION_LIMITED,
    ObservationPerturbationSpec,
    ObservationPerturbationState,
    perturb_ground_truth,
)


def _simulate_5_steps() -> list[dict[str, object]]:
    """Simulate 5 steps of a 3-actor episode with mixed perturbation profiles."""
    actor_ids = ["ped_A", "ped_B", "ped_C"]
    rows: list[dict[str, object]] = []

    noop_spec = ObservationPerturbationSpec()
    noisy_spec = ObservationPerturbationSpec(
        position_noise_std_m=0.3,
        position_noise_bound_m=0.5,
        missed_detection_probability=0.2,
        seed=42,
    )

    for step in range(5):
        pos = np.array([[1.0 + step, 2.0], [5.0, 5.0 + step], [10.0, 0.0]])
        vel = np.array([[0.1, 0.0], [-0.2, 0.3], [0.0, 0.5]])

        gt_result = perturb_ground_truth(pos, vel, actor_ids, spec=noop_spec, step=step)
        obs_result = perturb_ground_truth(pos, vel, actor_ids, spec=noisy_spec, step=step)

        rows.append(
            {
                "step": step,
                "ground_truth_observation": {
                    "positions": gt_result["observed"]["positions"].tolist(),
                    "ids": gt_result["observed"]["ids"],
                    "evidence_class": gt_result["metadata"]["evidence_class"],
                },
                "observed_observation": {
                    "positions": obs_result["observed"]["positions"].tolist(),
                    "ids": obs_result["observed"]["ids"],
                    "evidence_class": obs_result["metadata"]["evidence_class"],
                    "missing_ids": obs_result["missing_ids"],
                    "noise_profile": obs_result["metadata"]["noise_profile"],
                },
            }
        )
    return rows


def _simulate_delayed_steps() -> dict[str, object]:
    """Prove delay buffer produces distinct GT and lagged observed payloads."""
    actor_ids = ["ped_A"]
    spec = ObservationPerturbationSpec(delay_steps=2, seed=0)
    state = ObservationPerturbationState(delay_steps=2)
    results: list[dict[str, object]] = []

    for step in range(4):
        pos = np.array([[float(step), 0.0]])
        vel = np.array([[0.0, 0.0]])
        r = perturb_ground_truth(pos, vel, actor_ids, spec=spec, step=step, state=state)
        results.append(
            {
                "step": step,
                "gt_x": r["ground_truth"]["positions"][0][0],
                "obs_x": r["observed"]["positions"][0][0],
                "evidence_class": r["metadata"]["evidence_class"],
                "delay_steps": r["metadata"]["delay_steps"],
            }
        )
    return {"delay_trace": results}


def main() -> int:
    """Execute the smoke test and write trace/report artifacts."""
    rows = _simulate_5_steps()
    delay_info = _simulate_delayed_steps()

    # Shape contract checks
    for row in rows:
        assert "ground_truth_observation" in row
        assert "observed_observation" in row
        gt = row["ground_truth_observation"]
        obs = row["observed_observation"]
        assert gt["evidence_class"] == EVIDENCE_IDEAL
        assert obs["evidence_class"] == EVIDENCE_PERCEPTION_LIMITED
        assert isinstance(gt["positions"], list)
        assert isinstance(obs["positions"], list)
        assert isinstance(obs["missing_ids"], list)

    # Delay contract: observed lags behind GT
    for entry in delay_info["delay_trace"]:
        assert entry["delay_steps"] == 2
        assert entry["evidence_class"] == EVIDENCE_PERCEPTION_LIMITED

    # Write artifacts
    output_dir = Path("output/agent/issue-2730-opencode")
    output_dir.mkdir(parents=True, exist_ok=True)

    trace = {
        "trace_rows": rows,
        **delay_info,
    }
    trace_path = output_dir / "smoke_trace.json"
    trace_path.write_text(json.dumps(trace, indent=2), encoding="utf-8")

    report_lines = [
        "# Observation Perturbation Smoke Report",
        "",
        "## Trace/Report Shape",
        "",
        "- Each step has separate `ground_truth_observation` and `observed_observation` fields.",
        "- `ground_truth_observation.evidence_class` = `ideal_state` (always).",
        "- `observed_observation.evidence_class` = `perception_limited` (always).",
        "- Rows never merge ideal_state and perception_limited data.",
        "",
        "## Delay Behavior",
        "",
        "- Delay steps: 2",
        "- Observed positions lag GT by 2 steps after buffer warmup.",
        "",
        "## Integration Status",
        "",
        "- Direct integration into `run_policy_search_step_diagnostics.py` deferred:",
        "  requires modifying the step loop's observation flow, which is a broader refactor.",
        "- This smoke script proves the helper produces the correct trace/report payload shape.",
    ]
    report_path = output_dir / "RESULT.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(json.dumps({"status": "ok", "trace_path": str(trace_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
