"""Programmatic demo for research report + ablation generation (Phase 7).

Usage:
    uv run python examples/advanced/17_research_report_demo.py --out output/research_reports/demo
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from robot_sf.research.orchestrator import AblationOrchestrator, ReportOrchestrator


def _synthetic_records(seeds: list[int]) -> list[dict[str, float]]:
    records = []
    for s in seeds:
        records.append(
            {
                "seed": s,
                "policy_type": "baseline",
                "success_rate": random.uniform(0.75, 0.85),
                "collision_rate": random.uniform(0.05, 0.1),
                "timesteps_to_convergence": random.uniform(520000, 560000),
            }
        )
        records.append(
            {
                "seed": s,
                "policy_type": "pretrained",
                "success_rate": random.uniform(0.82, 0.9),
                "collision_rate": random.uniform(0.03, 0.07),
                "timesteps_to_convergence": random.uniform(300000, 340000),
            }
        )
    return records


def build_report(out_dir: Path) -> Path:
    seeds = [1, 2, 3]
    records = _synthetic_records(seeds)
    baseline_ts = [r["timesteps_to_convergence"] for r in records if r["policy_type"] == "baseline"]
    pretrained_ts = [
        r["timesteps_to_convergence"] for r in records if r["policy_type"] == "pretrained"
    ]
    orch = ReportOrchestrator(output_dir=out_dir)
    telemetry = {"steps_per_sec": 22.3, "cpu_util": "34%", "memory_gb": 4}
    return orch.generate_report(
        experiment_name="demo_report",
        metric_records=records,
        run_id="demo_run",
        seeds=seeds,
        baseline_timesteps=baseline_ts,
        pretrained_timesteps=pretrained_ts,
        baseline_rewards=[[random.uniform(0.0, 1.0) for _ in range(40)] for _ in seeds],
        pretrained_rewards=[[random.uniform(0.2, 1.2) for _ in range(40)] for _ in seeds],
        telemetry=telemetry,
    )


def build_ablation(out_dir: Path) -> Path:
    params = {"bc_epochs": [5, 10], "dataset_size": [100, 200]}
    ab_orch = AblationOrchestrator(
        experiment_name="demo_ablation",
        seeds=[1, 2],
        ablation_params=params,
        threshold=40.0,
        output_dir=out_dir / "ablation",
    )
    variants = ab_orch.run_ablation_matrix()
    return ab_orch.generate_ablation_report(variants)


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo: generate research report + ablation")
    parser.add_argument("--out", type=Path, required=True, help="Output directory root")
    args = parser.parse_args()
    report_path = build_report(args.out)
    ablation_path = build_ablation(args.out)
    print(f"Report written: {report_path}")
    print(f"Ablation report written: {ablation_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
