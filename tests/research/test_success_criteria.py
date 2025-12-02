"""Phase 7 success criteria smoke tests.

Validates presence of required artifacts and sections; complements earlier
unit/integration tests without duplicating deep logic.
"""

from __future__ import annotations

from robot_sf.research.orchestrator import AblationOrchestrator, ReportOrchestrator


def test_success_criteria_report(tmp_path):
    """Test success criteria report.

    Args:
        tmp_path: Auto-generated placeholder description.

    Returns:
        Any: Auto-generated placeholder description.
    """
    seeds = [1, 2]
    metric_records = [
        {
            "seed": s,
            "policy_type": "baseline",
            "success_rate": 0.8,
            "collision_rate": 0.05,
            "timesteps_to_convergence": 525000,
        }
        for s in seeds
    ] + [
        {
            "seed": s,
            "policy_type": "pretrained",
            "success_rate": 0.86,
            "collision_rate": 0.04,
            "timesteps_to_convergence": 315000,
        }
        for s in seeds
    ]
    baseline_ts = [
        r["timesteps_to_convergence"] for r in metric_records if r["policy_type"] == "baseline"
    ]
    pretrained_ts = [
        r["timesteps_to_convergence"] for r in metric_records if r["policy_type"] == "pretrained"
    ]
    orch = ReportOrchestrator(output_dir=tmp_path / "rpt")
    report_path = orch.generate_report(
        experiment_name="sc_report",
        metric_records=metric_records,
        run_id="sc_run",
        seeds=seeds,
        baseline_timesteps=baseline_ts,
        pretrained_timesteps=pretrained_ts,
        baseline_rewards=[[0.1, 0.2, 0.3] for _ in seeds],
        pretrained_rewards=[[0.2, 0.3, 0.4] for _ in seeds],
        telemetry={"steps_per_sec": 22.0},
    )
    assert report_path.exists()
    content = report_path.read_text(encoding="utf-8")
    assert "## Hypothesis Evaluation" in content
    assert "## Seed Summary" in content
    assert "## Telemetry" in content
    data_dir = report_path.parent / "data"
    assert (data_dir / "metrics.json").exists()
    assert (data_dir / "hypothesis.json").exists()


def test_success_criteria_ablation(tmp_path):
    """Test success criteria ablation.

    Args:
        tmp_path: Auto-generated placeholder description.

    Returns:
        Any: Auto-generated placeholder description.
    """
    ab_orch = AblationOrchestrator(
        experiment_name="sc_ablation",
        seeds=[1],
        ablation_params={"bc_epochs": [5, 10], "dataset_size": [100]},
        threshold=40.0,
        output_dir=tmp_path / "abl",
    )
    variants = ab_orch.run_ablation_matrix()
    report_path = ab_orch.generate_ablation_report(variants)
    assert report_path.exists()
    content = report_path.read_text(encoding="utf-8")
    assert "## Ablation Matrix" in content
    # At least one variant has a decision PASS/FAIL
    assert any(v.get("decision") in {"PASS", "FAIL"} for v in variants)
