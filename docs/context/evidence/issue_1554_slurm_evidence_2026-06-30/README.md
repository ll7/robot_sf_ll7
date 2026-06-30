# Issue #1554 Slurm Evidence Packet

This packet synthesizes completed jobs 13192, 13198, and 13203. It is a queue-decision artifact only; it does not edit paper or dissertation claims.

- status: `analysis_before_submit`
- jobs synthesized: `13192, 13198, 13203`
- next queue recommendation: Do not enqueue another duplicate S20/H500 planner-family run yet. Analyze job 13198 first, including the soft SNQI contract warning, then submit only a targeted follow-up for an explicit analysis gap or validated config/schema unblock.

## Job Findings

### Job 13192

- campaign: `2026-06-issue1554-best-ppo-compare-run`
- config: `configs/benchmarks/paper_experiment_matrix_v1_best_ppo_compare.yaml`
- public commit: `182a4ebb5b44165ffc3608a3c20580fbf9e8add6`
- Slurm result: `COMPLETED` / `0:0`
- role: `comparison_run`
- finding: Best-PPO comparison run completed and produced two ok planner rows with 144 episodes per planner. This is comparison evidence, not a paper/dissertation claim by itself.
- artifact summary: {"matrix_rows": 2, "planner_row_status_counts": {"ok": 2}, "planner_rows": 2, "warnings": []}
- limitations: Small comparison slice only; use for queue direction, not broad planner-ranking claims.

### Job 13198

- campaign: `2026-06-issue1554-s20-h500-split-mem180-run`
- config: `configs/benchmarks/paper_experiment_matrix_v1_scenario_horizons_h500_s20.yaml`
- public commit: `12a188de7246aad3b9088ea76e6a25a20029f976`
- Slurm result: `COMPLETED` / `0:0`
- role: `result_matrix`
- finding: The 180 GB split planner-family S20/H500 run completed in 01:42:35 and produced nine ok planner rows with 960 episodes per planner.
- artifact summary: {"matrix_rows": 9, "planner_row_status_counts": {"ok": 9}, "planner_rows": 9, "warnings": ["SNQI contract status=fail with snqi_contract.enforcement=warn; campaign marked soft contract warning."]}
- limitations: SNQI contract warning blocks paper-grade interpretation until analyzed.; Analyze this result before scheduling another duplicate S20/H500 planner-family run.

### Job 13203

- campaign: `2026-06-issue1554-current-main-smoke-preflight`
- config: `configs/benchmarks/camera_ready_smoke_all_planners.yaml`
- public commit: `7c1e63997c724064468f64ace698731deacf201e`
- Slurm result: `COMPLETED` / `0:0`
- role: `smoke_preflight`
- finding: Current-main smoke/preflight completed in 26 seconds with workers=1, one scenario, and eight planners.
- artifact summary: {"planner_count": 8, "scenario_count": 1, "warnings": [], "workers": 1}
- limitations: Smoke/preflight only; it proves launcher/config plumbing but does not replace result analysis.

## Claim Blockers

- job `13198`: SNQI contract warning blocks paper-grade interpretation until analyzed.; Analyze this result before scheduling another duplicate S20/H500 planner-family run.

## Forbidden Actions

- no Slurm/GPU submission
- no artifact deletion
- no paper/dissertation claim edits
