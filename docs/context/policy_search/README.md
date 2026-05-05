# Policy Search Context

This directory is the file-based execution surface for the local policy-search
project requested on 2026-04-29.

Use it for three things only:

- stable project contract and candidate registry,
- reasoning, validation, and result notes for each iteration,
- SLURM handoff instructions for work that is too expensive to run locally.

## Layout

- `candidate_registry.yaml`: canonical candidate list and config pointers.
- `experiment_ledger.md`: compact execution log for implemented candidates.
- `contracts/`: project contract, gates, taxonomy, and runbook.
- `reasoning/`: bounded planning and design notes.
- `reports/`: per-run markdown reports emitted by the policy-search runner.
- `validation/`: local smoke or narrow validation notes.
- `SLURM/`: deferred training TODO plus detailed handoff instructions.

## Current Diagnostic Notes

- `portfolio_overview_2026-05-05.md`: current candidate portfolio overview generated from the
  policy-search registry and tracked reports; includes current leaders, success/collision/near-miss
  evidence, why the best candidates look promising, coverage gaps, and reproduction commands.
- `reports/2026-05-05_full_matrix_all_candidates_analysis.md`: completed 29-candidate
  full-matrix sweep analysis, release-candidate decision, provenance caveat, and recommended next
  steps.
- `reports/2026-05-05_full_matrix_h500_analysis.md`: h500 long-horizon full-matrix analysis,
  current leader, promotion split, research directions, and scenario-specific horizon policy.
- `validation/2026-05-02_hybrid_rule_failure_diagnostics.md`: issue #874 diagnosis of the
  remaining `hybrid_rule_v3_fast_progress_static_escape` static-route and leave-group failures.

## Reproducible Entry Points

- Candidate runner: `uv run python scripts/validation/run_policy_search_candidate.py`
- Candidate portfolio overview: `uv run python scripts/tools/summarize_policy_search_portfolio.py`
- Candidate comparison: `uv run python scripts/tools/compare_policy_search_candidates.py`
- Failure taxonomy: `uv run python scripts/tools/build_policy_search_failure_report.py`
- Pareto plot: `uv run python scripts/tools/plot_policy_search_pareto_front.py`
- Promotion decision: `uv run python scripts/tools/promote_policy_search_candidate.py`
- SLURM candidate sweep: `scripts/dev/sbatch_policy_search_sweep.sh --stage full_matrix --all-implemented`

## Scope Boundary

Local work may run smoke and narrow validation stages.

Full matrix evaluations, robustness extensions, and all learning-heavy campaigns
must be routed through `SLURM/todo.md` and the linked handoff notes.
