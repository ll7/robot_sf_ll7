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

## Reproducible Entry Points

- Candidate runner: `uv run python scripts/validation/run_policy_search_candidate.py`
- Candidate comparison: `uv run python scripts/tools/compare_policy_search_candidates.py`
- Failure taxonomy: `uv run python scripts/tools/build_policy_search_failure_report.py`
- Pareto plot: `uv run python scripts/tools/plot_policy_search_pareto_front.py`
- Promotion decision: `uv run python scripts/tools/promote_policy_search_candidate.py`

## Scope Boundary

Local work may run smoke and narrow validation stages.

Full matrix evaluations, robustness extensions, and all learning-heavy campaigns
must be routed through `SLURM/todo.md` and the linked handoff notes.