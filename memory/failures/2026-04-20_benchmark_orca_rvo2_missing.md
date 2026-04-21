---
name: Camera-ready benchmark halts on ORCA when rvo2 is not installed
description: `uv sync --extra orca` is required before submitting `SLURM/Auxme/issue_791_benchmark.sl`; `socnav_missing_prereq_policy: fallback` does not cover the rvo2 import check.
type: failure
---

Camera-ready benchmark job 11798 (2026-04-18, l40s) ran for 23:35 then halted with
`benchmark_success=false` after ORCA raised
`RuntimeError('rvo2 is required for the benchmark-ready ORCA planner. Install via
uv sync --extra orca or set allow_fallback=True.')`.

Root cause: rvo2 is an optional extra. The `orca` planner entry in
`configs/benchmarks/paper_experiment_matrix_v1_issue_791_eval_aligned_compare.yaml`
set `socnav_missing_prereq_policy: fallback`, but that key only governs socnav-related
prerequisites, not the rvo2 import. With `stop_on_failure: true` at the top level,
the first ORCA failure halts the entire campaign, so downstream planners (including
the new PPO leader) are never evaluated.

**Why:** the benchmark fallback policy
(`docs/context/issue_691_benchmark_fallback_policy.md`) is fail-closed by design —
missing prereqs should surface as clear errors rather than silent fallbacks. So the
contract is working as intended; the fix is to install rvo2.

**How to apply:** before submitting any camera-ready benchmark that includes the
`orca` planner, run:

```bash
uv sync --extra orca
uv run python -c "import rvo2; print(rvo2.__file__)"
```

Only submit the sbatch after the import check passes. If rvo2 cannot be installed on
the target cluster, either remove orca from the planner list or set
`allow_fallback=True` on orca and treat orca's result as a caveat, not a baseline
claim.

Resolved for this workstation on 2026-04-20 via `uv sync --extra orca`; benchmark
retry submitted as job 11871.
