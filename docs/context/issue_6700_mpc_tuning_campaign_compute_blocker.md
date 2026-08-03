# Issue #6700: MPC tuning-sensitivity SLURM campaign — compute run blocked

> Status: `blocked` (compute not run). Evidence tier: `diagnostic-only`. This note records why the
> preregistered two-phase matched-budget MPC tuning-sensitivity SLURM campaign could not be
> launched from this lane, and what must be true before a coordinator actually submits it.

## Plain-language summary

Issue #6700 asks to run a large SLURM benchmark campaign on a compute cluster. That submission was
not performed because (1) the run is not authorized on this lane, (2) this host has no SLURM job
tooling, and (3) the two hard prerequisites the issue itself sets — the two-phase packet refreeze
and a passed native-solver canary — are not present on `main`. Rather than fabricate results or
spend a production budget that cannot be spent, this note records the blocker and the exact gate
that must pass before submission.

## What the campaign requires (from #5579, frozen 2026-08-03)

- Tuning surface: 3 declared scenarios (`classic_bottleneck_medium`, `classic_cross_trap_high`,
  `francis2023_intersection_wait`) × dev seeds 101-103.
- Held-out surface: `paper_eval_s10` seeds 111-120 on the full eligible benchmark matrix excluding
  the 3 tuning scenarios.
- Canary gate: both target arms (`prediction_mpc`, `prediction_mpc_cbf`) must produce 6/6 eligible
  rows over the 3 tuning scenarios at seed 101 in the target SLURM environment with native
  execution before any production budget is spent.
- Estimand: paired difference in collision-free route-completion rate, 2 tuned MPC arms × 4 frozen
  incumbents = 8 contrasts, 95% paired-bootstrap intervals, Holm-Bonferroni multiplicity
  correction.

## Why it is blocked on this lane

1. **Not authorized to submit.** The factory authorization for this lane is `compute_submit:
   false`, and the lease instruction forbids submitting SLURM jobs without recording an explicit
   blocker. A real campaign launch would violate that bound.
2. **No SLURM tooling on this host.** `sbatch`, `squeue`, `sacct`, and `srun` are not installed
   (verified `command -v`/`which` returned nothing on host `imech036`). There is no way to reach a
   terminal job state from here.
3. **Prerequisite packet refreeze not on `main`.** The two-phase tuning/held-out config with a
   frozen scenario list hash, the SLURM submission script under `SLURM/`, and the canary config are
   prerequisites stated in #6700 ("Blocked on packet refreeze child … being merged"). None of these
   exist on the current `main` (`HEAD d4666302b3ac03d204bf18d2ca7d9a594f7289f4`, Aug 2026). The
   in-repo config `configs/analysis/issue_5579_mpc_tuning_sensitivity.yaml`
   (SHA-256 `ceca5c8d…ab513`) still encodes the earlier single-phase run on seeds [111,112,113], not
   the frozen two-phase split.
4. **No passed native canary evidence.** Issue #5579 records the prior single-phase local run as
   `blocked`/`diagnostic-only` (295 eligible / 101 excluded: 92 solver-failure + 9 fallback, all
   0.0 success). That is not a passing canary and does not discharge the gate.

## Decision-rule outcome

Because the native canary cannot be established from this lane, the stop rule applies: do not spend
production budget. Retain the under-tuning consideration as `blocked`/`inconclusive`. No
dissertation claim is promoted or re-worded; any claim change is a separate author decision.

## What was validated locally (allowed non-compute check)

- `uv run python scripts/benchmark/run_mpc_tuning_sensitivity_issue_5579.py --check` exits 0 and
  reports the single-phase bound (`status: ok`, damage: 3 scenarios, 3 seeds, 2 target arms, 20
  candidate points, 40 target + 4 incumbent rows, 396 episode-row bound). This checks config shape
  only; it does not run episodes and does not imply the frozen two-phase contract.
- `uv run pytest tests/benchmark/test_mpc_tuning_sensitivity_issue_5579.py -q` → 9 passed.
- `git fetch origin main` confirms `HEAD == origin/main`; the branch carries no commits beyond `main`.

The `--check` is the one validation in the task's validation list that is execution-free and
exits 0. The remaining validation bullets (COMPLETED SLURM job with 6/6 canary rows, held-out rows
free of fallback/degraded/solver-failure/duplicates, a result ledger with a real job ID) require an
actual SLURM submission and are therefore not satisfied and not claimable here.

## Next smallest enabling step

Merge the packet-refreeze child (two-phase config with frozen scenario-hash, `SLURM/` submission
script, canary config, updated tests), then run the native solver canary (6/6 eligible at seed
101) on a SLURM-capable lane with `compute_submit` authorization. Issue #6700 remains the parent
compute issue to execute once those gates are green.

## Related surfaces

- Parent campaign contract: `#5579`.
- Compute issue: `#6700`.
- Seed sets: `configs/benchmarks/seed_sets_v1.yaml` (dev=[101..103], paper_eval_s10=[111..120]).
- Prior diagnostic evidence: `evidence/issue_5579_mpc_tuning_budget_sensitivity_2026-07-14/`.