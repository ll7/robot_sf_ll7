# Issue #6942 ORCA adapter-hedge preregistration

Status: **proposal / blocked**. This packet freezes a possible future validation campaign; it does
not run native ORCA, submit compute, or change a planner, benchmark, paper, or dissertation claim.

## Research question

Can a paired representative campaign determine whether the holonomic-to-unicycle projection used
by Robot SF materially contributes to the measured ORCA gap, while keeping projection effects
separate from ORCA planner behavior?

The preceding [#6615 adapter harness](issue_6615_orca_adapter_validation.md) established that the
projection is measurable and emits `orca_adapter_trace.v1`. Its four fixed synthetic cases are a
native diagnostic smoke, not representative evidence and not a hedge-changing result.

## Frozen protocol

The tracked packet is
[`configs/benchmarks/issue_6942_orca_adapter_hedge_preregistration.yaml`](../../configs/benchmarks/issue_6942_orca_adapter_hedge_preregistration.yaml).
It declares:

- six medium-density classic interaction scenarios from the existing #6474 matrix;
- the `paper_eval_s30` seed set (111–140), giving 180 fixed scenario-seed cells;
- 600 simulator physics steps at `dt=0.1` seconds with identical paired snapshots;
- a native `rvo2` world-velocity counterfactual versus the same native proposal after the
  existing heading-safe unicycle projection;
- `orca_adapter_trace.v1` fields, exact step pairing, non-finite rejection, and no imputation;
- primary angle-error and forward-speed-loss estimands with preregistered materiality thresholds;
- paired scenario-seed bootstrap uncertainty, Holm correction, provenance requirements, and
  complete-case rules; and
- stop conditions for approval, native dependency, fallback/degraded execution, pairing drift,
  missing traces, changed hashes, incomplete cells, or a comparator that cannot isolate the
  projection.

The native arm is a counterfactual action reference. It must not be described as proof that the
repository's differential-drive robot can execute holonomic commands. Episode-level success,
collision, and time-to-goal deltas are secondary diagnostics only; they cannot support an ORCA
quality, safety, ranking, native-equivalence, or real-world claim.

## Gates before any execution

1. Domain-aware approval must accept or reject the frozen scenario population, native counterfactual,
   estimands, missingness rules, and diagnostic-only claim boundary.
2. A native `rvo2` canary must prove trace fields and exact provenance. The existing #6615 canary
   can satisfy field availability only; its synthetic rows cannot enter the representative sample.
3. A separately reviewed representative runner must capture source/config/input hashes before the
   first row and fail closed on any fallback, degraded, missing, non-finite, or incomparable row.
4. An independent evidence review must decide whether the result remains diagnostic or can support
   a stronger, explicitly bounded statement. The packet itself authorizes no claim upgrade.

Validate the packet with:

```bash
scripts/dev/run_worktree_shared_venv.sh -- python \
  scripts/validation/check_issue_6942_orca_adapter_preregistration.py --json
```

The expected result is structured `status: blocked`, because the approval and campaign gates are
intentionally still closed.
