# Analysis verification

Five offline controls use synthetic dev identities and protect the analysis
boundary; they do not simulate a protected evaluation episode.

- Complete matched rosters must report the hand-calculated success/collision
  reversal (tau-b -1) and 120 eligible records.
- One missing or fallback record must withhold the joint ranking and leave its
  missing/excluded denominator visible. Optional diagnostic proxy unavailability
  must not invalidate executed goal/collision outcomes.
- One concordance with one tie on each axis gives tau-b 1/2; a constant ordering
  has no defined tau.
- Exactly three successes each out of 30 must tie, even when partitioned across
  seeds as (0,0,3) and (0,1,2). The pre-fix analyzer averaged fractions first and
  spuriously produced tau -1. Aggregating integer counts before normalization
  fixes the error.

Test-value assessment: these protect truthful ranks, ties and missingness. A
credible regression averages fractions before summation, drops unavailable
identities, or mistakes a proxy's unavailability for failed execution. Existing
benchmark contract tests validate row/schema contracts, not this joint-factor
analysis. No production test seam is added; fixtures exercise the real loader,
cell reconciliation, paired aggregation and summary writer.

Differential proof: the exact-count-tie test failed on the pre-fix new analyzer
with `-1.0 is None`. The same test passes after the integer-count correction.
The four initial controls pass; the tie regression brings the focused suite to
five. An in-memory negative control that ignores all exclusions fails the
fallback test with 120 instead of the independently expected 119 eligible
records. The control mutates no source file or artifact.

```bash
scripts/dev/run_worktree_shared_venv.sh -- uv run pytest \
  -p tests.support.heldout_seed_guard \
  tests/validation/test_examiner_robustness_analysis.py -q
scripts/dev/run_worktree_shared_venv.sh -- uv run pytest \
  -p tests.support.heldout_seed_guard tests/contract -q -n 2
```

Observed: five focused tests, 93 contract tests, and 24 existing canonical
resume/batch-runner tests passed (122 total). Real MPPI prefix checks wrote zero
rows and preserved their copied bytes. Both cap arms' five-step serial/process
canaries have equal outcomes, steps, effective physics and prediction metrics.
The completed continuations retain their exact original prefixes and yield
300 unique identities each. Final-head gate
results and hosted-check classification are recorded separately. Raw numerical
proof logs are preserved with the validation artifacts. A passing analysis
control does not establish historical release parity or empirical crowd fit.
