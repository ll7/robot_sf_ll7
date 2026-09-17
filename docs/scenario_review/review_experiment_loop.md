# SREV-24 review experiment loop

`srev24-review-experiment-loop` is a finite, local diagnostic session around
the SREV-22 bounded executor. It selects at most three declared candidates in
`(priority, intervention_id)` order, reserves a complete control/treatment pair
before dispatch, and persists every state transition in an atomic session
journal. It does not create a new simulator runner, planner, replay engine, or
scientific-admission path.

All output is diagnostic-only. A survived, falsified, or inconclusive pair is
retained as an observation; failures, unsupported recipes, cancelled work, and
other negative outcomes are retained as well. No fixture or one-session result
supports a population, causal, benchmark, or paper-facing claim.

## Explicit start boundary

Execution requires caller authorization with `--autonomous` (or
`run(..., autonomous=True)`). `--read-only` and `read_only: true` always win and
never launch an executor. A request without an autonomous policy returns
`unavailable` with `autonomous_start_authorization_required`; this is an
operational state, not an approval question.

The native path requires the launcher-owned `executor-admission.v1` companion.
The recipe cannot nominate its own source root or receipt. SREV-24 performs the
same merged SREV-22/#9417 source and preservation preflight, then delegates
execution to `review_execute`; source mutation, stale receipts, unsafe roots,
and preservation mismatches fail closed. Injected executors are intended for
tests/offline callers and must receive an already admitted proof via the API.

## Defaults and accounting

The hard session defaults are three candidates, six simulator executions, 600
elapsed seconds, one local central processing unit (CPU) process, and no implicit retries. A recipe or
caller may narrow these values but may not widen the SREV ceilings. Controls,
treatments, failures, retries, and fidelity attempts consume execution budget.
The loop records both `executions_consumed` and `reserved_executions`; a
candidate is not started unless two execution slots are available. A failed
control-fidelity check records the control and blocks treatment interpretation.
Evaluated or inconclusive candidates are never silently retried.

## CLI

The checked-in source-bound fixture uses the immutable SREV-22 source and its
admission/preservation receipts:

```bash
uv run python -m robot_sf.analysis_workbench.review_experiment_loop \
  --input tests/fixtures/scenario_review/review_experiment_loop/request.json \
  --config tests/fixtures/scenario_review/review_experiment_loop/config.json \
  --admission-config tests/fixtures/scenario_review/review_experiment_loop/admission.json \
  --output output/scenario_review/srev-24-smoke \
  --autonomous
```

The fixture narrows the selected prefix to the two supported speed candidates
so the smoke can complete without claiming support for the known
`single_pedestrian_start_delay_offset` limitation. Its source remains the
SREV-22 tiny crossing fixture; that is provenance for a diagnostic smoke only.

The output directory contains:

- `experiment-loop-journal.json` (canonical `experiment-loop-session.v1`),
- `session-journal.json` (compatibility alias),
- `experiment-loop-report.json` (`experiment-loop-report.v1`), and
- the nested SREV-22 executor artifacts under `executor/`.

Complete results reference the report and journal. Partial, failed, cancelled,
and unavailable results do not advertise complete artifacts, but the journal
and report remain on disk for diagnosis and explicit resume.

## Resume and crash recovery

Use the same request, recipe, source proof, and identity with `--resume`. A
resume may extend the candidate, execution, wall-time, or retry ceiling only
within the recipe and component hard limits; it may never reduce a prior
ceiling. Consumed executions and elapsed time remain in the journal, so an
extension cannot reset the budget or create a fresh child session. The journal
binds request/recipe/source/config-identity digests, candidate order, policy,
operation IDs, reservations, attempts, and elapsed time. It is atomically
replaced after reservation, before dispatch, after each result, and at
settlement.

Operation IDs are deterministic (`session:candidate:<id>:control|treatment`,
with a retry suffix only when explicitly configured). A completed operation is
never dispatched again. A journal state of `dispatching` is recovered through
the executor's idempotent `result_for`/`recover` interface when available;
without a result, it is retained as a failed unknown operation rather than
blindly run a second time. Tampered source proof, recipe, candidate order,
identity, or journal accounting returns a failed resume result.

## API and status vocabulary

```python
from robot_sf.analysis_workbench.review_experiment_loop import run

result = run(
    request,
    autonomous=True,
    admission_config=launcher_admission,
    base=output_root,
    resume=True,
)
```

`complete` means every selected candidate reached a terminal recorded state
with at least one complete pair and no failed candidate. `partial` means a
budget, wall, or candidate failure stopped a session after useful state was
recorded. `cancelled` records caller cancellation. `unavailable` means the
source/capability/recipe cannot be run (including an all-unsupported set).
`failed` means malformed input, failed control fidelity, output/infrastructure
failure, or an unrecoverable execution error; candidate failures leave a
non-complete session even when later candidates settle. Every result carries exact
request/recipe/source/tool provenance and the diagnostic boundary.
