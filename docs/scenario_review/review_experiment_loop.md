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
and preservation mismatches fail closed. The native adapter resumes the child
ledger one selected candidate at a time: each child dispatch is one complete
control/treatment pair, so cancellation or a wall/budget stop cannot let a
later candidate run ahead of the outer journal. Cancellation is a boundary
between pairs; an already-started native pair settles both outer operation
records before the next candidate is considered.

Injected executors are a test/offline seam, not an admission authority. They
must receive a measured proof containing the exact request and recipe digests,
the diagnostic boundary fields, a source reference matching the request, and a
SHA-256 measured from a regular file below the invocation base. A status-only
mapping, a source outside that base, an absolute/traversal URI, or an
`/etc/passwd`-like host path is rejected before the executor is called.

## Defaults and accounting

The hard session defaults are three candidates, six simulator executions, 600
elapsed seconds, one local central processing unit (CPU) process, and no implicit retries. A recipe or
caller may narrow these values but may not widen the SREV ceilings. Controls,
treatments, failures, retries, and fidelity attempts consume execution budget.
The loop records both `executions_consumed` and `reserved_executions`; a
candidate is not started unless two execution slots are available. A failed
control-fidelity check records the control and blocks treatment interpretation.
On the native path, `fidelity_attempts` includes the child executor's measured
control check as well as the loop-level check; a child control-fidelity failure
therefore remains accounted even though no treatment is dispatched.
The wall deadline is checked before dispatch and again between control and
treatment, so treatment is never newly dispatched after the control budget is
exhausted. Evaluated or inconclusive candidates are never silently retried.

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
never dispatched again. Each retained outcome indexes every operation ID for
that candidate, including failed retry attempts; older final-pair-only indexes
are accepted once and normalized on load. Recovery binds the exact operation
record to its candidate and kind; operation IDs are opaque, so valid candidate
IDs containing `:retry:` are not parsed or stripped. A journal state of
`dispatching` is recovered through the executor's idempotent
`result_for`/`recover` interface when available; native recovery first checks
the exact supported child report/attempt-ledger schemas, request/recipe/config
digests, admitted source proof, candidate prefix, and current child dispatch,
then asks the child executor to validate its own ledger before using a retained
pair. A stale or self-consistent but unbound nested artifact cannot complete an
outer operation. Without a result, a dispatching operation is retained as a
failed unknown operation rather than blindly run a second time. Resume validates
status/state/outcome, complete control+treatment result shape, operation,
reservation, and consumed-execution invariants before accepting any terminal
journal. For a complete pair, the survived/falsified/inconclusive verdict,
measured activation flags, reason, and negative flag are recomputed from the
retained control/treatment telemetry; stored summaries that disagree are
rejected. Native resume also fails closed when `execute-report.json` and
`attempt-ledger.json` disagree, and requires a contiguous retry chain whose
predecessor is a failed, retryable attempt. A current-schema journal must carry
its persisted elapsed-time floor; that floor is monotonic, so lowering the
journal's elapsed value cannot reset the wall deadline. A successful operation
without finite telemetry is converted to a failed operation before it is
persisted, keeping the terminal journal immediately resumable. Negative export
classification is derived from canonical status/outcome rather than a mutable
flag. These checks detect inconsistent or stale local state; they do not turn a
mutable diagnostic journal into cryptographic proof of execution. Tampered
source proof, recipe, candidate order, identity, or journal accounting returns
a failed resume result.

The session directory also has an atomic `.experiment-loop.lock`. A controller
holds it from journal load/reservation through settlement and releases it only
after the result is durable. A second controller attempting a concurrent resume
fails closed with `session_lock_owned`; stale locks are not guessed or stolen.

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
