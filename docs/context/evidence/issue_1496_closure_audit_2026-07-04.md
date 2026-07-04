# Issue #1496 Closure Audit

Plain-language summary: issue #1496 is not ready to close because merged PRs
only delivered fail-closed readiness checks; the required durable oracle
imitation dataset retrieval, warm-start training, benchmark comparison, and
result synthesis remain blocked.

- Issue: <https://github.com/ll7/robot_sf_ll7/issues/1496>
- Audit date: 2026-07-04
- Evidence status: closure audit only, not benchmark evidence
- Current issue state: open

## Live Thread Summary

The issue asks whether oracle-driven imitation pretraining, optionally followed
by Dataset Aggregation (DAgger)-style refinement, improves sample efficiency,
final benchmark strength, and hard-slice recovery relative to
reinforcement-learning-only baselines under the repository benchmark protocol.

Maintainer comments keep the issue blocked until durable oracle-imitation
dataset evidence exists: manifest, checksum, split/leakage validation, and
retrievable artifact pointer. Later comments after PRs #4119, #4127, and #4133
explicitly say those PRs were readiness/preflight slices only and did not run
warm-start training, benchmark comparison, artifact promotion, or claim updates.

## Acceptance Mapping

| Acceptance criterion from #1496 | Evidence | Status |
| --- | --- | --- |
| Consume durable dataset manifest and checksums produced by #1470. | PR #3767 added the issue #1496 warm-start readiness manifest and checker. PR #4111 added the durable trace URI registry prerequisite. The checked-in manifest still fails closed because #1470/#2655 durable trace retrieval remains unresolved. | Not met; fail-closed readiness coverage exists. |
| Validate split/leakage contract before training. | PR #3767 wired the dataset launch-packet validator into the warm-start readiness checker. PR #4111 extended the gate to require training-ready trace registry entries. | Partially met as a preflight gate; no durable ready dataset has passed it. |
| Train at least one behavior-cloning warm-start policy from the durable oracle dataset. | No merged PR records a training manifest, checkpoint manifest, or training run for #1496. PR #4127 intentionally declared these as required future durable outputs. | Not met. |
| Compare against a documented reinforcement-learning-only baseline with the same training budget and fixed scenario/seed policy. | The readiness manifest names the RL-only baseline config, but no benchmark report or comparison run exists. | Not met. |
| Report sample-efficiency gains separately from final-performance gains. | PR #4127 added expected future output manifest paths, including a benchmark report, but no report has been produced. | Not met. |
| Document hard-slice recovery handling and ineligible examples. | No merged PR for #1496 contains the training/evaluation report needed to classify hard-slice recovery. | Not met. |
| Feed the result into hard-guarded hybrid-learning synthesis issue #1489 and classify continue, revise, stop, or insufficient evidence. | No merged PR updates #1489 or a claim/synthesis surface with #1496 results. | Not met. |
| Keep local-only output from being treated as durable evidence. | PR #3767 introduced fail-closed readiness. PR #4002 added a decision manifest. PR #4091 improved blocker reporting. PR #4119 exposed `readiness_decision` and `out_of_scope_actions`. PR #4127 declared required durable output manifests. PR #4133 tightened unsupported-output and shared-path validation. | Met for readiness/preflight behavior. |

## Merged PR Evidence

| PR | Audit interpretation |
| --- | --- |
| #3767, `feat(training): oracle-imitation warm-start readiness preflight (#1496)` | Added the first read-only warm-start readiness manifest, checker, and tests. This prevents accidental training from unresolved inputs, but does not satisfy the training-campaign contract. |
| #4002, `issue #1496: add oracle-imitation warm-start readiness decision manifest` | Added machine-readable decision-manifest output for the readiness checker. |
| #4091, `Issue #1496: improve oracle imitation warm-start readiness blockers` | Improved fail-closed blocker reporting. |
| #4106, `Issue #1496: add oracle collection readiness decision manifest` | Added collection-side decision-manifest support relevant to prerequisite artifact readiness. |
| #4111, `Issue #1496: gate warm-start readiness on trace registry` | Required a durable trace URI registry before warm-start readiness can pass. |
| #4119, `Issue #1496: expose warm-start readiness decision` | Exposed readiness decision and out-of-scope action fields; maintainer comment says the checked-in manifest still returns `artifact_retrieval_blocked`. |
| #4127, `Issue #1496: declare oracle warm-start output manifests` | Declared future training, checkpoint, and benchmark-report manifests as required durable outputs. |
| #4133, `Issue #1496: tighten warm-start output manifest preflight` | Tightened fail-closed validation for unsupported output keys and duplicate output manifest paths. |

## Closure Decision

Do not close #1496 based on current merged evidence. The acceptance criteria
that would answer the research question require durable dataset retrieval,
training, benchmark comparison, artifact promotion, and synthesis. Those actions
are outside this central processing unit (CPU)-only closure audit and remain
blocked by missing durable inputs and unavailable compute authorization.

No full benchmark campaign was run. No Slurm or GPU job was submitted. No paper
or dissertation claim text was changed. No transient queue-routing state was
written to tracked files.

## Smallest Remaining Slice

No additional micro-guard is justified by this audit: existing merged PRs already
cover the fail-closed readiness contract. The next nameable capability is not a
checker refresh; it is durable input recovery followed by the first authorized
training/comparison run:

1. Publish or recover durable trace artifact URIs and checksums for the
   oracle-imitation dataset required by #1470/#2655.
2. Re-run the #1496 warm-start readiness checker and require `ready`.
3. Run the configured behavior-cloning warm-start training and emit durable
   training/checkpoint manifests.
4. Run the fixed-budget RL-only versus warm-start benchmark comparison and emit
   the benchmark report manifest.
5. Update #1489 or the current synthesis surface with a diagnostic, benchmark,
   or blocked classification matching the evidence tier.

## Audit Validation

Audit-time validation should stay docs-only: inspect this evidence mapping,
verify the linked issue and PR references, and run the repository docs evidence
integrity check.
