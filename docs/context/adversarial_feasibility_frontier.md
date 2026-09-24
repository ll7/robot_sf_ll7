# Adversarial feasibility-frontier report

**Status:** current fixture-backed report implementation for issue #9654. Full empirical
acceptance remains pending a completed small persisted loop from #9653.

The report builder consumes a versioned `adversarial-coevolution-evidence.v1` JSON bundle and
produces `adversarial-feasibility-frontier.v1` JSON, a compact Markdown report, and a two-panel
publication-style figure. It does not launch a planner, search, replay, or simulator. It summarizes
the records in the persisted input bundle, checks every referenced artifact against its SHA-256,
and keeps each source path and revision in the output.

## Command

Once #9653 writes an evidence bundle in this format, run:

```bash
uv run python scripts/tools/build_adversarial_feasibility_frontier_report.py \
  --input output/adversarial-coevolution/round-evidence.json \
  --out-dir output/adversarial-coevolution/frontier-report
```

The command writes:

- `frontier_report.json` — round summaries, exact finite budgets, per-set denominators, candidate
  accounting, case status transitions, and checksummed evidence references;
- `frontier_report.md` — concise round-by-round performance and case tables, including invalid,
  failed, unknown, and replay-unavailable search rows;
- `frontier.png` and `frontier.pdf` — eligible complete success fractions by evaluation set and the
  cumulative known counterexample, solved-case, and unknown-feasibility counts;
- `frontier.provenance.json` — source-artifact digests and the claim boundary for the figure.

Use a new or empty output directory for each generation. The command refuses to overwrite any of
its expected report files, preserving prior report bundles.

## Input contract

The bundle identifies `experiment_id`, `evidence_kind` (`synthetic_fixture`, `simulator_run`, or
`historical_artifact`), source revision, simulator identity, and scenario-space identity. It contains
at least two contiguous rounds. Each round records planner/configuration identity, optimization
method and explicit objective definition/seeds/budget/selection rule, fixed/regression/held-out episode rows, and
falsification method/objective/search-space/failure predicate/seeds/budget/stop reason.

Optimizer, search, evaluation, corpus, and replay artifacts use relative paths inside the evidence
bundle and carry a full source revision, schema label, role, and SHA-256. Escaping paths, absent
files, changed bytes, missing budgets, abbreviated source revisions, duplicate identities, and
candidate-ledger/budget count mismatches fail closed. Candidate rows retain status, admissibility
verdict, target-failure observation, replay result, corpus disposition, and stable case ID. Case
observations link the discovery round and candidate to corpus and replay artifacts.

The v1 schema uses the current #9651 admissibility verdicts and #9652 planner statuses:
`solved`, `unsolved`, `mixed`, and `unknown`. The report preserves those upstream values; it does
not infer dynamic feasibility from a planner failure. `structurally_invalid`,
`geometric_or_kinodynamic_impossibility`, `admissible_feasibility_unknown`,
`empirically_feasible`, and `planner_specific_failure` remain separate. Only a complete, admitted,
replay-verified target failure with `empirically_feasible` or `planner_specific_failure` status is
counted as a confirmed counterexample. Admitted unknown-feasibility cases are tracked separately.

Evaluation rates use records marked complete, eligible, and normal-mode with a recorded outcome.
Fallback/degraded, failed, partial, missing, unknown, and ineligible rows remain in status counts
and excluded-record lists; they never enter the success or collision denominator. Missing expected
rows are reported as missing accounting and are not synthesized as successful or failed episodes.

## Evidence limits and integration

Fixture output is implementation evidence only. The input's `evidence_kind` is descriptive metadata;
it is not an independent authorization for benchmark or publication claims. A generated report does
not establish search-space coverage, global optimality, mathematical feasibility, or real-world
safety. A round with zero replay-verified counterexamples states its exact candidate and simulator
budgets and explicitly says that no counterexample found does not mean none exists.

This adapter is fixture-first because #9653 has not yet landed a durable round-artifact schema. The
implementation for Issue #9653 must either emit this v1 bundle or add an explicit, tested adapter
from its persisted round artifacts. Do not hand-enter summary numbers or label fixtures as
simulator runs. Acceptance for Issue #9654 additionally requires generating this report directly
from a real completed 2+ round #9653 run, with held-out/regression evidence and representative replay
links.

The focused fixture contract is exercised in
`tests/adversarial/test_feasibility_frontier_report.py`. The tests create synthetic source artifacts
in temporary directories, check digests and fail-closed cases, and render the figure without
starting a simulator.
