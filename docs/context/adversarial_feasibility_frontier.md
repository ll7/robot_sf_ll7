# Adversarial feasibility-frontier report

**Status:** current fixture-backed report implementation for issue #9654. Full empirical
acceptance remains pending a completed small persisted loop from #9653.

The report builder consumes a versioned `adversarial-coevolution-evidence.v2` JSON bundle and
produces `adversarial-feasibility-frontier.v2` JSON, a compact Markdown report, and a two-panel
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
  cumulative known counterexample count, per-round solved/unsolved/mixed/unknown/not-observed case
  status, current unknown-feasibility count, and per-round structurally invalid and
  geometric/kinodynamic-impossibility candidate counts;
- `frontier.provenance.json` — source-artifact digests, the generating checkout's `repo_commit`,
  the evidence `source_revision`, evidence kind, visible figure title, and claim boundary. The
  evidence kind is printed in the figure so a detached fixture image remains visibly synthetic.

Use a new or empty output directory for each generation. The command refuses to overwrite any of
its expected report files, preserving prior report bundles.

## Input contract

The bundle identifies `experiment_id`, `evidence_kind` (`synthetic_fixture`, `simulator_run`, or
`historical_artifact`), source revision, simulator identity, and scenario-space identity. It contains
at least two contiguous rounds. Each round records planner/configuration identity, optimization
method and explicit objective definition/seeds/budget/selection rule, fixed/regression/held-out
episode rows, and falsification method/objective/search-space/failure predicate/seeds/budget/stop
reason.

Optimizer, search, evaluation, corpus, replay, and later admissibility-evidence artifacts use
relative paths inside the evidence bundle and carry a full source revision, schema label, role, and
SHA-256. Each per-round corpus status artifact uses `frontier-corpus-case-status.v1`; its case ID,
origin round/candidate, planner/config identity, and recorded status fields must match the enclosing
observation. A checksummed artifact for a different case cannot substantiate that observation.
Round-level optimizer, search, and evaluation references require the `optimization`,
`falsification-search`, `fixed-evaluation`, `regression-evaluation`, and `held_out-evaluation` roles,
respectively. Case observations require `falsification-search`, `corpus`, and
`admissibility-evidence` on their corresponding references. Escaping paths, absent files, changed
bytes, missing budgets, abbreviated source revisions, duplicate identities, and candidate-ledger/budget
count mismatches fail closed. Candidate rows retain
evaluation status, admissibility verdict, target-failure observation, replay result, corpus
disposition, and stable case ID. Case observations use the `falsification-search` role for their
origin search artifact and the `corpus` role for their corpus artifact. A case ID's origin round and
origin candidate stay unchanged across observations. Case observations link the discovery round and
candidate to corpus and replay artifacts. For non-historical cases, the search reference must be the
origin round's exact checksummed search artifact, and the replay reference must be the replay artifact
in that admitted origin candidate. An artifact from another round or case is not interchangeable.
A same-round observation must agree with its origin candidate's admissibility and replay status; it
cannot mark a case solved when the search recorded its target failure. Later planner-status changes,
such as `unsolved` to `solved`, are allowed. Later feasibility updates are limited to
`admissible_feasibility_unknown` → `empirically_feasible` or `planner_specific_failure`, and require
complete evidence plus a checksummed artifact whose role is `admissibility-evidence`. The report
does not infer stronger feasibility from a replay alone. Transitions are checked against the latest
recorded verdict: an unknown-to-confirmed upgrade counts once, repeated observations at the confirmed
verdict remain in the corpus without repeated discovery credit, and verdict downgrades fail closed.
A historical case is confirmed as a planner counterexample only from the later of its first confirmed
feasibility verdict and its first replay-verified unsolved or mixed planner outcome. A later replay
cannot backdate confirmation or feasibility-upgrade credit into an earlier round. A historical
unknown-feasibility case receives follow-up discovery credit only if a persisted
historical observation has both a verified replay and an unsolved or mixed planner outcome. Otherwise
its evidence-backed feasibility upgrade is retained in
`feasibility_upgrades_without_verified_counterexample_case_ids` and is not counted as a verified
counterexample. A replay artifact used to claim verified replay must declare the `replay` role as well as pass its
path and digest checks. A stable case ID may be admitted only once; later verified repeats use
`corpus_disposition=duplicate`, remain visible in candidate accounting, and do not count as new
unique discoveries. Pre-loop historical confirmed cases are included in the known-corpus frontier,
but not in the current loop's new-discovery count. The unknown-feasibility cumulative count means
cases ever admitted with that initial verdict; the current unknown count follows observations through
the rounds and can decrease after a valid evidence-backed upgrade.

The v1 schema uses the current #9651 admissibility verdicts and #9652 planner statuses:
`solved`, `unsolved`, `mixed`, and `unknown`. The report preserves those upstream values; it does
not infer dynamic feasibility from a planner failure. `structurally_invalid`,
`geometric_or_kinodynamic_impossibility`, `admissible_feasibility_unknown`,
`empirically_feasible`, and `planner_specific_failure` remain separate. Only a complete, admitted,
replay-verified target failure with `empirically_feasible` or `planner_specific_failure` status is
counted as a confirmed counterexample. Admitted unknown-feasibility cases are tracked separately. A
no-discovery statement counts unique newly confirmed cases, reports verified matches to known corpus
cases only when they were confirmed before the current round, and remains qualified by the finite
search budget. A duplicate of an unknown-feasibility case is not reported as a verified repeat until
persisted follow-up evidence confirms both feasibility and target-planner failure.

Evaluation rows keep three canonical runtime axes separate: `execution_mode` is `native`, `adapter`,
`mixed`, or `unknown`; `readiness_status` is `native`, `adapter`, `fallback`, or `degraded`; and
`availability_status` is `available`, `partial-failure`, `failed`, or `not_available`. A row enters
the benchmark-eligible set only when its evidence is complete, it is explicitly eligible, readiness
is `native` or `adapter`, availability is `available`, and execution mode is resolved as `native`,
`adapter`, or `mixed`. Success and collision rates then use separate denominators, each based only on
eligible rows with that outcome recorded. Fallback/degraded, failed, partial, missing, unknown, and
ineligible rows remain in status counts and excluded-record lists. Missing expected rows and missing
outcomes are reported separately; neither is synthesized as a success or failure.

The optimizer artifact content uses `frontier-optimizer-selection.v1` and records its experiment,
round, source revision, selected planner ID, and selected config SHA-256. The search artifact uses
`frontier-falsification-source.v1` and records the same round identity, target planner/config, and the
complete candidate ledger fields consumed by the report. The selected optimizer identity and search
target must match the enclosing round planner/configuration, and the report's candidate rows must
match that checksummed search ledger.

Each evaluation artifact uses `frontier-evaluation-source.v2`. It records the experiment, round,
source revision, evaluation-set name, planner/config identity, ordered `expected_episode_ids`, an
ordered `expected_episode_identities` manifest (`record_id`, `scenario_id`, integer
`scenario_seed`), and the complete normalized episode rows consumed by the report. The outer
evaluation set repeats these manifests and rows; they must match the checksummed source exactly,
and every reported row's scenario/seed pair must match its expected identity. Expected identities
are retained when a row is missing. If canonical source artifacts do not expose stable episode and
scenario/seed identities, accounting is unknown and report generation fails closed. A count match
alone cannot establish which episodes were evaluated. A scenario/seed pair may appear only once
within a set, and held-out identities must be disjoint from both fixed and regression identities in
each round. Fixed evaluation cohorts must retain the same scenario/seed manifest across rounds.
Held-out or regression cohorts may legitimately change, but summaries explicitly mark a changed
cohort as `non_comparable_cohort`; the figure omits the connecting trend segment across that change.
Success-rate segments also require the same eligible outcome sample in adjacent rounds. If fallback,
missing, or otherwise excluded rows change the rate denominator, the summary uses
`non_comparable_success_sample` and the figure leaves a gap. Each evaluation summary includes the
ordered episode-ID digest, sorted scenario/seed identity digest, and success-rate sample digest.
These normalized envelopes are a fixture-first #9654 input contract;
adapters from future #9653 artifacts must prove the same identity bindings rather than filling
summary fields independently.

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

The figure sidecar distinguishes `repo_commit` (the checkout that generated the figure) from
`source_revision` (the revision named by the evidence bundle). Neither field changes the evidence
claim boundary. The focused fixture contract is exercised in
`tests/adversarial/test_feasibility_frontier_report.py`. The tests create synthetic source artifacts
in temporary directories, check digests and fail-closed cases, cover repeated known cases and a flat
campaign with no verified discovery, and render the figure without starting a simulator.
