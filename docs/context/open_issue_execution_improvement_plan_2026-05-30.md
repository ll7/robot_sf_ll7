# Open Issue Execution Improvement Plan 2026-05-30

Related audit: [open_issues_training_split_audit_2026-05-30.md](open_issues_training_split_audit_2026-05-30.md)
Related workflow: [goal_driven_agent_loops_2026-05-13.md](goal_driven_agent_loops_2026-05-13.md)
Related SLURM note: [slurm_issue_batch_status_2026-05-21.md](slurm_issue_batch_status_2026-05-21.md)
Related learned-policy checklist:
[policy_search/contracts/learned_local_policy_eligibility.md](policy_search/contracts/learned_local_policy_eligibility.md)
Related artifact policy: [artifact_evidence_vocabulary.md](artifact_evidence_vocabulary.md)

## Purpose

This note refines a broad repository-improvement memo into a smaller execution plan. The original
memo correctly identified a recurring problem: the repository has strong issue framing, but agents
can still exhaust the implementable queue because many open issues are epics, decision gates,
external-data blockers, CARLA blockers, SLURM campaigns, or benchmark-heavy analyses.

The improvement is not to create many new labels or duplicate schemas. The useful next work is to
extend the existing issue workflow and learned-policy evidence surfaces with a few bounded,
agent-executable slices.

## What To Keep From The Memo

### 1. Add An Issue-Splitting Mode

This is the strongest workflow gap. When `goal-issue-implementation` finds no implementable issue,
the next useful action is often to split one parent or research issue into the smallest safe child
issue.

Proposed issue:

```text
workflow: add issue-splitting mode for parent-to-child decomposition
```

Acceptance criteria:

- add a mode to `issue-contract-maintainer`, `gh-issue-creator`, or a narrowly scoped new skill;
- read one parent, epic, decision, or research issue;
- identify the smallest next child issue that is independently implementable;
- check for existing child issues before proposing a new one;
- draft or create the child with scope, non-goals, validation command, and parent link;
- update the parent with `Next Implementable Child` only when the relationship is clear;
- do not change Project fields in the same pass.

Why this is better than a broad queue rewrite: `goal-issue-implementation` already has an
eligibility filter and queue exhaustion audit. The missing step is a safe decomposition path after
the queue is exhausted.

### 2. Add A Dummy Learned-Policy Adapter Fixture

This is the most concrete learned-policy implementation slice. It avoids external repositories,
checkpoints, and training while giving future learned-policy work a runnable reference.

Proposed issue:

```text
test: add dummy learned local-policy adapter fixture
```

Acceptance criteria:

- implement a deterministic learned-policy adapter stub with no real ML model;
- declare observation and action contracts using the learned-policy eligibility checklist;
- emit policy metadata and one predictable action;
- include a fail-closed path for unsupported observation/action requests;
- add a focused test that exercises the actual adapter boundary;
- document that the fixture is interface proof only, not benchmark evidence.

### 3. Specialize Existing Artifact Vocabulary For Learned Policies

The repository already has a canonical artifact vocabulary. A learned-policy manifest should extend
that vocabulary instead of becoming a parallel evidence system.

Proposed issue:

```text
workflow: define learned-policy artifact manifest fields
```

Acceptance criteria:

- define a small manifest shape under the existing artifact-evidence policy;
- include `policy_id`, `checkpoint_uri`, checksum, training config, training commit, observation
  schema, action schema, normalizer URI, license, and benchmark eligibility;
- include an example for one existing training lane, such as learned risk, shielded PPO, ORCA
  residual BC, or oracle imitation;
- require missing artifacts to fail closed with an actionable message;
- do not migrate checkpoints or change model loading in the same issue.

### 4. Tighten SLURM State Using The Existing SLURM Note

The pasted memo proposed a new SLURM ledger. A better first step is to extend the existing SLURM
batch-status/context-note pattern and only introduce machine-readable files when a concrete
consumer exists.

Proposed issue:

```text
workflow: standardize SLURM issue status blocks in context notes
```

Acceptance criteria:

- update the existing SLURM status note pattern with a compact reusable block;
- include issue number, state, job id if any, commit, config, output root, artifact status, and
  next action;
- apply the block to the current training-campaign issues that are already classified as
  SLURM-needed in the open-issues training split audit;
- keep raw logs and checkpoints out of git;
- do not submit jobs from machines without SLURM access.

## What To Change In The Memo

### Do Not Add A Large New State Taxonomy Yet

The memo proposed labels such as `state:parent`, `state:ready-local`, `state:ready-slurm`,
`state:blocked-external`, and `state:analysis-only`. That may eventually be useful, but it is not
the first fix. More labels increase project-board and automation maintenance cost.

Use existing labels and issue-body metadata first:

- `decision-required` for maintainer input;
- `resource:local`, `resource:slurm`, `resource:carla`, `resource:external-data` for execution
  environment;
- `type:*` labels for broad work class;
- explicit parent/child links in issue bodies.

Only add new `state:*` labels after a specific automation or Project #5 need is documented.

### Queue-Audit Schema Status

`goal-issue-implementation` already requires a final read-only implementability audit before
declaring the queue exhausted. Issue #1719 implements `queue_audit.v1` as a compact companion
shape for that existing audit, not as a replacement database or Project #5 metadata layer.

Use the schema to:

- keep the exhausted-queue handoff comparable across runs,
- include the issue query used, remaining issue classes, and recommended action,
- preserve the best issue-splitting candidate without classifying parent, epic, blocked, or
  analysis-only issues as ready implementation work.

Issue splitting now has the narrower `issue_split_summary.v1` companion shape. Use it only for
parent-to-child decomposition outputs: parent issue, duplicate-check queries, proposed child
readiness, blockers, validation paths, and the recommended follow-up action.

### Do Not Duplicate Agent-Run Self-Review Workflow

Agent workflow self-review already exists as a reusable skill and `docs/context/` is the durable
note surface. Avoid a new `docs/context/agent_runs/` convention unless it becomes necessary.

Better follow-up:

- document when a self-review note is required after delegated runs;
- keep compact lessons in existing context notes or workflow-improvement inboxes.

Issue #1783 adds `agent_run_self_review.v1` for compact self-review handoffs. Use it as a
comparison-friendly summary for objective, issue/file scope, validation, blockers, reusable lessons,
and whether local validation overrode sparse agent output; it does not replace the existing inbox
or context-note surfaces.

### Treat Research Priority As Maintainer Decision, Not Workflow Fact

The pasted memo recommends prioritizing actuation-aware learned navigation, keeping predictive-v2
paused, and ranking adversarial search as the second methodology lane. Those are plausible, but
they are research-priority decisions. They should be expressed as recommendations or decision
questions, not as completed policy.

Use this safer wording:

- predictive-v2 remains blocked until #1490 is revised, narrowed, or closed;
- actuation-aware learned navigation is a strong candidate lane after AMV provenance and metadata
  blockers are resolved;
- adversarial search should stay bounded until the existing smoke/manifest chain is complete;
- CARLA parity remains host-dependent and should not produce pseudo-evidence from non-CARLA hosts.

## Recommended Execution Order

1. Resolve current decision blockers from the training split audit: #1582, #1604, #1606, #1612,
   then #1490.
2. Add the issue-splitting mode so exhausted queues can produce implementable child issues instead
   of only prose reports.
3. Implement the dummy learned-policy adapter fixture.
4. Add learned-policy artifact manifest fields as a specialization of the existing artifact
   vocabulary.
5. Standardize SLURM status blocks in context notes before adding a new machine-readable ledger.
6. Keep local-ready issues moving: #1653, #1674, #1675, #1676, #1608, #1610, and #1638.
7. Queue long training only from a SLURM-capable host, following the split in
   `open_issues_training_split_audit_2026-05-30.md`.

## Published Issue List

These issues were created on 2026-05-30 from this plan. Keep this note as the rationale and
sequencing record; use the GitHub issues for execution.

| Priority | Issue | Why it is bounded |
|---|---|---|
| 1 | [#1684](https://github.com/ll7/robot_sf_ll7/issues/1684) `workflow: add issue-splitting mode for parent-to-child decomposition` | Extends existing issue workflow; no benchmark or training semantics. |
| 2 | [#1685](https://github.com/ll7/robot_sf_ll7/issues/1685) `test: add dummy learned local-policy adapter fixture` | Local deterministic fixture; no external model or training job. |
| 3 | [#1686](https://github.com/ll7/robot_sf_ll7/issues/1686) `workflow: define learned-policy artifact manifest fields` | Documentation/schema slice under existing artifact policy. |
| 4 | [#1687](https://github.com/ll7/robot_sf_ll7/issues/1687) `workflow: standardize SLURM issue status blocks in context notes` | Context-note convention; no job submission. |
| 5 | [#1688](https://github.com/ll7/robot_sf_ll7/issues/1688) `docs: sharpen exhausted-queue audit examples in goal-issue-implementation` | Improves existing skill instead of adding a parallel schema. |
| 6 | [#1689](https://github.com/ll7/robot_sf_ll7/issues/1689) `workflow: define simulation trace export schema for analysis workbench` | First child for #1646; one fixture and parser test, no frontend. |
| 7 | [#1690](https://github.com/ll7/robot_sf_ll7/issues/1690) `docs: inventory root-layout candidates before structural cleanup` | Inventory-first path for root hygiene; avoids broad moves. |

## Deferred Or Rejected From The Original Memo

- Broad new `state:*` label taxonomy: defer until automation requires it.
- Separate queue-audit schema: defer; improve `goal-issue-implementation` output first.
- New agent-run directory convention: defer; use existing self-review and context-note workflows.
- Broad learned-policy integration skill: defer; start with a checklist/context note or issue mode
  unless existing skills prove insufficient.
- New SLURM ledger files: defer until a consumer exists; standardize context-note blocks first.
- Root-layout mass refactor: reject; run inventory-first and split high-risk moves.
- Three.js or full visualization frontend: defer until trace export schema exists.
- New long training campaigns from this host: reject; local machine context does not provide SLURM
  submission.

## Validation Notes

This plan was refined with Qwen-only delegated reads. Those worker artifacts live under the
gitignored `.git/codex-agent-runs/` cache, so they are local drafting aids rather than durable
review evidence.

The scout output was treated as a lead, not final truth. Local checks confirmed overlap with:

- `.agents/skills/goal-issue-implementation/SKILL.md`
- `docs/context/slurm_issue_batch_status_2026-05-21.md`
- `docs/context/artifact_evidence_vocabulary.md`
- `docs/context/policy_search/contracts/learned_local_policy_eligibility.md`
