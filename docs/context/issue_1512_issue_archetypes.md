# Issue #1512 Issue Archetypes and Evidence Tiers

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/1512>

## Goal

Define a short canonical convention for issue metadata so Robot SF issues can declare what kind of
work they represent and what evidence bar they target without repeating long policy text.

This note is a **classification convention only**. It does not change benchmark semantics, metrics,
seeds, Project fields, or priority labels.

## Canonical Archetype Values

Every issue should declare exactly one `archetype` from this set:

| Archetype | Use when the issue is primarily about |
|---|---|
| `blocked-asset` | A missing model, dataset, checkpoint, map, credential-free dependency, or other prerequisite that blocks progress |
| `preflight` | Freezing inputs, validating setup, or preparing a run before broader execution |
| `slurm-execution` | Launching or tracking the actual scheduled run on SLURM |
| `analysis` | Interpreting existing outputs, metrics, logs, or campaign artifacts |
| `synthesis` | Producing a report, paper-facing summary, or cross-run conclusion |
| `workflow` | Agent workflow, automation, scripting, CI, issue process, or repo tooling changes |
| `docs` | Documentation-only contract or guidance updates |
| `benchmark-campaign` | A benchmark campaign intended to produce benchmark evidence |
| `training-campaign` | A training campaign intended to produce model or training evidence |
| `implementation` | Building or changing simulator/benchmark/runtime code to add a capability or runnable path (not primarily a campaign, analysis, or docs change) |
| `test` | Adding, repairing, or de-flaking test coverage or test infrastructure |
| `refactor` | Restructuring existing code without changing intended behavior |
| `data` | Acquiring, staging, ingesting, or making reproducible a dataset/trace/artifact input (often paired with `blocked` evidence when the source is external) |

### `blocked-asset` Guardrail-Only Trackers

Do not create a standalone `blocked-asset` issue when the proposed payload is only a
guardrail, reminder, or "track pasted follow-up anyway" note already enforced by an
existing parent issue's `state:blocked` or `evidence:blocked` status. Record the reminder
on the parent issue instead, for example as a comment, label, or status note.

Standalone `blocked-asset` issues remain valid when they carry unique technical state such
as paths, checksums, missing-file specifics, asset identifiers, or per-step unblock
conditions.

Closed guardrail-only tracker examples: #2416, #2417, #2413, and #2414.

### Deprecated archetype spellings

These are accepted on read and treated as their canonical value; prefer the canonical form for new
issues:

| Deprecated | Canonical | Note |
|---|---|---|
| `agent_task` | `implementation` | Generic agent placeholder; declare the specific archetype (e.g. `docs`, `analysis`) when it is clear |

## Canonical Evidence-Tier Values

Every issue should declare exactly one `evidence_tier` from this set:

| Evidence tier | Meaning |
|---|---|
| `idea` | Early proposal or scoping work with no execution evidence yet |
| `launch_packet` | Inputs are frozen enough to hand off or submit, but this is not execution evidence |
| `preflight_valid` | Preconditions were checked and passed for the intended next execution step |
| `smoke` | Minimal local or targeted path proving the surface works at a basic level |
| `nominal` | Standard execution evidence for the intended path |
| `stress` | Evidence from harsher scenarios, uncertainty probes, or adversarial conditions |
| `full_matrix` | Full planned scenario/config matrix completed |
| `analysis_only` | Existing outputs were analyzed without producing new execution evidence |
| `synthesis` | Evidence is a synthesis of prior durable inputs rather than a new run |
| `paper_grade` | Paper-facing, benchmark-success, or release-grade evidence |
| `blocked` | The requested evidence could not be produced because a prerequisite is missing or invalid |

### Deprecated evidence-tier spellings

Accepted on read and treated as their canonical value; prefer the canonical form for new issues:

| Deprecated | Canonical | Note |
|---|---|---|
| `proposal` | `idea` | Early proposal/scoping work with no execution evidence yet |

## Short Metadata Block

Place this block near the top of the issue body:

```yaml
archetype: preflight
evidence_tier: launch_packet
linked_policy:
  - docs/context/issue_691_benchmark_fallback_policy.md
  - docs/context/artifact_evidence_vocabulary.md
```

Use repository-relative paths in `linked_policy`. Keep the block short; issue-specific details belong
in the body sections below it.

## Concise Issue Skeleton

```markdown
## Question / Goal
## Current status
## Preconditions
## Frozen inputs
## Execution or analysis task
## Evidence outputs
## Accept / revise / reject
## Non-evidence / failure modes
## Next issue unlocked
```

## Distinguishing Common Cases

Use the requested evidence-tier values to separate commonly conflated issue types:

| Situation | Archetype | Evidence tier | Interpretation |
|---|---|---|---|
| Launch packet prepared for later execution | `preflight` | `launch_packet` | Inputs are ready for handoff/submission, but no benchmark or training claim is established yet |
| Smoke run or setup probe | `preflight` or `workflow` | `smoke` | Minimal path check only; useful for interface proof, not campaign-strength evidence |
| Fallback or degraded row observed during review | `analysis` | `analysis_only` | Record the limitation explicitly; this is not nominal success evidence |
| Execution blocked by missing dependency or asset | `blocked-asset` | `blocked` | The issue documents why evidence could not be produced yet |
| Standard scheduled campaign run | `slurm-execution`, `benchmark-campaign`, or `training-campaign` | `nominal`, `stress`, or `full_matrix` | Execution evidence exists and should match the claimed campaign depth |
| Paper-facing benchmark result or manuscript table | `benchmark-campaign` or `synthesis` | `paper_grade` | Durable evidence and fail-closed benchmark interpretation are required |

Launch packets, smoke runs, fallback/degraded rows, and paper-grade claims must not share the same
`evidence_tier` value. In particular, fallback or degraded outcomes are never `paper_grade`.

## Linked Canonical Policies

This convention links to, and does not replace, the canonical policy notes:

- **Fail-closed benchmark fallback**:
  [issue_691_benchmark_fallback_policy.md](issue_691_benchmark_fallback_policy.md)
- **Durable artifact vocabulary**:
  [artifact_evidence_vocabulary.md](artifact_evidence_vocabulary.md)
- **Durable artifact reference audit**:
  [issue_1053_durable_artifact_references.md](issue_1053_durable_artifact_references.md)
- **Evidence bundle policy**: [evidence/README.md](evidence/README.md)

## Workflow Integration

- `.agents/skills/gh-issue-creator/SKILL.md` should read this note and emit only the canonical
  `archetype` and `evidence_tier` values above.
- When the prompt is underspecified, default conservatively to `archetype: workflow` and
  `evidence_tier: idea`, then state the assumption explicitly.

## Scope Boundary

- **In scope**: canonical enum-like values, metadata block, concise issue skeleton, links to
  canonical policies, and issue-creation workflow integration.
- **Out of scope**: rewriting old issues, changing benchmark semantics, modifying Project #5 fields,
  or enforcing these values automatically in GitHub.
