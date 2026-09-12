# Planning Convention

Use this file when the selected execution profile is `coordinated` or `evidence_critical`, or when
inspection reveals enough ambiguity or risk to escalate. Observe and Local tasks use short working
notes or no persistent plan. A plan records intent, decisions, and intended proof; it never outranks
current code, evidence, or repository invariants. Repository-internal precedence is owned by the
`Instruction Precedence` contract in `AGENTS.md`.

## When A Plan Is Required

| Profile | Persistent plan |
| --- | --- |
| Observe | not required |
| Local | not required; short working notes are sufficient |
| Coordinated | required before or during implementation |
| Evidence-critical | required before execution |

Escalate to a higher profile instead of planning around a larger-than-expected task.

## Implementation / Migration Template

Keep plans short and operational:

```md
# Goal
- One or two sentences on the desired outcome.

# Scope
- In scope and explicitly out of scope.
- Acceptance criteria.

# Evidence sources
- Files, issues, docs, configs, or upstream sources that define the contract.

# Steps
- Ordered implementation steps.

# Decisions and risks
- Observed evidence separated from assumptions.
- Deferred scope or follow-up issue candidates.

# Validation route
- Commands to run and the proof each one demonstrates.

# Recovery / handoff
- How to resume, revert, or hand off if interrupted.
```

## Evidence / Research Additions

For evidence-critical plans, add:

- target claim or hypothesis, comparator or baseline, and minimum valid evidence;
- decision or stop rule, including fallback/degraded exclusions;
- artifact and provenance plan, and the synthesis, registry, or context surface to update.

## Plan Review

A good plan lets a reviewer answer: what changed, why the scope is correct, how it was validated, and
what risk remains.
