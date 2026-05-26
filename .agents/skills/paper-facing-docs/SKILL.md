---
name: paper-facing-docs
description: Draft or review benchmark and manuscript-support docs conservatively, with explicit provenance,
  reproducibility, and caveat handling.
category: benchmark-evidence
kind: atomic
phase: context
requires_write: false
requires_slurm: false
requires_benchmark_artifacts: true
delegates_to: []
output_schema: skill_run_summary.v1
---

# Paper-Facing Docs

## When to use

Use this skill for any documentation that can affect benchmark interpretation, manuscript claims, or
public-facing provenance statements.

## Read First

- `docs/code_review.md`
- `docs/benchmark_camera_ready.md`
- `docs/benchmark_artifact_publication.md`
- relevant issue execution notes under `docs/context/`
- `docs/benchmark_spec.md`

## Workflow

1. Validate the claim surface against existing benchmark definitions.
2. Check reproducibility breadcrumbs (config, command, seed policy, artifact path).
3. Mark caveats with plain-language risk language.
4. Review wording for unsupported causality or overclaim.
5. Add explicit provenance pointers before handoff.

## Proof and Guardrails

- Preserve fail-closed semantics: do not present fallback/degraded runs as success.
- Any claim must point to concrete reproducible artifacts.
- Prefer tracked canonical documents and execution notes for evidence.

## Output

Paper-facing docs should:

- cite canonical benchmark/config surfaces,
- avoid overstating planner support,
- name remaining caveats in plain language.
## Guardrails

- Stay within the skill scope declared in `.agents/skills/skills.yaml`.
- Prefer repository scripts and canonical docs before ad-hoc commands.
- Record blockers and validation gaps instead of overstating completion.
