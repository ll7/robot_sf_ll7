---
name: planner-integration
description: Assess planner-family integration feasibility, adapter burden, provenance safety, and benchmark-readiness
  boundaries in robot_sf_ll7.
category: benchmark-evidence
kind: atomic
phase: context
requires_write: false
requires_slurm: false
requires_benchmark_artifacts: true
delegates_to: []
output_schema: skill_run_summary.v1
---

# Planner Integration

## When to use

Use this skill when assessing whether a planner family can be integrated safely for benchmark and
documentation use.

## Read First

- `docs/benchmark_planner_family_coverage.md`
- `docs/context/issue_601_crowdnav_feasibility_note.md`
- `docs/context/issue_629_planner_zoo_research_prompt.md`
- `docs/benchmark_planner_quality_audit.md`
- `docs/context/issue_691_benchmark_fallback_policy.md`

## Workflow

1. Validate observation/action contract and compatibility assumptions.
2. Assess adapter burden and failure modes.
3. Check upstream provenance, licenses, and distribution constraints.
4. Classify benchmark readiness vs research-only usage.
5. Define required validation and deprecation conditions.

## Proof and Guardrails

- Do not claim paper-ready support if benchmark mode is degraded or fallback-only.
- Require a reproducible harness for any wrapper-friendly claim.
- Preserve conservative wording until adapter contracts are validated.

## Output

State clearly:

- whether the planner is wrapper-friendly, model-only, or research-only,
- required first-step validation and evidence,
- safe claim boundary after integration.
## Guardrails

- Stay within the skill scope declared in `.agents/skills/skills.yaml`.
- Prefer repository scripts and canonical docs before ad-hoc commands.
- Record blockers and validation gaps instead of overstating completion.
