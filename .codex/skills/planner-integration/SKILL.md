---
name: planner-integration
description: "Assess planner-family integration feasibility, adapter burden, provenance safety, and benchmark-readiness boundaries in robot_sf_ll7."
---

# Planner Integration

Use this skill when evaluating, wrapping, or documenting a planner family for benchmark use.

## Read First

- `docs/benchmark_planner_family_coverage.md`
- `docs/context/issue_601_crowdnav_feasibility_note.md`
- `docs/context/issue_629_planner_zoo_research_prompt.md`
- `docs/benchmark_planner_quality_audit.md`

## Focus

- observation contract and adapter burden,
- action space and kinematics compatibility,
- upstream provenance and license safety,
- benchmark readiness vs experimental-only status,
- what would break paper-facing faithfulness.

## Output Expectations

State clearly:

- whether the planner is wrapper-friendly, model-only, or research-only,
- what source-harness validation is required first,
- what claim the repository can safely make after integration.
