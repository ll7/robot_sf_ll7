# SkillOpt Agent-Skill Adoption

[Back to Documentation Index](../README.md)

SkillOpt may be useful for improving repo-agent procedures, but it is not a simulator, benchmark,
planner, or paper-claim optimizer. Use it only for compact agent-skill text where repeated tasks can
be scored and validated against repository behavior.

## Suitable Targets

Good first targets are:

- PR readiness review: require changed-file review, risk classification, targeted checks, and
  `BASE_REF=origin/main scripts/dev/pr_ready_check.sh` before merge-ready claims on code changes.
- Benchmark-result interpretation guardrails: require seed/config/provenance checks and prevent
  one-run or fallback/degraded output from becoming benchmark evidence.
- Figure/table artifact generation: require deterministic scripts, metadata sidecars, compact
  tracked evidence, and captions that name metric, scenario, seed/config, and limitation.
- Issue decomposition: convert broad research ideas into implementable issues with acceptance
  criteria, source files, validation commands, and explicit "not guaranteed research result" wording.

## Not Suitable

Do not use SkillOpt to change:

- simulator dynamics or environment APIs;
- benchmark schemas, metrics, seed sets, scenario taxonomies, or aggregation rules;
- planner architecture decisions;
- model-provenance or paper-facing claims.

Those areas require normal implementation review and executable proof. Optimized skill text can
help agents ask better questions or run the right checks, but it cannot establish the claim.

## Transfer Gate

Only import a SkillOpt-derived agent skill or rubric after it has:

1. improved held-out task score in the source skill repository;
2. stayed compact enough to review manually;
3. passed review for stale assumptions and benchmark overclaiming;
4. improved at least one recent `robot_sf_ll7` workflow case;
5. preserved this repository's validation hierarchy from `docs/maintainer_values.md`.

For code changes, the final proof path remains the repository-native gate:

```bash
BASE_REF=origin/main scripts/dev/pr_ready_check.sh
```

Docs-only or instruction-only changes may use the cheaper path: inspect the diff and verify changed
links or referenced paths where practical.
