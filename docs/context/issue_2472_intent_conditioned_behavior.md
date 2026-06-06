# Issue #2472 Intent-Conditioned Pedestrian Behavior Scope (2026-06-06)

Status: scoped actor-model direction, not behavior-realism or benchmark evidence.

Related surfaces:

- GitHub issue: https://github.com/ll7/robot_sf_ll7/issues/2472
- Parent roadmap issue: https://github.com/ll7/robot_sf_ll7/issues/2469
- Scenario contract:
  `configs/scenarios/contracts/intent_conditioned_behavior_issue2472_contracts.yaml`
- First smoke candidate: `configs/scenarios/single/francis2023_intersection_wait.yaml`
- Scenario contract docs: `docs/scenario_contracts.md`
- Single-pedestrian knobs: `docs/single_pedestrians.md`
- Current trace schema: `robot_sf/analysis_workbench/schemas/simulation_trace_export.v1.json`

## Result

Issue #2472 asks for an intent-conditioned pedestrian behavior model direction. This pass defines a
minimal intent vocabulary, maps each intent to existing authored scenario knobs, records missing
trace fields, and chooses `francis2023_intersection_wait` as the first smoke candidate.

The result is a scenario-contract extension, not a runtime behavior model. Existing single
pedestrian controls already express useful authored behaviors through `trajectory`, `wait_at`,
`start_delay_s`, `speed_m_s`, `role`, `role_target_id`, and `role_offset`, but current trace export
does not include intent labels, intent phase, role provenance, or behavior parameters.

## Intent Vocabulary

- `crossing`: movement through a robot-relevant crossing or conflict zone.
- `waiting`: authored pause before release or crossing.
- `following`: motion relative to a robot or actor target.
- `overtaking`: passing a slower actor along a shared route.
- `group_join`: moving toward or joining another pedestrian group.

## Claim Boundary

This is proposal and interface evidence only. It does not prove intent-conditioned pedestrians are
realistic, replace Social Force, or change planner rankings. A scenario contract records authored
intent; scenario certification, runtime trace evidence, and data-grounded validation remain
separate gates.

## Required Future Proof

A useful executable smoke needs:

- runtime trace fields for `intent_label`, `intent_phase`, behavior role/target, behavior
  parameters, wait intervals, and release events;
- one deterministic scenario smoke on `francis2023_intersection_wait`;
- explicit fail-closed reporting if intent trace fields are absent;
- no planner comparison, realism claim, or Social Force replacement claim until the trace smoke
  passes and a later data-grounded issue defines the evidence target.

## Validation

Targeted validation:

```bash
uv run pytest tests/benchmark/test_intent_conditioned_behavior_contract.py -q
uv run pytest tests/benchmark/test_scenario_contract.py tests/benchmark/test_odd_contract.py::test_scenario_contract_fixture_can_reference_odd_declaration -q
uv run python scripts/tools/validate_scenario.py configs/scenarios/single/francis2023_intersection_wait.yaml
uv run ruff check tests/benchmark/test_intent_conditioned_behavior_contract.py
uv run ruff format --check tests/benchmark/test_intent_conditioned_behavior_contract.py
uv run python scripts/validation/check_docs_proof_consistency.py --path docs/context/catalog.yaml
git diff --check
```

## Follow-Up Boundary

The recommended next issue is an executable trace-metadata smoke for
`francis2023_intersection_wait` that emits intent label, intent phase, behavior parameter
provenance, and release-event fields. Stop there before training a behavior model or making any
planner-ranking claim.
