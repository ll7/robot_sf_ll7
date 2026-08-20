<!-- AI-GENERATED (robot_sf#7066, 2026-08-13) - NEEDS-REVIEW -->

# Issue #7066 outcome producer readiness

This note is the current execution-facing contract for producing the
Issue #6105 `adversarial_independent_outcomes.v2` packet. It supersedes the
historical producer-blocker wording in `step3_decision_packet.json`; that JSON
still preserves the pre-#7066 decision and must not be treated as a current
campaign result.

## Native execution gate

The frozen #6105 contract admits only explicit native `social_force`
execution. The producer does not create a native runtime or relabel an adapter
record as native: `execution_mode` must be `native`, `adapter_active` must not
be true, and the shared v2 evaluator must accept the resulting row. Adapter,
fallback, degraded, mixed, unavailable, and identity-mismatched records fail
closed before admission. The current standard Social Force benchmark metadata
uses the adapter path, so a normal adapter record is expected to be rejected
until a separately reviewed native runtime path is available.

The historical planner/reference commit remains the `execution_commit` in each
row. The merged code that produces the packet is recorded separately as
`producer_commit`, and each episode record is content-hashed. Scenario
provenance binds to the selected candidate: `scenario_id` must equal the frozen
scenario family, while `scenario_params.candidate_manifest_id` and
`scenario_params.scenario_seed` must equal the envelope's selected manifest and
the external binding's frozen scenario seed. A record from another candidate
or scenario is rejected even if its execution seed otherwise looks valid.
Scenario and episode execution seeds must also be JSON integers, not numeric
lookalikes such as floating-point values or booleans.

## Producer command

After an authorized runner has written one explicit
`issue_7066_execution_record.v1` JSON Lines envelope for the deterministic
replay and five confirmation executions for each selected candidate, run:

```text
uv run python scripts/adversarial/materialize_issue_6105_outcomes.py \
  --contract configs/adversarial/issue_3275_same_planner_contract.json \
  --bindings docs/context/evidence/issue_3275_same_planner_held_out/candidate_manifest_bindings.v2.json \
  --execution-records <raw execution JSONL> \
  --output docs/context/evidence/issue_3275_same_planner_held_out/independent_outcomes.json
```

The bridge is resumable and idempotent. It fails closed on missing or
duplicate selected candidates, seeds, replay signatures, native execution
metadata, configuration lineage, episode provenance, or confirmation rows. It
validates the emitted packet with the shared v2 admission contract before
writing it atomically.

## Evidence boundary

This change establishes a reproducible producer and a fail-closed input
contract. It does not submit compute, run the 144 planned executions, create
an outcome packet from fabricated data, or establish a continue/stop result.
Only a separately authorized native-environment execution that emits the
required explicit envelopes can produce empirical rows. Until that happens,
the decision remains diagnostic and inconclusive; fallback, degraded, mixed,
or unavailable execution is not admissible evidence.
