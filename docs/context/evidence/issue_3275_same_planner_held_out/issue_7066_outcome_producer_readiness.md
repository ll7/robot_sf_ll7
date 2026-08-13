<!-- AI-GENERATED (robot_sf#7066, 2026-08-13) - NEEDS-REVIEW -->

# Issue #7066 outcome producer readiness

This note is the current execution-facing contract for producing the
Issue #6105 `adversarial_independent_outcomes.v2` packet. It supersedes the
historical producer-blocker wording in `step3_decision_packet.json`; that JSON
still preserves the pre-#7066 decision and must not be treated as a current
campaign result.

## Admitted execution identity

The producer does not create a new native Social Force runtime and does not
relabel arbitrary rows as native. The amended frozen contract admits exactly
the canonical `SocialForcePlannerAdapter` identity:

- planner identifier: `social_force`
- policy semantics: `social_force_adapter`
- adapter: `SocialForcePlannerAdapter`
- upstream command space: `velocity_vector_xy`
- benchmark command space: `unicycle_vw`
- projection: `heading_safe_velocity_to_unicycle_vw`

The historical planner/reference commit remains the `execution_commit` in
each row. The merged code that produces the packet is recorded separately as
`producer_commit`. Every episode record must also carry the same producer
commit and a content hash.

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
duplicate selected candidates, seeds, replay signatures, canonical adapter
metadata, configuration lineage, episode provenance, or confirmation rows.
It validates the emitted packet with the shared v2 admission contract before
writing it atomically.

## Evidence boundary

This change establishes a reproducible producer and a fail-closed input
contract. It does not submit compute, run the 144 planned executions, create
an outcome packet from fabricated data, or establish a continue/stop result.
Only a separately authorized native-environment execution that emits the
required explicit envelopes can produce empirical rows. Until that happens,
the decision remains diagnostic and inconclusive; fallback, degraded, mixed,
or unavailable execution is not admissible evidence.
