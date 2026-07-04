# Issue #3203 Narrow Table Re-Export Readiness Packet

This packet predeclares a narrow claim boundary for the tracked scenario-horizon table re-export.
It does not promote the July 1, 2026 run to benchmark Results evidence.

## Claim Boundary

- Allowed: payload/provenance readiness for the tracked scenario-horizon table re-export.
- Required source artifact:
  `docs/context/evidence/issue_3203_scenario_horizon_reexport_2026-07-01/reports/campaign_summary.json`
- Explicitly excluded: SNQI validity, planner ranking, benchmark Results wording, paper claims, and
  dissertation claim edits.

The source artifact still has `snqi_contract_status=fail`; that failure remains the blocker for any
broader #3203 claim.

## Validation

```bash
uv run python scripts/validation/check_scenario_horizon_results_readiness.py \
  docs/context/evidence/issue_3203_scenario_horizon_reexport_2026-07-01/reports/campaign_summary.json \
  --claim-boundary-packet \
  docs/context/evidence/issue_3203_scenario_horizon_table_reexport_closure_2026-07-04/readiness_packet.json \
  --json
```

Expected status: `table_reexport_ready`.
