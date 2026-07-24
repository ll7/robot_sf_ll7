# Issue #3275 Frozen Same-Planner Contract and Held-Out Result Integrity Repair

This context document records the proposed, currently **blocked** held-out contract and
its result-integrity repair for issue #3275 (sub-issue #6103). It is not evidence of
an executed experiment or a decision about proposal quality.

## Current Fail-Closed Status

The #6139 correction merged through PR #6172. Its content-addressed
recertification artifact is now bound by the contract and confirms that the accepted
archive bytes are unchanged. It also classifies 6 of the 12 pinned group-crossing fit
records as `stress_only`. The contract requires `eligible_only` fit inputs, so the
normal runner rejects the contract before selecting candidates. It remains blocked
pending an eligible-only fit-set decision, the #6104 candidate manifest, power
sensitivity from that final fit lineage, and the cross-family feature-semantics audit.

## 1. Frozen Study Inputs

- **Tracked Archive Path**: `docs/context/evidence/issue_5305_certified_archive/archive.json`
- **Archive Raw SHA-256**: `79e022587b35c1c42bc07cfefaf882af473e96841a99ef57f98a4cee26636445`
- **Archive Payload SHA-256**: `1d8db3638e3539dca2e06849c0727150aff41ea85a71ee2d22cb1a343bc6d57e`
- **#6139 Recertification**: `docs/context/evidence/issue_5305_certified_archive/recertification_issue_6139.json`
  (file SHA-256 `0d643f2c36d0f1f11e2be2351359567215d47ed216d156018fc6909a79a42cfe`)
- **Target Planner**: `social_force`
- **Target Planner Config SHA-256**: `dfdebd497e19a046e41cb2b1e7d7a7f54cd592ac0a465e4149efff19efa16735`
- **Fit Scenario Family**: `classic_group_crossing_medium`
- **Fit Entry Count**: 12 historic `social_force` entries (SHA-256 digest `2d5482648f18c2d779256abbad447b53ce3e43d1814d7f6a28e590cb81061b60`); 6 are currently `stress_only` and therefore not admitted.
- **Excluded Entries**: 5 `goal` entries from `classic_cross_trap_medium` explicitly excluded from fitting.
- **Search Space**: `configs/adversarial/issue_3275_cross_trap_ttc_space.yaml`

## 2. Integrity Repairs Applied

1. **Fit-Only Isolation**: The proposal model can only be constructed from admitted `social_force` fit entries. The 5 `goal` cross-trap entries are excluded prior to model initialization. Modifying or adding excluded entries cannot alter the proposal model state or candidate ranking.
2. **Train-Only Ranking Enforcement**: Candidate scoring and ranking use only fit entries.
3. **Independent Outcomes Authoritative**: When independent planner execution outcomes are present, top-level proposal/random metrics, comparison improvements, and the `#2921` stop rule are derived exclusively from independent planner execution outcomes. Archive-nearness objective values are moved to a diagnostic-only namespace (`diagnostic_archive_nearness_metrics`) and cannot drive decision rules. Opposite-sign tests confirm that decisions strictly follow independent execution.
4. **Row-Level Outcome Lineage (`adversarial_independent_outcomes.v2`)**: Each candidate execution row requires complete lineage (manifest SHA, planner SHA, seeds, execution commit, certification status, replay/record hash). Fallback or degraded execution statuses fail closed.
5. **Frozen Estimand & Decision Rule**: Primary estimand is proposal minus random candidate-level certified failure yield. The only admissible decision vocabulary is `continue`, `stop`, or `inconclusive`; invalid or fallback outcomes are `inconclusive`.
6. **Feature Semantics Audit**: Spatial and dynamic features are frozen for `classic_group_crossing_medium`. Cross-family transfer without verified invariant feature mapping is rejected to prevent uncalibrated cross-map scoring; this audit is currently unresolved.

## 3. Side-Effect-Free Contract Check Command

```bash
uv run python scripts/adversarial/run_proposal_vs_random_issue_2921.py --check-contract configs/adversarial/issue_3275_same_planner_contract.json
```
