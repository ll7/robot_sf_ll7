# Issue #3275 Frozen Same-Planner Contract and Held-Out Result Integrity Repair

This context document records the frozen contract and held-out result integrity repair for issue #3275 (sub-issue #6103).

## 1. Frozen Study Inputs

- **Tracked Archive Path**: `docs/context/evidence/issue_5305_certified_archive/archive.json`
- **Archive Raw SHA-256**: `79e022587b35c1c42bc07cfefaf882af473e96841a99ef57f98a4cee26636445`
- **Archive Payload SHA-256**: `1d8db3638e3539dca2e06849c0727150aff41ea85a71ee2d22cb1a343bc6d57e`
- **Target Planner**: `social_force`
- **Target Planner Config SHA-256**: `dfdebd497e19a046e41cb2b1e7d7a7f54cd592ac0a465e4149efff19efa16735`
- **Fit Scenario Family**: `classic_group_crossing_medium`
- **Fit Entry Count**: 12 `social_force` entries (SHA-256 digest `2d5482648f18c2d779256abbad447b53ce3e43d1814d7f6a28e590cb81061b60`)
- **Excluded Entries**: 5 `goal` entries from `classic_cross_trap_medium` explicitly excluded from fitting.
- **Search Space**: `configs/adversarial/crossing_ttc_space.yaml`

## 2. Integrity Repairs Applied

1. **Fit-Only Isolation**: The proposal model is constructed strictly from the 12 frozen `social_force` fit entries. The 5 `goal` cross-trap entries are excluded prior to model initialization. Modifying or adding excluded entries cannot alter the proposal model state or candidate ranking.
2. **Train-Only Ranking Enforcement**: Candidate scoring and ranking use only fit entries.
3. **Independent Outcomes Authoritative**: When independent planner execution outcomes are present, top-level proposal/random metrics, comparison improvements, and the `#2921` stop rule are derived exclusively from independent planner execution outcomes. Archive-nearness objective values are moved to a diagnostic-only namespace (`diagnostic_archive_nearness_metrics`) and cannot drive decision rules. Opposite-sign tests confirm that decisions strictly follow independent execution.
4. **Row-Level Outcome Lineage (`adversarial_independent_outcomes.v2`)**: Each candidate execution row requires complete lineage (manifest SHA, planner SHA, seeds, execution commit, certification status, replay/record hash). Fallback or degraded execution statuses fail closed.
5. **Frozen Estimand & Decision Rule**: Primary estimand is proposal minus random candidate-level certified failure yield. Decision rules (`continue | stop | revise | blocked`) follow predeclared thresholds.
6. **Feature Semantics Audit**: Spatial and dynamic features are frozen for `classic_group_crossing_medium`. Cross-family transfer without verified invariant feature mapping is rejected to prevent uncalibrated cross-map scoring.

## 3. Side-Effect-Free Contract Check Command

```bash
uv run python scripts/adversarial/run_proposal_vs_random_issue_2921.py --check-contract configs/adversarial/issue_3275_same_planner_contract.json
```
