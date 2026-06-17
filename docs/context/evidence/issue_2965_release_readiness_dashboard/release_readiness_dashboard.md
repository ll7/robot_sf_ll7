# Release Readiness Dashboard

Generated (UTC): 2026-06-17T05:04:38.615568Z
Schema: release-readiness-dashboard.v1

## Claim Boundaries
- Source-local evidence only; no benchmark/paper claim is promoted beyond cited source notes.
- Rows classified as `diagnostic_only` are caveated and should not be promoted to benchmark/paper claims.

## Sources
- **claim_map**: docs/context/issue_2943_fast_results_claim_map_v0.md
- **handoff**: docs/context/issue_2689_release_evidence_handoff_2026_06_15.md
- **catalog**: docs/context/catalog.yaml

## Ready Claims
- cm-v0.benchmark_release.row_claim_enforcement: benchmark_row_claim.v1 row-level enforcement on leaderboard sidecars (schema)
- cm-v0.mechanism.trace_schema: mechanism_trace.v1 source contract for local-navigation intervention rows (schema)
- cm-v0.prediction.full_planner_integration: Forecast variant integrated into a real planner consuming ProbabilisticPredictor (smoke)

## Diagnostic-Only Claims
- cm-v0.benchmark_release.seed_table_semantics: Release 0.0.2 secondary-report table semantics clarified (diagnostic)
- cm-v0.prediction.denominator_health: Horizon x timestep denominator-coverage audit (diagnostic)
- cm-v0.prediction.native_replay: Forecast-variant closed-loop replay path is native (not fallback) (diagnostic)

## Blocked Claims
- cm-v0.benchmark_release.odd_coverage: ODD/hazard-scenario coverage matrix for v0.1 release (blockers: blocked by source gate)
- cm-v0.benchmark_release.suite_freeze: Nominal/stress/adversarial/AMV suite freeze for v0.1 release (blockers: 2910)

## Missing Hazard Coverage
- cm-v0.benchmark_release.odd_coverage: odd_hazard_coverage.v1 schema and JSON/Markdown matrix exist, but all represented rows remain config-only/weakly-covered and several hazard families remain blocked or absent

## Missing Durable Artifact Pointers
- None

## Next Executable Issue per Blocked/Incomplete Requirement
- Secondary-table paper promotion (p1_after_gate): 2612
- ODD/AMV suite freeze for v0.1 (p1_after_gate): 2910
- Forecast variant benchmark campaign (p1_after_gate): 2966
- Full benchmark claim matrix for #2910 release (parked_blocked): 2612
- CARLA replay transfer evidence (parked_blocked): 2158
- AMV calibrated-actuation paper claim (parked_blocked): 2230
- SocNavBench planner rows (parked_blocked): 2397
- Forecast variant safety/success claims (parked_blocked): 2941

Diagnostic-only rows are allowed for local execution diagnostics and must not be promoted as benchmark/paper evidence.
