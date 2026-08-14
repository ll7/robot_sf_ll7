# Issue #6095 S10 ORCA/PPO Nominal-vs-Stress Discriminability Calibration

**Status**: the frozen S10 nominal and stress campaigns completed on 2026-08-14. The current
paired report is **diagnostic_ready_for_domain_review** with interpretation allowed for the
configured nominal-versus-stress diagnostic. Both arms have staged, identity-matched,
computed-file checkpoint receipts; runtime checkpoint load status remains `not_run`, so this is
not runtime hash/load or paper-facing evidence.

[The report builder](../../scripts/benchmark/build_issue_6095_discriminability_report.py) fails
closed on matrix, row-identity, runtime-contract, fallback/degraded, or PPO-provenance violations
and writes a machine-readable JSON report plus a compact Markdown handoff.

## Execution result (2026-08-14)

Both campaigns used commit `fcc495b955c9eab00bc60842b5cae63f74cf2e2c`, seeds 111--120, horizon
100, `dt=0.1`, and differential-drive kinematics. The nominal campaign was job 14408 with 80
episodes; the provenance-only stress rerun was job 14411 with 960 episodes. Both campaign
receipts report valid execution with zero fallback/degraded, unavailable, or failed rows. The
earlier stress job 14083 remains historical context only; the current paired report uses job
14411.

The raw episode contract is homogeneous within each arm: ORCA is `adapter`, PPO is `native`, and
both records expose `tracked_agents_no_noise`. The campaign summaries still carry fairness mismatch
warnings (including an obsolete `ppo=mixed` description), so this report does not rank planners or
interpret the result as an algorithm comparison.

The rebuilt report is `diagnostic_ready_for_domain_review` with no blockers. It observed higher
nominal success for both planners in S10 and in the first-three-seed S3 subset:

| seed schedule | nominal ORCA | stress ORCA | nominal PPO | stress PPO |
|---|---:|---:|---:|---:|
| S3 | 0.2500 | 0.0556 | 0.3333 | 0.1528 |
| S10 | 0.2500 | 0.0771 | 0.4000 | 0.1729 |

The stress floor classified 7 scenarios as both-planners-some-success, 6 as exactly-one-planner-
some-success, and 35 as both-planners-zero-success. Of the both-zero scenarios, 23 had a
non-equal observed collision or near-miss outcome (13 collision-discriminated; 20 near-miss-
discriminated). These are descriptive configured-matrix observations, not accepted planner or
safety claims.

The compact handoff, including all 48 stress scenario classifications and artifact checksums, is
under `docs/context/evidence/issue_6095_s10_discriminability_2026-08-14/`. The raw episodes remain
in the private operations retrieval roots pending durable artifact promotion.

## Summary

Two issue-specific benchmark configs compare ORCA and PPO across the nominal
`nominal_v1.yaml` matrix (4 scenarios) and the stress
`classic_interactions_francis2023.yaml` matrix (48 scenarios). Both use
the frozen S10 (ten-seed) schedule 111-120 (`paper_eval_s10`), horizon 100, dt=0.1, differential-drive
kinematics.

## Configs

- `configs/benchmarks/issue_6095_nominal_discriminability_v1.yaml`
- `configs/benchmarks/issue_6095_stress_discriminability_v1.yaml`

Both configs include only ORCA (`algo: orca`) and PPO
(`algo: ppo`, `algo_config: configs/baselines/ppo_15m_grid_socnav.yaml`).

Both configs also use
`configs/benchmarks/route_clearance_certifications_v1.yaml`, preserving the
reviewed caveats for intentionally tight route geometry. ORCA uses
`socnav_missing_prereq_policy: fail-fast`: a missing native prerequisite ends
the campaign preflight rather than creating a fallback result row.

## Proposal-stage preflight record (2026-07-22)

Both configs passed metadata-only preflight together on rebased source revision
`d791c08f70b9af20f93babd5f1f17b06d581a185`. The packet records each
repository-relative source config and its full SHA-256:

| Property | Nominal | Stress |
|---|---|---|
| Scenarios | 4 | 48 |
| Planners | orca, ppo | orca, ppo |
| Seeds | 111-120 (10) | 111-120 (10) |
| Expected rows | 80 | 960 |
| Config hash | `60448a7228d1a450` | `0375e182d186a8bc` |
| Scenario matrix hash | `e5fc81d3eef3` | `6b1f3a702703` |
| Source config SHA-256 | `3bf27cc362055e6874125f93b793c70f099ce6049641b60c1cb69974b3a55df7` | `e8f8b56097964568da4784054d23e1c590c14d32634c8ec6d465f735d1208dc6` |
| Route-clearance warnings | 2: 1 certified, 1 unresolved | 15: 15 certified, 0 unresolved |
| Native ORCA prerequisite | `rvo2` import passed | `rvo2` import passed |
| PPO checkpoint resolved | yes | yes |
| PPO checkpoint status | stageable_remote | stageable_remote |
| Submit safe | no (metadata-only) | no (metadata-only) |
| PPO checkpoint SHA256 | `2b30df81...` | `2b30df81...` |

The 15 stress warnings carry the reviewed Issue #1105 geometry caveats; they
are not uncaveated planner-attribution evidence. The nominal
`empty_map_8_directions_east` warning remains an unresolved map-level warning,
so a later campaign must preserve that caveat rather than infer a planner
effect from it.

## Tests

`tests/benchmark/test_issue_6095_s10_discriminability_configs.py` validates:

- Planner entries are ORCA and PPO only
- PPO algo_config points at existing `ppo_15m_grid_socnav.yaml`
- Seed policy uses `paper_eval_s10` (111-120)
- Resolved seed inventory matches S10 exactly
- Horizon=100, dt=0.1, kinematics=differential_drive
- Both configs share same planner rows, seed policy, horizon/dt/kinematics
- Expected row counts: 80 (nominal), 960 (stress)
- Configs reference different scenario matrices
- Route-clearance certification provenance, fail-fast ORCA prerequisite policy,
  repository-relative route-override preview provenance, and a worktree-independent
  nominal scenario matrix hash

## Evidence

Compact preflight outputs archived at:
`docs/context/evidence/issue_6095_s10_discriminability_2026-07-22/`

The tracked packet keeps provenance paths repository-relative and uses LF line
endings for CSV artifacts. It is a portable config/preflight record, not raw
campaign evidence.

The 2026-08-14 execution handoff is archived at:
`docs/context/evidence/issue_6095_s10_discriminability_2026-08-14/`.
It records both campaign IDs, job IDs, report checksums, the complete stress
floor classification, and staged checkpoint receipts with the explicit runtime-load caveat.
The raw JSONL episodes are not tracked in Git.

Rebuild the report from retrieved campaign roots with:

```bash
uv run python scripts/benchmark/build_issue_6095_discriminability_report.py \
  --nominal-root <retrieved-nominal-root> \
  --stress-root <retrieved-stress-root> \
  --output-dir <report-output-dir>
```

## Execution Requirements

The governed SLURM execution is complete for the frozen 80-row nominal and
960-row stress matrices. The current bounded diagnostic may proceed to domain review because
the rerun replaced the metadata-only stress receipt with a staged, checksum-verified receipt.
Runtime load status is still `not_run`; any stronger runtime, planner-ranking, safety, or
paper-facing claim requires additional proof and resolution of the fairness and overlap caveats.

## PPO Model Provenance

- **Checkpoint**: `ppo_expert_issue_791_reward_curriculum_eval_aligned_large_capacity_20260417`
- **Source job**: 11724 (auxme-imech093, L40s, 8h04m)
- **WandB**: ll7/robot_sf/ibo3aqus
- **Training config**: `configs/training/ppo/ablations/expert_ppo_issue_791_reward_curriculum_promotion_10m_env22_eval_aligned_large_capacity.yaml`
- **Checkpoint SHA256**: `2b30df812bfcc737924b126b0763d69c567fe20716dc1c1eba8f56f926b49c1d`

### Overlap Caveat

Per the PPO baseline doc (`configs/baselines/ppo_15m_grid_socnav.yaml`):

> This policy was trained on the eval superset
> `configs/scenarios/sets/ppo_full_maintained_eval_v1.yaml`.

That eval superset includes `classic_interactions_francis2023.yaml` (the stress
matrix) and `atomic_navigation_minimal_full_v1.yaml`. This means:
- **Stress matrix evaluation is in-distribution** for PPO.
- **Nominal matrix evaluation** uses scenarios (`empty_map_8_directions_east`,
  `single_ped_crossing_orthogonal`, `classic_doorway_low`,
  `classic_bottleneck_low`) that do not appear in the documented training set
  components, but may overlap through atomic archetype inclusion.
- This does not block the ORCA-vs-PPO comparison (both planners see the same
  scenarios), but limits any generalization or planner-family superiority claims.

## Claims and Limitations

The campaigns are valid execution evidence: all planned rows completed with no
fallback/degraded rows, and the report verifies raw planner/scenario/seed
identity, commit, horizon, time step, kinematics, and model identity. The
combined analysis is **diagnostic-only** because the checkpoint receipts prove staged,
identity-matched files but do not prove that the declared hash was the file loaded at runtime.

Conditional observations support the configured-matrix stress-floor hypothesis:
both planners have higher nominal success in S3 and S10, and 23 of 35
both-zero stress scenarios have non-equal collision or near-miss outcomes.
These observations must not be promoted to a planner-family ranking, safety,
transfer, unseen-scenario generalization, or paper-facing claim. The campaign
summary also carries fairness mismatch warnings; the report preserves them and
does not rank ORCA against PPO. See issue #6095 for the full analysis contract.

## References

- Issue #6095: benchmark campaign specification
- Issue #1344: paired nominal/stress AMV protocol (parent config pattern)
- `docs/context/issue_691_benchmark_fallback_policy.md`
- `docs/context/artifact_evidence_vocabulary.md`
- `configs/baselines/ppo_15m_grid_socnav.yaml`
