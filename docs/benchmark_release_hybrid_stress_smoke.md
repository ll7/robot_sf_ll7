# S30/H600 hybrid pre-release stress smoke

This diagnostic smoke is the recovery gate between a corrected planner/runtime
commit and a new full benchmark-data run. It is not a release, a software
version, or paper-facing benchmark evidence.

The contract is frozen in:

- campaign config: configs/benchmarks/paper_experiment_matrix_v2_h600_hybrid_stress_smoke.yaml
- scenario selection: configs/scenarios/sets/paper_matrix_v2_h600_hybrid_stress_smoke.yaml
- release manifest: configs/benchmarks/releases/paper_experiment_matrix_v2_h600_s30_hybrid_stress_smoke_v0_1.yaml

It retains the 14-arm production roster so the existing checkpoint-staging and
release-runner path is exercised. The required recovery targets are the four
hybrid arms:

- scenario_adaptive_hybrid_orca_v2_bottleneck_yield
- scenario_adaptive_hybrid_orca_v2_collision_guard
- hybrid_rule_v3_fast_progress_static_escape
- hybrid_rule_v3_fast_progress_static_escape_continuous

The selected stress cells are all run at seed 116 and horizon 600:

| Scenario | Mechanism |
| --- | --- |
| classic_urban_crossing_medium | urban crossing |
| classic_cross_trap_high | cross-trap |
| classic_doorway_high | doorway |
| francis2023_exiting_elevator | elevator exit |
| francis2023_robot_crowding | robot crowding |

This is a 14 × 5 × 1 = 70 episode-cell smoke. The cells were selected from
the four-arm fallback intersections in the durable issue #4365 job13376/job13378
artifacts, with the urban-crossing and other representative cells also
observed in the rejected issue #7742 job14730 artifact. The historical rows are
diagnostic selection evidence only and must never be reused as release evidence.

## Admission contract

A stress smoke is admissible only when the final campaign artifacts bind all
rows to one exact checked-out source SHA and the manifest/config/scenario hashes
match the launch packet. The tracked `review_base_commit` is audit context only;
it cannot be the final commit containing the manifest without creating a
self-reference loop. The runner records the exact checked-out `HEAD`, compares
it with `SLURM_EXPECTED_PUBLIC_COMMIT` when the private launch supplies that
pin, and rejects campaign metadata or rows with a different or mixed commit.
Do not submit a packet whose launch/runtime source identity is not exact.

The result must be rejected if any declared row, planner summary, or campaign
summary contains:

- fallback, degraded, unavailable, or failed status;
- a positive fallback counter or explicit fallback/degraded flag;
- the legacy `selected_source=all_candidates_rejected` or
  `selected_source=static_reorient` path;
- the legacy `planner_mode=EMERGENCY_STOP` or `planner_mode=REORIENT` mode;
- any positive `emergency_stop_count`, including one attached to an otherwise
  candidate-evaluated normal source;
- missing/false benchmark-success evidence, failed jobs, malformed rows,
  duplicate identities, mixed source/config provenance, or incomplete coverage.

Here, a failed *execution row* is different from a completed episode whose
scientific outcome is `collision` or `failure`. Those terminal outcomes are
retained as data and affect the reported component metrics; requiring every
navigation episode to succeed would censor the benchmark. The planner-level
run must still complete every declared identity and report `benchmark_success`.

A zero `fallback_count` does not make a legacy emergency-stop or
all-candidates-rejected row admissible. `fallback_count` remains an
independent counter: any positive value is still rejected. Dynamic escape and
other normal candidate-evaluated sources do not create an emergency-stop
exception. A current hybrid planner may instead emit an explicit native
`*_protective_stop` source and increment `protective_stop_count` when every
motion candidate fails its hard safety filters. That zero command is part of
the planner being evaluated, does not invoke another planner, and is therefore
not fallback or degraded execution. It remains outcome evidence: preserve and
report the count, and never reinterpret it as success or suppress its effect on
component metrics. The smoke contract never authorizes ranking, an SNQI claim,
or promotion of the stress slice into the 20,160-cell release.

## imech192 submission

Use the private-ops release wrapper on **imech192**, with the same module
initialization and resource envelope as the corrected full release packet. Keep
the execution on the canonical cluster lane; do not move this smoke to LiCCA
or run an ad-hoc shell command that bypasses the wrapper.

Before admission:

1. Check out the exact source worktree SHA named by the private launch packet.
2. Recompute and verify the campaign-config, scenario-matrix, scenario-source,
   and hybrid-config SHA-256 values.
3. Stage the checkpoint receipt against the stress campaign config and require
   submit_safe: true; the receipt must be fresh and config-bound even though
   only the hybrid arms are under test.
4. Run release preflight and confirm 14 arms, five scenarios, seed 116, H600,
   differential-drive, and 70 expected cells.
5. Bind the same source/config/scenario/manifest hashes into the private launch
   packet. Record the exact module setup and startup sentinel in the private
   receipt.
6. Submit through submit_and_record.sh on imech192. A non-zero diagnostic
   admission exit or any forbidden marker rejects the smoke; do not reuse its
   campaign ID. A zero smoke exit is `diagnostic_stress_smoke_passed`, never
   `release_benchmark_success` or `release_status: ok`.

A canonical local command shape is:

    uv run python scripts/benchmark/preflight_campaign_checkpoints.py \
      --config configs/benchmarks/paper_experiment_matrix_v2_h600_hybrid_stress_smoke.yaml \
      --stage \
      --report-path output/release/checkpoints/hybrid_stress_smoke_staging_receipt.json

    uv run python scripts/tools/run_benchmark_release.py \
      --manifest configs/benchmarks/releases/paper_experiment_matrix_v2_h600_s30_hybrid_stress_smoke_v0_1.yaml \
      --checkpoint-receipt output/release/checkpoints/hybrid_stress_smoke_staging_receipt.json \
      --campaign-id <new-fixed-stress-smoke-id>

The private wrapper should use the exact paths/hashes above, the corrected
release commit, and a fresh campaign id. The wrapper must preserve the full
campaign root even on a non-zero exit so a rejected smoke remains auditable.

## Decision boundary

- Passing smoke: eligible to prepare a fresh full S30/H600 preflight at the
  same exact source/config/checkpoint inputs. It does not authorize submission
  of the full campaign by itself.
- Failing smoke: retain the artifacts as diagnostic evidence, classify the
  failure, and correct the runtime/config/cluster issue before a new smoke id.
  Never relabel or filter failing rows and never reuse a rejected smoke as
  benchmark evidence.
- A code/config correction always requires a new immutable source SHA, fresh
  checkpoint receipt, fresh smoke id, and a rerun of this contract before the
  20,160-cell campaign.
