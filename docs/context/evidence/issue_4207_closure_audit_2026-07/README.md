# Issue #4207 Closure Audit

This audit maps issue #4207 acceptance criteria to merged pull request (PR) evidence. The issue is
not ready to close: the diagnostic certification-transfer probe exists, but trained learned and
predictive planner arms still lack checkpoint-backed artifact provenance and a fresh central
processing unit (CPU) rerun without fallback execution.

Source issue thread checked: <https://github.com/ll7/robot_sf_ll7/issues/4207>, including seven
issue comments through 2026-07-04 20:38:42 UTC. This audit intentionally does not run a full
benchmark campaign, submit Slurm work, or promote a paper/dissertation claim.

## Criterion Evidence

| Acceptance criterion | Evidence | Status |
| --- | --- | --- |
| Pre-register a diagnostic certification-transfer probe config and release-gate spec. | PR #4221 added `configs/benchmarks/issue_4207_certification_transfer_probe.yaml`, `configs/benchmarks/release_gates/issue_4207_certification_transfer_gates.yaml`, and the CPU runner. PR #4502 added `--validate-only` preflight and fail-closed gate evaluability for `blocked_no_evaluable_gate_family`. | Met |
| Record pedestrian-model provenance for certification, evaluation, and development identities. | PR #4221 added separate `certification_pedestrian_model`, `evaluation_pedestrian_model`, and `development_pedestrian_model` report fields. Tests cover the distinction in `tests/benchmark/test_certification_transfer_issue_4207.py`. | Met |
| Produce a compact evidence artifact with a diagnostic claim boundary, not a deployment or paper claim. | PR #4221 added `docs/context/evidence/issue_4207_certification_transfer_probe_2026-07/`; PR #4412 refreshed the physics-verified packet at `docs/context/evidence/issue_4207_interacting_physics_2026-07/`. Both packet READMEs and `claim_boundary.md` state diagnostic-only scope. | Met |
| Transfer flips are valid model-assumption fragility results, not failed experiments. | PR #4221 introduced `certification_transfer_matrix.csv`, `flip_cases.csv`, and transfer statuses including `fragile_pass_to_fail` and `conservative_fail_to_pass`. PR #4315 exercised flip handling in the interacting synthetic smoke packet. | Met |
| Missing gate metrics produce `not_evaluable`, never `pass`. | PR #4221 added gate evaluation semantics and tests. PR #4502 added preflight that fails closed before execution when a required gate metric cannot be aggregated. | Met |
| Demonstrate the probe on an interacting scenario so stable transfer rows are not vacuous. | PR #4308 added interaction-validity classification and `model_sensitivity_exercised`. PR #4315 added a synthetic interacting smoke fixture. PR #4412 refreshed a physics-verified packet with `physics_near_field_confirmed=true`, `model_sensitivity_exercised=true`, and `max_robot_ped_within_5m_frac=0.10558655435990806`. | Met for geometry/near-field proof |
| Prevent fallback or missing-checkpoint learned/predictive rows from being read as trained-planner comparison evidence. | PR #4433 added `trained_planner_claim_status` and exclusions. PR #4480 added `trained_planner_readiness` and required `algo_config`, `checkpoint`, and `training_manifest` provenance fields. Current physics packet reports 12 `excluded_missing_checkpoint_or_config` rows and 4 `not_a_trained_planner` rows. | Met as fail-closed contract |
| Run a real trained-planner comparison with checkpoint-backed learned/predictive arms and no fallback execution. | The issue comments after PR #4412, #4433, and #4480 all keep #4207 open for this blocker. No resolved checkpoint/training-manifest inputs are present in the checked configs or compact packet. | Not met |

## Closure Decision

Leave issue #4207 open. The smallest remaining empirical slice is to attach real trained-arm
artifacts for the learned/predictive arms, prove `algo_config`, `checkpoint`, and
`training_manifest` provenance, and rerun the CPU certification-transfer probe so at least one
trained learned/predictive arm is `eligible` rather than `excluded_missing_checkpoint_or_config` or
`excluded_fallback_execution`.

This audit does not add another checker/guardrail. It is the high-churn consolidation artifact
requested by the #4207 fragmentation guard: blockers remaining, intentional non-claims, and the
next empirical action are recorded in one place.

