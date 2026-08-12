# Continual-Adaptation Campaign Evidence Bundle (Issue #6657)

Campaign integration protocol-fixture evidence for continual-adaptation
nominal/shift/forgetting wiring. This deterministic bundle names the
validator-derived adapted-policy identifier distinct from the baseline and
exercises the promotion gate in `check_continual_adaptation_run`.

## Claim Boundary

Metadata-only campaign integration fixture. The fixture's schema value
`promotion_decision: promote` is not an empirical promotion decision. This
bundle does not represent an executed adaptation, evaluation, promoted policy,
or benchmark/paper evidence. It demonstrates only the wiring between the
continual-adaptation protocol contract (issue #6582), the merged proximal policy
optimization (PPO) manifest builder, and the benchmark campaign machinery.

## Contents

- `nominal_result.json` -- Metadata-only nominal result fixture.
- `shift_result.json` -- Metadata-only shift result fixture.
- `forgetting_result.json` -- Metadata-only forgetting result fixture.
- `evidence_bundle.yaml` -- Deterministic versioned bundle containing exact
  SHA-256 references to those three fixture files.

## Provenance

- Schema: `continual_adaptation_evidence.v1`
- Protocol: `continual_adaptation_run.v1`
- Baseline: `ppo_ammv_baseline_v3`
- Derived adapted-policy identifier:
  `ppo_ammv_baseline_v3#continual-adaptation@sha256:7e1d8e6036fa246c`
- Evidence boundary:
  `protocol_contract_only_no_training_no_checkpoint_write_no_safety_wrapper_mutation_no_policy_promotion_no_benchmark_or_paper_evidence`
- Exact-checksum validation:
  `uv run python scripts/benchmark/run_continual_adaptation_campaign.py --manifest configs/benchmark/continual_adaptation_promotion_fixture.yaml --validate`
