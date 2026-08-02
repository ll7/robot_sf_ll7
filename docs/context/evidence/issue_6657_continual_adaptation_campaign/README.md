# Continual-Adaptation Campaign Evidence Bundle (Issue #6657)

Campaign integration evidence for continual-adaptation nominal/shift/forgetting
evaluation wiring. This bundle names the validator-derived adapted-policy
identifier distinct from the baseline and satisfies the promotion gate in
`check_continual_adaptation_run`.

## Claim Boundary

Metadata-only campaign integration fixture. This bundle does not represent an
executed adaptation, a promoted policy, or benchmark/paper evidence. It
demonstrates the wiring between the continual-adaptation protocol contract
(issue #6582) and the benchmark campaign machinery.

## Contents

- `nominal_result.json` -- Nominal evaluation result reference.
- `shift_result.json` -- Shift evaluation result reference.
- `forgetting_result.json` -- Forgetting evaluation result reference.
- `evidence_bundle.yaml` -- Versioned evidence bundle naming the derived
  adapted-policy identifier.

## Provenance

- Schema: `continual_adaptation_evidence.v1`
- Protocol: `continual_adaptation_run.v1`
- Baseline: `ppo_ammv_baseline_v3`
- Derived adapted-policy identifier:
  `ppo_ammv_baseline_v3#continual-adaptation@sha256:7e1d8e6036fa246c`
- Evidence boundary:
  `protocol_contract_only_no_training_no_checkpoint_write_no_safety_wrapper_mutation_no_policy_promotion_no_benchmark_or_paper_evidence`
