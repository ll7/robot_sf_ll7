# Issue #1496 BC-warm-start comparison packet

The checked-in packet freezes the first outcome-free comparison between an RL-only PPO arm and
a behavior-cloning (BC) warm-start followed by the approved PPO fine-tuning path. It is a launch
and validation contract, not a training result.

## What is fixed

- Both arms use the five matched training seeds declared in the packet and `15,000,000`
  post-initialization environment steps.
- Validation uses the `validation` split for checkpoint selection. Held-out evaluation uses the
  canonical evaluation seeds from the oracle split contract.
- Only `best_validation` and `final` checkpoints are retained. DAgger, new data collection, and
  benchmark execution are outside this packet.
- Sample efficiency is the validation trajectory as a function of environment steps. Final
  performance is held-out evaluation at the predeclared selected checkpoint. Missing evaluations
  are reported, never imputed.

## Commands

Contract-only validation (safe in a public checkout):

```bash
uv run python scripts/validation/check_oracle_imitation_comparison.py \
  --config configs/training/ppo_imitation/oracle_imitation_comparison_issue_1496.yaml \
  --check --json
```

The strict loader gate additionally needs an authorized private artifact root. It verifies every
shard's path, size, and SHA-256 before any model construction:

```bash
uv run python scripts/validation/check_oracle_imitation_comparison.py \
  --config configs/training/ppo_imitation/oracle_imitation_comparison_issue_1496.yaml \
  --artifact-root <authorized-private-root> --check-artifacts --json
```

For local tests, a synthetic or disjoint row inventory can run the one-update startup canary:

```bash
uv run python scripts/validation/check_oracle_imitation_comparison.py \
  --config <fixture-comparison.yaml> --canary --canary-output /tmp/issue-1496-canary --json
```

The canary proves row loading, finite feature/action shape handling, a bounded model update,
checkpoint round-trip, and evaluation startup. Its loss values are diagnostics only; they are not
BC quality, benchmark, or publication evidence.

The nominal private artifact remains blocked from a public checkout by design. A later compute
lane must re-run the strict gate against the durable dataset pointer, complete the startup
canary, and preserve training/checkpoint/evaluation manifests before any result can contribute to
#1496 or #1489.
