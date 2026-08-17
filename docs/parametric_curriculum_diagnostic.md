# Leakage-safe parametric curriculum diagnostic

This diagnostic checks whether a structured social-navigation curriculum can be represented and
replayed without leaking training scenarios into a held-out evaluation split. It is a fixture-only
methodology check: it does not train a policy, run the simulator, or establish a safety or
benchmark result.

## What it records

The config declares interpretable scenario dimensions for density, pedestrian speed, constriction,
interaction type, robot speed, and adversariality. The command creates three matched method cards:

- `no_curriculum`: fixed midpoint training vectors;
- `random_curriculum`: seeded uniform draws;
- `structured_curriculum`: deterministic progression through the declared ranges.

Each method gets an independently hashed training manifest and shares a separately hashed evaluation
manifest. The report fails closed when scenario identities or parameter-vector hashes overlap, and it
recreates every manifest from its seed to verify replay.

## Run the smoke

```bash
uv run python scripts/validation/run_parametric_curriculum_diagnostic.py \
  --config configs/training/ppo/ablations/issue_7316_parametric_curriculum_smoke.yaml \
  --output /tmp/issue-7316-parametric-curriculum.json
```

The output is `parametric_curriculum_diagnostic.v1` with `diagnostic-only` evidence. Its
`training_executed`, `simulator_executed`, and `benchmark_evidence` fields remain false. A later
training issue must provide matched policy/compute/seed budgets, independent manifests, held-out
metrics, and the required domain-aware approval before any outcome claim is considered.

The design is methodologically inspired by the structured-parametric curriculum paper linked from
issue #7316; it is not a reproduction of that source or its results.
