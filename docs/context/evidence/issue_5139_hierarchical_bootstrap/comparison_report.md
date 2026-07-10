<!-- AI-GENERATED NEEDS-REVIEW -->

# Flat vs Hierarchical Bootstrap Interval Comparison (issue #5139)

**Evidence status:** `diagnostic-only`

**Claim boundary:** diagnostic-only: synthetic structured episode bundle characterizing the implemented flat vs hierarchical bootstrap procedures. NOT benchmark evidence and NOT a paper claim. No repository campaign bundle matches the (archetype, density, scenario_id, seed) aggregation schema yet; the pre-registered 30-seed successor campaign this work unblocks has not run. Re-run on a real campaign bundle to obtain nominal benchmark evidence.

## Plain-language summary

The analysis layer previously resampled episodes as if they were all independent (flat bootstrap). Because episodes are actually grouped inside scenario cells and seeds, that flat method understates how uncertain a result really is. This artifact compares the old flat interval widths against the new hierarchical (scenario-then-episode) cluster bootstrap widths on a structured synthetic bundle, so the size of that understatement (the anti-conservatism) is documented. It is a diagnostic characterization, not benchmark evidence.

## Bundle

Synthetic structured bundle: 90 episodes across 3 (archetype, density) groups, 6 scenario cells per group, 5 seeds per cell, with cell-level success/collision probabilities inducing intra-cluster correlation.

- bootstrap_samples: 1000
- bootstrap_confidence: 0.95
- master_seed: 123
- bundle_seed: 20260710
- modes compared: flat, hierarchical_scenario, hierarchical_seed

## Width ratios (hierarchical / flat)

ratio > 1 means the hierarchical interval is wider than flat (flat was anti-conservative).

| archetype | density | metric | mode | flat_width | hierarchical_width | ratio |
| --- | --- | --- | --- | --- | --- | --- |
| bottleneck | low | collision_rate | hierarchical_scenario | 0.3343 | 0.6840 | 2.05 |
| bottleneck | low | collision_rate | hierarchical_seed | 0.3343 | 0.1600 | 0.48 |
| bottleneck | low | success_rate | hierarchical_scenario | 0.3343 | 0.6840 | 2.05 |
| bottleneck | low | success_rate | hierarchical_seed | 0.3343 | 0.1600 | 0.48 |
| bottleneck | low | time_to_goal | hierarchical_scenario | 1.8847 | 3.8768 | 2.06 |
| bottleneck | low | time_to_goal | hierarchical_seed | 1.8847 | 1.8340 | 0.97 |
| crossing | high | collision_rate | hierarchical_scenario | 0.3363 | 0.6914 | 2.06 |
| crossing | high | collision_rate | hierarchical_seed | 0.3363 | 0.3201 | 0.95 |
| crossing | high | success_rate | hierarchical_scenario | 0.3363 | 0.6914 | 2.06 |
| crossing | high | success_rate | hierarchical_seed | 0.3363 | 0.3201 | 0.95 |
| crossing | high | time_to_goal | hierarchical_scenario | 1.8583 | 4.3738 | 2.35 |
| crossing | high | time_to_goal | hierarchical_seed | 1.8583 | 1.9390 | 1.04 |
| crossing | low | collision_rate | hierarchical_scenario | 0.3363 | 0.6611 | 1.97 |
| crossing | low | collision_rate | hierarchical_seed | 0.3363 | 0.1307 | 0.39 |
| crossing | low | success_rate | hierarchical_scenario | 0.3363 | 0.6611 | 1.97 |
| crossing | low | success_rate | hierarchical_seed | 0.3363 | 0.1307 | 0.39 |
| crossing | low | time_to_goal | hierarchical_scenario | 1.8672 | 4.1059 | 2.20 |
| crossing | low | time_to_goal | hierarchical_seed | 1.8672 | 1.8529 | 0.99 |

## Summary

### Rate metrics - hierarchical_scenario mode (primary anti-conservatism comparison)

- count: 6
- min ratio: 1.97
- median ratio: 2.05
- mean ratio: 2.02
- max ratio: 2.06

### Rate metrics - hierarchical_seed mode (optional seed-cluster variant)

- count: 6
- min ratio: 0.39
- median ratio: 0.48
- mean ratio: 0.61
- max ratio: 0.95

### Continuous metrics - hierarchical_scenario mode

- count: 3
- min ratio: 2.06
- median ratio: 2.20
- mean ratio: 2.20
- max ratio: 2.35

### Continuous metrics - hierarchical_seed mode

- count: 3
- min ratio: 0.97
- median ratio: 0.99
- mean ratio: 1.00
- max ratio: 1.04

ratio > 1 means the hierarchical interval is wider than the flat interval on the same records, i.e. the flat interval was anti-conservative (understated uncertainty) under clustering. The hierarchical_scenario (scenario-then-episode) mode is the documented successor-campaign procedure and is the primary anti-conservatism comparison; hierarchical_seed is the optional seed-level cluster variant. A seed-mode ratio < 1 can occur when the cell-level rate already absorbs most between-cell variance, leaving little between-seed dispersion.

## Reproduce

```bash
uv run python scripts/analysis/compare_flat_vs_hierarchical_intervals_issue_5139.py \
  --output-dir docs/context/evidence/issue_5139_hierarchical_bootstrap
```
