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
- modes compared: flat, hierarchical_scenario, hierarchical_seed

## Width ratios (hierarchical / flat)

ratio > 1 means the hierarchical interval is wider than flat (flat was anti-conservative).

| archetype | density | metric | mode | flat_width | hierarchical_width | ratio |
| --- | --- | --- | --- | --- | --- | --- |
| bottleneck | low | collision_rate | hierarchical_scenario | 0.3309 | 0.6714 | 2.03 |
| bottleneck | low | collision_rate | hierarchical_seed | 0.3309 | 0.1600 | 0.48 |
| bottleneck | low | success_rate | hierarchical_scenario | 0.3309 | 0.6714 | 2.03 |
| bottleneck | low | success_rate | hierarchical_seed | 0.3309 | 0.1600 | 0.48 |
| bottleneck | low | time_to_goal | hierarchical_scenario | 1.8902 | 4.0945 | 2.17 |
| bottleneck | low | time_to_goal | hierarchical_seed | 1.8902 | 1.8458 | 0.98 |
| crossing | high | collision_rate | hierarchical_scenario | 0.3309 | 0.6073 | 1.84 |
| crossing | high | collision_rate | hierarchical_seed | 0.3309 | 0.1600 | 0.48 |
| crossing | high | success_rate | hierarchical_scenario | 0.3309 | 0.6073 | 1.84 |
| crossing | high | success_rate | hierarchical_seed | 0.3309 | 0.1600 | 0.48 |
| crossing | high | time_to_goal | hierarchical_scenario | 1.9643 | 4.4223 | 2.25 |
| crossing | high | time_to_goal | hierarchical_seed | 1.9643 | 1.9706 | 1.00 |
| crossing | low | collision_rate | hierarchical_scenario | 0.3343 | 0.6533 | 1.95 |
| crossing | low | collision_rate | hierarchical_seed | 0.3343 | 0.2613 | 0.78 |
| crossing | low | success_rate | hierarchical_scenario | 0.3343 | 0.6533 | 1.95 |
| crossing | low | success_rate | hierarchical_seed | 0.3343 | 0.2613 | 0.78 |
| crossing | low | time_to_goal | hierarchical_scenario | 1.8848 | 4.1709 | 2.21 |
| crossing | low | time_to_goal | hierarchical_seed | 1.8848 | 1.8590 | 0.99 |

## Summary

### Rate metrics - hierarchical_scenario mode (primary anti-conservatism comparison)

- count: 6
- min ratio: 1.84
- median ratio: 1.95
- mean ratio: 1.94
- max ratio: 2.03

### Rate metrics - hierarchical_seed mode (optional seed-cluster variant)

- count: 6
- min ratio: 0.48
- median ratio: 0.48
- mean ratio: 0.58
- max ratio: 0.78

### Continuous metrics - hierarchical_scenario mode

- count: 3
- min ratio: 2.17
- median ratio: 2.21
- mean ratio: 2.21
- max ratio: 2.25

### Continuous metrics - hierarchical_seed mode

- count: 3
- min ratio: 0.98
- median ratio: 0.99
- mean ratio: 0.99
- max ratio: 1.00

ratio > 1 means the hierarchical interval is wider than the flat interval on the same records, i.e. the flat interval was anti-conservative (understated uncertainty) under clustering. The hierarchical_scenario (scenario-then-episode) mode is the documented successor-campaign procedure and is the primary anti-conservatism comparison; hierarchical_seed is the optional seed-level cluster variant. A seed-mode ratio < 1 can occur when the cell-level rate already absorbs most between-cell variance, leaving little between-seed dispersion.

## Reproduce

```bash
uv run python scripts/analysis/compare_flat_vs_hierarchical_intervals_issue_5139.py \
  --output-dir docs/context/evidence/issue_5139_hierarchical_bootstrap
```
