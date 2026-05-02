# Highest-Success Policy Autoresearch

Date: 2026-04-30

## Contract

Goal: find the highest local policy-search success rate available in this
worktree without running SLURM-only stages.

Primary metric:

```bash
LOGURU_LEVEL=WARNING uv run python scripts/validation/run_policy_search_candidate.py \
  --candidate <candidate> \
  --stage nominal_sanity \
  --output-dir output/ai/autoresearch/highest_success_policy/<run>
```

Metric extraction: `summary.success_rate` from the emitted `summary.json`.
Collision rate is retained as a constraint/caveat. Fallback or degraded
execution would not count as benchmark-strengthening evidence.

Local boundary: `smoke`, `nominal_sanity`, and `stress_slice` only. Full matrix
and robustness extension remain SLURM handoff work.

Scratch log:
`output/ai/autoresearch/highest_success_policy/results.tsv`.

## Results

| Experiment | Candidate | Stage | Success | Collision | Decision |
|---|---|---|---:|---:|---|
| Baseline | `hybrid_rule_v3_teb_like_rollout` | nominal_sanity | 0.1667 | 0.0000 | revise |
| Baseline | `hybrid_rule_v3_static_margin0_waypoint2` | nominal_sanity | 0.1667 | 0.1111 | revise |
| Incumbent | `hybrid_rule_v3_fast_progress` | nominal_sanity | 0.3333 | 0.0000 | revise |
| Discarded | `hybrid_rule_v3_fast_progress_3p5` | nominal_sanity | 0.2778 | 0.0000 | revise |
| Discarded | `hybrid_rule_v3_waypoint2_fast_progress` | nominal_sanity | 0.2222 | 0.0556 | revise |
| Discarded | `hybrid_rule_v3_fast_dynamic_relaxed` | nominal_sanity | 0.2778 | 0.0000 | revise |
| Validation | `hybrid_rule_v3_fast_progress` | stress_slice | 0.2917 | 0.0000 | tracked |

## Outcome

The best policy found in this bounded local pass is the existing
`hybrid_rule_v3_fast_progress` candidate:

- nominal_sanity: 0.3333 success, 0.0000 collision
- stress_slice: 0.2917 success, 0.0000 collision

The three new candidate hypotheses were reverted because none improved
nominal_sanity success over `hybrid_rule_v3_fast_progress`. The 3.5 m/s variant
and fast dynamic-relaxed variant stayed collision-free but regressed success to
0.2778. The waypoint2 fast-progress variant regressed success to 0.2222 and
introduced collisions.

## Remaining Risk

The incumbent still misses the nominal_sanity gate (`0.80`) by a wide margin.
Most remaining failures are `timeout_low_progress`; two crossing failures in the
incumbent run reached emergency-stop behavior. A future search should either
alter the route/local-minima logic more directly or move expensive broader
policy search to the SLURM workflow rather than continuing small local constant
tweaks.
