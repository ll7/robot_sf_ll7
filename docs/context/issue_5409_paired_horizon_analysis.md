# Issue #5409 paired horizon analysis

Issue #5409 asks whether changing the fixed episode horizon from 500 to 600 steps
changes planner conclusions. The two arms are comparable only when the roster,
scenario matrix, seeds, kinematics, and execution provenance are held constant.

After both camera-ready campaign roots are complete, run the repository handoff:

```bash
uv run python scripts/benchmark/build_issue_5409_paired_horizon_report.py \
  --h500-root /path/to/issue5409_horizon_ablation_h500 \
  --h600-root /path/to/issue5409_horizon_ablation_h600 \
  --output-dir /path/to/issue5409_horizon_ablation_pair
```

The command emits three compact JSON artifacts in `--output-dir`:

- `matched_key_completeness.json` records the declared denominator and missing,
  extra, duplicate, metric-incomplete, and provenance-invalid keys;
- `paired_horizon_deltas.json` records per-key `h600 - h500` values plus planner and
  scenario-family point estimates; and
- `paired_uncertainty_summary.json` reports deterministic 95% percentile-bootstrap
  intervals over seed-level means at planner and scenario-family level.

The handoff is fail-closed. It writes `status: blocked` and no numeric rows when a
campaign is incomplete, a key is missing or duplicated, provenance drifts, checkpoint
staging is not submit-safe, or any row is fallback, degraded, unavailable, failed,
partial, or diagnostic-only. A `status: ready` artifact is nominal evidence for this
fixed ablation only; it is not paper-grade evidence and does not by itself establish a
horizon finding.

The default contract is the launch packet's 12 planners, 48 scenarios, seeds
`[111, 112, 113]`, 1,728 rows per arm, and scenario-matrix hash `c10df617a87c`.
Use the CLI options only for small fixture or contract-specific validation, not to
relax the issue #5409 acceptance boundary.

Before a rerun, execute the issue-specific guarded-PPO availability preflight on the
same public commit and environment. It must resolve both the staged checkpoint and
the registry observation contract. The h500/h600 configs keep `guarded_ppo` in the
full roster without a static dependency gate because that preflight now resolves
`available`; if it fails, the campaign remains blocked and the paired handoff must
emit no numeric evidence.
