# Issue #2168 AMMV-Aware Social Force Pair Diagnostic

Issue: [#2168](https://github.com/ll7/robot_sf_ll7/issues/2168)
Status: diagnostic result as of 2026-06-03.

This note records the paired AMMV-aware Social Force diagnostic requested as the next proof step
after [issue_2154_ammv_social_force_model.md](issue_2154_ammv_social_force_model.md). The result
is mechanism evidence only. It is not benchmark-strength, calibrated, or paper-facing evidence.

## Scope

The diagnostic selected one named research-v1 AMMV slice:
`classic_head_on_corridor_low` from `configs/scenarios/issue_2168_ammv_social_force_pair.yaml`.
The benchmark probe compared default Social Force against
`configs/baselines/social_force_ammv_aware.yaml` with seeds `111`, `112`, and `113`.

The benchmark rows were retained as a diagnostic adapter check because they ran in `adapter` mode
and did not surface AMMV metadata. A direct `SocialForcePlanner` mechanism probe then bypassed the
adapter path to verify whether the AMMV term can activate in a controlled close-agent setup.

## Evidence

Tracked evidence pack:
`docs/context/evidence/issue_2168_ammv_social_force_pair_2026-06-03/`

Primary files:

- `summary.json`: compact metrics, checksums for the disposable local JSONL inputs, commands, and
  mechanism-probe output.
- `README.md`: human-readable interpretation.
- `SHA256SUMS`: checksums for the tracked summary and README.

Validation and execution commands:

```bash
uv run robot_sf_bench validate-config --matrix configs/scenarios/issue_2168_ammv_social_force_pair.yaml
uv run robot_sf_bench preview-scenarios --matrix configs/scenarios/issue_2168_ammv_social_force_pair.yaml
uv run robot_sf_bench run --matrix configs/scenarios/issue_2168_ammv_social_force_pair.yaml --out <worktree-local-default-jsonl> --base-seed 111 --repeats 1 --horizon 100 --dt 0.1 --record-forces --no-video --video-renderer none --algo social_force --workers 1 --no-resume --structured-output json
uv run robot_sf_bench run --matrix configs/scenarios/issue_2168_ammv_social_force_pair.yaml --out <worktree-local-ammv-jsonl> --base-seed 111 --repeats 1 --horizon 100 --dt 0.1 --record-forces --no-video --video-renderer none --algo social_force --algo-config configs/baselines/social_force_ammv_aware.yaml --workers 1 --no-resume --structured-output json
uv run python scripts/tools/run_ammv_social_force_pair_diagnostic.py --default-jsonl <worktree-local-default-jsonl> --ammv-jsonl <worktree-local-ammv-jsonl> --ammv-config configs/baselines/social_force_ammv_aware.yaml --scenario-config configs/scenarios/issue_2168_ammv_social_force_pair.yaml --output-dir docs/context/evidence/issue_2168_ammv_social_force_pair_2026-06-03
```

The tracked summary records SHA256 checksums for the disposable local JSONL inputs instead of
depending on paths under `output/`.

## Result

The benchmark probe produced three failure rows for each planner configuration. All rows timed out
at `max_steps`, reported zero collisions, and produced identical ordinary metrics:

- default min clearance: `0.484367` m;
- AMMV min clearance: `0.484367` m;
- default mean speed: `1.215808` m/s;
- AMMV mean speed: `1.215808` m/s;
- AMMV metadata surfaced in episode JSONL: `False`.

The direct mechanism probe did activate the AMMV term:

- AMMV max force magnitude: `2.641802`;
- AMMV max intrusion count: `1`;
- final robot lateral-offset delta: `0.201059` m;
- minimum robot-pedestrian clearance delta: `0.035677` m.

Observed implication: the AMMV term can activate and alter the robot-planner action in a controlled
close-agent probe, but the current benchmark adapter path does not expose an AMMV-specific paired
comparison. Confidence is about 0.9 for this diagnostic boundary because the direct mechanism probe
is deterministic and the adapter-mode rows were identical, but it does not measure pedestrian
lateral deviation or pedestrian speed adaptation.

## Claim Boundary

Classification: `diagnostic`.

This result supports only the mechanism claim that the AMMV term can activate in the direct
`SocialForcePlanner` path and that the current benchmark adapter path is not sufficient for an
AMMV-vs-default benchmark comparison. It does not support a performance improvement claim,
scenario-general claim, calibrated AMV behavior claim, or paper-facing benchmark result.

Unsupported requested fields:

- `pedestrian_lateral_deviation`: unsupported because the direct probe is robot-planner-only.
- `pedestrian_speed_adaptation`: unsupported because pedestrian dynamics are simulator-owned.

## Next Proof Step

Choose one of two directions:

1. Wire the AMMV-aware `SocialForcePlanner` into a benchmark execution path that surfaces AMMV
   diagnostics, then rerun the paired scenario with tracked row metadata.
2. Keep AMMV as a mechanism-only planner variant and use this result to decide whether the
   research-v1 matrix should compare adapter-mode Social Force or direct planner-mode Social Force.

Until one of those paths runs with AMMV metadata in executable benchmark rows, keep
`research-v1.amv.model_behavior` at `diagnostic`.
