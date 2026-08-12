# Issue #6938 — BRNE Progress Mechanism

Status: **mechanism signal reproduced; diagnostic-only**.

Bayesian Recursive Nash Equilibrium (BRNE) is a native external planner staged
locally at a pinned source commit. The exact corridor trace below isolates the
residual progress signal after the source-aligned action aggregation correction
in [#6934](https://github.com/ll7/robot_sf_ll7/issues/6934). It does not establish
planner quality, ranking, safety, realism, matched-compute parity, or paper
evidence.

## Frozen diagnostic

- Scenario: `classic_head_on_corridor_low` from
  `configs/benchmarks/issue_6464_brne_corridor_diagnostic.yaml`.
- Seeds: `111`, `112`, and `113`; horizon `500`; timestep `0.1` seconds.
- Native BRNE only for the mechanism interpretation; fallback is disabled.
- Upstream source: `MurpheyLab/brne` at
  `633a5cdcb39ab27f18b596cb8cb1968644f82391`, GPL-3.0, staged locally and not
  vendored or redistributed.
- Current Robot SF base: `b9232a265682309e4c804336987c2216674ca824`.
- Telemetry implementation: `0d704bd88`.

Reproduce with:

```bash
NUMBA_NUM_THREADS=1 LOGURU_LEVEL=WARNING TF_CPP_MIN_LOG_LEVEL=2 \
uv run python scripts/benchmark/run_brne_corridor_diagnostic_issue_6464.py \
  --config configs/benchmarks/issue_6464_brne_corridor_diagnostic.yaml \
  --output-dir output/benchmarks/issue_6938_brne_progress_<timestamp>
```

Focused planner/diagnostic tests passed (`56 passed`), and the native source
smoke passed (`6 passed`). The exact run completed with `3/3` pair coverage,
`3/3` native and eligible BRNE rows, no fallback/degraded rows, and no corridor
violations. The tracked compact handoff is
[`issue_6938_brne_progress_mechanism_summary.json`](evidence/issue_6938_brne_progress_mechanism_summary.json).

## Observed mechanism signal

The new trace separates the weighted command before safety clipping from the
selected command after clipping. All three native rows report the same pattern:

| Seed | Pre-clamp `v` first → last (m/s) | Clipped steps | `omega` range (rad/s) | Heading/goal delta first → last (rad) | Distance first → last (m) | Signed progress early / middle / late (m) | Terminal | Goal |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 111 | 0.400 → 0.040 | 0 | 0 → 0 | 0.118 → 0.219 | 4.298 → 2.330 | 0.657 / 0.660 / 0.652 | terminated | no |
| 112 | 0.400 → 0.040 | 0 | 0 → 0 | 0.112 → 0.198 | 4.591 → 2.619 | 0.657 / 0.661 / 0.654 | terminated | no |
| 113 | 0.400 → 0.040 | 0 | 0 → 0 | -0.058 → -0.094 | 5.130 → 3.141 | 0.661 / 0.666 / 0.662 | terminated | no |

The exact machine-readable report records the same selected and pre-clamp
actions, `plan_step_first` aggregation with control shape `[25, 42, 2]`, and
requested/effective sample counts `49/42`. Translational motion is non-degenerate
(`1.996 m` displacement per row), and signed goal-distance progress is positive
in every phase. Nevertheless, the action settles near `0.04 m/s` after the
initial command, so the three episodes terminate without a goal event.

The bounded heading/goal angular differences are direct trace observations; they
do not prove that heading or world-frame adaptation is correct in every setting.
They do, however, provide no gross frame-mismatch signal in this frozen slice.
The zero clipping count separates the observed low-speed command from the
safety clamp.

## Decision and boundary

This is a reproducible, trace-level mechanism signal: a common post-initial-step
low-speed command persists across all three native seeds while progress remains
positive but insufficient for goal reaching. It is not yet a causal explanation
of whether planner state, candidate weighting/normalization, or control-horizon
behavior produces the transition. Keep the result diagnostic-only.

Do not change the upstream source pin, action limits, fallback/degraded policy,
benchmark or ranking scope, safety/realism interpretation, or paper-facing
surfaces from this result. Raw episode JSONL and the staged GPL source remain
ignored, worktree-local artifacts; the tracked summary is a compact provenance
handoff rather than a raw episode archive.

The smallest next probe is [#6944](https://github.com/ll7/robot_sf_ll7/issues/6944),
which owns candidate command-generation, weighting/normalization, or
control-horizon explanations on the same frozen matrix. It must stop as
unresolved when the required native fields or a discriminating intervention
cannot be obtained.

## Report integrity

The exact local report directory is
`output/benchmarks/issue_6938_brne_progress_20260812T054636Z/`.

- `diagnostic_report.json` SHA-256:
  `1dd74349abdd75447e79e0819316af563be5508f08b25ef5ecf820f11db85137`
- `diagnostic_report.md` SHA-256:
  `d29cf3222e9ed8ef341c5598081c6f1a475c01e87ffca818d17cdef44c9c5722`
