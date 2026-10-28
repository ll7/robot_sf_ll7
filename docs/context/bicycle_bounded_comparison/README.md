# Bounded bicycle comparison (development-seed diagnostic)

**New current-version execution on the 0.0.8 release code. Development seeds only. Not release evidence; nothing here is admitted into the thesis.**

## Question

The three supported compact planners (goal, social force, ORCA) run in their 0.0.8 release configurations. We replace the release differential-drive (DD) plant with a bicycle-model (BI) plant. The two plants share the same collision radius (1.0 m), forward-speed cap (2.0 m/s), acceleration and braking (1 m/s²) and the no-reverse policy. The BI plant has a 0.90 m wheelbase and 0.79 rad steering limit, giving a minimum turning radius of 0.89 m.

Does this swap change success, collisions or planner ordering in six turning/encounter cells? And can the planners turn a bicycle toward a goal in an empty world?

The whole T60 profile is deliberately *not* loaded: its 0.64 m disc and 1.34 m/s cap would confound the comparison. Creep is off.

The design, its outcome rule and the analysis were frozen in [`contract.yaml`](contract.yaml) at commit `3cf7d7c3`, before any main episode ran. Production code is byte-identical to `373dbfde`; this directory only adds a harness and an analyzer.

**Stage 1** reviewed the existing bicycle records and found them insufficient for this question. See [`provenance/stage1_record_review.md`](provenance/stage1_record_review.md).

## What ran

- **Scenario episodes:** 1,080 = 3 planners × 6 cells × 2 plants × 30 seeds (1161–1190).
- **Empty-world probes:** 150 = 3 planners × 5 bearings (0/45/90/135/180°) × 2 plants × 5 seeds (1121–1125).
- **Cells:** widened doorway (3.6 m), ordinary doorway, bottleneck, head-on corridor, cross-trap, blind corner.
- **Coverage:** 0 missing pairs.
- **Mode:** 1,230/1,230 rows are `native`, i.e. the requested planner/plant path ran with no fallback and no degradation. ORCA and social force use the documented adapter on both plants; goal is native.
- **Creep:** 0 creep steps.
- **Pairing:** paired reset states are identical across plants.
- **Qualification:** pilot seeds 1001–1008 were used only for qualification and are not pooled; see [`evidence/episodes_pilot_qualification.csv`](evidence/episodes_pilot_qualification.csv).

## Results

Effects are BI minus DD in percentage points, paired by planner, cell and seed. Intervals are 95 % seed-block bootstrap intervals (10,000 draws). Pooled values are equal-cell means over the six cells.

| Planner | DD success | BI success | Δ success [95 % CI] | Δ collision [95 % CI] | Predeclared outcome |
|---|---:|---:|---:|---:|---|
| ORCA | 176/180 | 175/180 | −0.6 [−2.2, +1.1] | +0.6 [−1.1, +2.2] | O1 comparable in cells; O3 at 90/135/180° |
| Goal | 29/180 | 27/180 | −1.1 [−2.8, 0.0] | +1.1 [0.0, +2.8] | O1 comparable in cells (floor: 151/180 DD collisions); O3 at 180° |
| Social force | 65/180 | 53/180 | −6.7 [−11.7, −1.7] | −0.6 [−1.7, 0.0] | O5 open; ordinary doorway −53.3 [−70.0, −36.7]; O3 at 90/135/180° |

**Empty world.** On the DD plant, every planner reaches the goal at every bearing on every seed (75/75).

On the BI plant, the planners fail at the larger bearings while still reaching 0° and 45°:

| Planner | Bearings with 0/5 success | All five seeds end how? |
|---|---|---|
| ORCA | 90°, 135°, 180° | Timeout |
| Social force | 90°, 135°, 180° | Timeout |
| Goal | 180° only | Timeout |

In every one of these failures, the planner commands a zero-speed turn in place (v = 0, |ω| = 1 rad/s) that a creep-off bicycle cannot execute, so the robot never moves. A forward arc to each goal exists: the goal is 6 m away, the completion radius is 2 m, and the minimum turning radius is 0.89 m. The predeclared rule classifies this as a **controller/interface incompatibility (O3)**.

**Social-force ordinary doorway.** All 16 seeds where DD succeeded and BI failed end in timeout. In each, the last 10 s consist entirely of the same yaw-only command. ORCA succeeds on the bicycle on 29/30 seeds of this cell, so the cell is passable on those seeds. This is the same mechanism as the empty-world failures, not a geometric limit.

**Ordering.** All three pairwise planner differences exclude zero with the same sign on both plants, so there is no reversal: ORCA > social force > goal.

**Feasibility (O4).** No cell qualifies. Every cell has a bicycle success on at least 28 of 30 seeds.

Unrounded values are in [`evidence/summary.json`](evidence/summary.json) and [`evidence/paired_effects.csv`](evidence/paired_effects.csv).

## Interpretation limits

- One bicycle geometry; three planners; six authored cells. Development seeds only.
- The cell results come from route starts that roughly face the route. The empty-world probe shows that the same planners deadlock once the required heading change is large.
- The bicycle interface caps yaw rate by curvature (up to 2.25 rad/s); the DD plant caps it at 1.0 rad/s. This asymmetry is part of "the tested interface".
- These are intervals over seeds for a fixed roster. They are not simultaneous intervals and do not cover unseen scenarios.
- The 14-planner release roster, the learned policies and the hybrid were not tested.

## Reproduction

From the repository root at this branch, with Python 3.13.14 and `uv sync --frozen`:

```bash
for i in $(seq 0 9); do BOUNDED_OUTPUT=out uv run --frozen python \
  docs/context/bicycle_bounded_comparison/run_bounded.py \
  --contract docs/context/bicycle_bounded_comparison/contract.yaml \
  --stage main --worker $i --workers 10 & done; wait
cd docs/context/bicycle_bounded_comparison
uv run --frozen python analyze.py compact --results ../../../out/results --output /tmp/bb/episodes_main.csv
uv run --frozen python analyze.py analyze --episodes evidence/episodes_main.csv --contract contract.yaml --output /tmp/bb
sha256sum -c SHA256SUMS
```

Re-running compaction from the raw results and re-running the analysis reproduce all three evidence files byte for byte. Raw outputs (results, step traces and runner records) are preserved as the W&B artifact `wandb://ll7/robot_sf/campaign-xbike_bicycle_bounded_373dbfde_20261028:v0`; see [`provenance/custody.json`](provenance/custody.json).
