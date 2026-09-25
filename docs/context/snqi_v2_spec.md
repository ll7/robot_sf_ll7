# SNQI-v2 specification

SNQI-v2 (Social Navigation Quality Index, score version 2) adds an explicit,
safety-stratified simulator aggregate alongside the existing release score.
**Calibration is pending:** the versioned anchor file deliberately cannot load
until the complete development split has produced reviewed anchors.

SNQI-v2 is a declared benchmark aggregate over simulator quantities. It is not a
validated measure of human comfort or safety, and it admits no deployment ranking
on its own.

## Score and normalization

`1*S - 2*C - .25*T - .25*N - .25*F - .10*J - .10*K`

| Term | Input | Normalization |
| --- | --- | --- |
| S | success | Binary indicator |
| C | total_collision_count | Binary indicator of any collision |
| T | time_to_goal_ideal_ratio | On success, clip((ratio−1)/2, 0, 1); on failure, zero |
| N | near_misses / episode.steps | clip(fraction/.25, 0, 1) |
| F | robot_force_impulse_total, or preregistered alternative | clip(value/calibration p95, 0, 1) |
| J | jerk_mean | clip(value/calibration p95, 0, 1) |
| K | curvature_mean | clip(value/calibration p95, 0, 1) |

Lower penalty anchors are physical zero. T measures excess above ideal time;
its zero penalty is ratio 1, and ratio 3 gives full penalty. The loader requires
all seven explicit weights and their rationales, finite positive upper anchors,
the complete source registry, and both strict stratum inequalities. Undefined
active inputs fail; there is no zero imputation. A failure's undefined time ratio
is harmless because T is explicitly zero for failures.

Collision-free successes score in [0.05, 1]; collision-free failures in [-0.70, 0].
Collisions score at most -1, with overall minimum -2.70. These strict strata apply
to individual episodes. Planner means combine success rate, collision rate and
mean quality penalties; they do not guarantee lexicographic ordering by collision
rate and then success rate. Quality differences can outweigh small differences
in planner outcome rates.

## Changes from release 0.0.7

| Released-index defect | V2 correction |
| --- | --- |
| Implicit curvature weight 1 | Explicit K weight .10 with rationale |
| Calibration and episode scalarizers differed | Shared normalization and score implementation for campaign and offline reports |
| Quality could compensate for collisions | Strict episode safety strata |
| Force exceedance and comfort counted the same events | Removed both; duplicate/derived-source registry check |
| Total force included goals and walls | Robot-attributable force component from issue #9666 |
| Mixed raw and median-floor normalization | Zero lower anchors; normative T/N and frozen development p95 F/J/K |
| Weights fitted to planner outcomes | Declared weights; no ranking-based selection |

`metrics.snqi`, `compute_snqi_v0` and `compute_snqi_v1` retain their existing
semantics. V2 adds `metrics.snqi_v2` and `metrics.snqi_v2_terms`.

## Calibration before evaluation

Use `configs/benchmarks/snqi_v2/calibration.dev101_102.yaml`: the frozen 14-arm
release roster, all 48 scenarios, development seeds 101/102, horizon 600 and dt .1.
The config preserves planner/checkpoint references, requires force recording,
and disallows prerequisite fallback. V2 scoring is disabled for acquisition.
The canonical preflight and checkpoint staging gates must pass before submission.
Declared kinematic command adapters are legitimate members of the frozen roster
(for example, Social Force projects velocity vectors into unicycle commands).
They are distinct from fallback or degraded policy execution, which is rejected.
Calibration records `benchmark_execution: nonfallback` and each arm's actual
`command_mode_counts` over its 96 episodes; it never relabels adapter commands as native.

`v2_calibration.derive_calibration_anchors` requires every one of the 1,344 unique
arm/scenario/seed combinations. It computes episode-level Spearman correlation
between simulated force impulse and normalized N. At absolute rho >= .90 it
selects `robot_force_pp_equiv_impulse_total`; otherwise it retains the simulated
component. It records the choice and correlation, linear-interpolated p95 F/J/K,
run ID, source commit, episode hash, split identity and grid hash. The helper also
records the unsaturated exposure-fraction correlation for comparison; only clipped N
is the preregistered decision input. A constant
correlation, nonpositive p95, missing row or undefined selected source fails.
Derive the reviewed artifact with:

```bash
uv run python scripts/tools/analyze_snqi_contract.py \
  --campaign-root /durable/calibration-campaign \
  --freeze-v2-anchors configs/benchmarks/snqi_v2/anchors.v2.0.json
```

The helper confines source files to the archived `runs/` directory, verifies row
source commits against the campaign manifest, and records per-file hashes.

In particular, undefined pedestrian–pedestrian-equivalent force on one-step
traces is never replaced with zero. The 384-row four-arm diagnostic is not this
calibration split and cannot establish the switch.

Commit and merge the reviewed anchors and quote all three asset hashes before
running evaluation seeds 111–140. Calibration seeds cannot be used for v2
evaluation. Preserve raw calibration episodes and manifests in durable storage;
local `output/` is temporary.

## Mandatory weight family

V2-F uses NumPy PCG64 with seed 20260924. Draw Dirichlet(1,1,1,1,1), scale to .95,
and reject draws with any quality weight below .02 until exactly 2,000 remain.
The deterministic grid adds equal weights (.19 each), five heavy-term vectors
(.55/.10/.10/.10/.10), and five leave-one-out vectors: selected term .02 and the
remaining declared proportions rescaled to .93. S=1 and C=2 stay fixed.
Two separately labeled relaxed-stratum vectors use S=.5, C=.5 or 1, and the
declared quality weights. They are excluded from strict-family summary statistics.

Every scored report emits both `reports/snqi_v2_diagnostics.{json,md}` and
`reports/snqi_v2_family.{json,md}`. They include declared planner means and paired
seed-bootstrap 95% intervals, rank correlations against declared/success ordering,
top-1 frequencies, top-3 overlap, pairwise flips, and leave-one-out rank changes.
The separate existing ranking-stability helper resamples seeds independently per
planner and is explicitly labeled unpaired; the confidence intervals share seed draws.
Undefined correlations are null. Correlations use average ties; tied top-1 values
split credit; top-3 boundary ties use lexical arm identity. Bootstrap intervals
with very few seeds are diagnostic and do not establish population precision.


## Bounded report memory and producer custody

Campaign enrichment reads and rewrites one JSONL episode at a time. Reports retain only scalar
score inputs and episode/arm identities; force samples and planner/simulation traces stay on disk.
Memory therefore scales with the compact episode table plus the largest decoded episode. The
2,013 family vectors and paired coverage checks are unchanged. Offline report recomputation uses
the same compact projection.

Before replacement, enrichment validates the existing producer sidecar and its original JSONL
hash and row identities. It stages the new JSONL and sidecar, records the original input hash,
new output hash and specification provenance in `snqi_v2_enrichment`, and updates the sidecar's
raw-artifact hash. All paired-report checks pass before any source file is replaced. Individual
replacements are atomic; the file set is not a filesystem transaction. An interruption between
replacements leaves a hash mismatch that downstream custody checks reject. Repeating a completed
enrichment leaves episode and sidecar bytes unchanged. Extra disk space for staged JSONL files is
required until replacement completes. Legacy field values are preserved; JSON formatting may change.
