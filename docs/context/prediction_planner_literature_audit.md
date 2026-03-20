# Prediction Planner Literature Audit

Date: 2026-03-20
Related issue:
- `robot_sf_ll7#590` audit predictive planner: lineage, references, capability, performance, and literature positioning

## Purpose

This note audits what the repository can responsibly claim about `prediction_planner`.
It is not a paper-style survey of predictive planning in general. It is a repository-grounded
interpretation of:

- what is actually implemented in `robot_sf_ll7`,
- which benchmark contracts are in force,
- what current benchmark evidence shows,
- and where the current implementation stops short of external-family reproduction.

## Executive Verdict

`prediction_planner` is a real, benchmark-runnable in-repo predictive local planner, but it should
currently be described as a weak local implementation and experimental benchmark challenger, not as
evidence that the broader predictive-MPC or prediction-aware literature family performs poorly.

The repository supports four strong claims:

1. `prediction_planner` is implemented natively in this repo and is not a placeholder.
2. The benchmark contract for this planner is explicit: experimental tier, adapter execution mode,
   unicycle command output, checkpoint-dependent runtime.
3. The planner has been executed in the local benchmark stack with integrity-gated campaign
   evidence.
4. The current result quality is too weak for headline benchmark or paper comparison.

The repository does **not** support two stronger claims:

1. that this is a faithful byte-equivalent reproduction of a specific external predictive-planning
   paper or codebase,
2. that current `prediction_planner` scores measure the ceiling of the predictive-MPC family.

## Canonical Evidence Surfaces

Use these files as the primary evidence base for any future discussion:

- implementation:
  - `robot_sf/planner/socnav.py`
  - `robot_sf/planner/predictive_model.py`
- benchmark contract:
  - `robot_sf/benchmark/algorithm_readiness.py`
  - `robot_sf/benchmark/algorithm_metadata.py`
  - `robot_sf/benchmark/predictive_planner_config.py`
  - `configs/algos/prediction_planner_camera_ready.yaml`
  - `model/registry.yaml`
- benchmark-facing interpretation:
  - `docs/baselines/prediction_planner.md`
  - `docs/benchmark_planner_family_coverage.md`
  - `configs/benchmarks/planner_quality_audit_v1.yaml`
- execution evidence:
  - `docs/benchmark_camera_ready.md`
  - `docs/context/benchmark_post_prediction_fix_2026-02-20.md`
  - `docs/context/issue_581_paper_evidence_delta.md`

## What Is Actually Implemented

### Runtime architecture

The runtime planner is `PredictionPlannerAdapter` in `robot_sf/planner/socnav.py`.
It does all of the following in-repo:

1. converts SocNav structured observations into model input,
2. loads a learned predictive model or optional fallback path,
3. predicts short-horizon pedestrian trajectories,
4. builds a deterministic finite `(v, omega)` candidate lattice,
5. rolls out robot motion over a short horizon,
6. scores candidates using goal, collision, near-field, TTC, occupancy, and progress-risk terms,
7. returns an adapter-compatible unicycle command.

This is important because it places the planner clearly in the category:

- implemented local predictive planner,
- deterministic sampled-rollout search,
- learned prediction + local optimization adapter,
- not full long-horizon tree search,
- not native external-repo evaluation code.

### Predictive model lineage

The learned forecast model lives in `robot_sf/planner/predictive_model.py`.
It is a compact message-passing trajectory predictor:

- MLP encoder,
- pairwise distance-based attention weighting,
- repeated message-passing blocks,
- decoder producing future trajectory deltas,
- ADE/FDE and masked trajectory loss helpers.

The module docstring already describes it as `RGL-inspired`, which is the right strength of claim.
Nothing in the implementation suggests a direct code import from a named external predictive-MPC
stack.

### Benchmark contract

The benchmark contract is explicit and conservative:

- readiness tier: `experimental`
- readiness note: `RGL-inspired predictive planner; requires trained checkpoint.`
- baseline category: `learning`
- policy semantics: `predictive_model_based_adapter`
- default execution mode: `adapter`
- default adapter name: `PredictionPlannerAdapter`
- planner command space: `unicycle_vw`

These values come from `robot_sf/benchmark/algorithm_readiness.py` and
`robot_sf/benchmark/algorithm_metadata.py`.

## Canonical Config And Provenance

The canonical benchmark config is:

- `configs/algos/prediction_planner_camera_ready.yaml`

Current canonical model id in that config:

- `predictive_proxy_selected_v2_full`

That model id is present in `model/registry.yaml`, so the benchmark-facing config and registry are
aligned.

Important repository note:

- some older predictive-planner docs still mention `predictive_proxy_selected_v1`,
- but the current camera-ready config and current training docs point to
  `predictive_proxy_selected_v2_full`,
- so benchmark-facing interpretation should treat the YAML config as canonical.

The current camera-ready config also makes the planner shape auditable. It enables:

- learned checkpoint resolution via `predictive_model_id`,
- sequence search (`predictive_sequence_search_enabled: true`),
- three-segment sequence search with bounded branching,
- adaptive horizon logic,
- phase logic and occupancy-aware scoring.

## Literature Positioning

### Safe claim

The planner is best positioned as:

- an in-repo predictive local-planning challenger,
- inspired by graph-based crowd prediction work and prediction-aware local planning,
- adapted to the Robot SF benchmark contract.

This matches:

- `docs/baselines/prediction_planner.md`
- `docs/benchmark_planner_family_coverage.md`
- `configs/benchmarks/planner_quality_audit_v1.yaml`

### Unsafe claim

The planner should not currently be positioned as:

- a faithful reproduction of Pred2Nav or another external predictive-MPC stack,
- a source-harness reproduction,
- family-level evidence against predictive planning in general.

The repository already documents why that stronger claim would be wrong:

- observation contract gap,
- action-parameterization gap,
- scenario-assumption gap,
- evaluation-harness gap.

That boundary is stated explicitly in `configs/benchmarks/planner_quality_audit_v1.yaml`.

## Current Capability And Quality

### What the planner can do today

Within this repository, `prediction_planner` can be described as:

- executable in benchmark campaigns,
- reproducibly configurable,
- checkpointed and registry-resolved,
- integrated into all-planners camera-ready presets,
- integrity-gated by the benchmark runner and evaluation tooling.

That is enough to treat it as a real experimental planner, not a paper-only placeholder.

### What the current evidence says about quality

The strongest benchmark evidence in the repo does not support headline use.

Evidence from `docs/context/benchmark_post_prediction_fix_2026-02-20.md`:

- in the corrected all-planners campaign, `prediction_planner` ran successfully with
  `status=ok`, `135` episodes, and `0` failed jobs.
- the same note reported strong short-term campaign metrics for that corrected execution path.

Evidence from `docs/context/issue_581_paper_evidence_delta.md`:

- on the later canonical SNQI-v3 paper-facing matrix, `prediction_planner` had
  `success=0.0709`, `collisions=0.2128`, `SNQI=-0.1924`.
- the planner is explicitly classified as `active but underperforming`.

Interpretation:

- the integration path works,
- the planner is benchmark-honest,
- but final paper-facing quality is weak on the frozen hard matrix.

Those are compatible findings, not contradictions: execution viability improved, but comparative
performance remained poor.

## Discrepancies And Caveats

### Historical doc drift

There is historical drift across predictive-planner notes:

- older readiness/baseline notes mention `predictive_proxy_selected_v1`,
- current camera-ready config uses `predictive_proxy_selected_v2_full`.

When in doubt, prefer the live config and registry over older transition notes.

### Family interpretation risk

A low score from this planner does **not** show that predictive planning as a literature family is
weak. It only shows that the current in-repo implementation is weak on the current matrix.

### Benchmark-readiness boundary

The planner is benchmark-runnable, but its readiness tier is still `experimental`.
That means it should remain outside baseline-ready or paper-baseline claims unless future promotion
criteria are satisfied.

## Recommended Claim Language

Use language close to this in future docs or paper support notes:

> `prediction_planner` is an in-repo, RGL-inspired predictive local planner integrated into the
> Robot SF benchmark as an experimental adapter-backed challenger. It is benchmark-runnable and
> reproducible within the repository, but current results should be interpreted as local
> implementation evidence rather than a faithful external-family reproduction.

## Follow-Up Recommendations

1. Keep `prediction_planner` labeled `experimental` until a new benchmark run shows materially
   stronger hard-matrix success without relying on interpretation drift.
2. Treat any future paper-facing use as implementation-level evidence only unless a source-harness
   reproduction is demonstrated for a concrete external predictive-planning reference.
3. Refresh the remaining stale predictive-planner docs so all benchmark-facing surfaces agree on the
   canonical model id and config path.
4. If stronger literature comparison is needed, open a separate reproduction issue against one named
   external predictive-planning source and require source-harness parity before Robot SF comparison.
5. Keep proof requirements strict for any future planner upgrade: benchmark or targeted executable
   evidence should accompany changes to predictive checkpoints, scoring, or rollout semantics.

## Proof Attached To This Audit

This issue should not rely on prose alone.
The associated proof for the stable benchmark-contract claims is:

- targeted tests covering readiness tier, metadata semantics, adapter execution mode, and canonical
  config expectations for `prediction_planner`,
- direct inspection of the current canonical config and registry surfaces,
- existing benchmark evidence notes already committed under `docs/context/`.
