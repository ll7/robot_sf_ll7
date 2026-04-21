# Issue-791 held-out OOD evaluation suite — design scaffold (DEPRIORITIZED)

Date: 2026-04-20
Related: `docs/context/issue_791_wave6_results_and_benchmark_orca_block.md`

> **Status (2026-04-20, maintainer decision): option (a) — narrow-claim path.**
> No OOD suite will be built. The paper framing is "a strong policy evaluated on
> a broad benchmark of social-navigation scenarios", **not** a generalization /
> transfer claim. Seed-variance + benchmark-set numbers are the primary evidence.
> This document is kept as a design record in case the framing changes.
>
> Consequence for text and reporting:
> - Do **not** use the words "generalize", "transfer", "unseen environment",
>   or "novel scenario" in manuscript/issue text.
> - Do **not** contrast 0.586 (old plateau) vs 0.929 (leader) as a "generalization
>   fix". That framing implies an OOD claim. Use "coverage alignment on the
>   benchmark set" or "benchmark-set performance improvement" instead.
> - The strength of the claim rests on (i) breadth of the benchmark scenario
>   matrix, (ii) seed-variance tightness across 10M replicas, (iii) clean
>   camera-ready bootstrap CIs.

The Wave-5/6 PPO leader (job 11724, success=0.929) was trained **and** evaluated on
`configs/scenarios/sets/ppo_full_maintained_eval_v1.yaml`. That number is therefore an
in-distribution score by construction. To support any generalization claim for the
manuscript, we need a scenario family the policy has never seen.

## Training-set inventory (what must be excluded)

`ppo_full_maintained_eval_v1.yaml` expands to:

- `configs/scenarios/classic_interactions.yaml` — 11 classic interaction archetypes
  (bottleneck, cross_trap, crossing, doorway, group_crossing, head_on_corridor,
  merging, overtaking, realworld_bottleneck, t_intersection, urban_crossing).
- `configs/scenarios/francis2023.yaml` — 25 Francis-2023 archetypes
  (`configs/scenarios/single/francis2023_*.yaml`).
- `configs/scenarios/sets/atomic_navigation_minimal_full_v1.yaml` — 5 atomic archetype
  sets (`configs/scenarios/archetypes/issue_596_*.yaml`).

Anything in those three groups is in-distribution.

## Candidate OOD sources already in the repo

| Set | OOD suitability | Notes |
|---|---|---|
| `atomic_navigation_validation_fixtures_v1.yaml` | high | Only includes `issue_596_goal_inside_obstacle_invalid` — designed as a validation fixture, genuinely held out. Too small to carry an OOD claim alone (single scenario). |
| `safety_barrier_static_slice_v1.yaml` | medium | Static barrier regime; pedestrian dynamics differ from training distribution. Candidate for corridor OOD slice. |
| `issue_805_teb_topology_slice.yaml` | medium | Topology regime for TEB planner evaluation; novel map coverage. |
| `verified_simple_subset_v1.yaml` | low | Subset overlap with training probable; needs scenario-level diff before use. |
| `paper_seed_variability_pilot_v1.yaml` | low | Subset of classic archetypes — in-distribution. |

Repository single-scenario sources not referenced by any training set:

- `configs/scenarios/single/planner_sanity_simple.yaml` — hand-built, unused by PPO training.
- `configs/scenarios/single/issue_596_goal_inside_obstacle_invalid.yaml` — single edge-case fixture.

## Recommended construction (two-tier)

**Tier A — density/flow regime shift on trained maps (easier, fast to build).**
Keep the trained-map coverage, vary density, flow direction, and ped group structure
outside the training-time sampling distribution. Example knobs:

- `ped_density` = {0.08, 0.12} (training uses ~0.02-0.06).
- `flow` = {mono, bi, converging} mixed within one episode.
- `max_peds_per_group` = {1, 6} (training uses 3).

Tier A is a partial OOD claim: robot sees same maps, different population dynamics.

**Tier B — held-out maps (stronger claim).**
Pick SVG maps under `maps/svg_maps/` that are **not** referenced by
`classic_interactions.yaml` or `francis2023.yaml`. Build minimal scenarios on those
maps with sensible waypoints. Scripted candidate auditor:

```bash
# Find SVG maps NOT referenced by the training scenario set
uv run python - <<'PY'
import yaml
from pathlib import Path

training_refs = set()
for p in [
    "configs/scenarios/classic_interactions.yaml",
    "configs/scenarios/francis2023.yaml",
    "configs/scenarios/sets/atomic_navigation_minimal_full_v1.yaml",
]:
    text = Path(p).read_text()
    for line in text.splitlines():
        if ".svg" in line:
            training_refs.add(Path(line.split("/")[-1].strip().strip('\"')).name)

all_maps = {p.name for p in Path("maps/svg_maps").glob("*.svg")}
print("held-out candidates:")
for m in sorted(all_maps - training_refs):
    print(" ", m)
PY
```

## Proposed benchmark integration

1. Author scenario set:
   `configs/scenarios/sets/ppo_full_maintained_ood_v1.yaml`
   combining Tier-A density/flow variants + Tier-B held-out maps.
2. Author benchmark config:
   `configs/benchmarks/paper_experiment_matrix_v1_issue_791_ood_compare.yaml`
   cloning the eval-aligned compare config and pointing `scenario_matrix` to
   `ppo_full_maintained_ood_v1.yaml`.
3. Rerun on l40s via `SLURM/Auxme/issue_791_benchmark.sl` with label
   `issue791-ood-holdout-leader-11724`.

## Validation gates for the OOD suite

- At least 20 distinct scenarios (Tier A + Tier B combined) to keep bootstrap CIs
  meaningful at 300 samples / 95%.
- No overlap with training: scripted check comparing scenario `map_file` + archetype
  label to the training set before the run starts.
- Report the IO split (which scenarios are Tier A vs Tier B) in the publication
  appendix so readers can judge the strength of the OOD claim.

## Why this is not yet submitted

Authoring the OOD set is a design choice, not a mechanical config clone — the right
density/flow knobs, map selection, and per-scenario waypoints need human judgment.
This note scaffolds the construction so it can be executed as a follow-up task. Until
the OOD suite exists and the leader evaluates on it, any paper text must say 0.929 is
an **in-distribution benchmark-set** score, not an OOD generalization claim.

## Discussion — is seed randomization enough? (2026-04-20)

**Question raised.** Training uses random pedestrian seeds and the simulator evolves
trajectories differently for each seed; with 10M × 22 envs the policy sees millions
of distinct rollouts. Can we call this OOD and skip the held-out construction?

**Answer — no. Seed variance is not a distribution shift.** Stochastic rollouts on a
fixed scenario manifest sample more realizations of the **same** joint distribution
over (map × ped-density × spawn/goal family × behavior regime). A policy trained to
convergence on that manifest has seen the stationary distribution of the underlying
MDP. Reporting "unseen seed" eval is a test of *in-distribution generalization* —
expected for any converged policy — not of *out-of-distribution generalization*,
which is what the social-navigation literature (SocNav, Francis-2023, crowd-nav)
means when it says "generalize".

The clearest evidence in our own data: before eval alignment the policy hit 0.586 on
the broader eval; after alignment the same recipe hits 0.929. The 0.343 gap is
exactly the IID/OOD gap. Adding more training steps does not close that gap because
more steps improve optimization quality on the training distribution but do not widen
the distribution itself. "More seeds" and "more steps" both live inside the same
manifest; they do not cross the distribution boundary.

**When seed randomization IS the right answer.**

- The claim is narrow: "we improve PPO on the social-navigation benchmark defined by
  scenario set X." Then IID seed-variance eval is the right evidence.
- The benchmark itself is treated as the reference task (Alyassi-style comparability
  mapping is in this camp). Camera-ready reporting on that matrix does not require
  OOD.
- The paper explicitly avoids the word "generalize" and instead says "benchmark-set
  performance" or "on-distribution success."

**When seed randomization is NOT enough.**

- Any claim containing the word "generalize", "transfer", "unseen", or "novel
  environment". Reviewers in this literature read those words as structural shift.
- Any comparison that implies the policy would work in a deployment setting that
  differs from the training distribution (different map topologies, different
  density, different ped-behavior models).
- Any statement that contrasts the 0.929 leader with the 0.586 pre-alignment number
  as evidence of policy progress — because *that gap itself is an OOD gap*, so the
  only honest way to argue "we fixed it" is to show OOD performance on a held-out
  split.

**Pragmatic recommendation for issue-791.**

1. **Primary evidence (no OOD suite needed).** Camera-ready benchmark rerun on
   `paper_experiment_matrix_v1_issue_791_eval_aligned_compare.yaml` plus the two 10M
   seed replicas (jobs 11872, 11873). Report these as benchmark-set + seed-variance
   numbers. This is sufficient **if** the paper frames the claim narrowly.
2. **Secondary evidence (recommended, not blocking).** A small Tier-B held-out-maps
   slice (3–5 scenarios on unused SVG maps) run through the same benchmark pipeline.
   Cheap to author, cheap to run (one benchmark submission), and disarms the OOD
   reviewer objection. Keep it out of the headline number; report it in the appendix
   as "held-out maps" with explicit caveat that Tier-A density/flow shift was not
   probed.
3. **Skip.** Full Tier-A density/flow suite + full comparability matrix on an OOD
   split. That is a week of work and would widen the claim; current scope does not
   justify it.

**What NOT to do:** do not claim "seed randomization = OOD" in paper text. Even with
several 10M seed runs, this framing will not survive peer review on a social
navigation manuscript. If the goal is to avoid OOD construction entirely, the safer
path is to narrow the written claim ("we improve PPO on scenario set X") and drop
any generalization language.

**Decision needed from maintainer.** Pick one:

- (a) Narrow-claim path: ship the paper on primary evidence only, no OOD suite.
- (b) Belt-and-suspenders path: also author the Tier-B held-out-maps slice and
  include it in the appendix (~half-day work + one benchmark submission).
- (c) Full path: build Tier-A + Tier-B as originally scaffolded.

Default recommendation is **(b)** — it is the smallest addition that defends against
the most likely reviewer objection, and it leaves the "full generalization study"
as explicit future work rather than an unexamined gap.
