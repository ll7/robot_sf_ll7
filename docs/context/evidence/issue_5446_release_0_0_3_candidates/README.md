<!-- AI-GENERATED (robot_sf#5446) - NEEDS-REVIEW -->
# Issue #5446 mining run over release v0.0.3 evidence

First real-data execution of the #5446 miner, the #5615 candidate-to-trace
resolver, and the #5616 campaign atlas builder, run against a pinned,
read-only publication release bundle. No simulation or campaign was launched
by this work; only mining/resolution/rendering tooling ran over existing
episode evidence.

## Source evidence (read-only, never modified)

- Bundle: `paper_experiment_matrix_v2_h600_s30_extended_release_v0_0_3_final_publication_bundle/payload`
  (outside this repo; local path recorded in the reproduction commands below)
- `benchmark_release.release_id`: `paper_experiment_matrix_v2_h600_s30_v0_0_3`, `release_tag`: `0.0.3`
- Row-level pinned execution commit (`result_provenance.repo_commit`, identical on
  every one of the 20,160 episode rows across all 14 arms): `a307ef276d701f8d14dead1aa0513f44ee97c0b0`
- `manifest.json.git_hash` (the campaign-report-rebuild commit, distinct from
  the row-level execution commit above): `e2ac534c9d6bb750346b1e0724638c91306e410a`
- `manifest.json.benchmark_release.canonical_campaign_config_sha256`:
  `143ab63a235f40326c93c93044fba95e808388751f04d8ca979b89d1142ca465`
- `manifest.json.benchmark_release.scenario_matrix_sha256`:
  `d9e148e4b544b4c7e2b6ba98e599aef47046d114e0e25645f021946674cb9dc5`
- 14 arms x 30 seeds (`paper_eval_s30`, seeds 111-140) x 48 scenario cells =
  20,160 episode rows total; each arm's `episodes.jsonl` verified duplicate-free
  (unique `(scenario_id, seed)` per arm)
- This bundle carries **no per-step trace exports**
  (`simulation_trace_export.v1`) -- only `episodes.jsonl` summary rows plus
  per-arm `episodes.jsonl.provenance.json` sidecars. That absence is expected
  and is the reason the #5615 resolver step below reports every candidate as
  unresolved.

## Worktree / tool identity

- Branch: `issue-5446-mining-run`, based on `origin/main` at `f2056abff4c155803f6d0d02d74ae3c26490de19`
- Miner: `robot_sf/benchmark/seed_flip_mining.py` /
  `scripts/analysis/mine_seed_flips_and_inversions_issue_5446.py`
  (`seed_flip_inversion_candidates.v1`, merged for #5446)
- Resolver: `robot_sf/benchmark/candidate_trace_resolution.py` /
  `scripts/analysis/resolve_candidate_traces_issue_5615.py`
  (`candidate_trace_resolution.v1`, PR #5624, merge `b9ea59c82335e30dab65247f006fd7356c012a6f`)
- Atlas: `robot_sf/benchmark/campaign_atlas.py` /
  `scripts/analysis/build_campaign_atlas_issue_5616.py`
  (`campaign_atlas.v1`, PR #5637, merge `b9b7138aba730171c104a1c98c83c29faa986d32`)
- New adapter (thin reshaping only, no mining/resolution/rendering semantics
  changed): `scripts/analysis/adapt_release_bundle_issue_5446_mining_run.py`
- New joiner for the #5447 trace re-export ask:
  `scripts/analysis/build_issue_5446_trace_reexport_list.py`

## 1. Mine (#5446)

```bash
uv run python scripts/analysis/adapt_release_bundle_issue_5446_mining_run.py \
  --bundle-payload <bundle>/payload \
  --mining-rows-out /tmp/mining_rows.jsonl \
  --atlas-inventory-out /tmp/atlas_inventory.jsonl \
  --summary-json /tmp/adapter_summary.json

uv run python scripts/analysis/mine_seed_flips_and_inversions_issue_5446.py \
  --input /tmp/mining_rows.jsonl \
  --json seed_flip_inversion_candidates.v1.json \
  --md seed_flip_inversion_candidates.v1.md --top 50
```

### Adapter role

The bundle's per-row provenance the miner needs is real but nested under
different paths than the miner's flat row contract expects:
`result_provenance.repo_commit` -> top-level `repo_commit`, and
`algorithm_metadata.planner_kinematics.execution_mode` -> top-level
`execution_mode`. The adapter only projects these (plus a `scenario_family`
for the atlas, from the real `scenario_params.metadata.archetype` field). No
mining threshold, gate, or eligibility rule was touched; see
`scripts/analysis/adapt_release_bundle_issue_5446_mining_run.py` module
docstring.

### Result: eligibility gates applied to real data

| stage | rows |
| --- | --- |
| input rows (20,160 = 14 arms x 30 seeds x 48 scenarios) | 20,160 |
| eligible (native execution) | 2,880 |
| excluded | 17,280 |

Exclusion reasons (all fail-closed, no threshold relaxation):

- `non_native_execution:adapter`: 15,840 rows
- `non_native_execution:mixed`: 1,440 rows

**Only 2 of the 14 campaign arms have `execution_mode=native`: `goal` and
`ppo`.** All other planners run through a kinematics adapter (or a mixed
native/adapted mix) to convert their native command space onto the
differential-drive robot, so the native-execution gate (per the #5446 issue:
"native execution, never fallback/degraded/adapter") excludes them from
mining entirely, regardless of outcome quality. This yields 96 eligible
`(scenario, planner)` cells (48 scenarios x 2 native planners).

### Candidate counts by archetype

| archetype | available | n candidates | notes |
| --- | --- | --- | --- |
| `seed_flip` | yes | 45 | knife-edge (scenario, planner) cells across the 30 discovery seeds |
| `planner_upset` | yes | 5 | held-out (leave-one-scenario-out) `goal` vs `ppo` upsets |
| `causal_divergence` | **no** | 0 | no `temporal_boundary_margin` signal in this bundle's `metrics` block |
| `disagreement_recovery` | yes (signal only) | -- | cross-planner disagreement entropy attached per scenario; max 1.0 bit |

All 4 external sibling signals (`oracle_regret` #5302, `transfer` #5303,
`quality_diversity` #5308, `multiplicity` #5351) are reported `unavailable`
(no `--external` table supplied) -- honestly, not fabricated.

Pareto-selected candidates (non-dominated frontier, the two "worth
confirming" cases): 2 of 50.

1. `seed_flip::classic_doorway_medium::ppo` -- 15/15 split across 30 seeds,
   entropy = 1.000 bits, Wilson CI [0.33, 0.67].
2. `planner_upset::classic_realworld_double_bottleneck_high::goal>ppo` --
   `goal` (held-out strength 0.40) beats `ppo` (held-out strength 0.73) on
   this one scenario; held-out skill gap 0.330, outcome gap 0.233.

Determinism: re-ran the miner on the same `mining_rows.jsonl`; the manifest
JSON was byte-identical.

### Real-data finding: `algo` label collisions in the release bundle

The miner's default planner grouping (and this adapter's atlas `planner`
field) both key off the row's top-level `algo` field. In this bundle, 4 of
the 14 campaign arm directories collapse onto a single `algo` value:
`hybrid_rule_v3_fast_progress_static_escape`,
`hybrid_rule_v3_fast_progress_static_escape_continuous`,
`scenario_adaptive_hybrid_orca_v1`, and
`scenario_adaptive_hybrid_orca_v2_collision_guard` all report
`algo="hybrid_rule_local_planner"`. 14 arms -> 11 distinct `algo` labels.
This does not affect the mined candidates above (that planner's rows are
`execution_mode=adapter`/`mixed`, already excluded by the native gate), but it
does affect the campaign atlas: any atlas cell for `hybrid_rule_local_planner`
pools episodes from up to 4 architecturally distinct arm configurations as if
they were seed replicates of one planner. Reported here as observed bundle
data, not corrected -- correcting planner identity would be a semantic
decision out of scope for this adapter.

## 2. Resolve (#5615)

```bash
# without a campaign store
uv run python scripts/analysis/resolve_candidate_traces_issue_5615.py \
  --candidates seed_flip_inversion_candidates.v1.json \
  --json candidate_trace_resolution.v1.no_store.json --validate

# with a real (trace-less) campaign-result-store.v1 built from the same bundle
uv run python scripts/analysis/adapt_release_bundle_issue_5446_mining_run.py \
  --bundle-payload <bundle>/payload --store-rows-out /tmp/campaign_store
uv run python scripts/analysis/resolve_candidate_traces_issue_5615.py \
  --candidates seed_flip_inversion_candidates.v1.json \
  --campaign-store /tmp/campaign_store \
  --trace-roots <bundle>/payload/runs \
  --json candidate_trace_resolution.v1.with_store.json --validate
```

`campaign-result-store.v1` requires `pyarrow`/`duckdb`/`fastparquet`; installed
via `uv sync --extra analytics` in this worktree only (not required for the
miner/atlas steps).

### Result: fail-closed, as the issue's stop rule anticipates

| run | resolved | trace-missing | schema-mismatch | provenance-incomplete |
| --- | --- | --- | --- | --- |
| no `--campaign-store` | 0 | 0 | 0 | 50 / 50 |
| with a real `campaign-result-store.v1` + `--trace-roots` at the bundle | 0 | 0 | 0 | 50 / 50 |

Reason codes: `no_campaign_store_provided` (no-store run) /
`campaign_row_not_found` (with-store run) for all 50 candidates in both runs.

### Contract mismatch found on real data (#5446 <-> #5615)

`resolve_candidate_to_episode` (`robot_sf/benchmark/candidate_trace_resolution.py`)
looks up a campaign episode by `candidate.get("seed")` /
`candidate.get("episode_id")`. The #5446 `seed_flip_inversion_candidates.v1`
schema never populates those two fields on a candidate: a candidate is a
**`(scenario, planner)` cell aggregate**, and its per-seed outcomes live only
inside `reproducibility.raw_seed_outcomes` (`seed_flip`) or
`upset_outcome.raw_paired_outcomes` (`planner_upset`) -- both keyed dicts, not
scalar fields the resolver reads.

Consequently, on real mined candidates the resolver's episode-lookup key is
always `scenario_id=<real>|planner=<real>|seed=NA|episode_id=NA`. Without a
store this is `provenance-incomplete:no_campaign_store_provided`; with a real
store it becomes `provenance-incomplete:campaign_row_not_found`, because no
store row has `seed=NA`. **The resolver never reaches trace search for any
real #5446 candidate** -- not because traces are absent (true separately, see
above), but because the granularity of the two schemas does not line up. This
is a genuine composition gap between the merged #5446 and #5615 contracts,
observable only once real candidates (rather than the per-seed synthetic
fixtures in `tests/benchmark/test_candidate_trace_resolution_issue_5615.py`)
are run through the resolver. It is not a bug in either tool individually:
each honors its own issue's fail-closed contract.

## 3. Trace re-export list (actionable input for #5447)

Since the resolver cannot bridge cell-level candidates to per-seed episodes,
`scripts/analysis/build_issue_5446_trace_reexport_list.py` performs the one
join #5447 needs directly: for the 2 Pareto-**selected** candidates, expand
their embedded raw per-seed outcomes into `(scenario_id, planner, seed,
episode_id)` tuples, looking the real `episode_id` up in the same
`mining_rows.jsonl` the miner ran against. Un-selected/triage-only candidates
are not expanded (see the miner's own claim boundary: only selected
candidates are "worth confirming").

```bash
uv run python scripts/analysis/build_issue_5446_trace_reexport_list.py \
  --candidates seed_flip_inversion_candidates.v1.json \
  --mining-rows /tmp/mining_rows.jsonl \
  --json issue_5446_trace_reexport_list.v1.json
```

- **90 tuples**, all 90 with a real `episode_id` found (0 `not_found`).
  - 30 from `seed_flip::classic_doorway_medium::ppo` (planner `ppo`)
  - 30 + 30 from `planner_upset::classic_realworld_double_bottleneck_high::goal>ppo`
    (planners `goal` and `ppo`)
- Sample tuples:
  1. `classic_doorway_medium / ppo / seed 111 / classic_doorway_medium--111--fc6264c7e2bbbaf3`
  2. `classic_realworld_double_bottleneck_high / goal / seed 111 / classic_realworld_double_bottleneck_high--111--fb588846ea29cb6f`
  3. `classic_realworld_double_bottleneck_high / ppo / seed 111 / classic_realworld_double_bottleneck_high--111--544918024132112f`

This list, not the resolver output, is the actionable handoff to #5447: it
names exactly which pinned-SHA episodes need a targeted trace re-export
before case-capsule assembly can proceed on real data.

## 4. Atlas (#5616, stretch -- rendered)

```bash
uv run python scripts/analysis/build_campaign_atlas_issue_5616.py \
  --inventory /tmp/atlas_inventory.jsonl \
  --out-dir campaign_atlas \
  --campaign-id paper_experiment_matrix_v2_h600_s30_extended_release_v0_0_3_final \
  --commit a307ef276d701f8d14dead1aa0513f44ee97c0b0
```

The atlas inventory adapter maps `algo` -> `planner`,
`scenario_params.metadata.archetype` -> `scenario_family` (real per-scenario
archetype field, e.g. `bottleneck`, `doorway`, `crossing`), and folds the
row's literal `outcome.collision_event` / `outcome.timeout_event` /
`metrics.success` booleans into one outcome label (collision > timeout >
success > other; no new outcome semantics invented). No
`trajectory`/`event_anchors`/`predicate_timeline` are emitted -- the bundle
has none.

**Rendered**: the full population-level campaign atlas over all 385 eligible
`(scenario_family, planner)` cells (35 scenario families x 11 distinct `algo`
labels; see the label-collision finding above), 0 ineligible cells, with
Wilson-interval outcome counts per cell -- `campaign_atlas/campaign_atlas.svg.gz`
(1.2 MB uncompressed, gzipped to fit the repo's 1024 KB added-file limit; see
"Files" below) plus `campaign_atlas_summary.json` and
`campaign_atlas_catalog.yaml` (sha256-hashed artifact catalog; the catalog's
recorded SVG hash is over the uncompressed bytes). Determinism verified via
`--check-determinism` (PASS).

**Precise boundary** (as the #5616 issue anticipates): this atlas is
population-only. The **ensemble context view genuinely requires resolved
per-step traces** the bundle does not have. Requesting it
(`--ensemble-anchor near_miss_start`) does not crash or fabricate a plot --
every ensemble figure renders an explicit `Ensemble context view unavailable`
placeholder with the precise per-episode reason (`missing anchor
'near_miss_start', trajectory needs at least two points`). One example is
kept at `campaign_atlas/ensemble_boundary_demo/ensemble__doorway__goal.svg`
(scoped to the `doorway`/`goal` cell only; the full 385-cell ensemble sweep
was not committed here as it is not a compact artifact and every cell
produces the same "unavailable" boundary).

## Files

- `adapter_summary.json` / `adapter_summary_campaign_store.json`: adapter run
  summaries (per-arm row counts, execution-mode tally).
- `seed_flip_inversion_candidates.v1.json.gz` / `.md`: the real mined
  candidate manifest (issue #5446 primary deliverable). Gzipped (3.0 MB ->
  57 KB) because this repo's pre-commit `check-added-large-files` hook caps
  added files at 1024 KB; `gunzip -k` to read it directly.
- `candidate_trace_resolution.v1.no_store.json` / `.with_store.json`: the two
  real resolver runs (issue #5615).
- `issue_5446_trace_reexport_list.v1.json`: the actionable (scenario, planner,
  seed, episode-id) re-export ask for #5447.
- `campaign_atlas/`: the rendered population atlas (issue #5616,
  `campaign_atlas.svg.gz` for the same 1024 KB reason) plus one
  ensemble-boundary example.
- `SHA256SUMS`: checksums for every file in this bundle (over the committed,
  gzipped bytes for the two compressed files).

## Validation

```bash
gunzip -k -c docs/context/evidence/issue_5446_release_0_0_3_candidates/seed_flip_inversion_candidates.v1.json.gz | python3 -m json.tool >/dev/null
python3 -m json.tool docs/context/evidence/issue_5446_release_0_0_3_candidates/candidate_trace_resolution.v1.no_store.json >/dev/null
python3 -m json.tool docs/context/evidence/issue_5446_release_0_0_3_candidates/issue_5446_trace_reexport_list.v1.json >/dev/null
(cd docs/context/evidence/issue_5446_release_0_0_3_candidates && sha256sum -c SHA256SUMS)
uv run pytest -q tests/benchmark/test_scenario_seed_flip_planner_inversion_upset_issue_5446.py \
  tests/benchmark/test_candidate_trace_resolution_issue_5615.py \
  tests/benchmark/test_campaign_atlas_issue_5616.py \
  tests/analysis/test_adapt_release_bundle_issue_5446_mining_run.py \
  tests/analysis/test_build_issue_5446_trace_reexport_list.py
```

## Determinism note

The miner, resolver, and atlas builder were each re-run at least once on
identical input in this session; every re-run produced a byte-identical (or,
for the atlas, `--check-determinism`-verified hash-stable) output. The
`mining_rows.jsonl` / `atlas_inventory.jsonl` intermediates are not committed
here (112 MB / 6.6 MB respectively, regenerable in seconds from the pinned,
read-only bundle via the adapter command above) to keep this evidence bundle
compact; the committed manifests are the durable record.
