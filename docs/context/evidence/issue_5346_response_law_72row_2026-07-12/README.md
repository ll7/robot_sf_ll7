<!-- AI-GENERATED (robot_sf#5378, 2026-07-12) - NEEDS-REVIEW -->
# Issue #5346 — response-law 72-row mean-matched harness campaign (job 13379)

Date: 2026-07-12

Related issue (this registration): <https://github.com/ll7/robot_sf_ll7/issues/5378>
Executed campaign issue: <https://github.com/ll7/robot_sf_ll7/issues/5346>
Upstream harness issue: <https://github.com/ll7/robot_sf_ll7/issues/3574>

This packet registers the artifact bundle produced by the executed #5346 response-law campaign as
durable evidence. It performs **no interpretation upgrade, no claim promotion, and no re-run**; it
records what the campaign already produced and faithfully carries the campaign's own caveats.

## Claim boundary and status

- **Evidence status:** `diagnostic-only`. The bundled `analysis.md` states verbatim that the status
  is `diagnostic-only` "until a separately reviewed campaign provides attributable paired records
  and appropriate confidence bounds."
- **Claim boundary (from the sources):**
  - `analysis.md`: "This report does not run a benchmark and does not establish a
    heterogeneous-population effect, planner rank-stability result, realism claim, or sim-to-real
    claim." It also states: "Any ranking text below describes only the supplied records; it is not
    an empirical conclusion by itself."
  - `manifest.json`: `claim_boundary = "harness_only_no_ablation_result"`.
  - `integration_readiness.json`: `claim_boundary = "integration_readiness_only_no_ablation_result"`.
- This registration does not change any planner roster status, benchmark metric semantics, the
  frozen benchmark suite, or any downstream claim. It is a durable receipt of a diagnostic-only
  harness execution.

## Honest caveats (do not read this as an established result)

These caveats are sourced verbatim-faithfully from the bundled files and must not be softened.

1. **The manifest declares the campaign is still blocked.** `manifest.json` records
   `status = "blocked_pending_control_trace"` with six blockers: for both the `heterogeneous` and
   `mean_matched_homogeneous` arms of `issue_3574_classic_crossing_density_002`, the manifest reports
   that `algorithm_metadata.pedestrian_control_trace` is missing, along with the per-step
   `clearance_m` and `near_field_exposure_s` fields it expects under that trace. The harness stage
   therefore did not observe a real pedestrian control trace; it is a harness/integration receipt,
   not a completed ablation.
2. **Integration-readiness is narrow.** `integration_readiness.json` reports `ready = true` with
   `72/72` rows ready and empty per-row blockers, but its own `claim_boundary` is
   `"integration_readiness_only_no_ablation_result"`. "Ready" here means only that the supplied
   paired records passed the fail-closed integration schema check — it is **not** a statement that
   any ablation effect exists. Note the tension with caveat (1): the readiness check passed on the
   supplied records while the manifest still records the campaign as `blocked_pending_control_trace`.
3. **Single scenario only.** Every one of the 72 rows is the single scenario
   `issue_3574_classic_crossing_density_002` (3 planners × 2 population arms × 3 seeds, over the
   declared response-law fractions). The results do not span the scenario suite.
4. **CPU-level smoke slice, high uncertainty.** `analysis.md` states: "Since this is a CPU-level
   smoke validation run on a small slice, rank sensitivity estimates carry higher uncertainty" and
   "In full runs, a larger sample of seeds and scenarios is required to establish statistical
   significance."
5. **Rank reversals present.** `analysis.md` flags reversals between the heterogeneous and
   mean-matched-homogeneous arm rankings (e.g. heterogeneous `['social_force', 'goal', 'orca']`
   vs homogeneous `['social_force', 'orca', 'goal']`). Pairwise bootstrap probabilities are reported
   but are not significance claims.

## Bundle contents

Retrieved from cluster host `imech192` under the campaign worktree output path:
`/home/luttkule/git/robot_sf_ll7.worktrees/slurm-issue5346-response-law-72row-20260712/output/issue_3574_mean_matched_harness/` <!-- allow-abs-path: cluster provenance path, non-portable by design -->

| File | Size (bytes) | Description |
| --- | --- | --- |
| `manifest.json` | 647435 | Harness manifest (schema `mean_matched_heterogeneity_harness.v1`); declares status, blockers, config path, expected rows. |
| `integration_readiness.json` | 17475 | Fail-closed integration-readiness receipt (72/72 rows), schema `mean_matched_episode_readiness.v1`. |
| `analysis.md` | 13335 | Rendered ablation report with the diagnostic-only claim boundary and caveats. |
| `summary.json` | 288073 | Combined summary (ablation reports, integration readiness, per-archetype metric reports, rank sensitivity). |
| `rank_sensitivity.json` | 6502 | Paired-bootstrap rank probabilities per arm. |
| `ablation_results.csv` | 6668 | 72 rows: per scenario/seed/planner/arm mean and CVaR clearance. |
| `durable_evidence/ablation_results.csv` | 6668 | Campaign's own durable-subset mirror (byte-identical to the top-level file). |
| `durable_evidence/analysis.md` | 13335 | Mirror (byte-identical). |
| `durable_evidence/rank_sensitivity.json` | 6502 | Mirror (byte-identical). |
| `durable_evidence/summary.json` | 288073 | Mirror (byte-identical). |

`durable_evidence/` is the campaign's self-declared durable subset; its four files are byte-identical
copies of the corresponding top-level report files (verified by SHA-256). Both are kept as produced.

### Excluded (oversized raw records)

| File | Size | SHA-256 | Cluster location |
| --- | --- | --- | --- |
| `episode_records.jsonl` | 91762829 bytes (~87.5 MiB) | `33e468642cde3c75b05e2c941419879af3984c8dc2df21f30df9dfa5f9945c0b` | `imech192:/home/luttkule/git/robot_sf_ll7.worktrees/slurm-issue5346-response-law-72row-20260712/output/issue_3574_mean_matched_harness/episode_records.jsonl` | <!-- allow-abs-path: cluster provenance path for excluded raw records -->

<!-- allow-abs-path markers above annotate the two intentional cluster provenance paths; they are non-portable by design and retained for retrievability. -->

The bulk per-episode records file is excluded because it exceeds the 20 MB durable-evidence budget.
All derived reports above are computed from it. Its SHA-256 and cluster path are recorded here so the
raw records remain retrievable and verifiable.

## Provenance

- **Slurm job:** 13379 on `imech192` (a30-cpu partition), 2026-07-12 — COMPLETED, exit 0, ~3m32s,
  72/72 rows.
- **Producing commit:** `93c396797da205b46c1546a2ee2cf1bde3dfd37a` (ancestor of `origin/main`).
- **Producing config:** `configs/benchmarks/issue_3574_mean_matched_harness_smoke.yaml`
  (present at the producing commit; blob SHA-256 `44edf74e8803aa74b6799326a1b77a83bdf43923a0718da49a0a3e8674426372`).
- **Integrity:** `SHA256SUMS` in this directory covers every included file.

## Verification

```bash
# from the repository root, inside this evidence directory
cd docs/context/evidence/issue_5346_response_law_72row_2026-07-12
shasum -a 256 -c SHA256SUMS
```
