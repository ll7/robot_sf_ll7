# Issue #9431 corrected 0.0.7 campaign closeout

Plain-language summary: the corrected S30/H600 campaign completed all 20,160
planned episodes, passed the full-release acceptance gate, and has independently
recoverable evidence. The scheduler's terminal `FAILED` state belongs only to a
post-run filesystem promotion error; it is retained below rather than hidden.

## Exact identity

- software release: [`0.0.7`](https://github.com/ll7/robot_sf_ll7/releases/tag/0.0.7)
  at `76ec07dd9e6c8d2e49fda0e12bcfdfb373c34479`
- benchmark source and data tag:
  `paper-matrix-v2-h600-s30-2026-09-07f7e8d43084de748915e1b1eb8b2a1603357c6e`
  at `07f7e8d43084de748915e1b1eb8b2a1603357c6e`
- campaign ID:
  `issue9431_release_benchmark_data_0_0_7_07f7e8d43084_20260922`
- canonical effective-configuration SHA-256:
  `095331329b06673dc165109c8523579549f769c98542b207a712f6e2bf9ed6ad`
- scenario-matrix SHA-256:
  `03fc83302f707dd1b27c0fa81c4e45e36e8354a4413171d09365926f62bb5c2c`
- versioned scenario-source manifest SHA-256:
  `d9e148e4b544b4c7e2b6ba98e599aef47046d114e0e25645f021946674cb9dc5`
- resolved release-identity SHA-256:
  `9256f2a92578319238b06b0873aa1232482347a47b781a93c05390e93b480a63`
- release-metadata SHA-256:
  `d6603e1bc14dee6966935cf547d7e6121cbf2915f12c88d6be24979c56542151`

The software tag predates the final runtime-admission correction. The benchmark
data tag therefore names the exact source that produced the rows; the two
identities are deliberately not conflated or retagged.

## Campaign acceptance and custody

- matrix: 14 planner arms × 48 scenarios × 30 seeds (111--140), horizon 600
- accepted evidence: 20,160 unique episode identities, 14 successful arms,
  zero missing or unexpected identities, and zero fallback, degraded, failed,
  or unavailable rows
- release result SHA-256:
  `12cf2f039c7afb68f7ce26b7e2d9530defce77b99c935e11b065a3d3f2f40b24`
- publication archive SHA-256:
  `684da7c557c426756f22ddbf5cb3270141ee8ae385669a39d36f324852a6fb2f`
- archive contents: 93 files, 744,051,002 uncompressed bytes; all 93 member
  checksums pass
- preserved artifact:
  `ll7/robot_sf/campaign-issue9431_release_benchmark_data_0_0_7_07f7e8d43084_20260922:v0`
- [W&B preservation run](https://wandb.ai/ll7/robot_sf/runs/f6b8bbvp), state
  `COMMITTED`, with remote manifest readback verified
- preservation manifest digest:
  `sha256:5eb0d68e1483f3d82e75c33c3966a1c597330816f911a0caeaecbde119d4379f`
- preservation receipt digest:
  `sha256:cedf3d1c2761a39c3c5e14c8fbb3e5fa1fbb6379dfee08b4e366e8123d412c47`

Slurm job `15715` ended as `FAILED (2:0)` after the successful benchmark because
the wrapper's atomic publication-custody promotion returned
`renameat2(...): EINVAL`. The benchmark result itself records
`campaign_execution_status=completed`, `campaign_benchmark_success=true`, and
`release_acceptance.status=valid`. A custody-only replay then verified and
preserved the archive. The scheduler failure is therefore a provenance caveat,
not an episode-execution failure and not a silently reclassified success.

## Release diff

The checked-in [machine report](issue_9431_release_diff_0_0_6_to_0_0_7.json)
and [reader report](issue_9431_release_diff_0_0_6_to_0_0_7.md) pair the exact
`31cdfe0361abe2c520117a17f99c1b7a0aba4359..07f7e8d43084de748915e1b1eb8b2a1603357c6e`
source range by planner, scenario, and seed. They contain all 20,160 paired rows,
the 622 changed outcome identities, per-arm outcome counts, and deterministic
seed-block bootstrap intervals. The execution audit admits native, adapter, and
contract-valid mixed execution while rejecting fallback or degraded rows.

The source range includes the versioned social-force goal-approach repair, the
versioned goal-zone success-definition repair, and their runtime-admission
corrections. It is a descriptive release comparison, not a single-change causal
ablation. This explicitly supersedes the original issue's narrower
"nothing else changed" wording in favor of the maintainer-directed corrected
0.0.7 rerun.

The full-campaign rows do not contain simulation-step traces. Consequently,
goal-adjacent timeout labels are reported as `unavailable`, never inferred as
false. That part of the original diff criterion remains uncomputed; this report
does not claim a waiver or treat unavailable labels as negative outcomes. The
separately recorded traces below provide only bounded worked examples, not a
campaign-wide rate.

The committed diff files can be regenerated from the preserved predecessor
archive and successor publication bundle using
[`compare_issue_9431_release.py`](../../scripts/analysis/compare_issue_9431_release.py).
The comparator verifies both archive checksums, the predecessor source identity
inside its resolved manifest, and exact successor row-byte equality with the
pinned bundle. It requires the exact canonical 14-arm × 48-scenario × 30-seed
Cartesian identity (20,160 rows). A focused consistency test binds those frozen
scenario IDs, planner arms, and seeds to the versioned scenario/release manifests
and verifies the scenario-manifest SHA. At run time the comparator checks each
successor row's source and execution metadata. Runtime fallback/degraded markers
use the release-acceptance filter, so declarative config fields are not
misclassified as runtime execution. The generated report binds its campaign ID
to the pinned bundle's campaign manifest and records both matrix digests. After
retrieving the exact archives and extracting the successor bundle's `payload/`
directory, rerun:

```bash
uv run python scripts/analysis/compare_issue_9431_release.py \
  --predecessor-archive <retrieval>/benchmark_0_0_6_s30_h600_20260911_publication_bundle.tar.gz \
  --successor-root <extracted-bundle>/payload \
  --successor-bundle <retrieval>/issue9431_release_benchmark_data_0_0_7_07f7e8d43084_20260922_publication_bundle.tar.gz \
  --predecessor-sha256 61b865fdde65455a39a68221d7c65b0eff315bfa51b4c0bfe34aed3c5d4f3e8e \
  --predecessor-source-sha 31cdfe0361abe2c520117a17f99c1b7a0aba4359 \
  --successor-source-sha 07f7e8d43084de748915e1b1eb8b2a1603357c6e \
  --successor-bundle-sha256 684da7c557c426756f22ddbf5cb3270141ee8ae385669a39d36f324852a6fb2f \
  --output-json docs/analysis/issue_9431_release_diff_0_0_6_to_0_0_7.json \
  --output-markdown docs/analysis/issue_9431_release_diff_0_0_6_to_0_0_7.md
```

## Worked-example traces

Slurm job `15724` regenerated the requested head-on seeds 23/24 and
group-crossing seed 22 from public commit
`2e87f9de0523ef977bce19b40189ea0a74c61e13`, which contains the exact submitted
config. Six compact trace bundles are pinned under
`docs/context/evidence/issue_9431_trace_worked_examples_2026-09/`; every local
`SHA256SUMS` entry verifies.

- trace campaign: `issue9431_trace_18ep_0_0_7_2e87f9de0523_20260922`
- trace input-config SHA-256:
  `9d0ddf20842e1d959a1f7428a4efb998ccab48e33d992b51bceeabb5781d90b5`
- producer `SHA256SUMS` SHA-256:
  `c29f6571eb3a14a76c28d6f4ffb21b9145135b698f8503194326c7d44c46512d`
- preserved artifact:
  `wandb://ll7/robot_sf/campaign-issue9431_trace_18ep_0_0_7_2e87f9de0523_20260922:v0`
- preservation manifest digest:
  `sha256:6827dc8527528e8e81055d4ccb152fa67b3481e2be950d01dbf5b1856a8d68e5`
- full cold-restore report SHA-256:
  `f9529356b7018aa1dc2e5e5e6ff6e6aceca2232be4043446b9414391b0bca7ec`

All 18 source rows contain nonempty simulation/action/force traces. These
baseline planners do not expose specialized planner-internal decision steps;
the empty internal-decision arrays are retained and not reinterpreted. The six
tracked exports are worked examples only and do not establish a causal
mechanism, planner ranking, or population effect. The earlier job 15716 trace
package remains historical custody and is superseded for public provenance.
Job 15724's external admission, submission-intent, terminal, watcher, run
metadata, and execution-context receipts bind its run identity, but the CPU
wrapper did not emit `startup.json` or `admission.json` inside the result tree.
This is a recorded launch-contract exception: it prevents claiming the runtime
startup guard was attested, so the slice remains diagnostic-only. It does not
invalidate the pinned trace bytes or establish release-level evidence.

## Publication boundary

The GitHub software release and both W&B evidence packages are published. The
preassigned Zenodo identifiers were never published and return no citable DOI;
this closeout intentionally makes no Zenodo DOI claim. Issue #9431 remains open
until its goal-adjacent timeout comparison is either made computable from valid
evidence or explicitly waived by its owner; no such waiver is asserted here.
