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

The full-campaign rows do not contain simulation-step traces. Consequently,
goal-adjacent timeout labels are reported as `unavailable`, never inferred as
false. The separately recorded traces below provide only bounded mechanism
examples, not a campaign-wide rate.

## Worked-example traces

Slurm job `15716` regenerated the requested head-on seeds 23/24 and
group-crossing seed 22 from the benchmark source. Six compact trace bundles are
pinned under
`docs/context/evidence/issue_9431_trace_worked_examples_2026-09/`; every local
`SHA256SUMS` entry verifies.

- trace campaign: `issue9431_worked_traces_0_0_7_20260922`
- trace input-config SHA-256:
  `dcaf87a46182e1cb64283a7b2442b02ce5fec86d33e9d72df2bcc6a04ee19806`
- preserved package:
  `ll7/robot_sf/campaign-issue9431_trace_package_0_0_7_20260922_v2:v0`
- [W&B trace-preservation run](https://wandb.ai/ll7/robot_sf/runs/82w6s6zz)
- package manifest SHA-256:
  `6c7bae4b052ecadd3eda989eaebd9bb7346c31575212224f1e71a4933905f055`
- package `SHA256SUMS` SHA-256:
  `6469a5f37102fc776704d88aad1b7cd5ab1ec52fcb083f62e37eae76df1808b3`
- preservation receipt SHA-256:
  `c4baca06392978c505d26d298b8ba15d6b9d8d7ce2b30ce5660d6df79e86a382`

These traces are worked examples only. They do not establish a causal mechanism,
planner ranking, or population effect.

## Publication boundary

The GitHub software release and both W&B evidence packages are published. The
preassigned Zenodo identifiers were never published and return no citable DOI;
this closeout intentionally makes no Zenodo DOI claim.
