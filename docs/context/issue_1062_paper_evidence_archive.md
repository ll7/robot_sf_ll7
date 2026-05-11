# Issue 1062 Paper Evidence Archive Pointer

Date: 2026-05-09

Related issue: `ll7/robot_sf_ll7#1062`

## Goal

Make the paper-facing benchmark evidence recoverable from durable release pointers instead of from a
single machine's local `output/` tree.

## Current Source Of Truth

The current durable benchmark archive is GitHub release `0.0.2`:

- Release: `https://github.com/ll7/robot_sf_ll7/releases/tag/0.0.2`
- DOI: `https://doi.org/10.5281/zenodo.19563812`
- Archive:
  `https://github.com/ll7/robot_sf_ll7/releases/download/0.0.2/paper_experiment_matrix_7planners_v1_release_v0_0_2_20260414_134316_publication_bundle.tar.gz`
- Archive SHA-256:
  `64e8510ab7ba934103c709907f66a783c7b3dd2dd58aa4bd725e762da2734d90`

Release `0.0.2` is the scoped seven-planner release. It is not the blocked all-planners release
with `socnav_bench`; that planner remains excluded until the licensed SocNavBench assets are
available.

## Embedded Required Artifacts

The release publishes one tarball asset. The paper-critical files are durable through their paths
inside that archive:

| Artifact | Archive path | SHA-256 |
|---|---|---|
| Publication manifest | `paper_experiment_matrix_7planners_v1_release_v0_0_2_20260414_134316_publication_bundle/publication_manifest.json` | `29af662569e83a201a3ff8b2316a7a1f60d66f1f7cff042cb412a6c9875a56bc` |
| Checksums | `paper_experiment_matrix_7planners_v1_release_v0_0_2_20260414_134316_publication_bundle/checksums.sha256` | `a7ba5f7b4405d6becd493963506563b80891a0fefa78675683a8959e22682904` |
| SNQI diagnostics JSON | `paper_experiment_matrix_7planners_v1_release_v0_0_2_20260414_134316_publication_bundle/payload/reports/snqi_diagnostics.json` | `aba8803dccde08fb86a86fbb34962928656a20ed4e1ac9dd8df181efa3220c6f` |
| SNQI diagnostics Markdown | `paper_experiment_matrix_7planners_v1_release_v0_0_2_20260414_134316_publication_bundle/payload/reports/snqi_diagnostics.md` | `dbb178d7240a52f79b7e4d8117b9344d728310b980c5aeefffd660376b5e0224` |

The compact tracked metadata lives in
`docs/experiments/publication/20260414_benchmark_release_0_0_2/`.

## Recovery Path

From a fresh checkout:

```bash
mkdir -p output/benchmark_release_0_0_2
gh release download 0.0.2 \
  --pattern 'paper_experiment_matrix_7planners_v1_release_v0_0_2_20260414_134316_publication_bundle.tar.gz' \
  --dir output/benchmark_release_0_0_2
sha256sum output/benchmark_release_0_0_2/paper_experiment_matrix_7planners_v1_release_v0_0_2_20260414_134316_publication_bundle.tar.gz
tar -tzf output/benchmark_release_0_0_2/paper_experiment_matrix_7planners_v1_release_v0_0_2_20260414_134316_publication_bundle.tar.gz \
  | rg 'publication_manifest.json|checksums.sha256|snqi_diagnostics\.(json|md)'
```

The SHA-256 should match
`64e8510ab7ba934103c709907f66a783c7b3dd2dd58aa4bd725e762da2734d90`.

## Validation Performed

On 2026-05-09:

- `gh release view 0.0.2 --json tagName,name,url,isPrerelease,assets,body`
  confirmed the release and asset URL.
- `gh release download 0.0.2 --pattern ... --dir output/issue_1062_release_probe`
  recovered the archive into ignored local output.
- `sha256sum output/issue_1062_release_probe/*.tar.gz`
  produced the archive checksum recorded above.
- `tar -tzf ... | rg 'publication_manifest.json|checksums.sha256|snqi_diagnostics\.(json|md)'`
  confirmed the required manifest, checksum, and SNQI diagnostic paths inside the archive.
- `curl -I -L https://doi.org/10.5281/zenodo.19563812`
  resolved through Zenodo to HTTP 200.

No raw benchmark outputs, episode JSONL, videos, model caches, or publication tarballs were added
to git.

## Historical Notes Updated

The older durable-artifact audits remain useful historical records, but their publication-bundle
blocker is now resolved for the scoped `0.0.2` seven-planner release:

- `docs/context/issue_1051_camera_ready_evidence_provenance_audit.md`
- `docs/context/issue_1053_durable_artifact_references.md`

The May 4 all-planners compact evidence remains partial and is still not a replacement for the
publication archive.
