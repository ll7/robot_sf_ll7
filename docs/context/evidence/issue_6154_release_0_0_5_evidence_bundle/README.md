# Release 0.0.5 Evidence Bundle

Plain-language summary: this draft bundle assembles the three candidates that passed
the reverified frozen manifest from issue #6153. It is a release-preflight handoff,
not a published tag, GitHub Release, or Zenodo version.

- Schema: `release_0_0_5_evidence_bundle.v1`
- Issue: #6154
- Parent issue: #6149
- Zenodo concept DOI: `10.5281/zenodo.19482025`
- Frozen manifest: `docs/context/evidence/issue_6153_frozen_candidate_manifest.yaml`
- Frozen-manifest SHA-256: `84af86ca08b060132ed3d9f40d95254c8c73cc5096229623e173524d4ffe9177`

## Included Candidates

| Candidate | Issue | Classification | Acceptance checker | Claim boundary |
| --- | --- | --- | --- | --- |
| `candidate_5034` | #5034 | metric evidence promotion | passed | Targeted-smoke metric evidence for eligible native/adapter latency cells only. |
| `candidate_5305` | #5305 | certified adversarial archive | passed | 17 certified episode entries with stable failure-mechanism attributions and disjoint scenario splits. |
| `candidate_5592` | #5592 | preregistration packet | passed | Generalization-check contract across two scenario matrices only; no paper/dissertation ranking. |

## Excluded Candidates

| Candidate | Issue | Reason |
| --- | --- | --- |
| `candidate_4977` | #4977 | Campaign was not executed; its metric evidence is evaluated separately under #5034. |
| `candidate_5416` | #5416 | Current full acceptance checker is blocked by three geometrically infeasible rows; the owning issue retains the correction. |
| `candidate_5756` | #5756 | Campaign is still in progress and has no completed durable evidence directory. |

## Verification

Run these commands from the repository root. Each must report a passing status.

```bash
(cd docs/context/evidence/issue_6154_release_0_0_5_evidence_bundle && sha256sum -c SHA256SUMS)
uv run python scripts/tools/release_preflight_check.py \
  --checklist configs/benchmarks/releases/release_0_0_5_preflight_checklist.yaml --fail-on-blocked
uv run python scripts/repro/verify_release_checksums.py \
  --manifest configs/releases/release_0_0_5_checksum_manifest.yaml --no-download
```

The first command verifies every payload file and review marker except the
checksum manifest's own review marker; the third command verifies that final
marker through the outer release checksum manifest. The second verifies the
release preflight, including the frozen-manifest checksum and claim audit.

## Files

- `release_manifest.yaml` — candidate dispositions, claim boundaries, provenance, and residual risks.
- `SHA256SUMS` — SHA-256 checksums for every file in this bundle directory.
- `candidates/` — one checksum-verified reference manifest for each included candidate.
