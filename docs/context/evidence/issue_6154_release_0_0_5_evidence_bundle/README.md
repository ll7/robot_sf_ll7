# Release 0.0.5 Evidence Bundle

Plain-language summary: this bundle assembles four frozen release candidates from issue #6153's frozen candidate manifest into the 0.0.5 release evidence bundle. Two additional candidates are excluded per the frozen manifest disposition.

- Schema: `release_0_0_5_evidence_bundle.v1`
- Issue: #6154
- Parent issue: #6149
- Zenodo concept DOI: `10.5281/zenodo.19482025`
- Bundle assembly commit: `adadabacabfa663d6ace7eac10e23c67e6a18da7`

## Included Candidates

| Candidate | Issue | Classification | Acceptance Checker | Claim Boundary |
|---|---|---|---|---|
| `candidate_5034` | #5034 | metric_evidence_promotion | verified | control-action-latency metric-evidence promotion only |
| `candidate_5305` | #5305 | certified_adversarial_archive | passed | certified adversarial archive of 17 entries, disjoint splits |
| `candidate_5416` | #5416 | preregistration_packet | blocked (see caveat) | Preregistration and CPU contract only |
| `candidate_5592` | #5592 | preregistration_packet | ready | Generalization check contract, two matrices only |

## Excluded Candidates

| Candidate | Issue | Disposition | Rationale |
|---|---|---|---|
| `candidate_5756` | #5756 | exclude | Campaign in progress, no completed evidence directory |
| `candidate_4977` | #4977 | exclude | Campaign not executed, metric evidence promoted under #5034 |

## Caveats

1. **#5416 acceptance checker regression**: The `check_issue_5416_sipp_four_geometry_packet.py` checker now reports `status=blocked` instead of the frozen manifest's recorded `status=ready`. This is because PR #6172 updated the geometry certifier (`scenario_cert.v1`), which now correctly classifies `classic_doorway_low`, `classic_station_platform_medium`, and `classic_merging_low` as `geometrically_infeasible`. The preregistration packet itself documents these as expected exclusions via `certification_caveat: excluded_geometrically_infeasible_post_pr_6172`. Candidate remains in the bundle with this caveat; correction routed to owning issue #5416.

2. **Changelog entries**: The CHANGELOG.md release notes restate only each candidate's `allowed_claim` from the frozen manifest. No claim is widened beyond its recorded boundary.

3. **Provenance recording**: Execution-versus-publication commit lineage is recorded in the release_manifest.yaml. For #5034: source_commit=484d3fd0 matches manifest. For #5305: source_commit=ecf997d3 matches manifest. For #5416: source_commit=13cb68de matches manifest. For #5592: source_commit=96776636 matches manifest.

## Files

- `release_manifest.yaml` — full release evidence bundle manifest with per-candidate metadata, checksums, and residual risks.
- `SHA256SUMS` — sha256 checksums for all files in this bundle directory.
- `candidates/candidate_5034_reference.yaml` — reference to #5034 evidence with checksum verification.
- `candidates/candidate_5305_reference.yaml` — reference to #5305 evidence with checksum verification.
- `candidates/candidate_5416_reference.yaml` — reference to #5416 preregistration packet with checksum verification.
- `candidates/candidate_5592_reference.yaml` — reference to #5592 preregistration packet with checksum verification.
- `README.md` — this file.
