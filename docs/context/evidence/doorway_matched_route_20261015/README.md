<!-- AI-GENERATED (recovered-packet custody, 2026-11-05) - NEEDS-REVIEW -->

# Matched-route doorway width series (recovered development packet, 2026-10-15)

**Diagnostic only, development seeds 1001–1030, 0.0.8 release code, no source changes
(`373dbfde4f39667cf9e8732dabe7df5118bdeab1`).** Not release evidence; admits nothing
to the dissertation by itself.

The packet runs ORCA, SocNav sampling and a stationary control through one doorway map
whose gap is widened from 2.00 to 2.80 m while the robot route (`[[7,5],[25.5,5]]`),
pedestrians and starts stay fixed. 450 episodes: five widths × three policies × 30
development seeds. The full interpretation, final report and frozen contract are
in the files held outside this public tree (see Custody).

| Width (m) | 2.00 | 2.20 | 2.40 | 2.60 | 2.80 |
| --- | ---: | ---: | ---: | ---: | ---: |
| ORCA successes /30 | 0 | 0 | 0 | 0 | 25 |
| SocNav sampling successes /30 | 0 | 0 | 16 | 24 | 29 |
| Stationary control successes /30 | 0 | 0 | 0 | 0 | 0 |

(Copied from `analysis/terminal_summary.csv`.)

## Custody

The original packet was sent to the external data hub on 2026-10-15 and that copy was
later unavailable. On 2026-11-04 the archive was rebuilt from the original files retained
on the acquisition host; every delivery file matches the original `MANIFEST`
(SHA-256 `44affe4c373dd5fe513196a4f5bf5911b77c0b8f2c3a378ac28f7e8b8601e0f8`).
The 192 MB archive `matched-route-packet.tar.gz`
(SHA-256 `6ad5d53737c2ae5995a4d3492a945282e0fc9b0cde72c67a89a4959aca6d8ded`) is preserved
as W&B artifact `ll7/robot_sf/doorway-matched-route-20261015-recovered:v0`; see
`custody.json` for the readback and the independent recount of all 450 raw records.

## Files

* Delivery files, byte-identical to the original delivery: `MANIFEST`, `MANIFEST.sha256`,
  `matched-route-packet.tar.gz.sha256`, `prereg.sha256`, and the archive's own `PAYLOAD_MANIFEST` (664 files).
* Payload members copied from the archive, each matching its `PAYLOAD_MANIFEST` line:
  `analysis/` (terminal summary, thresholds, validation, tables, paired differences,
  independent verification, verification note, route contrasts).
* `custody.json` and this README are new; `SHA256SUMS` covers these files and the four
  omitted operational files.

The four operational files (`prereg.md`, `provenance.json`, `report.md` and `result.md`)
are held in W&B archive `ll7/robot_sf/doorway-matched-route-20261015-recovered:v0` and
were sent to the external data hub, whose copy was later unavailable; their fingerprints are
in `SHA256SUMS`.

Verify the files in this tree with `sha256sum --ignore-missing -c SHA256SUMS`; this does
not verify the four omitted files. To check the archive, download the W&B artifact
into an empty directory and run `sha256sum -c MANIFEST.sha256 && sha256sum -c MANIFEST`.

## Limits

The ORCA zeros at 2.40 and 2.60 m are not explained by this packet; it records outcomes,
not their cause. The series is one map and route on development seeds; it does not give a
physical doorway-width requirement, a release-seed result, or behaviour of other planners.
The `result.md` 27/30 ORCA count at 2.80 m belongs to an earlier run with a different
robot route, not to this series.
