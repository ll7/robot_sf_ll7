# Planned 0.0.8 metadata-only correction successor

AI-GENERATED NEEDS-REVIEW. **Planned, not published; not a PUT-ready contract.**

The author selected [D-087](decisions.md#d-087-correct-the-doi-description-through-a-metadata-only-successor-after-f2-publication)
on 2026-10-05. Publish the existing F2 identity and bound metadata unchanged first.
Then build and validate a metadata-only successor with main's accepted tooling.
The scientific source stays `66f402ba176b13e45210d0da0b2cf20fcdc0cc02`; the concept
DOI stays `10.5281/zenodo.23150471`, which the thesis cites. The predecessor
version DOI is `10.5281/zenodo.23150472`. No new calibration or freeze move is
authorised by this plan. Ordinary scientific, mint and publication gates still apply.

The [main planned metadata](planned_main_zenodo_metadata_successor.json) supplies
the future successor description, unchanged-row/no-simulation erratum boundary
and one `isNewVersionOf` relation to the predecessor. Its `metadata` object must
be rendered into a separately reviewed, concrete successor metadata file;
the planning wrapper is not submitted to Zenodo. Use:

- `{{source_sha}}`: F2, unchanged.
- `{{latest_main_base_commit}}`: `1261295887901e566a93da87a4559ea1040808a7`,
  the freeze-branch first parent of F2, unchanged. The internal slot name is
  historical; the description now labels its actual role.
- `{{release_tag}}`: `paper-matrix-v2-h600-s30-2026-10-66f402ba176b13e45210d0da0b2cf20fcdc0cc02-erratum.1`.
- `{{concept_doi}}`: `10.5281/zenodo.23150471`.
- `{{version_doi}}`: the distinct successor version DOI returned by the linked
  new-version action, later frozen into the reviewed erratum contract. No
  successor version DOI is asserted or reserved here.

The [doorway planned description](planned_doorway_zenodo_metadata_successor.json)
is companion publication prose only. It retains the raw-metrics-only,
not-validated-for-this-slice SNQI exclusion. Only main metadata is PUT,
verified and published under the one concept; never reserve a separate DOI or
PUT the companion JSON. Retain `snqi_claim_policy=advisory_no_ranking`.

The immutable F2 metadata and acquired calibration evidence remain unchanged.
Main's reusable templates are corrected only to prevent future descriptions
from inheriting the stale clause and base label. Main's corrected templates
must not be substituted into F2 identity generation or draft update.

Follow [the derived-metadata successor procedure](../../RELEASE.md#immutable-publication-errata)
and [publication follow-up #10143](https://github.com/ll7/robot_sf_ll7/issues/10143). Re-download and hash the actual immutable
published predecessor archive before building; keep its canonical inner
`benchmark-release-manifest.v0.2` at
`payload/release/release_manifest.resolved.json`. An outer
`benchmark-release-resolved-identity.v1` envelope is not that archival manifest.
If the real build refuses the actual archive, reopen D-087 and stop publication.

The offline enforcing test proves contract compatibility and unchanged-row
equality on synthetic artifacts. It does not prove that the still-unpublished
real predecessor has been published or that its eventual archive passes.
