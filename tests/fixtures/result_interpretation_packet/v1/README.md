# result_interpretation_packet fixtures

Compact, deterministic fixture packets for the `result_interpretation_packet.v1`
contract (issue #7029).  Each fixture references tracked repository evidence by
path and SHA-256 digest without copying raw episode data.

## Fixtures

| File | Source issue | Decision type |
|------|-------------|---------------|
| `issue_6474_comfort_exposure_supported.json` | #6474 | `supported` |
| `issue_6944_brne_candidate_transition_diagnostic.json` | #6944 | `not_supported` |
| `ch7_visualization_causal_abstention.json` | #6792 | `unavailable` |
| `issue_6962_lane_formation_diagnostic.json` | #6962 | `inconclusive` |

## Determinism

All SHA-256 digests were computed with `sha256sum` against tracked repository
files at the base commit.  Each source records both the generation commit and
the `tracked_commit` that contains the exact bytes behind the digest.  Fixtures
are single-line JSON for deterministic serialization.  No raw episode data is
copied; only tracked paths and digests are bound.  Generation commands must
name a script present at the recorded generation commit; review sidecars use
the explicit `evidence-review-marker.v1` marker.

## Regeneration

```bash
sha256sum docs/context/evidence/issue_6474_social_compliance_nominal_campaign_interpretation.md
sha256sum docs/context/evidence/issue_6944_brne_candidate_transition_summary.json
# etc.
```
