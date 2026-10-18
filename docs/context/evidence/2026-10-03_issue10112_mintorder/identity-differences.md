# Every dry identity difference

Old SHA-256: 103d3cacd02c5fd3dd007ae11b67d11cee122adbf8929faa5244af7919be43a0
New SHA-256: 63fa9684b27b548b25289e62eaaef4b81e3bfe03ad83999c94f92a96eb62870e

Both use output/freezeprep-dry/zenodo_metadata.resolved.json and diagnostic DOIs99000001/99000002. The old hash is reproduced by the unmodified c979e033 resolver. Generation and verification perform no episodes.

| Field | Explanation |
| --- | --- |
| `publication.metadata_sha256` | Resolved metadata incorporates the new source/tag coordinates; its tracked metadata template, DOI pair and output path are unchanged. |
| `release_tag` | Same diagnostic dry-tag pattern, with the newly selected source suffix. |
| `resolved_manifest.canonical_campaign_config_sha256` | D-083 YAML now declares the four acquisition/spec pins and same-source anchor-freeze contract. Authored budgets, planners, scenarios and sealed tuple are unchanged. |
| `resolved_manifest.identity_resolution.template_sha256` | Current main template adds SNQI-v2 asset bindings and repins the authored campaign. |
| `resolved_manifest.latest_main_base_commit` | Existing resolver derives the selected source first parent; this is the integrated branch parent, not a claim that main is frozen. |
| `resolved_manifest.metrics.snqi_v2_binding` | New explicit weights/family/pending-anchor/acquisition paths and hashes. Their source bytes are unchanged. Real acquired anchors remain a separate post-freeze artifact. |
| `resolved_manifest.planning_base_sha` | Same source-first-parent derivation as latest_main_base_commit. |
| `resolved_manifest.provenance.latest_main_base_commit` | Duplicate provenance carrier of the derived source parent. |
| `resolved_manifest.provenance.metadata_sha256` | Duplicate provenance carrier of the resolved metadata digest. |
| `resolved_manifest.provenance.planning_base_sha` | Duplicate provenance carrier of the derived source parent. |
| `resolved_manifest.provenance.source_commit` | Duplicate provenance carrier of the selected source. |
| `resolved_manifest.provenance.source_sha` | Duplicate provenance carrier of the selected source. |
| `resolved_manifest.release_id` | Template derives release_id from the supplied dry release tag. |
| `resolved_manifest.release_tag` | Duplicate manifest carrier of the supplied dry release tag. |
| `resolved_manifest.source_sha` | Manifest carrier of the selected source. |
| `resolved_manifest_sha256` | Canonical manifest digest changes because the source/tag/parents/asset pins and metadata digest change. |
| `source_commit` | New source includes implementation, authored-producer smoke admission, compatibility-facade forwarding, public CLI refusal witnesses, retention of the two source-declared ORCA scenario routes, focused witness corrections and integration of #10113. It is a dry candidate, not an orchestrator-named production freeze. |
| `template.sha256` | Envelope carrier of the changed tracked main template digest. |

The JSON ledger retains the exact old/new values for every changed field. No other field differs. All source assets remain pending/frozen as previously declared; the sealed tuple, matrix cardinality, scenario schedule and roster are byte-equal in the resolved manifests.

## Doorway identity

The main record above is the exact c979e033 → d56092ed dry comparison; it is not a new freeze at the round-1 head. The doorway has four additional field groups (six scalar leaves), recorded in [doorway-identity-differences.json](doorway-identity-differences.json):

| Field group | Old | New | Reason |
| --- | --- | --- | --- |
| canonical_campaign_config | three_width_doorway_v1 YAML | three_width_doorway_v2 YAML | Select the pending SNQI-v2 source successor. |
| canonical_campaign_name | paper_experiment_matrix_v2_h600_s30_three_width_doorway_v1 | paper_experiment_matrix_v2_h600_s30_three_width_doorway_v2 | The config name follows its successor. |
| snqi_weights path + sha256 | legacy camera_ready_v3 weights, 71a67c3c… | null + null | Legacy scoring is disabled; v2 weights belong to the four-asset v2 binding. |
| snqi_baseline path + sha256 | legacy camera_ready_v3 baseline, 329ca576… | null + null | Acquired v2 anchors replace legacy baseline scoring. |

The v2 set binds weights 684db941…, family 6ba0e5f4…, pending anchors bb1a526c… and scheduled dev1001/1002 acquisition fe55f5ef…. The complete hashes are in the JSON ledger. The legacy doorway concrete manifest is deliberately preserved and must be regenerated from the v0.2 template at the named freeze before any doorway execution. No doorway episode or production freeze is claimed.
