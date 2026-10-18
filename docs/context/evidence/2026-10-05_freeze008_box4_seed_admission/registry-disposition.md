# Exact resolver identity copies: registry representation findings

AI-GENERATED NEEDS-REVIEW. The raw resolver identities are preserved byte-for-byte
because changing their JSON fields would invalidate the scientific binding.
The generic evidence registry does not recognize all fields of
`benchmark-release-resolved-identity.v1`. This packet explicitly disposes its
38 new representation findings; it does not suppress a finding category or
change any linter, coverage threshold or frozen identity bytes.

| Copied identity | Code | Count | Disposition and verification |
| --- | --- | --- | --- |
| main/release_identity.resolved.json | hash_without_artifact_path | 18 | Schema-specific bindings: validated by F2 generate/verify and the exact whole-file SHA256, which also binds nested manifest/publication/template hashes. |
| doorway/release_identity.resolved.json | hash_without_artifact_path | 16 | Same frozen-schema custody check, with the companion template/config and its separate metadata. |
| main/release_identity.resolved.json | uncommitted_artifact_missing_location | 2 | Original runtime paths remain under ignored output/release-008; archived main metadata is main/zenodo_metadata.resolved.json, and the determinism receipt is the actual published Git blob listed in README. Restage originals at their declared paths for verification. |
| doorway/release_identity.resolved.json | uncommitted_artifact_missing_location | 2 | Archived companion metadata is doorway/zenodo_metadata.resolved.json; the same published determinism receipt is restaged at the declared ignored path. |

These are source-bound manifests, not missing raw campaign data. All copies have
SHA256-bound review sidecars. `build_evidence.py` refuses changed inputs and
checks the canonical configuration blobs; the primary frozen resolver verifies
the exact identity bytes and source notes receipt. No artificial artifact path
or location was inserted into the identity just to satisfy a generic scanner.
The baseline refresh records 492 findings (454 existing + these 38), keeping
all existing per-file findings unchanged. Independent review must explicitly
accept this representation disposition together with the admission proposal.
The new counts remain visible, with this explanation; no unsupported benchmark
claim or execution authority is supplied by a baseline refresh.
