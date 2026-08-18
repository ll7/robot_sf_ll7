# Tracked asset-rights inventory

`scripts/validation/asset_rights_inventory.v1.yaml` is the repository’s narrow asset-rights
boundary for issue #7299. It classifies tracked map, data, media, example, renderer, fixture, and
evidence families. It does not grant rights, infer permission from a paper or filename, or replace
the separate model/checkpoint review in issue #6855.

Run the strict release check with:

```bash
uv run python scripts/tools/check_asset_rights_inventory.py --json
```

The strict command exits `2` while a row is `blocked` or `external-pointer-only`. That is
intentional: unresolved source, attribution, license, checksum, or modification evidence must
remain fail-closed. Pull-request CI uses `--allow-known-blockers` so it can reject new
unclassified/overlapping/malformed rows while reporting the existing legal blockers truthfully.

Every tracked path in a declared scope must match exactly one row. A new release-relevant asset
therefore requires a row before it can be added. Each row records source, source revision or
access date, rights status, attribution, checksum policy, modification status, and evidence (or an
explicit unblock condition). `queue_id`, benchmark readiness, and model-weight rights are outside
this contract.

The checked-in OVGU import has an ODbL source, OpenStreetMap attribution, and a raw-input SHA-256
in its provenance sidecar. Its `exploratory_only` classification still prevents it from becoming
benchmark evidence automatically. OSM exports without equivalent per-family provenance, Francis
2023 and master-thesis-derived maps, SocNavBench pointers, example recordings, and example data
remain explicit blockers. Documentation, fixtures, specifications, and generated evidence are
explicitly excluded from the runtime release surface; exclusion is not clearance for any external
asset they reference.

External datasets staged outside Git continue to use
[`manage_external_data.py`](../../scripts/tools/manage_external_data.py) and its own terms and
checksum manifests. This inventory only sees tracked paths and never downloads, edits, stages, or
deletes data.
