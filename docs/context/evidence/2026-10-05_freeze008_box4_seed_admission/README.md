# F2 DOI-bound sealed seed admission — proposal for box 4 review

AI-GENERATED NEEDS-REVIEW. This packet proposes the `decision: admitted` ruling;
it is not an already completed scientific review, a checked box or execution
authorization. The stable review packet reference is
`freeze008-box4-independent-review-2026-10-05`. Independent box 4 review must
accept this exact ruling and both tracks; a separate private box 5 pin review
and subsequent smoke/mint/execution authorizations remain required.
The [#10137 review](https://github.com/ll7/robot_sf_ll7/pull/10137#issuecomment-5984099933)
accepted F2 calibration boxes 1/2; it did not admit sealed evaluation seeds.

## Frozen source, output and custody

F2 is `66f402ba176b13e45210d0da0b2cf20fcdc0cc02`; no freeze or code change.
Both templates deliberately keep source anchors `pending_calibration`, digest
`bb1a526c818efca16c6d4237127c27dc7812947758f78a18c8f1d28c2a65fa4c`.
Acquired anchors are a separate runtime `--snqi-v2-anchors` input, authenticated
by the proposed private Git-blob trust pin. They never replace the identity's
pending source anchors. The doorway's raw-only claim exclusion remains in
[publication scope](../../../release/0.0.8/doorway_claim_scope.md); computed SNQI
diagnostics for that slice are not admitted as results, ranked or compared.

Reservation state was independently supplied and validated locally: deposition
23150472, unsubmitted, concept DOI `10.5281/zenodo.23150471`, version DOI
`10.5281/zenodo.23150472`. Credential-free state SHA256:
`bb2b957dc5acba2105fe73d47085f7417432dbb6ace91bd01f2a7209d14be7aa`.
There was no token access or Zenodo API call. Only the main metadata is the
operator's draft-update input; never PUT the companion metadata.

Steps 1–4 ran in a clean F2 checkout, with origin/main fetched and `uv sync
--frozen`. The resolver script and imported release protocol were F2 files.
All outputs were real files inside ignored `output/release-008/` and Git status
was empty after generation and verification. Both tracks use the same tag and
DOI coordinates, exactly as F2 runbook §3a prescribes:
`paper-matrix-v2-h600-s30-2026-10-66f402ba176b13e45210d0da0b2cf20fcdc0cc02`.

Stage the actual published blobs, without symlinks or source YAML edits:

```bash
D=docs/context/evidence/2026-10-04_freeze008_f2_calibration
S=output/release-008/calibration
mkdir -p "$S"
for f in anchors.v2.0.acquired.json acquisition-proof.json determinism-receipt.json; do
  git show "527eb2509c4375d6ce81091d7d67260d35844310:$D/$f" > "$S/$f"
done
sha256sum "$S"/*.json
```

| Published blob | SHA256 |
| --- | --- |
| acquired anchors | `8d86636bcb33a27bab6ba97318516145112aaebe4a4a9713665ec2e39fbc7349` |
| acquisition proof | `e4375e44f9a58cfba600e42c50112f61880ebc92744946f6976899f79c5cc0d1` |
| determinism receipt | `cd29d6c9a3213e4dbfabc1b8b8c23305066a6a39c5eb088f54d8f8d539827464` |

For each of `main` and `doorway`, use its own template; all other coordinates
are shared. Generate once per template, then verify both with F2's resolver:

```bash
FREEZE_SHA=66f402ba176b13e45210d0da0b2cf20fcdc0cc02
TAG=paper-matrix-v2-h600-s30-2026-10-$FREEZE_SHA
for track in main doorway; do
  template=benchmark_data_release_s30_h600.template.yaml
  if [ "$track" = doorway ]; then
    template=three_width_doorway_release_0_0_8_v1.template.yaml
  fi
  uv run --frozen python scripts/tools/resolve_benchmark_release_identity.py generate \
    --template "configs/benchmarks/releases/$template" \
    --output "output/release-008/$track/release_identity.resolved.json" \
    --source-commit "$FREEZE_SHA" --release-tag "$TAG" \
    --concept-doi 10.5281/zenodo.23150471 --version-doi 10.5281/zenodo.23150472 \
    --determinism-receipt-path "$S/determinism-receipt.json" \
    --determinism-receipt-sha256 cd29d6c9a3213e4dbfabc1b8b8c23305066a6a39c5eb088f54d8f8d539827464
done
for track in main doorway; do
  uv run --frozen python scripts/tools/resolve_benchmark_release_identity.py verify \
    --identity "output/release-008/$track/release_identity.resolved.json"
done
git status --porcelain
```

| Concrete output | SHA256 |
| --- | --- |
| main identity (ruling's manifest_sha256) | `527f9dc5e9ee3004e93444789472eb25a29438e64741c4c4b7b95a10f0db3e71` |
| doorway identity | `6fce4eb41436213a2c4d9790b74811b3492126297ea654ef89a1f3d94e17e199` |
| main Zenodo metadata | `e21b48b69916e2ab669759eb485a540b2d3c398d59be3496c6bb1e28fba7790e` |
| doorway metadata (not PUT) | `819335ea5b096d8592b0d5c4c71408e4e3649f39632fd8e165d176480ff4ce37` |

Both `verify` commands passed, including frozen release-notes gates. The copied
identities/metadata/notes receipts preserve those exact bytes. To verify the
archive, restage them at their original ignored paths in clean F2; do not run
main's resolver on the copies or rewrite their internal paths.

## Separate configuration hashes

The runtime acquisition identity check recomputed
`sha256(json.dumps(_config_hash_payload(acquisition), sort_keys=True,
separators=(",", ":")).encode())` and called F2
`_validate_acquisition_identity(..., require_config_identity=True)`.
It exactly matches the acquired anchor's campaign_config_hash:
`6cd6a57f3583d150d51ac52b7a4a4fd80fa7b077795ac3c7973ef643d3f8fbbd`.
The frozen acquisition YAML byte digest remains
`fe55f5efb6fd885ae86fc978dffc01afd5928fba75128442a6dcd88ae9e94ff3`.
This acquisition hash is distinct from the sealed campaign config byte digest
that the private mint checks in the ruling:

- Main canonical config: `de627db9f2999b17ed56378b9d793ee13a4b1bc15d1eb442513360d45f6d76d5`.
- Companion canonical config: `e3a26ccfb9df1cea343fba9f83aba1e9505494cbead7121132922bffb5c75039`.

The ruling binds the main identity file hash, not the inner resolved-manifest
hash. It retains both companion bindings and the exact 30-seed tuple. Seeds
were read statically; no planner or environment was reset or stepped.

## Byte-preserving evidence regeneration

From a checkout containing the publication commit and the verified F2 outputs:

```bash
python docs/context/evidence/2026-10-05_freeze008_box4_seed_admission/build_evidence.py
```

The builder validates exact input digests, source/config Git blobs, both DOI
pairs, receipt and the sealed tuple; it copies resolver outputs without edits,
regenerates the proposal and refreshes every review sidecar. It does not grant
scientific admission or call a runner, scheduler or Zenodo. Independent review
must check source/DOI/config/split bindings and the separate companion claim
boundary before either PR can merge. No review boxes are ticked here.
