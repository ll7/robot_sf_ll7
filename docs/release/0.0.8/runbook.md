# 0.0.8 freeze → tag runbook

This is an ordered admission runbook, checked against public main
`c979e0337da4ad053d59a76225fbb9154140ee73` on 2026-10-03 and the private-ops
main `dba879cbe5fc5390c6c06af82daa739fc9b855b5` files read through `gh`. Commands below use existing tools. A missing
input or an explicitly blocked admission is a stop for that stage. This
preparation lane does not select/move a freeze, acquire sealed episodes, mint
an admitted packet, publish, tag or perform a DOI action.

The orchestrator names the freeze. Publish, tag and DOI remain **author-reserved**;
their execution was **delegated on 2026-09-28**. Delegation is not approval,
scientific admission, permission supplied by this document, or evidence that an
action happened. Calibration uses dev 1001/1002 only. Development/rehearsal
uses 1001–1030; retired 111–140 and the sealed 0.0.8 tuple must never reset or
step outside the separately admitted campaign. A refusal before reset is
recorded and work continues. Static resolved identity generation is safe.

Use a clean source checkout and a durable output root **inside the task's lane**.
Do not reuse stale identities or override guards. Set these concrete inputs
from the orchestrator's packet, with all paths under the lane:

```bash
export FREEZE_SHA='<named full 40-character SHA>'
export TAG='<dataset tag ending in that exact full SHA>'
export ARTIFACT_ROOT='<absolute durable lane artifact directory>'
export OPS='<clean robot_sf_ll7-private-ops main checkout inside the lane>'
export CALIBRATION_ID='<fresh development calibration campaign id>'
export CAMPAIGN_ID='<fresh sealed campaign id>'
export DOORWAY_CAMPAIGN_ID='<fresh companion campaign id>'
export SMOKE_ID='<fresh admitted development runtime smoke id>'
# Publication plan selects a fresh main bundle basename; use a distinct companion name.
export BUNDLE_NAME="${CAMPAIGN_ID}_publication_bundle"
# Hydrate this immutable 0.0.7 archive from the release identified in README.md:48-50.
export BASELINE_007_ARCHIVE="$ARTIFACT_ROOT/baseline/issue9431_release_benchmark_data_0_0_7_07f7e8d43084_20260922_publication_bundle.tar.gz"
# Bound to the generated and verified main identity by sha256sum in step 3a.
export MAIN_IDENTITY_SHA256='<not usable until step 3a binds the digest>'
export MINT_DATE='<fresh YYYYMMDD suffix from the preparation packet>'
export OPS_RUNTIME='<reviewed detached private-ops runtime checkout inside the lane>'
export CONCEPT_DOI='<actual reserved concept DOI>'
export VERSION_DOI='<actual reserved unpublished version DOI>'
export ZENODO_STATE='<operator-owned reservation state file>'
export ZENODO_METADATA='<concrete operator-reviewed Zenodo metadata JSON>'
export TOKEN_FILE='<operator-owned token file; never log its contents>'
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
```

`<...>` denotes required input, never a usable placeholder. Baseline input is
`issue9431_release_benchmark_data_0_0_7_07f7e8d43084_20260922_publication_bundle.tar.gz`,
SHA-256 `684da7c557c426756f22ddbf5cb3270141ee8ae385669a39d36f324852a6fb2f`,
source `07f7e8d43084de748915e1b1eb8b2a1603357c6e`. The archive must exist at
`BASELINE_007_ARCHIVE` and match that SHA-256 before step 6; the [root README](../../../README.md)
identifies its release/tag and the immutable baseline provenance. `BUNDLE_NAME`
is an export input chosen from the fresh campaign identity, not an exporter
receipt. Preserve all 0.0.7 artifacts.

Execution order: 1 freeze → 3a preparatory row mint/identities → 2 calibration →
3b same-source smoke → 3c final campaign mint → 4 sealed campaign → 5 export/
preflight → 6 comparator → 7 tag/publication/DOI. No final mint runs before smoke.

**Orchestrator ruling, 2026-10-03:** [#10112](https://github.com/ll7/robot_sf_ll7/issues/10112)
**blocks the freeze**: D-083 SNQI-v2 binding, the smoke contract and mint ordering
require release-source fixes, and SNQI v2 is a reported 0.0.8 number.
[#10110](https://github.com/ll7/robot_sf_ll7/issues/10110) **must also land before
the freeze**: the mint/preflight release-notes gate is source that the resolved
identity binds. Complete and review both fixes before naming/moving the final
freeze; then reprove its source and identity. This supersedes the earlier
mint/publication-only classification; this documentation is not the fixes.

## 1. Move the freeze branch — orchestrator only

```bash
git fetch origin main release/0.0.8-freeze
git merge-base --is-ancestor origin/release/0.0.8-freeze "$FREEZE_SHA"
git push origin "$FREEZE_SHA:refs/heads/release/0.0.8-freeze"
git ls-remote origin refs/heads/release/0.0.8-freeze
git switch --detach "$FREEZE_SHA"
uv sync --all-extras --reinstall-package robot-sf
```

Inputs: named candidate, exact-head test/audit/rehearsal/intake evidence and
current remote freeze ref. Output: remote freeze points to the named SHA;
clean detached checkout and rebuilt installed physics. Admission: full suite,
pin/seed/identity witnesses classified, hosted CI checked, all measured P1
blockers closed, #10112 and #10110 source fixes landed and reviewed under the
2026-10-03 ruling, D-070 intake disposition recorded; verify the remote SHA by
readback. Check ancestry of the train commits and #10081/#10045/#10103/#10108.
If non-fast-forward, stop for the orchestrator; never force. See
[freeze_audit.md](freeze_audit.md) for the candidate, which is not a ruling.
The guard compares actual imported `robot_sf` and all non-cache
`pysocialforce` files with source blobs. A stale installed copy is a refusal.

## 2. Acquire SNQI-v2 calibration and freeze the anchor artifact

Run acquisition inside one Slurm allocation of at most 32 CPUs, not a login
node. Stage before execution; do not pass staging options to run mode.

```bash
uv run python scripts/benchmark/preflight_campaign_checkpoints.py \
  --config configs/benchmarks/snqi_v2/calibration.dev1001_1002_scheduled_acquisition.yaml \
  --stage --json --report-path "$ARTIFACT_ROOT/calibration-checkpoints.json"
uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/snqi_v2/calibration.dev1001_1002_scheduled_acquisition.yaml \
  --output-root "$ARTIFACT_ROOT/calibration" --campaign-id "$CALIBRATION_ID" \
  --arm-isolation subprocess
uv run python scripts/tools/analyze_snqi_contract.py \
  --campaign-root "$ARTIFACT_ROOT/calibration/$CALIBRATION_ID" \
  --freeze-v2-anchors "$ARTIFACT_ROOT/anchors.v2.0.json"
```

Inputs: exact named source, authored schedule, 14 configs/models, 48 scenarios,
seeds **1001/1002** (1,344 cells), staged checksums. Outputs: raw per-arm
JSONL/sidecars, manifest/preview/summary and atomic frozen anchors with custody,
command-mode census, schema/sealed-seed commitment and force-source decision.
Admission: print the resolved dev seeds before dispatch; exact 14×48×2 census,
zero fallback/degraded rows, no imputation, actual F/J/K p95, T=3/N=0.25 and
strict calibration binding. A 3-seed rehearsal is not calibration custody.

**Post-freeze scientific stop:** D-083 and both current templates now bind the
weights, family, pending anchors and scheduled dev1001/1002 acquisition inputs.
The source anchor file stays `pending_calibration`. Freeze a separate acquired
artifact at the named source; do not edit tracked anchors or move the source
between acquisition and campaign. The public runner's strict loader checks its
source/configuration identity; complete raw custody can additionally be rederived.
Private scientific trust pins remain empty until independent review of actual
acquisition and sealed-source ruling. Use the checklist below before final mint.

## 3. Two-phase preparation and final mint

### 3a. Preparatory row mint and resolved identities

DOI coordinates must already be authentic reserved-unpublished coordinates.
If absent, the **author-reserved delegated DOI operator** performs the existing
reservation command before identity generation, not a second reservation at tag:

```bash
uv run robot-sf release zenodo reserve --token-file "$TOKEN_FILE" \
  --state "$ZENODO_STATE" --metadata "$ZENODO_METADATA"
```

For both the main and the fixed H400 2.2/2.8/3.6 m companion:

```bash
uv run python scripts/tools/resolve_benchmark_release_identity.py generate \
  --template configs/benchmarks/releases/benchmark_data_release_s30_h600.template.yaml \
  --output output/release-008/main/release_identity.resolved.json \
  --source-commit "$FREEZE_SHA" --release-tag "$TAG" \
  --concept-doi "$CONCEPT_DOI" --version-doi "$VERSION_DOI"
uv run python scripts/tools/resolve_benchmark_release_identity.py generate \
  --template configs/benchmarks/releases/three_width_doorway_release_0_0_8_v1.template.yaml \
  --output output/release-008/doorway/release_identity.resolved.json \
  --source-commit "$FREEZE_SHA" --release-tag "$TAG" \
  --concept-doi "$CONCEPT_DOI" --version-doi "$VERSION_DOI"
uv run python scripts/tools/resolve_benchmark_release_identity.py verify \
  --identity output/release-008/main/release_identity.resolved.json
uv run python scripts/tools/resolve_benchmark_release_identity.py verify \
  --identity output/release-008/doorway/release_identity.resolved.json
sha256sum output/release-008/{main,doorway}/release_identity.resolved.json
export MAIN_IDENTITY_SHA256="$(sha256sum output/release-008/main/release_identity.resolved.json | awk '{print $1}')"
```

Mint preparation rows with the existing private-ops tool (not the final campaign
mint, and not submission):

```bash
"$OPS/.venv/bin/python" "$OPS/ops/jobs/scripts/mint_snqi_v2_release_rows.py" \
  --public-root "$PWD" --public-sha "$FREEZE_SHA" --date "$MINT_DATE" \
  --private-ops-root "$OPS" --private-ops-runtime-worktree "$OPS_RUNTIME" \
  --out-dir "$ARTIFACT_ROOT/preparatory-rows"
```

Inputs: clean named public source, clean reviewed private runtime, fresh suffix,
reserved coordinates and calibration/smoke source contracts. Outputs: proposed
calibration/smoke rows and packets, plus both verified resolved identities;
`MAIN_IDENTITY_SHA256` is the actual main identity digest consumed in step 6.
Admission: inspect the proposed rows and complete independent queue/packet
admission before any smoke dispatch; no `--admit` or diagnostic output grants
release authority. The completed step-2 calibration custody remains mandatory.

### 3b. Same-source runtime smoke before final mint

The existing ordinary release runner command is:

```bash
uv run python scripts/benchmark/preflight_campaign_checkpoints.py \
  --config "$SMOKE_CONFIG" --stage --json --report-path "$ARTIFACT_ROOT/smoke-checkpoints.json"
uv run python scripts/tools/run_benchmark_release.py \
  --manifest "$SMOKE_MANIFEST" --label runtime-smoke-008 --campaign-id "$SMOKE_ID" \
  --checkpoint-receipt "$ARTIFACT_ROOT/smoke-checkpoints.json"
```

Inputs `SMOKE_CONFIG`/`SMOKE_MANIFEST`: a reviewed tracked same-source 0.0.8
successor, full 14-arm roster, dev **1003**, same learned-model fingerprints,
authored horizon/kinematics contract, exact imported runtime; allocation ≤32 CPUs.
Bind the produced receipt only after successful smoke and admission:

```bash
export SMOKE_RESULT="$PWD/output/benchmarks/camera_ready/$SMOKE_ID/release/release_result.json"
test -f "$SMOKE_RESULT"
```

Output: that release result, raw rows and separately authenticated environment receipt. `CAMPAIGN_ROOT` and the companion root are the respective `output/benchmarks/camera_ready/<id>` directories; `BUNDLE_DIR`/`BUNDLE_ARCHIVE` come from the exporter receipt, not an invented file name.
Admission: every arm successful/native, source/model bindings equal campaign,
result and staging age ≤24 h, independently accepted environment/stress receipt.
Record CPU model; learned episodes/resume must stay on the same node (D-072).

**Current reviewed-source contract:** use the tracked v0_6 config/manifest,
all 14 current D-083 keys, dev seed **1003**, dt=0.1, differential drive and the
authored blind-corner **H400**. The public validator selects this successor;
v0_2 and v0_5 remain historical bytes. Same-source native success plus authentic
separate environment/stress/staging/cold-custody receipts is still required.
D-086 all-roster development smoke remains permanently diagnostic and is
explicitly refused as ordinary release admission. A source contract is not an
admission receipt.

### 3c. Final campaign mint after admitted smoke

Public main has an identity resolver, **no production queue mint**. The existing
private-ops main production mint is:

```bash
"$OPS/.venv/bin/python" "$OPS/ops/jobs/scripts/mint_snqi_v2_full_campaign.py" \
  --request "$ARTIFACT_ROOT/full-mint-request.json" \
  --out-dir "$ARTIFACT_ROOT/full-mint" --report "$ARTIFACT_ROOT/full-mint-report.json"
```

Inputs: request matching `private-ops:docs/full_campaign_mint.md` and
`docs/post_freeze_release_chain.md`, clean public/private source SHAs, reserved
DOIs, both verified identities, independently pinned scientific anchors and
seed admission, both staged checkpoint receipts, authentic smoke result and
separate environment admission, stress/cold-custody receipts, fresh identities
and immutable preservation destination. Request `cpus=32` for this lane.
Stage main and companion checkpoint receipts with `preflight_campaign_checkpoints.py --config <each exact resolved campaign config path> --stage --json --report-path <main-checkpoints.json or doorway-checkpoints.json>`; resolve those campaign config paths from the verified identities.
Outputs: atomic **proposed, go=false, non-dispatchable** queue/packet pair and
mint report; no submission or authority grant. Admission: exact sealed tuple,
20,160 main +1,260 companion cells, disjoint cell identities, shared source/DOIs,
strict source/model/hash closure and independently reviewed trust pins; release
notes gate [#10110](https://github.com/ll7/robot_sf_ll7/issues/10110) must already
be implemented in the freeze-bound source and pass. Stop unless stage 3b produced
the authenticated `SMOKE_RESULT` and separate environment admission; supply
those exact receipts in `full-mint-request.json`.

**Ordering stop retained:** existing main cannot execute literal final mint →
first smoke: full mint requires a successful smoke already. Stage 3a is the
different preparatory-row tool, stage 3b must produce and admit that same-source
receipt, and only then can stage 3c run. The #10112 public/private successors
must land before freeze naming. Do not omit smoke inputs or use diagnostic
receipts as admission. The private executor forwards the reviewed `snqi_anchors`
artifact to both public runners; raw custody is an optional additional public
rederivation input, never an independent scientific trust grant.

## 4. Sealed campaign and fixed companion

After independent queue/packet/scientific admission, the canonical private driver
submits the minted row; do not bypass its environment/allocation/smoke/stress
checks. Read the row's actual canonical driver argv and read back job identity.
The existing public runner invocation inside that admitted allocation is:

```bash
uv run python scripts/tools/run_benchmark_release.py \
  --manifest output/release-008/main/release_identity.resolved.json \
  --label release-008 --campaign-id "$CAMPAIGN_ID" \
  --checkpoint-receipt "$ARTIFACT_ROOT/main-checkpoints.json" \
  --runtime-smoke-receipt "$SMOKE_RESULT"
uv run python scripts/tools/run_benchmark_release.py \
  --manifest output/release-008/doorway/release_identity.resolved.json \
  --label release-008-doorway --campaign-id "$DOORWAY_CAMPAIGN_ID" \
  --checkpoint-receipt "$ARTIFACT_ROOT/doorway-checkpoints.json" \
  --runtime-smoke-receipt "$SMOKE_RESULT"
```

Existing single-node submission wrapper (canonical admitted packet must own launch):

```bash
sbatch --cpus-per-task=32 SLURM/submit_release_single_node.sbatch \
  output/release-008/main/release_identity.resolved.json release-008 "$CAMPAIGN_ID" \
  "$ARTIFACT_ROOT/main-checkpoints.json" "$SMOKE_RESULT"
```

For the two-track path, the existing canonical driver is
`ops/jobs/scripts/submit_via_canonical_driver.sh`; it delegates to
`submit_and_record.sh` from clean private-ops main. Supply the reviewed minted
row's complete `submit_args` and bound packet, with `cpus=32`; admission/promotion
must precede submission. The guarded `submit_s30_h600_release.sh` then runs this
exact executor command inside the allocation:

```bash
"$PYTHON_PATH" "$OPS/ops/jobs/scripts/execute_release_chain.py" \
  --packet "$ADMITTED_PACKET" --out-dir "$RESULT_ROOT/release_chain"
```

`PYTHON_PATH`, `ADMITTED_PACKET` and `RESULT_ROOT` are authenticated wrapper
inputs. The wrapper supplies `RELEASE_CHAIN_VALIDATED_PACKET_SHA256` only after
its startup/admission checks; never set it manually to bypass the driver.
The executor retains both sequential runner calls in one allocation. Do not
replace this chain with two independent unbound submissions.
Outputs: separate raw roots, strict main/companion acceptance, both bundles and
terminal chain receipt. Admission: 20,160/1,260 exact cells, no duplicate/missing/
unexpected identities, no overlap, fixed widths/H400, same source/DOIs, native
execution and all five strict runner gates. #10081 now implements the public
companion contract. An infrastructure resume needs an immutable receipt, the
same source/config/CPU node and a classified interruption; code defects require
a corrected source/fresh ID, not resubmission. No step in FREEZEPREP executes this.

## 5. Publication export and preflight — local preparation

The release runner exports through the common exporter. If explicitly exporting
its accepted raw root, use the existing command (no overwrite of earlier custody):

```bash
uv run python scripts/tools/benchmark_publication_bundle.py export \
  --run-dir "$CAMPAIGN_ROOT" --out-dir "$ARTIFACT_ROOT/publication" \
  --bundle-name "$BUNDLE_NAME" --release-tag "$TAG" --doi "$VERSION_DOI"
uv run python scripts/tools/publication_preflight.py \
  --bundle-dir "$BUNDLE_DIR" --output "$ARTIFACT_ROOT/publication-preflight.json"
sha256sum "$BUNDLE_ARCHIVE"
```

Inputs: admitted main and companion outputs; repeat per track with distinct names.
Outputs: checksummed bundle directories/archives, inventories, strict preflight receipts.
Admission: actual files=manifest=checksums, roles/source/SNQI bound, strict
release_result/campaign-summary reconciliation, no private paths, complete chain
containment, recorded archive size/SHA, independent cold readback/preservation,
zero unresolved publication scanner findings and disclosure gate green.
`--no-require-release-reconciliation` is not release admission. Export is local
preparation; remote publication remains author-reserved.

## 6. D-062 comparator against pinned 0.0.7

```bash
uv run python scripts/analysis/compare_release_distributions.py \
  --baseline-bundle "$BASELINE_007_ARCHIVE" --successor-root "$CAMPAIGN_ROOT" \
  --successor-manifest output/release-008/main/release_identity.resolved.json \
  --successor-manifest-sha256 "$MAIN_IDENTITY_SHA256" \
  --successor-source-root "$PWD" --output-dir "$ARTIFACT_ROOT/comparator"
```

Inputs: checksum-verified immutable baseline above; exact accepted successor
rows/identity/source. Outputs: JSON/CSV/Markdown comparison, unchanged-definition
contrasts, Holm primary/separate BH exploratory family, supported-success census.
Admission: strict full census, schema/source binding, no paired-seed outcome
claim, changed definitions excluded, report the D-062 plant/world/seed caveats.
No `--diagnostic-partial` for the release; development output cannot promote it.

## 7. Tag, publish and DOI — author-reserved, delegated 2026-09-28

Only after all preceding admissions, final comparison/intake/scanner review and
explicit release authorization; same source, coordinates, notes and bundle digests:

```bash
git tag -a "$TAG" "$FREEZE_SHA" -m "Robot SF benchmark data 0.0.8"
git push origin "refs/tags/$TAG:refs/tags/$TAG"
git ls-remote origin "refs/tags/$TAG" "refs/tags/$TAG^{}"
uv run python scripts/tools/publish_camera_ready_release.py \
  --campaign-root "$CAMPAIGN_ROOT" --repo ll7/robot_sf_ll7 --tag "$TAG" \
  --expected-source-sha "$FREEZE_SHA" --create-draft --execute-upload \
  --output-json "$ARTIFACT_ROOT/github-publication-custody.json"
uv run robot-sf release zenodo upload --token-file "$TOKEN_FILE" \
  --state "$ZENODO_STATE" --manifest output/release-008/main/release_identity.resolved.json \
  "$BUNDLE_ARCHIVE"
uv run robot-sf release zenodo verify --token-file "$TOKEN_FILE" \
  --state "$ZENODO_STATE" --metadata "$ZENODO_METADATA" \
  --manifest output/release-008/main/release_identity.resolved.json
gh release edit "$TAG" --repo ll7/robot_sf_ll7 --draft=false
uv run robot-sf release zenodo publish --token-file "$TOKEN_FILE" \
  --state "$ZENODO_STATE" --metadata "$ZENODO_METADATA" \
  --manifest output/release-008/main/release_identity.resolved.json
```

Outputs: exact-source annotated tag, remotely verified dataset assets, publication
custody and published DOI. Admission: reserved state/metadata/manifest agree;
both-track publication containment admitted, versioned notes linked, GitHub-to-
Zenodo webhook disabled, remote readback matches local digests. Repeat upload
for all admitted assets in the reviewed two-track publication plan; a main-only
upload does not satisfy companion custody. No reservation reuse or second DOI.
If a tag already exists, verify target and stop on mismatch; never replace it.

## CHAIN-4 gap readback G01–G12

Read with `gh` on 2026-10-03: private-ops `docs/reviews/0.0.8/lanes/runbook_report.md`,
`docs/post_freeze_release_chain.md`, `docs/full_campaign_mint.md`, and merged
[#421](https://github.com/ll7/robot_sf_ll7-private-ops/pull/421). The historical
report is dated 2026-10-01; statuses below reconcile source changes, not new
production receipts. Access succeeded; no inaccessible gap is guessed closed.

| Gap | Current status and evidence |
|---|---|
| G01 integrated whole-roster rehearsal | Packaging completed by #10103 / D-086 on `b8d5e970…`, 672 smoke +2,016 development cells. Candidate `c979e033…` includes later #10108 hardening; exact-candidate intake/rehearsal acceptance still belongs to orchestrator. |
| G02 calibration custody | Open: acquire exact 1,344 dev 1001/1002 cells; 2,016 rehearsal rows cannot be relabelled as calibration. #10045 supplies strict freeze validator. |
| G03 v2 assets and manifest binding | #10112 binds pending acquisition/spec/assets to both templates; land reviewed successors before freeze. Real acquired anchors and independent scientific review remain post-freeze inputs. |
| G04 calibration versus frozen-source trust | #10112 requires same-source acquisition/configuration custody; private scientific trust set remains empty until independent review of actual acquired anchors and sealed ruling. |
| G05 authored-horizon mint and bounded diagnostic | Tooling fixed by merged private #421; static dev diagnostics non-dispatchable, production CPUs 32–60. Not proof production inputs are admitted. |
| G06 doorway H600 versus H400 | Fixed: #9999 authored slice schedule and #10081 strict companion acceptance; static source-pin witnesses bind H400. |
| G07 same-node two-track chain | Tooling fixed in private #421: distinct identities/checkpoints, shared source/DOI and sequential runners. Final reconciliation/preservation remain G12. |
| G08 preparation seeds | #10112/private successor selects scheduled dev1001/1002 acquisition and tracked v0_6 dev1003 smoke; historical v0_5 remains seed103. |
| G09 exact-freeze smoke/stress/staging/cold custody | #10112 selects v0_6/current keys/authored H400 and preparatory mint → smoke → final mint. Authentic same-freeze environment/stress/staging/preservation/cold receipts remain required. |
| G10 comparator end-to-end | Development path verified in #10103, correct relative source binding and release-mode refusal. Strict full sealed-census comparison still due after execution. |
| G11 DOI reservation order | Documented/fixed in private #421: reserve unpublished before identity/mint, publish last. Authentic reservation receipt remains operator input; none performed here. |
| G12 full publication chain | Public doorway-runner seam fixed by #10081. Width scientific review, two-bundle/result/projection reconciliation, scanner/cold preservation, intake and final authorization remain open; not supplied by a diagnostic mint/rehearsal. |

## Development packaging reference

D-086 identities use resolver `generate --development-rehearsal
--development-seeds 1001` (smoke) and `1001,1002,1003` (2,016-cell rehearsal),
fixed diagnostic DOIs `10.5281/zenodo.99000001` / `10.5281/zenodo.99000002`,
`development-rehearsal-<full SHA>` tag and the same D-083 authored template.
The Slurm wrapper's `ROBOT_SF_DEVELOPMENT_RUNTIME_SMOKE=1` uses fifth argument
`-`; subsequent development campaign consumes that exact-source smoke result.
These outputs retain `release_eligible: false`; comparator requires
`--diagnostic-partial`. Preserve the complete raw custody and diagnostic bundle;
they are never substitutes for steps 3–7 production admission.

## Before the freeze: #10112 source contract

[#10112](https://github.com/ll7/robot_sf_ll7/issues/10112) **blocks the freeze**.
Its scoring, smoke and ordering changes must land on main before the orchestrator
names the freeze commit. This ruling supersedes the earlier preparation audit's
classification of #10112 as a mint-only blocker. Historical manifests remain
unchanged; regenerate every identity/packet at the newly named clean source.

D-083 now declares `snqi_v2_spec` with the weights, family, pending anchor asset,
and `calibration.dev1001_1002_scheduled_acquisition.yaml`, each hash-bound.
Both v0.2 templates pin those same assets. The source anchor file remains
`pending_calibration`; loading a configuration is permitted for identity and
checkpoint preparation, but executing an unscored bound campaign refuses.
The doorway template selects its v2 campaign successor; its original v1 config
and concrete manifest remain historical bytes.

Acquisition freezes a separate artifact after the freeze, without editing the
tracked pending file or moving the named source. The release runner accepts
`--snqi-v2-anchors <artifact>` and checks the strict frozen loader, dev1001/1002
split, calibration source and exact acquisition configuration identity. Supply
`--snqi-v2-calibration-root <complete raw root>` to additionally rederive and
compare the anchors from every producer row/sidecar. Independent scientific
pins remain mandatory in production; these checks confer no scientific authority.

## Independent scientific review required before production mint

Leave these boxes unchecked here. An independent reviewer completes them after
actual acquisition; this integration and a rehearsal cannot fill the trust set.

- [ ] Review the complete dev1001/1002 14×48×2 acquisition custody, actual producer
  hashes/sidecars, native/adapter/mixed census, absence of fallback/degraded rows,
  force-source decision, schema/zero anchors, T=3, N=0.25 and real positive F/J/K p95.
- [ ] Bind the calibration source to the named freeze. Verify zero metric/runtime,
  planner/model, physics, schema and authored-budget drift between acquisition and
  campaign; keep the sealed seed commitment and the D-084 overtaking H600.
- [ ] Independently review anchor applicability to the fixed doorway width slice;
  retain its separate scientific and publication boundary (CHAIN-4 G12).
- [ ] Review the sealed evaluation ruling binding the freeze, concrete main and
  companion manifests/configs, canonical acquired anchor digest, exact sealed
  tuple and review reference. No request-created receipt can authenticate it.
- [ ] In a separate reviewed private code change, add the actual
  `REVIEWED_SCIENTIFIC_SOURCES` entry: `freeze_sha`, `review_ref`, and both
  `snqi_anchors` / `evaluation_seed_admission` sources, each with repository,
  relative path, full source commit and SHA-256. Test positive controls on the
  actual pinned Git blobs and negative controls on changed bytes/source/splits.
- [ ] Re-run production mint's strict loader and all scientific/source/custody
  gates at the exact admitted public/private revisions. The trust set is empty
  until that independently reviewed change lands; retain the refusal meanwhile.

## Rehearsing acquisition and scored packaging (D-086)

At a clean rehearsal source, use the same acquisition file on dev1001/1002,
freeze anchors into ignored output, then generate the D-086 seed-1001 preparatory
smoke and seed-1001/1002/1003 campaign identities described above. The preparatory
D-086 smoke may remain unscored and is permanently diagnostic. Before the scored
campaign wrapper, set `ROBOT_SF_SNQI_V2_ANCHORS=<artifact>` and
`ROBOT_SF_SNQI_V2_CALIBRATION_ROOT=<complete acquisition root>`; the wrapper forwards
both as explicit runner inputs. The runner revalidates acquisition custody,
computes `snqi_v2`/terms and exports the shared diagnostic bundle. Inspect the
actual 2,016 scored rows and bundle, retaining `release_eligible: false`.
Development anchors/receipts never admit a later source or a sealed campaign.

SNQI v2 enrichment retains the actual per-episode algorithm. The two reviewed
scenario-adaptive hybrid arms use their frozen ORCA branch on
`francis2023_leave_group`; they remain single configured arms in the paired
reports. Enrichment reads these declarations from the source planner config and
rejects an algorithm swap outside its declared scenario. It does not infer
permitted routing from observed rows or relax fallback/degraded checks.
