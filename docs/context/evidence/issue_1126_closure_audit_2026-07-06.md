# Issue #1126 Closure Audit

Date: 2026-07-06
Issue: <https://github.com/ll7/robot_sf_ll7/issues/1126>
Parent: <https://github.com/ll7/robot_sf_ll7/issues/1091>
Context note: [`issue_1126_sdd_curation_preflight.md`](../issue_1126_sdd_curation_preflight.md)

## Purpose

Closure audit for #1126 (`bench: curate first real SDD-derived benchmark scenario set`). This note
maps each acceptance criterion to evidence, records freshly reproduced fail-closed validation, and
states the closure decision. It asserts only data-preflight / support-tooling readiness: it does
**not** stage real Stanford Drone Dataset (SDD) annotations, run the importer against real data, run
a benchmark campaign, or promote any model/benchmark/paper claim.

## Closure decision: KEEP OPEN (blocked on external data)

#1126 stays open with label `state:blocked-external-input`. The **agent-executable, fixture-backed
slice** — the curation-step readiness gate, the fail-closed evidence contract, and the metadata-only
decision packet — is complete and covered by tests. The **remaining acceptance criteria require a
real licensed SDD annotation file staged locally** (bring-your-own, per the 2026-06-22 issue-audit
reframe and the 2026-07-05 gate update). No such file exists at any approved local source root in
this checkout, so real scene selection, real import, and the real smoke run cannot be performed and
must not be faked with fixtures.

Per the repository COMPLETE-FIRST rule an issue whose only remainder is a *compute* run counts as
complete; #1126's remainder is instead an **external raw-data staging** gate (licensed SDD +
checksum/license provenance), which is explicitly excluded from agent scope. It therefore stays open,
not closed.

## Authoritative scope

The issue body's `agent-exec-spec:v1` block (appended 2026-06-20) is the authoritative agent scoping
and overrides older body wording. It defines the agent-executable slice as: build/verify the
curation logic on importer fixtures; ensure missing SDD fails closed (never an implied
dataset-backed result); keep `proxy_schema_smoke` strictly distinct from `dataset_backed_prior`. It
marks real SDD annotation staging as `Blocked-until` #1497 (validated by #2413). The 2026-07-05
maintainer comment confirms the identical residual: "Remains open: BYO real SDD annotation staging
(raw-data gate + checksum/license provenance) before benchmark-ready promotion."

## Acceptance criteria → evidence

| Acceptance criterion (issue body) | Status | Evidence |
| --- | --- | --- |
| #1497 has staged official SDD annotations locally, or recorded a clear access/provenance failure that keeps this issue blocked | Blocked (honest fail-closed) | No SDD source at `output/StanfordDroneDataset`, `output/stanford_drone_dataset`, `third_party/sdd`, `third_party/stanford_drone_dataset`, `output/external_data/sdd`. `manage_external_data.py list` reports SDD `Missing local source path` with the license-gated manual acquisition instructions (reproduced below). |
| One scene/video selected with a recorded deterministic selection rule | Deferred (needs real data) | Deterministic selection rule implemented + fixture-tested: `probe_annotation_file` requires ≥1 track with ≥`min_track_points` usable `label` points after `lost`/label filtering (`sdd_curation_preflight.py`; PR #3765). Concrete scene identity cannot be recorded until BYO data is staged. |
| Import command, source identity, source checksums, license/source URL, and scale assumptions recorded | Met for the *handoff contract* (this PR); real values deferred | The decision packet (PR #4564) records the import command, dataset id, and label; **this PR fixes the generated import command so it is runnable and now records the `meters_per_pixel` scale assumption** (see "Contract fix" below). Real checksums/URL/scale values are filled at BYO staging time via `manage_external_data.py` + the packet's `--decision-meters-per-pixel`. |
| Generated scenario/map artifacts pass repository loading/parser validation | Deferred (needs real data) | Importer `import_sdd_scenarios.py` (#1091) and its loader are unit-tested on fixtures; a real generated artifact requires staged data. |
| At least one representative smoke run succeeds or output is explicitly rejected with reasons | Deferred (needs real data) | The decision packet's `required_next_commands.smoke_validation` names the exact post-import smoke step; it cannot execute without a real generated scenario. |
| Documentation states `benchmark_ready` vs `exploratory_only` | Met for the gate | The preflight fails closed to `proxy_schema_smoke` / `output_classification: blocked` while unstaged and only reaches `benchmark_candidate` when `dataset_backed` (PR #3765); the final `benchmark_ready` vs `exploratory_only` call is a documented post-smoke decision. |
| Only small reviewable artifacts or durable pointers committed | Met | No raw SDD or bulky output committed; decision packet is written to git-ignored `output/sdd_curation/issue_1126/`. `raw_data_policy.raw_sdd_committed = False`. |

### Contract fix delivered by this PR

Auditing the #4564 decision packet surfaced a real defect: its generated `import` handoff command
was **not runnable**. It used `--annotation`/`--output-dir` and omitted the *required*
`--meters-per-pixel`, while the canonical importer requires `--annotations`, `--out-dir`, and
`--meters-per-pixel`. A curator copy-pasting it would hit argparse exit `2` before touching data,
and the packet recorded no scale assumption at all (an explicit acceptance-criterion item).

This PR fixes `build_decision_packet` to emit `--annotations`/`--out-dir`/`--meters-per-pixel`, adds
a `--decision-meters-per-pixel` CLI flag, records `meters_per_pixel` in `curation_parameters`, and
emits a `<meters-per-pixel>` placeholder when the scale is unset (scene-specific until BYO staging).
New regression tests parse the generated command with the importer's own parser so it cannot drift
again.

### Contributing merged PRs

| PR | Commit | Merged | Contribution |
| --- | --- | --- | --- |
| #1091 (#1127) | `9aefd352b` | — | SDD trajectory scenario importer `scripts/tools/import_sdd_scenarios.py` |
| #3765 | `95b913a13` | 2026-06-27 | Fail-closed SDD curation readiness preflight (`sdd_curation_preflight.py` + tests) |
| #4564 | `87afce809` | 2026-07-05 | `build_decision_packet()` + `--write-decision-packet`/`--decision-*` flags (metadata-only handoff) |
| this PR | — | — | Fix the packet's import command to be importer-runnable; record `meters_per_pixel` scale |

Related staging owners (separate issues): #1497 BYO staging preflight (`87afce809`... see
`manage_external_data.py`), #2413 manifest validation, #3473 staging consolidation, #2657 staging
recipe.

## Reproduced validation (2026-07-06, worktree from `origin/main` @ `405eb5b5a`)

```bash
scripts/dev/run_worktree_shared_venv.sh -- python scripts/tools/import_sdd_scenarios.py --help
# exit 0

scripts/dev/run_worktree_shared_venv.sh -- python scripts/tools/manage_external_data.py list
# exit 0; sdd -> "Missing local source path" + license-gated manual acquisition instructions

scripts/dev/run_worktree_shared_venv.sh -- python scripts/tools/sdd_curation_preflight.py
# staging mode: proxy_schema_smoke; dataset_backed: False; benchmark promotion: False; output class: blocked

scripts/dev/run_worktree_shared_venv.sh -- python scripts/tools/sdd_curation_preflight.py --require-benchmark-ready
# exit 3 (fail closed) — refuses to treat the blocked state as benchmark-ready

scripts/dev/run_worktree_shared_venv.sh -- python scripts/tools/sdd_curation_preflight.py \
  --write-decision-packet output/sdd_curation/issue_1126/decision_packet.json \
  --decision-meters-per-pixel 0.0417
# exit 0; packet import command:
#   uv run python scripts/tools/import_sdd_scenarios.py --annotations '<staged-sdd>/.../annotations.txt' \
#   --out-dir output/sdd_curation/issue_1126 --dataset-id sdd_first_real_candidate --label Pedestrian \
#   --meters-per-pixel 0.0417 --min-track-points 8 --max-pedestrians 4   (parses against the importer)

scripts/dev/run_worktree_shared_venv.sh -- python -m pytest tests/tools/test_sdd_curation_preflight.py -q
# 15 passed

scripts/dev/run_worktree_shared_venv.sh -- python -m pytest tests/ -k "sdd or import_scenarios" -q
# 75 passed
```

## Remaining criteria checklist (unblock gate)

Real curation starts only after a contributor stages a licensed SDD copy. Remaining items, all
requiring the external raw-data gate:

- [ ] Stage a licensed SDD annotation directory via `manage_external_data.py stage sdd --source <path>`
      with checksum + license provenance so `resolve_sdd_scenario_prior_mode` reports `dataset_backed_prior`.
- [ ] Select one scene/video (deterministic rule already implemented) and record its identity, source
      URL/license, checksums, and calibrated `--decision-meters-per-pixel`.
- [ ] Run the packet's (now-runnable) importer command and load the generated scenario/map.
- [ ] Run one CPU smoke path; mark `benchmark_ready` or `exploratory_only`, or reject with reasons.

## Out of scope here

No SDD download/ingestion, no real curation run, no benchmark campaign, no SLURM/GPU submission, no
paper/dissertation claim edits.
