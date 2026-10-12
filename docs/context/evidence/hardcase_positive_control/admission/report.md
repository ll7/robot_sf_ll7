# Seed-1001 source-bound window admission

The observed critical-case chain passes through versioned corpus admission.
All ten registry findings were remediated under the maintainer disposition;
the registry has returned to 492 findings without adding debt. PR
[#10331](https://github.com/ll7/robot_sf_ll7/pull/10331) is ready for review and
ends at replay parity; corpus admission is outside its scope and tracked in
[#9652](https://github.com/ll7/robot_sf_ll7/issues/9652). The separate stacked PR
[#10334](https://github.com/ll7/robot_sf_ll7/pull/10334) implements the minimal
admission slice and window receipts for
[#5600](https://github.com/ll7/robot_sf_ll7/issues/5600). Neither PR is merged.

This is one observed simulator regression case. Feasibility and independent
perturbation persistence remain **unknown**. The separate robust-scenario
promotion verdict is **REJECT**, because its nine cells were not executed.
The #9652 observed-case contract permits recorded unknown feasibility and
requires a replay-verified critical outcome; it does not require that grid.
No benchmark, release, publication, real-world safety or generic checkpoint
claim is made. This positive control is not a new adversarial search discovery.

## Source identities and custody

- Source/main audit revision: `53721be3248600e08641a9816d094b605e3c93e4`.
- Replay implementation and both actual native replay manifests:
  `1baaa3eb08cfbc1866f18702eb939c1f60b400fd`.
- Admission implementation and actual first-entry receipt:
  `4fe503aadbaed2bd48dc82c58509bde71b4af2d8`.
- Scenario `classic_cross_trap_high`, development seed **1001**, analytic
  `social_force` with the release 0.0.8 planner configuration. Original source
  config hash `f330dd29077077ac`; effective planner hash `a4913be8bee6f3f3`;
  config-file SHA-256
  `9a7015a5f66b5d9ddb5d96c706d33ac8fe027bc0aee256e9ab87d2e89804f773`.
- Analytic model implementation digest
  `fe4e5b99c0a9f51ada80cfccbf1361bb4f1872c0c523e2b35ce89e94e7b32958`.
  No learned weights exist for this planner; the three implementation file
  hashes and source Git-object checks are recorded in [window_receipt.json](window_receipt.json).
- Full 600-frame source digest
  `7a73ee858ba8248969015c4f31273c014d15fd328bd97671876f9520c2bf2b34`.
  Window frames **370–389**, boundary samples 37.0–39.0 s, 20 transitions at
  0.1 s; source budget remains 600. Canonical window digest
  `37488c586d0a7fd203b9385377434516f108edb8bef011a6003a3236e135fed3`.
- Both windows contain 11 near-miss frames; first event **step 374 / 37.5 s**;
  no contact; minimum surface clearance **0.47727323760374385 m**. Position
  tolerance is **1e-9 m**, observed error **0 m**; all recorded trace channels
  also compare exactly. The window end is censored at 39 s. The source's later
  timeout at 60 s is not a window timeout termination.

The receipt binds actual compressed source and two window rows, distinct native
run manifests, map/config/model bytes, catalog entry, materialized scenario,
typed checkpoint and companions. The verifier reads those files and recomputes
parity/event facts; a receipt asserting pass is insufficient. Required source
and replay Git objects must be available. The six continuation implementation
files are unchanged after the recorded replay revision. Later updates correct custody metadata and types outside that six-file
checkpoint inventory; the retained numerical replay proofs remain applicable,
and affected tests were rerun after remediation.

## Per-stage results and exact commands

Commands below run from the admission worktree. The launcher uses the existing
shared environment at `../repo/.venv`. Only prior real
seed-1001 artifacts were read in this admission slice: **zero new episodes,
zero resets, zero search evaluations**. No sealed-seed replay/reset or frozen
artifact edits occurred.

| Stage | Result | Command / evidence |
| --- | --- | --- |
| Source provenance, detailed trace and extraction | PASS, retained | Source recipe below; [source provenance](../provenance.json), [catalog entry](../followup/packet/entry.json). |
| Materialization, two consecutive real window replays and exact parity | PASS, retained | Same source recipe; [comparison](../followup/packet/comparison.json), [previous report](../followup/report.md). |
| Source-bound window receipt and native-envelope persistence integration | PASS | Persistence command below; [v2 receipt](persistence/generated-54fdcdbc905bf047.persistence.json): exact replay PASS, critical event PASS. |
| First versioned observed-case admission | PASS | Admission command below: `admitted`, one entry, no near duplicates. |
| Refuse missing parity proof | PASS | Negative command below: exit 2, no corpus directory created; production CLI test also passed. |
| Idempotent re-admission / interrupted-index recovery | PASS | Same admission command repeated: `already_admitted`; original case bytes preserved; regression test verifies mtime/bytes and index recovery. |
| New tests and tests touching changed modules | PASS | Focused command below: **49 passed, 3 simulator-interlock skips** in 16.30 s. |
| Full benchmark comparison against main | PASS for zero introduced failures | Full command below: all **132** bad IDs/kinds match the completed main audit; **zero new**, zero missing baseline IDs. Raw pytest exit remains 1. |
| Final evidence-registry ratchet / readiness | PASS after remediation | All ten new locator/custody findings removed; existing 492 finding counts unchanged. Only derived evidence-tree inventory refreshed. Commands and proof below. |
| Robust-scenario promotion, separate from observed-case admission | FAIL / REJECT, unexecuted grid | Nine unknown cells; no cell was converted into pass or bypassed. This PR does not claim robust promotion. |

The already-finished source/replay command is retained for reproducibility and
was **not rerun** during this slice:

```bash
scripts/dev/run_worktree_shared_venv.sh --venv ../repo/.venv -- uv run python \
  docs/context/evidence/hardcase_positive_control/followup/reproduce.py \
  --output docs/context/evidence/hardcase_positive_control/followup/packet
```

For a new simulation run, choose an empty output directory; do not overwrite
the retained packet. The prior report explains source capture, lossless restore
and the original one-step goal-completion failure. The offline verifier remains:

```bash
scripts/dev/run_worktree_shared_venv.sh --venv ../repo/.venv -- uv run python \
  docs/context/evidence/hardcase_positive_control/followup/verify.py
```

Prepare the exact offline gate inputs from retained compressed records; this
performs no simulation. The existing gate-replay input preserves all nine
unknown cells. The original native source row is used without identity aliases:

```bash
python3 - <<'PY'
import gzip
from pathlib import Path
folder = Path('docs/context/evidence/hardcase_positive_control')
Path('../admission_source.jsonl').write_bytes(gzip.decompress((folder / 'source.jsonl.gz').read_bytes()))
Path('../admission_gate_replays.json').write_bytes(gzip.decompress((folder / 'followup/packet/gate_replay_results.json.gz').read_bytes()))
PY
scripts/dev/run_worktree_shared_venv.sh --venv ../repo/.venv -- uv run python \
  scripts/tools/evaluate_pipeline_persistence_gate.py \
  --candidate docs/context/evidence/hardcase_positive_control/entry.json \
  --episodes ../admission_source.jsonl \
  --config configs/analysis/issue_5600_persistence_gate.yaml \
  --output docs/context/evidence/hardcase_positive_control/admission/persistence \
  --replay-results ../admission_gate_replays.json \
  --window-receipt docs/context/evidence/hardcase_positive_control/admission/window_receipt.json \
  --receipt-root .
```

Observed persistence CLI exit 0 describes a successful evaluation, not robust
promotion: exact replay/event pass, promotion rejects unknown cells. Window
receipt support uses the existing persistence gate and its independent statuses;
v1 whole-episode input semantics remain in place.

```bash
scripts/dev/run_worktree_shared_venv.sh --venv ../repo/.venv -- uv run python \
  scripts/tools/admit_counterexample_window.py \
  --candidate docs/context/evidence/hardcase_positive_control/admission/candidate.json \
  --window-receipt docs/context/evidence/hardcase_positive_control/admission/window_receipt.json \
  --corpus docs/context/evidence/hardcase_positive_control/admission/corpus \
  --export-slice docs/context/evidence/hardcase_positive_control/admission/replay_slice.json
```

First execution admitted
`case-082cc08f6d75b6f7721974c7e52ef9cc8d4609e0754ca396e504dd4ebb14e524`.
The immediate second invocation omitted `--export-slice` and returned
`already_admitted`. Repeating against the retained corpus is intentionally
idempotent; use a new empty corpus destination to reproduce first admission.
[corpus.json](corpus/corpus.json) indexes the immutable case, and
[replay_slice.json](replay_slice.json) exports the existing canonical runner's
scenario/config/options, not a new runner.

The manual negative probe removed only `parity_proof` from the valid window
receipt, recalculated its candidate hash, and used an unused corpus destination:

```bash
scripts/dev/run_worktree_shared_venv.sh --venv ../repo/.venv -- uv run python \
  scripts/tools/admit_counterexample_window.py \
  --candidate ../admission_no_parity_candidate.json \
  --window-receipt ../admission_no_parity.json \
  --corpus ../admission_refused_corpus
```

Observed exit **2**: `REFUSED: window parity proof/schema invalid:
'parity_proof' is a required property`; no corpus was written. The portable
regression `test_refuse_real_window_without_parity_before_any_write` constructs
this mutation itself. Additional tests refuse false proof and reused run identity.

## Required test checks and baseline diagnosis

The full check was named before PR creation and run once after implementation
commit `4fe503aad`. Its processes have exited. No completed check was restarted
after the transient session errors. The same task interlock and dependency
profile used for the main audit prohibit test-driven simulator episodes/resets.
The already-executed real replay regressions belong to PR #10331.

```bash
export HARDCASE_NO_TEST_SIMULATION=1
export ROBOT_SF_PYTEST_SEED_GUARD=1
export ROBOT_SF_PYTEST_SEED_AUDIT=._focused_seed_audit.jsonl
export PYTHONPATH=../test_guard:$PWD/tests/support:$PWD
scripts/dev/run_worktree_shared_venv.sh --venv ../repo/.venv -- uv run pytest \
  tests/adversarial/test_counterexample_corpus.py \
  tests/adversarial/test_adversarial_package_imports.py \
  tests/benchmark/test_source_bound_window_receipt.py \
  tests/benchmark/test_scenario_persistence_gate.py \
  tests/benchmark/test_pipeline_persistence_gate_wiring.py \
  tests/benchmark/test_map_runner_preflight_profiles.py -q --tb=short
export ROBOT_SF_PYTEST_SEED_AUDIT=._test_seed_audit.jsonl
scripts/dev/run_worktree_shared_venv.sh --venv ../repo/.venv -- uv run pytest \
  tests/benchmark -q --tb=short --continue-on-collection-errors \
  --junitxml=../admission_benchmark.xml
```

Focused: **49 passed, 3 guarded skips**. Full directory: **12,122 passed,
119 failed, 340 skipped, 13 collection errors**, exit 1, **25m21s**. Main:
**12,116 passed, 119 failed, 336 skipped, 13 collection errors**, exit 1,
24m53s. All 132 failing/error identifiers and kinds match; six new benchmark
tests pass. Four additional skips are the separately passed real replay tests
in the parent. [benchmark_audit.json](benchmark_audit.json) retains the complete
matched identifier list, zero-introduced comparison and JUnit digest. The main
worktree stayed at its pinned revision; the finished baseline was not rerun.

| Pre-existing root cause | Failed | Collection errors | Existing tracking |
| --- | ---: | ---: | --- |
| Optional packages/profile metadata unavailable | 82 | 11 | [#7640](https://github.com/ll7/robot_sf_ll7/issues/7640) updated in the replay follow-up. |
| Collection-time partial torch module leaked through `sys.modules` | 11 | 2 | [#10332](https://github.com/ll7/robot_sf_ll7/issues/10332), filed in the replay follow-up. |
| Historical generation/source Git objects unavailable in fetched refs | 23 | 0 | [#10207](https://github.com/ll7/robot_sf_ll7/issues/10207), companion #10251; existing issue updated. |
| Intentional task simulator interlock: two multiprocessing skip-pickling errors and one child-canary assertion | 3 | 0 | Task coverage restriction, no public host/lab issue. |

The [previous grouped audit](../followup/benchmark_audit.json) contains root
cause membership and baseline exceptions. No unrelated baseline fixes or
dependency changes were made. This is a successful zero-regression comparison,
not a green raw benchmark suite.

Native-envelope regression was demonstrated RED before the fix: integer
`steps=600` masked nested trace frames. It now passes. Admission tests exercise
the production CLI with the actual retained seed-1001 artifacts, verify missing
parity refusal and preserve bytes/mtime on duplicate admission and index repair.

## Review and remaining boundaries

Ruff lint/format and local docs/evidence integrity pass. Parent evidence catalogue
registration and task-guard docstring/format fixes are documentation-only and
also live in PR #10331, so its replay fix can be reviewed independently.

The maintainer/orchestrator disposition was **REMEDIATE all ten findings; do
not baseline any**. All ten were removed, not grandfathered:

- Every config hash now has an explicit repository-relative `resolved_path` next
  to its existing `path`; the receipt schema permits it and the verifier refuses
  a disagreement between the two locators. Both native manifests resolve their
  schema/scenario/config inputs to committed paths, with sizes.
- Both uncommitted raw rows have a durable lane-host location, SHA-256 and size:
  `imech156-u:lanes/hardcase/custody_raw/window_replay_1.jsonl`
  and `window_replay_2.jsonl`, **206,932 bytes each**. Native artifact status remains
  `available`. The original native provenance validator passes both manifests.
- Exact offline regeneration is recorded in both manifests:

```bash
python3 docs/context/evidence/hardcase_positive_control/followup/restore_raw_rows.py \
  --output ../custody_raw
```

This reverses only the archived literal producer-prefix substitution and verifies
original raw SHA-256 values before writing. Both original digests are reproduced:
`1a2f954113d3a240d96dadc7c293da13008d128066217dae7bf4b9b3b4fffa50` and
`746b454f84736b729dea74ad419d2384bc3e41cac34b78ee5e7e75a170f99286`. No
simulation or reset is needed. Regeneration refuses to overwrite different bytes.

Custody-only manifest changes were rebound in the receipt, its candidate, the
first case, persistence receipt and corpus index. Case identity, original
discovery, admission commit, scenario/checkpoint bytes, all numeric trajectories
and event proof remain unchanged. Previous receipt/case custody hashes are
recorded in `admission_evidence.json`; prior committed versions remain in Git.
The normal admission API still never replaces an existing case. Its idempotency
regression passes after this explicitly authorized custody repair.

The CI-equivalent registry uses full 40-hex committed head/base identities
(`HARDCASE_HEAD` / `HARDCASE_BASE`), matching hosted frozen-base projection.
Untracked runtime lock files are not evidence-tree entries.

The prior debt baseline digest was
`7ad52af3e4328abdefb37bc89471dec0c84c089184a8d79757db1e587901aa71`.
The regenerated machine baseline has **exactly the same `findings_by_path` and
492 total findings**. No hardcase path was added to that finding map; no review
companion or waiver was added. Only inventory/generation metadata changed to
include the committed evidence tree (including the offline restorer).

```bash
HARDCASE_HEAD="$(git rev-parse HEAD)"
HARDCASE_BASE="$(git rev-parse lane/hardcase-positive-control)"
scripts/dev/run_worktree_shared_venv.sh --venv ../repo/.venv -- uv run python \
  scripts/dev/evidence_registry_ratchet.py --write-baseline --candidate-head "$HARDCASE_HEAD" --frozen-base "$HARDCASE_BASE"
scripts/dev/run_worktree_shared_venv.sh --venv ../repo/.venv -- uv run python \
  scripts/dev/evidence_registry_ratchet.py --check --candidate-head "$HARDCASE_HEAD" --frozen-base "$HARDCASE_BASE" \
  --report-output ../remediation_registry_final.json
scripts/dev/run_worktree_shared_venv.sh --venv ../repo/.venv -- uv run python \
  scripts/dev/check_docs_evidence_integrity.py --base-ref origin/main
scripts/dev/run_worktree_shared_venv.sh --venv ../repo/.venv -- uv run ruff check .
scripts/dev/run_worktree_shared_venv.sh --venv ../repo/.venv -- uv run ruff format --check .
uvx ty@0.0.58 check . --python ../typing-venv \
  --output-format gitlab --exit-zero > ../remediation_admission_pinned_ty.json
UV_PROJECT_ENVIRONMENT=../typing-venv uv run --no-sync python \
  scripts/dev/ty_advisory_ratchet.py --check --ty-output ../remediation_admission_pinned_ty.json
```

The type environment is a separate Python 3.12 environment with frozen all-extras
dependencies, matching CI. The gate pins **ty 0.0.58**; exploratory runs with the
project's newer ty or the partial simulation dependency profile were not accepted
as the CI equivalent. The accepted check passes: benchmark general findings
**1,683 ≤ 1,688**, total **6,430 ≤ 6,435** (exception-adjusted **6,427 ≤ 6,432**);
no module increases. Neither the typing baseline nor its exceptions were changed.
Typing fixes describe the existing episode metadata, mutable recipe dictionaries
and map point/triangle types. All six checkpoint-bound runtime files remain
byte-identical to the actual replay commit, so numerical replay proof stays valid.

Touched tests include every previous admission/persistence check plus the adapter,
scenario-generation and typed-episode checks affected by the parent annotation
corrections, and four new custody regression cases:

```bash
HARDCASE_NO_TEST_SIMULATION=1 ROBOT_SF_PYTEST_SEED_GUARD=1 \
ROBOT_SF_PYTEST_SEED_AUDIT=../remediation_focused_seed_audit.jsonl \
PYTHONPATH=../test_guard:$PWD/tests/support:$PWD \
scripts/dev/run_worktree_shared_venv.sh --venv ../repo/.venv -- uv run pytest \
  tests/tools/test_hardcase_window_custody.py \
  tests/adversarial/test_counterexample_corpus.py \
  tests/adversarial/test_adversarial_package_imports.py \
  tests/benchmark/test_source_bound_window_receipt.py \
  tests/benchmark/test_scenario_persistence_gate.py \
  tests/benchmark/test_pipeline_persistence_gate_wiring.py \
  tests/benchmark/test_map_runner_preflight_profiles.py \
  tests/benchmark/test_generated_scenario_replay_adapter.py \
  tests/benchmark/test_data_driven_scenario_generation.py \
  tests/benchmark/test_map_runner_episode_typed.py \
  tests/benchmark/test_map_runner_decomposition_helpers.py \
  tests/benchmark/test_generated_window_continuation.py -q --tb=short
```

Result: **156 passed, 14 simulator-interlock skips**. The new custody test failed
on all four real files before remediation (**4 failed in 0.35 s**), then passed.
Canonical byte-bound `NEEDS-REVIEW` sidecars were added with the existing
shared writer; no numeric artifact was edited to add a marker. Temporary corrupt
pytest fixtures carry justified writer exemptions and remain outside durable
evidence. The idempotency regression also reproduced the reader treating a
`.review.json` sidecar as a case (**1 failed in 2.09 s**), then passed after
skipping that known review suffix. This protects repeat admission/index recovery
with repository-standard metadata; earlier coverage had no review sidecar; the
production writer/reader are used without a test-only seam.

It protects real hash/location resolution; dropping a locator recreates the actual
registry failure; previous replay tests did not invoke the generic registry; it
uses the existing production parser with no test-only seam. The added alias
negative protects against contradicting `path`/`resolved_path`; existing path
escape tests did not cover two explicit locators. The earlier completed full
benchmark comparison is retained, not misrepresented as a new post-remediation
run; this follow-up requested the touched tests and required gates.

PR contract equivalent uses the actual published body, latest replay base and
Git-derived changed/added lists, without posting a CI comment:

```bash
GH_TOKEN="$(gh auth token)" scripts/dev/run_worktree_shared_venv.sh \
  --venv ../repo/.venv -- uv run python \
  scripts/ci/pr_contract_check.py --pr-body-file ../admission_pr_body.md \
  --pr-title 'feat(corpus): admit source-bound replay-verified critical windows' \
  --pr-number 10334 --repo ll7/robot_sf_ll7 \
  --changed-files-file ../remediation_changed_files.txt \
  --added-files-file ../remediation_added_files.txt \
  --base-ref lane/hardcase-positive-control
```

The registry, documentation, Ruff lint/format, pinned typing ratchet, PR contract
equivalent and touched tests must all pass before readiness. No new public
host/driver/lab issue, frozen edit, sealed replay/reset, agent fan-out or merge
occurs in this remediation. #6839/#5600 record the REMEDIATE outcome.

The minimal slice deliberately leaves broader #9652 historical import and
future-planner status recomputation for follow-up. Unknown feasibility and
perturbation persistence remain explicit. Only this supported analytic
social-force continuation is accepted by receipt v1; unsupported planners or
missing/mismatched artifacts fail closed. Both PRs remain open and unmerged.
The next action is review of the replay fix and then the separate admission
slice. The registry disposition is fulfilled and the local required checks pass;
review readiness does not claim domain approval or authorize a merge. No further campaign
or robust promotion is authorized by this evidence.



Review follow-up (2026-10-12): this branch was rebased onto replay review fix
79981142be80cf99fcb86953b87319f59bc2ea20. The parent pins this source-bound
regression's LLVM CPU target before importing Numba kernels; its retained
source/trace/checkpoint bytes and strict oracle are unchanged. New intentional
custody annotations change only native manifest bytes, so their hashes were
rebound in the receipt, candidate, embedded case, scoped persistence record,
index and checksum/review sidecars. Case identity, discovery and original
admission code commit remain unchanged. Documentation commands now use lane
relative paths. No new episode or reset was needed for corpus revalidation.
All inherited six checkpoint-bound runtime files still match the actual
replay-producing commit. Final heads and hosted results are in the lane report.

TERMINAL=AUTHOR_DECISION_REQUIRED
