# Issue #9653 Planner–Falsifier Loop Contract

`robot_sf.adversarial.coevolution` owns the bounded round state machine for
alternating planner optimization, challenge evaluation, falsification, and
counterexample admission. It wraps the canonical planner optimizer and
`robot_sf.adversarial.search.run_adversarial_search` through normalized adapters;
it does not replace those owners.

## Current implementation boundary

The coordinator and deterministic fixtures are implemented. The optimizer,
challenge-evaluator, and search adapters are explicit callables. Replay,
feasibility/admissibility, and corpus admission are also explicit callables and
must be connected only through reviewed upstream contracts. This branch has no
default production adapter set or command-line runner yet, and it has not run a
simulator, search, replay, or empirical co-evolution experiment. The tests are
implementation-integrity evidence only.

The normalized adapter result shapes and strict checks are in
[`coevolution.py`](../../robot_sf/adversarial/coevolution.py). The config loader
requires `adversarial_coevolution_config.v1`, at least two rounds, explicit
optimizer/falsification budgets, base seeds, a supported falsification sampler,
held-out IDs, and the named optimizer/template/search-space inputs. Every round
derives and records the exact sampler, seeds, case IDs, budgets, and source
revision before invoking an adapter; the search result must echo the selected
sampler. Files referenced by those configs, such as map or route assets, must
also be listed in `additional_input_files` when their digests are not already
recorded by the corresponding owner manifest.

## Evidence and admission rules

Each owner phase writes a separate machine-readable result with an input digest,
output digest, and file digest. Paired round/run manifest updates first persist
a write-ahead transaction containing both intended records and their digests;
resume completes that transaction if the process stopped between the two atomic
replacements. Resume then verifies the config, named input files, source
revision, non-output working-tree bytes, Python/platform identity, and completed
phase artifacts, including the selected planner config path and bytes.
Completed phases are reused. A phase interrupted while marked `running`, or
marked `failed`, is not automatically repeated because that could duplicate
simulator work; its diagnostic remains in the run manifest. Nested scenario
and evidence payloads in regression cases are copied and recursively frozen
before adapters receive the round request; `to_json()` returns a detached copy.

A discovered case reaches the next round only when the search row is an
evaluated target-planner failure with normal execution, replay matches exactly,
feasibility is empirically demonstrated, and the admissibility adapter says
admissible. Invalid, fallback, degraded, replay-mismatched, infeasible, and
unknown rows remain separately recorded and are not admitted. The corpus
adapter must preserve the coordinator's stable case ID.

No-new-case stopping waits until the configured minimum round count and means
only that no new admissible counterexample was found under the recorded finite
budget. Optimization plateau and maximum-round budget exhaustion have separate
stop reasons. None establishes that no counterexample exists, mathematical
feasibility, global optimality, or real-world safety.

## Verification

Run the fixture contract suite with:

```bash
scripts/dev/run_worktree_shared_venv.sh -- uv run pytest tests/adversarial/test_coevolution.py -q
```

It covers round-one admission flowing into round-two regression evaluation,
minimum-two-round no-discovery behavior, distinct invalid/unknown/replay-mismatch/
fallback/degraded/failed rows, sampler identity validation, immutable nested
regression payloads, optimizer improvement and plateau, round-budget exhaustion,
infrastructure failure, crash-window manifest recovery, and resume digest
validation without re-running completed phases. These fixtures do not establish
planner or search performance.
