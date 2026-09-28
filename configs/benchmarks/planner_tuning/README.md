# Planner development split v1 — issue #9748

Author decision: [28 September 2026](https://github.com/ll7/robot_sf_ll7/issues/9748#issuecomment-5864715937).
The protocol uses **pre-registered development-only scenario-parameter variants, using disjoint
seeds and identities not present in the release matrix.** It does not use newly edited SVG
spawn/goal geometry and does not claim an independent scenario domain.

The split contract is [`../planner_development_split_v1.yaml`](../planner_development_split_v1.yaml).
It is a reusable seed/scenario contract, **not a runnable campaign or a planner roster**. A tuning
campaign must explicitly copy its `scenario_matrix` and `seed_policy`; do not rely on runner defaults.
Its scenarios can also be loaded directly with `robot_sf.training.scenario_loader.load_scenarios`.

## Frozen variants

Every row has seeds **1001–1030**, `metadata.development_only: true`, and a distinct name:

| Development identity | Source scenario | Parameter changes |
| --- | --- | --- |
| `dev_v1__classic_doorway_medium` | `classic_doorway_medium` | `ped_density: 0.065`, `route_spawn_jitter_frac: 0.30` |
| `dev_v1__classic_group_crossing_medium` | `classic_group_crossing_medium` | `ped_density: 0.10` |
| `dev_v1__francis2023_perpendicular_traffic` | `francis2023_perpendicular_traffic` | `ped_density: 0.12` |
| `dev_v1__francis2023_crowd_navigation` | `francis2023_crowd_navigation` | `ped_density: 0.10` |

All other simulation and robot settings and the existing map paths are retained from the source
rows pinned in the freeze record. The original row names are provenance metadata, not runtime
aliases. Old plausibility measurements and verification claims are deliberately not copied to the
changed variants. Planner-specific overrides belong to the development planner snapshot and must
use the development names when intended to apply to these rows; never rename a runtime row back
to a release identity to make an override match.

The definitions live here, outside `configs/scenarios/` discovery and release includes. No release
matrix, release seed registry, historical artifact, SVG, planner implementation, or release config
is edited by this change.

## Freeze before tuning

[`../planner_development_freeze_v1.json`](../planner_development_freeze_v1.json) records the source
commit, author decision, exact split/scenario file SHA-256 values, and each raw scenario row's
SHA-256. Row hashing is UTF-8 JSON with sorted keys, compact comma/colon separators, no ASCII
escaping, and no NaN values. Each reused map is pinned by its Git blob SHA-1, computed over
`b"blob " + ascii(byte_length) + b"\0" + file_bytes`; this is a Git identity, not a SHA-256.

Commit, review, and merge these materialized definitions **before the first tuning trial**. The
freeze record does not assert that review has already happened. There is no automatic re-freeze
command: changed definitions or map bytes fail the guard and need a separately approved protocol
version, not updated hashes to make an old experiment pass. Keep the freeze files clean during
execution. The initial `tuning_log_v1.json` has no trials and asserts no planner result or final
planner freeze.

## Required acceptance check

Run from the repository root before tuning and again before accepting the tuning log:

```bash
scripts/dev/run_worktree_shared_venv.sh -- uv run python scripts/validation/check_planner_tuning_split.py
```

Store all new tuning configs and structured ledgers under `configs/benchmarks/planner_tuning/` so
the default check and regression test discover them. Files elsewhere must be passed explicitly:

```bash
scripts/dev/run_worktree_shared_venv.sh -- uv run python scripts/validation/check_planner_tuning_split.py \
  --config "$TUNING_CONFIG" --log "$TUNING_LOG"
```

The command fails on release seeds **111–140**, any seed outside the registered development range,
implicit/unknown seed policies, changed freeze/map bytes, release scenario selection, and malformed
or incomplete logs. It follows config/include/base-config references and resolves the selected
member of named seed sets; unused sets in a shared registry are not tuning selections. Scalar,
list, numeric-string and inclusive range seed declarations are checked, as are seed references in
commands and comments. Opaque command-line seed-set indirection is rejected; declare it as a
structured `seed_policy` instead. YAML duplicate keys and unresolved/ambiguous file references also
fail. Scenario-definition overrides are forbidden; planner candidate overrides are separate.

Logs are JSON/YAML `planner-tuning-log.v1` ledgers with `split`, `scenario_file_sha256` and a `trials`
list, or JSONL trial records, each with the same split and definition hash. Every trial records a
unique `trial_id`, explicit `seeds`, development `scenario_ids`, the immutable planner snapshot's
`config_path` and `config_sha256`, and a nonempty `changes` description of what was tried. Keep the
config/include closure immutable too; retain run commands and results as extra trial fields. Logs
and configs referencing release seeds fail even if labelled diagnostic. Unstructured console logs
are not accepted as the provenance ledger. A check can validate supplied artifacts, not discover
unrecorded runs or prove that a human never inspected a release result.

Release exclusion uses the canonical scenario loader, including its include/selection/override
semantics. It checks the combined classic/Francis matrices and `scenario.matrix_path` in
benchmark-release manifests. Pass each additional prospective matrix via `--release-matrix`.
Adding a development identity or development marker to a checked release matrix fails.

## Tuning, final planner freeze, and evaluation

All modified-planner tuning, including hybrid v4, socnav_sampling v2 and social force v2, uses this
split. The whole schedule is registered; smaller diagnostic subsets must be recorded explicitly in
the log and must remain within the four identities and seeds 1001–1030. Correctness can also be
justified by independent unit/oracle cases, never by choosing changes from release-seed results.

After tuning and before evaluation, commit the final planner configuration and its resolved
config/include closure hashes, source revision, and complete tuning log/hash. This later **planner
freeze** is distinct from the pre-tuning **scenario-definition freeze** implemented here. Run the
held-out release seeds only after that freeze, and make no further planner changes for that same
release after evaluation. A failing guard is a stop, not permission to relabel or omit a trial.

PR #9747's prior runs on release seeds are **pre-protocol diagnostics**, excluded from tuning
provenance and reported benchmark evidence. Their historical files remain unchanged. This split
combines parameter differences, disjoint seeds, and pre-registration; it is not proof of broad
generalization, planner superiority, or deployment safety. See the
[0.0.8 protocol release note](../../../docs/releases/0.0.8_planner_development_protocol.md).
