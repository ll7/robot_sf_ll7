# Planner optimizer run contract v1

`planner_optimizer.v1` is a bounded configuration-search workflow over one
registered planner. It composes the canonical candidate loader and evaluator
from `scripts/validation/run_policy_search_candidate.py`; it does not optimize
scenario parameters or create a planner implementation.

## Run command

```bash
uv run python scripts/validation/optimize_planner_config.py \
  --config configs/policy_search/planner_optimizer_issue9650.yaml \
  --output-dir docs/context/policy_search/validation/issue_9650_planner_optimizer_pilot_v1
```

The output directory must be new or empty. A lock and a `status: running`
manifest are written before evaluations begin. If execution is interrupted,
keep the partial trial rows and evaluator outputs; the same directory refuses a
second run so an operator cannot silently repeat simulations. Start a new
`run_id` only as a deliberate new experiment.

## Selection semantics

The optimizer records and selects by this ordered tuple:

1. fraction of expected episodes with valid, complete, non-fallback execution;
2. collision-free fraction among valid episodes;
3. near-miss-free fraction among valid episodes using the policy-search
   surface-clearance semantics;
4. task-completion fraction among valid episodes;
5. lower mean successful `time_to_goal_ideal_ratio` (then
   `time_to_goal_norm_success_only` when the ideal ratio is absent);
6. lower mean `ped_force_q95` (then `force_q95` when the pedestrian metric is
   absent).

Invalid evaluations are preserved and do not receive a fabricated successful
score. Missing near-miss, efficiency, or comfort observations remain `null`; per-metric
coverage counts are recorded. An efficiency mean requires a value for every
successful episode; a comfort mean requires a value for every valid episode.
Missing values are given a tied low sentinel only at the Optuna sampler
boundary so the multiobjective sampler can continue. The durable selection
tuple keeps the missing value as `null`. No weighted scalar is reported.

Random and Optuna TPE receive the same number of training evaluations over the
same fixed scenario/seed identities. Candidate selection uses only training
results. The baseline and selected candidate are evaluated on the held-out
suite only after search selection is frozen. Scenario/seed overlap is rejected
before evaluation.

## Durable outputs

Every run writes:

- `run_manifest.json` (`planner_optimizer_run.v1`): source revision, baseline,
  config and suite digests, explicit parameter bounds, objective order, equal
  budgets, seeds, episode identities, train-only method results, selected
  candidate, held-out comparison, and artifact pointers;
- `trials.jsonl`: one row for every Random/TPE proposal, including invalid
  evaluations, sampled values, metric components, runtime, error reason, and
  episode-record pointer;
- `train_baseline.json`, `heldout_comparison.json`, `report.md`;
- `trials/<method>/trial_NNN/` and `heldout/`: canonical evaluator JSONL,
  normalized `stage_summary.json`, and effective algorithm config;
- `best_candidate.yaml`: a regular policy-search candidate YAML with
  `algo`, `base_config_path`, and `params`;
- `candidate_registry.yaml`: a minimal registry that points the existing
  policy-search CLI at that export.

The #9653 consumer should treat the manifest and exported YAML/registry as the
integration boundary. It can read `selected.candidate_name`, pass the export
registry to the existing candidate runner, use `selected_candidate_config` as
the planner artifact, and add admitted counterexamples to a later training
suite. A completed search does not imply an improvement; compare the stored
lexicographic tuple and frozen held-out result.

## Evidence boundary

`valid_episode_fraction` is an execution/admissibility observation for the
configured benchmark records. It is not a mathematical feasibility oracle.
Finite-budget failure to improve or failure to find a better config is a valid
null result. No planner result establishes real-world safety or global
optimality.
