# Issue 1181: `ml-intern` bounded experiment-assistant assessment

Date: 2026-05-13
Issue: #1181
Follow-up: #1191

## Decision

`huggingface/ml-intern` is a plausible **bounded experiment assistant** for `robot_sf_ll7`, but it
should **not** become the repository runtime or replace the existing local/HPC/SLURM workflows.

Update on 2026-05-15 for #1191: do **not** spend API budget on an `ml-intern` runtime smoke for the
current Robot SF workflow. The incremental value of the CLI is small here because the repository
already has the core execution discipline through Codex skills, context notes, SLURM-aware local
machine policy, proof-first validation, and GitHub workflow. Treat `ml-intern` runtime execution as
optional/deferred, while extracting its useful workflow concepts into the Codex-native workflow.

Recommended near-term use:

- repo comprehension after reading the Robot SF context stack,
- validation-plan drafting,
- config/path scaffolding,
- bounded local smoke execution,
- and a separate cost/privacy decision before any remote sandbox, HF Job, or paid CLI runtime check.

Not recommended in this issue:

- long PPO or benchmark campaigns,
- paper-facing benchmark execution,
- CARLA/Unreal-heavy runs,
- SLURM replacement,
- or any workflow that treats fallback/degraded execution as valid benchmark evidence.

## Issue #1191 Outcome: Extract Concepts, Defer Runtime Smoke

Issue #1191 was initially scoped as an `imech192` SLURM setup-and-submit smoke for `ml-intern`.
During the 2026-05-15 audit, the setup check found:

- `ml-intern` was not installed in the worktree environment,
- no explicit model credential or local OpenAI-compatible endpoint was present,
- a Codex/ChatGPT subscription is not a substitute for `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
  `HF_TOKEN`, or a local endpoint credential,
- and Robot SF already has the valuable workflow primitives that `ml-intern` would be asked to
  demonstrate.

Recommendation: keep the `ml-intern` CLI out of the required Robot SF workflow for now. Use Codex
and repo-local skills to apply the useful ideas directly. A future `ml-intern` smoke remains
reasonable only for a narrow Hugging Face-specific task where its native model, dataset, Space, or
HF Jobs context provides a clear advantage over the existing workflow.

## Extracted Practices to Keep

The useful `ml-intern` concepts are workflow controls, not the specific executable:

1. **No-edit planning pass**
   - Start uncertain agent tasks by reading the context stack and proposing a bounded proof ladder.
   - For Robot SF, the first prompt should name the exact files to read and require the assistant to
     stop after a plan unless execution is explicitly in scope.
2. **Budgeted agent runs**
   - Before exploratory work, declare allowed commands, forbidden commands, max iterations or time
     budget, stop condition, and cost/risk class.
   - Avoid broad "improve this repo" prompts against paid or trace-uploading external tools.
3. **Trace/privacy defaults**
   - Do not paste secrets, unpublished benchmark interpretations, private review text, or raw
     internal logs into external agent tools.
   - Disable trace upload where the tool supports it, especially for pre-publication or
     benchmark-facing work.
4. **One-smoke-before-campaign rule**
   - New assistant workflows, remote sandboxes, and experiment runners must prove one small command
     first, then fail closed or proceed with an explicit follow-up.
   - A fallback, degraded, or broadened run is not success evidence.
5. **Optional HF-specific lane**
   - Reconsider `ml-intern` for Hugging Face-heavy work such as model, dataset, Space, or HF Jobs
     exploration.
   - Keep normal Robot SF issue execution on Codex plus repo-local skills unless the HF-specific
     value is concrete.

## Why this fits Robot SF

Robot SF already has a strong config-first and proof-first workflow:

- repo guidance and execution policy live in `AGENTS.md` and `docs/dev_guide.md`,
- issue-specific durable notes live under `docs/context/`,
- examples expose small headless proof paths in `examples/README.md`,
- PPO training has a canonical config-driven entrypoint in `scripts/training/train_ppo.py`,
- and benchmark/training claims already require explicit validation rather than chat-only reasoning.

That makes `ml-intern` most useful as an assistant that reads repo context first and proposes or
executes **small, reviewable** steps against existing commands, rather than inventing a parallel
workflow.

## Upstream contract checked on 2026-05-13

Observed from the upstream `huggingface/ml-intern` README / PyPI page:

- install path is repo-local and tool-based:
  - `git clone ...`
  - `uv sync`
  - `uv tool install -e .`
- CLI supports:
  - interactive mode: `ml-intern`
  - headless mode: `ml-intern "prompt"`
- supported model routing includes Anthropic, OpenAI, HF-router-backed models, and local
  OpenAI-compatible endpoints (`ollama/`, `vllm/`, `lm_studio/`, `llamacpp/`)
- default tool runtime is **local filesystem** (`bash`, `read`, `write`, `edit`)
- remote sandbox tooling is opt-in via `--sandbox-tools` and requires `HF_TOKEN`
- sandbox mode is intended for creating/replacing HF Space sandboxes and for testing remotely before
  larger HF Jobs
- traces are auto-uploaded by default to a **private** personal Hugging Face dataset
  (`{hf_user}/ml-intern-sessions`)
- trace upload can be disabled with:

```json
{ "share_traces": false }
```

## Trace / data-handling recommendation

The default private trace upload is still a **real upload**, so unpublished research hypotheses,
private experiment notes, internal review text, proprietary logs, or unredacted benchmark findings
should not be sent to `ml-intern` casually.

For Robot SF, the safe default is:

1. use repo-local prompts that reference tracked files and commands,
2. avoid pasting unpublished result narratives or secrets,
3. prefer sanitized prompts for assessment/planning,
4. disable trace uploads explicitly when handling sensitive or pre-publication material.

This matters more for Robot SF than for a generic toy repo because benchmark wording, promotion
claims, and artifact provenance are part of correctness.

## Recommended Robot SF use cases

Good first uses:

- “read these repo files and propose a minimal validation plan”
- “check whether this config/command path exists and summarize the workflow”
- “review a PPO config and suggest bounded dry-run proof commands”
- “compare two local docs/config surfaces and point out contract mismatches”

Possible later use, but only after local proof and a new cost/privacy decision:

- one remote sandbox smoke for a bounded training or validation command,
- one explicitly scoped HF Job follow-up with pinned install steps, timeout, hardware, and durable
  artifact routing.

## Non-goals and red lines

Do **not** treat `ml-intern` as:

- the canonical training launcher,
- a substitute for `scripts/dev/` readiness gates,
- a reason to skip `AGENTS.md` / `docs/dev_guide.md`,
- a new dependency to add to `pyproject.toml`,
- or a benchmark execution path whose fallback/degraded results are acceptable evidence.

If a future remote run cannot preserve repo-tracked configs, explicit artifact persistence, and
proof-first validation, it should stop rather than broadening the claim.

## Verified local proof ladder

The first validation ladder should stay local and bounded:

1. **Repo comprehension only**
   - Read:
     - `AGENTS.md`
     - `docs/ai/repo_overview.md`
     - `examples/README.md`
     - `scripts/training/train_ppo.py`
   - Ask for a minimal validation plan before any edits or long commands.
2. **One headless quickstart candidate**
   - `uv run python examples/quickstart/01_basic_robot.py`
3. **One targeted pytest candidate**
   - `uv run pytest tests/integration/test_train_expert_ppo.py -q`
4. **One PPO dry-run candidate**
   - `uv run python scripts/training/train_ppo.py --config configs/training/ppo/expert_ppo_issue_576_br06_v3_15m_all_maps_randomized.yaml --dry-run --log-level WARNING`

The point of this ladder is not to outsource repository judgment. For current Robot SF work, this
ladder is better used as a Codex-native pattern. It should check that any assistant stays inside the
existing contracts while helping with bounded work.

## First-use prompt for Robot SF

Use a no-edit prompt first:

```text
You are assessing the Robot SF repository as a bounded experiment assistant.

Before proposing any command, read these files first:
- AGENTS.md
- docs/ai/repo_overview.md
- examples/README.md
- scripts/training/train_ppo.py
- docs/training/ppo_training_workflow.md

Constraints:
- Do not edit files.
- Do not propose long training, SLURM jobs, CARLA/Unreal runs, or paper-facing benchmark campaigns.
- Do not add dependencies.
- Treat fallback or degraded benchmark behavior as failure, not success.
- Prefer existing repo commands and configs.

Task:
1. Summarize the repository's execution rules relevant to a bounded assistant.
2. Propose a minimal proof ladder using:
   - one headless quickstart command,
   - one targeted pytest command,
   - one PPO --dry-run command.
3. For each command, explain why it is low risk and what evidence it would produce.
4. Call out any trace/privacy risk from using ml-intern on this repository.
5. Stop after the plan; do not run anything.
```

## Verified local paths referenced by this assessment

Checked on this branch:

- `AGENTS.md`
- `docs/ai/repo_overview.md`
- `docs/context/README.md`
- `examples/README.md`
- `examples/quickstart/01_basic_robot.py`
- `scripts/training/train_ppo.py`
- `docs/training/ppo_training_workflow.md`
- `configs/training/ppo/expert_ppo_issue_576_br06_v3_15m_all_maps_randomized.yaml`
- `tests/integration/test_train_expert_ppo.py`

## Validation

Commands run on this branch:

```bash
source .venv/bin/activate
uv run python examples/quickstart/01_basic_robot.py
uv run pytest tests/integration/test_train_expert_ppo.py -q
uv run python scripts/training/train_ppo.py \
  --config configs/training/ppo/expert_ppo_issue_576_br06_v3_15m_all_maps_randomized.yaml \
  --dry-run \
  --log-level WARNING
BASE_REF=origin/main PYTEST_NUM_WORKERS=8 scripts/dev/pr_ready_check.sh
```

Observed outcomes:

- `examples/quickstart/01_basic_robot.py` completed successfully and produced the expected headless
  environment reset plus 10-step rollout output.
- `uv run pytest tests/integration/test_train_expert_ppo.py -q` passed: `43 passed in 22.72s`.
- `train_ppo.py --dry-run` exited successfully against the documented canonical PPO config.
- `BASE_REF=origin/main PYTEST_NUM_WORKERS=8 scripts/dev/pr_ready_check.sh` passed on this branch:
  `3435 passed, 17 skipped, 3 warnings in 255.63s`.
- The quickstart and dry-run both surfaced a known SVG obstacle repair warning for
  `robot_sf/maps/uni_campus_big.svg`; this did not block the bounded assessment and is exactly the
  kind of real repository signal a bounded assistant should report rather than hide.
- The PPO dry-run also surfaced the expected cadence/seed warnings from the training config:
  deprecated `evaluation.frequency_episodes` is ignored in favor of `evaluation.step_schedule`, and
  `randomize_seeds` causes the provided training seeds to be ignored.

## Follow-up boundary

The #1191 decision supersedes the earlier default of running a follow-up `ml-intern` smoke. The
current follow-up boundary is:

- Codex-native extraction is in scope for #1191.
- `ml-intern` runtime execution is optional/deferred, not required.
- A future runtime smoke should be opened only when there is a concrete Hugging Face-specific value
  proposition and an explicit cost/privacy budget decision.

If a future runtime smoke is approved, it should be a separate issue with:

- pinned install/setup steps,
- explicit hardware/runtime assumptions,
- timeout and failure handling,
- durable artifact routing,
- and an explicit statement that it does not replace the canonical local/SLURM workflow.

Follow-up tracker: #1191.
