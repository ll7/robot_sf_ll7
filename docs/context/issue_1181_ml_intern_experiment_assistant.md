# Issue 1181: `ml-intern` bounded experiment-assistant assessment

Date: 2026-05-13
Issue: #1181

## Decision

`huggingface/ml-intern` is a plausible **bounded experiment assistant** for `robot_sf_ll7`, but it
should **not** become the repository runtime or replace the existing local/HPC/SLURM workflows.

Recommended near-term use:

- repo comprehension after reading the Robot SF context stack,
- validation-plan drafting,
- config/path scaffolding,
- bounded local smoke execution,
- and, only after local proof, a separate follow-up issue for one small remote sandbox/HF Job check.

Not recommended in this issue:

- long PPO or benchmark campaigns,
- paper-facing benchmark execution,
- CARLA/Unreal-heavy runs,
- SLURM replacement,
- or any workflow that treats fallback/degraded execution as valid benchmark evidence.

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

Possible later use, but only after local proof:

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

The point of this ladder is not to outsource repository judgment. It is to check whether
`ml-intern` can stay inside Robot SF's existing contracts while helping with bounded work.

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
  `3434 passed, 18 skipped, 3 warnings in 255.63s`.
- The quickstart and dry-run both surfaced a known SVG obstacle repair warning for
  `robot_sf/maps/uni_campus_big.svg`; this did not block the bounded assessment and is exactly the
  kind of real repository signal a bounded assistant should report rather than hide.
- The PPO dry-run also surfaced the expected cadence/seed warnings from the training config:
  deprecated `evaluation.frequency_episodes` is ignored in favor of `evaluation.step_schedule`, and
  `randomize_seeds` causes the provided training seeds to be ignored.

## Follow-up boundary

If this note holds up after the local proof ladder, the next acceptable follow-up is **one separate
issue** for a single remote sandbox/HF Job smoke with:

- pinned install/setup steps,
- explicit hardware/runtime assumptions,
- timeout and failure handling,
- durable artifact routing,
- and an explicit statement that it does not replace the canonical local/SLURM workflow.

Follow-up tracker opened: #1191.
