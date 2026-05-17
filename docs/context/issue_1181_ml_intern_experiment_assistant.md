# Issue #1181: `ml-intern` Bounded Experiment-Assistant Assessment

Date: 2026-05-13
Issue: #1181
Follow-up: #1191

## Decision

`huggingface/ml-intern` is a plausible **bounded experiment assistant** for `robot_sf_ll7`, but it
should **not** become the repository runtime or replace the existing local/HPC/SLURM workflows.

Update, 2026-05-15: the recommended near-term action is to **extract the reusable workflow
concepts into Robot SF's Codex-native skill and documentation practice**, not to install or run the
`ml-intern` CLI. A runtime smoke remains optional and deferred because it would add a separate agent
runtime, credential surface, trace/privacy path, and API-cost risk without improving the current
proof-first issue workflow.

Keep these workflow ideas:

- **No-edit planning pass**: read the Robot SF context stack first, produce a validation plan, and
  stop before file edits or expensive commands.
- **Budgeted agent runs**: scope each agent pass to a small issue, explicit command budget, and
  concrete proof artifact instead of open-ended exploration.
- **Trace/privacy defaults**: assume uploaded traces are sensitive unless explicitly disabled or
  sanitized; do not paste unpublished benchmark findings, secrets, or private review text.
- **One-smoke-before-campaign rule**: prove one small local command before proposing any longer
  training, benchmark, SLURM, HF Job, or remote sandbox run.
- **Optional HF-specific lane**: keep any future Hugging Face sandbox or Job smoke in a separate
  opt-in issue with pinned setup, credentials, timeout, and artifact-routing decisions.

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

For current Robot SF practice, those concepts are already better expressed through Codex-native
skills, issue-scoped branches, `scripts/dev/` validation gates, and durable context notes. The
external runtime is therefore not required to keep the useful discipline.

## Upstream Contract Checked on 2026-05-13

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

## Runtime Decision Checked on 2026-05-15

Observed evidence from the issue-audit pass:

- `ml-intern` was not installed in the checked worktree environment.
- No explicit `OPENAI_API_KEY`, Hugging Face token suitable for `ml-intern`, local model endpoint,
  or other provider credential was available for a real CLI run.
- A Codex or ChatGPT subscription is not the same thing as a runtime API credential.
- Robot SF already has repo-local substitutes for the useful workflow primitives:
  `AGENTS.md`, `docs/dev_guide.md`, `.agents/skills/`, proof-first validation scripts,
  `docs/context/`, artifact-routing policy, and GitHub issue/PR workflow.

Recommendation based on that evidence:

- Do not install, add as a dependency, or run `ml-intern` for this follow-up.
- Treat issue #1191 as a documentation and workflow-extraction task.
- Preserve the option for a later runtime smoke only if a future issue supplies explicit
  credentials, cost/privacy approval, timeout, hardware/runtime assumptions, and durable artifact
  routing.

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

Codex-native equivalents should follow the same shape without requiring `ml-intern`:

- use a no-edit planning pass for unfamiliar surfaces,
- cap the command scope before execution,
- verify referenced paths before suggesting commands,
- run one lightweight smoke before larger experiments,
- record reusable decisions in `docs/context/`,
- and keep private traces or external uploads opt-in rather than default.

Possible later external-runtime use, but only after local proof and explicit approval:

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
  `3435 passed, 17 skipped, 3 warnings in 255.63s`.
- The quickstart and dry-run both surfaced a known SVG obstacle repair warning for
  `robot_sf/maps/uni_campus_big.svg`; this did not block the bounded assessment and is exactly the
  kind of real repository signal a bounded assistant should report rather than hide.
- The PPO dry-run also surfaced the expected cadence/seed warnings from the training config:
  deprecated `evaluation.frequency_episodes` is ignored in favor of `evaluation.step_schedule`, and
  `randomize_seeds` causes the provided training seeds to be ignored.

Issue #1191 Update Validation on 2026-05-15:

```bash
ls AGENTS.md docs/dev_guide.md .agents/skills docs/context/README.md docs/README.md \
  docs/ai/repo_overview.md docs/context/issue_1181_ml_intern_experiment_assistant.md
uv run pytest tests/docs -q
git diff --check
BASE_REF=origin/main PYTEST_NUM_WORKERS=8 scripts/dev/pr_ready_check.sh
```

Observed outcomes:

- Referenced repository and documentation entry-point paths exist.
- `uv run pytest tests/docs -q` passed: `7 passed in 10.65s`.
- `git diff --check` passed with no whitespace errors.
- The first readiness run after editing passed: `3535 passed, 13 skipped, 3 warnings in 266.37s`.
- PR handoff should use a post-commit readiness rerun so changed-file detection compares committed
  docs against `origin/main`.

## Follow-Up Boundary

Issue #1191 resolved the first follow-up by extracting the useful workflow ideas into Robot SF's
existing Codex-native practice instead of running the `ml-intern` CLI.

Runtime execution remains **optional and deferred**, not a required Robot SF workflow. The next
acceptable runtime follow-up, if one is ever needed, is **one separate issue** for a single remote
sandbox/HF Job smoke with:

- pinned install/setup steps,
- explicit hardware/runtime assumptions,
- timeout and failure handling,
- durable artifact routing,
- and an explicit statement that it does not replace the canonical local/SLURM workflow.

Historical follow-up tracker: #1191.
