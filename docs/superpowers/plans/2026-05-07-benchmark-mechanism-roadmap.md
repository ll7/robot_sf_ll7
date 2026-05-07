# Benchmark Mechanism Roadmap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute the approved benchmark-mechanism roadmap by capturing the strategic issue state and preparing the first h500 trace-backed evidence slice.

**Architecture:** This is a planning, GitHub issue, and evidence-preparation workflow before any benchmark/tooling implementation. The first pass preserves the selected paper-first strategy in issues and context notes, then narrows `#1049` into an executable trace pilot with explicit validation gates.

**Tech Stack:** Markdown docs, GitHub CLI (`gh`), repository context notes, `scripts/validation/run_policy_search_step_diagnostics.py`, benchmark comparison outputs, `docs/context/evidence/`, shell commands through `rtk`, and repository PR-readiness scripts.

---

## File Structure

- Reference only: `docs/superpowers/specs/2026-05-07-benchmark-mechanism-roadmap-design.md`
  - Approved design contract for this implementation plan.
- Reference only: `docs/context/issue_1044_h500_followup_benchmark_plan.md`
  - Current h500 claim boundary, reporting plan, and pilot-slice examples.
- Reference only: `docs/context/issue_1045_h500_solvability_mechanisms.md`
  - Current aggregate mechanism analysis and representative case source.
- Reference only: `docs/context/policy_search/reasoning/2026-05-05_h500_research_plan.md`
  - Deferred planner-improvement concepts and promotion gates.
- Reference only: `docs/context/issue_928_carla_t0_t1_replay_contract.md`
  - CARLA T0/T1 transfer boundary and fail-closed semantics.
- Reference only: `docs/context/issue_1001_architecture_seam_audit.md`
  - Benchmark infrastructure pressure points if trace tooling exposes a blocker.
- Modify if needed: `docs/context/README.md`
  - Add a short link to this roadmap only if the execution creates a durable context note.
- Create if needed: `docs/context/evidence/issue_1049_h500_mechanism_pilot_2026-05-07/README.md`
  - Compact evidence index for the pilot summaries. Do not copy raw videos or large JSONL files into git.
- Create if needed: `docs/context/issue_1049_h500_mechanism_pilot.md`
  - Durable note summarizing selected cells, commands, evidence pointers, and mechanism conclusions.
- GitHub issue update: `#1049`
  - Current primary execution issue for the 2-4 week slice.
- GitHub issue creation or update: deferred planner-improvement issue
  - Captures why planner work matters, why it waits for trace evidence, and what would promote it.
- GitHub issue update: `#872`
  - Captures why broad CARLA transfer remains strategically relevant but lower priority than benchmark-meaning work.

## Task 1: Confirm Context And Clean Starting State

**Files:**
- Read: `docs/superpowers/specs/2026-05-07-benchmark-mechanism-roadmap-design.md`
- Read: `docs/context/issue_1044_h500_followup_benchmark_plan.md`
- Read: `docs/context/issue_1045_h500_solvability_mechanisms.md`
- Read: `docs/context/policy_search/reasoning/2026-05-05_h500_research_plan.md`
- Read: `docs/context/issue_928_carla_t0_t1_replay_contract.md`

- [ ] **Step 1: Read the approved design spec**

Run:

```bash
rtk sed -n '1,240p' docs/superpowers/specs/2026-05-07-benchmark-mechanism-roadmap-design.md
```

Expected: output states the selected direction is `Benchmark Meaning And Planner Failure Mechanisms` and lists deferred Planner Improvement and CARLA Transfer alternatives.

- [ ] **Step 2: Read the h500 claim boundary and pilot source notes**

Run:

```bash
rtk sed -n '1,260p' docs/context/issue_1044_h500_followup_benchmark_plan.md
rtk sed -n '1,260p' docs/context/issue_1045_h500_solvability_mechanisms.md
```

Expected: `issue_1044` includes the fixed-vs-h500 claim boundary and pilot slice; `issue_1045` includes aggregate mechanism classes such as `budget_limited_clean_completion`, `exposure_enabled_completion`, and `safety_regressed_completion`.

- [ ] **Step 3: Read deferred alternative sources**

Run:

```bash
rtk sed -n '1,180p' docs/context/policy_search/reasoning/2026-05-05_h500_research_plan.md
rtk sed -n '1,140p' docs/context/issue_928_carla_t0_t1_replay_contract.md
```

Expected: the policy-search note lists recovery, comfort, selector, and MPC-proposer directions; the CARLA note defines T0/T1 fail-closed transfer boundaries.

- [ ] **Step 4: Check current GitHub issue state**

Run:

```bash
rtk gh issue view 1049 --json number,title,state,labels,url
rtk gh issue view 872 --json number,title,state,labels,url
rtk gh issue view 1003 --json number,title,state,labels,url
```

Expected: `#1049` is open and benchmark/research/validation labeled; `#872` is open as the CARLA parent; `#1003` is open as the high-priority CARLA T1 smoke slice.

- [ ] **Step 5: Inspect the working tree**

Run:

```bash
rtk git status --short --branch
```

Expected: the branch may be ahead by the roadmap spec commit and this plan may be uncommitted. Do not revert unrelated user changes.

## Task 2: Capture The Primary Strategy On Issue `#1049`

**Files:**
- GitHub issue update/comment: `#1049`
- Reference: `docs/superpowers/specs/2026-05-07-benchmark-mechanism-roadmap-design.md`

- [ ] **Step 1: Create the strategy comment body**

Ensure the temporary directory exists, then create a local temporary file at
`output/tmp/issue_1049_strategy_comment.md` with this exact content. Tasks 2-4 all reuse
`output/tmp/`, so creating it once here is sufficient for the rest of the plan:

```bash
mkdir -p output/tmp
```


```markdown
## Strategy alignment

`#1049` is the first 2-4 week slice of the benchmark-mechanism roadmap in
`docs/superpowers/specs/2026-05-07-benchmark-mechanism-roadmap-design.md`.

The selected 1-2 month program is paper-first benchmark interpretation:

- produce trace-backed evidence for representative fixed-vs-h500 cells,
- classify why outcomes change,
- separate observed evidence from hypotheses,
- use the findings to create targeted follow-up issues for planner, reporting, or scenario-contract work.

This issue should remain evidence-first. It should not turn into a planner-tuning branch unless the
trace evidence identifies a narrow repeated planner failure mechanism.

Promotion output expected from this issue:

- selected representative cells,
- fixed and h500 trace/video evidence where feasible,
- compact evidence under `docs/context/evidence/`,
- updated h500 context notes,
- follow-up issue links for any planner-improvement, reporting, or certification work justified by the traces.
```

Expected: the file exists under ignored `output/tmp/` and contains no raw generated benchmark data.

- [ ] **Step 2: Post the comment to `#1049`**

Run:

```bash
rtk gh issue comment 1049 --body-file output/tmp/issue_1049_strategy_comment.md
```

Expected: GitHub returns the created comment URL. Save the URL in the task notes or final report.

- [ ] **Step 3: Confirm the comment is visible**

Run:

```bash
rtk gh issue view 1049 --comments --json comments --jq '.comments[-1].body'
```

Expected: output starts with `## Strategy alignment` and mentions `docs/superpowers/specs/2026-05-07-benchmark-mechanism-roadmap-design.md`.

- [ ] **Step 4: Delete the temporary comment file**

Run:

```bash
rm -f output/tmp/issue_1049_strategy_comment.md
```

Expected: the temporary file is removed. Do not delete any other `output/` contents.

## Task 3: Create The Deferred Planner-Improvement Issue

**Files:**
- GitHub issue: new issue titled `Strategy: defer planner-improvement program until h500 mechanisms are trace-backed`
- Reference: `docs/context/policy_search/reasoning/2026-05-05_h500_research_plan.md`
- Reference: `docs/superpowers/specs/2026-05-07-benchmark-mechanism-roadmap-design.md`

- [ ] **Step 1: Check whether a matching deferred issue already exists**

Run:

```bash
rtk gh issue list --state open --search "planner-improvement h500 mechanisms in:title,body repo:ll7/robot_sf_ll7" --json number,title,url
```

Expected: either no matching issue exists, or the output identifies an existing issue that clearly covers the deferred planner-improvement strategy.

- [ ] **Step 2: Create the issue body if no matching issue exists**

If Step 1 returns no matching issue, create `output/tmp/deferred_planner_improvement_issue.md` with this exact content:

```markdown
## Goal / Problem

Capture the deferred planner-improvement program identified in
`docs/superpowers/specs/2026-05-07-benchmark-mechanism-roadmap-design.md`.

The h500 evidence suggests important planner opportunities: deadlock recovery, route-local-minimum
escape, comfort-preserving high success, selector safety accounting, and MPC-as-proposer behind hard
safety filters. These are strategically relevant, but they should not be the first 1-2 month
investment track.

## Why This Matters

- The h500 policy-search notes show persistent failures in `classic_merging_medium`,
  `classic_station_platform_medium`, `francis2023_narrow_doorway`, and related long-horizon cases.
- Current high-success h500 candidates still carry near-miss and safety tradeoffs.
- A successful planner-improvement program would strengthen the research platform and may later
  improve benchmark results.

## Why Not Now

- The paper-first priority is to explain existing benchmark and h500 outcomes before tuning new
  planner behavior.
- Without trace-backed mechanism evidence, planner changes risk becoming aggregate-score tuning.
- The current benchmark-mechanism roadmap should first identify which failures are route-budget
  artifacts, waiting/yielding behavior, recovery gaps, risk-taking, or scenario-contract issues.

## Promotion Conditions

Promote this to an active implementation program only after `#1049` or a successor trace pilot
identifies a narrow repeated planner failure mechanism and a targeted slice can prove improvement
without increasing collision or near-miss risk beyond the strict incumbent envelope.

Candidate promotion gates:

- targeted h500 blocker slice solves at least one previously no-success scenario,
- collision rate remains within the strict h500 incumbent envelope,
- near-miss exposure does not improve merely by converting successes into timeouts,
- per-step diagnostics show when the new planner mechanism activates and why.

## Candidate Workstreams To Revisit

- Deadlock and route-local-minimum recovery.
- Comfort-preserving high success.
- Selector with safety accounting.
- MPC as a proposer behind hard static and dynamic safety filters.

## Related Context

- `docs/superpowers/specs/2026-05-07-benchmark-mechanism-roadmap-design.md`
- `docs/context/policy_search/reasoning/2026-05-05_h500_research_plan.md`
- `docs/context/policy_search/reports/2026-05-05_full_matrix_h500_analysis.md`
- `docs/context/issue_1044_h500_followup_benchmark_plan.md`
- `#1049`

## Definition Of Done

- [ ] A trace-backed issue identifies the specific failure mechanism to target.
- [ ] The first active planner-improvement child issue names a scenario/seed slice and strict
      safety envelope.
- [ ] Any implemented planner mechanism reports activation diagnostics and rejection reasons.
- [ ] Full h500 promotion is blocked until a targeted slice passes.
```

Expected: the body contains explicit `Why This Matters`, `Why Not Now`, and `Promotion Conditions` sections.

- [ ] **Step 3: Open the deferred planner issue if no matching issue exists**

Run:

```bash
rtk gh issue create \
  --title "Strategy: defer planner-improvement program until h500 mechanisms are trace-backed" \
  --body-file output/tmp/deferred_planner_improvement_issue.md \
  --label "research" \
  --label "benchmark" \
  --label "enhancement"
```

Expected: GitHub returns the new issue URL. Record the issue number.

- [ ] **Step 4: If a matching issue already exists, comment instead of creating a duplicate**

If Step 1 returns an existing issue, save its number in `EXISTING_ISSUE_NUMBER`, then create
`output/tmp/deferred_planner_improvement_comment.md` with this content and post it to that issue:

```markdown
## Strategy refresh

This issue remains strategically relevant, but it should stay deferred behind the h500
trace-backed mechanism pilot in `#1049`.

Why not now:

- the paper-first need is benchmark interpretation,
- planner changes should be guided by trace evidence rather than aggregate h500 scores,
- promotion should wait until a repeated failure mechanism is identified with a targeted slice and
  strict safety envelope.

Roadmap source: `docs/superpowers/specs/2026-05-07-benchmark-mechanism-roadmap-design.md`.
```

Run (replace `<NUMBER>` with the matching issue number from Step 1; do not use `1049`,
which is the primary execution issue, not a deferred planner-improvement issue):

```bash
EXISTING_ISSUE_NUMBER=<NUMBER>
rtk gh issue comment "$EXISTING_ISSUE_NUMBER" --body-file output/tmp/deferred_planner_improvement_comment.md
```

Expected: GitHub returns the comment URL for the deferred planner-improvement issue.

- [ ] **Step 5: Remove temporary issue files**

Run:

```bash
rm -f output/tmp/deferred_planner_improvement_issue.md output/tmp/deferred_planner_improvement_comment.md
```

Expected: both temporary files are absent.

## Task 4: Refresh The CARLA Parent Issue With Deferred Strategy

**Files:**
- GitHub issue comment: `#872`
- Reference: `#1003`
- Reference: `docs/context/issue_928_carla_t0_t1_replay_contract.md`
- Reference: `docs/superpowers/specs/2026-05-07-benchmark-mechanism-roadmap-design.md`

- [ ] **Step 1: Create the CARLA strategy comment body**

Create `output/tmp/issue_872_carla_strategy_comment.md` with this exact content:

```markdown
## Strategy refresh

CARLA transfer remains strategically relevant, but the benchmark-mechanism roadmap keeps it behind
the current paper-first benchmark-interpretation program.

Why relevant:

- CARLA replay can test whether Robot SF scenario conclusions survive a higher-fidelity simulator
  boundary.
- The T0 export/schema stack is mostly in place.
- `#1003` is the next executable T1 oracle replay smoke slice.

Why not the primary 1-2 month program now:

- the immediate paper value is higher from explaining existing Robot SF benchmark outcomes,
- CARLA work is still at proof-slice stage,
- broad CARLA parity should not displace h500 mechanism evidence until at least one fail-closed T1
  smoke path is proven.

Promotion condition:

- promote broader CARLA transfer when `#1003` proves a fail-closed T1 smoke path and a
  CARLA-capable environment can replay at least one certified scenario enough to support a concrete
  transfer question.

Roadmap source: `docs/superpowers/specs/2026-05-07-benchmark-mechanism-roadmap-design.md`.
Transfer boundary: `docs/context/issue_928_carla_t0_t1_replay_contract.md`.
```

Expected: the body includes `Why relevant`, `Why not`, and `Promotion condition` content.

- [ ] **Step 2: Post the comment to `#872`**

Run:

```bash
rtk gh issue comment 872 --body-file output/tmp/issue_872_carla_strategy_comment.md
```

Expected: GitHub returns the comment URL.

- [ ] **Step 3: Confirm the comment is visible**

Run:

```bash
rtk gh issue view 872 --comments --json comments --jq '.comments[-1].body'
```

Expected: output starts with `## Strategy refresh` and mentions `#1003`.

- [ ] **Step 4: Delete the temporary comment file**

Run:

```bash
rm -f output/tmp/issue_872_carla_strategy_comment.md
```

Expected: the temporary file is removed.

## Task 5: Select The First H500 Mechanism Pilot Cells

**Files:**
- Read: `docs/context/issue_1045_h500_solvability_mechanisms.md`
- Read: `docs/context/evidence/issue_1045_h500_solvability_mechanisms_2026-05-07/`
- Create: `docs/context/issue_1049_h500_mechanism_pilot.md`

- [ ] **Step 1: Inspect the mechanism evidence bundle**

Run (the directory may not exist on a fresh checkout if `#1045` evidence has not been
generated yet; the existence guard prevents a misleading error):

```bash
EV_DIR=docs/context/evidence/issue_1045_h500_solvability_mechanisms_2026-05-07
if [ -d "$EV_DIR" ]; then
  find "$EV_DIR" -maxdepth 2 -type f | sort
else
  echo "missing: $EV_DIR — confirm #1045 evidence is on this branch before continuing"
fi
```

Expected: output lists the compact evidence files generated for `#1045`, including summaries or case records. If the directory is missing, stop and confirm the branch state before selecting cells.

- [ ] **Step 2: Inspect candidate rows for the required mechanism classes**

Run:

```bash
rtk rg -n "budget_limited_clean_completion|exposure_enabled_completion|safety_regressed_completion" docs/context/evidence/issue_1045_h500_solvability_mechanisms_2026-05-07 docs/context/issue_1045_h500_solvability_mechanisms.md
```

Expected: output identifies candidate planner-scenario-seed cells or a table that points to them.

- [ ] **Step 3: Create the pilot context note skeleton**

Create `docs/context/issue_1049_h500_mechanism_pilot.md` with this exact initial structure:

```markdown
# Issue #1049 H500 Mechanism Pilot

Related issue: https://github.com/ll7/robot_sf_ll7/issues/1049

Roadmap source: `docs/superpowers/specs/2026-05-07-benchmark-mechanism-roadmap-design.md`

## Goal

Run a small fixed-vs-h500 trace-backed pilot so h500 claims can distinguish route-budget relief,
waiting/yielding, delayed progress, recovery/replanning, risk-taking, and safety regression.

## Selected Cells

| Mechanism target | Planner | Scenario | Seed | Fixed outcome | H500 outcome | Reason selected |
| --- | --- | --- | ---: | --- | --- | --- |
| Clean budget relief | ORCA | `classic_bottleneck_low` | 111 | fixed-horizon timeout candidate | h500 completion candidate | Representative route/time-budget case named in the h500 follow-up plan. |
| Exposure-enabled completion | ORCA or PPO | `francis2023_parallel_traffic` | 111 | fixed-horizon incomplete or timeout candidate | h500 completion with longer exposure candidate | Representative exposure/waiting question named in the h500 follow-up plan. |
| Safety-regressed completion | prediction planner | `francis2023_accompanying_peer` | 111 | fixed-horizon non-collision candidate | h500 completion with safety regression candidate | Representative safety-regression case named in the h500 follow-up plan. |

The exact planner/seed values must be replaced only if the `#1045` evidence bundle shows a better
matching representative cell. If a replacement is made, record the reason in the final column.

## Evidence Commands

Commands will be filled with the exact fixed and h500 trace commands after runner coverage is
confirmed. Raw outputs under `output/` are disposable unless promoted through a compact evidence
bundle or durable artifact pointer.

## Mechanism Findings

No mechanism findings are recorded yet. Each finding must distinguish observed trace evidence from
hypothesis.

## Follow-Up Boundaries

- Planner-improvement follow-ups require trace evidence of a repeated planner mechanism.
- Reporting follow-ups require a concrete missing table, rate, or evidence field.
- Scenario-certification follow-ups require a scenario-contract ambiguity, not only a planner
  failure.
```

Expected: the file exists and includes three selected mechanism targets.

- [ ] **Step 4: Replace selected cells if evidence contradicts the initial candidates**

If Step 2 shows the initial candidates do not match the required mechanism classes, edit only the
`Selected Cells` table in `docs/context/issue_1049_h500_mechanism_pilot.md`.

For each replacement, use this row format:

```markdown
| Exposure-enabled completion | PPO | `francis2023_crowd_navigation` | 112 | fixed-horizon timeout candidate | h500 completion with increased near-miss exposure candidate | Selected from `#1045` evidence because it better represents exposure-enabled completion than the initial row. |
```

Expected: the table contains one clean budget-relief row, one exposure-enabled row, and one safety-regressed row.

- [ ] **Step 5: Link the pilot note from the context README**

First confirm the target section still exists:

```bash
rg -n "^## Benchmark Run Notes" docs/context/README.md
```

If the section is present, add this bullet under it. If the README has since reorganized,
place the bullet near the other h500 notes (`issue_1044_h500_followup_benchmark_plan.md`,
`issue_1045_h500_solvability_mechanisms.md`) and record the new section name in the pilot
note's handoff:

```markdown
* [Issue #1049 H500 Mechanism Pilot](issue_1049_h500_mechanism_pilot.md)
  records the selected fixed-vs-h500 trace cells, evidence commands, mechanism interpretation, and
  follow-up boundaries for the benchmark-mechanism roadmap.
```

Expected: `docs/context/README.md` links the new pilot note near the other h500 notes.

## Task 6: Validate The Pilot Trace Command Surface

**Files:**
- Read: `scripts/validation/run_policy_search_step_diagnostics.py`
- Read: `scripts/tools/policy_analysis_run.py`
- Modify: `docs/context/issue_1049_h500_mechanism_pilot.md`

- [ ] **Step 1: Inspect diagnostics runner CLI options**

Run:

```bash
rtk python scripts/validation/run_policy_search_step_diagnostics.py --help
```

Expected: help output shows options for candidate, scenario, seed, horizon/output path, or clearly shows which equivalent runner options are available.

- [ ] **Step 2: Inspect policy-analysis runner options for fallback trace/video path**

Run:

```bash
rtk python scripts/tools/policy_analysis_run.py --help | sed -n '1,220p'
```

Expected: help output shows scenario, policy, seed-set or seed controls, output directory, and video/reporting options. If the output is too long, rerun with `rtk python scripts/tools/policy_analysis_run.py --help | rg -n "scenario|policy|seed|video|output|horizon"`.

- [ ] **Step 3: Update the pilot note with runnable command candidates**

Add a `Command Surface Check` subsection under `Evidence Commands` in
`docs/context/issue_1049_h500_mechanism_pilot.md`:

```markdown
### Command Surface Check

Diagnostics runner checked:

```bash
rtk python scripts/validation/run_policy_search_step_diagnostics.py --help
```

Policy-analysis runner checked:

```bash
rtk python scripts/tools/policy_analysis_run.py --help
```

Use `run_policy_search_step_diagnostics.py` for policy-search candidates when it can select the
needed planner/scenario/seed/horizon directly. Use `policy_analysis_run.py` or video replay tooling
for baseline planners not covered by the policy-search diagnostics runner.
```

Expected: the note records the command surfaces before any expensive trace run.

- [ ] **Step 4: Decide whether a tiny tooling issue is needed**

If neither runner can run fixed and h500 traces for the selected cells, add this subsection to the
pilot note:

```markdown
### Tooling Gap

The existing diagnostics runners do not directly cover all selected fixed-vs-h500 cells. The next
step is a narrow helper issue that adds command support for selected planner/scenario/seed/horizon
trace export. This helper must not change benchmark semantics or planner behavior.
```

Expected: the gap is recorded without implementing tooling in this task.

## Task 7: Run Documentation And Link Validation

**Files:**
- Validate: `docs/context/issue_1049_h500_mechanism_pilot.md`
- Validate: `docs/context/README.md`
- Validate: GitHub issue links from Tasks 2-4

- [ ] **Step 1: Check Markdown files for conflict markers and unresolved markers**

Run:

```bash
rtk python - <<'PY'
from pathlib import Path

markers = [chr(60) * 7, "=" * 7, chr(62) * 7, "UNRESOLVED", "NEEDS_DECISION"]
paths = [
    Path("docs/context/issue_1049_h500_mechanism_pilot.md"),
    Path("docs/context/README.md"),
]

matches = []
for path in paths:
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        if any(marker in line for marker in markers):
            matches.append(f"{path}:{lineno}: {line}")

if matches:
    raise SystemExit("\n".join(matches))

print("no unresolved markers")
PY
```

Expected: output is `no unresolved markers`. If the command prints a match from `README.md`, inspect it and leave it unchanged only if it predates this work.

- [ ] **Step 2: Check referenced local paths exist**

Run:

```bash
rtk python - <<'PY'
from pathlib import Path
paths = [
    'docs/superpowers/specs/2026-05-07-benchmark-mechanism-roadmap-design.md',
    'docs/context/issue_1044_h500_followup_benchmark_plan.md',
    'docs/context/issue_1045_h500_solvability_mechanisms.md',
    'docs/context/policy_search/reasoning/2026-05-05_h500_research_plan.md',
    'docs/context/issue_928_carla_t0_t1_replay_contract.md',
    'docs/context/issue_1049_h500_mechanism_pilot.md',
]
missing = [path for path in paths if not Path(path).is_file()]
if missing:
    raise SystemExit('missing paths: ' + ', '.join(missing))
print('all roadmap paths exist')
PY
```

Expected: output is `all roadmap paths exist`.

- [ ] **Step 3: Run diff hygiene**

Run:

```bash
rtk git diff --check
```

Expected: no whitespace errors.

- [ ] **Step 4: Review the changed files**

Run:

```bash
rtk git diff -- docs/context/issue_1049_h500_mechanism_pilot.md docs/context/README.md
```

Expected: diff contains only the new pilot note and README link.

## Task 8: Commit The Issue-Capture And Pilot-Preparation Work

**Files:**
- Commit: `docs/context/issue_1049_h500_mechanism_pilot.md`
- Commit: `docs/context/README.md`
- Do not commit: `output/tmp/*`

- [ ] **Step 1: Confirm no temporary output files are staged**

Run:

```bash
rtk git status --short
```

Expected: tracked changes include only context docs. No `output/tmp` file appears as staged or untracked for commit.

- [ ] **Step 2: Stage the docs**

Run:

```bash
rtk git add docs/context/issue_1049_h500_mechanism_pilot.md docs/context/README.md
```

Expected: the files are staged.

- [ ] **Step 3: Commit the preparation work**

Run:

```bash
rtk git commit -m "docs: prepare h500 mechanism pilot"
```

Expected: commit succeeds.

- [ ] **Step 4: Capture final state**

Run:

```bash
rtk git status --short --branch
rtk git log --oneline -n 5
```

Expected: branch is ahead of `origin/main` by the roadmap spec commit, this implementation-plan commit if committed, and the pilot-preparation commit. Working tree is clean unless unrelated user changes exist.

## Task 9: Report Handoff And Execution Choices

**Files:**
- Report only: no file edits

- [ ] **Step 1: Summarize created or updated GitHub issue links**

In the final handoff, list:

- `#1049` strategy-alignment comment URL,
- deferred planner-improvement issue number or existing issue comment URL,
- `#872` CARLA strategy-refresh comment URL.

Expected: the user can open every issue link and see why the strategy is relevant and why it is or is not current priority.

- [ ] **Step 2: Summarize local docs and validation**

In the final handoff, list:

- `docs/context/issue_1049_h500_mechanism_pilot.md`,
- `docs/context/README.md`,
- validation commands run,
- any runner/tooling gap discovered before trace execution.

Expected: the user can decide whether to start the trace pilot immediately or split a small tooling helper first.

- [ ] **Step 3: Stop before running expensive traces**

Do not run fixed-vs-h500 traces in this plan unless the user explicitly asks to execute the pilot.

Expected: the next step is a separate execution plan or issue implementation for the selected pilot cells.
