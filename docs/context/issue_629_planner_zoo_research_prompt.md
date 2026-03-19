# Issue 629 Planner Zoo Deep Research Prompt

Date: 2026-03-19
Related issues:
- `robot_sf_ll7#629` Planner Zoo deep research for external local planner repositories and integration candidates
- `robot_sf_ll7#624` planner quality audit workflow
- `robot_sf_ll7#601` CrowdNav family feasibility note
- `robot_sf_ll7#626` SoNIC source-harness and model-only probe

## Goal

Find external local planner repositories, pretrained models, and codebases that have a high likelihood
of being successfully integrated into `robot_sf_ll7` with minimal wrapper work while preserving
provenance and benchmark credibility.

This prompt is intentionally biased toward candidates that can become original-code-backed benchmark
entries rather than large reimplementations inspired by literature.

## Canonical research prompt

```text
You are performing deep repository and codebase research for a social robot navigation benchmark project.

Goal:
Find external local planner repositories, pretrained models, and codebases that have a high likelihood of being successfully integrated into our benchmark stack with minimal wrapper work, while preserving provenance and benchmark credibility.

Primary objective:
Identify candidate repositories for a “local planner zoo” in which the original upstream code remains recognizable and attributable, ideally via upstream remote reference plus a thin local wrapper or a tracked import strategy such as a git subtree or similarly provenance-preserving integration method.

Project context:
- The target project is a Python social navigation benchmark built around Gymnasium-style environments.
- Current stack requires Python >=3.11.
- Current env stack uses `gymnasium>=0.29,<1.2`.
- Current project uses PyTorch and Stable-Baselines3; optional stacks are acceptable only if they do not force a full incompatible environment reset.
- Benchmark execution currently centers on structured social-navigation observations and differential-drive/unicycle-style execution, even when adapters are used.
- The benchmark values reproducibility, source-harness faithfulness, and paper-facing provenance more than novelty alone.
- Existing audit conclusion: only a small subset of current local planners are headline-credible; we are explicitly looking for stronger external anchors that are realistically integrable.

Research scope:
Search for external repositories implementing local planners for robot or crowd navigation, including:
- classical/reactive planners,
- optimization/MPC-style planners,
- learned local policies,
- hybrid planners with clear local action generation.

Prioritize mixed families, but rank candidates by integration feasibility first.

Hard requirements:
1. Repository must be open source with a clear, permissive or at least usable license.
   - Strongly prefer MIT, BSD, Apache-2.0.
   - GPL projects may be useful as inspiration but should be clearly marked as non-vendorable unless the downstream integration model is explicitly compatible.
   - Projects with missing or unclear licenses should be marked “reference only”.
2. Must have accessible source code, not just a paper.
3. Must have a runnable entrypoint, test script, inference script, or clearly defined evaluation harness.
4. Must be realistically compatible with our current Python/Gymnasium-oriented ecosystem, or require only a narrow, explicit shim.
5. Must expose a local action-generation behavior that could plausibly be adapted to our benchmark.
6. Must have a clear provenance story: upstream repo URL, key files, model/checkpoint availability, and whether import via subtree or wrapper is plausible.
7. Must be benchmark-credible: the result should remain attributable to the original method family, not mostly our own reimplementation.

Strong preferences:
- Python-first repositories.
- Pretrained checkpoints available.
- Clear eval/test script.
- Minimal dependence on deprecated Gym-only stacks.
- Minimal hard dependence on Docker/NVIDIA unless inference can still be reproduced locally.
- Observation and action contracts that are either already close to our benchmark or can be translated with an explicit adapter.
- Code quality that suggests a small wrapper is enough.
- Repositories whose structure would make them plausible candidates for provenance-preserving import or upstream-pinned wrapping.

Compatibility requirements to assess explicitly for every candidate:
1. Python/runtime compatibility
   - Python version assumptions
   - Gym vs Gymnasium assumptions
   - PyTorch / TensorFlow / JAX dependency burden
   - OS / CUDA / Docker dependence
2. Observation compatibility
   - What observation/state tensor or structured state the planner expects
   - Whether that is directly available in our benchmark
   - Whether a thin adapter is sufficient
3. Action compatibility
   - Holonomic / velocity-vector / waypoint / unicycle / differential-drive / acceleration output
   - Whether post-policy adaptation to `unicycle_vw` or equivalent is feasible
4. Kinematics compatibility
   - Whether the planner fundamentally assumes holonomic motion
   - Whether that assumption is fatal, manageable, or adapter-friendly
5. Reward/training coupling
   - Whether the method is tightly tied to its own reward shaping or simulator semantics
   - Whether inference-only reuse is still credible
6. Source-harness reproducibility
   - Can the original source harness likely be run locally first?
   - If not, can model-only inference still be reused credibly?
7. Integration shape
   - direct wrapper,
   - model-only inference adapter,
   - source-harness reproduction first,
   - inspiration only,
   - do not use
8. License/import suitability
   - vendorable / subtree-feasible / wrapper-only / reference-only / blocked

Required output format:
Produce a ranked “local planner zoo candidate table” with at least these columns:
- candidate name
- upstream repo URL
- planner family
- license
- language/runtime
- pretrained weights available? (yes/no/unclear)
- source test/inference path available? (yes/no)
- observation compatibility
- action/kinematics compatibility
- Gymnasium/Python compatibility
- integration shape recommendation
- provenance-preserving import suitability
- expected wrapper effort (low/medium/high)
- benchmark credibility risk (low/medium/high)
- overall recommendation:
  - integrate next
  - prototype only
  - assessment only
  - inspiration only
  - reject

For the top candidates, add a short parity note:
- what the source actually evaluates,
- what we would need to preserve,
- what would likely break benchmark faithfulness,
- why this candidate is or is not a good fit for our benchmark.

Required ranking logic:
Rank candidates primarily by:
1. likelihood of successful implementation here,
2. benchmark faithfulness and provenance quality,
3. license/import safety,
4. availability of code/checkpoints/tests,
5. expected planner quality,
6. only then novelty or literature prestige.

Avoid:
- ranking purely by published performance,
- recommending repositories with unclear licenses as integration candidates,
- recommending stacks that require wholesale environment replacement,
- hand-waving away observation/action mismatches,
- treating paper claims as enough without checking code/test assets.

Important framing:
We are not just looking for “good planners”.
We are looking for planners that could become original-code-backed benchmark entries with clear provenance and limited wrapper logic.
If a method is strong but likely to become a large reimplementation rather than a faithful import/wrap, classify it lower.

Also identify:
- the best candidate for immediate prototype,
- the best candidate for classical/reactive breadth,
- the best candidate for learned-policy breadth,
- the best candidate that is likely subtree-friendly,
- the most likely dead end despite good paper results.

Finally, recommend a concrete execution sequence for the top 3 candidates:
1. what to assess first,
2. what to try to run first,
3. what to wrap only after source-harness validation,
4. what to avoid entirely.
```

## Evaluation rubric

Apply these gates in order:

1. License and provenance gate
- reject or downgrade repos with missing/unclear licenses
- require upstream URL, runnable code path, and import suitability classification

2. Runtime and ecosystem gate
- prefer Python-first repos compatible with Python >=3.11
- prefer Gymnasium-ready or narrow-shim candidates
- downgrade candidates requiring wholesale environment replacement

3. Observation/action/kinematics gate
- require explicit observation contract notes
- require explicit action output and post-policy adaptation notes
- downgrade candidates whose motion assumptions are fundamentally misaligned with unicycle execution

4. Reproducibility gate
- prefer source-harness runnable repos
- if source harness is blocked, require evidence that model-only inference is still plausible

5. Benchmark credibility gate
- prefer candidates that can be benchmarked as original-code-backed entries
- downgrade candidates that would mostly become our own reimplementation

## Required output interpretation

Use these final recommendation classes only:
- `integrate next`
- `prototype only`
- `assessment only`
- `inspiration only`
- `reject`

## Execution sequence for resulting research

1. Gate on license + runnable source path first.
2. Promote only provenance-preserving candidates.
3. Pick top 3 and classify.
4. Open one implementation issue per surviving top candidate.
5. Keep benchmark claims conservative until source-harness or model-only parity is demonstrated.

## Current defaults

- planner-family scope: `mixed`
- runtime target: Python `>=3.11`, `gymnasium>=0.29,<1.2`
- benchmark target: structured Robot SF observations with explicit action adaptation when needed
- provenance preference: upstream URL plus thin wrapper first; subtree-friendly import is a plus
- unclear license: `reference only`
