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

## Deep-research intake update (2026-03-19)

The following shortlist was added after an external deep-research pass to refine the next likely
planner-zoo additions. This is a research intake, not a blanket approval to import code. The same
license, provenance, and source-harness gates still apply.

Important caveats:

- `CrowdNav-SB3` is recorded here as a promising direction, but the originally supplied upstream
  reference was not a canonical GitHub repository URL. Treat it as a provisional candidate until
  the exact upstream source, license, and test path are re-verified.
- `PySocialForce` is useful as a candidate family, but the originally supplied repository
  attribution may differ from the canonical upstream. Keep the family assessment, and re-verify the
  exact upstream repo before any import decision.
- `SocNavGym` remains `prototype only` because GPL-3.0 blocks direct vendoring into this repository.
- `SDA` remains `assessment only` because Habitat coupling makes direct benchmark integration
  unlikely without substantial reimplementation.

### Candidate table

| Candidate name | Upstream repo URL | Planner family | License | Pretrained weights? | Obs. compatibility | Action/kinematics | Gymnasium compat. | Integration shape | Wrapper effort | Recommendation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `CrowdNav-SB3` | provisional upstream; exact canonical repo still needs re-verification | Learned (RL/attention) | unresolved in intake; re-verify before import | Yes (claimed in intake) | High (state tensors) | Holonomic or unicycle adapter | High if upstream claim holds | Direct wrapper | Low | `integrate next` if repo claim is verified |
| [`PySocialForce`](https://github.com/yuxiang-gao/PySocialForce) | <https://github.com/yuxiang-gao/PySocialForce> | Classical (social force) | MIT | N/A | High (agent states) | Velocity vector | High | Direct wrapper | Low | `integrate next` |
| [`SocNavGym`](https://github.com/gnns4hri/SocNavGym) | <https://github.com/gnns4hri/SocNavGym> | Hybrid (GNN + RL) | GPL-3.0 | Yes | High (structured) | Unicycle or continuous | Native (`0.29.1` in intake) | Source harness | Medium | `prototype only` |
| [`SDA`](https://github.com/L-Scofano/SDA) | <https://github.com/L-Scofano/SDA> | Learned (history-based) | MIT | Yes | Medium (history-based) | Unicycle | Medium | Model-only adapter | High | `assessment only` |
| [`RVO2-python`](https://github.com/chengji253/RVO2-python) | <https://github.com/chengji253/RVO2-python> | Geometric (ORCA) | MIT | N/A | High (position and velocity) | Velocity vector | High | Subtree or wrapper | Low | `integrate next` |
| [`PythonRobotics` DWA](https://github.com/AtsushiSakai/PythonRobotics) | <https://github.com/AtsushiSakai/PythonRobotics> | Classical (reactive) | MIT | N/A | Medium (scan-style or handcrafted state) | Unicycle | Medium | Inspiration or native port | Low | `inspiration only` |

### Top-candidate parity notes

#### CrowdNav-SB3

- Source evaluation:
  - intake characterizes this as an SB3-native port of the historical CrowdNav SARL/CADRL family
  - intended scenarios are circle-crossing and random multi-agent crowd navigation
- Preserve:
  - multi-input policy wiring
  - attention handling over variable pedestrian sets
  - the source observation flattening or tensorization path
- Risks:
  - exact upstream provenance is still unresolved from the intake
  - parity breaks immediately if we benchmark a wrapper over materially different state extraction
- Interpretation:
  - promising as the top learned-policy candidate, but only after the exact upstream repo and test
    harness are verified

#### PySocialForce

- Source evaluation:
  - vectorized Helbing-style social-force implementation in Python
- Preserve:
  - attractive, repulsive, and social force composition
  - the baseline’s native desired-velocity interpretation
- Risks:
  - outputs velocity vectors, so Robot SF would still need an explicit `unicycle_vw` action adapter
- Interpretation:
  - strong candidate for classical breadth because the implementation is lightweight, MIT-licensed,
    and likely wrapper-friendly

#### SocNavGym

- Source evaluation:
  - structured Gymnasium environment with social-navigation-specific agents and learned policies
- Preserve:
  - environment-specific observation graph or structured-state contract
  - policy expectations around that observation layout
- Risks:
  - GPL-3.0 blocks vendoring
  - credible use would require wrapper-only or external-dependency integration, not subtree import
- Interpretation:
  - strong prototype candidate, not a direct import candidate

### Selection summary

- Best for immediate prototype:
  - `CrowdNav-SB3`, if and only if the upstream repo/license/test path are re-verified
- Best for classical breadth:
  - `PySocialForce`
- Best for learned breadth:
  - `SocNavGym`
- Best subtree-friendly candidate:
  - `RVO2-python`
- Most likely dead end despite strong paper value:
  - `SDA`, because Habitat coupling makes parity costly

### Recommended execution sequence

1. Assess and, if still justified, subtree or wrap `RVO2-python` as the clean geometric baseline.
2. Verify `CrowdNav-SB3` provenance, then run its source-side inference path before any wrapper work.
3. Wrap `PySocialForce` with an explicit unicycle adapter once the geometric baseline contract is
   stable.
4. Avoid Habitat-locked imports such as `SDA` until the core planner-zoo harness is stable.

## Second-pass integration intake (2026-03-19)

This second-pass intake sharpens the shortlist toward candidates whose upstream code can remain
intact, runnable, and attributable with only a thin wrapper. The central conclusion is operational:
the best near-term additions are not necessarily the most novel methods, but the ones that preserve
provenance while fitting the current benchmark with explicit, reviewable adapters.

### Executive summary

- Best immediate production candidate:
  - `Python-RVO2`
  - reason: permissive license, stable bindings, obvious example entrypoint, and a very small
    provenance surface
- Best Gymnasium-native breadth anchor:
  - `Social-Navigation-PyEnvs`
  - reason: already aligned with `gymnasium==0.29.1`, includes differential-drive framing, and
    spans ORCA, social-force, and CrowdNav-family learned policies
- Best learned-policy breadth candidate:
  - `CrowdNav_HEIGHT`
  - reason: MIT license, explicit test scripts, downloadable checkpoints, and a clearer learned
    method identity than most legacy CrowdNav-lineage repos
- Most likely dead end despite good paper results:
  - `SoNIC-Social-Nav`
  - reason: public release remains test or visualization heavy, Docker or NVIDIA oriented, and does
    not yet give a complete public source-harness story

Main pattern:

- classical or reactive planners integrate cleanly
- CrowdNav-lineage learned methods remain benchmark-credible but runtime-legacy
- ROS-heavy MPC planners are method-credible but expensive to integrate faithfully into a
  Gymnasium-first Python benchmark

### Ranked candidate table

| Rank | Candidate name | Upstream repo URL | Planner family | License | Language/runtime | Pretrained weights? | Source test/inference path? | Observation compatibility | Action/kinematics compatibility | Gymnasium/Python compatibility | Integration shape recommendation | Provenance-preserving import suitability | Expected wrapper effort | Benchmark credibility risk | Overall recommendation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `Python-RVO2` | cited upstream in deep audit; confirm canonical remote before import | Classical/reactive ORCA | Apache-2.0 | C++ core + Cython/Python bindings | No | Yes (`example.py`) | Needs agent positions, radii, and preferred velocity; usually derivable from benchmark state | Velocity-space / holonomic output; explicit `unicycle_vw` projection still required | High; no Gym dependency | Direct wrapper around upstream simulator | Excellent | Low | Low | `integrate next` |
| 2 | `Social-Navigation-PyEnvs` | cited upstream in deep audit; confirm canonical remote before import | Mixed framework: ORCA, SFM/HSFM, CADRL, LSTM-RL, SARL | Apache-2.0 | Python | Unclear | Yes | Very good; structured crowd state and differential-drive robot already exist | Good; differential-drive robot and laser sensor reduce adaptation burden | Strong if the stated `gymnasium==0.29.1` claim holds | Source-harness reproduction first, then wrap selected planners | Good | Low to medium | Low to medium | `integrate next` |
| 3 | `CrowdNav_HEIGHT` | cited upstream in deep audit; confirm canonical remote before import | Learned graph-transformer local policy | MIT | Python + PyTorch + OpenAI Baselines | Yes | Yes (`test.py` and variants) | Good if CrowdNav-style structured robot/human state is supplied | Likely velocity-style local action; nonholonomic parity must be validated explicitly | Weak for main stack; better in a side env | Model-only inference adapter after source-harness validation | Good for wrapper-only import | Medium | Medium | `prototype only` |
| 4 | `CrowdNav` | <https://github.com/vita-epfl/CrowdNav> | Learned local policies + ORCA baseline | MIT | Python | Unclear | Yes (`train.py`, `test.py`) | Good for structured robot/human state if state contract is preserved | Benchmark-credible only if the source policy contract is preserved | Legacy stack; not main-stack native | Source-harness reproduction first | Good | Medium | Medium | `assessment only` |
| 5 | `soc-nav-training` | cited upstream in deep audit; confirm canonical remote before import | Learned SARL variant / curriculum training | MIT | Python + Torch + legacy Gym | Yes | Yes (`train.py`, `test.py`) | Conceptually close to CrowdNav-style structured state | Same CrowdNav-lineage caveat; preserve source policy semantics | Weak for main stack; better in a side env | Model-only inference adapter if source harness runs first | Good as a pinned external source | Medium | Medium | `prototype only` |
| 6 | `LT_DWA` | cited upstream in deep audit; confirm canonical remote before import | Classical/reactive kinodynamic DWA | MIT | ROS Noetic + C++/Python | No | Yes (ROS launch flow) | Moderate; expects ROS-world state and simulator-specific config | Strong differential-drive faithfulness | Weak for current stack; ROS side env required | Source-harness reproduction first, then ROS bridge only if justified | Fair | High | Medium | `assessment only` |
| 7 | `lpsnav` | cited upstream in deep audit; confirm canonical remote before import | Hybrid/social policy framework | MIT | Python | No | Yes (`sim.py`) | Moderate; own scenario/config contract must be mapped | Promising; desired speed and heading interface is adapter-friendly | Python-first and light, but version support still needs confirmation | Direct wrapper or subtree import after small reproduction | Very good | Low to medium | Medium | `prototype only` |
| 8 | `tud-amr/mpc_planner` | cited upstream in deep audit; confirm canonical remote before import | Optimization/MPC motion planner | Apache-2.0 | ROS/C++ + Python solver generation | No | Yes, via ROS/container setup | Moderate; needs path, obstacle, and cost inputs rather than benchmark-native social obs | Good for motion control, not a direct crowd-policy drop-in | Weak for main stack | Assessment only; bridge, not wrapper | Legally fine, but not lightweight | High | Medium | `assessment only` |
| 9 | `Pred2Nav` | <https://github.com/sriyash421/Pred2Nav> | Hybrid prediction + vecMPC crowd navigation | Unclear / missing | Python + PyTorch + TensorFlow + legacy Gym | No or unclear | Yes | Moderate; conceptually close to CrowdSim-style structured state | Attractive MPC action generation, but tightly coupled to its own state/prediction contract | Weak for current stack | `reference only` unless license is clarified | Reference-only | High | High | `inspiration only` |
| 10 | `SoNIC-Social-Nav` | <https://github.com/tasl-lab/SoNIC-Social-Nav> | Learned safe RL + uncertainty wrapper | MIT (root) | Python in Docker, NVIDIA-oriented | Yes | Yes (`test.py`, `visualize.py`) | Good on paper | Plausible for local policy reuse, but incomplete public harness blocks a faithful claim | Poor fit for main stack today | Do not use for integration now | Wrapper-only at best, with incomplete provenance | High | High | `reject` |
| 11 | `mpc_ros` | cited upstream in deep audit; confirm canonical remote before import | Nonlinear MPC path-tracking local planner | Apache-2.0 | ROS Melodic + C++/Python tooling | No | Yes (ROS build/demo) | Low for social-nav benchmark | Good for unicycle/differential tracking, weak as a social local planner | Poor fit to current stack | Do not use except as controller inspiration | Legally fine, but family mismatch is large | High | High | `reject` |
| 12 | `mpc_waypoint_tracking_controller` | cited upstream in deep audit; confirm canonical remote before import | Waypoint-tracking MPC local planner | Apache-2.0 | ROS/catkin C++ | No | Yes (demo launch flow) | Low for social benchmark observations | Good for `cmd_vel` generation, weak as a social local planner | Poor fit to current stack | Do not use except as control-backend inspiration | Legally fine, but benchmark-faithful reuse is weak | High | High | `reject` |

### Why the top candidates rank this way

#### 1. Python-RVO2

- Cleanest provenance-preserving import in the set
- compact upstream, permissive license, obvious example, and no Gym, ROS, Docker, or GPU burden
- the real limitation is kinematics, not packaging:
  - ORCA is fundamentally velocity-space and effectively holonomic
  - any `unicycle_vw` projection must be explicit and benchmark-visible

#### 2. Social-Navigation-PyEnvs

- best ecosystem-level match to the current stack
- unusually strong breadth anchor because it combines:
  - Gymnasium alignment
  - differential-drive framing
  - classical and learned crowd-navigation methods
- main unknown:
  - whether learned checkpoints are actually bundled and reusable without retraining

#### 3. CrowdNav_HEIGHT

- strongest learned-policy candidate from an operational benchmark perspective
- the ranking penalty is runtime age, not method quality:
  - better treated in a frozen side environment than forced into the main Python 3.11 stack

#### 4. CrowdNav and `soc-nav-training`

- still matter as the canonical lineage for CADRL/LSTM-RL/SARL-style local policies
- value is primarily lineage and benchmark credibility, not frictionless integration

#### 5. LT_DWA

- attractive because it is explicitly differential-drive and crowd-aware
- ranking penalty comes from the likely need for a ROS bridge or sidecar process rather than an
  in-process Gymnasium wrapper

### Parity notes for the top candidates

#### Python-RVO2

- Source actually evaluates:
  - reciprocal collision avoidance in velocity space through the RVO2/ORCA formulation
- Must preserve:
  - simulator parameters
  - preferred-velocity semantics
  - neighbor/radius/time-horizon settings
  - explicit ORCA-family identity
- Would break faithfulness:
  - rewriting the avoidance logic while still calling it ORCA
  - hiding the holonomic-to-unicycle projection in evaluation glue
- Why it fits:
  - minimal runtime burden and almost no provenance ambiguity

#### Social-Navigation-PyEnvs

- Source actually evaluates:
  - a social-navigation framework spanning ORCA, SFM/HSFM, and CrowdNav-family learned policies in
    a Gymnasium-style environment with a differential-drive robot
- Must preserve:
  - original environment dynamics
  - human model selection
  - policy identities and module boundaries
- Would break faithfulness:
  - re-encoding the policies into a benchmark-native environment while still claiming original-code
    backing
- Why it fits:
  - unusually good ecosystem alignment for a breadth candidate

#### CrowdNav_HEIGHT

- Source actually evaluates:
  - learned crowded-navigation inference through a heterogeneous interaction graph-transformer setup
- Must preserve:
  - original checkpoint
  - graph/state construction path
  - source test-harness behavior before wrapper work
- Would break faithfulness:
  - retraining in our environment and presenting the result as the upstream method
  - reimplementing the graph/state interface from the paper alone
- Why it fits:
  - strong learned-policy anchor with explicit assets, but only as an external wrapped method

#### LT_DWA

- Source actually evaluates:
  - kinodynamic local planning for differential wheeled robots in static and crowd environments via
    ROS launch flows
- Must preserve:
  - ROS-side configuration
  - wheel/kinodynamic assumptions
  - original launch/evaluation path
- Would break faithfulness:
  - reimplementing the search/cost logic in Python instead of bridging to the upstream planner
- Why it fits or does not fit:
  - strong breadth addition, but not a low-effort one

### Best-of selections

- Best candidate for immediate prototype:
  - `Python-RVO2`
- Best candidate for classical/reactive breadth:
  - `LT_DWA` if differential-drive faithfulness matters most
  - `Python-RVO2` if friction and provenance matter most
- Best candidate for learned-policy breadth:
  - `CrowdNav_HEIGHT`
- Best subtree-friendly candidate:
  - `Python-RVO2` first
  - `lpsnav` second
- Most likely dead end despite good reported results:
  - `SoNIC-Social-Nav`

### Concrete execution sequence for the top 3

1. `Python-RVO2`
   - assess first:
     - isolated Python 3.11 build/install
     - upstream example loop
     - benchmark-visible `velocity_vector -> unicycle_vw` projection policy
   - run first:
     - upstream `example.py` unmodified
     - then a minimal benchmark adapter
   - wrap only after source validation:
     - yes
   - avoid:
     - hiding holonomic-to-unicycle projection inside evaluation plumbing

2. `Social-Navigation-PyEnvs`
   - assess first:
     - clean install in current stack
     - which planners are callable without retraining
     - whether a planner-only API is possible without adopting the whole simulator
   - run first:
     - package install
     - one classical baseline
     - one learned CrowdNav-family path if available
   - wrap only after source validation:
     - wrap selected planner modules, not the full simulator, unless we explicitly adopt it as a
       benchmark backend
   - avoid:
     - rewriting policies into benchmark-native abstractions before proving upstream parity

3. `CrowdNav_HEIGHT`
   - assess first:
     - whether the authors' frozen side environment still runs
     - whether downloaded checkpoints execute end-to-end via `test.py`
     - exact state tensor and config requirements at inference
   - run first:
     - upstream checkpoint evaluation exactly as documented
     - only then a one-way adapter from benchmark observation into the upstream state contract
   - wrap only after source validation:
     - yes
   - avoid:
     - porting directly into the main Python 3.11 environment before the source harness is proven

### What to avoid entirely

- `Pred2Nav` until the licensing situation is clarified
- `SoNIC-Social-Nav` as a benchmark entry today
- ROS MPC/path-tracking controllers such as `mpc_ros` or `mpc_waypoint_tracking_controller` as
  social local planners

### Final recommendation

If the objective is a credible local-planner zoo with minimal wrapper work and strong provenance,
the practical first wave is:

1. `Python-RVO2` as the first production candidate
2. `Social-Navigation-PyEnvs` as the breadth scaffold and likely second integration target
3. `CrowdNav_HEIGHT` as the first learned external anchor, but only through side-environment
   validation and checkpoint-backed inference wrapping
