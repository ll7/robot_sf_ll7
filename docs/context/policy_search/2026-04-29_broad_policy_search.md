## Executive Summary

The best local navigation policies will probably not come from “one better planner.” They will come from a **policy generation system** that searches across several planner paradigms, combines them, and evaluates them under scenario-stratified evidence.

For `robot_sf_ll7`, the target should be:

```text
safe like ORCA
+ more progressive than ORCA
+ less collision-prone than PPO
+ robust across constrained geometry, dense crowds, and flowing interactions
+ interpretable enough to debug per scenario family
```

The broad search space has five high-value families:

1. **Optimization-based planners**: MPC, trajectory optimization, lattice search.
2. **Interaction-aware geometric planners**: ORCA, velocity obstacles, social force variants.
3. **Learning-based policies**: RL, imitation learning, offline RL, diffusion policies.
4. **Hybrid planners**: learned proposals with hard safety shields, or model-based search with learned costs.
5. **Adaptive planner ensembles**: scenario-conditioned planner selection and parameter adaptation.

The most plausible “best” policy is a **hybrid hierarchical stack**, not a pure PPO policy and not pure ORCA:

```text
global / route-level intent
→ local candidate trajectory generation
→ learned or engineered interaction-risk scoring
→ hard safety shield
→ deadlock recovery
→ scenario-aware adaptation
→ benchmark-driven iteration
```

The uploaded paper already supports this framing: the current evidence shows a trade-off where `ppo` improves success but increases collision exposure, while `orca` remains safer but less successful. The benchmark is explicitly multi-objective and scenario-stratified, which is exactly the evaluation setting needed for broad policy search.

## 1. First Principle: “Best Policy” Is Not One Objective

For local navigation in pedestrian-rich AMV settings, “best” cannot mean only highest success rate.

A strong policy must satisfy multiple objectives simultaneously:

```text
reach the goal
avoid collisions
avoid near misses
avoid intrusive proximity
avoid oscillation
avoid deadlock
remain smooth
remain physically feasible
generalize across scenario families
```

This matters because the current benchmark already shows a non-dominance pattern:

```text
ORCA: safer, less successful
PPO: more successful, less safe
```

So the best-policy search should be treated as a **Pareto optimization problem**, not as a scalar maximization problem.

A strong policy is one that moves the Pareto frontier upward and left:

```text
higher success
lower collision
lower near-miss exposure
lower time-to-goal
lower jerk / discomfort
```

## 2. General Approach Class A: Better Classical Optimization

This is the most underrated route. Many weak “learning” results happen because the classical baseline is not strong enough.

### A1. Model Predictive Control

Use short-horizon optimization over feasible robot controls:

```text
state → rollout candidate controls → evaluate cost → execute first command
```

Typical objective:

```text
J =
  progress_to_goal
+ obstacle_clearance_cost
+ pedestrian_clearance_cost
+ time_to_collision_cost
+ velocity_smoothness_cost
+ angular_smoothness_cost
+ deadlock_penalty
```

Strengths:

```text
interpretable
debuggable
physically feasible
easy to add safety constraints
good for constrained geometry
```

Weaknesses:

```text
can be slow
cost weights are hard to tune
may become conservative
needs good pedestrian prediction
```

For `robot_sf_ll7`, this is probably one of the best directions because the current failures appear scenario-dependent. Optimization-based planners let you inspect exactly why the robot chose a bad command.

### A2. Lattice / Sampling-Based Local Planner

Instead of solving a continuous MPC problem, sample many feasible `(v, omega)` commands and simulate them.

```text
sample commands
simulate 2–4 seconds
reject unsafe rollouts
score remaining rollouts
execute best command
```

This is simpler than full MPC and likely enough for your benchmark.

High-value features:

```text
dynamic obstacle prediction
static clearance
pedestrian clearance
TTC
goal progress
path deviation
smoothness
stuck detection
```

This may outperform both weak PPO and naïve ORCA if implemented carefully.

### A3. TEB-like Elastic Band

Timed Elastic Band is attractive because it optimizes a local trajectory with timing, obstacle constraints, and kinematic constraints.

Potential benefit:

```text
better in narrow passages, doorways, bottlenecks, and constrained geometry
```

Risk:

```text
complex tuning
may fail in dynamic crowds if human prediction is weak
```

For the paper context, a TEB-like implementation would be valuable because it directly addresses the constrained-geometry failure regime.

## 3. General Approach Class B: Stronger Interaction Models

### B1. ORCA / Velocity Obstacle Variants

ORCA is already your best safety reference. Instead of replacing it, treat it as a **safety primitive**.

Possible improvements:

```text
ORCA + goal-progress recovery
ORCA + deadlock escape
ORCA + scenario-dependent parameters
ORCA + learned risk cost
ORCA + MPC candidate filtering
```

The key idea:

```text
Use ORCA to define what is unsafe.
Use another planner to choose what is useful.
```

Pure ORCA can be too conservative or locally myopic. But ORCA as a shield or constraint layer is very strong.

### B2. Social Force Variants

Social force models are useful but often weak when naïvely implemented.

They become more valuable if treated as **cost terms**, not as complete planners:

```text
pedestrian discomfort field
group cohesion field
wall repulsion field
directional passing preference
proxemic penalty
```

A good use:

```text
MPC / sampler generates rollouts
social-force cost evaluates interaction quality
safety layer rejects dangerous rollouts
```

A weaker use:

```text
social force directly produces control
```

### B3. Reciprocal Interaction Prediction

Many planners fail because they assume pedestrians continue straight. Better policies estimate how humans may react.

Broad options:

```text
constant velocity prediction
social-force pedestrian prediction
ORCA-style reciprocal prediction
learned trajectory prediction
multi-modal prediction
```

For local planning, perfect prediction is unnecessary. What matters is conservative short-horizon risk estimation.

## 4. General Approach Class C: Learning-Based Policies

Learning can generate excellent local policies, but only if the training setup is excellent. Otherwise, it often produces exactly what you currently see: higher success with unacceptable collisions.

### C1. Pure Reinforcement Learning

Policy:

```text
observation → action
```

Strengths:

```text
can learn nontrivial interaction behavior
can exploit simulator scale
can produce fast runtime policies
```

Weaknesses:

```text
reward hacking
poor generalization
unsafe exploration
seed sensitivity
difficult debugging
large training budget
```

Pure RL is not the first thing I would trust for your current goal. It may become powerful later, but only after the evaluation and failure-analysis pipeline is mature.

### C2. RL With Safety Constraints

Better than pure PPO:

```text
constrained RL
Lagrangian PPO
shielded RL
cost-constrained policy optimization
risk-sensitive RL
```

Instead of optimizing only reward, define costs:

```text
collision_cost
near_miss_cost
comfort_violation_cost
jerk_cost
wrong-way cost
```

Then train with constraints:

```text
maximize success/progress
subject to collision_rate <= threshold
```

This is conceptually well aligned with your benchmark. But implementation complexity is higher.

### C3. Imitation Learning

Train from expert trajectories.

Possible experts:

```text
ORCA
MPC
human-designed planner
best-of-N trajectory optimizer
mixed expert ensemble
```

The strongest variant is not imitation of ORCA alone. It is imitation of an **oracle planner**:

```text
for each state:
  sample many feasible actions
  rollout each action
  score using future outcome
  choose best safe action
train policy to imitate that action
```

This can produce a policy that is faster than planning but inherits planning quality.

### C4. Offline RL From Planner Data

Generate a large dataset from many planners and then learn a better policy offline.

Dataset sources:

```text
ORCA episodes
MPC episodes
failed PPO episodes
randomized sampler episodes
best-of-N oracle episodes
recovery maneuvers
```

Train:

```text
state, action, outcome → policy/value/risk model
```

This is promising because `robot_sf_ll7` can generate controlled simulation data.

### C5. Diffusion / Generative Trajectory Policies

Instead of outputting one action, generate trajectory candidates.

Pipeline:

```text
condition on local scene
generate multiple local trajectories
filter infeasible trajectories
score safety and progress
execute first command
```

Strengths:

```text
multi-modal behavior
can represent different passing strategies
better than one deterministic action in ambiguous interactions
```

Weaknesses:

```text
more complex
requires good data
harder to integrate
may be overkill initially
```

This could become strong later, especially for social navigation where multiple reasonable maneuvers exist.

## 5. General Approach Class D: Hybrid Policies

This is the most promising class.

### D1. Learned Proposal + Safety Shield

Architecture:

```text
learned policy proposes action
model-based shield checks safety
if safe: execute learned action
if unsafe: replace with safe fallback
```

Example:

```text
PPO proposes (v, omega)
short-horizon rollout predicts collision
if unsafe:
  use best safe sampled command
```

This directly targets your current situation:

```text
PPO has better success
PPO has too many collisions
a shield may preserve progress while reducing collisions
```

This is one of the highest-value experiments.

### D2. Model-Based Planner + Learned Cost

Architecture:

```text
sampler/MPC generates candidate trajectories
learned model scores social risk or success probability
hard safety filter rejects collisions
```

This is often more reliable than pure learning.

The learned model does not directly control the robot. It only helps ranking candidates.

Possible learned costs:

```text
collision risk
near-miss risk
deadlock risk
human-discomfort risk
success probability
time-to-goal estimate
```

This is a strong research direction because it combines interpretability and learning.

### D3. Best-of-Both Planner

Run multiple planners in parallel and select the safest useful action.

```text
ORCA action
MPC action
PPO action
social-force action
braking action
```

Then evaluate each proposal with a shared safety/progress evaluator.

This can work surprisingly well:

```text
candidate action set from diverse planners
common rollout evaluator
hard safety constraints
choose best candidate
```

This avoids betting on one algorithmic family.

### D4. Hierarchical Planner

Separate the problem:

```text
strategic layer:
  choose pass-left / pass-right / yield / follow / overtake / wait

tactical layer:
  generate feasible local trajectory

control layer:
  track command safely
```

This is likely better for social navigation than direct action prediction.

Example discrete modes:

```text
go
yield
follow
overtake left
overtake right
pass through gap
wait at bottleneck
reverse / recover
```

Then each mode has its own local planner.

This is valuable for bottlenecks and doorways because many failures are not low-level control failures; they are wrong interaction-mode choices.

## 6. General Approach Class E: Adaptive and Meta-Policies

The benchmark contains qualitatively different scenario types. One static parameterization may not dominate everywhere.

### E1. Scenario-Type-Aware Planner

Use local scene features to identify the current interaction regime:

```text
narrow passage
doorway
crossing flow
head-on encounter
parallel traffic
dense crowd
blind corner
open space
```

Then adapt:

```text
speed limit
clearance margin
prediction horizon
goal-progress weight
yielding behavior
deadlock timeout
planner choice
```

This fits your benchmark because the paper already shows different behavior between classic constrained scenarios and Francis motifs.

### E2. Planner Ensemble / Selector

Instead of one planner, learn or engineer a selector:

```text
if narrow + oncoming pedestrian:
  use conservative MPC
elif open crossing:
  use progress-biased sampler
elif dense crowd:
  use ORCA-like reciprocal avoidance
elif stuck:
  use recovery planner
```

This is probably closer to a deployable navigation stack than a single monolithic policy.

### E3. Automatic Configuration Search

Many planners fail because parameters are poor, not because the planner family is weak.

Use automatic search:

```text
Bayesian optimization
CMA-ES
random search
successive halving
population-based training
multi-objective optimization
```

Search over:

```text
clearance margins
TTC threshold
speed limits
goal weights
pedestrian weights
smoothness weights
deadlock thresholds
prediction horizon
sampling density
```

Objective:

```text
maximize success
subject to collision and near-miss constraints
```

This may produce large gains without inventing a new planner.

## 7. General Approach Class F: Curriculum and Data Generation

If you train policies, the data curriculum matters as much as the algorithm.

### F1. Nominal-to-Stress Curriculum

Training should not start on the hardest matrix.

Curriculum:

```text
empty scenes
sparse pedestrians
single crossing
doorway
bottleneck
multi-agent crossing
dense crowd
blind corner
adversarial pedestrian
```

This builds competence before stress exposure.

### F2. Failure-Driven Scenario Generation

Use failed episodes to generate more training/evaluation cases.

Loop:

```text
run planner
identify failure cases
mutate failed scenarios
train/tune on variants
evaluate on held-out original matrix
```

Mutation dimensions:

```text
pedestrian start offset
pedestrian speed
robot start angle
goal offset
density
door width
obstacle placement
reaction delay
```

This is likely one of the strongest ways to improve policies systematically.

### F3. Adversarial Testing

Train or test against adversarial pedestrian behaviors:

```text
late crossing
sudden stop
narrow gap closure
group splitting
occluded emergence
opposite-flow bottleneck
```

This should not replace nominal evaluation, but it reveals brittle policies.

## 8. General Approach Class G: Better Observation and State Representation

Sometimes the policy is weak because the planner input is weak.

The paper notes that planners use different observation contracts, and that PPO reflects the learned policy together with its planner-facing observation representation.  Therefore, improving the policy may require improving the observation.

Candidate improvements:

```text
egocentric occupancy grid
velocity grid
time-to-collision grid
agent-relative polar features
short trajectory histories
goal ray / path corridor encoding
static obstacle distance transform
local topological bottleneck features
```

For learning-based planners, I would strongly consider:

```text
agent-centric features + occupancy grid + distance transform + goal corridor
```

For model-based planners, I would consider:

```text
structured agents + obstacle segments + local distance field
```

A major insight:

```text
A better observation representation can improve all planner families.
```

## 9. General Approach Class H: Recovery and Deadlock Handling

Many local planners are acceptable until they get stuck.

A strong AMV local stack needs explicit recovery behavior:

```text
detect no progress
detect oscillation
detect blocked path
detect repeated yielding
detect unsafe narrow passage
```

Recovery actions:

```text
wait
slow creep
back up slightly
rotate to improve heading
choose alternate side
increase clearance temporarily
switch planner
request new waypoint
```

This should be implemented as a separate subsystem, not hidden inside planner weights.

In practice, a mediocre planner with excellent recovery can outperform a theoretically stronger planner with no recovery.

## 10. What Could Realistically Generate the Best Policy?

Ranking by expected value for your setting:

### Rank 1: Hybrid MPC / sampler with hard safety guard

Most likely to produce a robust improvement.

```text
model-based rollout
hard collision rejection
TTC and clearance costs
social comfort costs
deadlock recovery
scenario-aware parameters
```

Why:

```text
directly addresses constrained geometry
interpretable
easy to debug
safe by construction if implemented well
does not require massive training
```

### Rank 2: PPO or learned policy with action shielding

High-value because PPO already has success signal.

```text
keep PPO as proposal generator
reject unsafe commands
fallback to safe sampler
```

Why:

```text
may preserve PPO's success advantage
can cut collision rate sharply
relatively easy to test
```

Risk:

```text
if PPO frequently proposes unsafe commands, the shield becomes the real planner
```

### Rank 3: Model-based planner with learned risk/value scoring

Probably the strongest medium-term approach.

```text
candidate rollout planner
learned risk model
hard safety constraints
```

Why:

```text
learning improves ranking
model-based layer preserves feasibility
good compromise between performance and interpretability
```

### Rank 4: Planner ensemble with scene-conditioned selector

Potentially strongest in final form.

```text
ORCA for dense reciprocal interactions
MPC for narrow geometry
PPO/shielded policy for open interactions
recovery planner for stuck states
```

Why:

```text
different scenario families need different behavior
```

Risk:

```text
selector bugs can dominate performance
```

### Rank 5: Pure RL from scratch

Potentially powerful but poor expected value as the next step.

Why lower priority:

```text
requires strong reward design
requires curriculum
requires safety constraints
hard to debug
may reproduce current PPO failure pattern
```

Use pure RL only after the benchmark funnel, scenario curriculum, and safety evaluation are mature.

## 11. The Broadest Useful Research Program

The strongest project would have three concurrent tracks.

### Track A: Strong model-based baseline

Goal:

```text
Build the best non-learning planner possible.
```

Candidates:

```text
MPC sampler
ORCA-sampler hybrid
TEB-like optimizer
scenario-adaptive ORCA
```

This gives you a serious baseline and a possible final policy.

### Track B: Learning as a component

Goal:

```text
Use learning where it helps, not where it creates safety opacity.
```

Candidates:

```text
risk model
value model
trajectory proposal model
mode selector
shielded PPO
```

Avoid:

```text
unshielded end-to-end action policy as the only decision layer
```

### Track C: Automatic improvement loop

Goal:

```text
Make policy improvement systematic rather than intuition-driven.
```

Loop:

```text
evaluate
classify failures
generate targeted candidate
run stress slice
compare against ORCA/PPO
promote/revise/discard
```

This is what will prevent the recurring pattern of “implemented a candidate, but it is not strong.”

## 12. My Main Recommendation

The best broad strategy is:

```text
Do not search for the best planner.
Search for the best architecture for generating, filtering, scoring, and selecting local motion candidates.
```

Concrete architecture:

```text
1. Generate candidate actions/trajectories from multiple sources:
   - ORCA
   - velocity lattice
   - MPC rollout
   - PPO proposal
   - recovery actions

2. Reject candidates that violate hard safety constraints:
   - static collision
   - dynamic collision
   - TTC below threshold
   - clearance below threshold

3. Score remaining candidates:
   - goal progress
   - social clearance
   - comfort
   - smoothness
   - deadlock risk
   - learned risk/value estimate

4. Select the best safe candidate.

5. Adapt weights by scene type:
   - doorway
   - bottleneck
   - crossing
   - parallel traffic
   - dense crowd
   - blind corner

6. Log every decision for failure analysis.
```

This architecture is broad enough to absorb many ideas without rewriting the stack each time.

## 13. The Key Conceptual Shift

Your problem is probably not that “strong local planners are impossible.”

The more likely issue is this:

```text
You are implementing planner candidates as isolated algorithms,
but the benchmark requires a robust decision stack.
```

A strong local navigation policy for AMVs should not be a single policy network or a single geometric rule. It should be a layered system:

```text
intent
prediction
candidate generation
safety filtering
multi-objective scoring
recovery
adaptation
evaluation feedback
```

That is the level at which the best policy is likely to emerge.
