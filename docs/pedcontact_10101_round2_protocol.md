<!-- AI-GENERATED / NEEDS-REVIEW: PEDCONTACT Round 2, issue #10101 -->
Historical Round 2 evidence. Superseded by [Round 3 corrections and current results](pedcontact_10101.md); the old two-sided V6 ranking and whole-crowd-fallback results are withdrawn.
# PEDCONTACT Round 2 corrections

Round 1's V6 necessary-condition pruning and zero-survivor conclusion are withdrawn.
Five identical, deterministic no-interferer baselines have effectively zero angular velocity.
A Gaussian tail exceeding that threshold is not an avoidance onset. The old 3 m capture
saturated onset near its first sample and made the calibration conclusion tautological.

## V6 correction to the #10075 validation suite

[Huber et al. 2014, sections 4.2 and 4.4](https://doi.org/10.1371/journal.pone.0089589)
measure filtered angular velocity and compare it with five human no-interferer trials.
Their trunk-marker trajectories contain sway, unlike this deterministic CM model.
We retain the 0.5 Hz Gaussian and five-sample median, own-X distance to synchronized
PoMD, and five baseline diagnostics. Onset now requires **0.05 rad/s for 0.3 s**,
equivalent to at least 0.86 degrees of accumulated heading change. This rejects numerical
noise and filtering tails while retaining a small observable turn. This explicit engineering
criterion is **not a numeric threshold supplied by Huber**, and matching their published
onset means under it does not establish empirical equivalence. Sensitivity at 0.025 and
0.1 rad/s is reported; the criterion is frozen before the grid, not tuned to pass a target.
The own-X analysis window is 25 m, beyond the configured 20 m social-force range.
Initial separation is 60 m instead of the human protocol's 12 m so the complete window
is observed. This deterministic extension is declared in raw case records. No published
mean or accepted range is changed; protected config bytes retain the historical protocol
text, superseded for these runs by the estimator identity and explicit row fields.

## Solver and radius

Uniform-grid candidate cells have size 2r plus a micrometre margin. Their search radius
expands conservatively for swept motion. A 0.10 m neighbour-list skin avoids rebuilding
the grid on every small constraint pass. It is rebuilt after any endpoint moves more
than 0.05 m from its reference; relative swept paths then differ by at most 0.10 m.
Admission requires the skin bound to remain valid. Large motion falls back to all pairs.
Alternating Gauss-Seidel with 1.6 over-relaxation handles contact chains; every admission
also checks actual pair and wall geometry. Sequential corrections propagate local chain
constraints within a pass. On the same congested probe, 1.6 reduced the maximum passes
from 74 at 1.2 to 40. The warmed measurement determines whether it meets the cost target.
Capsule AABBs cull irrelevant wall constraints. Start push-out repairs invalid corner
starts. Segment lines and both endpoint planes are independent constraints; shared vertices
cannot shadow the wall interior. At the 256-pass cap, deterministic endpoint push-out
is attempted, then an admissible previous state is used where possible. Fallback and
unresolved counters remain visible; no exception aborts the episode. Residual geometry
is still counted as a violation rather than admitted as a success.
Velocity correction removes closing normal components and reapplies the integration cap;
it never converts geometric displacement into velocity. Wall normals blend over 0.30 m,
avoiding medial-axis chatter while preserving the bounded finite-range strength.
A 0.04 m trial remained in a clipped lateral cycle at a near-wall start and is
superseded. The 0.30 m band makes the parallel-gap restoring-gradient bound
A(4/blend + 1/decay) <= 345/s² for the whole A<=9, decay>=.04 grid. At dt=.1
and tau=.5, the damped semi-implicit limit is 360/s². A real-force 300-step
witness at amplitudes 3/6/9 must damp below 10 micrometres, not just be continuous.
Selectors retain the backend radius unless pedestrian_radius_m is explicitly provided.
Both robot gate arms explicitly use [milestone 0.1.0's 0.28 m radius](https://github.com/ll7/robot_sf_ll7/milestone/11).

## Test value and pre-fix witnesses

The first ten new cases failed on ead48d9c at physical assertions, not imports or fixtures.
- Protect: feasible corner repair, non-aborting cap fallback, stationary repair velocity,
  continuity across a gap, unchanged unset radius, and onset sensitive to actual manoeuvres.
- Credible regressions: choosing one corner plane; raising at iteration cap; writing
  displacement/dt; nearest-only normals; assigning default ped_radius; zero-sway thresholds.
- Existing coverage missed: the original contact tests had no shared-vertex witness,
  exhausted solver, post-repair speed assertion, medial-axis limit, unset radius or onset
  nonsaturation control; the original onset test explicitly expected the saturated 3 m answer.
- No production test seam: real simulator steps and geometry; the fallback witness only
  temporarily lowers the existing iteration-limit constant. The radius oracle is the
  independent backend default; onset controls use analytic paths and tiny yaw perturbations.
A further normal-impact witness fails with the exact reviewed contact module: closing
velocity must be removed even when over-relaxation leaves a positive positional margin.
An oblique velocity-transfer witness protects the final per-body cap: removing normal
closing speed can increase one body's speed while reducing pair energy. Stationary
repair and symmetric head-on controls miss this case. It uses a real simulator with
zero driving forces, not a production-only test seam, and fails the reviewed kernel.
The original CALFIT preflight saturated-onset oracle is replaced by numerical-yaw
rejection and independently known five-metre manoeuvre-translation controls.
The warmed congestion probe and complete robot JSON records verify runtime and spawn
receipt findings directly, without fragile machine-speed pytest assertions.

## Fit policy

Measure every declared one of 108 settings on dev1001–1003. Evaluate V2 and V5 for every
wall setting and retain every item in the complete fitness ranking. No single-item pruning.
Ranking is passed-check count, then summed out-of-range residual, then candidate identifier;
missing measurements receive a large explicit penalty. The original Oct7 18:32UTC deadline
remains in force. Numerical range agreement is kept separate from physical validity and
from unresolved human-protocol equivalence, particularly V5 and V6.
