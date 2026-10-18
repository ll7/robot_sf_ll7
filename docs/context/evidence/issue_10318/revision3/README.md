# Continuous-interval pedestrian margin after the second FIX review

Development diagnostic only. Supersedes the sampled-position guard in [revision 2](../revision2/README.md). Independent review of the updated exact head remains required; this is not evidence admission or merge authorization.

## Defect and implementation

The reviewed head `65a3f648b428332cf582d0a2253e4363bab3b528` checked only sampled pedestrian/robot positions. The review's exact witness has a 0.350257135 m surface gap at the origin and first sample, but 0.3498 m at t=0.05 s. Its unsafe `(1.04, 0)` command was returned despite the configured 0.35 m margin.

Both learned and observed constant-velocity forecasts now include t=0 and check minimum synchronized separation on every interval. For relative chord endpoints a and b, minimize `|a + u (b-a)|` with `u = clip(-a·(b-a)/|b-a|², 0, 1)`; a constant relative chord uses u=0. Learned origins use the measured observed actor slots. Scoring-only inputs lacking observed slots extrapolate the first forecast chord to the origin. Production forecasts preserve their observed slots.

For the bound differential drive, subtract a conservative synchronized robot-motion cover from the chord distance. Over an interval of length h, bound actual linear speed by V, angular rate by W and linear acceleration by A, respecting measured velocity, clipping and every internal control substep. A continuous unicycle's chord interpolation error is at most `(A+V*W)*h²/8`. Mid-heading odometry differs from that continuous path by at most `V*W*h²/2`; covering both the interior and endpoint-interpolated error adds `V*W*h²`. The total cover is `(A+9*V*W)*h²/8`. Constant straight motion has zero cover; straight acceleration retains an acceleration cover. Reversals and curved internal substeps remain covered. Unbound command rollouts retain their declared piecewise straight sample-segment semantics.

A lower-bound separation below `robot_radius + pedestrian_radius + predictive_safe_distance` returns infinite cost through the shared action/sequence guard. Finite beam filtering and final command braking remain. The admission objective correction, population-capacity refusal and historical center-distance semantics are unchanged. No model, threshold, radius, horizon, configuration, scenario or goal weight is tuned.

## Regression proof and controls

Thirteen deterministic cases: the exact first-interval witness through action scoring, sequence scoring and returned command, isolating both/learned/observed sources; curved-drive cases at 0.1 s and 0.025 s control clocks through both scorers. Independent geometry proves clear sample endpoints and a sub-margin interior. Reviewed head: **13 failed**, intended finite-score/returned-clearance assertions. Fix focused module: **53 passed**, including the existing clear-progress, compliant slow-progress, touching-body and capacity controls.

Four test-value answers: protect interval-safe motion; the reviewed head demonstrably fails; expectations use analytic geometry and returned commands rather than internal call ordering; fixed eight-frame inputs and bounded searches are cheap and deterministic. Existing sampled-margin cases never cross inside a clear-ended interval. Only the external predictor is supplied with deterministic forecast bytes; no production test seam is added.

The complete requested selection (`tests/planner` plus every Python test file referencing `prediction_planner` or `PredictionPlannerAdapter`) ran once with `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1`: **4,763 passed, 17 skipped, 4 xfailed**, exit 0. Three skips exclude live non-development episode seeds; other skips are absent optional dependencies/checkpoints and the xfails are existing recorded planner-envelope issues. Exclusions are not passes. No full readiness rerun or CI polling.

## Development comparisons

[All eighteen affected pairs](affected.csv): reviewed versus interval guard, seeds 1001–1003, complete canonical 400-step/0.1 s traces, same resets, digest-matched loaded predictor, no fallback. **All traces are identical**, including outcomes, steps, contacts, episode minima and terminal gaps. Three completions remain; fifteen timeouts remain. Every measured minimum and terminal gap exceeds 0.35 m; worst minimum 0.517095510 m. Crowding still holds at reset because the observed population exceeds configured forecast capacity. [Affected summary](affected_summary.json).

Empty-world replay uses all three development seeds; the reviewed comparator includes preserved seeds 1001/1002 and fresh seed 1003 controls. [All paired rows](empty_world.csv), [summary](empty_summary.json). All 153 traces are identical: main 141/144 → 141/144 successes, width 9/9 → 9/9, zero contacts or new failures. The three main timeouts are the original physical narrow-doorway refusal on each seed. The table records every trace/trajectory change and contact/outcome comparison.

## Provenance and limitations

[Manifest](manifest.json) binds runtime revisions, named input and artifact hashes. All CSVs use LF line endings, including normalized historical tables whose hashes were refreshed. Complete raw traces, logs, test exclusion plugin and failed-old proof are preserved privately.

The interval veto covers piecewise linear pedestrian forecasts and the declared bound-drive motion, not arbitrary future pedestrian behavior or forecast error. Its conservative curve cover may reject safe candidates elsewhere. Population refusal may increase dense-scene timeouts. Residual timeouts and these tradeoffs remain under #10318. Independent exact-head review is the next action; no merge, release, held-out claim or evidence admission occurs here.
