# Independent 0.0.7–0.0.8 distributions

The releases use independent evaluation seeds. This tool reads published episode rows,
reuses the paired comparator's pinned source/config validators, and never creates episodes.
The unit is mapped arm × kinematics × scenario × track, using
`ARM_SLOTS_0_0_7_TO_0_0_8` with explicit flags for the four replacements.

```bash
uv run python -m scripts.analysis.compare_release_distributions \
  --baseline-bundle /path/to/accepted/0.0.7_publication_bundle.tar.gz \
  --successor-root /path/to/0.0.8/campaign \
  --successor-manifest /path/to/reviewed/successor-manifest.json \
  --successor-manifest-sha256 <reviewed-sha256> \
  --successor-source-root /path/to/source/repository \
  --output-dir /path/to/results
```

The checksum-bound successor manifest uses the existing `slot-paired-successor.v1`
format: source, campaign/scenario bytes, runtime identities and versioned planner files.
Both releases require 14 arms × 48 scenarios × 30 seeds. Missing, extra or duplicate
slots fail admission; baseline metrics must be uniformly v1 and successor metrics v2.
The probe gate is mandatory and reported separately, outside arm pooling.

`--diagnostic-partial` relaxes sample and probe **coverage** only. It never relaxes
checksums, source/config binding, schemas, duplicates, extra slots or the arm roster.
Its output is labelled **diagnostic, not the release comparison**. Do not use the flag
for release evidence. An admission error exits 2; an unchanged distribution exits 0.

Readers parse one JSONL line at a time and drop simulation/decision traces and robot-force
samples before retaining rows. Memory scales with compact rows and one input line.
Breakdown tables are never read: #10044's empty jerk cells cannot become zero; jerk
comes from episode rows and remains version-qualified. Null, NaN and infinite scalars
are excluded and counted. Success-only fields report successes/exclusions and need at
least five successes in each release for a sufficient cell. They are conditioned on
successful episodes in each release, including the historical path-efficiency display.
Every such CSV/Markdown cell carries **conditional on success; estimands differ when
success rates differ**. This warns against comparing differently selected populations.

## Reviewed definitions

The explicit registry is `scripts/analysis/registries/release_distribution_metrics.json`.
Its allowlist is:

| Metric | Reason / units |
|---|---|
| avg_speed | Mean recorded robot velocity magnitude (m/s); formula unchanged. |
| energy | Sum of recorded acceleration magnitudes; unchanged proxy, not physical energy. |
| clearing_distance_min | Minimum trajectory-to-obstacle distance (m); unchanged; absent obstacles missing. |
| clearing_distance_avg | Mean nearest-obstacle distance (m); unchanged; absent obstacles missing. |
| near_misses | Count within the existing fixed pedestrian-distance threshold; unchanged. |

These formulas are unchanged from baseline source `07f7e8d` to metric v2 and do not
depend on terminal-goal corrections or SNQI. The registry records reasons and rejects
any future intersection with `metric_definitions.CHANGED_METRICS`. It never infers an
allowlist as the complement of that set. Unreviewed shared fields are excluded.

Observed changed fields are shown as `name@v1` and `name@v2`, including nested numeric
fields. D-055 curvature is additionally version-qualified despite its absence from
`CHANGED_METRICS`. Path efficiency/lengths, +1-step goal times, jerk, deadlock families,
social mini-game and SNQI never receive differences, p/q-values or changed labels.
Robot-force metrics, reference-efficiency violations and SNQI-v2-specific fields are
successor-only.

Outcome rates use `route_complete`, `collision_event` (any collision), and `timeout_event`.
`max_steps` also counts as timeout in both releases. Following #9999's measured horizon
finding, v1 unsuccessful, noncollision `terminated` rows count as timeout when recorded
steps reach their authored `simulation_config.max_episode_steps`. Earlier terminations
do not count merely because of the label.

Each unit records source-bound map SHA-256, horizon/budget, dt, completion/simulation
controls, density, robot config and `algo_config_hash`. Seed defaults are excluded
from definition fingerprints. Replaced arms, scenario changes and planner config
changes are flagged. Any flagged difference is a **combined release difference**;
the tool never labels a planner effect.

## Statistical contract and outputs

The **28 predeclared primaries** are success and collision differences for each of the
14 mapped arms. The primary estimand is **scenario-conditional**: outcomes on this
fixed benchmark suite, with equal weight per non-probe scenario. Scenarios are held
fixed; seeds are resampled with replacement within each scenario independently per
release. Disjoint release seed sets are never paired. Shared scenario effects therefore
do not inflate the primary difference interval.

All statistics are descriptive. Unit rates use Wilson 95% intervals; independent rate
differences use Newcombe hybrid-score intervals and two-sided Fisher exact p-values.
Continuous means use percentile seed bootstrap intervals with independent samples
for their differences. Their two-sided p-values use the centred independent bootstrap
null distribution with a plus-one correction. Arm rates use the scenario-conditional
seed bootstrap for their primary intervals and centred-bootstrap p-values; pooled
per-episode Wilson intervals remain reference summaries only.

A separate **paired joint scenario resample sensitivity** interval draws one set of
scenario indices for both releases, matched by scenario_id, then independently samples
seeds within each selected scenario and release. Scenario pairing does not pair episode
seeds. This interval measures sensitivity to scenario composition, appears in JSON,
CSV and a Markdown column, and does not replace the primary interval in `changed`.
Pooling over changed scenario definitions does not establish an isolated planner effect.

The fixed analysis RNG seed **20260930** is never passed to an environment. Each cell
has a deterministic SHA-256-derived stream and at least **10,000** resamples. Sensitivity
uses a separate deterministic stream. Input ordering does not affect outputs; matrices
are batched in at most 256 replicates. The synthetic null split-half regression holds
strong scenario effects in common and pins the primary difference to zero with a narrow
interval, without running any episode.

Primary contrasts use **Holm at alpha=0.05** in their own predeclared family of 28.
The family size stays 28 even for a partial diagnostic sample. Other unit and arm cells
use **Benjamini–Hochberg at q=0.05** in a separate **exploratory** family. Every output
labels the family and adjustment method. `adjusted_p_value` holds the applicable
adjustment; `q_value` is populated only for BH exploratory cells. `changed` requires
both a passing family-adjusted threshold and a primary difference interval strictly
excluding zero. Changed-definition and successor-only fields are never tested.

`distribution.json` preserves input paths, both source commits, archive/manifest and
row-file SHA-256 values, schemas, registry/digest, RNG seed, methods, support, missing
counts, fingerprints, probe classes and adjusted tests. `distribution.csv` and
`distribution.md` contain every unit/metric and pooled arm rate, with definition status,
statistics, primary and sensitivity intervals, inference family, conditioning markers
and flags for thesis review. Quantitative findings belong in
those artifacts. These tooling changes admit no dissertation evidence or release claim.
