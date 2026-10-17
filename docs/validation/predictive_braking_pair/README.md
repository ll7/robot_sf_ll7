# Paired predictive braking development diagnostics

**Diagnostic-only. AI-generated, needs review. Predictive braking is not a safety guarantee.**

[Contract and reproduction commands](../../benchmark_predictive_braking_pair.md)

Execution source: `6c9d79cda6dfaafc8c24c846e774a7bbd2b657e8`, after merged
#10228. Both named hybrid arms resolve through the map runner from the 0.1.0
companion campaign. No learned PPO integration, released scenario change,
default activation, release admission or paper claim is made. The existing
default candidate, its base config and the pinned learned-policy campaign remain
byte-identical to base `36d701c639f870cbb94186df9d8ed205bf7b4c30`.

## Station, development seeds 1001–1030

Both arms use a 600-step diagnostic budget and the same scenario/seed pairs.
The normal campaign retains the authored 650-step station budget; these results
do not stand in for that full campaign.

| Metric | Default | Predictive |
| --- | ---: | ---: |
| Route-complete successes | 0/30 | 15/30 |
| Success proportion, descriptive Wilson 95% interval | 0–11.35% | 33.15–66.85% |
| Episodes with observed contact | 0/30 | 0/30 |
| Contact proportion, descriptive Wilson 95% interval | 0–11.35% | 0–11.35% |
| Near-miss onsets | 102 | 200 |
| Near-miss exposure | 233.5 s | 330.1 s |
| Violated complete nearby 2 s windows | 1,231 / 7,206 | 2,315 / 5,049 |
| 2 s bound-violation rate | 17.08% | 45.85% |

The success gain is **15 episodes / 50 percentage points**, alongside
**98 extra near-miss onsets (+96.1%)** and **96.6 s extra near-miss exposure
(+41.4%)**. Every remaining failure is a 600-step route-completion timeout
without observed contact; the [failure ledger](station_failures.csv) lists all
45 failed arm/seed episodes individually. Their root cause is not established
by this comparison.

The versioned audit checks the full Euclidean constant-velocity prediction tube
at every sampled offset of complete 2 s windows for pedestrians initially
within 2 m centre distance. It includes errors in any direction and respawn
discontinuities; this is not the earlier toward-robot-only refute diagnostic.
Both the nearby pedestrian population and eligible-window denominator vary by
arm because the robot's path and episode duration change. Windows overlap and
are correlated. These percentages describe those arm-conditioned samples;
they do not establish a causal population change or unconditional stop safety.

## Empty-world behaviour gate

The gate uses all 48 authored scenarios, seeds 1001 and 1002, both named arms
and unchanged authored budgets: **84/96 successes, 12/96 timeouts and zero
contacts in each arm**. All 14 compared executed/counter fields match exactly
in every one of the 96 pairs. There are no changed failures or missing,
fallback or degraded episodes. All 24 failed arm/seed episodes are classified
individually as inherited actor-free route-completion timeouts in the
[failure ledger](empty_failures.csv). Matching outcomes do not prove physical
feasibility or a timeout's root cause. Bound rates are undefined in the empty
world because there are no eligible nearby pedestrian-windows.

## Evidence and custody

- [Station episodes](station/episodes.csv), [producer manifest](station/manifest.json)
  and [per-scenario tradeoff table](station/paired.csv).
- [Empty-world episodes](empty/episodes.csv), [producer manifest](empty/manifest.json)
  and [per-scenario tradeoff table](empty/paired.csv).
- [Individual empty-world failures](empty_failures.csv) and [verification summary](summary.json).

The native manifests are preserved exactly. Tracked CSV copies normalize only
CRLF line endings to LF to pass the repository whitespace gate; their review
markers record `preserved_exact_bytes: false`. Field values and row order are
unchanged. The raw archive preserves the original producer CSV bytes. Each
episode records a compressed trace name and SHA-256. The closeout checks every
trace hash and recomputes its 2 s counts from the saved sampled positions and
observed velocities: **252 trajectories verified**, with every count matching
its episode CSV. The verification summary binds the complete raw archive
hash; that archive remains in private development storage outside disposable
worktree output. These compact files permit analysis reproduction without the
raw traces; the raw archive is required to independently re-audit trajectories.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
scripts/dev/run_worktree_shared_venv.sh -- uv run python \
  scripts/analysis/analyze_predictive_braking_pair.py \
  --episodes docs/validation/predictive_braking_pair/station/episodes.csv \
  --output output/predictive_pair/reproduced_station
```

The whole pedestrian campaign (48 scenarios × 30 seeds × two arms) has not
been executed here. The roster/preflight, station slice and actor-free gate are
development implementation evidence. Allowance calibration, a near-miss budget,
methodology approval and promotion remain separate under #10111.
