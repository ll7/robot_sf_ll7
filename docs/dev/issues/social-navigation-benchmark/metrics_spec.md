# Metric Definitions (Draft)

Let episode index $i$, timestep $t \in \{0,\dots,T-1\}$, robot state $r_t$, pedestrians $P_t = \{ p_t^k \}$.
Use Euclidean norm $\|\cdot\|$.

## Core Metrics
1. $\text{success} = 1$ if goal reached before horizon $H$ without collision; else $0$.
2. $\text{time\_to\_goal\_norm} = \frac{\text{steps\_to\_goal}}{H}$ if success else $1.0$.
3. $\text{collisions} = \left| \{ t : \min_k d(r_t, p_t^k) < d_{\text{coll}} \} \right|$.
4. $\text{near\_misses} = \left| \{ t : d_{\text{coll}} \le \min_k d(r_t, p_t^k) < d_{\text{near}} \} \right|$.
5. $\text{min\_distance} = \min_t \min_k d(r_t, p_t^k)$.
6. $\text{path\_efficiency} = \frac{L_{\text{shortest}}}{L_{\text{actual}}}$ (clipped $\le 1$).

### Convenience / Diagnostics
- $\text{avg\_speed} = \frac{1}{T} \sum_{t=0}^{T-1} \| v_t \|$ (average robot speed magnitude).
	Useful for sanity checks and distribution plots; not part of SNQI.

## Force / Comfort Metrics
7. $\text{force\_quantiles}(q50,q90,q95)$: quantiles of all pedestrian force magnitudes $\|F_t^k\|$.
8. $\text{force\_exceed\_events} = \left| \{ (t,k): \|F_t^k\| > \tau_{\text{force}} \} \right|$.
9. $\text{comfort\_exposure} = \frac{\text{force\_exceed\_events}}{|P|\, T_{\text{eff}}}$.

## Smoothness / Energy
10. $\text{jerk\_mean} = \frac{1}{T-2} \sum_{t=0}^{T-3} \| a_{t+1} - a_t \|$ where $a_t$ is robot acceleration.
11. $\text{energy} = \sum_{t=0}^{T-1} \| a_t \|$.

## Optional Field Metrics
12. $\text{force\_gradient\_norm\_mean} = \frac{1}{M} \sum_{m=1}^M \| \nabla F(x_m,y_m) \|$ along sampled path points $(x_m,y_m)$.

## Composite Index (SNQI) (Draft)
\[
	ext{SNQI} = w_1\,\text{success} - w_2\,\text{time\_to\_goal\_norm} - w_3\,\text{norm\_collisions} - w_4\,\text{norm\_near\_misses} - w_5\,\text{comfort\_exposure} - w_6\,\text{norm\_force\_exceed} - w_7\,\text{norm\_jerk}
\]

Normalization: $\text{norm}_x = \frac{x - b_{\text{med}}}{b_{p95} - b_{\text{med}} + \varepsilon}$ using baseline distribution (random or simple planner); clamp to $[0,1]$.

## Default Constants (Initial)
- $d_{\text{coll}} = 0.25\,\text{m}$
- $d_{\text{near}} = 0.5\,\text{m}$
- $\tau_{\text{force}} =$ force $q95$ in low-density baseline scenario
- $\varepsilon = 10^{-6}$

## Open Choices
- Weight selection procedure (grid search maximizing rank stability across scenario subsets)
- Whether to include path_efficiency directly or encode via time_to_goal_norm

## Coverage Status (as of 2025-09-10)
- Implemented & documented: success, time_to_goal_norm, collisions, near_misses, min_distance, path_efficiency, force_quantiles (q50/q90/q95), force_exceed_events, comfort_exposure, jerk_mean, energy, force_gradient_norm_mean, avg_speed (diagnostic).
- Documented but partially specified/optional: mean interpersonal distance (pending), curvature-based smoothness (pending), force field divergence (pending).

## Validation & Tests
- Empty crowd: collisions=0, near_misses=0, comfort_exposure=0
- Single pedestrian static: force_exceed_events=0 for low density
- High density stress: comfort_exposure increases monotonically with density label

## Next
Finalize constants & add symbol table; implement metric module + unit tests.
