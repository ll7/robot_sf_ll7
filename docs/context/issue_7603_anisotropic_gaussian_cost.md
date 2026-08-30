# Anisotropic Gaussian Human-Cost Planner Method Card (`anisotropic_gaussian_cost`)

**Status:** experimental / opt-in / smoke-only — implementation-integrity and deterministic-smoke
proof; not a faithful reproduction, not benchmark evidence, not a release-roster change.
**Issue:** [#7603](https://github.com/ll7/robot_sf_ll7/issues/7603) (programme parent
[#7319](https://github.com/ll7/robot_sf_ll7/issues/7319)).
**Owner module:** `robot_sf/planner/anisotropic_gaussian_cost.py`.
**Config:** `configs/algos/issue_7603_anisotropic_gaussian_cost.yaml`.

Plain-language summary: an opt-in local planner that steers a unicycle robot by combining an
attractive force toward a look-ahead goal target with repulsive negative-gradient forces derived
from motion-aligned anisotropic Gaussian human-cost fields around visible pedestrians, and repulsive
forces from static obstacles. When a pedestrian moves, their cost field elongates along their velocity
vector with velocity-scaled spread and front/rear asymmetry; when stationary, a deterministic
isotropic Gaussian limiting rule is applied without angular singularities. Hard speed and angular-rate
limits are enforced as strict predicates.

## 1. Source-to-implementation map

| Source element | Implementation | Notes |
| --- | --- | --- |
| motion-aligned anisotropic Gaussian field | `implemented` | longitudinal axis aligns with $\operatorname{atan2}(v_y, v_x)$ when $v \ge v_{\\min}$ |
| stationary pedestrian handling | `implemented` | isotropic Gaussian limiting rule with $\\sigma = \\text{stationary\\_sigma\\_m}$ |
| velocity-scaled expansion | `implemented` | $\\sigma_{\\text{long}} = \\sigma_{\\text{long},0} + \\beta \\cdot v$ |
| front/rear asymmetric elongation | `implemented` | forward sector ($d_{\\text{long}} > 0$) scaled by $\\text{asymmetry\\_front\\_ratio}$ |
| Mahalanobis / distance truncation | `implemented` | zero cost beyond $d_M > \\text{mahalanobis\\_cutoff}$ or $r > \\text{cutoff\\_distance\\_m}$ |
| repulsive gradient force | `implemented` | analytic negative gradient $-\\nabla_{\\mathbf{q}} C$ integrated into total control force |
| rate and speed constraints | `implemented` | linear and angular speeds and step-to-step rates are clipped as hard predicates |

## 2. Formulae and conventions

- **Pedestrian State**: position $\\mathbf{p}_h = (x_h, y_h)$, velocity $\\mathbf{v}_h = (v_x, v_y)$, speed $v = \\|\\mathbf{v}_h\\|$.
- **Query Displacement**: $\\mathbf{\\Delta} = \\mathbf{q} - \\mathbf{p}_h = (\\Delta x, \\Delta y)$, distance $r = \\|\\mathbf{\\Delta}\\|$.
- **Stationary Limiting Rule** ($v < v_{\\min} = 0.05\\text{ m/s}$):
  $$\\sigma = \\text{stationary\\_sigma\\_m}$$
  $$d_M = \\frac{r}{\\sigma}$$
  $$C(\\mathbf{q}) = A \\exp\\left(-\\frac{1}{2} d_M^2\\right) \\quad \\text{if } d_M \\le d_{M,\\max} \\text{ and } r \\le r_{\\max}$$
  $$\\mathbf{F}_{\\text{rep}} = C(\\mathbf{q}) \\frac{\\mathbf{\\Delta}}{\\sigma^2}$$
- **Moving Mode** ($v \ge v_{\\min}$):
  $$\\theta = \\operatorname{atan2}(v_y, v_x), \\quad \\hat{\\mathbf{u}} = (\\cos\\theta, \\sin\\theta), \\quad \\hat{\\mathbf{n}} = (-\\sin\\theta, \\cos\\theta)$$
  $$d_{\\text{long}} = \\mathbf{\\Delta} \\cdot \\hat{\\mathbf{u}}, \\quad d_{\\text{lat}} = \\mathbf{\\Delta} \\cdot \\hat{\\mathbf{n}}$$
  $$\\sigma_{\\text{long}} = (\\sigma_{\\text{long},0} + \\beta_{\\text{long}} v) \\cdot (\\alpha_{\\text{front}} \\text{ if } d_{\\text{long}} > 0 \\text{ else } 1.0)$$
  $$\\sigma_{\\text{lat}} = \\sigma_{\\text{lat},0} + \\beta_{\\text{lat}} v$$
  $$d_M^2 = \\left(\\frac{d_{\\text{long}}}{\\sigma_{\\text{long}}}\\right)^2 + \\left(\\frac{d_{\\text{lat}}}{\\sigma_{\\text{lat}}}\\right)^2$$
  $$C(\\mathbf{q}) = A \\exp\\left(-\\frac{1}{2} d_M^2\\right) \\quad \\text{if } d_M \\le d_{M,\\max} \\text{ and } r \\le r_{\\max}$$
  $$\\mathbf{F}_{\\text{rep}} = C(\\mathbf{q}) \\left[ \\frac{d_{\\text{long}}}{\\sigma_{\\text{long}}^2} \\hat{\\mathbf{u}} + \\frac{d_{\\text{lat}}}{\\sigma_{\\text{lat}}^2} \\hat{\\mathbf{n}} \\right]$$

## 3. Deviations and Unsupported Elements

- No learned trajectory prediction network; uses instantaneous linear velocity extrapolation.
- No full non-linear MPC trajectory optimization; uses direct force coupling with unicycle projection.
- Diagnostic/comparator core only; does not alter default benchmark or release rosters.

## 4. Deterministic Smoke Receipt

- **Parsed planner-config digest** (`AnisotropicGaussianCostConfig.digest()`, SHA-256):
  `4e605b1d0041d626614b96c87fb5e4a7dfc89ab33529415b3b22c1fd3652262a`.
- **Exact fixed-smoke command**:
  ```bash
  scripts/dev/run_worktree_shared_venv.sh -- uv run pytest tests/planner/test_anisotropic_gaussian_cost.py::test_fixed_smoke_scenarios
  ```
