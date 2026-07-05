"""Recompute Chapter 8 dissertation statistics from explicit source data.

The helpers in this module are intentionally small and data-driven. They do not
load dissertation prose or infer claims; callers must provide the numeric source
values and expected targets in a manifest.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class StatisticResult:
    """Computed statistic plus fail-closed status metadata."""

    statistic_id: str
    statistic_kind: str
    status: str
    computed: dict[str, Any]
    expected: dict[str, Any]
    blockers: tuple[str, ...] = ()

    def to_json(self) -> dict[str, Any]:
        """Return stable JSON-ready representation for packet artifacts."""

        return {
            "id": self.statistic_id,
            "kind": self.statistic_kind,
            "status": self.status,
            "computed": self.computed,
            "expected": self.expected,
            "blockers": list(self.blockers),
        }


def partial_eta_squared(groups: dict[str, list[float]]) -> float:
    """Compute one-way partial eta squared from named treatment groups.

    Returns:
        Partial eta squared value for the supplied group observations.
    """

    clean_groups = _validate_groups(groups)
    values = np.concatenate([np.asarray(group, dtype=float) for group in clean_groups.values()])
    overall_mean = float(np.mean(values))
    ss_effect = 0.0
    ss_error = 0.0
    for group in clean_groups.values():
        group_array = np.asarray(group, dtype=float)
        group_mean = float(np.mean(group_array))
        ss_effect += len(group_array) * (group_mean - overall_mean) ** 2
        ss_error += float(np.sum((group_array - group_mean) ** 2))
    denominator = ss_effect + ss_error
    if denominator <= 0.0:
        raise ValueError("partial_eta_squared denominator is zero")
    return float(ss_effect / denominator)


def spearman_rho(x_values: list[float], y_values: list[float]) -> float:
    """Compute Spearman rank correlation for paired finite samples.

    Returns:
        Spearman rho statistic.
    """

    x_array = _validate_sequence(x_values, name="x_values", min_length=2)
    if not isinstance(y_values, list) or len(x_array) != len(y_values):
        raise ValueError("x_values and y_values must have the same length")
    y_array = _validate_sequence(y_values, name="y_values", min_length=2)
    if len(x_array) != len(y_array):
        raise ValueError("x_values and y_values must have the same length")
    rho = stats.spearmanr(x_array, y_array).statistic
    if not math.isfinite(float(rho)):
        raise ValueError("spearman_rho is not finite")
    return float(rho)


def bootstrap_mean_ci(
    values: list[float],
    *,
    samples: int,
    confidence_level: float,
    seed: int,
) -> dict[str, float | int | list[float]]:
    """Compute a deterministic bootstrap confidence interval for the mean.

    Returns:
        Summary with original mean, bootstrap mean, confidence interval, seed, and sample count.
    """

    value_array = _validate_sequence(values, name="values", min_length=2)
    if samples <= 0:
        raise ValueError("samples must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(value_array), size=(samples, len(value_array)))
    means = value_array[indices].mean(axis=1)
    alpha = 1.0 - confidence_level
    ci = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {
        "samples": samples,
        "seed": seed,
        "confidence_level": confidence_level,
        "mean": float(np.mean(value_array)),
        "bootstrap_mean": float(np.mean(means)),
        "ci": [float(ci[0]), float(ci[1])],
    }


def evaluate_statistic(spec: dict[str, Any]) -> StatisticResult:
    """Evaluate one manifest statistic and return fail-closed status.

    Returns:
        Structured result row for the reproducibility packet.
    """

    statistic_id = _required_str(spec, "id")
    statistic_kind = _required_str(spec, "statistic_kind")
    data = spec.get("data")
    expected = _expected_block(spec)
    blockers: list[str] = []

    if not isinstance(data, dict) or not data:
        return StatisticResult(
            statistic_id=statistic_id,
            statistic_kind=statistic_kind,
            status="blocked_missing_source_data",
            computed={},
            expected=expected,
            blockers=("data block is missing or empty",),
        )

    try:
        computed = _compute_by_kind(statistic_kind, data)
    except (KeyError, TypeError, ValueError) as exc:
        return StatisticResult(
            statistic_id=statistic_id,
            statistic_kind=statistic_kind,
            status="blocked_invalid_source_data",
            computed={},
            expected=expected,
            blockers=(str(exc),),
        )

    status = "recomputed"
    if expected:
        try:
            blockers.extend(_expected_blockers(computed, expected))
        except (KeyError, TypeError, ValueError) as exc:
            return StatisticResult(
                statistic_id=statistic_id,
                statistic_kind=statistic_kind,
                status="blocked_invalid_source_data",
                computed=computed,
                expected=expected,
                blockers=(f"invalid expected block: {exc}",),
            )
        status = "matches_expected" if not blockers else "computed_mismatch"
    else:
        status = "computed_expected_value_missing"
        blockers.append("expected value block is missing")

    return StatisticResult(
        statistic_id=statistic_id,
        statistic_kind=statistic_kind,
        status=status,
        computed=computed,
        expected=expected,
        blockers=tuple(blockers),
    )


def _compute_by_kind(statistic_kind: str, data: dict[str, Any]) -> dict[str, Any]:
    if "rows" in data:
        rows = data["rows"]
        if statistic_kind == "partial_eta_squared":
            metric = data["metric"]
            return _variance_decomposition_ch8(rows, metric)
        if statistic_kind == "spearman_rho":
            x_field = data["x_field"]
            y_field = data["y_field"]
            return _spearman_ch8(rows, x_field, y_field)
        if statistic_kind == "bootstrap_mean_ci":
            metric = data["metric"]
            samples = int(data["samples"])
            seed = int(data["seed"])
            planner = data.get("planner")
            res = _rank_stability_bootstrap_ch8(rows, metric, samples, seed)
            if planner:
                p_res = res.get(planner)
                if not p_res:
                    raise ValueError(f"planner {planner} not found in bootstrap results")
                return {
                    "samples": samples,
                    "seed": seed,
                    "observed_rank": p_res["observed"],
                    "rank_ci": [p_res["ci_lo"], p_res["ci_hi"]],
                }
            else:
                return {
                    "samples": samples,
                    "seed": seed,
                    "rank_ci_by_planner": {
                        p: [d["ci_lo"], d["ci_hi"]] for p, d in res.items() if p != "ppo"
                    },
                }
        raise ValueError(f"unsupported statistic_kind: {statistic_kind}")

    if statistic_kind == "partial_eta_squared":
        return {"value": partial_eta_squared(data["groups"])}
    if statistic_kind == "spearman_rho":
        return {"value": spearman_rho(data["x_values"], data["y_values"])}
    if statistic_kind == "bootstrap_mean_ci":
        return bootstrap_mean_ci(
            data["values"],
            samples=int(data["samples"]),
            confidence_level=float(data["confidence_level"]),
            seed=int(data["seed"]),
        )
    raise ValueError(f"unsupported statistic_kind: {statistic_kind}")


def _expected_block(spec: dict[str, Any]) -> dict[str, Any]:
    expected = spec.get("expected")
    return expected if isinstance(expected, dict) else {}


def _expected_blockers(computed: dict[str, Any], expected: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    tolerance = float(expected.get("tolerance", 1e-9))
    _check_value(computed, expected, blockers, tolerance)
    _check_ci(computed, expected, blockers, tolerance)
    _check_rank_ci(computed, expected, blockers)
    _check_rank_ci_by_planner(computed, expected, blockers)
    _check_eta_squared(computed, expected, blockers, tolerance)
    if "samples" in expected and computed.get("samples") != expected["samples"]:
        blockers.append(
            f"samples {computed.get('samples')} does not match expected {expected['samples']}"
        )
    return blockers


def _check_value(
    computed: dict[str, Any], expected: dict[str, Any], blockers: list[str], tolerance: float
) -> None:
    if "value" in expected:
        expected_value = float(expected["value"])
        computed_value = float(computed.get("value", computed.get("mean", math.nan)))
        if not math.isclose(computed_value, expected_value, rel_tol=tolerance, abs_tol=tolerance):
            blockers.append(f"value {computed_value} does not match expected {expected_value}")


def _check_ci(
    computed: dict[str, Any], expected: dict[str, Any], blockers: list[str], tolerance: float
) -> None:
    if "ci" in expected:
        computed_ci = computed.get("ci")
        expected_ci = expected["ci"]
        if not isinstance(computed_ci, list) or len(computed_ci) != 2:
            blockers.append("computed ci is missing")
        elif not isinstance(expected_ci, list) or len(expected_ci) != 2:
            blockers.append("expected ci must contain two values")
        else:
            for index, (actual, target) in enumerate(zip(computed_ci, expected_ci, strict=True)):
                if not math.isclose(
                    float(actual), float(target), rel_tol=tolerance, abs_tol=tolerance
                ):
                    blockers.append(f"ci[{index}] {actual} does not match expected {target}")


def _check_rank_ci(computed: dict[str, Any], expected: dict[str, Any], blockers: list[str]) -> None:
    if "rank_ci" in expected:
        computed_rank_ci = computed.get("rank_ci")
        expected_rank_ci = expected["rank_ci"]
        if not isinstance(computed_rank_ci, list) or len(computed_rank_ci) != 2:
            blockers.append("computed rank_ci is missing or invalid")
        elif list(computed_rank_ci) != list(expected_rank_ci):
            blockers.append(
                f"rank_ci {computed_rank_ci} does not match expected {expected_rank_ci}"
            )


def _check_rank_ci_by_planner(
    computed: dict[str, Any], expected: dict[str, Any], blockers: list[str]
) -> None:
    if "rank_ci_by_planner" in expected:
        computed_by_planner = computed.get("rank_ci_by_planner")
        expected_by_planner = expected["rank_ci_by_planner"]
        if not isinstance(computed_by_planner, dict):
            blockers.append("computed rank_ci_by_planner is missing or invalid")
        else:
            for planner, expected_ci in expected_by_planner.items():
                actual_ci = computed_by_planner.get(planner)
                if not isinstance(actual_ci, list) or len(actual_ci) != 2:
                    blockers.append(f"computed rank_ci for {planner} is missing or invalid")
                elif list(actual_ci) != list(expected_ci):
                    blockers.append(
                        f"rank_ci for {planner} {actual_ci} does not match expected {expected_ci}"
                    )


def _check_eta_squared(
    computed: dict[str, Any], expected: dict[str, Any], blockers: list[str], tolerance: float
) -> None:
    for key in ("scenario_family_eta_squared", "planner_eta_squared", "interaction_eta_squared"):
        if key in expected:
            expected_val = float(expected[key])
            computed_val = float(computed.get(key, math.nan))
            if not math.isclose(computed_val, expected_val, rel_tol=tolerance, abs_tol=tolerance):
                blockers.append(f"{key} {computed_val} does not match expected {expected_val}")


def _parse_float(raw: Any) -> float | None:
    if raw is None:
        return None
    s = str(raw).strip().lstrip("'").strip()
    if s == "" or s.lower() == "nan":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _mean_ch8(xs: list[float | None]) -> float:
    valid = [x for x in xs if x is not None]
    return sum(valid) / len(valid) if valid else 0.0


def _variance_decomposition_ch8(rows: list[dict[str, Any]], metric: str) -> dict[str, float]:
    cells = {}
    planners, families = set(), set()
    for r in rows:
        v = _parse_float(r.get(metric))
        if v is None:
            continue
        p = r.get("planner_key")
        fam = r.get("scenario_family")
        if p and fam:
            cells[(p, fam)] = v
            planners.add(p)
            families.add(fam)
    P, F = sorted(planners), sorted(families)
    vals = list(cells.values())
    n = len(vals)
    if n == 0:
        raise ValueError(f"no valid cells found for metric: {metric}")
    grand = sum(vals) / n
    ss_total = sum((v - grand) ** 2 for v in vals)
    pmean = {p: _mean_ch8([cells[(p, fam)] for fam in F if (p, fam) in cells]) for p in P}
    fmean = {fam: _mean_ch8([cells[(p, fam)] for p in P if (p, fam) in cells]) for fam in F}
    ss_planner = sum(
        len([fam for fam in F if (p, fam) in cells]) * (pmean[p] - grand) ** 2 for p in P
    )
    ss_family = sum(
        len([p for p in P if (p, fam) in cells]) * (fmean[fam] - grand) ** 2 for fam in F
    )
    ss_inter = ss_total - ss_planner - ss_family
    return {
        "scenario_family_eta_squared": ss_family / ss_total if ss_total else 0.0,
        "planner_eta_squared": ss_planner / ss_total if ss_total else 0.0,
        "interaction_eta_squared": ss_inter / ss_total if ss_total else 0.0,
    }


def _ranks_ch8(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            out[order[k]] = avg
        i = j + 1
    return out


def _pearson_ch8(x: list[float], y: list[float]) -> float | None:
    n = len(x)
    if n < 3:
        return None
    mx, my = sum(x) / n, sum(y) / n
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y, strict=True))
    if sxx == 0 or syy == 0:
        return None
    return sxy / math.sqrt(sxx * syy)


def _spearman_ch8(rows: list[dict[str, Any]], x_field: str, y_field: str) -> dict[str, float]:
    pairs = []
    for r in rows:
        x_val = _parse_float(r.get(x_field))
        y_val = _parse_float(r.get(y_field))
        if x_val is not None and y_val is not None:
            pairs.append((x_val, y_val))
    if len(pairs) < 3:
        raise ValueError("less than 3 pairs found for Spearman correlation")
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    val = _pearson_ch8(_ranks_ch8(xs), _ranks_ch8(ys))
    if val is None or not math.isfinite(val):
        raise ValueError("computed Spearman rho is not finite")
    return {"value": float(val)}


def _rank_stability_bootstrap_ch8(
    rows: list[dict[str, Any]], metric: str, n_boot: int, seed: int
) -> dict[str, dict[str, Any]]:
    planners = sorted({r["planner_key"] for r in rows if r.get("planner_key")})
    families = sorted({r["scenario_family"] for r in rows if r.get("scenario_family")})
    cell = {p: {} for p in planners}
    for r in rows:
        v = _parse_float(r.get(metric))
        if v is not None:
            cell[r["planner_key"]][r["scenario_family"]] = v

    def rank_by_mean(sampled_families: list[str]) -> dict[str, int]:
        means = {}
        for p in planners:
            vals = [cell[p][f] for f in sampled_families if f in cell[p]]
            means[p] = _mean_ch8(vals) if vals else 0.0
        order = sorted(planners, key=lambda p: -means[p])
        return {p: order.index(p) + 1 for p in planners}

    observed = rank_by_mean(families)
    state = seed

    def _rand_int(n: int) -> int:
        nonlocal state
        state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
        return state % n

    boot_ranks = {p: [] for p in planners}
    for _ in range(n_boot):
        sample = [families[_rand_int(len(families))] for _ in range(len(families))]
        for p, r in rank_by_mean(sample).items():
            boot_ranks[p].append(r)
    result = {}
    for p in planners:
        rs = sorted(boot_ranks[p])
        lo = rs[int(0.025 * n_boot)]
        hi = rs[int(0.975 * n_boot)]
        result[p] = {"observed": observed[p], "ci_lo": lo, "ci_hi": hi}
    return result


def _required_str(spec: dict[str, Any], key: str) -> str:
    value = spec.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"statistic spec requires non-empty string field: {key}")
    return value


def _validate_groups(groups: dict[str, list[float]]) -> dict[str, list[float]]:
    if not isinstance(groups, dict) or len(groups) < 2:
        raise ValueError("groups must contain at least two named groups")
    clean: dict[str, list[float]] = {}
    for name, values in groups.items():
        if not isinstance(name, str) or not name:
            raise ValueError("group names must be non-empty strings")
        clean[name] = _validate_sequence(values, name=f"groups.{name}", min_length=1).tolist()
    if sum(len(values) for values in clean.values()) <= len(clean):
        raise ValueError("groups need at least one residual degree of freedom")
    return clean


def _validate_sequence(values: list[float], *, name: str, min_length: int) -> np.ndarray:
    if not isinstance(values, list) or len(values) < min_length:
        raise ValueError(f"{name} must contain at least {min_length} values")
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D sequence")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array
