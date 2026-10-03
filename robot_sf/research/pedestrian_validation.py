"""Source-bound trajectory estimators for the development pedestrian release gate.

Protocol choices and equivalence limits are in the #10074 source audit. Distances
are metres, time is seconds. Missing/censored measurements remain None.
"""

from __future__ import annotations

import itertools

import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.optimize import least_squares


def _trace(positions, times):
    p, t = np.asarray(positions, dtype=float), np.asarray(times, dtype=float)
    if p.ndim < 2 or p.shape[0] != len(t) or p.shape[-1] != 2 or len(t) < 3:
        raise ValueError("trajectory requires at least three XY samples and matching times")
    if not np.isfinite(p).all() or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError("trajectory must be finite with increasing times")
    return p, t


def acceleration_fit(speeds, times, *, reaction_s=0.35, fit_duration_s=3.0) -> dict[str, object]:
    """Fit v0 and tau from early acceleration, using equation 4.1 of Moussaid.

    Clock starts at the cue. Fit only post-reaction acceleration, not the final
    asymptote. Simulator velocities substitute for supplementary marker data.


    Returns:
        Measurement values with explicit missingness and units.
    """
    v, t = np.asarray(speeds, float), np.asarray(times, float)
    if v.shape != t.shape or not np.isfinite(v).all() or np.any(np.diff(t) <= 0):
        raise ValueError("invalid acceleration samples")
    mask = (t >= reaction_s) & (t <= reaction_s + fit_duration_s)
    u, observed = t[mask] - reaction_s, v[mask]
    if len(u) < 4 or np.ptp(observed) < 1e-6:
        return {
            "fitted_desired_speed_m_s": None,
            "fitted_tau_s": None,
            "fit_rmse_m_s": None,
            "fit_n": len(u),
            "reason": "no identifiable acceleration",
        }
    initial = float(np.interp(reaction_s, t, v))

    def residual(params) -> np.ndarray:
        desired, tau = params
        return desired + (initial - desired) * np.exp(-u / tau) - observed

    fit = least_squares(
        residual,
        [max(float(observed.max()), 0.1), 0.5],
        bounds=([0.0, 0.01], [10.0, 10.0]),
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
    )
    return {
        "fitted_desired_speed_m_s": float(fit.x[0]),
        "fitted_tau_s": float(fit.x[1]),
        "fit_rmse_m_s": float(np.sqrt(np.mean(fit.fun**2))),
        "fit_n": len(u),
        "reason": None,
    }


def crossing_times(positions, times, plane_m) -> list[float | None]:
    """First forward crossing per person, interpolated; initial beyond-line excluded.

    Returns:
        Measurement values with explicit missingness and units.
    """
    p, t = _trace(positions, times)
    if p.ndim == 2:
        p = p[:, None, :]
    result = []
    for x in p[:, :, 0].T:
        events = np.flatnonzero((x[:-1] < plane_m) & (x[1:] >= plane_m))
        if len(events) and x[0] < plane_m:
            k = int(events[0])
            result.append(float(t[k] + (plane_m - x[k]) / (x[k + 1] - x[k]) * (t[k + 1] - t[k])))
        else:
            result.append(None)
    return result


def aperture_drop(positions, times, *, plane_m, capture_upstream_m=4.0) -> dict[str, object]:
    """Wilmut temporal phases and 3-SD event; explicit cubic LS trend equivalent.

    CM substitutes for C7, cubic LS polynomial for unspecified Higuchi trend and
    Woltring processing. Reduction is approach mean minus minimum trend before
    passage. Censored passage is null. These substitutions are never source exact.


    Returns:
        Measurement values with explicit missingness and units.
    """
    p, t = _trace(positions, times)
    passage = crossing_times(p, t, plane_m)[0]
    capture = crossing_times(p, t, plane_m - capture_upstream_m)[0]
    if capture is None and np.isclose(p[0, 0], plane_m - capture_upstream_m):
        capture = float(t[0])
    if passage is None or capture is None or passage - capture <= 2.0:
        return {
            "speed_drop_m_s": None,
            "reduction_event": None,
            "reason": "no complete capture-to-passage interval longer than approach phase",
        }
    interval = np.unique(np.r_[capture, t[(t > capture) & (t < passage)], passage])
    xy = np.column_stack([np.interp(interval, t, p[:, d]) for d in range(2)])
    speed = np.linalg.norm(np.gradient(xy, interval, axis=0, edge_order=2), axis=1)
    clock = interval - capture
    trend = np.polynomial.Polynomial.fit(clock, speed, 3)(clock)
    approach = trend[clock <= 2.0]
    mean, sd = float(approach.mean()), float(approach.std(ddof=1))
    drop = mean - float(trend[clock > 2.0].min())
    event = drop > 3 * sd
    return {
        "speed_drop_m_s": max(0.0, drop) if event else 0.0,
        "approach_speed_m_s": mean,
        "approach_sd_m_s": sd,
        "reduction_event": bool(event),
        "passage_time_s": passage,
        "capture_time_s": capture,
        "trend_rmse_m_s": float(np.sqrt(np.mean((speed - trend) ** 2))),
        "reason": None,
    }


def bottleneck_flow(
    positions, times, *, width_m, plane_m, expected_n, steady_min_s=5.0
) -> dict[str, object]:
    """Finite-N flow plus density/velocity-stable window: documented Liao equivalent.

    Exit plane explicit. One-second bins inside (plane-1, plane), five-bin density
    and speed CV <= .10. Longest contiguous stable interval >= steady_min_s wins
    (earliest breaks ties). Rectangle density substitutes for source Voronoi.


    Returns:
        Measurement values with explicit missingness and units.
    """
    p, t = _trace(positions, times)
    if width_m <= 0 or p.ndim != 3 or p.shape[1] != expected_n:
        raise ValueError("invalid bottleneck width or supply")
    crossings = crossing_times(p, t, plane_m)
    observed = sorted(v for v in crossings if v is not None)
    complete = len(observed) == expected_n
    duration = observed[-1] - observed[0] if len(observed) > 1 else 0.0
    all_flow = expected_n / duration if complete and duration > 0 else None
    velocity = np.linalg.norm(np.gradient(p, t, axis=0, edge_order=2), axis=-1)
    bins = np.arange(np.ceil(t[0]), np.floor(t[-1]) + 1, dtype=float)
    densities, speeds = [], []
    for left, right in itertools.pairwise(bins):
        frame = (t >= left) & (t < right)
        pp = p[frame]
        inside = (
            (pp[..., 0] >= plane_m - 1.0)
            & (pp[..., 0] < plane_m)
            & (np.abs(pp[..., 1]) < width_m / 2)
        )
        densities.append(float(inside.sum(axis=1).mean() / width_m) if len(pp) else 0.0)
        speeds.append(float(velocity[frame][inside].mean()) if inside.any() else 0.0)
    density, speed = np.asarray(densities), np.asarray(speeds)
    stable = np.zeros(len(density), bool)
    for start in range(len(density) - 4):
        d, v = density[start : start + 5], speed[start : start + 5]
        if min(d.min(), v.min()) > 0 and d.std() / d.mean() <= 0.10 and v.std() / v.mean() <= 0.10:
            stable[start : start + 5] = True
    starts = np.flatnonzero(stable & ~np.r_[False, stable[:-1]]) if len(stable) else []
    ends = np.flatnonzero(stable & ~np.r_[stable[1:], False]) + 1 if len(stable) else []
    windows = [
        (float(bins[a]), float(bins[b]))
        for a, b in zip(starts, ends, strict=True)
        if bins[b] - bins[a] >= steady_min_s
    ]
    window = max(windows, key=lambda w: (w[1] - w[0], -w[0])) if windows else None
    count = sum(window[0] <= c < window[1] for c in observed) if window else 0
    steady = count / (window[1] - window[0]) if window and count >= 2 else None
    return {
        "all_data_flow_persons_s": all_flow,
        "all_data_specific_flow_persons_m_s": None if all_flow is None else all_flow / width_m,
        "steady_flow_persons_s": steady,
        "steady_specific_flow_persons_m_s": None if steady is None else steady / width_m,
        "steady_window_s": window,
        "steady_crossed": count,
        "counting_plane_m": plane_m,
        "source_crossing_times_s": crossings,
        "source_crossed": len(observed),
        "complete_supply": complete,
        "stationarity_density_m2": density.tolist(),
        "stationarity_speed_m_s": speed.tolist(),
        "reason": None if complete else "finite-N all-data flow right-censored",
    }


def circumvention_clearance(positions, times, *, centre_xy, obstacle_radius_m) -> dict[str, object]:
    """CM-to-cylinder edge and abeam clearance equivalent for lateral PS extent.

    Does not estimate a fitted ellipse or longitudinal 2 m radius. No abeam
    crossing remains censored. Straight through-obstacle paths are flagged.


    Returns:
        Measurement values with explicit missingness and units.
    """
    p, t = _trace(positions, times)
    centre = np.asarray(centre_xy, float)
    distance = np.linalg.norm(p - centre, axis=-1) - obstacle_radius_m
    crossing = crossing_times(p, t, float(centre[0]))[0]
    lateral = (
        None
        if crossing is None
        else abs(float(np.interp(crossing, t, p[:, 1])) - centre[1]) - obstacle_radius_m
    )
    return {
        "minimum_cm_to_edge_m": float(distance.min()),
        "lateral_cm_to_edge_m": lateral,
        "abeam_time_s": crossing,
        "passed": bool(p[-1, 0] > centre[0] + obstacle_radius_m),
        "reason": None
        if crossing is not None
        else "obstacle not circumvented; lateral PS censored",
    }


def huber_filtered(positions, times) -> np.ndarray:
    """0.5 Hz Gaussian (-3 dB), five-sample median; linear endpoint padding.

    Gaussian cutoff normalization/endpoints are not published: these are explicit
    equivalents. Native simulator samples cannot recreate 250 Hz trunk markers.


    Returns:
        Measurement values with explicit missingness and units.
    """
    p, t = _trace(positions, times)
    dt = float(np.mean(np.diff(t)))
    if not np.allclose(np.diff(t), dt):
        raise ValueError("Huber filtering requires uniformly sampled positions")
    sigma = np.sqrt(np.log(2)) / (2 * np.pi * 0.5 * dt)
    pad = int(np.ceil(4 * sigma)) + 3
    pre = p[0] + np.arange(-pad, 0)[:, None] * (p[1] - p[0])
    post = p[-1] + np.arange(1, pad + 1)[:, None] * (p[-1] - p[-2])
    filtered = gaussian_filter1d(np.vstack([pre, p, post]), sigma=sigma, axis=0, mode="nearest")
    return median_filter(filtered, size=(5, 1), mode="nearest")[pad:-pad]


def turning_onset(
    positions,
    interferer,
    times,
    baselines,
    *,
    yaw_threshold_rad_s=0.05,
    analysis_window_m=3.0,
    persistence_s=0.3,
) -> dict[str, object]:
    """Physical yaw onset; own X distance to synchronized PoMD.

    Huber 2014 section 4.4 describes own-X distances about synchronized PoMD,
    a [-3,+3] m analysis region and baseline-derived angular-speed onset
    (https://doi.org/10.1371/journal.pone.0089589). The author ruling of
    2026-10-03 keeps the Fig. 4C engineering equivalent at 0.05 rad/s for
    deterministic bodies without sway, sustained for 0.3 s. A turn already
    underway at the left boundary is right-censored in positive onset distance:
    its onset is at least the window extent. Baselines remain diagnostics.

    Returns:
        Measurement values with explicit missingness and units.
    """
    p, t = _trace(positions, times)
    q, _ = _trace(interferer, times)
    p, q = huber_filtered(p, t), huber_filtered(q, t)
    closest = int(np.argmin(np.linalg.norm(p - q, axis=1)))

    def angular(path) -> np.ndarray:
        vel = np.gradient(path, t, axis=0, edge_order=2)
        heading = np.unwrap(np.arctan2(vel[:, 1], vel[:, 0]))
        return np.abs(np.gradient(heading, t, edge_order=2))

    maxima = []
    for base in baselines:
        bp = huber_filtered(base, t)
        mask = np.abs(bp[:, 0] - p[closest, 0]) <= analysis_window_m
        if not mask.any():
            raise ValueError("no baseline samples in captured region")
        maxima.append(float(angular(bp)[mask].max()))
    if len(maxima) != 5:
        raise ValueError("Huber onset requires five no-interferer baseline trials")
    if yaw_threshold_rad_s <= 0 or analysis_window_m <= 0 or persistence_s <= 0:
        raise ValueError("onset criterion must be positive")
    threshold = float(yaw_threshold_rad_s)
    omega = angular(p)
    valid = (p[:, 0] >= p[closest, 0] - analysis_window_m - 1e-9) & (np.arange(len(t)) < closest)
    required = max(1, int(np.ceil(persistence_s / float(np.mean(np.diff(t))) - 1e-9)))
    above = valid & (omega > threshold)
    runs = np.convolve(above.astype(int), np.ones(required, dtype=int), mode="valid")
    excess = np.flatnonzero(runs == required)
    onset = int(excess[0]) if len(excess) else None
    first_valid = np.flatnonzero(valid)
    right_censored = False
    if onset is not None and len(first_valid) and onset == first_valid[0] and onset > 0:
        boundary_x = p[closest, 0] - analysis_window_m
        previous_x, next_x = p[onset - 1, 0], p[onset, 0]
        if previous_x <= boundary_x <= next_x and next_x > previous_x:
            fraction = (boundary_x - previous_x) / (next_x - previous_x)
            boundary_yaw = omega[onset - 1] + fraction * (omega[onset] - omega[onset - 1])
            right_censored = bool(boundary_yaw > threshold)
    measured = None if onset is None else float(abs(p[closest, 0] - p[onset, 0]))
    return {
        "onset_m": analysis_window_m if right_censored else measured,
        "onset_right_censored": right_censored,
        "onset_lower_bound_m": analysis_window_m if right_censored else measured,
        "first_observed_onset_m": measured,
        "onset_time_s": None if onset is None else float(t[onset]),
        "baseline_turning_rad_s": float(np.mean(maxima)),
        "onset_threshold_rad_s": threshold,
        "onset_persistence_s": persistence_s,
        "analysis_window_m": analysis_window_m,
        "onset_definition": "fixed_yaw_source_window_lower_bound_v3",
        "baseline_maxima_rad_s": maxima,
        "pomd_time_s": float(t[closest]),
        "pomd_own_x_m": float(p[closest, 0]),
        "wrong_side": bool(p[closest, 1] < p[0, 1]),
        "reason": None if onset is not None else "no sustained physical yaw before PoMD",
    }


def pair_overlap(positions, radius_m, groups=()) -> dict[str, object]:
    """Unordered post-step pairs split by actual groups; starts reported separately.

    Strict thresholds. Zero denominator has null fractions and minimum.


    Returns:
        Measurement values with explicit missingness and units.
    """
    p = np.asarray(positions, float)
    if p.ndim != 3 or p.shape[-1] != 2 or len(p) < 2 or not np.isfinite(p).all():
        raise ValueError("invalid pair-step trajectory")
    if not np.isfinite(radius_m) or radius_m <= 0:
        raise ValueError("radius must be positive and finite")
    first, second = np.triu_indices(p.shape[1], 1)
    group_pairs, members = set(), set()
    for group in groups:
        ids = list(group)
        if len(set(ids)) != len(ids) or any(i in members or not 0 <= i < p.shape[1] for i in ids):
            raise ValueError("invalid or overlapping group membership")
        members.update(ids)
        group_pairs.update(tuple(sorted((a, b))) for j, a in enumerate(ids) for b in ids[j + 1 :])
    same = np.array(
        [(int(a), int(b)) in group_pairs for a, b in zip(first, second, strict=True)], bool
    )
    result = {}
    for name, mask in (("group", same), ("non_group", ~same), ("all", np.ones(len(first), bool))):
        a, b = first[mask], second[mask]
        count_radius = count_fixed = total = 0
        minimum = np.inf
        for start in range(1, len(p), 64):
            dist = np.linalg.norm(p[start : start + 64, a] - p[start : start + 64, b], axis=-1)
            total += dist.size
            count_radius += int(np.count_nonzero(dist < 2 * radius_m))
            count_fixed += int(np.count_nonzero(dist < 0.45))
            if dist.size:
                minimum = min(minimum, float(dist.min()))
        initial = np.linalg.norm(p[0, a] - p[0, b], axis=-1)
        result[name] = {
            "pair_steps": total,
            "below_2r_count": count_radius,
            "below_0_45_count": count_fixed,
            "share_below_2r": count_radius / total if total else None,
            "share_below_0_45": count_fixed / total if total else None,
            "minimum_centre_distance_m": minimum if total else None,
            "initial_overlapping_pairs": int(np.count_nonzero(initial < 2 * radius_m)),
        }
    return result
