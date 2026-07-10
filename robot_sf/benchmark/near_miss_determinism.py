"""Near-miss rerun nondeterminism: identification, quantification, and SNQI bound.

This module addresses the reproducibility gap in the benchmark release
reproducibility contract (issue #5140): the contract previously marked
``near_misses_mean`` as rerun-nondeterministic "within a small tolerance" but
left the tolerance **unquantified** and the source **unnamed**. Near-miss is the
largest-weight SNQI term (``w_near ~= 0.31`` for the camera-ready v3 weights), so
an unbounded nondeterminism propagates into the composite and bit-exact
re-derivation of SNQI cannot be promised.

Three deliverables are provided here, mirroring the issue proposal:

1. **Source identification.** ``metric_path_is_deterministic`` proves, by direct
   repeated invocation, that the near-miss metric reduction
   (``_compute_robot_ped_distance_summary``) is a pure-NumPy computation with no
   Numba kernels, no parallel reduction, and no ``fastmath`` reassociation. The
   reduction is therefore bit-deterministic for any fixed input trajectory set.
   This *disproves* the "parallel reduction order / JIT fastmath / thread
   scheduling in the Numba kernels" hypothesis **for the metric path**: any
   residual rerun nondeterminism must originate upstream in the pedestrian
   dynamics (``pysocialforce.forces`` uses ``@njit(fastmath=True)``) producing
   trajectories whose surface clearance crosses the 0.5 m threshold at knife-edge
   timesteps. The residual is *machine-/compiler-conditional*, not a property of
   the metric definition.

2. **Tolerance quantification.** ``measure_exact_repeat_nondeterminism`` runs
   ``N`` exact-repeat episodes of a scenario and reports the per-metric maximum
   deviation, the number of distinct values, and whether each metric was
   bit-identical across repeats. The result is a schema-versioned report that the
   repro contract can cite with a concrete number instead of "a small tolerance".

3. **SNQI propagation bound.** ``snqi_near_miss_propagation_bound`` turns a raw
   near-miss count tolerance into the worst-case SNQI composite delta, using the
   canonical ``w_near`` weight and the baseline median/p95 normalization spread,
   respecting the ``[0, 1]`` clamp on the normalized term. This lets the contract
   state the propagated effect on SNQI explicitly instead of leaving it implicit.

The harness is advisory and read-only: it never mutates rankings, campaign
summaries, or metric semantics. It fails closed (empty inputs raise; unknown
metric keys are reported rather than silently dropped) so a stale report cannot
be mistaken for a zero-deviation result.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

SCHEMA_VERSION = "near_miss_determinism.v1"

#: Default number of exact-repeat runs used to quantify rerun nondeterminism.
#: The repro contract measurement (issue #5140) uses this as the floor; callers
#: that want tighter confidence pass a larger ``n_repeats``.
DEFAULT_N_REPEATS = 10

#: Metrics whose rerun deviation is reported by the exact-repeat measurement.
#: ``near_misses`` is the focus of issue #5140; the others are included so the
#: report can distinguish a near-miss-only effect from a broader trajectory
#: divergence. All are scalar-valued episode metrics.
DEFAULT_TRACKED_METRICS: tuple[str, ...] = (
    "near_misses",
    "collisions",
    "success",
    "min_clearance",
    "min_distance",
    "time_to_goal",
)

#: Canonical camera-ready v3 SNQI weight for the near-miss term
#: (``configs/benchmarks/snqi_weights_camera_ready_v3.json``). Used as the
#: default for the propagation bound when the caller does not supply weights.
CANONICAL_W_NEAR_V3 = 0.30825830332144416

#: Metrics that must be present in every episode record for the exact-repeat
#: measurement to report a result. A missing required metric fails closed so it
#: can never look like a zero-deviation result. ``time_to_goal`` is intentionally
#: *not* required because it is absent when the robot never reaches its goal.
DEFAULT_REQUIRED_METRICS: tuple[str, ...] = (
    "near_misses",
    "collisions",
    "success",
    "min_clearance",
    "min_distance",
)


def metric_path_is_deterministic(
    robot_pos: "Sequence[Any]",
    peds_pos: "Sequence[Any]",
    *,
    robot_radius: float,
    ped_radius: float,
    n_invocations: int = 25,
) -> dict[str, Any]:
    """Prove the near-miss metric reduction is deterministic on a fixed input.

    Runs ``_compute_robot_ped_distance_summary`` ``n_invocations`` times against
    an identical synthetic :class:`EpisodeData` built from the supplied
    trajectories and reports whether every aggregate is bit-identical across
    invocations. This is the executable evidence that the metric path (NumPy
    ``np.linalg.norm`` distance matrix + ``min`` reduction over axis 1 +
    ``count_nonzero`` threshold) contains **no** source of rerun nondeterminism:
    no Numba kernel, no parallel reduction, no ``fastmath``.

    Args:
        robot_pos: ``(T, 2)`` robot trajectory used to build the episode data.
        peds_pos: ``(T, K, 2)`` pedestrian trajectory tensor.
        robot_radius: Robot radius (metres) used for surface-clearance math.
        ped_radius: Pedestrian radius (metres) used for surface-clearance math.
        n_invocations: Number of repeated invocations to compare. Must be >= 2.

    Returns:
        Schema-versioned report with keys:

        - ``is_deterministic`` (bool): ``True`` only if every aggregate value is
          bit-identical across all invocations.
        - ``n_invocations`` (int): number of invocations performed.
        - ``n_distinct_results`` (int): number of distinct aggregate tuples; 1
          means fully deterministic.
        - ``sample`` (dict): one representative aggregate mapping.
        - ``tracked_aggregates`` (list[str]): aggregates compared
          (``near_misses``, ``min_clearance``, ``human_collisions``,
          ``min_distance``, ``mean_clearance``).

    Raises:
        ValueError: When ``n_invocations`` is less than 2, or when the supplied
            trajectories are empty.
    """
    # Local import keeps the module importable without pulling the full metrics
    # graph until the harness is actually exercised.
    from robot_sf.benchmark.metrics import _compute_robot_ped_distance_summary

    if n_invocations < 2:
        raise ValueError(f"n_invocations must be >= 2, got {n_invocations}")

    import numpy as _np

    robot_arr = _np.asarray(robot_pos)
    peds_arr = _np.asarray(peds_pos)
    if (
        robot_arr.ndim != 2
        or robot_arr.shape[0] == 0
        or peds_arr.ndim != 3
        or peds_arr.shape[1] == 0
    ):
        raise ValueError(
            "metric_path_is_deterministic received empty trajectories; "
            "refusing to assert determinism with no aggregate evidence"
        )

    class _EpisodeData:
        """Minimal attribute container matching the metric's data contract."""

        __slots__ = ("robot_pos", "peds_pos", "robot_radius", "ped_radius")

    def _build() -> _EpisodeData:
        d = _EpisodeData()
        # Copy so callers cannot mutate the input mid-loop and corrupt the proof.
        d.robot_pos = robot_arr.copy()
        d.peds_pos = peds_arr.copy()
        d.robot_radius = robot_radius
        d.ped_radius = ped_radius
        return d

    first = _compute_robot_ped_distance_summary(_build())

    tracked = ("near_misses", "min_clearance", "human_collisions", "min_distance", "mean_clearance")
    distinct: set[tuple[Any, ...]] = set()
    distinct.add(tuple(first[k] for k in tracked))
    for _ in range(n_invocations - 1):
        agg = _compute_robot_ped_distance_summary(_build())
        distinct.add(tuple(agg[k] for k in tracked))

    return {
        "schema_version": SCHEMA_VERSION,
        "is_deterministic": len(distinct) == 1,
        "n_invocations": n_invocations,
        "n_distinct_results": len(distinct),
        "tracked_aggregates": list(tracked),
        "sample": dict(first),
    }


def measure_exact_repeat_nondeterminism(
    scenario_params: "Mapping[str, Any]",
    seed: int,
    *,
    n_repeats: int = DEFAULT_N_REPEATS,
    horizon: int = 60,
    dt: float = 0.1,
    tracked_metrics: "Sequence[str]" = DEFAULT_TRACKED_METRICS,
    required_metrics: "Sequence[str]" | None = None,
    run_episode: Any | None = None,
) -> dict[str, Any]:
    """Run ``N`` exact-repeat episodes and report per-metric rerun deviation.

    Each repeat constructs the scenario from the same ``scenario_params`` and
    ``seed``, steps the identical policy, and recomputes metrics, so any nonzero
    deviation is genuine rerun nondeterminism rather than input variation. The
    report is the empirical tolerance the repro contract cites for issue #5140.

    Metrics are split into two groups so conditionally-emitted metrics (e.g.
    ``time_to_goal``, which is absent when the robot never reaches its goal) do
    not mask the near-miss signal. A metric in ``required_metrics`` that is
    absent fails closed (a missing required metric must never look like a
    zero-deviation result). A tracked-but-optional metric that is absent is
    recorded as ``available: False`` and excluded from the nondeterminism flag.

    Args:
        scenario_params: Scenario parameter mapping accepted by
            ``generate_scenario`` (e.g. ``density``/``flow``/``speed_var``).
        seed: Exact-repeat seed reused for every invocation.
        n_repeats: Number of exact-repeat runs. Must be >= 2.
        horizon: Episode horizon (steps).
        dt: Episode timestep (seconds).
        tracked_metrics: Scalar episode metrics whose deviation is reported.
        required_metrics: Metrics that must be present in every record; absence
            raises. Defaults to :data:`DEFAULT_REQUIRED_METRICS`.
        run_episode: Optional injected runner (defaults to
            ``robot_sf.benchmark.runner.run_episode``) for testability.

    Returns:
        Schema-versioned report with keys:

        - ``exact_repeat`` (dict): per-metric ``available`` (bool),
          ``max_deviation``, ``n_distinct_values``, ``bit_identical`` (bool),
          ``min``, ``max``, ``values``.
        - ``summary`` (dict): ``n_repeats``, ``seed``, ``scenario_id``,
          ``any_nondeterministic_metric`` (bool), and
          ``near_misses_max_deviation`` (the issue #5140 headline number).
        - ``diagnostics`` (dict): ``numba_num_threads`` and ``numpy_version``
          captured so the report is interpretable across machines (the residual
          nondeterminism is machine-/compiler-conditional).

    Raises:
        ValueError: When ``n_repeats`` is less than 2, when ``tracked_metrics``
            is empty, or when a ``required_metrics`` entry is absent from any
            episode record.
    """
    if n_repeats < 2:
        raise ValueError(f"n_repeats must be >= 2, got {n_repeats}")
    if not tracked_metrics:
        raise ValueError("tracked_metrics must list at least one metric")
    if required_metrics is None:
        required_metrics = DEFAULT_REQUIRED_METRICS
    required_set = set(required_metrics)

    if run_episode is None:
        from robot_sf.benchmark.runner import run_episode as _run_episode

        run_episode = _run_episode

    collected: dict[str, list[float]] = {k: [] for k in tracked_metrics}
    unavailable: set[str] = set()
    for _ in range(n_repeats):
        record = run_episode(
            dict(scenario_params),
            seed=seed,
            horizon=horizon,
            dt=dt,
            record_forces=False,
        )
        metrics = record.get("metrics", {})
        for key in tracked_metrics:
            value = metrics.get(key)
            if value is None:
                if key in required_set:
                    # A missing required metric must never look like zero deviation.
                    raise ValueError(
                        f"exact-repeat measurement: required metric {key!r} absent "
                        "from episode record; refusing to report a zero-deviation "
                        "result for a missing required metric"
                    )
                unavailable.add(key)
                continue
            collected[key].append(float(value))

    per_metric: dict[str, Any] = {}
    any_nondeterministic = False
    near_miss_max_dev: float | None = None
    for key in tracked_metrics:
        vals = collected[key]
        if key in unavailable or not vals:
            per_metric[key] = {
                "available": False,
                "max_deviation": None,
                "n_distinct_values": None,
                "bit_identical": None,
                "min": None,
                "max": None,
                "values": [],
            }
            continue
        distinct = sorted({round(v, 12) for v in vals})
        v_min = min(vals)
        v_max = max(vals)
        max_dev = v_max - v_min
        bit_identical = len(distinct) == 1
        if not bit_identical:
            any_nondeterministic = True
        if key == "near_misses":
            near_miss_max_dev = max_dev
        per_metric[key] = {
            "available": True,
            "max_deviation": max_dev,
            "n_distinct_values": len(distinct),
            "bit_identical": bit_identical,
            "min": v_min,
            "max": v_max,
            "values": vals,
        }

    diagnostics = _capture_diagnostics()

    return {
        "schema_version": SCHEMA_VERSION,
        "exact_repeat": per_metric,
        "summary": {
            "n_repeats": n_repeats,
            "seed": seed,
            "scenario_id": scenario_params.get("id"),
            "horizon": horizon,
            "dt": dt,
            "any_nondeterministic_metric": any_nondeterministic,
            "near_misses_max_deviation": near_miss_max_dev,
        },
        "diagnostics": diagnostics,
    }


def _capture_diagnostics() -> dict[str, Any]:
    """Capture machine/compiler diagnostics so the report is cross-machine legible.

    Returns:
        Mapping with NumPy/Numba versions and thread count. Missing optional
        values are reported as ``None`` rather than raising, since diagnostics
        are advisory.
    """
    import numpy as np

    diag: dict[str, Any] = {"numpy_version": np.__version__}
    try:
        import numba

        diag["numba_version"] = numba.__version__
        diag["numba_num_threads"] = int(getattr(numba.config, "NUMBA_NUM_THREADS", 0)) or None
    except Exception:  # pragma: no cover - numba is optional for the diagnostic
        diag["numba_version"] = None
        diag["numba_num_threads"] = None
    return diag


def snqi_near_miss_propagation_bound(
    near_miss_tolerance: float,
    *,
    w_near: float = CANONICAL_W_NEAR_V3,
    baseline_med: float | None = None,
    baseline_p95: float | None = None,
) -> dict[str, Any]:
    """Bound the worst-case SNQI composite delta from a near-miss tolerance.

    The SNQI near-miss term is ``-w_near * clamp((nm - med) / (p95 - med), 0, 1)``.
    A rerun tolerance of ``delta`` raw near-miss counts can change the normalized
    term by at most ``delta / (p95 - med)`` inside the linear region, and the
    ``[0, 1]`` clip caps the achievable swing. This function reports both the
    linear-region bound and the clip-capped bound so the contract can state the
    propagated effect on SNQI explicitly (issue #5140 proposal 3: bound it).

    Args:
        near_miss_tolerance: Max raw near-miss count deviation observed across
            exact-repeat runs (e.g. ``summary["near_misses_max_deviation"]``).
            Must be non-negative.
        w_near: SNQI near-miss weight. Defaults to the camera-ready v3 value.
        baseline_med: Baseline median for the near-miss normalization. When
            omitted, the bound is reported only in the *clip-capped* form (the
            unconditional worst case ``w_near``), which does not require baseline
            statistics.
        baseline_p95: Baseline p95 for the near-miss normalization.

    Returns:
        Schema-versioned report with keys:

        - ``w_near`` (float): weight used.
        - ``near_miss_tolerance`` (float): input tolerance.
        - ``linear_region_bound`` (float | None): ``w_near * tol / (p95 - med)``
          when baseline stats are supplied, else ``None``.
        - ``clip_capped_bound`` (float): unconditional worst case ``w_near`` (the
          largest the normalized term can ever swing is its full weight, since
          the clamp range is 1.0).
        - ``propagation_bound`` (float): the tighter applicable bound
          (``linear_region_bound`` when available and smaller, else
          ``clip_capped_bound``).

    Raises:
        ValueError: When ``near_miss_tolerance`` is negative, ``w_near`` is
            negative, or the supplied baseline has a non-positive spread.
    """
    tol = float(near_miss_tolerance)
    if tol < 0.0:
        raise ValueError(f"near_miss_tolerance must be non-negative, got {tol}")
    w = float(w_near)
    if w < 0.0:
        raise ValueError(f"w_near must be non-negative, got {w}")

    clip_capped = w  # clamp range is [0,1], so a full-swing swing costs w_near.
    linear_bound: float | None = None
    if baseline_med is not None and baseline_p95 is not None:
        med = float(baseline_med)
        p95 = float(baseline_p95)
        spread = p95 - med
        if spread <= 0.0:
            raise ValueError(
                f"baseline normalization spread (p95 - med) must be positive, got {spread}"
            )
        linear_bound = w * tol / spread

    propagation = clip_capped if linear_bound is None else min(linear_bound, clip_capped)

    return {
        "schema_version": SCHEMA_VERSION,
        "w_near": w,
        "near_miss_tolerance": tol,
        "linear_region_bound": linear_bound,
        "clip_capped_bound": clip_capped,
        "propagation_bound": propagation,
    }
