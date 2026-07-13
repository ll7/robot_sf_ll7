"""Latency-stress preflight contract helpers for learned-policy diagnostics."""

from __future__ import annotations

import os
import platform
import sys
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

_PLANNER_UPDATE_MODES = {"every-step", "hold-last"}
_DEFAULT_NON_SUCCESS_STATUSES = (
    "fallback",
    "degraded",
    "timeout",
    "not_available",
    "failed",
)
_THREAD_SETTING_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


@dataclass(frozen=True)
class LatencyStressProfile:
    """Serializable preflight-only latency stress profile for benchmark diagnostics."""

    name: str
    observation_delay_steps: int = 0
    action_delay_steps: int = 0
    planner_update_mode: str = "every-step"
    planner_update_period_steps: int = 1
    inference_timeout_ms: float | None = None
    profile_version: str = "v0"
    claim_scope: str = "synthetic-only"
    non_success_statuses: tuple[str, ...] = field(
        default_factory=lambda: _DEFAULT_NON_SUCCESS_STATUSES
    )

    def to_metadata(self, *, dt: float | None = None) -> dict[str, Any]:
        """Return a JSON-safe metadata payload.

        ``dt`` is optional so config/preflight paths can emit a stable contract before a
        runner-specific step size is known.
        """
        validate_latency_stress_profile(self)
        observation_delay_ms = None
        action_delay_ms = None
        planner_update_interval_ms = None
        if dt is not None and float(dt) > 0.0:
            step_ms = float(dt) * 1000.0
            observation_delay_ms = float(self.observation_delay_steps * step_ms)
            action_delay_ms = float(self.action_delay_steps * step_ms)
            planner_update_interval_ms = float(self.planner_update_period_steps * step_ms)
        return {
            "schema_version": "latency-stress-profile.v1",
            "name": self.name,
            "profile_version": self.profile_version,
            "claim_scope": self.claim_scope,
            "observation_delay_steps": int(self.observation_delay_steps),
            "observation_delay_ms": observation_delay_ms,
            "action_delay_steps": int(self.action_delay_steps),
            "action_delay_ms": action_delay_ms,
            "planner_update_mode": self.planner_update_mode,
            "planner_update_period_steps": int(self.planner_update_period_steps),
            "planner_update_interval_ms": planner_update_interval_ms,
            "inference_timeout_ms": (
                float(self.inference_timeout_ms) if self.inference_timeout_ms is not None else None
            ),
            "non_success_statuses": list(self.non_success_statuses),
            "contract_scope": "preflight-and-provenance-only",
        }


def known_planner_update_modes() -> tuple[str, ...]:
    """Return supported planner update modes."""
    return tuple(sorted(_PLANNER_UPDATE_MODES))


def _optional_text(payload: dict[str, Any], key: str, *, default: str) -> str:
    """Return a normalized optional string field.

    ``None`` means absent for optional fields. Non-string values fail before they can be silently
    coerced into misleading contract metadata.
    """
    raw = payload.get(key, default)
    if raw is None:
        raw = default
    if not isinstance(raw, str):
        raise TypeError(f"latency_stress_profile.{key} must be a string")
    return raw.strip()


def _optional_int(payload: dict[str, Any], key: str, *, default: int) -> int:
    """Return a normalized optional integer field."""
    raw = payload.get(key, default)
    if raw is None:
        raw = default
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise TypeError(f"latency_stress_profile.{key} must be an integer")
    return int(raw)


def _optional_float(payload: dict[str, Any], key: str) -> float | None:
    """Return a normalized optional float field."""
    raw = payload.get(key)
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, int | float):
        raise TypeError(f"latency_stress_profile.{key} must be a number")
    return float(raw)


def _normalize_non_success_statuses(raw: Any) -> tuple[str, ...]:
    """Normalize non-success status labels without coercing invalid entries.

    Returns:
        Normalized status labels.
    """
    if raw is None:
        return _DEFAULT_NON_SUCCESS_STATUSES
    if isinstance(raw, str):
        return (raw.strip(),)
    if isinstance(raw, (list, tuple)):
        statuses: list[str] = []
        for value in raw:
            if not isinstance(value, str):
                raise TypeError(
                    "latency_stress_profile.non_success_statuses entries must be strings"
                )
            statuses.append(value.strip())
        return tuple(statuses)
    raise TypeError("latency_stress_profile.non_success_statuses must be a list")


def load_latency_stress_profile(payload: Any) -> LatencyStressProfile | None:
    """Normalize optional latency-stress payloads into the typed profile contract.

    Returns:
        A validated profile, or ``None`` when the payload is absent.
    """
    if payload is None:
        return None
    if isinstance(payload, LatencyStressProfile):
        validate_latency_stress_profile(payload)
        return payload
    if not isinstance(payload, dict):
        raise TypeError("latency_stress_profile must be a mapping when provided")
    profile = LatencyStressProfile(
        name=_optional_text(payload, "name", default=""),
        profile_version=_optional_text(payload, "profile_version", default="v0") or "v0",
        claim_scope=_optional_text(payload, "claim_scope", default="synthetic-only")
        or "synthetic-only",
        observation_delay_steps=_optional_int(payload, "observation_delay_steps", default=0),
        action_delay_steps=_optional_int(payload, "action_delay_steps", default=0),
        planner_update_mode=(
            _optional_text(payload, "planner_update_mode", default="every-step").lower()
            or "every-step"
        ),
        planner_update_period_steps=_optional_int(
            payload,
            "planner_update_period_steps",
            default=1,
        ),
        inference_timeout_ms=_optional_float(payload, "inference_timeout_ms"),
        non_success_statuses=_normalize_non_success_statuses(
            payload.get("non_success_statuses", _DEFAULT_NON_SUCCESS_STATUSES)
        ),
    )
    validate_latency_stress_profile(profile)
    return profile


def validate_latency_stress_profile(profile: LatencyStressProfile) -> None:  # noqa: C901, PLR0912
    """Validate that one latency-stress profile is usable as a diagnostic contract."""
    if not isinstance(profile.name, str):
        raise TypeError("latency_stress_profile.name must be a string")
    if not profile.name.strip():
        raise ValueError("latency_stress_profile.name must be non-empty")
    if not isinstance(profile.profile_version, str):
        raise TypeError("latency_stress_profile.profile_version must be a string")
    if not profile.profile_version.strip():
        raise ValueError("latency_stress_profile.profile_version must be non-empty")
    if not isinstance(profile.claim_scope, str):
        raise TypeError("latency_stress_profile.claim_scope must be a string")
    if profile.claim_scope.strip() != "synthetic-only":
        raise ValueError("latency_stress_profile.claim_scope must be 'synthetic-only'")
    if isinstance(profile.observation_delay_steps, bool) or not isinstance(
        profile.observation_delay_steps,
        int,
    ):
        raise TypeError("latency_stress_profile.observation_delay_steps must be an integer")
    if profile.observation_delay_steps < 0:
        raise ValueError("latency_stress_profile.observation_delay_steps must be >= 0")
    if isinstance(profile.action_delay_steps, bool) or not isinstance(
        profile.action_delay_steps,
        int,
    ):
        raise TypeError("latency_stress_profile.action_delay_steps must be an integer")
    if profile.action_delay_steps < 0:
        raise ValueError("latency_stress_profile.action_delay_steps must be >= 0")
    if not isinstance(profile.planner_update_mode, str):
        raise TypeError("latency_stress_profile.planner_update_mode must be a string")
    if profile.planner_update_mode not in _PLANNER_UPDATE_MODES:
        known = ", ".join(known_planner_update_modes())
        raise ValueError(
            "Unsupported latency_stress_profile.planner_update_mode "
            f"'{profile.planner_update_mode}'. Expected one of: {known}"
        )
    if isinstance(profile.planner_update_period_steps, bool) or not isinstance(
        profile.planner_update_period_steps,
        int,
    ):
        raise TypeError("latency_stress_profile.planner_update_period_steps must be an integer")
    if profile.planner_update_mode == "every-step" and profile.planner_update_period_steps != 1:
        raise ValueError(
            "latency_stress_profile.planner_update_period_steps must be 1 for every-step"
        )
    if profile.planner_update_mode == "hold-last" and profile.planner_update_period_steps < 2:
        raise ValueError(
            "latency_stress_profile.planner_update_period_steps must be >= 2 for hold-last"
        )
    if (
        profile.inference_timeout_ms is not None
        and (isinstance(profile.inference_timeout_ms, bool))
    ) or (
        profile.inference_timeout_ms is not None
        and not isinstance(profile.inference_timeout_ms, int | float)
    ):
        raise TypeError("latency_stress_profile.inference_timeout_ms must be a number")
    if profile.inference_timeout_ms is not None and profile.inference_timeout_ms <= 0.0:
        raise ValueError("latency_stress_profile.inference_timeout_ms must be > 0 when set")
    if isinstance(profile.non_success_statuses, str) or not isinstance(
        profile.non_success_statuses,
        tuple,
    ):
        raise TypeError("latency_stress_profile.non_success_statuses must be a tuple")
    if any(not isinstance(value, str) for value in profile.non_success_statuses):
        raise TypeError("latency_stress_profile.non_success_statuses entries must be strings")
    normalized_statuses = tuple(
        value.strip() for value in profile.non_success_statuses if value.strip()
    )
    if not normalized_statuses:
        raise ValueError("latency_stress_profile.non_success_statuses must be non-empty")
    required = set(_DEFAULT_NON_SUCCESS_STATUSES)
    if not required.issubset(normalized_statuses):
        missing = ", ".join(sorted(required.difference(normalized_statuses)))
        raise ValueError(
            "latency_stress_profile.non_success_statuses must include fallback, degraded, "
            f"timeout, not_available, and failed; missing: {missing}"
        )


def not_available_latency_metrics() -> dict[str, str]:
    """Return explicit placeholders for latency metrics not measured by this preflight contract."""
    return {
        "observation_age_steps": "not_available",
        "observation_age_ms": "not_available",
        "held_action_ratio": "not_available",
        "planner_update_interval_steps": "not_available",
        "planner_update_interval_ms": "not_available",
        "inference_timeout_count": "not_available",
        "inference_fallback_count": "not_available",
        "synthetic_actuation_delay_steps": "not_available",
    }


class LatencyContext:
    """Context manager to manage the thread-local active LatencyMeasurementHarness."""

    _local = threading.local()

    @classmethod
    def get_current(cls) -> LatencyMeasurementHarness | None:
        """Get the active latency measurement harness from the thread-local context.

        Returns:
            LatencyMeasurementHarness | None: The active harness if set, else None.
        """
        return getattr(cls._local, "current", None)

    def __init__(self, harness: LatencyMeasurementHarness):
        """Initialize the context with a harness instance."""
        self.harness = harness

    def __enter__(self) -> LatencyContext:
        """Enter the context and set the active harness thread-locally.

        Returns:
            LatencyContext: The context manager instance.
        """
        self._old = getattr(self._local, "current", None)
        self._local.current = self.harness
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit the context and restore the previous active harness."""
        self._local.current = self._old


class LatencyMeasurementHarness:
    """Harness to measure planning step latencies and sub-component breakdowns."""

    @classmethod
    def get_current(cls) -> LatencyMeasurementHarness | None:
        """Get the active latency measurement harness from the thread-local context.

        Returns:
            LatencyMeasurementHarness | None: The active harness if set, else None.
        """
        return LatencyContext.get_current()

    def __init__(
        self,
        deadline_ms: float = 100.0,
        target_hardware: str | None = None,
        measured_host_is_embedded: bool = False,
        *,
        config_hash: str | None = None,
        measured_host_identity: str | None = None,
    ):
        """Initialize the latency measurement harness with a budget and target hardware."""
        self.deadline_ms = deadline_ms
        self.target_hardware = target_hardware
        self.measured_host_is_embedded = measured_host_is_embedded
        self.config_hash = config_hash
        self.measured_host_identity = measured_host_identity
        self.cycles: list[dict[str, float]] = []
        self.current_accumulator: dict[str, float] | None = None
        self.step_start_time: float | None = None

    def __enter__(self) -> LatencyMeasurementHarness:
        """Enter the latency measurement context and apply instrumentation.

        Returns:
            LatencyMeasurementHarness: The harness instance.
        """
        apply_latency_instrumentation()
        self._context = LatencyContext(self)
        self._context.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit the latency measurement context."""
        self._context.__exit__(exc_type, exc_val, exc_tb)

    def wrap_policy(  # noqa: C901
        self, policy_fn: Callable[[dict[str, Any]], Any]
    ) -> Callable[[dict[str, Any]], Any]:
        """Wrap policy_fn and instrument any underlying adapter and filters.

        Returns:
            Callable[[dict[str, Any]], Any]: The wrapped policy callable.
        """
        adapter = getattr(policy_fn, "_planner_adapter", None)
        cbf_filter = getattr(policy_fn, "_cbf_filter", None)
        policy_meta = getattr(policy_fn, "_meta", None)
        if isinstance(policy_meta, Mapping):
            policy_config_hash = policy_meta.get("config_hash")
            if policy_config_hash is not None and not isinstance(policy_config_hash, str):
                raise TypeError("policy metadata config_hash must be a string")
            if policy_config_hash:
                if self.config_hash is not None and self.config_hash != policy_config_hash:
                    raise ValueError("Latency harness config_hash disagrees with policy metadata")
                self.config_hash = policy_config_hash

        if adapter:
            instrument_adapter_for_latency(adapter)

        if cbf_filter and not hasattr(cbf_filter.filter_command, "_instrumented"):
            orig_filter = cbf_filter.filter_command

            def wrapped_filter(*args: Any, **kwargs: Any) -> Any:
                t0 = time.perf_counter()
                res = orig_filter(*args, **kwargs)
                active_harness = LatencyMeasurementHarness.get_current()
                if active_harness is not None:
                    active_harness.add_time(
                        "collision_risk_safety_filter", (time.perf_counter() - t0) * 1000.0
                    )
                return res

            wrapped_filter._instrumented = True  # type: ignore[attr-defined]
            cbf_filter.filter_command = wrapped_filter

        def wrapped_policy(obs: dict[str, Any]) -> Any:
            active_harness = LatencyMeasurementHarness.get_current()
            if active_harness is not None and active_harness.current_accumulator is not None:
                t0 = time.perf_counter()
                res = policy_fn(obs)
                if hasattr(policy_fn, "_last_step_native"):
                    wrapped_policy._last_step_native = policy_fn._last_step_native  # type: ignore[attr-defined]
                total_policy_time = (time.perf_counter() - t0) * 1000.0
                used_components = (
                    active_harness.current_accumulator["observation_construction"]
                    + active_harness.current_accumulator["prediction"]
                    + active_harness.current_accumulator["action_conversion"]
                    + active_harness.current_accumulator["collision_risk_safety_filter"]
                )
                active_harness.current_accumulator["planner_computation"] += max(
                    0.0, total_policy_time - used_components
                )
                return res
            res = policy_fn(obs)
            if hasattr(policy_fn, "_last_step_native"):
                wrapped_policy._last_step_native = policy_fn._last_step_native  # type: ignore[attr-defined]
            return res

        # Preserve any planner reset/close attributes
        for attr in [
            "_planner_reset",
            "_planner_close",
            "_planner_bind_env",
            "_planner_stats",
            "_planner_native_env_action",
            "_last_step_native",
        ]:
            if hasattr(policy_fn, attr):
                setattr(wrapped_policy, attr, getattr(policy_fn, attr))

        return wrapped_policy

    def start_cycle(self) -> None:
        """Start a new measurement cycle for the planning step."""
        self.current_accumulator = {
            "observation_construction": 0.0,
            "prediction": 0.0,
            "planner_computation": 0.0,
            "collision_risk_safety_filter": 0.0,
            "action_conversion": 0.0,
        }
        self.step_start_time = time.perf_counter()

    def add_time(self, component: str, duration_ms: float) -> None:
        """Accumulate execution duration for a specific planning component."""
        if self.current_accumulator is not None:
            self.current_accumulator[component] += duration_ms

    def end_cycle(self) -> None:
        """End the current measurement cycle and record total and component latencies."""
        if self.step_start_time is not None and self.current_accumulator is not None:
            total_time = (time.perf_counter() - self.step_start_time) * 1000.0
            sum_other = (
                self.current_accumulator["observation_construction"]
                + self.current_accumulator["prediction"]
                + self.current_accumulator["collision_risk_safety_filter"]
                + self.current_accumulator["action_conversion"]
            )
            # Ensure planner_computation is non-negative
            self.current_accumulator["planner_computation"] = max(
                self.current_accumulator["planner_computation"], 0.0
            )

            # Ensure total_time is at least the sum of components
            total_time = max(
                total_time, sum_other + self.current_accumulator["planner_computation"]
            )

            # Guarantee component-sum consistency by adjusting planner_computation
            self.current_accumulator["planner_computation"] = total_time - sum_other

            self.cycles.append(
                {
                    "observation_construction_ms": self.current_accumulator[
                        "observation_construction"
                    ],
                    "prediction_ms": self.current_accumulator["prediction"],
                    "planner_computation_ms": self.current_accumulator["planner_computation"],
                    "collision_risk_safety_filter_ms": self.current_accumulator[
                        "collision_risk_safety_filter"
                    ],
                    "action_conversion_ms": self.current_accumulator["action_conversion"],
                    "total_ms": total_time,
                }
            )
        self.step_start_time = None
        self.current_accumulator = None

    def get_metrics(self) -> dict[str, Any]:
        """Compute and retrieve summary latency statistics and provenance.

        Returns:
            dict[str, Any]: Dictionary containing latency metrics and environment info.
        """
        if not self.cycles:
            return {}
        if len(self.cycles) < 2:
            raise ValueError(
                "Latency measurement requires at least two cycles before steady-state evidence "
                "can be classified"
            )

        cold_start = self.cycles[0]["total_ms"]
        steady_cycles = self.cycles[1:]
        steady_totals = [c["total_ms"] for c in steady_cycles]

        p50 = float(np.percentile(steady_totals, 50))
        p95 = float(np.percentile(steady_totals, 95))
        p99 = float(np.percentile(steady_totals, 99))
        max_val = float(np.max(steady_totals))

        misses = sum(1 for t in steady_totals if t > self.deadline_ms)
        miss_rate = float(misses / len(steady_totals))

        avg_obs = float(np.mean([c["observation_construction_ms"] for c in steady_cycles]))
        avg_pred = float(np.mean([c["prediction_ms"] for c in steady_cycles]))
        avg_plan = float(np.mean([c["planner_computation_ms"] for c in steady_cycles]))
        avg_safety = float(np.mean([c["collision_risk_safety_filter_ms"] for c in steady_cycles]))
        avg_action = float(np.mean([c["action_conversion_ms"] for c in steady_cycles]))

        prov = collect_environment_provenance(
            config_hash=self.config_hash,
            measured_host_identity=self.measured_host_identity,
        )
        validate_provenance_completeness(prov)
        classification = classify_feasibility(
            steady_state_latencies=steady_totals,
            deadline_ms=self.deadline_ms,
            target_hardware=self.target_hardware,
            measured_host_is_embedded=self.measured_host_is_embedded,
            measured_host_identity=prov["measured_host_identity"],
        )

        return {
            "cold_start_latency_ms": cold_start,
            "steady_state_latency_p50_ms": p50,
            "steady_state_latency_p95_ms": p95,
            "steady_state_latency_p99_ms": p99,
            "steady_state_latency_max_ms": max_val,
            "deadline_miss_rate": miss_rate,
            "classification": classification,
            "steady_state_averages": {
                "observation_construction_ms": avg_obs,
                "prediction_ms": avg_pred,
                "planner_computation_ms": avg_plan,
                "collision_risk_safety_filter_ms": avg_safety,
                "action_conversion_ms": avg_action,
            },
            "cycles": self.cycles,
            "provenance": prov,
        }


def classify_feasibility(
    *,
    steady_state_latencies: list[float],
    deadline_ms: float,
    target_hardware: str | None = None,
    measured_host_is_embedded: bool = False,
    measured_host_identity: str | None = None,
) -> str:
    """Classify the compute feasibility cell based on hardware and deadlines.

    Returns:
        str: The budget classification label.
    """
    del (
        measured_host_is_embedded
    )  # Retained for API compatibility; identity verification is required.
    if target_hardware is not None:
        if not isinstance(measured_host_identity, str) or not measured_host_identity.strip():
            return "target_hardware_unmeasured"
        if _normalize_hardware_identity(target_hardware) != _normalize_hardware_identity(
            measured_host_identity
        ):
            return "target_hardware_unmeasured"

    if not steady_state_latencies:
        return "target_hardware_unmeasured"
    if any(lat > deadline_ms for lat in steady_state_latencies):
        return "misses_budget_on_measured_host"
    return "meets_budget_on_measured_host"


def _normalize_hardware_identity(value: str) -> str:
    """Normalize host/target labels for conservative identity comparisons.

    Returns:
        A lowercase alphanumeric identity suitable for exact comparisons.
    """
    return "".join(character.lower() for character in value.strip() if character.isalnum())


def collect_environment_provenance(  # noqa: C901
    *,
    config_hash: str | None = None,
    measured_host_identity: str | None = None,
) -> dict[str, Any]:
    """Collect hardware and software details for run provenance.

    Returns:
        dict[str, Any]: Environmental provenance dictionary containing CPU model,
        affinity, BLAS/thread settings, package versions, and repo commit.
    """
    cpu_model = "unknown"
    try:
        if platform.system() == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line:
                        cpu_model = line.split(":", 1)[1].strip()
                        break
        else:
            cpu_model = platform.processor() or "unknown"
    except OSError:
        pass

    cpu_affinity = []
    try:
        if hasattr(os, "sched_getaffinity"):
            cpu_affinity = sorted(os.sched_getaffinity(0))
    except OSError:
        pass

    thread_settings: dict[str, Any] = {
        "environment": {var: os.environ.get(var) for var in _THREAD_SETTING_ENV_VARS},
        "threadpools": [],
    }
    try:
        from threadpoolctl import threadpool_info  # noqa: PLC0415

        thread_settings["threadpools"] = [
            {
                key: info.get(key)
                for key in (
                    "user_api",
                    "internal_api",
                    "num_threads",
                    "version",
                    "threading_layer",
                    "architecture",
                )
                if info.get(key) is not None
            }
            for info in threadpool_info()
        ]
    except ImportError:
        pass

    dependency_versions = {
        "python": sys.version.split()[0],
    }
    for pkg in ["numpy", "numba", "torch", "stable_baselines3", "scipy"]:
        try:
            import importlib  # noqa: PLC0415

            mod = importlib.import_module(pkg)
            dependency_versions[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            pass

    repo_commit = "unknown"
    try:
        from robot_sf.benchmark.utils import _git_hash_fallback  # noqa: PLC0415

        repo_commit = _git_hash_fallback()
    except (OSError, ValueError, AttributeError):
        pass

    return {
        "cpu_model": cpu_model,
        "cpu_affinity": cpu_affinity,
        "thread_settings": thread_settings,
        "dependency_versions": dependency_versions,
        "git_commit": repo_commit,
        "config_hash": config_hash,
        "measured_host_identity": measured_host_identity or cpu_model,
    }


def validate_provenance_completeness(prov: dict[str, Any]) -> None:  # noqa: C901
    """Ensure all required environment details are present and non-empty."""
    required_keys = [
        "cpu_model",
        "cpu_affinity",
        "dependency_versions",
        "git_commit",
        "thread_settings",
        "config_hash",
        "measured_host_identity",
    ]
    for key in required_keys:
        if key not in prov or not prov[key]:
            raise ValueError(f"Latency provenance missing required field: {key}")

    if prov["cpu_model"] == "unknown":
        raise ValueError("Latency provenance cpu_model cannot be 'unknown'")

    if not prov["git_commit"] or prov["git_commit"] == "unknown":
        raise ValueError("Latency provenance git_commit cannot be 'unknown' or empty")

    if not isinstance(prov["config_hash"], str) or not prov["config_hash"].strip():
        raise ValueError("Latency provenance config_hash cannot be missing or empty")

    if (
        not isinstance(prov["measured_host_identity"], str)
        or not prov["measured_host_identity"].strip()
        or prov["measured_host_identity"] == "unknown"
    ):
        raise ValueError("Latency provenance measured_host_identity cannot be missing or unknown")

    thread_settings = prov["thread_settings"]
    if not isinstance(thread_settings, Mapping):
        raise ValueError("Latency provenance thread_settings must be a mapping")
    environment_settings = thread_settings.get("environment")
    threadpools = thread_settings.get("threadpools")
    if not isinstance(environment_settings, Mapping) or not isinstance(threadpools, list):
        raise ValueError(
            "Latency provenance thread_settings must include environment and threadpools"
        )
    has_effective_thread_settings = bool(threadpools) or any(
        value not in (None, "") for value in environment_settings.values()
    )
    if not has_effective_thread_settings:
        raise ValueError("Latency provenance lacks effective thread settings")

    deps = prov["dependency_versions"]
    if "python" not in deps or "numpy" not in deps:
        raise ValueError("Latency provenance dependency_versions must include python and numpy")


def instrument_adapter_for_latency(adapter: Any) -> None:  # noqa: C901
    """Wraps adapter methods to measure component durations dynamically."""
    # 1. Wrap state extraction / grid caching
    for name in ["_extract_state", "_cache_grid_payload"]:
        original = getattr(adapter, name, None)
        if original and not hasattr(original, "_instrumented"):

            def make_wrapped(orig_func: Callable[..., Any]) -> Callable[..., Any]:
                def wrapped(*args: Any, **kwargs: Any) -> Any:
                    t0 = time.perf_counter()
                    res = orig_func(*args, **kwargs)
                    active_harness = LatencyMeasurementHarness.get_current()
                    if active_harness is not None:
                        active_harness.add_time(
                            "observation_construction", (time.perf_counter() - t0) * 1000.0
                        )
                    return res

                wrapped._instrumented = True  # type: ignore[attr-defined]
                return wrapped

            setattr(adapter, name, make_wrapped(original))

    # 2. Wrap prediction methods
    for name in ["_predict_ped_positions", "_predict_future"]:
        original = getattr(adapter, name, None)
        if original and not hasattr(original, "_instrumented"):

            def make_wrapped(orig_func: Callable[..., Any]) -> Callable[..., Any]:
                def wrapped(*args: Any, **kwargs: Any) -> Any:
                    t0 = time.perf_counter()
                    res = orig_func(*args, **kwargs)
                    active_harness = LatencyMeasurementHarness.get_current()
                    if active_harness is not None:
                        active_harness.add_time("prediction", (time.perf_counter() - t0) * 1000.0)
                    return res

                wrapped._instrumented = True  # type: ignore[attr-defined]
                return wrapped

            setattr(adapter, name, make_wrapped(original))


_instrumentation_applied = False


def apply_latency_instrumentation() -> None:  # noqa: C901
    """Dynamically patches map-runner and policy common helper functions to record planning sub-components."""
    global _instrumentation_applied
    if _instrumentation_applied:
        return

    try:
        from robot_sf.benchmark import map_runner_episode  # noqa: PLC0415

        original_safety_wrapper = map_runner_episode._apply_safety_wrapper_step
        original_cbf_safety = map_runner_episode._apply_cbf_safety_filter_step
        original_noise = map_runner_episode.apply_observation_noise
        original_tracking = map_runner_episode._apply_tracking_precision_to_observation
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "Latency instrumentation could not be installed: required map-runner hooks are "
            "unavailable"
        ) from exc

    # Patch safety wrappers outside policy.
    def wrapped_safety_wrapper(*args: Any, **kwargs: Any) -> Any:
        harness = LatencyMeasurementHarness.get_current()
        if harness is not None:
            t0 = time.perf_counter()
            res = original_safety_wrapper(*args, **kwargs)
            harness.add_time("collision_risk_safety_filter", (time.perf_counter() - t0) * 1000.0)
            return res
        return original_safety_wrapper(*args, **kwargs)

    # Patch the CBF safety path as a separate component wrapper.
    def wrapped_cbf_safety(*args: Any, **kwargs: Any) -> Any:
        harness = LatencyMeasurementHarness.get_current()
        if harness is not None:
            t0 = time.perf_counter()
            res = original_cbf_safety(*args, **kwargs)
            harness.add_time("collision_risk_safety_filter", (time.perf_counter() - t0) * 1000.0)
            return res
        return original_cbf_safety(*args, **kwargs)

    # Patch observation processing outside policy.
    def wrapped_noise(*args: Any, **kwargs: Any) -> Any:
        harness = LatencyMeasurementHarness.get_current()
        if harness is not None:
            t0 = time.perf_counter()
            res = original_noise(*args, **kwargs)
            harness.add_time("observation_construction", (time.perf_counter() - t0) * 1000.0)
            return res
        return original_noise(*args, **kwargs)

    def wrapped_tracking(*args: Any, **kwargs: Any) -> Any:
        harness = LatencyMeasurementHarness.get_current()
        if harness is not None:
            t0 = time.perf_counter()
            res = original_tracking(*args, **kwargs)
            harness.add_time("observation_construction", (time.perf_counter() - t0) * 1000.0)
            return res
        return original_tracking(*args, **kwargs)

    map_runner_episode._apply_safety_wrapper_step = wrapped_safety_wrapper
    map_runner_episode._apply_cbf_safety_filter_step = wrapped_cbf_safety
    map_runner_episode.apply_observation_noise = wrapped_noise
    map_runner_episode._apply_tracking_precision_to_observation = wrapped_tracking
    _instrumentation_applied = True
