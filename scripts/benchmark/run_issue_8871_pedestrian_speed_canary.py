#!/usr/bin/env python3
"""Run the disjoint-seed issue #8871 pedestrian-speed activation canary.

The command consumes the already frozen 72-row packet produced by
``build_issue_8871_pedestrian_speed_canary.py``.  It validates source/config/
checkpoint identity, preflights every planner, and then executes each packet
row through the native map runner.  The only retained measurements are the
speed-intervention diagnostics needed by the frozen #6561 activation rule;
episode outcomes and benchmark metrics are deliberately discarded.

The scenario loader on current ``origin/main`` does not expose the four
desired-speed fields in its scenario override allowlist.  This runner therefore
binds those frozen controls to the loaded ``SimulationSettings`` immediately
before the environment is created, and verifies the resulting values.  The
binding is scoped to this process and does not alter a tracked scenario or
production manifest.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import fields
from importlib import metadata as importlib_metadata
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

import yaml

from scripts.validation.build_issue_8871_pedestrian_speed_canary import (
    DEFAULT_CONFIG as DEFAULT_CANARY_CONFIG,
)
from scripts.validation.build_issue_8871_pedestrian_speed_canary import (
    build_manifest,
    load_canary_config,
)
from scripts.validation.check_issue_6561_activation_preflight import (
    NOT_EVIDENCE_BANNER,
    classify_activation,
    load_preflight,
    validate_preflight,
)
from scripts.validation.check_issue_6561_pedestrian_speed_protocol import load_protocol

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PRECHECK_CONFIG = (
    REPO_ROOT / "configs/benchmarks/issue_6561_pedestrian_speed_activation_preflight.yaml"
)
DEFAULT_PROTOCOL_CONFIG = REPO_ROOT / "configs/benchmarks/issue_6561_pedestrian_speed_protocol.yaml"
DIAGNOSTICS_SCHEMA_VERSION = "robot_sf.issue_6561_pedestrian_speed_activation_diagnostics.v1"
RUNNER_SCHEMA_VERSION = "robot_sf.issue_8871_pedestrian_speed_canary_execution.v1"
EXPECTED_ROWS = 72
EXPECTED_SEED = 311
TREATED_REGIMES = frozenset({"slow_distributed", "typical_distributed"})
VALID_EXECUTION_MODES = frozenset({"native", "native_command", "adapter", "mixed"})
FORBIDDEN_ROW_STATUSES = frozenset(
    {
        "fallback",
        "degraded",
        "failed",
        "missing",
        "duplicate",
        "unavailable",
        "provenance_invalid",
        "intervention_not_activated",
    }
)


class CanaryError(ValueError):
    """Raised when the canary cannot be admitted without guessing."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CanaryError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    except OSError as exc:
        raise CanaryError(f"cannot read {path}: {exc}") from exc
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CanaryError(f"{field} must be numeric, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise CanaryError(f"{field} must be finite, got {value!r}")
    return number


def _repo_path(raw: Any, field: str) -> Path:
    _require(isinstance(raw, str) and raw.strip(), f"{field} must be a path")
    path = Path(raw)
    _require(not path.is_absolute() and ".." not in path.parts, f"{field} leaves repository")
    resolved = REPO_ROOT / path
    _require(resolved.is_file(), f"{field} does not exist: {raw}")
    return resolved


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise CanaryError(f"cannot determine repository HEAD: {exc}") from exc


def _git_clean() -> bool:
    try:
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise CanaryError(f"cannot inspect repository cleanliness: {exc}") from exc
    return not result.stdout.strip()


def load_execution_manifest(
    path: Path,
    *,
    config_path: Path = DEFAULT_CANARY_CONFIG,
    current_head: str | None = None,
) -> dict[str, Any]:
    """Load and recompile the frozen packet against the current source bytes."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CanaryError(f"cannot read manifest {path}: {exc}") from exc
    _require(isinstance(payload, dict), "execution manifest must be a mapping")
    source_commit = payload.get("source_commit")
    _require(
        isinstance(source_commit, str)
        and len(source_commit) == 40
        and all(char in "0123456789abcdef" for char in source_commit),
        "manifest source_commit must be a full lowercase SHA",
    )
    actual_head = current_head or _git_head()
    _require(
        source_commit == actual_head, f"manifest source drift: {source_commit} != {actual_head}"
    )
    expected = build_manifest(load_canary_config(config_path), source_commit=source_commit)
    _require(payload.get("manifest_hash") == expected["manifest_hash"], "manifest hash drifted")
    _require(payload.get("identities") == expected["identities"], "manifest identities drifted")
    _require(payload.get("expected_rows") == EXPECTED_ROWS, "manifest row count is not 72")
    identities = payload.get("identities")
    _require(
        isinstance(identities, list) and len(identities) == EXPECTED_ROWS,
        "manifest identities must contain 72 rows",
    )
    _require(all(isinstance(row, Mapping) for row in identities), "manifest identity row malformed")
    keys = [row.get("identity_key") for row in identities if isinstance(row, Mapping)]
    _require(
        len(keys) == EXPECTED_ROWS and len(set(keys)) == EXPECTED_ROWS,
        "manifest identities are not unique",
    )
    _require(
        all(
            row.get("seed") == EXPECTED_SEED and row.get("registered") is False
            for row in identities
        ),
        "manifest contains a registered or non-frozen seed",
    )
    return payload


def _load_scenarios(protocol: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Load exactly the six source-bound scenario entries named by the protocol."""
    from robot_sf.training.scenario_loader import load_scenarios

    selected = protocol["scenario_contract"]["selected_scenarios"]
    scenarios: dict[str, dict[str, Any]] = {}
    for index, spec in enumerate(selected):
        source_path = _repo_path(spec.get("source_path"), f"scenario[{index}].source_path")
        _require(
            _sha256(source_path) == spec.get("source_sha256"),
            f"scenario[{index}] source hash drifted",
        )
        matches = [
            dict(row)
            for row in load_scenarios(source_path)
            if str(row.get("name")) == str(spec.get("scenario_id"))
        ]
        _require(
            len(matches) == 1,
            f"scenario {spec.get('scenario_id')!r} is not unique in {source_path}",
        )
        scenario_id = str(spec["scenario_id"])
        _require(scenario_id not in scenarios, f"duplicate protocol scenario {scenario_id}")
        scenarios[scenario_id] = matches[0]
    return scenarios


def _load_planner_specs(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Validate planner config bytes and return the frozen roster."""
    specs: list[dict[str, Any]] = []
    for index, raw in enumerate(protocol["planner_contract"]["roster"]):
        spec = dict(raw)
        config_raw = spec.get("config_path")
        if config_raw is None:
            _require(
                spec.get("planner_id") == "orca", f"planner[{index}] missing config must be ORCA"
            )
            spec["config_path"] = None
            spec["raw_config"] = {}
        else:
            config_path = _repo_path(config_raw, f"planner[{index}].config_path")
            _require(
                _sha256(config_path) == spec.get("config_sha256"),
                f"planner[{index}] config hash drifted",
            )
            try:
                raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            except yaml.YAMLError as exc:
                raise CanaryError(f"cannot parse planner config {config_path}: {exc}") from exc
            _require(isinstance(raw_config, dict), f"planner[{index}] config must be a mapping")
            spec["config_path"] = config_path
            spec["raw_config"] = raw_config
        specs.append(spec)
    return specs


def _runtime_controls(row: Mapping[str, Any]) -> dict[str, Any]:
    controls = row.get("runtime_controls")
    _require(isinstance(controls, Mapping), f"{row.get('identity_key')}: runtime_controls missing")
    return dict(controls)


def build_execution_scenario(base: Mapping[str, Any], row: Mapping[str, Any]) -> dict[str, Any]:
    """Add diagnostic-only trace switches and frozen row controls to a scenario copy."""
    scenario = copy.deepcopy(dict(base))
    simulation_config = scenario.get("simulation_config")
    simulation_config = dict(simulation_config) if isinstance(simulation_config, Mapping) else {}
    simulation_config["max_episode_steps"] = int(row["horizon_steps"])
    simulation_config["oracle_force_trace_enabled"] = True
    simulation_config["sampler_capture_enabled"] = True
    simulation_config.update(_runtime_controls(row))
    scenario["simulation_config"] = simulation_config
    scenario["seeds"] = [int(row["seed"])]
    return scenario


def _apply_runtime_speed_controls(config: Any, controls: Mapping[str, Any], *, seed: int) -> None:
    """Bind and verify frozen desired-speed controls on native SimulationSettings."""
    sim_config = getattr(config, "sim_config", None)
    _require(sim_config is not None, "native environment config has no sim_config")
    for field in ("ped_speed_tier", "desired_speed_mean", "desired_speed_std"):
        setattr(sim_config, field, controls.get(field))
    raw_seed = controls.get("desired_speed_seed")
    desired_seed = (
        None if raw_seed is None else (seed if raw_seed == "episode_seed" else int(raw_seed))
    )
    sim_config.desired_speed_seed = desired_seed
    validator = getattr(sim_config, "_validate_desired_speed_config", None)
    _require(callable(validator), "native SimulationSettings lacks desired-speed validation")
    validator()
    for field in (
        "ped_speed_tier",
        "desired_speed_mean",
        "desired_speed_std",
        "desired_speed_seed",
    ):
        expected = desired_seed if field == "desired_speed_seed" else controls.get(field)
        observed = getattr(sim_config, field)
        _require(
            observed == expected,
            f"native desired-speed binding drifted for {field}: {observed!r} != {expected!r}",
        )


def _runtime_binding_context(controls: Mapping[str, Any], *, seed: int):
    """Return a scoped map-runner env-builder override for the missing YAML fields."""
    from contextlib import contextmanager

    from robot_sf.benchmark.map_runner import map_runner_episode

    @contextmanager
    def _context():
        original = map_runner_episode._build_env_config

        def _build(scenario: dict[str, Any], *, scenario_path: Path) -> Any:
            config = original(scenario, scenario_path=scenario_path)
            scenario_controls = scenario.get("simulation_config", {})
            _require(
                isinstance(scenario_controls, Mapping), "execution scenario controls are missing"
            )
            _require(
                dict(scenario_controls.items())
                and all(scenario_controls.get(key) == controls.get(key) for key in controls),
                "scenario controls changed before native config binding",
            )
            _apply_runtime_speed_controls(config, controls, seed=seed)
            return config

        map_runner_episode._build_env_config = _build
        try:
            yield
        finally:
            map_runner_episode._build_env_config = original

    return _context()


def _execution_disposition(  # noqa: C901, PLR0912
    record: Mapping[str, Any],
) -> tuple[str, str | None]:
    """Classify map-runner metadata using the existing native/adaptor contract."""
    metadata = record.get("algorithm_metadata")
    if not isinstance(metadata, Mapping):
        return "failed", "missing algorithm_metadata"
    status = str(metadata.get("status", "")).strip().lower()
    if status.startswith(
        ("fallback", "degraded", "policy_step_error_fallback", "policy_step_timeout_fallback")
    ):
        return "degraded", f"algorithm_metadata.status={status}"
    timeout = metadata.get("policy_step_timeout")
    if timeout is not None and not isinstance(timeout, Mapping):
        return "failed", "malformed policy_step_timeout"
    if isinstance(timeout, Mapping):
        try:
            fallback_actions = float(timeout.get("fallback_actions", 0.0))
        except (TypeError, ValueError):
            return "failed", "malformed policy_step_timeout.fallback_actions"
        if not math.isfinite(fallback_actions) or fallback_actions < 0.0:
            return "failed", "malformed policy_step_timeout.fallback_actions"
        if fallback_actions > 0.0:
            return "degraded", "policy_step_timeout.fallback_actions>0"
    if metadata.get("fallback_or_degraded") is True:
        return "degraded", "algorithm_metadata.fallback_or_degraded=true"
    foresight = metadata.get("foresight_prediction")
    if foresight is not None and not isinstance(foresight, Mapping):
        return "failed", "malformed foresight_prediction"
    if isinstance(foresight, Mapping):
        if foresight.get("fallback_used") is True:
            return "degraded", "foresight_prediction.fallback_used=true"
        if foresight.get("evidence_eligible") is False:
            return "degraded", "foresight_prediction.evidence_eligible=false"
    kinematics = metadata.get("planner_kinematics")
    if not isinstance(kinematics, Mapping):
        return "failed", "missing planner_kinematics"
    mode = str(kinematics.get("execution_mode", "")).strip().lower()
    if mode in {"fallback", "degraded"}:
        return "degraded", f"planner_kinematics.execution_mode={mode}"
    if mode not in VALID_EXECUTION_MODES:
        return "failed", f"unsupported planner_kinematics.execution_mode={mode}"
    if status == "ok":
        return "native", None
    return "failed", f"unsupported algorithm_metadata.status={status or '<missing>'}"


def _speed(value: Any, field: str) -> float:
    velocity = _velocity(value, field)
    return math.hypot(velocity[0], velocity[1])


def _velocity(value: Any, field: str) -> list[float]:
    """Return a finite two-dimensional runtime velocity for receipt diagnostics."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 2:
        raise CanaryError(f"{field} must be a two-dimensional velocity")
    return [_finite(value[0], f"{field}[0]"), _finite(value[1], f"{field}[1]")]


def _p95(values: Sequence[float]) -> float:
    _require(bool(values), "cannot compute p95 of an empty sequence")
    ordered = sorted(values)
    rank = max(1, math.ceil(0.95 * len(ordered)))
    return ordered[rank - 1]


def extract_activation_diagnostics(  # noqa: PLR0915
    record: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    target_tolerance_m_s: float = 0.2,
) -> dict[str, Any]:
    """Extract finite activation diagnostics from the native oracle transition trace."""
    metadata = record.get("algorithm_metadata")
    _require(isinstance(metadata, Mapping), "record missing algorithm_metadata")
    trace = metadata.get("simulation_step_trace")
    _require(isinstance(trace, Mapping), "record missing simulation_step_trace")
    steps = trace.get("steps")
    _require(isinstance(steps, list) and steps, "simulation_step_trace.steps is empty")
    dt = _finite(trace.get("dt"), "simulation_step_trace.dt")
    expected_dt = _finite(row.get("dt_seconds"), "row.dt_seconds")
    _require(
        math.isclose(dt, expected_dt, abs_tol=1e-9), f"trace dt drifted: {dt} != {expected_dt}"
    )
    reset = trace.get("reset")
    _require(isinstance(reset, Mapping), "simulation_step_trace.reset is missing")
    reset_peds = reset.get("pedestrians")
    _require(
        isinstance(reset_peds, list) and reset_peds, "reset pedestrian velocities are unavailable"
    )
    initial_velocity_by_pedestrian: dict[str, list[float]] = {}
    initial_speeds: list[float] = []
    for ped in reset_peds:
        _require(isinstance(ped, Mapping), "reset pedestrian entry is malformed")
        raw_id = ped.get("actor_id", ped.get("id"))
        _require(raw_id is not None, "reset pedestrian id is missing")
        ped_id = str(raw_id)
        _require(
            ped_id and ped_id not in initial_velocity_by_pedestrian,
            "reset pedestrian ids duplicate",
        )
        velocity = _velocity(ped.get("velocity"), "reset.pedestrians.velocity")
        initial_velocity_by_pedestrian[ped_id] = velocity
        initial_speeds.append(math.hypot(*velocity))

    preferred: dict[str, float] = {}
    trajectories: dict[str, list[tuple[int, float]]] = {}
    final_velocity_by_pedestrian: dict[str, list[float]] = {}
    for step_index, raw_step in enumerate(steps):
        _require(isinstance(raw_step, Mapping), f"trace step {step_index} is malformed")
        transitions_block = raw_step.get("oracle_transition_trace")
        _require(
            isinstance(transitions_block, Mapping),
            f"trace step {step_index} lacks oracle transitions",
        )
        transitions = transitions_block.get("transitions")
        _require(
            isinstance(transitions, list) and transitions,
            f"trace step {step_index} has no oracle transitions",
        )
        for transition in transitions:
            _require(isinstance(transition, Mapping), "oracle transition is malformed")
            ped_id = transition.get("simulator_pedestrian_id")
            _require(
                isinstance(ped_id, str) and ped_id, "oracle transition pedestrian id is missing"
            )
            dynamics = transition.get("dynamics")
            _require(isinstance(dynamics, Mapping), f"oracle transition {ped_id} lacks dynamics")
            preferred_speed = _finite(
                dynamics.get("preferred_speed_mps"), f"oracle[{ped_id}].preferred_speed_mps"
            )
            preferred.setdefault(ped_id, preferred_speed)
            post = transition.get("post_integration")
            _require(
                isinstance(post, Mapping), f"oracle transition {ped_id} lacks post_integration"
            )
            velocity_xy = _velocity(
                post.get("velocity_xy"), f"oracle[{ped_id}].post_integration.velocity_xy"
            )
            velocity = math.hypot(*velocity_xy)
            trajectories.setdefault(ped_id, []).append((step_index + 1, velocity))
            final_velocity_by_pedestrian[ped_id] = velocity_xy
    _require(preferred, "oracle trace contains no pedestrian desired speeds")
    _require(set(trajectories) == set(preferred), "oracle desired-speed and velocity ids differ")

    controls = _runtime_controls(row)
    configured_mean = controls.get("desired_speed_mean")
    configured_std = controls.get("desired_speed_std")
    diagnostics: dict[str, Any] = {
        "configured_desired_speed_mean_m_s": None
        if configured_mean is None
        else _finite(configured_mean, "configured desired speed mean"),
        "configured_desired_speed_std_m_s": None
        if configured_std is None
        else _finite(configured_std, "configured desired speed std"),
        "realized_desired_speed_mean_m_s": fmean(preferred.values()),
        "realized_desired_speed_std_m_s": pstdev(preferred.values()),
        "initial_spawn_speed_mean_m_s": fmean(initial_speeds),
        "initial_spawn_speed_peak_m_s": max(initial_speeds),
        # These maps retain the actual simulator state and ``peds.max_speeds``
        # values behind the aggregate activation fields.  They are diagnostic
        # provenance, not benchmark outcomes.
        "runtime_max_speed_m_s_by_pedestrian": dict(sorted(preferred.items())),
        "initial_spawn_velocity_xy_by_pedestrian": dict(
            sorted(initial_velocity_by_pedestrian.items())
        ),
        "final_post_integration_velocity_xy_by_pedestrian": dict(
            sorted(final_velocity_by_pedestrian.items())
        ),
    }
    if configured_mean is None:
        diagnostics.update(
            {
                "time_to_desired_speed_target_seconds": None,
                "acceleration_transient_steps": None,
                "desired_speed_activation_fraction": None,
            }
        )
        return diagnostics

    target = _finite(configured_mean, "configured desired speed mean")
    target_steps: list[int] = []
    reached = 0
    horizon_steps = int(row["horizon_steps"])
    for ped_id, samples in trajectories.items():
        first = next(
            (step for step, speed in samples if abs(speed - target) <= target_tolerance_m_s),
            None,
        )
        # Keep a missing target finite at the full episode horizon, while
        # distinguishing it from a target reached on the final recorded step.
        target_steps.append(horizon_steps if first is None else first)
        if first is not None:
            reached += 1
    p95_steps = _p95(target_steps)
    diagnostics.update(
        {
            "time_to_desired_speed_target_seconds": p95_steps * dt,
            "acceleration_transient_steps": p95_steps,
            "desired_speed_activation_fraction": reached / len(target_steps),
        }
    )
    for key in (
        "configured_desired_speed_mean_m_s",
        "configured_desired_speed_std_m_s",
        "realized_desired_speed_mean_m_s",
        "realized_desired_speed_std_m_s",
        "initial_spawn_speed_mean_m_s",
        "initial_spawn_speed_peak_m_s",
        "time_to_desired_speed_target_seconds",
        "acceleration_transient_steps",
        "desired_speed_activation_fraction",
    ):
        value = diagnostics[key]
        _require(value is not None, f"treated diagnostic {key} is unavailable")
        _finite(value, f"diagnostics.{key}")
    return diagnostics


def _registry_checkpoint(model_id: str, checkpoint_root: Path) -> dict[str, Any]:
    from robot_sf.models.registry import get_registry_entry

    try:
        entry = get_registry_entry(model_id, REPO_ROOT / "model/registry.yaml")
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        raise CanaryError(f"checkpoint registry entry unavailable for {model_id}: {exc}") from exc
    release = entry.get("github_release")
    expected = release.get("sha256") if isinstance(release, Mapping) else None
    _require(
        isinstance(expected, str) and len(expected) == 64,
        f"checkpoint {model_id} lacks a pinned SHA-256",
    )
    local_path = Path(str(entry.get("local_path", "")))
    filename = local_path.name
    candidates = [checkpoint_root / model_id / filename, checkpoint_root / filename]
    if local_path.parts:
        candidates.append(checkpoint_root / local_path)
    selected = next((candidate for candidate in candidates if candidate.is_file()), None)
    _require(selected is not None, f"checkpoint {model_id} is not staged below {checkpoint_root}")
    observed = _sha256(selected)
    _require(
        observed == expected.lower(),
        f"checkpoint {model_id} SHA-256 drifted: {observed} != {expected}",
    )
    return {
        "model_id": model_id,
        "path_label": f"{model_id}/{filename}",
        "sha256": observed,
        "size_bytes": selected.stat().st_size,
        "expected_sha256": expected.lower(),
        "path": selected,
    }


def _required_model_ids(planner_specs: Sequence[Mapping[str, Any]]) -> list[str]:
    ids: list[str] = []
    for spec in planner_specs:
        cfg = spec.get("raw_config")
        if not isinstance(cfg, Mapping):
            continue
        for key in ("model_id", "predictive_model_id"):
            value = cfg.get(key)
            if isinstance(value, str) and value.strip() and value not in ids:
                ids.append(value)
        if cfg.get("predictive_foresight_enabled"):
            value = cfg.get("predictive_foresight_model_id")
            if isinstance(value, str) and value.strip() and value not in ids:
                ids.append(value)
    return ids


def _bind_checkpoint_paths(
    algo: str, config: Mapping[str, Any], checkpoints: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """Use staged bytes while retaining the frozen model identity in the outer receipt."""
    effective = dict(config)
    if algo == "ppo":
        model_id = effective.get("model_id")
        if isinstance(model_id, str) and model_id in checkpoints:
            effective["model_id"] = None
            effective["model_path"] = str(checkpoints[model_id]["path"])
        foresight_id = effective.get("predictive_foresight_model_id")
        if effective.get("predictive_foresight_enabled") and foresight_id in checkpoints:
            effective["predictive_foresight_checkpoint_path"] = str(
                checkpoints[foresight_id]["path"]
            )
    elif algo == "prediction_planner":
        model_id = effective.get("predictive_model_id")
        if isinstance(model_id, str) and model_id in checkpoints:
            effective["predictive_checkpoint_path"] = str(checkpoints[model_id]["path"])
    return effective


def _device_report(planner_specs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    report: dict[str, Any] = {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
    }
    try:
        import torch
    except ImportError:
        report.update({"torch": None, "cuda_available": False, "cuda_device_count": 0})
    else:
        report.update(
            {
                "torch": str(torch.__version__),
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_device_count": int(torch.cuda.device_count()),
            }
        )
    requested: dict[str, Any] = {}
    for spec in planner_specs:
        cfg = spec.get("raw_config")
        if isinstance(cfg, Mapping):
            requested[str(spec["planner_id"])] = {
                key: cfg.get(key)
                for key in (
                    "device",
                    "predictive_device",
                    "predictive_foresight_device",
                )
                if key in cfg
            }
    report["requested_devices"] = requested
    report["execution_policy"] = (
        "CPU-compatible; CUDA availability is recorded and never substituted into the frozen config"
    )
    return report


def _environment_report(planner_specs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Record importable runtime versions used by the native preflight."""
    package_versions: dict[str, str | None] = {}
    for distribution in (
        "numpy",
        "gymnasium",
        "stable-baselines3",
        "sb3-contrib",
        "torch",
        "pysocialforce",
    ):
        try:
            package_versions[distribution] = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError:
            package_versions[distribution] = None
    return {
        "device": _device_report(planner_specs),
        "package_versions": package_versions,
        "native_trace_requirements": {
            "oracle_force_trace_enabled": True,
            "sampler_capture_enabled": True,
            "record_simulation_step_trace": True,
            "fallback_or_degraded_execution_allowed": False,
        },
    }


def preflight_canary(  # noqa: C901
    manifest: Mapping[str, Any],
    *,
    config_path: Path = DEFAULT_CANARY_CONFIG,
    preflight_config_path: Path = DEFAULT_PRECHECK_CONFIG,
    checkpoint_root: Path,
) -> dict[str, Any]:
    """Validate native planner, model, scenario, and trace prerequisites."""
    _require(config_path.is_file(), f"canary config not found: {config_path}")
    _require(
        preflight_config_path.is_file(), f"preflight config not found: {preflight_config_path}"
    )
    _require(DEFAULT_PROTOCOL_CONFIG.is_file(), "pedestrian-speed protocol config is unavailable")
    _require(
        _sha256(DEFAULT_PROTOCOL_CONFIG) == manifest["protocol_sha256"],
        "manifest protocol bytes differ from the current protocol config",
    )
    protocol = load_protocol()
    scenarios = _load_scenarios(protocol)
    planner_specs = _load_planner_specs(protocol)
    checkpoints: dict[str, dict[str, Any]] = {}
    for model_id in _required_model_ids(planner_specs):
        checkpoints[model_id] = _registry_checkpoint(model_id, checkpoint_root)
    planner_report: list[dict[str, Any]] = []
    blockers: list[str] = []
    from robot_sf.benchmark.map_runner.map_runner import (
        _preflight_policy,
        _resolve_policy_search_candidate_runtime,
    )

    for spec in planner_specs:
        planner_id = str(spec["planner_id"])
        algorithm = str(spec["algorithm"])
        config_path_value = spec.get("config_path")
        raw_config = dict(spec.get("raw_config") or {})
        effective_by_scenario: dict[str, dict[str, Any]] = {}
        scenario_errors: list[str] = []
        for scenario_id, scenario in scenarios.items():
            try:
                resolved_algorithm, effective = _resolve_policy_search_candidate_runtime(
                    default_algo=algorithm,
                    algo_config_path=str(config_path_value) if config_path_value else None,
                    algo_config=raw_config,
                    scenario=scenario,
                )
                effective = _bind_checkpoint_paths(resolved_algorithm, effective, checkpoints)
                effective_by_scenario[scenario_id] = {
                    "algorithm": resolved_algorithm,
                    "config": effective,
                }
            except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
                scenario_errors.append(f"{scenario_id}: {type(exc).__name__}: {exc}")
        if scenario_errors:
            blockers.extend(f"{planner_id} {message}" for message in scenario_errors)
            planner_report.append(
                {"planner_id": planner_id, "status": "blocked", "errors": scenario_errors}
            )
            continue
        # One construction per distinct effective config is enough; the actual episode
        # path still receives the scenario-specific config selected above.
        distinct: dict[str, tuple[str, dict[str, Any], list[str]]] = {}
        for scenario_id, resolved in effective_by_scenario.items():
            key = _canonical_hash(resolved)
            if key not in distinct:
                distinct[key] = (
                    str(resolved["algorithm"]),
                    dict(resolved["config"]),
                    [scenario_id],
                )
            else:
                distinct[key][2].append(scenario_id)
        checks: list[dict[str, Any]] = []
        for algorithm_key, effective, scenario_ids in distinct.values():
            try:
                _effective, status = _preflight_policy(
                    algo=algorithm_key,
                    algo_config=effective,
                    benchmark_profile="experimental",
                    missing_prereq_policy="fail-fast",
                    robot_kinematics="differential_drive",
                )
                if status.get("status") != "ok":
                    raise CanaryError(str(status))
                checks.append(
                    {"algorithm": algorithm_key, "scenario_ids": scenario_ids, "status": "ok"}
                )
            except (KeyError, OSError, RuntimeError, TypeError, ValueError, CanaryError) as exc:
                reason = f"{type(exc).__name__}: {exc}"
                blockers.append(f"{planner_id}: {reason}")
                checks.append(
                    {
                        "algorithm": algorithm_key,
                        "scenario_ids": scenario_ids,
                        "status": "blocked",
                        "reason": reason,
                    }
                )
        planner_report.append(
            {
                "planner_id": planner_id,
                "status": "ok" if all(check["status"] == "ok" for check in checks) else "blocked",
                "effective_config_count": len(distinct),
                "checks": checks,
                "model_ids": [
                    model_id for model_id in _required_model_ids([spec]) if model_id in checkpoints
                ],
            }
        )
    # Explicit model deserialization catches schema/load failures that construction-only
    # preflight cannot see for lazy prediction adapters.
    for spec in planner_specs:
        planner_id = str(spec["planner_id"])
        algorithm = str(spec["algorithm"])
        raw_config = dict(spec.get("raw_config") or {})
        first_scenario = next(iter(scenarios.values()))
        try:
            resolved_algorithm, effective = _resolve_policy_search_candidate_runtime(
                default_algo=algorithm,
                algo_config_path=str(spec["config_path"]) if spec.get("config_path") else None,
                algo_config=raw_config,
                scenario=first_scenario,
            )
            effective = _bind_checkpoint_paths(resolved_algorithm, effective, checkpoints)
            _load_model_for_preflight(resolved_algorithm, effective)
        except (KeyError, OSError, RuntimeError, TypeError, ValueError, CanaryError) as exc:
            blockers.append(f"{planner_id} model load: {type(exc).__name__}: {exc}")
    preflight_payload, protocol_from_preflight = load_preflight(preflight_config_path)
    validate_preflight(preflight_payload, protocol_from_preflight)
    # Loading the checker config here keeps the activation classifier bound to the same protocol.
    _require(
        protocol_from_preflight["manifest_contract"]["manifest_hash"]
        == protocol["manifest_contract"]["manifest_hash"],
        "preflight protocol differs from execution protocol",
    )
    result = {
        "schema_version": RUNNER_SCHEMA_VERSION,
        "status": "ok" if not blockers else "blocked",
        "source_commit": manifest["source_commit"],
        "manifest_hash": manifest["manifest_hash"],
        "expected_rows": EXPECTED_ROWS,
        "scenario_count": len(scenarios),
        "planner_count": len(planner_specs),
        "checkpoint_root_label": checkpoint_root.name,
        "checkpoints": [
            {key: value for key, value in checkpoint.items() if key != "path"}
            for checkpoint in checkpoints.values()
        ],
        "planners": planner_report,
        "environment": _environment_report(planner_specs),
        "source": {
            "worktree_clean_tracked": _git_clean(),
            "canary_config_path": str(config_path.relative_to(REPO_ROOT))
            if config_path.is_relative_to(REPO_ROOT)
            else config_path.name,
            "canary_config_sha256": _sha256(config_path),
            "protocol_config_path": str(DEFAULT_PROTOCOL_CONFIG.relative_to(REPO_ROOT)),
            "protocol_config_sha256": _sha256(DEFAULT_PROTOCOL_CONFIG),
            "activation_preflight_config_path": str(preflight_config_path.relative_to(REPO_ROOT))
            if preflight_config_path.is_relative_to(REPO_ROOT)
            else preflight_config_path.name,
            "activation_preflight_config_sha256": _sha256(preflight_config_path),
        },
        "activation_checker": {
            "config": str(preflight_config_path.relative_to(REPO_ROOT))
            if preflight_config_path.is_relative_to(REPO_ROOT)
            else preflight_config_path.name,
            "status": "loaded",
            "diagnostic_schema": DIAGNOSTICS_SCHEMA_VERSION,
        },
        "blockers": blockers,
    }
    return result


def _load_model_for_preflight(algo: str, config: Mapping[str, Any]) -> None:
    """Force lazy learned adapters to deserialize their staged checkpoint bytes."""
    if algo == "ppo":
        from robot_sf.baselines.ppo import PPOPlanner, PPOPlannerConfig

        allowed = {field.name for field in fields(PPOPlannerConfig)}
        planner = PPOPlanner(
            {key: value for key, value in config.items() if key in allowed}, seed=None
        )
        try:
            foresight = getattr(planner, "_predictive_foresight", None)
            if foresight is not None:
                adapter = getattr(foresight, "_adapter", None)
                ensure_model = getattr(adapter, "_ensure_model", None)
                _require(callable(ensure_model), "PPO predictive foresight loader is unavailable")
                _require(
                    ensure_model() is not None, "PPO predictive foresight checkpoint did not load"
                )
            _require(
                planner.get_metadata().get("status") == "ok", "PPO checkpoint did not load natively"
            )
        finally:
            planner.close()
    elif algo == "prediction_planner":
        from robot_sf.planner.socnav import PredictionPlannerAdapter, SocNavPlannerConfig

        allowed = {field.name for field in fields(SocNavPlannerConfig)}
        planner = PredictionPlannerAdapter(
            SocNavPlannerConfig(**{key: value for key, value in config.items() if key in allowed}),
            allow_fallback=False,
        )
        try:
            _require(
                planner._ensure_model() is not None, "prediction checkpoint did not load natively"
            )
        finally:
            close = getattr(planner, "close", None)
            if callable(close):
                close()


def _safe_error(exc: Exception) -> str:
    return f"{type(exc).__name__}: {str(exc).replace(str(REPO_ROOT), '<repo>')}"


def _row_record(
    row: Mapping[str, Any],
    *,
    status: str,
    reason: str | None,
    record: Mapping[str, Any] | None,
    diagnostics: Mapping[str, Any] | None,
) -> dict[str, Any]:
    metadata = record.get("algorithm_metadata") if isinstance(record, Mapping) else None
    kinematics = metadata.get("planner_kinematics") if isinstance(metadata, Mapping) else None
    return {
        "identity_key": row["identity_key"],
        "scenario_id": row["scenario_id"],
        "regime_id": row["regime_id"],
        "planner_id": row["planner_id"],
        "seed": int(row["seed"]),
        "horizon_steps": int(row["horizon_steps"]),
        "dt_seconds": float(row["dt_seconds"]),
        "row_status": status,
        "execution_disposition_reason": reason,
        "planner_execution_mode": (
            str(kinematics.get("execution_mode")) if isinstance(kinematics, Mapping) else None
        ),
        "algorithm_metadata_status": (
            str(metadata.get("status")) if isinstance(metadata, Mapping) else None
        ),
        "diagnostics": dict(diagnostics) if isinstance(diagnostics, Mapping) else None,
    }


def _append_journal_event(handle: Any, event: str, **payload: Any) -> None:
    """Durably append one row lifecycle event for interruption-safe reconciliation."""
    record = {"event": event, **payload}
    handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def run_canary(  # noqa: PLR0915
    manifest_path: Path,
    output_path: Path,
    *,
    config_path: Path = DEFAULT_CANARY_CONFIG,
    preflight_config_path: Path = DEFAULT_PRECHECK_CONFIG,
    checkpoint_root: Path,
    max_rows: int | None = None,
    journal_path: Path | None = None,
) -> dict[str, Any]:
    """Run all selected packet rows and write a fail-closed diagnostics receipt."""
    _require(not output_path.exists(), f"refusing to overwrite existing output: {output_path}")
    journal_path = journal_path or output_path.with_name(output_path.name + ".journal.jsonl")
    _require(
        journal_path.resolve() != output_path.resolve(),
        "diagnostic journal must be separate from the final output",
    )
    _require(not journal_path.exists(), f"refusing to overwrite existing journal: {journal_path}")
    manifest = load_execution_manifest(manifest_path, config_path=config_path)
    _require(_git_clean(), "source worktree has tracked changes; clean it before execution")
    preflight = preflight_canary(
        manifest,
        config_path=config_path,
        preflight_config_path=preflight_config_path,
        checkpoint_root=checkpoint_root,
    )
    _require(
        preflight["status"] == "ok", "native preflight blocked: " + "; ".join(preflight["blockers"])
    )
    protocol = load_protocol()
    scenarios = _load_scenarios(protocol)
    planner_specs = _load_planner_specs(protocol)
    planner_by_id = {str(spec["planner_id"]): spec for spec in planner_specs}
    selected_rows = list(manifest["identities"])
    if max_rows is not None:
        _require(max_rows > 0, "max_rows must be positive")
        selected_rows = selected_rows[:max_rows]
    from robot_sf.benchmark.map_runner import map_runner_episode
    from robot_sf.benchmark.map_runner.map_runner import (
        _resolve_policy_search_candidate_runtime,
        build_map_policy,
    )

    checkpoints = {
        checkpoint["model_id"]: checkpoint
        for checkpoint in (
            _registry_checkpoint(model_id, checkpoint_root)
            for model_id in _required_model_ids(planner_specs)
        )
    }
    policy_cache: dict[str, tuple[Callable[..., Any], dict[str, Any]]] = {}

    def _policy_builder(
        algo: str,
        algo_config: dict[str, Any],
        *,
        robot_kinematics: str | None = None,
        robot_command_mode: str | None = None,
        adapter_impact_eval: bool = False,
    ) -> tuple[Callable[..., Any], dict[str, Any]]:
        key = _canonical_hash({"algo": algo, "config": algo_config})
        if key not in policy_cache:
            policy_cache[key] = build_map_policy(
                algo,
                algo_config,
                robot_kinematics=robot_kinematics,
                robot_command_mode=robot_command_mode,
                adapter_impact_eval=adapter_impact_eval,
            )
        return policy_cache[key]

    execution_rows: list[dict[str, Any]] = []
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    with journal_path.open("x", encoding="utf-8") as journal:
        _append_journal_event(
            journal,
            "header",
            schema_version=RUNNER_SCHEMA_VERSION,
            source_commit=manifest["source_commit"],
            manifest_hash=manifest["manifest_hash"],
            selected_rows=len(selected_rows),
        )
        try:
            for row in selected_rows:
                _append_journal_event(
                    journal,
                    "row_started",
                    identity_key=row["identity_key"],
                    row_index=len(execution_rows),
                )
                scenario_id = str(row["scenario_id"])
                planner_id = str(row["planner_id"])
                base = scenarios.get(scenario_id)
                spec = planner_by_id.get(planner_id)
                if base is None or spec is None:
                    row_result = _row_record(
                        row,
                        status="unavailable",
                        reason="manifest source/planner missing",
                        record=None,
                        diagnostics=None,
                    )
                else:
                    try:
                        raw_config = dict(spec.get("raw_config") or {})
                        resolved_algo, effective_config = _resolve_policy_search_candidate_runtime(
                            default_algo=str(spec["algorithm"]),
                            algo_config_path=(
                                str(spec["config_path"]) if spec.get("config_path") else None
                            ),
                            algo_config=raw_config,
                            scenario=base,
                        )
                        effective_config = _bind_checkpoint_paths(
                            resolved_algo, effective_config, checkpoints
                        )
                        scenario = build_execution_scenario(base, row)
                        controls = _runtime_controls(row)
                        with _runtime_binding_context(controls, seed=int(row["seed"])):
                            record = map_runner_episode.run_map_episode(
                                scenario=scenario,
                                seed=int(row["seed"]),
                                horizon=int(row["horizon_steps"]),
                                dt=float(row["dt_seconds"]),
                                record_forces=False,
                                snqi_weights=None,
                                snqi_baseline=None,
                                algo=resolved_algo,
                                scenario_path=_repo_path(
                                    next(
                                        item["source_path"]
                                        for item in protocol["scenario_contract"][
                                            "selected_scenarios"
                                        ]
                                        if item["scenario_id"] == scenario_id
                                    ),
                                    "scenario source",
                                ),
                                algo_config=effective_config,
                                algo_config_path=None,
                                benchmark_track="experimental",
                                record_simulation_step_trace=True,
                                close_policy=False,
                                policy_builder=_policy_builder,
                            )
                        status, reason = _execution_disposition(record)
                        diagnostics = (
                            extract_activation_diagnostics(record, row)
                            if status == "native"
                            else None
                        )
                        row_result = _row_record(
                            row,
                            status=status,
                            reason=reason,
                            record=record,
                            diagnostics=diagnostics,
                        )
                    except (
                        CanaryError,
                        IndexError,
                        KeyError,
                        OSError,
                        RuntimeError,
                        TypeError,
                        ValueError,
                    ) as exc:
                        row_result = _row_record(
                            row,
                            status="failed",
                            reason=_safe_error(exc),
                            record=None,
                            diagnostics=None,
                        )
                execution_rows.append(row_result)
                _append_journal_event(
                    journal,
                    "row_finished",
                    identity_key=row_result["identity_key"],
                    row_status=row_result["row_status"],
                    execution_disposition_reason=row_result["execution_disposition_reason"],
                )
        finally:
            for policy, _meta in policy_cache.values():
                close = getattr(policy, "_planner_close", None)
                if callable(close):
                    close()
    expected_keys = {str(row["identity_key"]) for row in selected_rows}
    observed_keys = [str(row["identity_key"]) for row in execution_rows]
    exact_rows = len(selected_rows) == EXPECTED_ROWS and len(execution_rows) == EXPECTED_ROWS
    complete_keys = (
        len(observed_keys) == len(set(observed_keys)) == EXPECTED_ROWS
        and set(observed_keys) == expected_keys
    )
    treated_rows = [row for row in execution_rows if row["regime_id"] in TREATED_REGIMES]
    native_complete = (
        exact_rows
        and complete_keys
        and all(
            row["row_status"] == "native" and isinstance(row.get("diagnostics"), Mapping)
            for row in execution_rows
        )
    )
    checker_result: dict[str, Any] | None = None
    verdict = "invalid_incomplete"
    if native_complete and len(treated_rows) == 48:
        preflight_payload, protocol_for_checker = load_preflight(preflight_config_path)
        checker_payload = {
            "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
            "seeds": [EXPECTED_SEED],
            "rows": treated_rows,
        }
        checker_result = classify_activation(
            preflight_payload, protocol_for_checker, checker_payload
        )
        verdict = "activation_pass" if checker_result["ok"] else "intervention_not_activated"
    result: dict[str, Any] = {
        "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
        "runner_schema_version": RUNNER_SCHEMA_VERSION,
        "banner": NOT_EVIDENCE_BANNER,
        "issue": 8871,
        "parent_issue": 6561,
        "source_commit": manifest["source_commit"],
        "protocol_sha256": manifest["protocol_sha256"],
        "canary_manifest_hash": manifest["manifest_hash"],
        "seeds": [EXPECTED_SEED],
        "expected_rows": EXPECTED_ROWS,
        "observed_rows": len(execution_rows),
        "selected_rows": len(selected_rows),
        "verdict": verdict,
        "preflight": preflight,
        "rows": treated_rows,
        "all_rows": execution_rows,
        "activation_result": checker_result,
        "claim_boundary": "activation diagnostics only; no benchmark or publication evidence",
        "diagnostic_journal": {
            "path_label": journal_path.name,
            "sha256": _sha256(journal_path),
            "event_contract": "header; row_started; row_finished; unfinished rows require reconciliation",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(output_path)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, required=True, help="Frozen 72-row packet JSON")
    parser.add_argument(
        "--output", type=Path, help="Diagnostics receipt JSON (required for execution)"
    )
    parser.add_argument(
        "--journal",
        type=Path,
        help="Append-only per-row journal (defaults to <output>.journal.jsonl)",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CANARY_CONFIG)
    parser.add_argument("--preflight-config", type=Path, default=DEFAULT_PRECHECK_CONFIG)
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=REPO_ROOT / "output/model_cache",
        help="Staged registry cache containing <model_id>/<artifact> files",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate prerequisites without running episodes",
    )
    parser.add_argument(
        "--max-rows", type=int, help="Deterministic smoke prefix; always returns invalid_incomplete"
    )
    return parser.parse_args()


def main() -> int:
    """Run the preflight or exact packet and return a fail-closed status code."""
    args = _parse_args()
    try:
        manifest = load_execution_manifest(args.manifest, config_path=args.config)
        checkpoint_root = args.checkpoint_root.expanduser()
        if args.preflight_only:
            report = preflight_canary(
                manifest,
                config_path=args.config,
                preflight_config_path=args.preflight_config,
                checkpoint_root=checkpoint_root,
            )
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0 if report["status"] == "ok" else 2
        if args.output is None:
            raise CanaryError("--output is required unless --preflight-only is used")
        result = run_canary(
            args.manifest,
            args.output,
            config_path=args.config,
            preflight_config_path=args.preflight_config,
            checkpoint_root=checkpoint_root,
            max_rows=args.max_rows,
            journal_path=args.journal,
        )
        print(
            json.dumps(
                {
                    key: result[key]
                    for key in ("verdict", "canary_manifest_hash", "observed_rows", "preflight")
                },
                indent=2,
                sort_keys=True,
            )
        )
        return {"activation_pass": 0, "intervention_not_activated": 1, "invalid_incomplete": 2}[
            result["verdict"]
        ]
    except (CanaryError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"error: {_safe_error(exc)}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
