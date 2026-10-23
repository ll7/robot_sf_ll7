"""Compile receipt contracts through the same resolver and constructors as map episodes."""

from __future__ import annotations

import dataclasses
import enum
import importlib.metadata
import inspect
import json
import os
import platform
import shutil
import tempfile
from pathlib import Path
from typing import Any

from robot_sf.benchmark.receipt_stages import file_digest, stage_key


def plain(value: Any) -> Any:
    """Snapshot effective configuration without object addresses or mutable references.

    Returns:
        The validated stage result."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: plain(getattr(value, field.name)) for field in dataclasses.fields(value)
        }
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(plain(item) for item in value)
    if isinstance(value, enum.Enum):
        return plain(value.value)
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return plain(value.tolist())
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError(f"unsupported effective configuration type: {type(value).__name__}")


def check_metric_contract(cfg: Any) -> None:
    """Reject historical normalization before constructing the first environment."""
    from robot_sf.benchmark.metric_definitions import (  # noqa: PLC0415 - stage-local heavy dependency.
        METRIC_SCHEMA_VERSION,
        require_anchor_compatibility,
    )

    for path in (cfg.snqi_baseline_path,):
        if path is not None:
            require_anchor_compatibility(
                {"metric_schema_version": METRIC_SCHEMA_VERSION}, json.loads(path.read_bytes())
            )
    if cfg.snqi_v2_spec is not None:
        require_anchor_compatibility(
            {"metric_schema_version": METRIC_SCHEMA_VERSION}, cfg.snqi_v2_spec.provenance()
        )


def selected_episode_ids(derived: list[tuple]) -> list[list[Any]]:
    """Expand the real selected matrix and reject duplicate or missing combinations.

    Returns:
        The validated stage result."""
    ids, seen = [], set()
    for suite, _path, scenarios, cfg in derived:
        declared_seeds = list(cfg.seed_policy.seeds)
        for scenario in scenarios:
            if sorted(scenario["seeds"]) != sorted(declared_seeds):
                raise ValueError("missing or duplicate scenario/seed combination")
            for planner in cfg.planners:
                for kin in cfg.kinematics_matrix:
                    for seed in declared_seeds:
                        identity = (
                            planner.key,
                            kin,
                            scenario.get("id") or scenario.get("scenario_id") or scenario["name"],
                            seed,
                        )
                        if identity in seen:
                            raise ValueError("duplicate episode combination")
                        seen.add(identity)
                        ids.append([suite, *identity])
    return sorted(ids)


def expected_episode_ids(root: Path) -> list[list[Any]]:
    """Expand the authoritative promotion roster independently of a selected execution slice.

    Returns:
        The validated stage result."""
    from scripts.validation.run_empty_world_sweep import assert_dev_seeds  # noqa: PLC0415 - defer heavy runtime imports until the stage executes.

    scope = json.loads(
        (root / "configs/benchmarks/releases/behaviour_gate_0_1_0.json").read_bytes()
    )
    arms, slots, seeds = scope["arms"], scope["maps"], assert_dev_seeds(list(range(1001, 1031)))
    if len(arms) != len(set(arms)) or len(slots) != len(set(slots)):
        raise ValueError("duplicate arm or scenario in promotion contract")
    ids = [[arm, slot, seed] for arm in arms for slot in slots for seed in seeds]
    if len(arms) != 14 or len(slots) != 51 or len(ids) != 21420:
        raise ValueError("missing promotion combinations: expected 14 arms x 51 slots x 30 seeds")
    return sorted(ids)


def check_output(root: Path, *, minimum_free_bytes: int) -> dict:
    """Exercise actual create/fsync/rename paths and check the declared storage budget.

    Returns:
        The validated stage result."""
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("receipt output must be a real directory")
    usage = shutil.disk_usage(root)
    if usage.free < minimum_free_bytes:
        raise ValueError("insufficient receipt output disk space")
    descriptor, temporary = tempfile.mkstemp(prefix=".receipt-path-check-", dir=root)
    target = Path(temporary + ".published")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(b"receipt-path-check\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        if target.read_bytes() != b"receipt-path-check\n":
            raise ValueError("receipt output readback failed")
    finally:
        Path(temporary).unlink(missing_ok=True)
        target.unlink(missing_ok=True)
    return {"minimum_free_bytes": minimum_free_bytes, "free_bytes_at_preflight": usage.free}


def _policy_objects(policy: Any) -> list[Any]:
    """Traverse owned adapters, bound lifecycle hooks and real policy closures.

    Returns:
        Distinct live objects owned by the constructed policy.
    """
    result, pending, seen = [], [policy], set()
    while pending:
        item = pending.pop()
        if id(item) in seen:
            continue
        seen.add(id(item))
        result.append(item)
        if inspect.isfunction(item):
            pending.extend(
                value
                for value in inspect.getclosurevars(item).nonlocals.values()
                if not isinstance(value, (str, int, float, bool, dict, list, tuple, type(None)))
            )
        owner = getattr(item, "__self__", None)
        if owner is not None:
            pending.append(owner)
        for name in (
            "_planner_adapter",
            "_predictor",
            "predictor",
            "_planner",
            "planner",
            "_foresight",
            "_predictive_foresight",
            "_adapter",
            "fallback_adapter",
            "prior_adapter",
            "_planner_close",
            "_planner_reset",
            "_planner_bind_env",
            "_planner_stats",
        ):
            child = getattr(item, name, None)
            if child is not None:
                pending.append(child)
    return result


def _loaded_predictors(policy: Any) -> list[Any]:
    """Find actual predictor instances, including nested/closure-only learned planners.

    Returns:
        Constructed predictor adapters owned by the policy.
    """
    from robot_sf.planner.socnav_prediction import PredictionPlannerAdapter  # noqa: PLC0415

    return [item for item in _policy_objects(policy) if isinstance(item, PredictionPlannerAdapter)]


def _captured_parameters(item: Any) -> dict:
    """Read native policy function defaults and bound scalar inputs.

    Returns:
        Captured effective scalar parameters.
    """
    captured = {}
    if inspect.isfunction(item):
        for name, parameter in inspect.signature(item).parameters.items():
            if parameter.default is not inspect.Parameter.empty and isinstance(
                parameter.default, (str, int, float, bool)
            ):
                captured[item.__qualname__ + ".default." + name] = plain(parameter.default)
        for name, value in inspect.getclosurevars(item).nonlocals.items():
            if isinstance(value, (str, int, float, bool)):
                captured[item.__qualname__ + "." + name] = plain(value)
    return captured


def _policy_readback(policy: Any) -> dict:
    """Read owned planner configuration rather than substituting authored metadata.

    Returns:
        Actual object configurations and primitive values captured by native policies.
    """
    configurations, captured = [], {}
    for item in _policy_objects(policy):
        if dataclasses.is_dataclass(item) and not isinstance(item, type):
            parameters = plain(item)
            parameters.update(
                {
                    name: value
                    for name, value in vars(item).items()
                    if name not in parameters and isinstance(value, (str, int, float, bool))
                }
            )
            configurations.append(
                {
                    "type": type(item).__module__ + "." + type(item).__name__,
                    "field": "constructed_object",
                    "parameters": parameters,
                }
            )
        for field in ("config", "cfg", "_config"):
            config = getattr(item, field, None)
            if dataclasses.is_dataclass(config) or isinstance(config, dict):
                configurations.append(
                    {
                        "type": type(item).__module__ + "." + type(item).__name__,
                        "field": field,
                        "parameters": plain(config),
                    }
                )
        captured.update(_captured_parameters(item))
    if not configurations and not captured:
        raise ValueError("constructed planner has no observable effective configuration")
    return {"owned_configurations": configurations, "captured_parameters": captured}


def _forecast_readback(policy: Any) -> list[dict]:
    """Check requested horizons against the loaded model's own declared and actual capability.

    Returns:
        The validated stage result."""
    import torch  # noqa: PLC0415 - defer heavy runtime imports until the stage executes.

    observations = []
    for predictor in _loaded_predictors(policy):
        model = predictor._ensure_model()
        if model is None:
            raise ValueError("receipt predictor did not load its checkpoint")
        declared = int(model.config.horizon_steps)
        state, mask, *_ = predictor._build_model_input({})
        with torch.no_grad():
            output = model(
                torch.from_numpy(state[None]).to(predictor._device),
                torch.from_numpy(mask[None]).to(predictor._device),
            )
        actual = int(output["future_positions"].shape[-2])
        requests = [int(predictor.config.predictive_horizon_steps)]
        if predictor.config.predictive_adaptive_horizon_enabled:
            requests.append(
                requests[0] + max(0, int(predictor.config.predictive_horizon_boost_steps))
            )
        adapter = getattr(policy, "_planner_adapter", None)
        if adapter is not None and hasattr(adapter.config, "horizon_steps"):
            requests.append(int(adapter.config.horizon_steps))
        if actual != declared or any(not 1 <= request <= declared for request in requests):
            raise ValueError(
                f"forecast capability mismatch: requested={requests}, declared={declared}, actual={actual}; never clamp"
            )
        observations.append({"requested": requests, "declared": declared, "actual": actual})
    return observations


def inspect_construction(scenario: dict, seed: int, **kwargs: Any) -> dict:
    """Certify real constructors under the production episode's default-policy scope.

    Returns:
        Effective constructor snapshot, including its resolved default policy.
    """
    from robot_sf.common.hybrid_defaults import episode_defaults  # noqa: PLC0415

    with episode_defaults(**kwargs) as policy:
        snapshot = _inspect_construction(scenario, seed, **kwargs)
        snapshot["hybrid_default_policy"] = dict(policy)
        return snapshot


def _inspect_construction(scenario: dict, seed: int, **kwargs: Any) -> dict:
    """Construct real episode context, policy, simulator; read back their effective parameters.

    No reset or step is performed. Signature binding deliberately uses the production defaults;
    context resolution and policy construction are the very helpers used by run_map_episode.

    Returns:
        The validated stage result."""
    from robot_sf.benchmark.map_runner import map_runner as runner  # noqa: PLC0415 - defer heavy runtime imports until the stage executes.
    from robot_sf.benchmark.map_runner import map_runner_episode as episode  # noqa: PLC0415 - defer heavy runtime imports until the stage executes.

    values = inspect.signature(episode.run_map_episode).bind(
        scenario, seed, policy_builder=runner._build_policy, **kwargs
    )
    values.apply_defaults()
    records: list[dict[str, str]] = []
    arguments = dict(values.arguments)
    arguments["runtime_input_records"] = records
    ctx = episode._resolve_episode_run_context(
        **{
            name: arguments[name]
            for name in inspect.signature(episode._resolve_episode_run_context).parameters
        }
    )
    contract = episode._prepare_policy_and_observation_contract(
        scenario=ctx.scenario,
        algo=ctx.algo,
        policy_cfg=ctx.policy_cfg,
        config=ctx.config,
        observation_mode=arguments["observation_mode"],
        observation_level=arguments["observation_level"],
        robot_kinematics=ctx.robot_kinematics,
        robot_command_mode=ctx.robot_command_mode,
        adapter_impact_eval=arguments["adapter_impact_eval"],
        benchmark_track=ctx.benchmark_track,
        track_schema_version=ctx.track_schema_version,
        actuation_profile=ctx.actuation_profile,
        policy_builder=runner._build_policy,
    )
    env = None
    try:
        forecasts = _forecast_readback(contract.policy_fn)
        env = episode.make_robot_env(config=ctx.config, seed=seed, debug=False)
        if contract.planner_bind_env:
            contract.planner_bind_env(env)
        native_env: Any = env.unwrapped
        return {
            "resolved_policy": plain(ctx.policy_cfg),
            "simulator": plain(native_env.simulator.config),
            "robot": plain(native_env.simulator.robots[0].config),
            "environment": {
                name: plain(getattr(native_env.config, name))
                for name in (
                    "observation_mode",
                    "include_goal_next_valid",
                    "robot_config",
                    "sim_config",
                )
            },
            "planner": _policy_readback(contract.policy_fn),
            "forecast_capabilities": forecasts,
            "runtime_inputs": records,
            "effective_horizon": ctx.horizon_val,
        }
    finally:
        if env is not None:
            env.close()
        if contract.planner_close:
            contract.planner_close()


def environment_identity() -> dict:
    """Conservatively bind the installed runtime rather than guessing a dependency subset.

    Returns:
        The validated stage result."""
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": sorted(
            (dist.metadata["Name"], dist.version) for dist in importlib.metadata.distributions()
        ),
    }


def input_digests(root: Path, snapshots: Any) -> dict[str, str]:  # noqa: C901 - inspect nested effective inputs.
    """Digest resolved files and check declared runtime-input checksums.

    Returns:
        The validated stage result."""
    paths: set[Path] = set()

    def visit(value: Any) -> None:  # noqa: C901 - recursive asset inventory.
        if isinstance(value, dict):
            from robot_sf.models.registry import get_registry_entry, resolve_model_path  # noqa: PLC0415

            for key in ("model_id", "predictive_model_id"):
                model_id = value.get(key)
                if isinstance(model_id, str) and model_id:
                    checkpoint = resolve_model_path(model_id, allow_download=False).resolve()
                    entry = get_registry_entry(model_id)
                    expected = (entry.get("github_release") or {}).get("sha256")
                    if expected and file_digest(checkpoint) != expected:
                        raise ValueError("registry checkpoint digest mismatch")
                    paths.add(checkpoint)
            if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
                candidate = Path(value["path"])
                candidate = candidate if candidate.is_absolute() else root / candidate
                if not candidate.is_file() or file_digest(candidate) != value["sha256"]:
                    raise ValueError("runtime asset digest mismatch")
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
        elif isinstance(value, str) and len(value) < 4096:
            candidate = Path(value)
            candidate = candidate if candidate.is_absolute() else root / candidate
            if candidate.is_file():
                paths.add(candidate.resolve())

    visit(snapshots)
    return {str(path): file_digest(path) for path in sorted(paths)}


def episode_inputs(cfg: Any, arm: Any, scenario: dict, kin: str, root: Path) -> tuple[dict, dict]:
    """Apply the campaign's own kinematics, horizon and profile bindings to a slot.

    Returns:
        The validated stage result."""
    from robot_sf.benchmark.camera_ready._config import (  # noqa: PLC0415 - stage-local heavy dependency.
        _apply_fixed_campaign_horizon,
        _scenario_with_kinematics,
    )
    from robot_sf.benchmark.camera_ready.campaign import (  # noqa: PLC0415 - stage-local heavy dependency.
        _latency_stress_metadata,
        _resolve_arm_safety_wrapper,
        _synthetic_actuation_metadata,
    )

    horizon = arm.horizon_override if arm.horizon_override is not None else cfg.horizon
    dt = arm.dt_override if arm.dt_override is not None else cfg.dt
    scoped = _scenario_with_kinematics(
        scenario, kinematics=kin, holonomic_command_mode=cfg.holonomic_command_mode
    )
    if cfg.telemetry is not None:
        scoped["telemetry"] = dict(cfg.telemetry)
    scoped = _apply_fixed_campaign_horizon(
        [scoped],
        horizon=horizon,
        horizon_policy=cfg.horizon_policy,
        protocol_version=cfg.protocol_version,
    )[0]
    return scoped, {
        "horizon": horizon or 0,
        "dt": dt or 0.0,
        "record_forces": cfg.record_forces,
        "snqi_weights": json.loads(cfg.snqi_weights_path.read_bytes())
        if cfg.snqi_weights_path
        else None,
        "snqi_baseline": json.loads(cfg.snqi_baseline_path.read_bytes())
        if cfg.snqi_baseline_path
        else None,
        "algo": arm.algo,
        "scenario_path": root / "scoped_scenarios.json",
        "provenance_scenario_path": cfg.scenario_matrix_path,
        "algo_config_path": str(arm.algo_config_path) if arm.algo_config_path else None,
        "adapter_impact_eval": arm.adapter_impact_eval,
        "observation_mode": arm.observation_mode or cfg.observation_mode,
        "observation_noise": cfg.observation_noise,
        "synthetic_actuation_profile": _synthetic_actuation_metadata(
            cfg.synthetic_actuation_profile
        ),
        "latency_stress_profile": _latency_stress_metadata(cfg.latency_stress_profile, dt=dt),
        "safety_wrapper": _resolve_arm_safety_wrapper(cfg=cfg, planner=arm),
        "record_simulation_step_trace": cfg.record_simulation_step_trace,
        "record_planner_decision_trace": cfg.record_planner_decision_trace,
    }


def compile_contract(
    derived: list[tuple],
    *,
    source_sha: str,
    root: Path,
    output_dir: Path,
    minimum_free_bytes: int,
    full_roster: bool,
) -> dict:
    """Freeze all identities before the first trajectory and retain construction readback.

    Returns:
        The validated stage result."""
    from robot_sf.benchmark import metric_definitions  # noqa: PLC0415
    from robot_sf.benchmark.camera_ready._config import _load_campaign_scenarios  # noqa: PLC0415 - defer heavy runtime imports until the stage executes.

    selected = selected_episode_ids(derived)
    expected = expected_episode_ids(root)
    if full_roster and sorted([arm, slot, seed] for _, arm, _, slot, seed in selected) != expected:
        raise ValueError("missing or unexpected full promotion combination")
    disk = check_output(output_dir, minimum_free_bytes=minimum_free_bytes)
    from robot_sf.benchmark.map_runner import map_runner as runner  # noqa: PLC0415

    snapshots = []
    for suite, cfg_path, _scenarios, cfg in derived:
        resolved = _load_campaign_scenarios(cfg)
        for arm in cfg.planners:
            for kin in cfg.kinematics_matrix or ("differential_drive",):
                inputs = [episode_inputs(cfg, arm, scenario, kin, root) for scenario in resolved]
                kwargs = inputs[0][1]
                admitted = runner.run_map_batch(
                    [scoped for scoped, _ in inputs],
                    out_path=output_dir
                    / "construction_preflight"
                    / suite
                    / f"{arm.key}__{kin}"
                    / "episodes.jsonl",
                    schema_path=root / "robot_sf/benchmark/schemas/episode.schema.v1.json",
                    provenance_scenario_path=cfg.scenario_matrix_path,
                    benchmark_profile=arm.benchmark_profile,
                    socnav_missing_prereq_policy=(
                        "fail-fast"
                        if cfg.checkpoint_provenance_enforcement == "error"
                        else arm.socnav_missing_prereq_policy
                    ),
                    preflight_only=True,
                    workers=1,
                    resume=False,
                    **kwargs,
                )
                expected_jobs = sum(len(scoped["seeds"]) for scoped, _ in inputs)
                if not admitted.get("preflight_complete") or admitted["jobs"] != expected_jobs:
                    raise ValueError(
                        "production batch preflight skipped or omitted receipt combinations"
                    )
                for scenario, (scoped, kwargs) in zip(resolved, inputs, strict=True):
                    snapshot = inspect_construction(scoped, min(scoped["seeds"]), **kwargs)
                    snapshots.append(
                        {
                            "suite": suite,
                            "arm": arm.key,
                            "kinematics": kin,
                            "slot": scenario.get("id")
                            or scenario.get("scenario_id")
                            or scenario["name"],
                            "effective": snapshot,
                        }
                    )
    assets = input_digests(root, [snapshots, [plain(cfg) for _, _, _, cfg in derived]])
    # Hash the historical metric implementation without substituting newer metric code.
    metric_sources = {
        relative: file_digest(root / relative)
        for relative in (
            "robot_sf/benchmark/metrics.py",
            "robot_sf/benchmark/metric_definitions.py",
            "robot_sf/benchmark/constants.py",
            "robot_sf/benchmark/robot_force_contract.py",
            "robot_sf/benchmark/map_runner/map_runner_episode.py",
        )
    }
    declared_digest = getattr(metric_definitions, "metric_definitions_sha256", None)
    execution = {
        "source_sha": source_sha,
        "acquisition_bundle": json.loads((output_dir / "harness.json").read_bytes())
        if (output_dir / "harness.json").is_file()
        else None,
        "harness_sha256": file_digest(root / "scripts/validation/run_empty_world_sweep.py"),
        "environment": environment_identity(),
        "selected_episode_ids": selected,
        "effective_inputs": snapshots,
        "assets": assets,
        "metric_definitions_sha256": declared_digest() if declared_digest is not None else None,
        "metric_semantics_sha256": stage_key("metric-semantics", metric_sources),
        "metric_semantics_sources": metric_sources,
    }
    return {
        "schema_version": "receipt-contract.v1",
        "source_sha": source_sha,
        "execution_key": stage_key("execution", execution),
        "execution": execution,
        "expected_episode_ids": expected,
        "expected_episodes": len(expected),
        "selected_episodes": len(selected),
        "full_roster": full_roster,
        "row_schema_sha256": file_digest(
            root / "robot_sf/benchmark/schemas/episode.schema.v1.json"
        ),
        "metric_schema_version": metric_definitions.METRIC_SCHEMA_VERSION,
        "disk": disk,
    }


def prepare_completion_views(
    derived: list[tuple], output_dir: Path, contract: dict, *, first_seed_only: bool = False
) -> None:
    """Recover immutable rows before the campaign's normal missing-job filter runs."""
    from robot_sf.benchmark.receipt_stages import atomic_json, recover_episodes  # noqa: PLC0415 - defer heavy runtime imports until the stage executes.
    from robot_sf.benchmark.schema_validator import load_schema  # noqa: PLC0415 - defer heavy runtime imports until the stage executes.

    schema = load_schema(Path(__file__).parent / "schemas/episode.schema.v1.json")
    for suite, _path, _scenarios, cfg in derived:
        campaign = output_dir / "campaigns" / f"empty_world_{suite}"
        descriptor = {
            "schema_version": "receipt-execution.v1",
            "source_sha": contract["source_sha"],
            "execution_key": contract["execution_key"],
            "run_id": str(output_dir.resolve()),
        }
        atomic_json(campaign / "receipt_execution.json", descriptor, immutable=True)
        seeds = list(cfg.seed_policy.seeds)
        dispatch = {
            "execution_key": contract["execution_key"],
            "seeds": [min(seeds)] if first_seed_only else seeds,
        }
        atomic_json(campaign / "receipt_dispatch.json", dispatch)
        for arm in cfg.planners:
            for kin in cfg.kinematics_matrix or ("differential_drive",):
                directory = campaign / "runs" / f"{arm.key}__{kin}"
                if directory.exists():
                    recover_episodes(directory / "episodes.jsonl", schema, descriptor)
                    atomic_json(directory / "receipt_dispatch.json", dispatch)
